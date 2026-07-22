#!/usr/bin/env python
"""KS-SCE self-consistent field calculation on a 1D grid, with Pulay DIIS.

Cleaned-up and working rewrite of main.py. Fixes with respect to the old
script:

  * the potential is added on the diagonal of H (the old code broadcast a 1D
    vector over the whole matrix, producing a non-symmetric Hamiltonian)
  * orbitals are normalised on the grid, so the density integrates to N
    (required by the SCE comotion functions)
  * the SCE potential is actually used in the SCF (the old loop only ever
    used 0.5*Hartree even though it computed vSCE)
  * the initial guess is centred on the external wells at +/-R (was +/-R/2)
  * proper DIIS with safeguards, a convergence test and energy monitoring

Convergence strategy. For this stretched two-well system the KS-SCE
solution lies in the dissociation regime: the self-consistent SCE potential
builds a step that brings the highest occupied levels of the two wells into
alignment, so the ground state is an ensemble state with fractional
occupations (q electrons in the left well, N-q in the right) and a naive
aufbau SCF flip-flops between the wells forever, no matter the damping.
The stable formulation used here:

  * inner loop:  DIIS SCF at *fixed* occupations (q, N-q), with the two
    frontier orbitals tracked by maximum overlap so that a level crossing
    cannot swap their occupations;
  * outer loop:  bisection on eps_L(q) - eps_R(q).  By Janak's theorem
    dE/dq = eps_L - eps_R, so the aligned point is the energy minimum;
    if the levels never align, the best integer occupation wins.

The weakly-correlated functionals (LDA, Hartree, EXX) go through a standard
aufbau DIIS SCF with damping and a level shift instead.

The script runs every functional in `functionals`, prints a comparison
table and overlays the densities and Hxc potentials in one figure
(ks_comparison.png).

Run:  python main_fixed.py [--show]
"""

import os
import sys

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

from jax import config
config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import OneDKSSCE

np.set_printoptions(precision=6)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
nbPts = 600                     # number of grid points
b = 0.6                         # softening parameter of the e-e interaction
wint = OneDKSSCE.softCoulomb    # softCoulomb / absCoulomb / wireCoulomb
R = 10.0                        # nuclei at -R and +R
L = 8 * (R + 10)                # box length
site_softening = [2.25, 0.7]    # softening of the external wells at -R and +R
N = 2                           # electrons
functionals = ("SCE", "LDA", "Hartree")   # runs each; "EXX" also available
functional = "SCE"              # current functional (set by main)

max_cycle = 300                 # inner SCF cycles
conv_tol = 1e-10                # inner SCF: target on the density change
conv_accept = 1e-6              # inner SCF: acceptance threshold for the best
                                # iterate; exactly at the level degeneracy the
                                # grid SCE potential limits self-consistency
                                # to ~1e-7 in the density
diis_space = 8                  # size of the DIIS subspace
diis_engage = 1e-3              # engage DIIS once drho falls below this;
                                # in the earlier nonlinear regime plain damping
                                # is more robust, in the linear regime DIIS
                                # kills the soft charge-transfer mode
mixing = 0.2                    # linear density mixing before DIIS engages

# aufbau SCF (Hartree/EXX only)
aufbau_max_cycle = 500
aufbau_mixing = 0.1
level_shift = 0.3               # virtual-level shift; keeps the occupied
                                # orbital from flipping when HOMO and LUMO
                                # get close (does not change the fixed point)
q_scan = np.arange(0.0, N + 1e-9, 0.25)   # coarse scan of the charge split
align_tol = 1e-5                # outer bisection: threshold on the q bracket
                                # (E is stationary at q*, so the E error is
                                # ~ curvature * align_tol^2 ~ 1e-10)

assert N == 2, "the energy expressions assume a 2-electron system"

# ---------------------------------------------------------------------------
# Grid and one-body Hamiltonian
# ---------------------------------------------------------------------------
def set_geometry(R_new=None, site_softening_new=None, nbPts_new=None, L_new=None):
    """(Re)build the grid-dependent module state.  Companion scripts call
    this to change the geometry (e.g. to stretch the molecule)."""
    global R, site_softening, nbPts, L, x, delta, weights, kin, vext, h1
    global ref_left, ref_right, _coulomb_matrix

    if R_new is not None:
        R = R_new
        L = 8 * (R + 10)
    if L_new is not None:
        L = L_new
    if site_softening_new is not None:
        site_softening = site_softening_new
    if nbPts_new is not None:
        nbPts = nbPts_new

    x = np.linspace(-L / 2, L / 2, nbPts)
    delta = x[1] - x[0]
    weights = np.full(nbPts, delta)

    kin = OneDKSSCE.kinetic(nbPts, delta, 1)
    vext = OneDKSSCE.vExtMol_reduced(x, [-R, R], wint, site_softening)
    h1 = kin + np.diag(vext)
    _coulomb_matrix = None

    # Ground states of the isolated wells: initial guess and references for
    # tracking the left/right frontier orbitals through the SCF.
    _, C = scipy.linalg.eigh(kin - np.diag(np.asarray(wint(np.abs(x + R), site_softening[0]))))
    ref_left = C[:, 0]
    _, C = scipy.linalg.eigh(kin - np.diag(np.asarray(wint(np.abs(x - R), site_softening[1]))))
    ref_right = C[:, 0]


set_geometry()

# ---------------------------------------------------------------------------
# Hxc potential and energies
# ---------------------------------------------------------------------------
_coulomb_matrix = None


def coulomb_matrix():
    global _coulomb_matrix
    if _coulomb_matrix is None:
        _coulomb_matrix = np.asarray(wint(np.abs(x[:, None] - x[None, :]), b))
    return _coulomb_matrix


def lda_xc_potential(rho):
    """1D LDA xc potential (OneDKSSCE parametrisation, rs = 1/(2 rho)).
    Evaluated only where the density is non-negligible: the xc potential
    vanishes as rho -> 0, and the rs -> inf limit overflows numerically."""
    v = np.zeros_like(rho)
    mask = rho > 1e-10
    with np.errstate(all='ignore'):
        rs = 0.5 / rho[mask]
        v[mask] = OneDKSSCE.LDA_correl_pot(rs) + OneDKSSCE.LDA_exchange_pot(rs)
    return v


def lda_xc_energy(rho):
    """1D LDA xc energy int rho (ex + ec) dx."""
    mask = rho > 1e-10
    with np.errstate(all='ignore'):
        rs = 0.5 / rho[mask]
        e = OneDKSSCE.LDA_correl_nrj(rs) + OneDKSSCE.LDA_exchange_nrj(rs)
    return np.sum(weights[mask] * rho[mask] * e)


def hxc_potential(rho):
    """Hxc potential on the grid (1D array) for the density rho."""
    if functional == "SCE":
        return np.asarray(
            OneDKSSCE.compute_kssce_potential_gauge_zero(rho, weights, x, N, wint, b))
    v = coulomb_matrix() @ (weights * rho)
    v -= 0.5 * (v[0] + v[-1])
    if functional == "EXX":
        v *= 0.5            # for N=2 exact exchange cancels half the Hartree term
    elif functional == "LDA":
        v = v + lda_xc_potential(rho)
    return v


def sce_energy(rho):
    """V_ee^SCE = 1/2 int rho(x) w(|x - f(x)|) dx for N=2, averaged over the
    left- and right-integrated comotion functions (as in OneDKSSCE)."""
    f_fwd = np.asarray(OneDKSSCE.compute_all_comotion_functions(rho, weights, x, N))[1]
    f_rev = np.flip(np.asarray(OneDKSSCE.compute_all_comotion_functions(
        rho[::-1], weights[::-1], x[::-1], N)), axis=1)[1]

    def pair_energy(f):
        return 0.5 * np.sum(weights * rho * np.asarray(wint(np.abs(x - f), b)))

    return 0.5 * (pair_energy(f_fwd) + pair_energy(f_rev))


def total_energy(orb_left, orb_right, q, rho):
    """Total energy from the two occupied orbitals with occupations (q, N-q)."""
    ts = q * (orb_left @ kin @ orb_left) + (N - q) * (orb_right @ kin @ orb_right)
    e_ext = np.sum(weights * vext * rho)
    if functional == "SCE":
        e_hxc = sce_energy(rho)
    else:
        p = weights * rho
        e_hxc = 0.5 * p @ coulomb_matrix() @ p
        if functional == "EXX":
            e_hxc *= 0.5
        elif functional == "LDA":
            e_hxc += lda_xc_energy(rho)
    return ts + e_ext + e_hxc, ts, e_ext, e_hxc


# ---------------------------------------------------------------------------
# DIIS
# ---------------------------------------------------------------------------
def diis_extrapolate(fock_list, err_list):
    n = len(fock_list)
    B = np.empty((n + 1, n + 1))
    B[-1, :] = B[:, -1] = -1.0
    B[-1, -1] = 0.0
    for i in range(n):
        for j in range(i, n):
            B[i, j] = B[j, i] = np.vdot(err_list[i], err_list[j])
    rhs = np.zeros(n + 1)
    rhs[-1] = -1.0
    try:
        coeff = np.linalg.solve(B, rhs)
    except np.linalg.LinAlgError:
        coeff = np.linalg.lstsq(B, rhs, rcond=None)[0]
    return sum(w * F for w, F in zip(coeff[:-1], fock_list))


# ---------------------------------------------------------------------------
# Inner loop: DIIS SCF at fixed occupations (q, N-q)
# ---------------------------------------------------------------------------
def scf_fixed_q(q, verbose=False):
    """Converge the KS equations with q electrons in the left frontier
    orbital and N-q in the right one, tracked by maximum overlap.

    Plain damped iteration until the density change drops below
    diis_engage, then undamped commutator DIIS to machine precision."""
    orb_l, orb_r = ref_left, ref_right
    rho = (q * orb_l**2 + (N - q) * orb_r**2) / weights
    fock_list, err_list = [], []
    D = None
    diis_on = False
    best = None

    for cycle in range(1, max_cycle + 1):
        F = h1 + np.diag(hxc_potential(rho))

        if D is not None and diis_on:
            err = F @ D - D @ F
            fock_list.append(F)
            err_list.append(err)
            if len(fock_list) > diis_space:
                fock_list.pop(0)
                err_list.pop(0)
            if len(fock_list) >= 2:
                F = diis_extrapolate(fock_list, err_list)

        eigenvalues, C = scipy.linalg.eigh(F, subset_by_index=[0, 7])
        i_l = np.argmax(np.abs(C.T @ orb_l))
        i_r = np.argmax(np.abs(C.T @ orb_r))
        if i_l == i_r:                       # tracking collapsed on one orbital
            return None
        orb_l, orb_r = C[:, i_l], C[:, i_r]
        eps_l, eps_r = eigenvalues[i_l], eigenvalues[i_r]

        rho_new = (q * orb_l**2 + (N - q) * orb_r**2) / weights
        D = q * np.outer(orb_l, orb_l) + (N - q) * np.outer(orb_r, orb_r)

        drho = np.sum(weights * np.abs(rho_new - rho))
        if verbose:
            e_now = total_energy(orb_l, orb_r, q, rho_new)[0]
            print(f"    cycle {cycle:3d}  E = {e_now:.10f}  drho = {drho:.3e}")
        if best is None or drho < best["drho"]:
            best = dict(q=q, eps_l=eps_l, eps_r=eps_r, rho=rho_new,
                        orb_l=orb_l, orb_r=orb_r, cycles=cycle, drho=drho)
        if drho < conv_tol:
            break
        if not diis_on and drho < diis_engage:
            diis_on = True
        rho = rho_new if diis_on else (1 - mixing) * rho + mixing * rho_new

    best["E"] = total_energy(best["orb_l"], best["orb_r"], q, best["rho"])[0]
    best["converged"] = best["drho"] < conv_accept
    return best


def scf_aufbau(verbose=False):
    """Standard closed-shell aufbau DIIS SCF (both electrons in the lowest
    orbital), suitable for the weakly-correlated functionals."""
    rho = (ref_left**2 + ref_right**2) / weights
    fock_list, err_list = [], []
    D = None
    diis_on = False
    best = None

    for cycle in range(1, aufbau_max_cycle + 1):
        F = h1 + np.diag(hxc_potential(rho))

        if D is not None and diis_on:
            err = F @ D - D @ F
            fock_list.append(F)
            err_list.append(err)
            if len(fock_list) > diis_space:
                fock_list.pop(0)
                err_list.pop(0)
            if len(fock_list) >= 2:
                F = diis_extrapolate(fock_list, err_list)

        if D is not None and level_shift:
            F = F + level_shift * (np.eye(nbPts) - D / N)

        eigenvalues, C = scipy.linalg.eigh(F, subset_by_index=[0, 7])
        c = C[:, 0]
        rho_new = N * c**2 / weights
        D = N * np.outer(c, c)

        drho = np.sum(weights * np.abs(rho_new - rho))
        if verbose:
            print(f"    cycle {cycle:3d}  drho = {drho:.3e}")
        if best is None or drho < best["drho"]:
            best = dict(q=None, eps_l=eigenvalues[0], eps_r=eigenvalues[1],
                        rho=rho_new, orb_l=c, orb_r=c, cycles=cycle, drho=drho)
        if drho < conv_tol:
            break
        if not diis_on and drho < diis_engage:
            diis_on = True
        rho = (rho_new if diis_on
               else (1 - aufbau_mixing) * rho + aufbau_mixing * rho_new)

    # the level shift displaces the virtual eigenvalues: recompute them cleanly
    eigenvalues = scipy.linalg.eigh(h1 + np.diag(hxc_potential(best["rho"])),
                                    eigvals_only=True, subset_by_index=[0, 7])
    best["eps_l"], best["eps_r"] = eigenvalues[0], eigenvalues[1]
    best["E"] = total_energy(best["orb_l"], best["orb_r"], 1.0, best["rho"])[0]
    best["converged"] = best["drho"] < conv_accept
    return best


# ---------------------------------------------------------------------------
# Outer loop: find the charge split that aligns the frontier levels
# ---------------------------------------------------------------------------
def run():
    print(f"KS-{functional}: {nbPts} points, L = {L}, R = {R}, b = {b}, N = {N}")

    if functional != "SCE":
        res = scf_aufbau()
        state = "converged" if res["converged"] else "NOT converged"
        print(f"\nAufbau SCF {state} in {res['cycles']} cycles, "
              f"drho = {res['drho']:.1e}")
        return res

    print(f"\nScanning the charge split q (electrons in the left well):")
    print(f"{'q':>6} {'E':>16} {'eps_L':>10} {'eps_R':>10} {'eps_L-eps_R':>12} {'cycles':>7}")

    scan = []
    for q in q_scan:
        res = scf_fixed_q(q)
        if res is None or not res["converged"]:
            info = "tracking failed" if res is None else f"not converged, drho={res['drho']:.1e}"
            print(f"{q:6.2f}   [{info}]")
            continue
        scan.append(res)
        print(f"{q:6.2f} {res['E']:16.8f} {res['eps_l']:10.5f} {res['eps_r']:10.5f} "
              f"{res['eps_l'] - res['eps_r']:12.5f} {res['cycles']:7d}")

    if not scan:
        raise RuntimeError("no converged SCF in the q scan")

    # bracket a sign change of eps_L - eps_R around the scan minimum
    gaps = [r["eps_l"] - r["eps_r"] for r in scan]
    bracket = next(((scan[i], scan[i + 1]) for i in range(len(scan) - 1)
                    if gaps[i] < 0 <= gaps[i + 1]), None)

    if bracket is None:
        best = min(scan, key=lambda r: r["E"])
        print(f"\nNo level alignment found: integer-occupation minimum at q = {best['q']}.")
        return best

    lo, hi = bracket
    print(f"\nLevel alignment bracketed in q = [{lo['q']}, {hi['q']}]; bisecting...")
    while hi["q"] - lo["q"] > align_tol:
        q_mid = 0.5 * (lo["q"] + hi["q"])
        mid = scf_fixed_q(q_mid)
        if mid is None or not mid["converged"]:
            raise RuntimeError(f"inner SCF failed at q = {q_mid}")
        if mid["eps_l"] - mid["eps_r"] < 0:
            lo = mid
        else:
            hi = mid
    best = min(lo, hi, key=lambda r: r["E"])
    print(f"Converged: q* = {best['q']:.8f}, eps_L - eps_R = "
          f"{best['eps_l'] - best['eps_r']: .3e}")
    return best


def report(res):
    """Print the summary of one converged calculation and record the pieces
    needed for the comparison (energy components, Hxc potential)."""
    q, rho = res["q"], res["rho"]
    e_tot, ts, e_ext, e_hxc = total_energy(res["orb_l"], res["orb_r"],
                                           1.0 if q is None else q, rho)
    if q is None:
        print(f"\nOccupations:         closed-shell aufbau ({N} in the lowest orbital)")
        print(f"HOMO, LUMO:          {res['eps_l']:.6f}, {res['eps_r']:.6f}")
    else:
        print(f"\nOccupations (left, right):  ({q:.6f}, {N - q:.6f})")
        print(f"Frontier levels:            eps_L = {res['eps_l']:.6f}, "
              f"eps_R = {res['eps_r']:.6f}")
    print(f"Kinetic energy:      {ts: .10f}")
    print(f"External energy:     {e_ext: .10f}")
    print(f"Hxc energy:          {e_hxc: .10f}")
    print(f"Total energy:        {e_tot: .10f}")
    print(f"Integral of density: {np.sum(weights * rho):.10f}")

    res["functional"] = functional
    res["energy_parts"] = (e_tot, ts, e_ext, e_hxc)
    res["vhxc"] = hxc_potential(rho)


COLORS = {"SCE": "tab:red", "LDA": "tab:green", "Hartree": "tab:blue",
          "EXX": "tab:purple"}


def plot_comparison(results, show):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(x, vext, color='0.5', ls='--', label=r'$v_{ext}$')
    for res in results:
        color = COLORS.get(res["functional"])
        label = res["functional"]
        if res["q"] is not None:
            label += f" (q = {res['q']:.3f})"
        ax1.plot(x, res["rho"], color=color, label=label)
        ax2.plot(x, res["vhxc"], color=color, label=res["functional"])
    ax1.set(xlabel='x', title='Self-consistent densities', xlim=(-50, 50))
    ax2.set(xlabel='x', title='Converged Hxc potentials', xlim=(-50, 50))
    for ax in (ax1, ax2):
        ax.grid()
        ax.legend()
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'ks_comparison.png')
    fig.savefig(out, dpi=150)
    print(f"\nComparison plot saved to {out}")
    if show:
        plt.show()


def main(show=False):
    global functional
    results = []
    for functional in functionals:
        print("\n" + "=" * 70)
        res = run()
        report(res)
        results.append(res)

    print("\n" + "=" * 70)
    print("Energy comparison:")
    print(f"{'functional':>10} {'E_tot':>14} {'T_s':>12} {'E_ext':>12} "
          f"{'E_Hxc':>12} {'occupations':>16}")
    for res in results:
        e_tot, ts, e_ext, e_hxc = res["energy_parts"]
        occ = ("aufbau (2, 0)" if res["q"] is None
               else f"({res['q']:.3f}, {N - res['q']:.3f})")
        print(f"{res['functional']:>10} {e_tot:14.8f} {ts:12.6f} {e_ext:12.6f} "
              f"{e_hxc:12.6f} {occ:>16}")

    plot_comparison(results, show)


if __name__ == "__main__":
    main(show="--show" in sys.argv)
