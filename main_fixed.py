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

All settings live in config.toml next to this script: geometry (per-well
softening and depth, so homo- and heteronuclear setups are both one edit
away), grid, e-e interaction and the list of Hxc functionals to run
("SCE", "vcond", "vresp", "LDA", "Hartree", "EXX").  "vcond" is the SCE
conditional potential w(|x - f(x)|) and "vresp" the SCE response potential
vSCE - vcond; neither is a genuine Hxc potential, they are run as the sole
Hxc component to inspect the effect of each piece of vSCE, with the SCE
interaction functional kept as the (non-variational) energy expression.
The script runs every requested functional, prints a comparison table and
overlays the densities and Hxc potentials in one figure (ks_comparison.png).

As a final sanity check on the SCF, every converged orbital is verified to
solve its own Schroedinger equation: the Hamiltonian is rebuilt from the
converged orbitals (density -> Hxc potential -> H) and H phi / phi is
plotted pointwise for each occupied orbital, one panel per functional, in a
single board (schrodinger_check.png).  For a true self-consistent
eigenstate the curve is flat at the orbital energy wherever the orbital is
non-negligible.  The check is purely diagnostic: nothing acts on it.

Run:  python main_fixed.py [other_config.toml] [--show]
"""

import os
import sys
import tomllib

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

from jax import config
config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import OneDKSSCE

np.set_printoptions(precision=6)

# ---------------------------------------------------------------------------
# Configuration (config.toml)
# ---------------------------------------------------------------------------
AVAILABLE_FUNCTIONALS = ("SCE", "vcond", "vresp", "LDA", "Hartree", "EXX")
SCE_FAMILY = ("SCE", "vcond", "vresp")   # ensemble machinery + SCE energy
AVAILABLE_INTERACTIONS = ("softCoulomb", "absCoulomb", "wireCoulomb")
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "config.toml")

functional = "SCE"              # current functional (set by main)


def load_config(path=DEFAULT_CONFIG):
    """Read a TOML settings file, validate it and (re)build the module
    state.  Every parameter of the calculation is set here."""
    global nbPts, b, wint, N, functionals
    global max_cycle, conv_tol, conv_accept, diis_space, diis_engage, mixing
    global aufbau_max_cycle, aufbau_mixing, level_shift, q_scan, align_tol

    with open(path, "rb") as f:
        cfg = tomllib.load(f)

    system = cfg["system"]
    N = system["N"]
    assert N == 2, "the energy expressions assume a 2-electron system"
    b = system["b"]
    interaction = system["interaction"]
    if interaction not in AVAILABLE_INTERACTIONS:
        raise ValueError(f"unknown interaction {interaction!r}; "
                         f"choose from {AVAILABLE_INTERACTIONS}")
    wint = getattr(OneDKSSCE, interaction)

    functionals = tuple(cfg["functionals"]["run"])
    unknown = [f for f in functionals if f not in AVAILABLE_FUNCTIONALS]
    if unknown:
        raise ValueError(f"unknown functional(s) {unknown}; "
                         f"choose from {AVAILABLE_FUNCTIONALS}")

    wells = cfg["wells"]
    if not (len(wells["softening"]) == len(wells["depth"]) == 2):
        raise ValueError("wells.softening and wells.depth must have 2 entries "
                         "(wells at -R and +R)")

    grid = cfg["grid"]
    nbPts = grid["nbPts"]

    scf = cfg["scf"]
    max_cycle = scf["max_cycle"]        # inner SCF cycles
    conv_tol = scf["conv_tol"]          # target on the density change
    conv_accept = scf["conv_accept"]    # acceptance threshold for the best
                                        # iterate; exactly at the level
                                        # degeneracy the grid SCE potential
                                        # limits self-consistency to ~1e-7
    diis_space = scf["diis_space"]
    diis_engage = scf["diis_engage"]    # engage DIIS below this drho; in the
                                        # earlier nonlinear regime plain
                                        # damping is more robust, in the
                                        # linear regime DIIS kills the soft
                                        # charge-transfer mode
    mixing = scf["mixing"]              # linear mixing before DIIS engages

    aufbau = scf["aufbau"]              # closed-shell SCF (LDA/Hartree/EXX)
    aufbau_max_cycle = aufbau["max_cycle"]
    aufbau_mixing = aufbau["mixing"]
    level_shift = aufbau["level_shift"] # keeps the occupied orbital from
                                        # flipping when HOMO and LUMO get
                                        # close (does not move the fixed point)

    ensemble = scf["ensemble"]          # SCE outer loop over the charge split
    q_scan = np.arange(0.0, N + 1e-9, ensemble["q_step"])
    align_tol = ensemble["align_tol"]   # bisection threshold on the q bracket
                                        # (E is stationary at q*, so the E
                                        # error is ~ curvature * align_tol^2)

    global out_figure, check_figure
    out_figure = cfg.get("output", {}).get("figure", "ks_comparison.png")
    check_figure = cfg.get("output", {}).get("check_figure",
                                             "schrodinger_check.png")

    # full parsed file, so companion scripts can read their own sections
    global config
    config = cfg

    set_geometry(R_new=wells["R"], site_softening_new=wells["softening"],
                 site_depth_new=wells["depth"], L_new=grid.get("L"))


def well_character():
    """'homonuclear' if the two wells are identical, else 'heteronuclear'."""
    return ("homonuclear"
            if site_softening[0] == site_softening[1]
            and site_depth[0] == site_depth[1]
            else "heteronuclear")

# ---------------------------------------------------------------------------
# Grid and one-body Hamiltonian
# ---------------------------------------------------------------------------
def set_geometry(R_new=None, site_softening_new=None, nbPts_new=None, L_new=None,
                 site_depth_new=None):
    """(Re)build the grid-dependent module state.  Companion scripts call
    this to change the geometry (e.g. to stretch the molecule)."""
    global R, site_softening, site_depth, nbPts, L, x, delta, weights
    global kin, vext, h1, ref_left, ref_right, _coulomb_matrix

    if R_new is not None:
        R = R_new
        L = 8 * (R + 10)
    if L_new is not None:
        L = L_new
    if site_softening_new is not None:
        site_softening = site_softening_new
    if site_depth_new is not None:
        site_depth = site_depth_new
    if nbPts_new is not None:
        nbPts = nbPts_new

    x = np.linspace(-L / 2, L / 2, nbPts)
    delta = x[1] - x[0]
    weights = np.full(nbPts, delta)

    kin = OneDKSSCE.kinetic(nbPts, delta, 1)
    # v_ext(x) = -sum_i Z_i w(|x - X_i|, b_i) with wells at -R and +R
    wells = list(zip((-R, R), site_softening, site_depth))
    vext = -sum(Z * np.asarray(wint(np.abs(x - X), bs)) for X, bs, Z in wells)
    h1 = kin + np.diag(vext)
    _coulomb_matrix = None

    # Ground states of the isolated wells: initial guess and references for
    # tracking the left/right frontier orbitals through the SCF.
    refs = []
    for X, bs, Z in wells:
        _, C = scipy.linalg.eigh(kin - np.diag(Z * np.asarray(wint(np.abs(x - X), bs))))
        refs.append(C[:, 0])
    ref_left, ref_right = refs


site_depth = [1.0, 1.0]
load_config()

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
    if functional in SCE_FAMILY:
        # vSCE = vcond + vresp: "vcond" and "vresp" run one component alone
        if functional == "vcond":
            return np.asarray(OneDKSSCE.compute_vcondsce_potential_gauge_zero(
                rho, weights, x, N, wint, b))
        vsce = np.asarray(OneDKSSCE.compute_kssce_potential_gauge_zero(
            rho, weights, x, N, wint, b))
        if functional == "vresp":
            return vsce - np.asarray(OneDKSSCE.compute_vcondsce_potential_gauge_zero(
                rho, weights, x, N, wint, b))
        return vsce
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
    if functional in SCE_FAMILY:
        # for vcond/vresp the potential is not the functional derivative of
        # any energy: the SCE interaction energy at the SCF density is
        # reported so the columns stay comparable across the family
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
def scf_ensemble():
    """SCE-family ensemble solution: scan the charge split q, then bisect
    eps_L(q) - eps_R(q) to the aligned point (Janak).  Falls back to the
    best integer occupation when the levels never align, to the best
    bracket endpoint when a bisection step stops converging, and to None
    when nothing in the scan converges at all."""
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
        return None

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
    aligned = True
    while hi["q"] - lo["q"] > align_tol:
        q_mid = 0.5 * (lo["q"] + hi["q"])
        mid = scf_fixed_q(q_mid)
        if mid is None or not mid["converged"]:
            # keep the best converged bracket endpoint instead of losing the
            # whole functional (vcond hits this in the heteronuclear run)
            aligned = False
            print(f"  inner SCF failed at q = {q_mid:.6f}; stopping the "
                  f"refinement at bracket width {hi['q'] - lo['q']:.2e}")
            break
        if mid["eps_l"] - mid["eps_r"] < 0:
            lo = mid
        else:
            hi = mid
    best = min(lo, hi, key=lambda r: r["E"])
    state = "Converged" if aligned else "Best available (alignment not refined)"
    print(f"{state}: q* = {best['q']:.8f}, eps_L - eps_R = "
          f"{best['eps_l'] - best['eps_r']: .3e}")
    return best


def run():
    print(f"KS-{functional}: {nbPts} points, L = {L}, R = {R}, b = {b}, N = {N}")
    print(f"Wells at +/-R ({well_character()}): softening {list(site_softening)}, "
          f"depth {list(site_depth)}")

    if functional not in SCE_FAMILY:
        res = scf_aufbau()
        state = "converged" if res["converged"] else "NOT converged"
        print(f"\nAufbau SCF {state} in {res['cycles']} cycles, "
              f"drho = {res['drho']:.1e}")
        return res

    res = scf_ensemble()
    if res is None:
        raise RuntimeError("no converged SCF in the q scan")
    return res


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


COLORS = {"SCE": "tab:red", "vcond": "tab:orange", "vresp": "tab:cyan",
          "LDA": "tab:green", "Hartree": "tab:blue", "EXX": "tab:purple"}


# ---------------------------------------------------------------------------
# Schroedinger-equation check on the converged orbitals (diagnostic only)
# ---------------------------------------------------------------------------
CHECK_TOL = 1e-3        # acceptable |H phi / phi - eps|, and the fixed y
                        # half-range of every check panel: the SCF acceptance
                        # drho < 1e-6 propagates to potential errors of only
                        # ~1e-5 (interaction kernel norm ~ 1/sqrt(b)), so an
                        # accepted run sits well inside this band and any
                        # visible structure marks a genuine failure
CHECK_MASK = 1e-4       # evaluate the ratio only where |phi| > mask * max|phi|


def schrodinger_check_data(res):
    """H phi / phi curves for the occupied orbitals of one converged result.

    Everything is rebuilt from the orbitals alone: density -> Hxc potential
    -> H = h1 + diag(v), then the pointwise ratio (H phi) / phi.  For an
    exact self-consistent eigenstate the ratio is flat at the orbital energy
    wherever the orbital is non-negligible.  Must be called while the module
    geometry matches the result (companion scripts call it inside their
    geometry loop)."""
    global functional
    functional_saved = functional
    functional = res["functional"]
    try:
        q = res["q"]
        if q is None:
            orbitals = [(r"\phi", res["orb_l"], res["eps_l"])]
            rho = N * res["orb_l"]**2 / weights
        else:
            orbitals = [(r"\phi_L", res["orb_l"], res["eps_l"]),
                        (r"\phi_R", res["orb_r"], res["eps_r"])]
            rho = (q * res["orb_l"]**2 + (N - q) * res["orb_r"]**2) / weights
        H = h1 + np.diag(hxc_potential(rho))
        curves = []
        for name, phi, eps in orbitals:
            mask = np.abs(phi) > CHECK_MASK * np.abs(phi).max()
            ratio = (H @ phi)[mask] / phi[mask]
            curves.append(dict(name=name, x=x[mask], ratio=ratio, eps=eps,
                               dev=np.max(np.abs(ratio - eps))))
    finally:
        functional = functional_saved
    return curves


def draw_check_panel(ax, res, curves):
    """One tile of the check board: the H phi / phi curves of one result,
    the orbital energies as dashed lines, the worst deviation in the title
    (red if it exceeds CHECK_TOL).  The y range is pinned to eps +/-
    CHECK_TOL: autoscale would magnify sub-tolerance noise into apparent
    structure, whereas inside a fixed band flat means converged and any
    visible wiggle or clipping is a deviation that actually matters."""
    color = COLORS.get(res["functional"])
    for c, ls in zip(curves, ("-", ":")):
        ax.axhline(c["eps"], color='0.3', lw=0.8, ls='--')
        ax.plot(c["x"], c["ratio"], color=color, ls=ls,
                label=f'$H{c["name"]}/{c["name"]}$  dev {c["dev"]:.1e}')
    ax.set_ylim(min(c["eps"] for c in curves) - CHECK_TOL,
                max(c["eps"] for c in curves) + CHECK_TOL)
    ax.ticklabel_format(useOffset=False, axis='y')
    dev = max(c["dev"] for c in curves)
    occ = ("aufbau" if res["q"] is None else f"q = {res['q']:.3f}")
    state = "" if res["converged"] else ", SCF NOT converged"
    ax.set_title(f'{res["functional"]} ({occ}{state})  '
                 f'max dev {dev:.1e}',
                 fontsize=9, color="tab:red" if dev > CHECK_TOL else "black")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=7)


def plot_schrodinger_check(results, show):
    """One board with all the H phi = eps phi checks, one panel per result."""
    ncols = min(3, len(results))
    nrows = (len(results) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows),
                             squeeze=False)
    for ax in axes.ravel()[len(results):]:
        ax.set_axis_off()
    for ax, res in zip(axes.ravel(), results):
        draw_check_panel(ax, res, schrodinger_check_data(res))
        ax.set_xlim(-R - 20, R + 20)
    for ax in axes[-1, :]:
        ax.set_xlabel('x')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$H\phi/\phi$')
    fig.suptitle(r'Schroedinger check: $H\phi/\phi$ from the rebuilt '
                 r'Hamiltonian (flat at $\epsilon$ = converged; '
                 rf'y range fixed to $\epsilon \pm$ {CHECK_TOL:g})', fontsize=11)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), check_figure)
    fig.savefig(out, dpi=150)
    print(f"Schroedinger check board saved to {out}")
    if show:
        plt.show()


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
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_figure)
    fig.savefig(out, dpi=150)
    print(f"\nComparison plot saved to {out}")
    if show:
        plt.show()


def main(show=False):
    global functional
    results = []
    for functional in functionals:
        print("\n" + "=" * 70)
        try:
            res = run()
        except RuntimeError as exc:
            print(f"KS-{functional} failed: {exc}")
            continue
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
    plot_schrodinger_check(results, show)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--show"]
    if args:
        load_config(args[0])
    main(show="--show" in sys.argv)
