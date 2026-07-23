#!/usr/bin/env python
"""Stretching a two-well molecule: ground state vs bond distance.

Generalisation of the original homonuclear study to any pair of wells.
Every [[stretch.studies]] entry of config.toml is one study (the default
config ships a homonuclear and a heteronuclear molecule); plain keys of
[stretch] are defaults shared by all studies.  For each bond distance R
and each functional the candidate solutions are converged with the
machinery of main_fixed.py:

  * the closed-shell aufbau state, occupations (2, 0);
  * an ensemble state: for identical wells the charge split is pinned to
    q = 1 by symmetry; for unequal wells the SCE-family split q* is found
    by scanning q and aligning the frontier levels eps_L(q) = eps_R(q),
    exactly as in main_fixed.py (weakly-correlated functionals keep the
    q = 1 candidate);

and the converged candidate with the lowest energy is the ground state.

Output per study: a 2 x n_distances figure (top: densities, bottom: Hxc
potentials, occupations in the legend) and the Schroedinger check board
of main_fixed.py (functionals x distances) — H phi / phi from the
Hamiltonian rebuilt out of the converged orbitals; diagnostic only,
nothing acts on it.

Run:  python stretch.py [other_config.toml] [--show]
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main_fixed as m


def stretch_studies():
    """The studies to run: every [[stretch.studies]] entry merged over the
    plain keys of [stretch] (shared defaults), or the flat [stretch]
    section alone when no studies array is given (the original
    homonuclear-study layout)."""
    base = dict(m.config.get("stretch", {}))
    entries = base.pop("studies", [{}])
    studies = []
    for entry in entries:
        s = {**base, **entry}
        name = s.get("name", "stretch")
        settings = dict(
            name=name,
            distances=s.get("distances", [2.0, 4.0, 10.0]),
            softening=s.get("softening", [0.7, 0.7]),
            depth=s.get("depth", [1.0, 1.0]),
            dx=s.get("dx", 0.2667),        # grid spacing, fixed while L grows
            functionals=tuple(s.get("functionals", m.functionals)),
            figure=s.get("figure", f"{name}_stretch.png"),
            check_figure=s.get("check_figure", f"{name}_stretch_check.png"),
        )
        labels = s.get("labels", [])
        settings["labels"] = (labels if len(labels) == len(settings["distances"])
                              else [""] * len(settings["distances"]))
        unknown = [f for f in settings["functionals"]
                   if f not in m.AVAILABLE_FUNCTIONALS]
        if unknown:
            raise ValueError(f"unknown functional(s) {unknown} in study "
                             f"{name!r}; choose from {m.AVAILABLE_FUNCTIONALS}")
        studies.append(settings)
    return studies


def _fmt(occ):
    return f"{occ:.3f}".rstrip('0').rstrip('.')


def solve(functional):
    """Ground state of the current geometry for one functional: the
    lowest-energy converged candidate among aufbau and the ensemble
    solution (q = 1 by symmetry for identical wells, the level-aligning
    q* of main_fixed.scf_ensemble otherwise)."""
    m.functional = functional

    candidates = []
    aufbau = m.scf_aufbau()
    if aufbau["converged"]:
        aufbau["occ"] = "(2, 0)"
        aufbau["gap"] = aufbau["eps_r"] - aufbau["eps_l"]
        candidates.append(aufbau)
    if functional in m.SCE_FAMILY and m.well_character() == "heteronuclear":
        ensemble = m.scf_ensemble()
    else:
        ensemble = m.scf_fixed_q(1.0)
    if ensemble is not None and ensemble["converged"]:
        q = ensemble["q"]
        ensemble["occ"] = f"({_fmt(q)}, {_fmt(m.N - q)})"
        ensemble["gap"] = abs(ensemble["eps_l"] - ensemble["eps_r"])
        candidates.append(ensemble)
    if not candidates:
        return None

    best = min(candidates, key=lambda r: r["E"])
    best["functional"] = functional
    best["vhxc"] = m.hxc_potential(best["rho"])
    # the check needs the current geometry: compute it here, plot later
    best["check"] = m.schrodinger_check_data(best)
    return best


def plot_study(columns, s, show):
    fig, axes = plt.subplots(2, len(columns), figsize=(13, 7),
                             sharex='col', sharey='row', squeeze=False)
    for icol, col in enumerate(columns):
        ax_rho, ax_pot = axes[0, icol], axes[1, icol]
        ax_rho.plot(col["x"], col["vext"], color='0.5', ls='--',
                    label=r'$v_{ext}$')
        for res in col["results"]:
            color = m.COLORS.get(res["functional"])
            label = f"{res['functional']} {res['occ']}"
            ax_rho.plot(col["x"], res["rho"], color=color, label=label)
            ax_pot.plot(col["x"], res["vhxc"], color=color, label=label)
        R = col["R"]
        title = f'R = {R:g}' + (f'  ({col["regime"]})' if col["regime"] else '')
        ax_rho.set(title=title, xlim=(-R - 14, R + 14))
        ax_pot.set(xlabel='x')
        for ax in (ax_rho, ax_pot):
            ax.grid()
        ax_rho.legend(fontsize=8, loc='lower left')
        ax_pot.legend(fontsize=8, loc='upper left')
    axes[0, 0].set_ylabel(r'density, $v_{ext}$')
    axes[1, 0].set_ylabel(r'$v_{Hxc}$')
    fig.suptitle(f'{s["name"].capitalize()} stretch ({m.well_character()} '
                 'wells): ' + ' vs '.join(s["functionals"])
                 + ' (occupations of the ground state in the legend)')
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), s["figure"])
    fig.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


def plot_check(columns, s, show):
    """One board with every Schroedinger check: functionals x distances."""
    functionals = s["functionals"]
    fig, axes = plt.subplots(len(functionals), len(columns),
                             figsize=(4.6 * len(columns),
                                      2.9 * len(functionals)),
                             squeeze=False)
    for icol, col in enumerate(columns):
        for irow, functional in enumerate(functionals):
            ax = axes[irow, icol]
            res = next((r for r in col["results"]
                        if r["functional"] == functional), None)
            if res is None:
                ax.text(0.5, 0.5, f"{functional}: no converged solution",
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
                continue
            m.draw_check_panel(ax, res, res["check"])
            R = col["R"]
            ax.set_xlim(-R - 14, R + 14)
            ax.set_title(f'R = {R:g}: ' + ax.get_title(), fontsize=9,
                         color=ax.title.get_color())
    for ax in axes[-1, :]:
        ax.set_xlabel('x')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$H\phi/\phi$')
    fig.suptitle(f'{s["name"].capitalize()} stretch — Schroedinger check: '
                 r'$H\phi/\phi$ from the rebuilt Hamiltonian '
                 r'(flat at $\epsilon$ = converged; '
                 rf'y range fixed to $\epsilon \pm$ {m.CHECK_TOL:g})',
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       s["check_figure"])
    fig.savefig(out, dpi=150)
    print(f"Schroedinger check board saved to {out}")


def run_study(s, show):
    columns = []
    for icol, (R, regime) in enumerate(zip(s["distances"], s["labels"])):
        L = 8 * (R + 10)
        m.set_geometry(R_new=R, site_softening_new=s["softening"],
                       site_depth_new=s["depth"],
                       nbPts_new=int(round(L / s["dx"])))
        if icol == 0:
            print(f"Stretch study {s['name']!r} ({m.well_character()} wells: "
                  f"softening {s['softening']}, depth {s['depth']})")
            print(f"{'R':>5} {'functional':>10} {'occ':>15} {'E_tot':>13} "
                  f"{'gap':>10} {'drho':>9}")
        results = []
        for functional in s["functionals"]:
            res = solve(functional)
            if res is None:
                print(f"{R:5.1f} {functional:>10}   [no converged solution]")
                continue
            results.append(res)
            print(f"{R:5.1f} {functional:>10} {res['occ']:>15} "
                  f"{res['E']:13.7f} {res['gap']:10.2e} {res['drho']:9.1e}")
        columns.append(dict(R=R, regime=regime, x=m.x.copy(),
                            vext=m.vext.copy(), results=results))

    plot_study(columns, s, show)
    plot_check(columns, s, show)


def main(show=False):
    for s in stretch_studies():
        print("\n" + "=" * 70)
        run_study(s, show)
    if show:
        plt.show()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--show"]
    if args:
        m.load_config(args[0])
    main(show="--show" in sys.argv)
