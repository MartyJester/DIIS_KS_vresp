#!/usr/bin/env python
"""Stretching a two-well molecule: ground state vs bond distance.

All settings come from the [stretch] section of config.toml: the bond
distances, the per-well softening and depth (identical wells = homonuclear,
where by symmetry the dissociated molecule holds exactly one electron per
well), the grid spacing and the functionals to compare.  For each distance
R and each functional two candidate solutions are converged with the
machinery of main_fixed.py:

  * the closed-shell aufbau state, occupations (2, 0);
  * the q = 1 ensemble state, occupations (1, 1);

and the converged candidate with the lowest energy is the ground state.
The three default distances illustrate the three regimes of the SCE
solution for the homonuclear molecule:

  R = 2   covalent bond      aufbau ground state, open gap,
                             large midbond density
  R = 4   transition         the aufbau SCE SCF stops converging (charge-
                             sloshing instability), the ensemble takes over,
                             the SCE barrier cuts the bond
  R = 10  dissociated        one electron per well, gap ~ 0, two
                             independent "atoms"

Output: 2 x n_distances figure (top: densities, bottom: Hxc potentials),
one column per distance, occupations of each solution in the legend.

Run:  python homonuclear_stretch.py [other_config.toml] [--show]
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main_fixed as m


def stretch_settings():
    """Read the [stretch] section of the loaded config, with the original
    homonuclear study as the default for anything left unspecified."""
    s = m.config.get("stretch", {})
    settings = dict(
        distances=s.get("distances", [2.0, 4.0, 10.0]),
        softening=s.get("softening", [0.7, 0.7]),
        depth=s.get("depth", [1.0, 1.0]),
        dx=s.get("dx", 0.2667),        # grid spacing, fixed while L grows
        functionals=tuple(s.get("functionals", m.functionals)),
        figure=s.get("figure", "homonuclear_stretch.png"),
    )
    labels = s.get("labels", [])
    settings["labels"] = (labels if len(labels) == len(settings["distances"])
                          else [""] * len(settings["distances"]))
    unknown = [f for f in settings["functionals"]
               if f not in m.AVAILABLE_FUNCTIONALS]
    if unknown:
        raise ValueError(f"unknown functional(s) {unknown} in [stretch]; "
                         f"choose from {m.AVAILABLE_FUNCTIONALS}")
    return settings


def solve(functional):
    """Ground state of the current geometry for one functional: the
    lowest-energy converged candidate among aufbau and the q = 1 ensemble."""
    m.functional = functional

    candidates = []
    aufbau = m.scf_aufbau()
    if aufbau["converged"]:
        aufbau["occ"] = (2, 0)
        aufbau["gap"] = aufbau["eps_r"] - aufbau["eps_l"]
        candidates.append(aufbau)
    ensemble = m.scf_fixed_q(1.0)
    if ensemble is not None and ensemble["converged"]:
        ensemble["occ"] = (1, 1)
        ensemble["gap"] = abs(ensemble["eps_l"] - ensemble["eps_r"])
        candidates.append(ensemble)
    if not candidates:
        return None

    best = min(candidates, key=lambda r: r["E"])
    best["functional"] = functional
    best["vhxc"] = m.hxc_potential(best["rho"])
    return best


def main(show=False):
    s = stretch_settings()

    columns = []
    for icol, (R, regime) in enumerate(zip(s["distances"], s["labels"])):
        L = 8 * (R + 10)
        m.set_geometry(R_new=R, site_softening_new=s["softening"],
                       site_depth_new=s["depth"],
                       nbPts_new=int(round(L / s["dx"])))
        if icol == 0:
            print(f"Stretch study ({m.well_character()} wells: "
                  f"softening {s['softening']}, depth {s['depth']})")
            print(f"{'R':>5} {'functional':>10} {'occ':>7} {'E_tot':>13} "
                  f"{'gap':>10} {'drho':>9}")
        results = []
        for functional in s["functionals"]:
            res = solve(functional)
            if res is None:
                print(f"{R:5.1f} {functional:>10}   [no converged solution]")
                continue
            results.append(res)
            print(f"{R:5.1f} {functional:>10} {str(res['occ']):>7} "
                  f"{res['E']:13.7f} {res['gap']:10.2e} {res['drho']:9.1e}")
        columns.append(dict(R=R, regime=regime, x=m.x.copy(),
                            vext=m.vext.copy(), results=results))

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
    fig.suptitle(f'{m.well_character().capitalize()} stretch: '
                 + ' vs '.join(s["functionals"])
                 + ' (occupations of the ground state in the legend)')
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), s["figure"])
    fig.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")
    if show:
        plt.show()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--show"]
    if args:
        m.load_config(args[0])
    main(show="--show" in sys.argv)
