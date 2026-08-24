#!/usr/bin/env python3
"""
H Lyman-alpha Stark-Zeeman Splitting at B = 500 T

Higher-field companion to test_lyman_alpha.py (100 T), loosely inspired by
the Figure 3 benchmark of Ferri, Peyrusse & Calisti (2022,
DOI: 10.1063/5.0058552) — that figure used C VI (Z = 6); this script uses
H (Z = 1) throughout.

Conditions:
  - Radiator: H (Z = 1), Lyman-alpha (n=2 → n=1)
  - B = 500 T
  - Ne = 5e25 m^-3
  - Te = 100 eV
  - No convolution (pure Stark-Zeeman profile)

Energy range ±0.20 eV around E0.  The pi component is multiplied by -1 for
visual separation of polarizations.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.utils import energy_ev_to_wavelength_nm, RYDBERG_EV
from starkzee.static_profile import calculate_static_profile


def run_test():
    # ── Physical parameters ────────────────────────────────────────────────
    Z  = 1        # Hydrogen
    B  = 500.0    # Tesla
    Ne = 5e25     # m^-3
    Te = 100.0    # eV

    # Unperturbed Lyman-alpha for H (n=2 → n=1)
    E0 = (Z**2) * RYDBERG_EV * (1.0 - 1.0/4.0)   # = (3/4) Z² Ry

    # Energy grid: ±0.20 eV around E0
    detuning_grid = np.linspace(-0.2, 0.2, 600)
    energies_ev   = E0 + detuning_grid

    print("=" * 70)
    print("test_lyman_alpha_500T: H Lyman-alpha — B = 500 T")
    print(f"  Z={Z}, B={B} T, Ne={Ne:.1e} m^-3, Te={Te} eV")
    print(f"  E0 = {E0:.6f} eV  ({energy_ev_to_wavelength_nm(E0):.6f} nm)")
    print("=" * 70)

    print("-> Computing static Stark-Zeeman profile (unconvolved)…")
    pi, sig_plus, sig_minus = calculate_static_profile(
        n_u=2, n_l=1, Z=Z, B=B, Ne_m3=Ne, Te_ev=Te,
        energies_ev=energies_ev,
        num_f=30, num_mu=10,
        use_screening=True,
        quadratic_zeeman=True,
        frequency_dependent_width=True,
    )

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(detuning_grid, sig_plus,  color='#10b981', linewidth=2.2,
            label=r'$\sigma_+$  ($q = +1$)')
    ax.plot(detuning_grid, sig_minus, color='#f59e0b', linewidth=2.2,
            label=r'$\sigma_-$  ($q = -1$)')
    ax.plot(detuning_grid, -pi,       color='#ef4444', linewidth=2.2,
            linestyle='--', label=r'$-\pi$  ($q = 0$)')

    ax.axhline(0, color='gray', linestyle=':', linewidth=0.9)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6,
               label=f'$E_0 = {E0:.4f}$ eV')

    ax.set_title(
        f"SZ Lyman-$\\alpha$ of H  (B = {B:.0f} T, "
        f"$N_e = {Ne:.0e}$ m$^{{-3}}$, $T_e = {Te:.0f}$ eV)\n"
        r"Pure Stark-Zeeman profile",
        fontsize=12, pad=12,
    )
    ax.set_xlabel(r"Energy detuning from $E_0$ (eV)", fontsize=11)
    ax.set_ylabel("Intensity (arb. units)", fontsize=11)
    ax.set_xlim(detuning_grid.min(), detuning_grid.max())

    ax.grid(True, linestyle=':', alpha=0.55)
    ax.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    plot_file = os.path.join(os.path.dirname(__file__), "lyman_alpha_500T.png")
    plt.savefig(plot_file, dpi=300)
    print(f"Plot saved → {plot_file}")
    plt.show()


if __name__ == "__main__":
    run_test()
