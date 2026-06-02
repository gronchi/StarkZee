#!/usr/bin/env python3
"""
Test and Verification Script: C VI Lyman-alpha Stark-Zeeman Splitting (Figure 3 Benchmark)
========================================================================================

Reproduces Figure 3 of:
  Stark-Zeeman line-shape model for multi-electron radiators in hot dense plasmas
  subjected to large magnetic fields (AIP, 2021, DOI: 10.1063/5.0058552)

Conditions:
  - Radiator: C VI (Z = 6), hydrogen-like carbon, Lyman-alpha (n=2 -> n=1)
  - B = 100 T
  - Ne = 5e25 m^-3
  - Te = 100 eV
  - No Doppler or instrumental convolution (pure Stark-Zeeman profile)

The pi component is multiplied by -1 to match the paper's visual convention.
The x-axis is energy detuning (eV) from the unperturbed transition center E0.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure starkzee package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.utils import energy_ev_to_wavelength_nm, RYDBERG_EV
from starkzee.static_profile import calculate_static_profile


def run_lyman_alpha_test():
    # ── Physical parameters matching paper Fig. 3 ──────────────────────────
    Z  = 6        # Carbon VI
    B  = 100.0    # Tesla
    Ne = 5e25     # m^-3
    Te = 100.0    # eV

    # Unperturbed Lyman-alpha transition energy for C VI: n=2 -> n=1
    E_upper = -(Z**2) * RYDBERG_EV / 4.0   # n=2 level
    E_lower = -(Z**2) * RYDBERG_EV          # n=1 level
    E0 = E_upper - E_lower                  # ~367.35 eV for C VI

    # Energy grid: ±0.20 eV around E0
    detuning_grid = np.linspace(-0.2, 0.2, 500)
    energies_ev   = E0 + detuning_grid

    print("=" * 70)
    print("test_lyman_alpha: C VI Lyman-alpha Stark-Zeeman — Fig. 3 Benchmark")
    print(f"  Z={Z}, B={B} T, Ne={Ne:.1e} m^-3, Te={Te} eV")
    print(f"  Unperturbed center E0 = {E0:.6f} eV  "
          f"({energy_ev_to_wavelength_nm(E0):.6f} nm)")
    print("=" * 70)

    # ── Compute static (unconvolved) Stark-Zeeman profile ──────────────────
    print("-> Computing static Stark-Zeeman profile (unconvolved)…")
    pi, sig_plus, sig_minus = calculate_static_profile(
        n_u=2, n_l=1, Z=Z, B=B, Ne_m3=Ne, Te_ev=Te,
        energies_ev=energies_ev,
        num_f=30, num_mu=10,
        use_screening=True,
        quadratic_zeeman=True,
        frequency_dependent_width=True,
    )

    # ── Plot — Figure 3 layout ──────────────────────────────────────────────
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
        f"SZ Lyman-$\\alpha$ of C VI  (B = {B:.0f} T, "
        f"$N_e = {Ne:.0e}$ cm$^{{-3}}$, $T_e = {Te:.0f}$ eV)\n"
        r"Pure Stark-Zeeman profile — reproduces Fig. 3",
        fontsize=12, pad=12,
    )
    ax.set_xlabel(r"Energy detuning from $E_0$ (eV)", fontsize=11)
    ax.set_ylabel("Intensity (arb. units)", fontsize=11)
    ax.set_xlim(-0.2, 0.2)

    ax.grid(True, linestyle=':', alpha=0.55)
    ax.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    plot_file = os.path.join(os.path.dirname(__file__), "starkzee_profile.png")
    plt.savefig(plot_file, dpi=300)
    print(f"Plot saved → {plot_file}")
    plt.show()


if __name__ == "__main__":
    run_lyman_alpha_test()
