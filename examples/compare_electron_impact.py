#!/usr/bin/env python3
"""
compare_electron_impact.py
========================
Compare the electron-impact G-functions and physical broadening widths (HWHM)
between StarkZee and ZEST.

This script imports ZEST (from a neighboring folder) and displays a 2x2 comparison:
1. ZEST Figure 1 Reproduction: Dimensionless G(Δω) vs reduced detuning Δω / ω_p
   with kappa_m * lambda_D = 30.0. An overlay curve demonstrates the exact
   mathematical equivalence of StarkZee's gbk_model under parameter mapping.
2. Physical G-Function Comparison: Standard physical conditions (density Ne,
   temperature Te) comparing ZEST and StarkZee cutoffs.
3. HWHM Widths vs. Detuning: Physical electron impact width HWHM in eV,
   highlighting the effect of StarkZee's strong-collision constant C_n.
4. Summary Text: Explaining the physical differences between the models.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

# Insert paths to import both local packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Find and import ZEST
try:
    import zest
except ImportError:
    # Try neighboring folders
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../zest")))
    try:
        import zest
    except ImportError:
        print("Error: Could not import 'zest'. Make sure the zest repository is at ../zest relative to StarkZee.")
        sys.exit(1)

import starkzee.broadening as sz_broad
from starkzee.utils import RYDBERG_EV

def run_comparison():
    print("Running StarkZee vs ZEST comparison...")
    
    # ── 1. Physical Plasma Parameters ──
    ne = 1e23  # Electron density (m^-3)
    te_k = 1e5 # Temperature (Kelvin)
    Z = 1      # Core charge (1 for Hydrogen)
    n = 2      # Upper principal quantum number (n=2 for Lyman-alpha)
    B = 0.0    # Magnetic field (T)

    # Convert temperature to eV
    Te_ev = te_k * const.k / const.e

    # ── 2. Cutoffs & Parameters for Both Packages ──
    lambda_D = zest.get_classical_debye_length(ne, te_k)
    omega_p = zest.electron.get_plasma_frequency(ne)
    
    # Zest physical cutoff
    kappa_m_phys = zest.electron.get_cutoff_kappa_m(Z, n, te_k)
    
    # StarkZee physical cutoffs
    omega_e = sz_broad.calculate_configuration_frequency(ne, Te_ev)
    omega_c_sz_rad = max(omega_p, sz_broad.calculate_larmor_frequency(B), omega_e)
    
    # Conversions to eV
    omega_p_ev = omega_p * const.hbar / const.e
    omega_c_sz_ev = omega_c_sz_rad * const.hbar / const.e

    # ── 3. Evaluate G-Functions for ZEST Figure 1 (Dimensionless) ──
    # Reduced detuning x-axis grid: dw / omega_p from 0.0 to 15.0
    w_red = np.linspace(0.0, 15.0, 300)
    dw_grid = w_red * omega_p
    
    # For Figure 1, ZEST forces kappa_m * lambda_D = 30.0
    kappa_m_lambda_D_fig1 = 30.0
    kappa_m_fig1 = kappa_m_lambda_D_fig1 / lambda_D
    
    g_lee_fig1 = zest.electron.g_lee(dw_grid, ne, te_k, lambda_D, kappa_m_fig1)
    g_gbk_fig1 = zest.electron.g_gbk(dw_grid, ne, te_k, lambda_D, kappa_m_fig1)
    g_dufty_fig1 = zest.electron.g_dufty(dw_grid, ne, te_k, lambda_D, kappa_m_fig1)

    # Map StarkZee to ZEST Figure 1 settings to verify mathematical overlay
    # Te_ev_mapped aligns StarkZee's y parameter to Zest's arg under kappa_m_lambda_D = 30.0
    Te_ev_mapped = (kappa_m_lambda_D_fig1 * omega_p_ev)**2 * (n**2 / (2*Z))**2 / RYDBERG_EV
    delta_omega_ev = dw_grid * const.hbar / const.e
    g_sz_gbk_mapped = sz_broad.gbk_model(delta_omega_ev, omega_p_ev, Te_ev_mapped, Z, n=n)

    # ── 4. Evaluate Physical G-Functions (No forced ratios) ──
    g_lee_phys = zest.electron.g_lee(dw_grid, ne, te_k, lambda_D, kappa_m_phys)
    g_gbk_phys = zest.electron.g_gbk(dw_grid, ne, te_k, lambda_D, kappa_m_phys)
    g_dufty_phys = zest.electron.g_dufty(dw_grid, ne, te_k, lambda_D, kappa_m_phys)
    
    # StarkZee physical G-functions:
    g_sz_gbk_wp_only_phys = sz_broad.gbk_model(delta_omega_ev, omega_p_ev, Te_ev, Z, n=n)
    g_sz_gbk_full_phys = sz_broad.gbk_model(delta_omega_ev, omega_c_sz_ev, Te_ev, Z, n=n)

    # ── 5. Evaluate Physical HWHM Widths ──
    r2_avg = sum(
        (2*l + 1) * (n**2 / (2.0 * Z**2)) * (5.0*n**2 + 1.0 - 3.0*l*(l + 1.0))
        for l in range(n)
    ) / n**2
    
    # ZEST HWHM Widths (rad/s -> converted to eV)
    w_zest_gbk_ev = zest.electron.get_electron_width(ne, te_k, Z, n, dw_grid, model='gbk', r_sq=r2_avg) * const.hbar / const.e
    w_zest_lee_ev = zest.electron.get_electron_width(ne, te_k, Z, n, dw_grid, model='lee', r_sq=r2_avg) * const.hbar / const.e
    w_zest_dufty_ev = zest.electron.get_electron_width(ne, te_k, Z, n, dw_grid, model='dufty', r_sq=r2_avg) * const.hbar / const.e
    
    # StarkZee HWHM Width (eV, includes strong collision constant Cn=1.5 and max cutoff)
    w_sz_gbk_ev = sz_broad.electron_impact_width(delta_omega_ev, ne, Te_ev, B, Z, n=n)

    # StarkZee intermediate HWHM Widths to show the individual effects of Cn and omega_e
    if n <= 2:
        Cn = 1.5
    elif n == 3:
        Cn = 1.0
    elif n == 4:
        Cn = 0.75
    elif n == 5:
        Cn = 0.5
    else:
        Cn = 0.40
    prefactor = sz_broad.calculate_electron_impact_prefactor(ne, Te_ev)
    w_sz_wp_only_no_Cn_ev = prefactor * r2_avg * g_sz_gbk_wp_only_phys
    w_sz_wp_only_with_Cn_ev = prefactor * r2_avg * (Cn + g_sz_gbk_wp_only_phys)

    # ── 6. Print Summary ──
    print(f"\n--- Parameter Summary ---")
    print(f"Density Ne:                 {ne:.2e} m^-3")
    print(f"Temperature Te:             {te_k:.2e} K ({Te_ev:.3f} eV)")
    print(f"Upper level n:              {n}")
    print(f"Debye length lambda_D:      {lambda_D:.6e} m")
    print(f"Plasma frequency omega_p:   {omega_p:.6e} rad/s")
    print(f"Config frequency omega_e:   {omega_e:.6e} rad/s")
    print(f"StarkZee cutoff omega_c:    {omega_c_sz_rad:.6e} rad/s (dominated by omega_e)")
    print(f"Zest physical cutoff km:    {kappa_m_phys:.6e} m^-1 (phys km*lD = {kappa_m_phys * lambda_D:.2f})")
    print(f"StarkZee <r^2>_n:           {r2_avg:.2f} a0^2")
    print(f"-------------------------")
    print(f"At Line Center (dw = 0):")
    print(f"  ZEST G_GBK (Fig 1 model): {g_gbk_fig1[0]:.4f}")
    print(f"  StarkZee G_GBK (mapped):  {g_sz_gbk_mapped[0]:.4f}  (Difference: {abs(g_gbk_fig1[0]-g_sz_gbk_mapped[0]):.2e})")
    print(f"  ZEST G_GBK (physical):    {g_gbk_phys[0]:.4f}")
    print(f"  StarkZee G_GBK (physical): {g_sz_gbk_full_phys[0]:.4f}")
    print(f"  ZEST HWHM Width (GBK):    {w_zest_gbk_ev[0]*1e3:.4f} meV")
    print(f"  StarkZee HWHM Width:      {w_sz_gbk_ev[0]*1e3:.4f} meV")
    print(f"-------------------------\n")

    # ── 7. Create 2x2 Plots ───────────────────────────────────────────────────
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot (0,0): ZEST Figure 1 G-Function Reproduction (kappa_m * lD = 30)
    axes[0, 0].plot(w_red, g_dufty_fig1, label='ZEST Dufty RPA', color='#1f77b4', lw=2.5)
    axes[0, 0].plot(w_red, g_lee_fig1, label='ZEST Lee', color='#ff7f0e', lw=2.0, ls='--')
    axes[0, 0].plot(w_red, g_gbk_fig1, label='ZEST GBK', color='#2ca02c', lw=2.0, ls=':')
    axes[0, 0].plot(w_red, g_sz_gbk_mapped, label='StarkZee GBK (mapped)', color='#d62728', lw=1.8, ls='-.')
    
    axes[0, 0].set_title(r"ZEST Fig 1 Reproduction ($\kappa_m \lambda_D = 30.0$)", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel(r"$\Delta\omega / \omega_p$", fontsize=10)
    axes[0, 0].set_ylabel(r"$G(\Delta\omega)$", fontsize=10)
    axes[0, 0].set_xlim(0.0, 15.0)
    axes[0, 0].legend(fontsize=9, frameon=True, facecolor='white')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)

    # Plot (0,1): Physical G-Function Comparison
    axes[0, 1].plot(w_red, g_dufty_phys, label='ZEST Dufty RPA', color='#1f77b4', lw=2.5)
    axes[0, 1].plot(w_red, g_lee_phys, label='ZEST Lee', color='#ff7f0e', lw=2.0, ls='--')
    axes[0, 1].plot(w_red, g_gbk_phys, label='ZEST GBK (wp cutoff)', color='#2ca02c', lw=2.0, ls=':')
    axes[0, 1].plot(w_red, g_sz_gbk_wp_only_phys, label='StarkZee GBK (wp cutoff)', color='#d62728', lw=1.8, ls='-.')
    axes[0, 1].plot(w_red, g_sz_gbk_full_phys, label='StarkZee GBK (full physical cutoff)', color='#9467bd', lw=2.5)

    axes[0, 1].set_title("Physical G-Functions (no forced ratios)", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel(r"$\Delta\omega / \omega_p$", fontsize=10)
    axes[0, 1].set_ylabel(r"$G(\Delta\omega)$", fontsize=10)
    axes[0, 1].set_xlim(0.0, 15.0)
    axes[0, 1].legend(fontsize=9, frameon=True, facecolor='white')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    # Plot (1,0): HWHM Width vs. Detuning
    axes[1, 0].plot(delta_omega_ev, w_zest_dufty_ev * 1e3, label='ZEST Dufty RPA HWHM', color='#1f77b4', lw=2.5)
    axes[1, 0].plot(delta_omega_ev, w_zest_lee_ev * 1e3, label='ZEST Lee HWHM', color='#ff7f0e', lw=2.0, ls='--')
    axes[1, 0].plot(delta_omega_ev, w_zest_gbk_ev * 1e3, label='ZEST GBK HWHM', color='#2ca02c', lw=2.0, ls=':')
    axes[1, 0].plot(delta_omega_ev, w_sz_wp_only_no_Cn_ev * 1e3, label='StarkZee (wp cutoff, no Cn)', color='#d62728', lw=1.8, ls='-.')
    axes[1, 0].plot(delta_omega_ev, w_sz_wp_only_with_Cn_ev * 1e3, label='StarkZee (wp cutoff, with Cn)', color='#e377c2', lw=1.8, ls='--')
    axes[1, 0].plot(delta_omega_ev, w_sz_gbk_ev * 1e3, label='StarkZee (full cutoff, with Cn)', color='#9467bd', lw=2.5)

    axes[1, 0].set_title("Physical Electron Broadening Width HWHM", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel(r"Detuning $\Delta E$ (eV)", fontsize=10)
    axes[1, 0].set_ylabel(r"Half-Width at Half-Maximum $W_e$ (meV)", fontsize=10)
    axes[1, 0].legend(fontsize=9, frameon=True, facecolor='white')
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)

    # Plot (1,1): Descriptive Text / Explanations
    axes[1, 1].axis('off')
    text_info = (
        "Key Model Differences & Insights:\n\n"
        "1. Dimensionless Equivalence (Top-Left):\n"
        "   - Under the ZEST Figure 1 conditions (kappa_m * lD = 30.0),\n"
        "     the StarkZee GBK curve (mapped) overlays the ZEST GBK curve\n"
        "     exactly (difference is ~10^-16), verifying mathematical parity.\n\n"
        "2. Cutoff Frequencies (Top-Right):\n"
        "   - ZEST G-functions cut off only at the plasma frequency omega_p.\n"
        "   - StarkZee's GBK model uses omega_c = max(omega_p, omega_L, omega_e).\n"
        "   - In this plasma, the configuration-change frequency omega_e is\n"
        "     larger than omega_p, yielding a tighter cutoff (smaller G-value).\n\n"
        "3. Strong-Collision Constant (Bottom-Left):\n"
        "   - StarkZee includes a physical strong-collision constant C_n\n"
        "     (C_n = 1.5 for n <= 2) to prevent decay to zero at high detunings.\n"
        "   - ZEST's default get_electron_width does not include C_n,\n"
        "     decaying to zero in the far wings."
    )
    axes[1, 1].text(0.05, 0.1, text_info, fontsize=10.5, family='sans-serif',
                     bbox=dict(facecolor='white', edgecolor='#cccccc', boxstyle='round,pad=1.0', alpha=0.9))

    fig.suptitle(
        f"Electron-Impact Broadening: StarkZee vs. ZEST\n"
        f"$N_e = {ne:.1e}$ m$^{{-3}}$, $T_e = {Te_ev:.2f}$ eV, $Z={Z}$, $n={n}$",
        fontsize=14, fontweight='bold', y=0.98
    )
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_comparison()
