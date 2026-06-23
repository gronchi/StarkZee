#!/usr/bin/env python3
"""
figure2_gbk_magnetic.py
=======================
Compare G(Δω) for hydrogen Lyman-α at Ne = 1e23 m⁻³, Te = 5 eV,
for four B-field values (0, 100, 500, 1000 T).

Reproduces Fig. 2 of Ferri, Peyrusse & Calisti, Matter Radiat. Extremes 7,
015901 (2022).  Each figure has two panels:
  Left : StarkZee G(Δω) for four B-field values.
  Right: ZEST-equivalent G(Δω) for GBK / Lee / Dufty models (B = 0).

All ZEST-equivalent models are implemented in starkzee.broadening; no
dependency on the ZEST library is required.

Formula check (Ferri Eq. 19–20)
---------------------------------
The Maxwell-averaged collision operator is

    Φ(Δω) = -(4π/3) N_e √(2m_e/π k_B T_e) (ħ/m_e)² R⃗·R⃗ [C_n + G(Δω)]

C_n is the strong-collision constant outside G; G itself is

    G(Δω) = ½ E₁(y),   y = (ħn²/2Z)² (Δω² + ω_c²) / (E_H k_B T_e)

In eV units (omega_c_ev = ħω_c/e, etc.) this simplifies to

    y = (n²/2Z)² (delta_omega_ev² + omega_c_ev²) / (Ryd_ev × Te_ev)

where Ryd = e²/2a₀ = 13.6057 eV.  GBK, Ferri and ZEST all use this same definition
(GBK calls it "E_H"; ZEST and modern convention call it "Ryd").  No factors are missing.

Note on ω_e and the 2π discrepancy
------------------------------------
Ferri's paper text (near Eq. 20) writes:

    ω_e = 2π / τ_e,   τ_e = r_e / v_th,   v_th = √(k_B T_e / m_e)

At Ne = 10¹⁷ cm⁻³, Te = 5 eV this gives ω_e ≈ 290 meV, which exceeds ω_L for
all B up to ~2500 T.  All four B-field curves would then collapse onto one line —
inconsistent with their Fig. 2 showing distinct curves at B = 500 T and 1 kT.

Their Fig. 2 caption states "G(Δω) calculated for B = 100 T is superposed on the
non-magnetized results because ω_e is larger than ω_L for these plasma conditions."
ω_L(100 T) ≈ 11.6 meV, so ω_e > 11.6 meV (both the 2π and no-2π definitions
satisfy this).  But at B = 500 T, ω_L ≈ 58 meV, and the figure shows separation,
so ω_e < 58 meV is required.  Only ω_e = 1/τ_e ≈ 46 meV (no 2π) satisfies this.

Could the 2π arise from a different v_th definition?
The valid window is 11.6 meV < ω_e < 58 meV, i.e., v_th in (0.73, 3.7) × 10⁶ m/s
(without 2π).  Standard definitions at Te = 5 eV:
  √(kT/m)   = 9.38×10⁵ m/s → ω_e = 46 meV  ✓  (Ferri's stated definition)
  √(2kT/m)  = 1.33×10⁶ m/s → ω_e = 65 meV  ✗  (> 58 meV; B=500T still overlaps B=0)
  √(8kT/πm) = 1.22×10⁶ m/s → ω_e = 60 meV  ✗
  √(3kT/m)  = 1.62×10⁶ m/s → ω_e = 79 meV  ✗
No standard definition places ω_e in the valid window with the 2π included (that
would require v_th ≈ 2π × smaller, i.e., a factor of ~6.3 — unphysical).

Note on units: ω_p and ω_L are angular frequencies by derivation (from equations
of motion); 1/τ_e is a rate in s⁻¹ (Hz).  To put ω_e on the same footing as
ω_p and ω_L one must multiply by 2π, giving ω_e = 2π/τ_e — which is exactly
what Ferri write.  So their text is dimensionally correct.

The inconsistency is in their figure: ω_e = 2π/τ_e ≈ 290 meV contradicts the
distinct B-field curves shown.  Their code (PPPB) most likely used ω_e = 1/τ_e
(the rate in s⁻¹ treated directly as an angular frequency, skipping the 2π
conversion) — a common shortcut in plasma physics where collision frequency ν
and angular collision frequency ω are conflated.  StarkZee follows the same
convention: ω_e = 1/τ_e = v_th / r_e ≈ 46 meV, consistent with the figure.
"""

import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.constants import hbar as HBAR, e as E_CHARGE, epsilon_0 as EPSILON_0, m_e as M_ELECTRON

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import starkzee.broadening as sz_broad


def _setup_style():
    mpl.rcParams.update({
        'font.family':           'serif',
        'font.size':             9,
        'axes.labelsize':        10,
        'axes.titlesize':        9,
        'axes.linewidth':        0.8,
        'legend.fontsize':       7,
        'legend.framealpha':     0.9,
        'legend.edgecolor':      '0.75',
        'legend.handlelength':   2.5,
        'xtick.labelsize':       9,
        'ytick.labelsize':       9,
        'xtick.direction':       'in',
        'ytick.direction':       'in',
        'xtick.top':             True,
        'ytick.right':           True,
        'xtick.minor.visible':   True,
        'ytick.minor.visible':   True,
        'xtick.major.width':     0.8,
        'ytick.major.width':     0.8,
        'xtick.minor.width':     0.5,
        'ytick.minor.width':     0.5,
        'xtick.major.size':      4,
        'ytick.major.size':      4,
        'xtick.minor.size':      2,
        'ytick.minor.size':      2,
        'lines.linewidth':       1.5,
        'figure.dpi':            150,
    })


def main():
    _setup_style()

    # ── Plasma conditions ──────────────────────────────────────────────────
    Ne_m3 = 1e23         # Ne = 1e17 cm^-3 = 1e23 m^-3
    Te_ev = 5.0
    Z     = 1
    n     = 2             # Lyman-α upper level (n=2 → n=1)

    omega_p    = sz_broad.calculate_plasma_frequency(Ne_m3)
    omega_p_ev = omega_p * HBAR / E_CHARGE

    omega_e    = sz_broad.calculate_configuration_frequency(Ne_m3, Te_ev)
    omega_e_ev = omega_e * HBAR / E_CHARGE

    B_crit = omega_e * M_ELECTRON / E_CHARGE   # B where ω_L = ω_e

    print(f"omega_p  = {omega_p:.4e} rad/s  ({omega_p_ev*1e3:.2f} meV)")
    print(f"omega_e  = {omega_e:.4e} rad/s  ({omega_e_ev*1e3:.2f} meV)")
    print(f"B_crit (omega_L = omega_e) = {B_crit:.1f} T")
    print()

    # ── Detuning axis in units of ω_p ─────────────────────────────────────
    w_red          = np.linspace(0.0, 40.0, 1000)   # Δω / ω_p
    delta_omega_ev = w_red * omega_p_ev

    # ── B values, line styles, labels ─────────────────────────────────────
    B_values   = [0.0, 100.0, 500.0, 1000.0]
    labels     = [
        r'$B = 0$',
        r'$B = 100\,\mathrm{T}$',
        r'$B = 500\,\mathrm{T}$',
        r'$B = 1\,\mathrm{kT}$',
    ]
    linestyles = [
        '-',
        (0, (5, 2)),
        (0, (5, 2, 1, 2)),
        (0, (1, 1)),
    ]
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = prop_cycle[:4]

    def _label_panel(ax, letter):
        ax.text(0.04, 0.96, f'{letter})', transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top', ha='left')

    def _style_axes(ax_left, ax_right, label_left, label_right, xlim=30):
        for ax in (ax_left, ax_right):
            ax.set_xlabel(r'$\Delta\omega\,/\,\omega_p$')
            ax.set_xlim(0, xlim)
            ax.set_ylim(bottom=0)
            ax.legend(loc='upper right', frameon=True)
            ax.grid()
        ax_left.set_ylabel(r'$G(\Delta\omega)$')
        ax_left.set_title(label_left, pad=6)
        ax_right.set_title(label_right, pad=6)
        _label_panel(ax_left,  'a')
        _label_panel(ax_right, 'b')

    # =========================================================================
    # ── FIGURE 1: ZEST Dimensionless Parameters (kappa_m * lambda_D = 30.0) ──
    # =========================================================================
    print("=== Figure 1: ZEST Dimensionless Parameters (kappa_m * lambda_D = 30.0) ===")

    Te_ev_fig1 = 1.0

    kappa_m_fig1    = sz_broad.calculate_cutoff_kappa_m(Z, n, Te_ev_fig1)
    lambda_D_target = 30.0 / kappa_m_fig1
    Ne_m3_fig1      = EPSILON_0 * Te_ev_fig1 / (E_CHARGE * lambda_D_target**2)

    omega_p_fig1    = sz_broad.calculate_plasma_frequency(Ne_m3_fig1)
    omega_p_ev_fig1 = omega_p_fig1 * HBAR / E_CHARGE
    omega_e_fig1    = sz_broad.calculate_configuration_frequency(Ne_m3_fig1, Te_ev_fig1)

    delta_omega_ev_fig1 = w_red * omega_p_ev_fig1

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2), sharex=True, sharey=True, constrained_layout=True)

    print(f"--- StarkZee G(dw) (Te = 1 eV, Ne = {Ne_m3_fig1:.6e} m^-3) ---")
    for B, label, ls, color in zip(B_values, labels, linestyles, colors):
        omega_L     = sz_broad.calculate_larmor_frequency(B)
        omega_c_rad = max(omega_p_fig1, omega_e_fig1, omega_L)
        omega_c_ev  = omega_c_rad * HBAR / E_CHARGE
        G = sz_broad.gbk_model(delta_omega_ev_fig1, omega_c_ev, Te_ev_fig1, Z, n=n)
        ax1.plot(w_red, G, linestyle=ls, lw=2.0, color=color, label=label)
        omega_L_ev = omega_L * HBAR / E_CHARGE
        print(f"  B = {B:6.0f} T:  omega_L = {omega_L_ev*1e3:7.2f} meV,  "
              f"omega_c = {omega_c_ev*1e3:7.2f} meV,  G(0) = {G[0]:.3f}")

    # GBK/ZEST reference: same formula, omega_c = omega_p only (B=0)
    g_zest_ref_fig1 = sz_broad.gbk_zest_model(delta_omega_ev_fig1, Ne_m3_fig1, Te_ev_fig1, Z, n)
    ax1.plot(w_red, g_zest_ref_fig1, linestyle=(0,(3,1,1,1)), lw=1.5,
             label=r'GBK/ZEST ($\omega_c = \omega_p$)')
    print(f"  GBK/ZEST ref B=0:  G(0) = {g_zest_ref_fig1[0]:.3f}  (omega_c = omega_p)")

    print(f"\n--- ZEST-equivalent G(dw) (Te = 1 eV, Ne = {Ne_m3_fig1:.6e} m^-3) ---")
    g_gbk   = sz_broad.gbk_zest_model(delta_omega_ev_fig1, Ne_m3_fig1, Te_ev_fig1, Z, n)
    g_lee   = sz_broad.lee_model(delta_omega_ev_fig1, Ne_m3_fig1, Te_ev_fig1, Z, n)
    g_dufty = sz_broad.dufty_model(delta_omega_ev_fig1, Ne_m3_fig1, Te_ev_fig1, Z, n)

    ax2.plot(w_red, g_dufty, label='Dufty RPA', lw=1.5)
    ax2.plot(w_red, g_lee,   label='Lee',        lw=1.5, ls='--')
    ax2.plot(w_red, g_gbk,   label='GBK (ZEST)', lw=1.5, ls=(0,(5,2)))

    print(f"  GBK G(0):   {g_gbk[0]:.4f}")
    print(f"  Lee G(0):   {g_lee[0]:.4f}")
    print(f"  Dufty G(0): {g_dufty[0]:.4f}")

    _style_axes(
        ax1, ax2,
        label_left=r'StarkZee, $\omega_c = \max(\omega_p,\,\omega_e,\,\omega_L)$',
        label_right=r'ZEST-equivalent ($\kappa_m \lambda_D = 30$)',
        xlim=30,
    )
    fig1.suptitle(
        r"Lyman-$\alpha$, $T_e = 1\,\mathrm{eV}$, "
        r"$N_e = $" + f"${Ne_m3_fig1:.2e}$" + r"$\,\mathrm{m}^{-3}$",
        fontsize=9
    )
    # =========================================================================
    # ── FIGURE 2: Physical Plasma Parameters (Ne = 1e23 m^-3, Te = 5.0 eV) ──
    # =========================================================================
    print(f"\n=== Figure 2: Physical Parameters (Ne = {Ne_m3:.1e} m^-3, Te = 5.0 eV) ===")

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2), sharex=True, sharey=True, constrained_layout=True)

    print(f"--- StarkZee G(dw), omega_e = 1/tau_e = {omega_e_ev*1e3:.2f} meV ---")
    for B, label, ls, color in zip(B_values, labels, linestyles, colors):
        omega_L     = sz_broad.calculate_larmor_frequency(B)
        omega_c_rad = max(omega_p, omega_e, omega_L)
        omega_c_ev  = omega_c_rad * HBAR / E_CHARGE
        G = sz_broad.gbk_model(delta_omega_ev, omega_c_ev, Te_ev, Z, n=n)
        ax1.plot(w_red, G, linestyle=ls, lw=2.0, color=color, label=label)
        omega_L_ev = omega_L * HBAR / E_CHARGE
        print(f"  B = {B:6.0f} T:  omega_L = {omega_L_ev*1e3:7.2f} meV,  "
              f"omega_c = {omega_c_ev*1e3:7.2f} meV,  G(0) = {G[0]:.3f}")

    # GBK/ZEST reference: same formula, omega_c = omega_p only (B=0)
    g_zest_ref = sz_broad.gbk_zest_model(delta_omega_ev, Ne_m3, Te_ev, Z, n)
    ax1.plot(w_red, g_zest_ref, linestyle=':', lw=2.0, color='#8e44ad',
             label=r'GBK/ZEST ($B=0$, $\omega_c = \omega_p$)')
    print(f"  GBK/ZEST ref B=0:  G(0) = {g_zest_ref[0]:.3f}  (omega_c = omega_p)")

    print(f"\n--- ZEST-equivalent G(dw) (Te = 5.0 eV, Ne = {Ne_m3:.1e} m^-3) ---")
    g_gbk   = sz_broad.gbk_zest_model(delta_omega_ev, Ne_m3, Te_ev, Z, n)
    g_lee   = sz_broad.lee_model(delta_omega_ev, Ne_m3, Te_ev, Z, n)
    g_dufty = sz_broad.dufty_model(delta_omega_ev, Ne_m3, Te_ev, Z, n)

    ax2.plot(w_red, g_dufty, label='Dufty RPA', lw=1.5)
    ax2.plot(w_red, g_lee,   label='Lee',        lw=1.5, ls='--')
    ax2.plot(w_red, g_gbk,   label='GBK (ZEST)', lw=1.5, ls=(0,(5,2)))

    print(f"  GBK G(0):   {g_gbk[0]:.4f}")
    print(f"  Lee G(0):   {g_lee[0]:.4f}")
    print(f"  Dufty G(0): {g_dufty[0]:.4f}")

    _style_axes(
        ax1, ax2,
        label_left=r'StarkZee, $\omega_c = \max(\omega_p,\,\omega_e,\,\omega_L)$',
        label_right='ZEST-equivalent models',
        xlim=40,
    )
    fig2.suptitle(
        r"Lyman-$\alpha$, $T_e = 5\,\mathrm{eV}$, $N_e = 10^{23}\,\mathrm{m}^{-3}$"
        r" — Ferri et al. (2022), Fig. 2",
        fontsize=9
    )
    plt.show()


if __name__ == '__main__':
    main()
