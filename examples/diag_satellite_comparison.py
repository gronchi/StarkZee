#!/usr/bin/env python3
"""
diag_satellite_comparison.py
=============================
Comparison of ±2μ_B×B Stark-Zeeman satellite features:
  Hβ (n=4→2) — satellite IS visible as distinct peak  (~2% amplitude)
  Hα (n=3→2) — satellite NOT visible as distinct peak (~0.07% = tail level)

Physical mechanism:
  Upper eigenstate near E0_n + 2μ_B×B (dominated by |nd, ml=2⟩) has a small
  Stark-induced admixture β × |np, ml=1⟩ that makes a σ+ transition to the
  lower eigenstate near E0_n2 (zero Zeeman, dominated by |2s, ml=0⟩ + |2p, ml=0⟩).

  β ≈ Fx × r_within_n × A0 / (μ_B×B)

  For Hα (n=3): r_within ≈ 10.1 a0 → β(F=F0) ≈ 0.027 → sat ≈ 0.013%  << tail
  For Hβ (n=4): r_within ≈ 20.8 a0 → β(F=F0) ≈ 0.056; PLUS n=4 has |4f,ml=2⟩
               degenerate with |4d,ml=2⟩ at +2μ_B×B → additional mixing → sat ≈ 2%
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.static_profile import calculate_static_profile
from starkzee.utils import reduced_mass_rydberg_ev, BOHR_MAGNETON_EV_T

B = 1000.0
muB_B = BOHR_MAGNETON_EV_T * B
Ne, Te = 1e17, 5.0
Z = 1

print(f"B = {int(B)} T,  μ_B×B = {muB_B*1e3:.1f} meV,  2×μ_B×B = {2*muB_B*1e3:.1f} meV")
print()

# ─── Hα (n=3→2) ──────────────────────────────────────────────────────────────
n_u, n_l = 3, 2
E0_Ha = (Z**2) * reduced_mass_rydberg_ev(Z, 1) * (1/n_l**2 - 1/n_u**2)
# Wide grid: σ+ region and satellite region (+40 to +170 meV from E0)
det_Ha = np.linspace(0.040, 0.170, 2600)
en_Ha  = E0_Ha + det_Ha

print("Hα: computing σ+ profile (no-QZ and with-QZ) …")
_, sp_Ha, _ = calculate_static_profile(
    n_u, n_l, Z, B, Ne, Te, en_Ha,
    num_f=50, num_mu=12, use_screening=True, quadratic_zeeman=False,
    frequency_dependent_width=False, Ti_ev=Te)
_, sp_Ha_yq, _ = calculate_static_profile(
    n_u, n_l, Z, B, Ne, Te, en_Ha,
    num_f=50, num_mu=12, use_screening=True, quadratic_zeeman=True,
    frequency_dependent_width=False, Ti_ev=Te)
sp_Ha_max = sp_Ha.max()
# QZ shifts the satellite away from the naive linear 2*muB_B position (it also
# shifts the main cluster itself), so search a window around it rather than
# reading a single exact point — same convention as diag_halpha_satellites.py.
sat_window_Ha = np.abs(det_Ha - 2*muB_B) < 0.015
print(f"  σ+ peak = {sp_Ha_max:.3e} at +{det_Ha[np.argmax(sp_Ha)]*1e3:.1f} meV")
print(f"  No-QZ max in ±15 meV around +{2*muB_B*1e3:.1f} meV: "
      f"{sp_Ha[sat_window_Ha].max()/sp_Ha_max*100:.4f}%")
print(f"  With-QZ max in ±15 meV around +{2*muB_B*1e3:.1f} meV: "
      f"{sp_Ha_yq[sat_window_Ha].max()/sp_Ha_max*100:.4f}%")
print()

# ─── Hβ (n=4→2) ──────────────────────────────────────────────────────────────
n_u4, n_l4 = 4, 2
E0_Hb = (Z**2) * reduced_mass_rydberg_ev(Z, 1) * (1/n_l4**2 - 1/n_u4**2)
det_Hb = np.linspace(0.040, 0.170, 2600)
en_Hb  = E0_Hb + det_Hb

print("Hβ: computing σ+ profile (no-QZ and with-QZ) …")
_, sp_Hb, _ = calculate_static_profile(
    n_u4, n_l4, Z, B, Ne, Te, en_Hb,
    num_f=50, num_mu=12, use_screening=True, quadratic_zeeman=False,
    frequency_dependent_width=False, Ti_ev=Te)
_, sp_Hb_yq, _ = calculate_static_profile(
    n_u4, n_l4, Z, B, Ne, Te, en_Hb,
    num_f=50, num_mu=12, use_screening=True, quadratic_zeeman=True,
    frequency_dependent_width=False, Ti_ev=Te)
sp_Hb_max = sp_Hb.max()
sat_window_Hb = np.abs(det_Hb - 2*muB_B) < 0.015
print(f"  σ+ peak = {sp_Hb_max:.3e} at +{det_Hb[np.argmax(sp_Hb)]*1e3:.1f} meV")
print(f"  No-QZ max in ±15 meV around +{2*muB_B*1e3:.1f} meV: "
      f"{sp_Hb[sat_window_Hb].max()/sp_Hb_max*100:.4f}%")
print(f"  With-QZ max in ±15 meV around +{2*muB_B*1e3:.1f} meV: "
      f"{sp_Hb_yq[sat_window_Hb].max()/sp_Hb_max*100:.4f}%")

from scipy.signal import find_peaks
peaks, _ = find_peaks(sp_Hb_yq, height=sp_Hb_max*0.005)
print(f"  With-QZ σ+ peaks > 0.5%: {[(det_Hb[p]*1e3, sp_Hb_yq[p]/sp_Hb_max*100) for p in peaks]}")
print()

# ─── Figure ──────────────────────────────────────────────────────────────────
det_meV_Ha = det_Ha * 1e3
det_meV_Hb = det_Hb * 1e3
sat_meV = 2*muB_B*1e3

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle(
    f"B = {int(B)} T,  $N_e = 10^{{17}}$ m$^{{-3}}$,  $T_e = 5$ eV  — "
    r"σ$^+$ polarization: solid = with QZ, dashed = no QZ",
    fontsize=12)

for ax, det_meV, sp_nq, sp_yq, sp_max, sat_window, label, color in [
        (axes[0], det_meV_Ha, sp_Ha, sp_Ha_yq, sp_Ha_max, sat_window_Ha, r"H$\alpha$  (n=3→2)", "tab:blue"),
        (axes[1], det_meV_Hb, sp_Hb, sp_Hb_yq, sp_Hb_max, sat_window_Hb, r"H$\beta$  (n=4→2)",  "tab:red"),
]:
    sp_nq_norm = sp_nq / sp_max * 100
    sp_yq_norm = sp_yq / sp_max * 100

    ax.semilogy(det_meV, sp_nq_norm, 'k--', lw=1.2, alpha=0.7, label='No QZ')
    ax.semilogy(det_meV, sp_yq_norm, color=color, lw=1.8, label=f'{label}, with QZ')

    # Mark satellite position
    ax.axvline(sat_meV, color='orange', ls='--', lw=1.5, label=f'+2μ_B×B = +{sat_meV:.0f} meV')
    ax.axvline(muB_B*1e3, color='gray', ls=':', lw=1.0, label=f'+μ_B×B = +{muB_B*1e3:.0f} meV')

    # Annotation: max with-QZ value in a +-15 meV window around the naive satellite
    # position (computed, not hardcoded) -- QZ shifts the actual peak off that
    # position, so a single exact-point lookup would misrepresent the feature.
    sat_idx = np.argmax(sp_yq_norm[sat_window])
    sat_val = sp_yq_norm[sat_window][sat_idx]
    sat_pos = det_meV[sat_window][sat_idx]
    ax.annotate(f'{sat_val:.3f}%\nat +{sat_pos:.0f} meV',
                xy=(sat_pos, sat_val), xytext=(sat_pos+5, sat_val*2),
                fontsize=9, color='orange',
                arrowprops=dict(arrowstyle='->', color='orange', lw=1.0))

    distinct = sat_val > 0.5   # heuristic: a resolved peak vs. Lorentzian tail level
    box_text = f"With-QZ satellite: {sat_val:.2f}%\n" + ("distinct peak" if distinct else "tail level, no distinct peak")
    ax.text(0.40, 0.70, box_text, transform=ax.transAxes,
            fontsize=10, color='darkred' if distinct else 'navy', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='wheat' if distinct else 'lightblue', alpha=0.7))

    ax.set_xlabel(f'Detuning from $E_0$ (meV)', fontsize=11)
    ax.set_ylabel('σ⁺ intensity (% of main peak)', fontsize=11)
    ax.set_title(label, fontsize=12, color=color)
    ax.legend(fontsize=9, loc='lower left')
    ax.set_xlim(40, 170)
    ax.set_ylim(0.005, 200)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xticks([40, 58, 80, 100, 116, 140, 160])
    ax.xaxis.set_tick_params(labelsize=9)

plt.tight_layout()
out = "satellite_comparison.png"
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f"Saved {out}")

# ─── Second figure: wavelength-space showing where features would appear ──────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle(f"B = {int(B)} T — total (π + ½σ±) transverse profile in wavelength space", fontsize=11)

def ev_to_nm(e): return 1239.84197 / e
def nm_to_ev(w): return 1239.84197 / w

for ax, n_u_val, E0_n, label, color in [
        (axes2[0], 3, E0_Ha, r"H$\alpha$ (n=3→2)", "tab:blue"),
        (axes2[1], 4, E0_Hb, r"H$\beta$ (n=4→2)",  "tab:red"),
]:
    lam_main  = ev_to_nm(E0_n)
    lam_sigp  = ev_to_nm(E0_n + muB_B)
    lam_sigm  = ev_to_nm(E0_n - muB_B)
    lam_satp  = ev_to_nm(E0_n + 2*muB_B)
    lam_satm  = ev_to_nm(E0_n - 2*muB_B)

    # Plot the profile using previous sigma+ data
    det_plot = np.linspace(-0.17, 0.17, 5000)
    en_plot  = E0_n + det_plot
    print(f"  Computing full profile for {label} …", flush=True)
    pi_p, sp_p, sm_p = calculate_static_profile(
        n_u_val, 2, Z, B, Ne, Te, en_plot,
        num_f=50, num_mu=12, use_screening=True, quadratic_zeeman=False,
        frequency_dependent_width=False, Ti_ev=Te)
    total = pi_p + 0.5*(sp_p + sm_p)
    peak  = total.max()
    wl_plot = ev_to_nm(en_plot)   # wavelength in nm (note: reversed x-axis)
    # Sort by wavelength ascending
    sort_idx = np.argsort(wl_plot)
    wl_s = wl_plot[sort_idx]
    tot_s = total[sort_idx]

    ax.semilogy(wl_s*10, tot_s/peak, color=color, lw=1.6)
    for lam, lbl, ls in [
            (lam_satp, f'σ⁺ sat\n{lam_satp*10:.0f} Å', '--'),
            (lam_sigp, f'σ⁺\n{lam_sigp*10:.0f} Å', ':'),
            (lam_main, f'π\n{lam_main*10:.0f} Å', '-'),
            (lam_sigm, f'σ⁻\n{lam_sigm*10:.0f} Å', ':'),
            (lam_satm, f'σ⁻ sat\n{lam_satm*10:.0f} Å', '--'),
    ]:
        ax.axvline(lam*10, color='gray' if 'sat' not in lbl else 'orange', ls=ls, lw=0.9)
        ax.text(lam*10, 2.0, lbl, fontsize=7, ha='center', color='orange' if 'sat' in lbl else 'gray')

    ax.set_xlim(wl_s[0]*10, wl_s[-1]*10)
    ax.set_ylim(1e-5, 3.0)
    ax.set_xlabel(r'Wavelength (Å)', fontsize=10)
    ax.set_ylabel('Norm. intensity', fontsize=10)
    ax.set_title(label, fontsize=11, color=color)
    ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
out2 = "satellite_wavelength.png"
plt.savefig(out2, dpi=200, bbox_inches='tight')
print(f"Saved {out2}")
plt.show()
print("\nDone.")
