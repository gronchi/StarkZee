#!/usr/bin/env python3
"""
diag_halpha_satellites.py
=========================
Look for the ±2μ_B×B Stark-Zeeman satellite features in Hα (n=3→2) at B=1000 T.

Physical mechanism
------------------
Transverse Stark field Fx mixes states with Δl=±1, Δml=±1 within the same n shell.
The eigenstate near E0_n3 + 2μ_B×B (dominated by |3d,ml=2⟩) acquires an admixture
β × |3p,ml=1⟩ due to Fx coupling. This β component can make a σ+ transition to the
lower eigenstate near E0_n2 (dominated by |2s,ml=0⟩ or |2p,ml=0⟩) — producing a
photon at energy E0 + 2μ_B×B instead of the regular E0 + μ_B×B.

β ≈ Fx × ⟨3p|r|3d⟩ × A0 / (μ_B × B) → scales as Fx/B → satellite intensity ∝ ⟨F²⟩/B²

For n=3: β_rms ≈ F0 × 10.06a0 × A0 / μ_B×B = 8e6 × 5.32e-10 / 57.9e-3 ≈ 0.074
Satellite intensity ~ β² × OS_fraction ~ 0.5%  (weak — that's why it's hard to see)

For n=4: β_rms ≈ F0 × 15.9a0 × A0 / μ_B×B ≈ 0.116 → intensity ~ 1.3%  (visible)

This script runs with high resolution: num_f=60, num_mu=12 and a ±200 meV window,
comparing B=100T, 500T, 1000T.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.static_profile import calculate_static_profile
from scipy.constants import e as E_CHARGE, m_e as M_E
from starkzee.utils import RYDBERG_EV, BOHR_MAGNETON_EV_T, A0

Z, n_u, n_l = 1, 3, 2
Ne, Te = 1e17, 5.0
E0 = RYDBERG_EV * (1.0/n_l**2 - 1.0/n_u**2)   # ≈ 1.889 eV

muB_1000 = BOHR_MAGNETON_EV_T * 1000.0          # ≈ 57.9 meV

print(f"Hα: E0 = {E0:.4f} eV")
print(f"μ_B × 1000 T = {muB_1000*1e3:.2f} meV")
print(f"Satellite expected at ±{2*muB_1000*1e3:.1f} meV from E0")
print()

# ── Compute profiles at B=1000 T with wide window ─────────────────────────────
B = 1000.0
muB_B = BOHR_MAGNETON_EV_T * B
DET_HALF = 0.20         # ±200 meV window to see both σ± main peaks AND satellites
NPTS     = 8000         # ~0.05 meV resolution

det = np.linspace(-DET_HALF, DET_HALF, NPTS)
energies = E0 + det

# Run with different resolution settings to check convergence
configs = [
    ("Low  (num_f=20, num_mu=6)",  20, 6),
    ("Mid  (num_f=40, num_mu=10)", 40, 10),
    ("High (num_f=60, num_mu=12)", 60, 12),
]

results = {}
for label, num_f, num_mu in configs:
    print(f"B={int(B)}T, {label} …", flush=True)
    pi_nq, sp_nq, sm_nq = calculate_static_profile(
        n_u, n_l, Z, B, Ne, Te, energies,
        num_f=num_f, num_mu=num_mu,
        use_screening=True, quadratic_zeeman=False,
        frequency_dependent_width=False)
    print("  without QZ done.", flush=True)
    pi_yq, sp_yq, sm_yq = calculate_static_profile(
        n_u, n_l, Z, B, Ne, Te, energies,
        num_f=num_f, num_mu=num_mu,
        use_screening=True, quadratic_zeeman=True,
        frequency_dependent_width=False)
    print("  with QZ done.", flush=True)
    results[label] = dict(pi_nq=pi_nq, sp_nq=sp_nq, sm_nq=sm_nq,
                          pi_yq=pi_yq, sp_yq=sp_yq, sm_yq=sm_yq)

    # Quick peak analysis for no-QZ total spectrum
    total_nq = pi_nq + 0.5*(sp_nq + sm_nq)
    peak = total_nq.max()
    satellite_region = np.abs(np.abs(det*1e3) - 2*muB_B*1e3) < 15.0  # ±15 meV around ±2μ_BB
    main_region      = np.abs(det) < 0.030
    sat_peak = total_nq[satellite_region].max() if satellite_region.any() else 0
    main_peak = total_nq[main_region].max()
    print(f"  Satellite / main peak ratio (no-QZ, ±15meV around ±2μ_B×B): "
          f"{sat_peak/main_peak*100:.3f}%")

# ── Plot: full spectrum showing satellites ─────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle(
    r"Hα (n=3→2) full Stark-Zeeman profile, B=1000 T — searching for ±2μ$_B$B satellites"
    "\n"
    r"$N_e=10^{17}$ cm$^{-3}$, $T_e=5$ eV, transverse obs. (I$_\pi$ + ½(I$_{\sigma+}$ + I$_{\sigma-}$))",
    fontsize=11)

det_meV = det * 1e3
satellite_eV = 2 * muB_B  # 115.8 meV at B=1000T
main_eV = muB_B            # 57.9 meV

for ax, (label, num_f, num_mu), color in zip(
        axes, configs, ['tab:blue', 'tab:orange', 'tab:red']):
    r = results[label]
    total_nq = r['pi_nq'] + 0.5*(r['sp_nq'] + r['sm_nq'])
    total_yq = r['pi_yq'] + 0.5*(r['sp_yq'] + r['sm_yq'])

    peak = max(total_yq.max(), total_nq.max(), 1e-30)

    ax.semilogy(det_meV, total_nq/peak, 'k--', lw=1.2, alpha=0.75, label='No QZ')
    ax.semilogy(det_meV, total_yq/peak, color=color, lw=1.8, label='With QZ')

    # Mark expected satellite positions
    for sign, tag in [(+1, 'σ+'), (-1, 'σ−')]:
        ax.axvline(sign*satellite_eV*1e3, color='orange', ls=':', lw=1.2,
                   label=f'±2μ_B×B = ±{satellite_eV*1e3:.0f} meV ({tag} sat)' if sign==1 else '')
        ax.axvline(sign*main_eV*1e3, color='gray', ls=':', lw=1.0,
                   label=f'±μ_B×B = ±{main_eV*1e3:.0f} meV (σ±)' if sign==1 else '')

    ax.set_ylim(1e-5, 3.0)
    ax.set_ylabel('Norm. intensity', fontsize=9)
    ax.set_title(f'{label}', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)

axes[-1].set_xlabel('Detuning from $E_0$ (meV)', fontsize=11)
axes[-1].set_xlim(-210, 210)

plt.tight_layout()
out = 'halpha_satellites_convergence.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f"\nSaved {out}")

# ── Zoomed-in plot: just the satellite regions ─────────────────────────────────
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9))
fig2.suptitle(r"Zoomed satellite regions: Hα at B=1000 T", fontsize=12)

windows = [
    ('+2μ_B×B region (σ+ sat)',  +satellite_eV*1e3, 30),
    ('+μ_B×B region (σ+ main)', +main_eV*1e3, 30),
    ('zero  (π main)', 0.0, 30),
]
for col, (title, center_meV, hw_meV) in enumerate(windows):
    mask = np.abs(det_meV - center_meV) < hw_meV
    for row, (label, num_f, num_mu), color in zip(range(3), configs, ['tab:blue','tab:orange','tab:red']):
        if col == 0:
            ax = axes2[0 if row < 2 else 1, col]
        # actually let me just use 1 row per resolution for comparison
        pass

    for ax, (label, num_f, num_mu), color in zip([axes2[0,col], axes2[1,col]],
                                                   configs[:2], ['tab:blue','tab:orange']):
        r = results[label]
        total_nq = r['pi_nq'] + 0.5*(r['sp_nq'] + r['sm_nq'])
        total_yq = r['pi_yq'] + 0.5*(r['sp_yq'] + r['sm_yq'])
        peak = total_nq.max()

        ax.plot(det_meV[mask], total_nq[mask]/peak*100, 'k--', lw=1.2, alpha=0.8, label='No QZ')
        ax.plot(det_meV[mask], total_yq[mask]/peak*100, color=color, lw=1.8, label='With QZ')
        ax.set_title(f'{title}\n{label}', fontsize=8)
        ax.set_xlabel('Detuning (meV)', fontsize=8)
        ax.set_ylabel('% of main peak', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
out2 = 'halpha_satellites_zoom.png'
plt.savefig(out2, dpi=200, bbox_inches='tight')
print(f"Saved {out2}")

plt.show()
print("\nDone.")
