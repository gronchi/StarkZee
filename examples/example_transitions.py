"""
example_transitions.py — Discrete Stark-Zeeman transitions and full profiles.

Demonstrates:
  1. Listing individual transitions and their line strengths
  2. Stick spectra at B=0 and B=5T side by side (pure Zeeman)
  3. Full static profile with stick spectrum overlay
  4. Oscillator strengths and Einstein A coefficients for common lines
  5. Stark + Zeeman: how a microfield splits an otherwise degenerate manifold

Run:
    python example_transitions.py

Produces:  example_transitions.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.static_profile import discrete_transitions, calculate_static_profile
from starkzee.radiator import line_strength, oscillator_strength, einstein_a
from starkzee.utils import reduced_mass_rydberg_ev, energy_ev_to_wavelength_nm

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

POL_COLOR = {0: "#e74c3c", -1: "#3498db", 1: "#2ecc71"}  # π, σ+, σ−
POL_LABEL = {0: "π (q=0)", -1: "σ+ (q=−1)", 1: "σ− (q=+1)"}


def bohr_energy(n_u, n_l, Z, A=1):
    return (Z**2) * reduced_mass_rydberg_ev(Z, A) * (1.0/n_l**2 - 1.0/n_u**2)


def plot_stick_spectrum(ax, tr, E0, normalize=True, title="", show_legend=True):
    """Plot discrete transitions as vertical lines colored by polarization."""
    det = (tr['energy_ev'] - E0) * 1e3  # detuning in meV
    s = tr['strength']
    if normalize and s.max() > 0:
        s = s / s.max()
    for q in [0, -1, 1]:
        mask = tr['q'] == q
        if mask.any():
            ax.vlines(det[mask], 0, s[mask],
                      colors=POL_COLOR[q], label=POL_LABEL[q],
                      linewidth=1.8, alpha=0.85)
    ax.set_xlabel("Detuning from E₀ (meV)", fontsize=10)
    ax.set_ylabel("Norm. strength" if normalize else "|d|² (a₀²)", fontsize=10)
    ax.set_title(title, fontsize=11)
    if show_legend:
        ax.legend(fontsize=8, framealpha=0.7)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xlim(-5, 5)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Print transition table for H Ly-α at B=5T
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("H Ly-α (n=2→1, Z=1) discrete transitions at B=5 T")
print("=" * 65)
Z, n_u, n_l, B = 1, 2, 1, 5.0
E0 = bohr_energy(n_u, n_l, Z)

tr = discrete_transitions(n_u=n_u, n_l=n_l, Z=Z, B=B,
                           fine_structure=True, min_strength=1e-6)

print(f"{'#':>3}  {'E (eV)':>12}  {'ΔE (meV)':>10}  {'q':>4}  "
      f"{'|d|² (a₀²)':>12}  {'upper':>6}  {'lower':>6}")
print("-" * 65)
for k in range(len(tr['energy_ev'])):
    det_mev = (tr['energy_ev'][k] - E0) * 1e3
    pol = {0: "π", -1: "σ+", 1: "σ−"}[tr['q'][k]]
    print(f"{k+1:>3}  {tr['energy_ev'][k]:>12.6f}  {det_mev:>10.4f}  "
          f"{pol:>4}  {tr['strength'][k]:>12.6f}  "
          f"{tr['upper_idx'][k]:>6}  {tr['lower_idx'][k]:>6}")

S_ul = line_strength(n_u, n_l, Z)
print(f"\nSum of strengths : {tr['strength'].sum():.6f} a₀²")
print(f"line_strength()  : {S_ul:.6f} a₀²  (must match)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Atomic data table
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("Oscillator strengths and Einstein A coefficients")
print("=" * 65)
LINES = [
    ("H  Ly-α", 2, 1, 1),
    ("H  Ly-β", 3, 1, 1),
    ("H  Hα  ", 3, 2, 1),
    ("H  Hβ  ", 4, 2, 1),
    ("C VI Ly-α", 2, 1, 6),
    ("C VI Hα  ", 3, 2, 6),
]
print(f"{'Line':>12}  {'gf':>8}  {'S_ul (a₀²)':>12}  {'A_ul (s⁻¹)':>14}")
print("-" * 52)
for name, nu, nl, Zl in LINES:
    gf  = oscillator_strength(nu, nl, Zl)
    S   = line_strength(nu, nl, Zl)
    A   = einstein_a(nu, nl, Zl)
    print(f"{name:>12}  {gf:>8.4f}  {S:>12.4f}  {A:>14.4e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build figure
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(10, 9))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

ax_b0   = fig.add_subplot(gs[0, 0])
ax_b5   = fig.add_subplot(gs[0, 1])
ax_full = fig.add_subplot(gs[0, 2])
ax_stark = fig.add_subplot(gs[1, 0])
ax_hb   = fig.add_subplot(gs[1, 1])
ax_cvi  = fig.add_subplot(gs[1, 2])

# ── Panel 1: H Ly-α stick at B=0 ────────────────────────────────────────────
tr_b0 = discrete_transitions(n_u=2, n_l=1, Z=1, B=0.0,
                               fine_structure=True, min_strength=1e-6)
plot_stick_spectrum(ax_b0, tr_b0, E0,
                    title="H Ly-α  B=0 T\n(SO splits 2p₁/₂ from 2p₃/₂)")

# ── Panel 2: H Ly-α stick at B=5T ───────────────────────────────────────────
tr_b5 = discrete_transitions(n_u=2, n_l=1, Z=1, B=5.0,
                               fine_structure=True, min_strength=1e-6)
plot_stick_spectrum(ax_b5, tr_b5, E0,
                    title="H Ly-α  B=5 T\n(Zeeman + SO)")

# ── Panel 3: Full static profile overlaid with sticks ───────────────────────
Ne, Te = 1e23, 5.0
energies = E0 + np.linspace(-0.005, 0.005, 1000)
pi, sp, sm = calculate_static_profile(
    n_u=2, n_l=1, Z=1, B=5.0, Ne_m3=Ne, Te_ev=Te,
    energies_ev=energies, num_f=25, num_mu=8,
    fine_structure=True, frequency_dependent_width=False
)
det_ev  = (energies - E0) * 1e3
total   = pi + sp + sm
total_n = total / total.max()

ax_full.plot(det_ev, total_n, color="k", linewidth=1.5, label="Total profile")
ax_full.plot(det_ev, pi / total.max(), color=POL_COLOR[0],
             linewidth=1.0, linestyle="--", alpha=0.7, label="π")
ax_full.plot(det_ev, (sp + sm) / total.max(), color=POL_COLOR[-1],
             linewidth=1.0, linestyle=":", alpha=0.7, label="σ+σ−")

# Overlay sticks (normalized to profile peak)
s5 = tr_b5['strength']
s5_n = s5 / s5.max() * 0.5
det5 = (tr_b5['energy_ev'] - E0) * 1e3
for q in [0, -1, 1]:
    mask = tr_b5['q'] == q
    ax_full.vlines(det5[mask], 0, s5_n[mask],
                   colors=POL_COLOR[q], alpha=0.4, linewidth=1.2)

ax_full.set_xlabel("Detuning (meV)", fontsize=10)
ax_full.set_ylabel("Norm. intensity", fontsize=10)
ax_full.set_title(f"H Ly-α  B=5 T  Ne={Ne:.0e} m⁻³\nProfile + stick spectrum",
                  fontsize=11)
ax_full.set_xlim(-5, 5)
ax_full.legend(fontsize=8, framealpha=0.7)

# ── Panel 4: Stark + Zeeman — how a microfield splits lines ─────────────────
B_panel = 2.0
E0_lya = bohr_energy(2, 1, 1)
for Fz_vm, alpha in [(0.0, 1.0), (5e7, 0.65), (2e8, 0.35)]:
    tr_f = discrete_transitions(n_u=2, n_l=1, Z=1, B=B_panel,
                                 Fz=Fz_vm, fine_structure=False,
                                 min_strength=1e-6)
    det = (tr_f['energy_ev'] - E0_lya) * 1e3
    s   = tr_f['strength'] / tr_f['strength'].max()
    label = f"F={Fz_vm:.0e} V/m" if Fz_vm > 0 else "F=0"
    ax_stark.vlines(det, 0, s * alpha, alpha=alpha,
                    colors=["#7f8c8d", "#e67e22", "#8e44ad"][
                        [0.0, 5e7, 2e8].index(Fz_vm)],
                    linewidth=1.5, label=label)

ax_stark.set_xlabel("Detuning (meV)", fontsize=10)
ax_stark.set_ylabel("Norm. strength", fontsize=10)
ax_stark.set_title(f"H Ly-α  B={B_panel} T + Stark field\n"
                   f"(sticks: all polarizations combined)", fontsize=11)
ax_stark.legend(fontsize=8, framealpha=0.7)
ax_stark.set_xlim(-8, 8)
ax_stark.axhline(0, color="grey", linewidth=0.5)

# ── Panel 5: H Hβ stick at B=5T ─────────────────────────────────────────────
E0_hb = bohr_energy(4, 2, 1)
tr_hb = discrete_transitions(n_u=4, n_l=2, Z=1, B=5.0,
                               fine_structure=True, min_strength=1e-5)
det_hb = (tr_hb['energy_ev'] - E0_hb) * 1e3
s_hb   = tr_hb['strength'] / tr_hb['strength'].max()
for q in [0, -1, 1]:
    mask = tr_hb['q'] == q
    if mask.any():
        ax_hb.vlines(det_hb[mask], 0, s_hb[mask],
                     colors=POL_COLOR[q], linewidth=1.2, alpha=0.85,
                     label=POL_LABEL[q])
ax_hb.set_xlabel("Detuning (meV)", fontsize=10)
ax_hb.set_ylabel("Norm. strength", fontsize=10)
ax_hb.set_title("H Hβ (n=4→2)  B=5 T\nMany Stark-Zeeman components", fontsize=11)
ax_hb.legend(fontsize=8, framealpha=0.7)
ax_hb.set_xlim(-12, 12)
ax_hb.axhline(0, color="grey", linewidth=0.5)

# ── Panel 6: C VI Ly-α stick at B=5T — large SO dominates ───────────────────
E0_cvi = bohr_energy(2, 1, 6)
tr_cvi = discrete_transitions(n_u=2, n_l=1, Z=6, B=5.0,
                               fine_structure=True, min_strength=1e-6)
det_cvi = (tr_cvi['energy_ev'] - E0_cvi) * 1e3
s_cvi   = tr_cvi['strength'] / tr_cvi['strength'].max()
for q in [0, -1, 1]:
    mask = tr_cvi['q'] == q
    if mask.any():
        ax_cvi.vlines(det_cvi[mask], 0, s_cvi[mask],
                      colors=POL_COLOR[q], linewidth=1.8, alpha=0.85,
                      label=POL_LABEL[q])
ax_cvi.set_xlabel("Detuning from E₀ (meV)", fontsize=10)
ax_cvi.set_ylabel("Norm. strength", fontsize=10)
ax_cvi.set_title("C VI Ly-α (Z=6)  B=5 T\nSO dominates (ξ ≫ μ_B B)", fontsize=11)
ax_cvi.legend(fontsize=8, framealpha=0.7)
ax_cvi.set_xlim(-32, 32)
ax_cvi.axhline(0, color="grey", linewidth=0.5)

fig.suptitle("starkzee — Discrete Stark-Zeeman Transitions", fontsize=14, y=1.01)
plt.show()
