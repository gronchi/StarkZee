#!/usr/bin/env python3
"""
Balmer series (Hα – Hε) at DIII-D-like edge conditions
B = 12 T,  Ne = 1e21 m⁻³,  Te = Ti = 10 eV  —  transverse observation

Ti_ev is supplied so the static solver folds in thermal Doppler broadening
(the compute_static_profile default is a bare, un-Dopplered profile). Without
it, the natural/electron-impact linewidth at these conditions is narrower
than the wavelength-grid spacing, and the resulting undersampled Lorentzians
show up as spiky, sawtooth-textured curves rather than the smooth envelope a
real (Doppler-broadened) spectrum would have.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.line_profile import LineProfile

# ── Plasma conditions ─────────────────────────────────────────────────────────
Z   = 1
B   = 12.0   # T
Ne  = 1e21   # m⁻³
Te  = 1.0    # eV
Ti  = 10.0   # eV (Doppler; see module docstring)

# (n_u, n_l, label, half-window [nm])
# Stark broadening grows rapidly with n, so a fixed half-window truncates the
# wings of the higher lines instead of letting them decay to ~0 (at a fixed
# 1.0 nm window, the profile is still at 6% of its peak at the edge for Hε
# vs 0.4% for Hα) -- each half-window below is sized so its line's profile
# has decayed to <0.2% of peak by the edge.
LINES = [
    (3, 2, "Hα", 1.75),
    (4, 2, "Hβ", 2.00),
    (5, 2, "Hγ", 2.50),
    (6, 2, "Hδ", 3.50),
    (7, 2, "Hε", 5.00),
]

PROFILE_KWARGS = dict(
    num_f=30, num_mu=8,
    quadratic_zeeman=False,
    fine_structure=False,
    frequency_dependent_width=False,
)

# ── Compute ───────────────────────────────────────────────────────────────────
profiles = {}
for n_u, n_l, label, hw_nm in LINES:
    lp = LineProfile(n_u=n_u, n_l=n_l, B=B, Ne_m3=Ne, Te_ev=Te, Ti_ev=Ti, species='H')

    wl_grid = np.linspace(lp.E0_wavelength_nm - hw_nm, lp.E0_wavelength_nm + hw_nm, 1000)
    print(f"Computing {label} (n={n_u}→{n_l}), λ₀={lp.E0_wavelength_nm:.2f} nm …", flush=True)

    lp.compute_profile(wl_grid, grid_type='wavelength_nm', **PROFILE_KWARGS)
    lp.compute_discrete(Fz=0.0, Fx=0.0, quadratic_zeeman=False)
    profiles[label] = lp
    print(f"  → {len(lp.discrete.energy_ev)} discrete transitions", flush=True)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(LINES), 1, figsize=(10, 9), sharex=False)
fig.suptitle(
    rf"Balmer series — Stark-Zeeman  |  B = {B} T,  "
    rf"$N_e$ = {Ne:.0e} m$^{{-3}}$,  $T_e$ = {Te} eV,  $T_i$ = {Ti} eV",
    fontsize=12,
)

COLORS_Q = {0: "black", 1: "C3", -1: "C2"}
LABELS_Q  = {0: "π", 1: "σ+", -1: "σ−"}

for ax, (n_u, n_l, label, _) in zip(axes, LINES):
    lp   = profiles[label]
    disc = lp.discrete

    # Broadened profiles — normalize to peak transverse intensity
    peak = lp.profile_transverse.max() or 1.0
    ax.plot(lp.detuning_nm, lp.profile_transverse / peak,
            color="C0", lw=1.8, label="Transverse (90°)")
    ax.plot(lp.detuning_nm, lp.profile_parallel / peak,
            color="C1", lw=1.2, ls="--", label="Parallel (0°)")

    # Stick spectrum at zero microfield
    max_s = disc.strength.max() or 1.0
    drawn = set()
    for dλ, q, s in zip(disc.detuning_nm, disc.q, disc.strength):
        kw = dict(color=COLORS_Q[int(q)], lw=1.0, alpha=0.6)
        if int(q) not in drawn:
            kw["label"] = LABELS_Q[int(q)]
            drawn.add(int(q))
        ax.axvline(dλ, ymin=0, ymax=s / max_s * 0.35, **kw)

    ax.set_xlim(lp.detuning_nm.min(), lp.detuning_nm.max())
    ax.set_ylim(bottom=0)
    ax.set_title(rf"{label}  ($n={n_u}\to{n_l}$,  $\lambda_0={lp.E0_wavelength_nm:.2f}$ nm)",
                 fontsize=10)
    ax.set_ylabel("Norm. intensity", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel(r"$\lambda - \lambda_0$  (nm)", fontsize=11)

plt.tight_layout()
out = "example_balmer_lineprofile.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved {out}")
plt.show()
