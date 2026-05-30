#!/usr/bin/env python3
"""
Balmer series (Hα – Hε) at DIII-D edge conditions
B = 12 T,  Ne = 1e20 m⁻³,  Te = 5 eV  —  transverse observation
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
Te  = 500.0    # eV

# (n_u, n_l, label, half-window [nm])
LINES = [
    (3, 2, "Hα", 1.0),
    (4, 2, "Hβ", 1.0),
    (5, 2, "Hγ", 1.0),
    (6, 2, "Hδ", 1.0),
    (7, 2, "Hε", 1.0),
]

PROFILE_KWARGS = dict(
    num_f=30, num_mu=8,
    include_quadratic=False,
    include_fine_structure=False,
    frequency_dependent_width=False,
)

# ── Compute ───────────────────────────────────────────────────────────────────
profiles = {}
for n_u, n_l, label, hw_nm in LINES:
    lp = LineProfile(n_u=n_u, n_l=n_l, Z=Z, B=B, Ne_m3=Ne, Te_ev=Te)

    wl_grid = np.linspace(lp.E0_wavelength_nm - hw_nm, lp.E0_wavelength_nm + hw_nm, 1000)
    print(f"Computing {label} (n={n_u}→{n_l}), λ₀={lp.E0_wavelength_nm:.2f} nm …", flush=True)

    lp.compute_profile(wl_grid, grid_type='wavelength_nm', **PROFILE_KWARGS)
    lp.compute_discrete(Fz=0.0, Fx=0.0, include_quadratic=False)
    profiles[label] = lp
    print(f"  → {len(lp.discrete.energy_ev)} discrete transitions", flush=True)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(LINES), 1, figsize=(10, 9), sharex=False)
fig.suptitle(
    rf"Balmer series — Stark-Zeeman  |  B = {B} T,  "
    rf"$N_e$ = {Ne:.0e} m$^{{-3}}$,  $T_e$ = {Te} eV",
    fontsize=12,
)

COLORS_Q = {0: "black", 1: "C3", -1: "C2"}
LABELS_Q  = {0: "π", 1: "σ+", -1: "σ−"}

for ax, (n_u, n_l, label, _) in zip(axes, LINES):
    lp   = profiles[label]
    disc = lp.discrete

    # Broadened profiles — normalise to peak transverse intensity
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
