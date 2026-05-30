"""
example_halpha.py — H Balmer-alpha (n=3→2) Stark-Zeeman profiles.

Shows static profiles for B = 1, 5, 10 T at two electron densities:
  • Ne = 10^20 m^-3 : Zeeman-dominated  (±5 meV window)
  • Ne = 10^23 m^-3 : Stark-dominated   (±30 meV window)
Te = 5 eV throughout.

Run:
    python example_halpha.py

Produces:  example_halpha.png
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.static_profile import calculate_static_profile, discrete_transitions
from starkzee.utils import RYDBERG_EV

# ── Parameters ────────────────────────────────────────────────────────────────

Z, n_u, n_l = 1, 3, 2
Te_ev       = 5.0
E0 = (Z**2) * RYDBERG_EV * (1.0 / n_l**2 - 1.0 / n_u**2)

B_VALS  = [1.0, 5.0, 10.0]        # Tesla
NE_ROWS = [
    (1e20, 5.0),   # (Ne m^-3, detuning half-range meV)
    (1e23, 30.0),
]
NPTS = 1200

POL_COLOR = {0: "#e74c3c", -1: "#3498db", 1: "#2ecc71"}

# ── Precompute stick spectra (field-independent per B) ────────────────────────

print("Computing stick spectra …")
sticks = {}
for B in B_VALS:
    sticks[B] = discrete_transitions(
        n_u=n_u, n_l=n_l, Z=Z, B=B,
        include_fine_structure=True, min_strength=1e-5
    )

# ── Figure ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

for row, (Ne, det_range) in enumerate(NE_ROWS):
    ne_exp   = int(np.log10(Ne))
    det_mev  = np.linspace(-det_range, det_range, NPTS)
    energies = E0 + det_mev * 1e-3

    for col, B in enumerate(B_VALS):
        ax = axes[row, col]

        print(f"  B={B:.0f} T,  Ne=1e{ne_exp} cm⁻³ …", flush=True)

        pi, sp, sm = calculate_static_profile(
            n_u=n_u, n_l=n_l, Z=Z, B=B,
            Ne_m3=Ne, Te_ev=Te_ev,
            energies_ev=energies,
            num_f=25, num_mu=8,
            include_fine_structure=True,
            frequency_dependent_width=False,
        )
        total = pi + sp + sm
        norm  = total.max() or 1.0

        # ── Profile curves ────────────────────────────────────────────────
        ax.fill_between(det_mev, total / norm, alpha=0.10, color="k")
        ax.plot(det_mev, total / norm, "k",            lw=1.8, label="Total")
        ax.plot(det_mev, pi / norm,    color=POL_COLOR[0],
                lw=1.1, ls="--", alpha=0.85, label="π")
        ax.plot(det_mev, (sp + sm) / norm, color=POL_COLOR[-1],
                lw=1.1, ls=":",  alpha=0.85, label="σ+σ−")

        # ── Stick spectrum overlay (scaled to 0.45 of plot height) ────────
        tr      = sticks[B]
        det_stk = (tr["energy_ev"] - E0) * 1e3
        s_stk   = tr["strength"] / tr["strength"].max() * 0.45
        for q in [0, -1, 1]:
            mask = (tr["q"] == q) & (np.abs(det_stk) <= det_range)
            if mask.any():
                ax.vlines(det_stk[mask], 0, s_stk[mask],
                          colors=POL_COLOR[q], alpha=0.35, lw=1.0)

        # ── Axes decoration ───────────────────────────────────────────────
        ax.set_title(
            f"B = {B:.0f} T  |  $N_e = 10^{{{ne_exp}}}$ m$^{{-3}}$",
            fontsize=10
        )
        ax.set_xlabel("Detuning from $E_0$ (meV)", fontsize=9)
        ax.set_ylabel("Norm. intensity",            fontsize=9)
        ax.set_xlim(-det_range, det_range)
        ax.set_ylim(0, 1.15)
        ax.axhline(0, color="grey", lw=0.4)
        ax.tick_params(labelsize=8)

        if row == 0 and col == 0:
            ax.legend(fontsize=8, framealpha=0.75, loc="upper right")

print("Done.  Saving example_halpha.png …")

fig.suptitle(
    f"H Balmer-α  (n=3→2, Z=1)  ·  Static Stark-Zeeman profile  ·"
    f"  $T_e$ = {Te_ev:.0f} eV",
    fontsize=13
)
plt.savefig("example_halpha.png", dpi=150, bbox_inches="tight")
plt.show()
