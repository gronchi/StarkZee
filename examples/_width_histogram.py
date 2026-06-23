import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.broadening import electron_impact_width
from starkzee.radiator import einstein_a
from scipy.constants import hbar as _HBAR, e as _E_CHARGE

# Parameters from example_halpha.py
Z, n_u, n_l = 1, 5, 2
Te_ev = 0.5
E0 = 1.8892   # H-alpha ~eV

B_VALS  = [0.0, 3.0, 10.0]
NE_ROWS = [(1e17, "1e17"), (1e22, "1e22")]
det_range_ev = 2e-3   # ±1 meV

# Natural linewidth
gamma_upper = sum(einstein_a(n_u, k, Z) for k in range(1, n_u))
gamma_lower = sum(einstein_a(n_l, k, Z) for k in range(1, n_l))
w_natural_ev = _HBAR * (gamma_upper + gamma_lower) / 2.0 / _E_CHARGE
print(f"Natural linewidth: {w_natural_ev*1e6:.3f} µeV")

detuning = np.linspace(-det_range_ev, det_range_ev, 1000)

fig, axes = plt.subplots(len(NE_ROWS), len(B_VALS), figsize=(11, 6), sharey='row')
fig.suptitle("Distribution of electron-impact width w(Δ) over ±1 meV grid  [H-alpha, Te=0.5 eV]")

for row, (Ne, Ne_label) in enumerate(NE_ROWS):
    for col, B in enumerate(B_VALS):
        ax = axes[row, col]

        w = electron_impact_width(detuning, Ne, Te_ev, B, Z, n=n_u) + w_natural_ev
        w_uev = w * 1e6  # convert to meV

        w_min, w_max = w_uev.min(), w_uev.max()
        variation = (w_max - w_min) / w_uev.mean() * 100

        ax.hist(w_uev, bins=60, color='steelblue', edgecolor='none', alpha=0.8)
        ax.axvline(w_uev.mean(), color='k', lw=1.2, label=f'mean = {w_uev.mean():.4f} meV')
        ax.axvline(w_min, color='r', lw=0.8, ls='--')
        ax.axvline(w_max, color='r', lw=0.8, ls='--', label=f'range = {variation:.1f}%')

        ax.set_title(f"Ne={Ne_label} m⁻³,  B={B:.0f} T", fontsize=9)
        ax.set_xlabel("w(Δ)  [meV]", fontsize=8)
        ax.set_ylabel("count" if col == 0 else "", fontsize=8)
        ax.legend(fontsize=7.5)

        print(f"Ne={Ne_label}, B={B:.0f}T:  mean={w_uev.mean():.5f} meV, "
              f"min={w_min:.5f}, max={w_max:.5f}, variation={variation:.2f}%")

plt.tight_layout()
plt.show()
plt.savefig("examples/_width_histogram.png", dpi=130, bbox_inches='tight')
print("saved _width_histogram.png")
