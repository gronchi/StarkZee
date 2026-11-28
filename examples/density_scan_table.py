"""Balmer-alpha profiles over a density scan from one set of dressed states.

StaticProfileTable diagonalizes the Stark-Zeeman Hamiltonian once per (field, angle)
node of a fixed geometric field grid; every (Ne, Te, Ti) then reuses those states and
recomputes only the microfield weights, the widths and the line-shape sum. The script
times the scan against independent calculate_static_profile calls and reports the
quadrature difference between the two rules (both against a fine reference).

Run:  python examples/density_scan_table.py [--plot]
"""
import sys
import time
import warnings

import numpy as np

from starkzee.profile_table import StaticProfileTable
from starkzee.static_profile import calculate_static_profile, line_reference_energy

warnings.simplefilter("ignore", UserWarning)
B, Te, Ti = 2.0, 5.0, 5.0
E0 = line_reference_energy(3, 2, 1)
grid = np.linspace(E0 - 4e-3, E0 + 4e-3, 800)
densities = np.geomspace(1e19, 1e21, 9)

table = StaticProfileTable(3, 2, 1, B, (densities[0], densities[-1]), num_f=30)
t0 = time.perf_counter()
scan = [table.profile(grid, ne, Te, Ti_ev=Ti) for ne in densities]
t_table = time.perf_counter() - t0
t0 = time.perf_counter()
scan_fft = [table.profile(grid, ne, Te, Ti_ev=Ti, method="fft") for ne in densities]
t_fft = time.perf_counter() - t0
t0 = time.perf_counter()
direct = [calculate_static_profile(3, 2, 1, B, ne, Te, grid, num_f=30, Ti_ev=Ti) for ne in densities]
t_direct = time.perf_counter() - t0

print(f"{len(densities)} densities: table {t_table:.2f} s (first pass fills the cache, "
      f"{table.cached_points} nodes), table + fft {t_fft:.2f} s, independent calls {t_direct:.2f} s")
for ne, a, b in zip(densities, scan, direct):
    ref = calculate_static_profile(3, 2, 1, B, ne, Te, grid, num_f=300, Ti_ev=Ti)
    peak = np.max(ref)
    print(f"  n_e = {ne:.1e} m^-3: quadrature error (vs num_f=300) table {np.max(np.abs(np.array(a) - ref)) / peak:.1e}, "
          f"standard {np.max(np.abs(np.array(b) - ref)) / peak:.1e}")

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt
    for ne, (pi, sp, sm) in zip(densities, scan):
        total = pi + 0.5 * (sp + sm)
        plt.plot((grid - E0) * 1e3, total / total.max(), label=f"{ne:.1e}")
    plt.xlabel("E - E0 [meV]")
    plt.ylabel("normalized intensity")
    plt.legend(title="$n_e$ [m$^{-3}$]", fontsize=7)
    plt.title(f"Balmer-alpha, B = {B} T, $T_e$ = $T_i$ = {Te} eV")
    plt.show()
