"""C IV n = 5 -> 4 (253 nm) Stark-Zeeman profiles from NIST levels.

Lithium-like C IV: the n = 5 (5s-5g) and n = 4 (4s-4f) shells are Stark-mixed
by the plasma microfield; reduced elements from the Coulomb approximation
(within 1.3-2.6% of the NIST grade-A line strengths). Conditions of the
gas-liner pinch measurements of Glenzer et al., J. Phys. B 27, 5507 (1994),
also used in Ferri et al., Matter Radiat. Extremes 7, 015901 (2022), Fig. 6:
n_e = 2e18 cm^-3, Te = Ti = 10 eV.

Electron broadening uses the per-state impact operator (``electron_operator=True``),
which is fast at every B. The width depends strongly on the electron option: at
B = 0 the FWHM is 19.5 A (scalar widths), 10.8 A (per-state operator) and 5.4 A
(full PPP operator with interference, ``electron_interference=True,
frequency_dependent_width=False``; about 10 s at B = 0, much slower at B > 0).

Run:  python examples/c_iv_n5_n4_stark_zeeman.py [--plot]
"""
import sys
import warnings

import numpy as np

from starkzee.single_valence import air_wavelength_A, c_iv_5_4
from starkzee.static_profile import calculate_static_profile

EV_TO_CM = 8065.543937

warnings.simplefilter("ignore", UserWarning)
ne, Te = 2e24, 10.0
profiles = {}
for B in (0.0, 100.0, 300.0, 500.0):
    radiator = c_iv_5_4(B)
    E0 = radiator.line_energy_ev
    grid = np.linspace(E0 - 0.05, E0 + 0.05, 1601)
    pi, sp, sm = calculate_static_profile(5, 4, 4, B, ne, Te, grid, num_f=16, num_mu=6, radiator=radiator,
                                          Ti_ev=Te, electron_operator=True)
    lam = np.array([air_wavelength_A(E * EV_TO_CM) for E in grid])
    profiles[B] = (lam, pi, 0.5 * (sp + sm))
    sig = 0.5 * (sp + sm)
    fwhm = {k: np.ptp(lam[p >= p.max() / 2]) for k, p in (("pi", pi), ("sigma", sig))}
    peaks = lam[1:-1][(sig[1:-1] > sig[:-2]) & (sig[1:-1] > sig[2:]) & (sig[1:-1] > 0.5 * sig.max())]
    print(f"B = {B:5.0f} T: FWHM pi {fwhm['pi']:.2f} A, sigma {fwhm['sigma']:.2f} A;"
          f" sigma peaks at {np.round(peaks, 2)} A")

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for B, (lam, pi, sig) in profiles.items():
        norm = (pi + sig).max()
        a1.plot(lam, sig / norm, label=f"B = {B:.0f} T")
        a2.plot(lam, pi / norm, label=f"B = {B:.0f} T")
    a1.set_title(r"$\sigma$"), a2.set_title(r"$\pi$")
    for a in (a1, a2):
        a.set_xlabel("air wavelength [Å]")
    a1.set_ylabel("intensity (normalized to the B = 0 peak of each)")
    a1.legend()
    fig.suptitle(r"C IV n = 5 $\to$ 4, $n_e$ = 2e18 cm$^{-3}$, $T_e$ = $T_i$ = 10 eV")
    plt.tight_layout()
    plt.show()
