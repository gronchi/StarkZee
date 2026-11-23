"""He I 447.1 nm (2p 3P - 4d 3D) Stark-Zeeman profiles from NIST data.

The n = 4 triplet manifold (4s, 4p, 4d, 4f) is Stark-mixed by the plasma
microfield; the field-induced 2p - 4f component near 447.0 nm grows and moves
away from the allowed line as the density increases. Reduced elements: Coulomb
approximation, with the 2p-4d magnitude scaled to NIST.

Run:  python examples/he_i_4471_stark_zeeman.py [--plot]
"""
import sys
import warnings

import numpy as np

from starkzee.he_i import forbidden_component_energy_ev, he_i_4471
from starkzee.static_profile import calculate_static_profile

HC_EV_A = 12398.419843320026          # hc [eV Angstrom]

warnings.simplefilter("ignore", UserWarning)
B, Te, Ti = 2.0, 5.0, 2.0
radiator = he_i_4471(B)
E_a, E_f = radiator.line_energy_ev, forbidden_component_energy_ev()
grid = np.linspace(E_a - 3e-3, E_a + 4e-3, 1401)
profiles = {}
for ne in (1e20, 1e21, 3e21):
    pi, sp, sm = calculate_static_profile(4, 2, 1, B, ne, Te, grid, num_f=16, num_mu=6,
                                          radiator=radiator, Ti_ev=Ti)   # Doppler mass from the radiator
    profiles[ne] = pi + 0.5 * (sp + sm)                  # observed perpendicular to B
    print(f"n_e = {ne:.0e} m^-3: peak {profiles[ne].max():.3e} /eV")

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt
    lam = HC_EV_A / grid                                  # vacuum wavelength [A]
    for ne, p in profiles.items():
        plt.plot(lam, p / p.max(), label=f"$n_e$ = {ne:.0e} m$^{{-3}}$")
    plt.axvline(HC_EV_A / E_f, ls=":", c="k", lw=0.8)
    plt.xlabel("vacuum wavelength [Å]")
    plt.ylabel("normalized intensity")
    plt.title(f"He I 447.1 nm, B = {B} T, $T_e$ = {Te} eV, $T_i$ = {Ti} eV")
    plt.legend()
    plt.show()
