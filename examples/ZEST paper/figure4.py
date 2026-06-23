"""
Reproduce Figure 4 of the ZEST paper.
Plots Hydrogen Lyman-10 line shapes at Ne = 10^13 cm^-3, Te = 1 eV for B = 0, 10, 20 T,
using StarkZee with the ZEST electron model and FFM ion dynamics.
Doppler broadening is omitted.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from starkzee.ffm import calculate_ffm_profile
from starkzee.utils import (
    reduced_mass_rydberg_ev,
    energy_ev_to_wavenumber_cm,
    wavenumber_cm_to_energy_ev,
)


def _setup_style():
    mpl.rcParams.update({
        'font.family':           'serif',
        'font.size':             9,
        'axes.labelsize':        10,
        'axes.titlesize':        9,
        'axes.linewidth':        0.8,
        'legend.fontsize':       9,
        'legend.framealpha':     0.9,
        'legend.edgecolor':      '0.75',
        'legend.handlelength':   2.5,
        'xtick.labelsize':       9,
        'ytick.labelsize':       9,
        'xtick.direction':       'in',
        'ytick.direction':       'in',
        'xtick.top':             True,
        'ytick.right':           True,
        'xtick.minor.visible':   True,
        'ytick.minor.visible':   True,
        'xtick.major.width':     0.8,
        'ytick.major.width':     0.8,
        'xtick.minor.width':     0.5,
        'ytick.minor.width':     0.5,
        'xtick.major.size':      4,
        'ytick.major.size':      4,
        'xtick.minor.size':      2,
        'ytick.minor.size':      2,
        'lines.linewidth':       1.5,
    })


def simulate_figure4():
    print("Simulating Figure 4...")

    n_u, n_l, Z = 10, 1, 1
    ne    = 1e19   # m^-3  (10^13 cm^-3)
    Te_ev = 1.0    # eV
    Ti_ev = 1.0    # eV — drives FFM ion jumping rate only (no Doppler in FFM)

    E0_ev = Z**2 * reduced_mass_rydberg_ev(Z, 1) * (1.0/n_l**2 - 1.0/n_u**2)
    E0_cm = energy_ev_to_wavenumber_cm(E0_ev)

    wavenumbers_rel = np.linspace(-35.0, 35.0, 600)   # cm^-1
    energies_ev = wavenumber_cm_to_energy_ev(E0_cm + wavenumbers_rel)

    b_fields    = [0.0, 10.0, 20.0]
    line_styles = {
        0.0:  ('black', '-',  '0 T'),
        10.0: ('red',   ':',  '10 T'),
        20.0: ('green', '--', '20 T'),
    }

    profiles = {}
    for B in b_fields:
        print(f"  B = {B} T ...")
        pi, sp, sm = calculate_ffm_profile(
            n_u=n_u, n_l=n_l, Z=Z, B=B,
            Ne_m3=ne, Te_ev=Te_ev, Ti_ev=Ti_ev, A_ion=1,
            energies_ev=energies_ev,
            num_f=100, num_mu=12,
            use_screening=False,
            electron_model='zest',
            quadratic_zeeman=False,
            fine_structure=False,
            parallel_stark=True,
        )
        profiles[B] = pi + sp + sm

    max_B0 = np.max(profiles[0.0])

    fig, ax = plt.subplots(layout='constrained')

    for B in b_fields:
        color, linestyle, label = line_styles[B]
        I_tot = profiles[B]
        E_cg  = np.sum(energies_ev * I_tot) / np.sum(I_tot)
        wn_rel = energy_ev_to_wavenumber_cm(energies_ev) - energy_ev_to_wavenumber_cm(E_cg)
        ax.plot(wn_rel, I_tot / max_B0, label=label, color=color, linestyle=linestyle)

    ax.set_title("Figure 4: Hydrogen Lyman-10 line shapes", fontweight='bold')
    ax.set_xlabel(r"Photon energy ($\mathrm{cm}^{-1}$)")
    ax.set_ylabel("Normalized Intensity")
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 1.1)
    ax.set_xlim(-30.0, 30.0)
    ax.set_xticks([-30, -20, -10, 0, 10, 20, 30])
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(loc='lower center')
    plt.show()


if __name__ == '__main__':
    _setup_style()
    simulate_figure4()
