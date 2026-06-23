"""
Reproduce Figure 6 of the ZEST paper.
Plots Hydrogen Lyman-alpha line shapes at Ne = 10^17 cm^-3, T = 100 eV.
Comparing Quasi-Static (QS) and FFM ion dynamics.
Doppler is omitted; Holtsmark (unscreened) microfield; ZEST electron model.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from starkzee.static_profile import calculate_static_profile
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


def simulate_figure6():
    print("Simulating Figure 6...")

    n_u, n_l, Z = 2, 1, 1
    ne    = 1e23    # m^-3  (10^17 cm^-3)
    Te_ev = 100.0   # eV
    Ti_ev = 100.0   # eV

    E0_ev = Z**2 * reduced_mass_rydberg_ev(Z, 1) * (1.0/n_l**2 - 1.0/n_u**2)
    E0_cm = energy_ev_to_wavenumber_cm(E0_ev)

    wavenumbers_grid = np.linspace(-125.0, 125.0, 800)   # cm^-1
    energies_ev = wavenumber_cm_to_energy_ev(E0_cm + wavenumbers_grid)

    print("  QS profile ...")
    pi_qs, sp_qs, sm_qs = calculate_static_profile(
        n_u=n_u, n_l=n_l, Z=Z, B=0.0,
        Ne_m3=ne, Te_ev=Te_ev, Ti_ev=None,
        energies_ev=energies_ev,
        num_f=800, num_mu=6,
        use_screening=False,
        electron_model='zest',
    )
    I_qs = pi_qs + sp_qs + sm_qs

    print("  FFM profile ...")
    pi_ffm, sp_ffm, sm_ffm = calculate_ffm_profile(
        n_u=n_u, n_l=n_l, Z=Z, B=0.0,
        Ne_m3=ne, Te_ev=Te_ev, Ti_ev=Ti_ev, A_ion=1,
        energies_ev=energies_ev,
        num_f=800, num_mu=6,
        use_screening=False,
        electron_model='zest',
    )
    I_ffm = pi_ffm + sp_ffm + sm_ffm

    wn_per_ev = energy_ev_to_wavenumber_cm(1.0)
    I_qs_cm   = I_qs  / wn_per_ev
    I_ffm_cm  = I_ffm / wn_per_ev

    area_qs  = np.trapezoid(I_qs_cm,  wavenumbers_grid)
    area_ffm = np.trapezoid(I_ffm_cm, wavenumbers_grid)
    print(f"  QS area = {area_qs:.5e},  FFM area = {area_ffm:.5e}")

    I_qs_norm  = I_qs_cm  / area_qs
    I_ffm_norm = I_ffm_cm / area_ffm

    fig, ax = plt.subplots(layout='constrained')
    ax.plot(wavenumbers_grid, I_ffm_norm, label='FFM', color='black', linestyle='-')
    ax.plot(wavenumbers_grid, I_qs_norm,  label='QS',  color='black', linestyle=':')

    ax.set_title(r"Figure 6. Lyman-$\alpha$ line shapes of hydrogen", fontweight='bold')
    ax.set_xlabel(r"Photon energy relative to center of gravity ($\mathrm{cm}^{-1}$)")
    ax.set_ylabel("Normalized Intensity (cm)")
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 1.0)
    ax.set_xlim(-125.0, 125.0)
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    _setup_style()
    simulate_figure6()
