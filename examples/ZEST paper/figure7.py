"""
Reproduce Figure 7 of the ZEST paper.
Plots Hydrogen Lyman-beta line shapes at Ne = 10^17 cm^-3 and T = 1, 10, 100 eV.
FFM ion dynamics, Holtsmark microfield, ZEST electron model.
Doppler broadening is omitted.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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


def simulate_figure7():
    print("Simulating Figure 7...")

    n_u, n_l, Z = 3, 1, 1
    ne = 1e23   # m^-3

    E0_ev = Z**2 * reduced_mass_rydberg_ev(Z, 1) * (1.0/n_l**2 - 1.0/n_u**2)
    E0_cm = energy_ev_to_wavenumber_cm(E0_ev)

    wavenumbers_calc = np.linspace(-600.0, 600.0, 1200)   # cm^-1
    energies_ev = wavenumber_cm_to_energy_ev(E0_cm + wavenumbers_calc)

    wn_per_ev = energy_ev_to_wavenumber_cm(1.0)

    temperatures = [1.0, 10.0, 100.0]
    profiles = {}

    for T in temperatures:
        print(f"  T = {T} eV ...")
        pi, sp, sm = calculate_ffm_profile(
            n_u=n_u, n_l=n_l, Z=Z, B=0.0,
            Ne_m3=ne, Te_ev=T, Ti_ev=T, A_ion=1,
            energies_ev=energies_ev,
            num_f=800, num_mu=6,
            use_screening=False,
            electron_model='zest',
        )
        I_cm = (pi + sp + sm) / wn_per_ev
        area = np.trapezoid(I_cm, wavenumbers_calc)
        profiles[T] = I_cm / area
        print(f"    area = {area:.5e},  peak = {np.max(profiles[T]):.5e}")

    fig, ax = plt.subplots(layout='constrained')
    ax.plot(wavenumbers_calc, profiles[1.0],   label='T=1 eV',   color='black',     linestyle='-')
    ax.plot(wavenumbers_calc, profiles[10.0],  label='T=10 eV',  color='red',       linestyle=':')
    ax.plot(wavenumbers_calc, profiles[100.0], label='T=100 eV', color='limegreen', linestyle='--')

    ax.set_title(r"Figure 7. Lyman-$\beta$ line shapes of Hydrogen at $N_e=10^{17}\,\mathrm{cm}^{-3}$",
                 fontweight='bold')
    ax.set_xlabel(r"Photon energy ($\mathrm{cm}^{-1}$)")
    ax.set_ylabel("Normalized Intensity (cm)")
    ax.set_ylim(0.0, 9e-3)
    ax.set_xlim(-150.0, 150.0)
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_yticks([0.0, 2e-3, 4e-3, 6e-3, 8e-3])

    def format_y(x, pos):
        if x == 0:
            return "0"
        return f"{int(round(x * 1e3))}$\\times 10^{{-3}}$"

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_y))
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    _setup_style()
    simulate_figure7()
