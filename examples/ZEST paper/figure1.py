"""
Reproduce Figure 1 of Gilleron & Pain (2018).
Plots G(dw) vs reduced detuning dw/omega_p for kappa_m * lambda_D = 30,
using StarkZee broadening functions with the ZEST electron model.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, e as E_CHARGE, hbar as HBAR

from starkzee.broadening import (
    gbk_zest_model, lee_model, dufty_model,
    calculate_plasma_frequency, calculate_cutoff_kappa_m,
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


def simulate_figure1():
    print("Simulating Figure 1...")

    Z, n = 1, 2
    Te_ev = 1e5 / 11604.525  # ≈ 8.617 eV  (matches ZEST te = 1e5 K)

    # Choose ne such that kappa_m * lambda_D = 30 for Z=1, n=2 at this Te.
    kappa_m = calculate_cutoff_kappa_m(Z, n, Te_ev)
    lambda_D_target = 30.0 / kappa_m
    ne = epsilon_0 * Te_ev / (lambda_D_target**2 * E_CHARGE)

    omega_p_ev = calculate_plasma_frequency(ne) * HBAR / E_CHARGE

    # Reduced detuning x-axis: dw / omega_p from 0 to 15
    w_red = np.linspace(0.0, 15.0, 150)
    dw_ev = w_red * omega_p_ev

    g_gbk   = gbk_zest_model(dw_ev, ne, Te_ev, Z, n)
    g_lee   = lee_model(dw_ev, ne, Te_ev, Z, n)
    g_dufty = dufty_model(dw_ev, ne, Te_ev, Z, n)

    fig, ax = plt.subplots(layout='constrained')

    ax.plot(w_red, g_dufty, label='Dufty RPA')
    ax.plot(w_red, g_lee,   label='Lee', linestyle='--')
    ax.plot(w_red, g_gbk,   label='GBK', linestyle=':')

    ax.set_title("Figure 1: Electron Broadening G-function vs. Detuning",
                 fontweight='bold')
    ax.set_xlabel(r"$\Delta\omega / \omega_p$")
    ax.set_ylabel(r"$G(\Delta\omega)$")
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.set_ylim(0,4)
    ax.set_xlim(0,15)
    ax.legend()
    plt.show()


if __name__ == '__main__':
    _setup_style()
    simulate_figure1()
