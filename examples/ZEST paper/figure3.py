"""
Reproduce Figure 3 of the ZEST paper.
Plots microfield distributions at a charged emitter for different Gamma and U (s),
using StarkZee microfield functions.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from starkzee.microfield import (
    potekhin_distribution,
    hooper_distribution,
    _P_from_Q_grid,
    _Q_Mayer,
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


def simulate_figure3():
    print("Simulating Figure 3...")

    beta = np.linspace(0.0, 5.0, 500)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,5),layout='constrained')

    # --- Left Column: U = 1 (s = 1.0) ---
    gammas_left = [10.0, 5.0, 1.0, 0.0]
    colors_left = ['C0', 'C1', 'C2', 'C3']

    for gamma, color in zip(gammas_left, colors_left):
        p = potekhin_distribution(beta, gamma=gamma, s=1.0, charged=True)
        ax1.plot(beta, p, label=f"$\\Gamma = {int(gamma)}$", color=color)

    p_hooper_screened = hooper_distribution(beta, a=1.0, charged=True)
    ax1.plot(beta, p_hooper_screened, label="Hooper ($a = 1.0$)",
             color='black', linestyle='--')

    ax1.set_title("Charged Emitter with Screening ($U = 1$)", fontweight='bold')
    ax1.set_xlabel(r"$\beta = E/E_0$")
    ax1.set_ylabel(r"$P(\beta)$")
    ax1.set_xlim(0.0, 5.0)
    ax1.set_ylim(0.0, 2.5)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # --- Right Column: U = 0 (s = 0.0) ---
    # Mayer curve (Gamma = 10) — passes gamma directly as gamma_eff to _Q_Mayer
    p_mayer = _P_from_Q_grid(_Q_Mayer, beta, 10.0)
    ax2.plot(beta, p_mayer, label=r"Mayer ($\Gamma = 10$)",
             color='k', linestyle=':')

    gammas_right = [10.0, 5.0, 1.0, 0.0]
    colors_right = ['C0', 'C1', 'C2', 'C3']

    for gamma, color in zip(gammas_right, colors_right):
        p = potekhin_distribution(beta, gamma=gamma, s=0.0, charged=True)
        ax2.plot(beta, p, label=f"$\\Gamma = {int(gamma)}$", color=color)

    p_holtz = potekhin_distribution(beta, gamma=0.0, s=0.0, charged=False)
    ax2.plot(beta, p_holtz, label="Holtzmark", color='gray', linestyle='--')

    p_hooper_unscreened = hooper_distribution(beta, a=0.0, charged=True)
    ax2.plot(beta, p_hooper_unscreened, label="Hooper ($a = 0.0$)",
             color='purple', linestyle=':')

    ax2.set_title("Charged Emitter without Screening ($U = 0$)", fontweight='bold')
    ax2.set_xlabel(r"$\beta = E/E_0$")
    ax2.set_ylabel(r"$P(\beta)$")
    ax2.set_xlim(0.0, 5.0)
    ax2.set_ylim(0.0, 2.5)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    fig.suptitle("Figure 3: Charged Emitter Microfield Distributions", fontweight='bold')
    plt.show()


if __name__ == '__main__':
    _setup_style()
    simulate_figure3()
