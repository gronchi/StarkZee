"""
Simulate and reproduce Figure 2 of Gilleron & Pain (2018).
Plots G(0)_Lee / G(0)_GBK vs x = kappa_m * lambda_D.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import scipy.special as special
import matplotlib as mpl
import matplotlib.pyplot as plt


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


def simulate_figure2():
    print("Simulating Figure 2...")

    # Grid of x = kappa_m * lambda_D from 10^0 to 10^4
    x = np.logspace(0.0, 4.0, 400)

    g_lee_0 = np.log(1.0 + x**2) - x**2 / (1.0 + x**2)
    g_gbk_0 = special.exp1(1.0 / (2.0 * x**2))
    ratio_exact = g_lee_0 / g_gbk_0

    ratio_approx = (np.log(x) - 0.5) / (np.log(x) + 0.058)

    fig, ax = plt.subplots(layout='constrained')

    ax.plot(x, ratio_exact, label='Eq. (12) / Eq. (14)')
    ax.plot(x, ratio_approx, label=r'$(\ln(x) - 0.5) / (\ln(x) + 0.058)$', linestyle='--')

    ax.set_xscale('log')
    ax.set_xlim(1.0, 10000.0)
    ax.set_ylim(0.2, 1.0)

    ax.set_title("Figure 2: Ratio of Electron Impact Widths at Line Center (Lee / GBK)",
                 fontweight='bold')
    ax.set_xlabel(r"$x = \kappa_m \lambda_D$")
    ax.set_ylabel(r"$G_{\text{Lee}}(0) / G_{\text{GBK}}(0)$")
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    ax.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    _setup_style()
    simulate_figure2()
