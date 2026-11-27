"""Generate the atomic-API and Stark-map figures used in the documentation.

From the repository root::

    python examples/atomic_api_and_stark_maps.py

Requires NumPy, SciPy and Matplotlib. Figures are written without opening a
window; use --output-dir to choose another destination. No ARC installation
or downloaded atomic data are required.
"""

from pathlib import Path
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# IMPORTS_BEGIN
from starkzee import atomic
from starkzee.line_profile import LineProfile
from starkzee.stark_map import StarkMap
# IMPORTS_END

BLUE, ORANGE, TEAL = "#2A6FB0", "#D9822B", "#2F9E8F"


def field_free_data():
    # FIELD_FREE_BEGIN
    r_a0 = np.linspace(0.0, 70.0, 1801)
    radial = [atomic.radial_wavefunction(r_a0, n=3, l=l)
              for l in (0, 1, 2)]
    upper_shells = np.arange(3, 9)
    strengths = np.array([atomic.line_strength(n, 2) for n in upper_shells])
    gf = np.array([atomic.oscillator_strength(n, 2) for n in upper_shells])
    rates = np.array([atomic.einstein_a(n, 2, A=1) for n in upper_shells])
    # FIELD_FREE_END
    for wavefunction in radial:
        np.testing.assert_allclose(
            trapezoid(r_a0**2 * np.abs(wavefunction)**2, r_a0),
            1.0, atol=2e-6, rtol=0,
        )
    return r_a0, radial, upper_shells, strengths, gf, rates


def dressed_data():
    # DRESSED_BEGIN
    lp = LineProfile(3, 2, B=5.0, Ne_m3=1e20, Te_ev=5.0, species="H")
    radial_3d = lp.radial_wavefunction(np.array([1.0, 5.0]), l=2)
    radial_3d_2p = lp.radial_dipole(l_u=2, l_l=1)
    shell_rate = lp.field_free_einstein_a_s
    lp.compute_discrete(Fz=1e6, Fx=1e6, min_strength=0.0)
    tr = lp.discrete
    transition_rates = tr.einstein_a_s
    upper_rates = tr.upper_partial_decay_rate_s
    # DRESSED_END
    # dressed strengths are in the reduced Bohr radius, line_strength in a0^2
    from starkzee.utils import reduced_mass_factor
    np.testing.assert_allclose(
        tr.strength.sum(), atomic.line_strength(3, 2) / reduced_mass_factor(1) ** 2, rtol=2e-13,
    )
    np.testing.assert_allclose(transition_rates.sum(), upper_rates.sum())
    assert upper_rates.shape == (18,)
    assert np.all(transition_rates >= 0)
    return lp, tr, transition_rates, upper_rates, shell_rate


def stark_map_data():
    # STARK_MAP_BEGIN
    fields_v_m = np.linspace(0.0, 3e6, 161)
    target = (3, 2, 2.5, 0.5)  # n, l, j, mj: 3d(5/2), mj=1/2
    maps = []
    for magnetic_field in (0.0, 5.0):
        result = StarkMap(shells=(2, 3, 4, 5), B=magnetic_field).compute(
            fields_v_m, angle_deg=0.0, target_state=target,
        )
        maps.append(result)
    # STARK_MAP_END
    for result in maps:
        np.testing.assert_allclose(result.highlights.sum(axis=1), 1.0, atol=2e-14)
        assert result.mj == 0.5
        assert result.energies_ev.shape == (161, 24)
    return maps


def field_free_figure(data):
    r, radial, shells, strengths, gf, rates = data
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), layout="constrained")
    for wavefunction, label, color in zip(radial, ("3s", "3p", "3d"), (BLUE, ORANGE, TEAL)):
        axes[0, 0].plot(r, r**2 * np.abs(wavefunction)**2, label=label, color=color, lw=1.7)
    axes[0, 0].set(xlim=(0, 45), xlabel=r"Radius ($a_0$)",
                   ylabel=r"$r^2 |R_{3l}(r)|^2$ ($a_0^{-1}$)")
    axes[0, 0].legend(frameon=False, ncol=3)
    axes[0, 0].set_title("(a) Radial probability density", loc="left")
    for ax, values, color, title, ylabel in (
        (axes[0, 1], strengths, BLUE, "(b) Balmer line strength", r"$S_{n2}$ ($a_0^2$)"),
        (axes[1, 0], gf, TEAL, "(c) Weighted absorption oscillator strength", r"$g_2 f_{2n}$"),
        (axes[1, 1], rates / 1e6, ORANGE, "(d) Shell-averaged partial E1 rate", r"$\overline{A}_{n\to2}$ ($10^6$ s$^{-1}$)"),
    ):
        ax.bar(shells, values, color=color, width=0.65)
        ax.set(xlabel=r"Upper shell $n$", ylabel=ylabel, xticks=shells)
        ax.set_title(title, loc="left")
        ax.set_ylim(bottom=0)
    fig.suptitle("Field-free H I atomic data", fontsize=13, fontweight="semibold")
    return fig


def dressed_figure(data):
    lp, tr, transition_rates, upper_rates, shell_rate = data
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.15), layout="constrained")
    detuning_mev = tr.detuning_ev * 1e3
    for q, label, color in ((0, r"$\pi$", BLUE), (-1, r"$\sigma^+$", ORANGE), (1, r"$\sigma^-$", TEAL)):
        select = tr.q == q
        axes[0].vlines(detuning_mev[select], 0, tr.strength[select],
                       colors=color, linewidth=1.3, alpha=0.8, label=label)
    axes[0].set(xlabel="Detuning from physical reference (meV)",
                ylabel=r"Transition strength ($a_0^2$)", ylim=(0, None))
    axes[0].legend(frameon=False, ncol=3)
    axes[0].set_title("(a) Individual dressed transitions", loc="left")
    axes[1].bar(np.arange(len(upper_rates)), upper_rates / 1e6,
                color=BLUE, width=0.75)
    axes[1].axhline(shell_rate / 1e6, color=ORANGE, linestyle="--", lw=1.4,
                    label="Field-free shell average")
    axes[1].set(xlabel="Upper dressed-state index", ylabel=r"Partial rate to $n=2$ ($10^6$ s$^{-1}$)",
                ylim=(0, None), xticks=np.arange(0, len(upper_rates), 3))
    axes[1].legend(frameon=False, loc="upper right", fontsize=8.5)
    axes[1].set_title("(b) Sum over lower states and polarizations", loc="left")
    fig.suptitle(r"H $\alpha$: $B=5$ T, $F_z=F_x=10^6$ V/m",
                 fontsize=13, fontweight="semibold")
    return fig


def stark_map_figure(maps):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), layout="constrained")
    for ax, result in zip(axes, maps):
        # STARK_PLOT_BEGIN
        result.plot(unit="GHz", reference="target_zero_field",
                    field_unit="V/cm", ax=ax, colorbar=False, point_size=8)
        # STARK_PLOT_END
        ax.set_ylim(-500, 500)
        ax.set_title(f"B = {result.B:g} T; parallel fields", loc="left")
        ax.set_ylabel(r"$(E-E_{\mathrm{ref}})/h$ (GHz)")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    scatter = axes[-1].collections[0]
    fig.colorbar(scatter, ax=axes, label=r"$|\langle 3d_{5/2},m_j=1/2|\psi\rangle|^2$",
                 pad=0.025, shrink=0.88)
    fig.suptitle(r"Hydrogen Stark maps: shells $n=2\ldots5$, $m_j=1/2$",
                 fontsize=13, fontweight="semibold")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT/'docs/figures/examples')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.titlesize': 10, 'axes.labelsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': 0.15, 'axes.axisbelow': True,
        'savefig.facecolor': 'white',
    })
    free = field_free_data()
    dressed = dressed_data()
    maps = stark_map_data()
    for name, fig in (
        ('atomic_field_free.png', field_free_figure(free)),
        ('atomic_dressed_transitions.png', dressed_figure(dressed)),
        ('hydrogen_stark_map.png', stark_map_figure(maps)),
    ):
        path = args.output_dir/name
        fig.savefig(path, dpi=220)
        plt.close(fig)
        print(path)
    print(f'Balmer-alpha strength sum: {dressed[1].strength.sum():.8f} a0^2')
    print(f'Shell-averaged H-alpha rate: {dressed[4]:.8g} s^-1')
    print(f'Each map: {maps[0].energies_ev.shape[0]} fields, '
          f'{maps[0].energies_ev.shape[1]} selected basis states')


if __name__ == '__main__':
    main()
