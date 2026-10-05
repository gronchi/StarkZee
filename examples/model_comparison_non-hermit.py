"""
model_comparison_non-hermit.py - Compare StarkZee's non-Hermitian PPP
impact-limit collision model, using the intra-shell closure, against analytical
and tabulated models for D-alpha and D-gamma.

This is the non-Hermitian collision-operator counterpart of model_comparison.py. Both
StarkZee curves opt into ``electron_interference=True``. The static solver uses
the generalized Lorentzian residues ``a_k + i c_k``; the FFM carries the same
complex residues into its analytical Markov expression.

The optical-coherence eigensolve is substantially more expensive than the
default diagonal approximation, particularly for D-gamma. Use ``--quick`` for
a preview, or ``--num-f`` and ``--num-mu`` for convergence work.

Run directly::

    python examples/model_comparison_non-hermit.py
    python examples/model_comparison_non-hermit.py out/figure.png
    python examples/model_comparison_non-hermit.py out/preview.png --quick
"""

import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from starkzee.convolutions import calculate_doppler_width_ev
from starkzee.line_profile import LineProfile
import starkzee.models as models


def _make_figure(n_u, num_f=60, num_mu=11, include_ffm=True,
                 group_tolerance_ev=None, group_width_tolerance_ev=None,
                 group_profile_rtol=1e-3):
    # ── parameters ────────────────────────────────────────────────────────────────────────
    n_l = 2                    # Balmer transition
    species = 'D'             # emitting species: 'H', 'D', or 'T'
    Ne_m3 = 1e20              # electron density          [m⁻³]
    Te_ev = 1                 # electron temperature      [eV]
    Ti_ev = 1                 # ion temperature           [eV]
    B = 3.0                   # magnetic field            [T]
    view_angle_deg = 90.0     # observation angle to B    [deg]
    # The full operator evaluates its PPP coefficient directly. Keeping the
    # intra selector makes any accompanying scalar diagnostics use the same
    # selected-shell radius closure.
    electron_model = 'pppb-intra'
    # ─────────────────────────────────────────────────────────────────────────────────────────────────────

    lp = LineProfile(
        n_u=n_u, n_l=n_l, B=B, Ne_m3=Ne_m3, Te_ev=Te_ev, Ti_ev=Ti_ev,
        species=species, view_angle_deg=view_angle_deg,
    )

    delta_E_D = calculate_doppler_width_ev(lp.E0, Ti_ev, A_emitter=lp.A)
    delta_lambda_D_nm = lp.E0_wavelength_nm * delta_E_D / lp.E0
    half_width_nm = max(2.1, 4.0 * delta_lambda_D_nm)

    wl_sz_nm = np.linspace(
        lp.E0_wavelength_nm - half_width_nm,
        lp.E0_wavelength_nm + half_width_nm,
        3000,
    )
    line_name = {3: 'D-alpha', 5: 'D-gamma'}[n_u]
    print(f'--------\n{line_name} PPP intra-shell collision timings:\n--------')

    # This path implements the manual's selected-shell impact-limit G(0)
    # operator, so observation-frequency-dependent widths must be disabled.
    t0 = time.time()
    lp.compute_static_profile(
        wl_sz_nm,
        grid_type='wavelength_nm',
        num_f=num_f,
        num_mu=num_mu,
        use_empirical_data=True,
        atom=species,
        electron_model=electron_model,
        electron_interference=True,
        frequency_dependent_width=False,
    )
    print(f'starkzee (static, PPP intra-shell collision): {time.time() - t0:.3g} sec')

    lp_ffm = None
    ffm_profile = None
    if include_ffm:
        ffm_diagnostics = {}
        lp_ffm = LineProfile(
            n_u=n_u, n_l=n_l, B=B, Ne_m3=Ne_m3, Te_ev=Te_ev, Ti_ev=Ti_ev,
            species=species, view_angle_deg=view_angle_deg,
        )
        t0 = time.time()
        lp_ffm.compute_ffm_profile(
            wl_sz_nm,
            grid_type='wavelength_nm',
            num_f=num_f,
            num_mu=num_mu,
            use_empirical_data=True,
            atom=species,
            electron_model=electron_model,
            electron_interference=True,
            sdt_frequency_dependent_width=False,
            sdt_bin_tol=None,
            numerical_inversion=False,
            interference_diagnostics=ffm_diagnostics,
            interference_group_tolerance_ev=group_tolerance_ev,
            interference_group_width_tolerance_ev=group_width_tolerance_ev,
            interference_group_profile_rtol=group_profile_rtol,
        )
        ffm_profile = lp_ffm.profile
        print(
            'starkzee (ffm, PPP intra-shell collision): '
            f'{time.time() - t0:.3g} sec'
        )
        for q in (0, 1, -1):
            diagnostic = ffm_diagnostics[q]
            print(
                f'  q={q:+d}: {diagnostic["negative_mode_count"]}/'
                f'{diagnostic["mode_count"]} negative a_k; '
                f'{diagnostic["negative_absolute_fraction"]:.3%} of '
                'sum(|a_k|)'
            )
            if "grouping" in diagnostic:
                grouping = diagnostic["grouping"]
                print(
                    f'    grouped {grouping["original_mode_count"]} -> '
                    f'{grouping["grouped_mode_count"]} modes; static error '
                    f'{grouping["static_profile_relative_max_error"]:.3e}; '
                    'difference from modulus FFM '
                    f'{grouping["modulus_ffm_relative_max_difference"]:.3e}'
                )

    # Use the non-Hermitian static profile's intensity-weighted centroid as the
    # common physical line center for models that do not include fine structure.
    center_air_nm = float(
        np.sum(lp.wavelengths_air_nm * lp.profile) / np.sum(lp.profile)
    )
    wl_cmp_nm = np.linspace(
        center_air_nm - half_width_nm,
        center_air_nm + half_width_nm,
        5000,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[12, 5])
    fig.suptitle(
        f'{line_name} ($n={n_u}\\to2$), PPP impact-limit collision operator '
        '(intra-shell closure)\n'
        f'$n_e = {Ne_m3:.2g}$ m$^{{-3}}$,  '
        f'$T_i = {Ti_ev:.3g}$ eV,  $T_e = {Te_ev:.3g}$ eV\n'
        f'$B = {B:.3g}$ T,  $\\theta = {view_angle_deg:.3g}$°'
    )

    cmp_funcs = {
        'voigt': models.voigt,
        'stehle': models.stehle,
        'stehle_param': models.stehle_param,
        # 'lomanowski': models.lomanowski,
        'rosato': models.rosato,
    }

    for name, func in cmp_funcs.items():
        try:
            t0 = time.time()
            profile = func(
                wl_cmp_nm, n_u, n_l, B, Ne_m3, Te_ev, Ti_ev,
                view_angle_deg=view_angle_deg, species=species,
            )
            print(f'{name}: {time.time() - t0:.3g} sec')
            profile_norm = profile / profile.max()
            if name == 'rosato':
                # Plotting-only cutoff at 10^-6 of the Rosato peak. The model
                # and its source tables remain unchanged.
                profile_norm = np.where(
                    profile_norm >= 1e-6, profile_norm, np.nan
                )
            ax1.plot(wl_cmp_nm, profile_norm, label=name)
            ax2.plot(wl_cmp_nm, profile_norm, label=name)
        except Exception as exc:
            print(f'{name} failed: {exc}')
            traceback.print_exc()

    y = lp.profile / lp.profile.max()
    y_ffm = (ffm_profile / ffm_profile.max()
             if ffm_profile is not None else None)
    for ax in (ax1, ax2):
        ax.plot(
            lp.wavelengths_air_nm,
            y,
            'k--',
            linewidth=2,
            label='starkzee (static, PPP intra-shell collision)',
        )
        if y_ffm is not None:
            ax.plot(
                lp_ffm.wavelengths_air_nm,
                y_ffm,
                color='black',
                linestyle=':',
                marker='o',
                markevery=max(1, len(y_ffm) // 30),
                markersize=3.5,
                markerfacecolor='white',
                linewidth=2,
                label='starkzee (ffm, PPP intra-shell collision)',
            )

    for ax in (ax1, ax2):
        ax.set_xlim(wl_cmp_nm.min(), wl_cmp_nm.max())
        ax.axvline(center_air_nm, ls='--', color='dimgrey', zorder=0)
        ax.legend(fontsize=9)
        ax.set_xlabel('wavelength (nm)', fontsize=10)

    ax1.set_xlim(center_air_nm - 0.2, center_air_nm + 0.2)
    ax2.set_xlim(center_air_nm - 2, center_air_nm + 2)
    ax2.semilogy()
    fig.tight_layout()

    return fig


def run(save_path=None, *, num_f=60, num_mu=11, include_ffm=True,
        group_tolerance_ev=None, group_width_tolerance_ev=None,
        group_profile_rtol=1e-3):
    """Generate separate PPP intra-shell D-alpha and D-gamma figures."""
    if save_path:
        alpha_path = Path(save_path)
        gamma_path = alpha_path.with_name(
            f'{alpha_path.stem}_dgamma{alpha_path.suffix}'
        )
        # Save each expensive result before starting the next one, so a later
        # high-n failure cannot discard an already completed D-alpha figure.
        for n_u, path in zip((3, 5), (alpha_path, gamma_path)):
            fig = _make_figure(
                n_u, num_f=num_f, num_mu=num_mu, include_ffm=include_ffm,
                group_tolerance_ev=group_tolerance_ev,
                group_width_tolerance_ev=group_width_tolerance_ev,
                group_profile_rtol=group_profile_rtol)
            fig.savefig(path, dpi=200)
            print(f'saved figure to {path}')
            plt.close(fig)
    else:
        figures = [
            _make_figure(
                n_u, num_f=num_f, num_mu=num_mu, include_ffm=include_ffm,
                group_tolerance_ev=group_tolerance_ev,
                group_width_tolerance_ev=group_width_tolerance_ev,
                group_profile_rtol=group_profile_rtol)
            for n_u in (3, 5)
        ]
        plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('save_path', nargs='?')
    parser.add_argument('--num-f', type=int, default=60,
                        help='microfield quadrature nodes (default: 60)')
    parser.add_argument('--num-mu', type=int, default=11,
                        help='orientation quadrature nodes (default: 11)')
    parser.add_argument(
        '--quick', action='store_true',
        help='preview with num_f=12 and num_mu=5; overrides both node options')
    parser.add_argument(
        '--static-only', action='store_true',
        help='skip the PPP intra-shell FFM calculation')
    parser.add_argument(
        '--group-tolerance-ev', type=float,
        help='experimental complex-SDT frequency grouping tolerance [eV]')
    parser.add_argument(
        '--group-width-tolerance-ev', type=float,
        help='experimental complex-SDT width grouping tolerance [eV]')
    parser.add_argument(
        '--group-profile-rtol', type=float, default=1e-3,
        help='maximum grouped/static peak-relative error (default: 1e-3)')
    args = parser.parse_args()
    if args.quick:
        args.num_f, args.num_mu = 12, 5
    run(
        save_path=args.save_path,
        num_f=args.num_f,
        num_mu=args.num_mu,
        include_ffm=not args.static_only,
        group_tolerance_ev=args.group_tolerance_ev,
        group_width_tolerance_ev=args.group_width_tolerance_ev,
        group_profile_rtol=args.group_profile_rtol,
    )
