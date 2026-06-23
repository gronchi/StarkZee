"""
compare_griem_wings.py — Compare StarkZee (Griem mode) against analytical models
including the Griem α₁₂ Voigt profile.

StarkZee is run in "Griem mode":
  - Holtsmark (unscreened) microfield  →  use_screening=False
  - No fine structure                  →  fine_structure=False

This matches the assumptions underlying Griem's published tables (quasi-static
ions with the unscreened Holtsmark distribution and GBK electron impact at B=0).
The Griem analytical model uses the same α₁₂ HWHM but as a pure Voigt profile
(symmetric Lorentzian core + Gaussian Doppler), without the full quasi-static
Stark pattern.

Run directly::

    python examples/compare_griem_wings.py
"""

import time
import traceback

import numpy as np
import matplotlib.pyplot as plt

from starkzee.line_profile import LineProfile
from starkzee.convolutions import calculate_doppler_width_ev
import starkzee.models as models


def run():
    # ── parameters ────────────────────────────────────────────────────────────
    n_u, n_l       = 3, 2      # transition  (Hα)
    species        = 'H'       # emitting species: 'H', 'D', or 'T'
    Ne_m3          = 1e20      # electron density          [m⁻³]
    Te_ev          = 1        # electron temperature      [eV]  → Stark width
    Ti_ev          = 1        # ion temperature           [eV]  → Doppler width
    B              = 1.0       # magnetic field            [T]
    view_angle_deg = 90.0      # observation angle to B    [deg]
    # ──────────────────────────────────────────────────────────────────────────

    # Line center and adaptive grid width
    lp = LineProfile(n_u=n_u, n_l=n_l, B=B, Ne_m3=Ne_m3, Te_ev=Te_ev, Ti_ev=Ti_ev,
                     species=species, view_angle_deg=view_angle_deg)

    delta_E_D        = calculate_doppler_width_ev(lp.E0, Ti_ev, A_emitter=1)
    delta_lambda_D_nm = lp.E0_wavelength_nm * delta_E_D / lp.E0
    half_width_nm    = max(2.0, 4.0 * delta_lambda_D_nm)

    wl_sz_nm = np.linspace(lp.E0_wavelength_nm - half_width_nm,
                            lp.E0_wavelength_nm + half_width_nm, 1000)
    print('--------\ntimings:\n--------')

    # StarkZee — full mode first: its centroid (with fine structure) matches the
    # NIST air wavelength and is used to centre all comparison models.
    t0 = time.time()
    lp.compute_profile(wl_sz_nm, grid_type='wavelength_nm', num_f=20, num_mu=6,
                       fine_structure=True, use_screening=True)
    print(f'starkzee (full): {time.time() - t0:.3g} sec')

    center_air_nm = float(np.sum(lp.wavelengths_air_nm * lp.profile) / np.sum(lp.profile))

    # StarkZee — Griem mode: Holtsmark (unscreened), no fine structure (MV+Darwin off).
    # Wings fall as |Δλ|^{-5/2} (quasi-static Holtsmark), not |Δλ|^{-2} (Lorentzian).
    lp_griem = LineProfile(n_u=n_u, n_l=n_l, B=B, Ne_m3=Ne_m3, Te_ev=Te_ev, Ti_ev=Ti_ev,
                           species=species, view_angle_deg=view_angle_deg)
    t0 = time.time()
    lp_griem.compute_profile(wl_sz_nm, grid_type='wavelength_nm', num_f=50, num_mu=10,
                             fine_structure=False, use_screening=False)
    print(f'starkzee (Griem mode): {time.time() - t0:.3g} sec')

    wl_cmp_nm = np.linspace(center_air_nm - half_width_nm,
                             center_air_nm + half_width_nm, 1000)

    # ── figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[12, 5])
    fig.suptitle(
        f'$n_e = {Ne_m3:.2g}$ m$^{{-3}}$,  '
        f'$T_i = {Ti_ev:.3g}$ eV,  $T_e = {Te_ev:.3g}$ eV\n'
        f'$B = {B:.3g}$ T,  $\\theta = {view_angle_deg:.3g}$°  '
        f'— StarkZee Griem mode vs analytical models'
    )

    # ── comparison models ─────────────────────────────────────────────────────
    cmp_funcs = {
        'griem':        models.griem,
        'stehle':       models.stehle,
        'lomanowski':   models.lomanowski,
        'rosato':       models.rosato,
    }

    for name, func in cmp_funcs.items():
        try:
            t0 = time.time()
            profile = func(
                wl_cmp_nm, n_u, n_l, B, Ne_m3, Te_ev, Ti_ev,
                view_angle_deg=view_angle_deg, species=species,
            )
            print(f'{name}: {time.time() - t0:.3g} sec')
            ax1.plot(wl_cmp_nm, profile / profile.max(), label=name)
            ax2.plot(wl_cmp_nm, profile / profile.max(), label=name)
        except Exception as exc:
            print(f'{name} failed: {exc}')
            traceback.print_exc()

    # ── StarkZee curves — centroid-aligned onto wl_cmp_nm ────────────────────
    # Both SZ curves are plotted on wl_cmp_nm so all curves share the same
    # x-axis.  The Griem mode has no fine structure, so its energy centroid sits
    # at the gross-structure wavelength (~0.009 nm red of NIST); we shift it to
    # center_air_nm before interpolating so the line centers overlay.
    def _align(prof, wl_air):
        centroid = float(np.sum(wl_air * prof) / np.sum(prof))
        shift = center_air_nm - centroid
        return np.interp(wl_cmp_nm, wl_air + shift, prof, left=0.0, right=0.0)

    y_griem = _align(lp_griem.profile, lp_griem.wavelengths_air_nm)
    y_griem /= y_griem.max()
    for ax in (ax1, ax2):
        ax.plot(wl_cmp_nm, y_griem, 'k--', linewidth=2,
                label='starkzee (Griem mode: Holtsmark, no FS)')

    y_full = _align(lp.profile, lp.wavelengths_air_nm)
    y_full /= y_full.max()
    for ax in (ax1, ax2):
        ax.plot(wl_cmp_nm, y_full, 'k:', linewidth=2,
                label='starkzee (full: Hooper + FS)')

    # ── formatting ────────────────────────────────────────────────────────────
    for ax in (ax1, ax2):
        ax.set_xlim(wl_cmp_nm.min(), wl_cmp_nm.max())
        ax.axvline(center_air_nm, ls='--', color='dimgrey', zorder=0)
        ax.legend(fontsize=10)
        ax.set_xlabel('wavelength (nm)', fontsize=10)
        ax.set_yticklabels([])
        ax.set_yticks([])

    ax1.set_xlim(center_air_nm - 0.5, center_air_nm + 0.5)
    ax2.semilogy()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run()
