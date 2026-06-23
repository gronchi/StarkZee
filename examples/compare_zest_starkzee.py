"""
model_comparison.py — Compare StarkZee's static profile against analytical 
and tabulated models.

Edit the parameters block inside run() to change plasma conditions.
All models receive the same (Ti, Te, Ne, B, angle), so differences are
purely due to the underlying physics model.

Run directly::

    python examples/model_comparison.py
"""

import time
import traceback

import numpy as np
import matplotlib.pyplot as plt

from starkzee.line_profile import LineProfile
from starkzee.convolutions import calculate_doppler_width_ev
import starkzee.models as models

import sys
sys.path.insert(0, '.')
import zest

def zest_wrapper(wl_cmp_nm, n_u, n_l, B, Ne_m3, Te_ev, Ti_ev, view_angle_deg, species):
    prof = zest.ZESTProfile(n_init=n_u, n_final=n_l, Z_c=1)
    Te_K = Te_ev * 11604.525
    Ti_K = Ti_ev * 11604.525
    center_nm = np.mean(wl_cmp_nm)
    d_lambda_m = (wl_cmp_nm - center_nm) * 1e-9
    lambda0_m = center_nm * 1e-9
    d_omega = - (2.0 * np.pi * zest.C_LIGHT / lambda0_m**2) * d_lambda_m
    omega_grid = prof.omega_0 + d_omega
    I0, Ip, Im = prof.get_profiles(
        omega_grid, ne=Ne_m3, te=Te_K, ti=Ti_K, B=B,
        Z_bar=1, charged=True, model='gbk',
        N_F=100, N_mu=6, include_dynamics=False,
        omit_doppler=False, microfield_model='hooper',
    )
    theta_rad = np.radians(view_angle_deg)
    sin2 = np.sin(theta_rad)**2
    cos2 = np.cos(theta_rad)**2
    I_tot = I0 * sin2 + (Ip + Im) * (1.0 + cos2) / 2.0
    return I_tot


def run():
    # ── parameters ────────────────────────────────────────────────────────────
    n_u, n_l       = 3, 2      # transition  (Hα)
    species        = 'H'       # emitting species: 'H', 'D', or 'T'
    Ne_m3          = 1e20      # electron density          [m⁻³]
    Te_ev          = .5       # electron temperature      [eV]  → Stark width
    Ti_ev          = .5       # ion temperature           [eV]  → Doppler width
    B              = 3.0       # magnetic field            [T]
    view_angle_deg = 90.0      # observation angle to B    [deg]
    # ──────────────────────────────────────────────────────────────────────────

    # Line center and adaptive grid width
    lp = LineProfile(n_u=n_u, n_l=n_l, B=B, Ne_m3=Ne_m3, Te_ev=Te_ev, Ti_ev=Ti_ev,
                     species=species, view_angle_deg=view_angle_deg)

    delta_E_D        = calculate_doppler_width_ev(lp.E0, Ti_ev, A_emitter=1)
    delta_lambda_D_nm = lp.E0_wavelength_nm * delta_E_D / lp.E0
    half_width_nm    = max(2.0, 4.0 * delta_lambda_D_nm)

    # StarkZee computes in vacuum nm, on a grid centered on the gross-structure
    # Rydberg line center.  Compute it up front so the comparison models can be
    # referenced to its actual line center (next).
    wl_sz_nm = np.linspace(lp.E0_wavelength_nm - half_width_nm,
                            lp.E0_wavelength_nm + half_width_nm, 1001)
    print('--------\ntimings:\n--------')
    t0 = time.time()
    lp.compute_profile(wl_sz_nm, grid_type='wavelength_nm',  num_f=20, num_mu=6, fine_structure=True)
    print(f'starkzee (PPPB): {time.time() - t0:.3g} sec')

    # StarkZee with ZEST-equivalent processing: spin-free (no fine structure),
    # no quadratic Zeeman, ZEST κ_m/ω_p GBK electron width used as an operator
    # diagonal (per-Stark-Zeeman-state ⟨r²⟩ scaling, c_k = 0), and the impact
    # approximation (frequency-independent width).  Same grid as the default run.
    lp_zest = LineProfile(n_u=n_u, n_l=n_l, B=B, Ne_m3=Ne_m3, Te_ev=Te_ev, Ti_ev=Ti_ev,
                          species=species, view_angle_deg=view_angle_deg)
    t0 = time.time()
    lp_zest.compute_profile(wl_sz_nm, grid_type='wavelength_nm', num_f=20, num_mu=6,
                            fine_structure=False, quadratic_zeeman=False,
                            electron_model='zest', electron_operator=True,
                            frequency_dependent_width=False)
    print(f'starkzee (ZEST): {time.time() - t0:.3g} sec')

    # Physical line center = intensity-weighted centroid of the StarkZee profile.
    # The StarkZee profile includes Dirac fine structure (and the full Stark-Zeeman
    # asymmetry), which shifts the line ~0.01 nm to the blue of the gross-structure
    # value lp.E0 — onto the physical (NIST) wavelength.  The comparison models have
    # no fine structure and are symmetric about their grid mean, so centering their
    # grid and the vertical guide line on this centroid makes every peak overlay;
    # otherwise StarkZee appears offset from the others and from the vertical line.
    center_air_nm = lp.E0_wavelength_air_nm

    # Comparison models compute in air nm, centered on the same physical line center.
    wl_cmp_nm = np.linspace(center_air_nm - half_width_nm,
                             center_air_nm + half_width_nm, 1000)

    # ── figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[12, 5], sharex=True)
    fig.suptitle(
        f'$n_e = {Ne_m3:.2g}$ m$^{{-3}}$,  '
        f'$T_i = {Ti_ev:.3g}$ eV,  $T_e = {Te_ev:.3g}$ eV\n'
        f'$B = {B:.3g}$ T,  $\\theta = {view_angle_deg:.3g}$°'
    )

    # ── comparison models ─────────────────────────────────────────────
    cmp_funcs = {
        'zest':         zest_wrapper,
    }

    for name, func in cmp_funcs.items():
        try:
            t0 = time.time()
            profile = func(
                wl_cmp_nm, n_u, n_l, B, Ne_m3, Te_ev, Ti_ev,
                view_angle_deg=view_angle_deg, species=species,
            )
            print(f'{name}: {time.time() - t0:.3g} sec')
            if name == 'zest':
                ax1.plot(wl_cmp_nm, profile / profile.max(), ':', label=name)
                ax2.plot(wl_cmp_nm, profile / profile.max(), ':', label=name)
            else:
                ax1.plot(wl_cmp_nm, profile / profile.max(), label=name)
                ax2.plot(wl_cmp_nm, profile / profile.max(), label=name)
        except Exception as exc:
            print(f'{name} failed: {exc}')
            traceback.print_exc()

    # ── StarkZee static profiles (computed above) ─────────────────────────────
    # Default (PPPB) and the ZEST-equivalent processing, both normalized to peak.
    # Each profile is shifted by its own intensity-weighted centroid offset so
    # that all curves appear centered at center_air_nm regardless of whether
    # fine structure is included.
    y_pppb = lp.profile / lp.profile.max()
    y_zest = lp_zest.profile / lp_zest.profile.max()
    c_pppb = float(np.sum(lp.wavelengths_air_nm * lp.profile) / np.sum(lp.profile))
    c_zest_sz = float(np.sum(lp_zest.wavelengths_air_nm * lp_zest.profile) / np.sum(lp_zest.profile))
    for ax in (ax1, ax2):
        ax.plot(lp.wavelengths_air_nm + (center_air_nm - c_pppb), y_pppb,
                'k--', linewidth=2, label='starkzee (PPPB)')
        ax.plot(lp_zest.wavelengths_air_nm + (center_air_nm - c_zest_sz), y_zest,
                color='#d95f02', linewidth=2, label='starkzee (ZEST)')

    # ── formatting ────────────────────────────────────────────────────────────
    for ax in (ax1, ax2):
        ax.set_xlim(wl_cmp_nm.min(), wl_cmp_nm.max())
        ax.axvline(center_air_nm, ls='--', color='dimgrey', zorder=0)
        ax.legend(fontsize=10)
        ax.set_xlabel('wavelength (nm)', fontsize=10)
        ax.set_yticklabels([])
        ax.set_yticks([])

    ax2.semilogy()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run()
