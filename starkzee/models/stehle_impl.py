"""
Stehle (MMM) Stark-Zeeman-Doppler lineshape.

Based on pystark package (https://github.com/jsallcock/pystark).
"""

import os
import numpy as np
from scipy.constants import c as C, e as E, k as K, m_e as M_E
from scipy.io import netcdf_file
from scipy.signal import fftconvolve

_NC_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data', 'stehle_tables.nc'))
_NC = None
try:
    _NC = netcdf_file(_NC_PATH, 'r', mmap=True)
except Exception:
    pass


# ── Fortran FINTRP: three-point hyperbolic/quadratic interpolation ─────────────
def _fintrp(x1, x2, x3, y1, y2, y3, x):
    if x == x2:
        return y2
    a12 = x1 - x2;  a22 = x1 - x3
    v1  = y1 - y2;  v2  = y1 - y3
    if ((y1 < y2 < y3) or (y1 > y2 > y3)):
        deter = v1 * a22 - v2 * a12
        if abs(deter) < 1e-40:
            return y1 + (x - x1) * (y3 - y1) / (x3 - x1)
        a21 = x1 * y1
        a11 = a21 - x2 * y2
        a21 = a21 - x3 * y3
        c   = (a22 * a11 - a12 * a21) / deter
        a   = (-v2 * a11 + v1 * a21) / deter
        b   = (y1 - a) * (x1 - c)
        return a + b / (x - c)
    else:
        x1c = x1 * x1
        a11 = x1c - x2 * x2
        a21 = x1c - x3 * x3
        deter = a11 * a22 - a12 * a21
        if abs(deter) < 1e-40:
            raise ValueError('FINTRP: degenerate inputs')
        a = (a22 * v1 - a12 * v2) / deter
        b = (-a21 * v1 + a11 * v2) / deter
        return (a * x + b) * x + (y1 - a * x1c - b * x1)


def _compute_stehle_stark(n_u, n_l, Ne_m3, Te_ev, wl_centre_m, freq_axis):
    """Pure Stark profile on *freq_axis* [Hz], area-normalised to 1.

    Direct port of pystark's make_stehle; reads from the local stehle_tables.nc.
    Returns profile in 1/Hz.
    """
    if _NC is None:
        raise FileNotFoundError(f"stehle_tables.nc not found at {_NC_PATH}")

    temp_k  = Te_ev * E / K
    dens_cm = Ne_m3 * 1e-6
    prefix  = f'n_{n_u}_{n_l}_'

    tab_temp_k   = np.array(_NC.variables[prefix + 'tempe'].data)
    num_tab_dens = int(np.asarray(_NC.variables[prefix + 'id_max'].data).item())
    fainom       = float(np.asarray(_NC.variables[prefix + 'fainom'].data).item())
    tab_dens_cm  = np.array(_NC.variables[prefix + 'dense'].data)
    pr0          = np.array(_NC.variables[prefix + 'pr0'].data)
    jtot         = np.array(_NC.variables[prefix + 'jtot'].data, dtype=int)
    dom          = np.array(_NC.variables[prefix + 'dom'].data)
    o1line       = np.array(_NC.variables[prefix + 'o1line'].data)
    o1lines      = np.array(_NC.variables[prefix + 'o1lines'].data)

    # Nudge density away from exact table nodes
    if np.abs(dens_cm - tab_dens_cm[0]) / dens_cm <= 1e-3:
        dens_cm = tab_dens_cm[0] * 1.001
    for id_ in range(1, num_tab_dens + 1):
        if np.abs(dens_cm - tab_dens_cm[id_]) / dens_cm <= 1e-3:
            dens_cm = tab_dens_cm[id_] * 0.999

    if dens_cm >= 2.0 * tab_dens_cm[num_tab_dens]:
        raise ValueError(f'Stehle: density {dens_cm:.3e} cm⁻³ exceeds table max')
    if dens_cm <= tab_dens_cm[0]:
        raise ValueError(f'Stehle: density {dens_cm:.3e} cm⁻³ below table min')
    if temp_k >= tab_temp_k[9]:
        raise ValueError(f'Stehle: temperature {temp_k:.1f} K exceeds table max')
    if temp_k <= tab_temp_k[0]:
        raise ValueError(f'Stehle: temperature {temp_k:.1f} K below table min')

    normal_hf = 1.25e-9 * (dens_cm ** (2. / 3.))          # normal Holtsmark field [ues]

    PR0_exp = 0.0898 * (dens_cm ** (1. / 6.)) / np.sqrt(temp_k)
    if PR0_exp > 1.:
        raise ValueError('Stehle: plasma too strongly correlated (r₀/λ_D > 1)')

    wl_centre_angst = wl_centre_m * 1e10
    c_angst         = C * 1e10
    angular_freq_0  = 2 * np.pi * c_angst / wl_centre_angst
    otrans          = -2 * np.pi * c_angst / wl_centre_angst ** 2
    olines          = o1lines / abs(otrans)

    # Build common detuning grid from all tabulated detunings
    dom0 = np.zeros(10000)
    inc  = 0
    for id_ in range(num_tab_dens + 1):
        for j in range(10):
            for i in range(1, jtot[id_, j]):
                inc += 1
                dom0[inc] = dom[id_, j, i]
    npik = np.count_nonzero(dom) + 1
    dom0[:npik] = np.sort(dom0[:npik])

    domm = np.zeros(100000)
    inc  = 0
    domm[0] = 0.0
    for i in range(1, npik):
        dif = dom0[i] - dom0[i - 1]
        if dif <= 1e-6 or dif / abs(dom0[i]) <= 0.1:
            continue
        inc += 1
        domm[inc] = dom0[i]
    jdom = inc + 1

    tprofs = np.zeros((30, 10, 10000))
    for id_ in range(num_tab_dens):
        for j in range(10):
            if pr0[id_, j] > 1.0:
                continue
            tprofs[id_, j, 0] = olines[id_, j, 0]
            if jtot[id_, j] == 0:
                continue
            for i in range(1, jdom + 1):
                domeg  = domm[i]
                ij_max = jtot[id_, j]
                found  = False
                for ij in range(1, ij_max - 1):
                    if (domeg - dom[id_, j, ij]) * (domeg - dom[id_, j, ij - 1]) <= 0.:
                        tprofs[id_, j, i] = _fintrp(
                            dom[id_, j, ij-1], dom[id_, j, ij], dom[id_, j, ij+1],
                            olines[id_, j, ij-1], olines[id_, j, ij], olines[id_, j, ij+1],
                            domeg)
                        found = True
                        break
                if not found:
                    if (domeg - dom[id_, j, ij_max-2]) * (domeg - dom[id_, j, ij_max-1]) <= 0.:
                        tprofs[id_, j, i] = _fintrp(
                            dom[id_, j, ij_max-3], dom[id_, j, ij_max-2], dom[id_, j, ij_max-1],
                            olines[id_, j, ij_max-3], olines[id_, j, ij_max-2], olines[id_, j, ij_max-1],
                            domeg)
                    elif domeg > dom[id_, j, ij_max]:
                        tprofs[id_, j, i] = fainom / (domeg ** 2.5)

    # Find density bounding interval
    id1, id2, dense1, dense2 = 0, 1, tab_dens_cm[0], tab_dens_cm[1]
    for id_ in range(num_tab_dens):
        if (dens_cm - tab_dens_cm[id_]) * (dens_cm - tab_dens_cm[id_ + 1]) <= 0.:
            id1, id2 = id_, id_ + 1
            dense1, dense2 = tab_dens_cm[id_], tab_dens_cm[id_ + 1]
            break
    if dens_cm >= tab_dens_cm[num_tab_dens]:
        id1, id2 = num_tab_dens - 1, num_tab_dens
        dense1, dense2 = tab_dens_cm[id1], tab_dens_cm[id2]

    # Find temperature bounding interval
    it1, it2, tempe1, tempe2 = 0, 1, tab_temp_k[0], tab_temp_k[1]
    for it in range(10):
        if (temp_k - tab_temp_k[it]) * (temp_k - tab_temp_k[it + 1]) <= 0.:
            it1, it2 = it, it + 1
            tempe1, tempe2 = tab_temp_k[it], tab_temp_k[it + 1]
            break

    # Interpolate in temperature
    uprofs = np.zeros((30, 10000))
    for id_ in range(id1, id2 + 1):
        for i in range(jdom):
            uprofs[id_, i] = (tprofs[id_, it1, i]
                              + (temp_k - tempe1)
                              * (tprofs[id_, it2, i] - tprofs[id_, it1, i])
                              / (tempe2 - tempe1))

    # Interpolate in density and convert units
    delta_nu  = np.zeros(jdom)
    wprofs_nu = np.zeros(jdom)
    for i in range(jdom):
        wprofs = (uprofs[id1, i]
                  + (dens_cm - dense1) * (uprofs[id2, i] - uprofs[id1, i])
                  / (dense2 - dense1))
        delta_omega   = domm[i] * normal_hf
        delta_nu[i]   = delta_omega / (2 * np.pi)
        wprofs_nu[i]  = (wprofs / normal_hf) * (2. * np.pi)

    # Build symmetric profile in frequency space
    delta_nu2  = np.concatenate((-delta_nu[::-1],  delta_nu))
    wprofs_nu2 = np.concatenate((wprofs_nu[::-1], wprofs_nu))

    freq_centre = C / wl_centre_m
    ls_sd = np.interp(freq_axis, delta_nu2 + freq_centre, wprofs_nu2, left=0., right=0.)
    return ls_sd


def _zeeman_split_freq(freq_axis, ls, B, view_angle_deg):
    if B == 0.:
        return ls
    theta      = np.deg2rad(view_angle_deg)
    rel_pi     = np.sin(theta)**2 / 2.
    rel_sigma  = (1. + np.cos(theta)**2) / 4.
    freq_shift = E / (4. * np.pi * M_E) * B
    ls_sm = rel_sigma * np.interp(freq_axis + freq_shift, freq_axis, ls, left=0., right=0.)
    ls_sp = rel_sigma * np.interp(freq_axis - freq_shift, freq_axis, ls, left=0., right=0.)
    return rel_pi * ls + ls_sm + ls_sp


def stehle(wavelengths_nm, n_u, n_l, B, Ne_m3, Te_ev, Ti_ev,
           view_angle_deg=90.0, species='H'):
    """Stehle (MMM) Stark-Zeeman-Doppler profile using local tabulated data."""
    from starkzee.utils import species_to_ZA
    from starkzee.models.analytical import _fwhm_doppler_nm, _SIGMA2FWHM

    Z, A = species_to_ZA(species)

    # Line centre from the grid midpoint — matches pystark when the caller passes a grid
    # centred on the physical line centre (e.g. lp.E0_wavelength_air_nm).
    lambda0_nm  = np.mean(wavelengths_nm)
    wl_centre_m = lambda0_nm * 1e-9
    freq_centre = C / wl_centre_m

    # freq_axis: just wide enough to cover the wavelength grid (pystark adds ~6% margin).
    max_dfreq = max(abs(C / (wavelengths_nm.min() * 1e-9) - freq_centre),
                    abs(C / (wavelengths_nm.max() * 1e-9) - freq_centre))
    half_hz   = max_dfreq * 1.06
    npts      = 2001
    freq_axis = np.linspace(freq_centre - half_hz, freq_centre + half_hz, npts)

    # Pure Stark profile in frequency space
    ls_s = _compute_stehle_stark(n_u, n_l, Ne_m3, Te_ev, wl_centre_m, freq_axis)

    # Doppler kernel on freq_axis_conv (500 extra points each side, like pystark).
    # fftconvolve(ls_s, ls_d, 'same') returns len(ls_s) points; the extra width in
    # ls_d prevents the convolution from picking up zero-padding artifacts at the edges.
    extra     = 1000
    dfreq     = (freq_axis[-1] - freq_axis[0]) / (len(freq_axis) - 1)
    freq_conv = np.linspace(freq_axis[0]  - extra // 2 * dfreq,
                             freq_axis[-1] + extra // 2 * dfreq,
                             len(freq_axis) + extra)
    fwhm_d_hz = (_fwhm_doppler_nm(lambda0_nm, Ti_ev, A) * 1e-9) * C / wl_centre_m**2
    sigma_hz  = fwhm_d_hz / _SIGMA2FWHM
    ls_d      = np.exp(-0.5 * ((freq_conv - freq_centre) / sigma_hz)**2)
    ls_d     /= ls_d.sum()

    ls_sd = fftconvolve(ls_s, ls_d, 'same')   # returns len(ls_s) = npts points

    # Zeeman splitting in frequency space
    ls_szd = _zeeman_split_freq(freq_axis, ls_sd, B, view_angle_deg)

    # Convert frequency → wavelength.
    # I(ν) [1/Hz] → I(λ) [1/m]: I(λ) = I(ν) · |dν/dλ| = I(ν) · c/λ²
    wl_from_freq_nm = C / freq_axis * 1e9            # nm, non-uniform & reversed
    wl_from_freq_m  = C / freq_axis                  # m
    ls_wl = ls_szd * C / wl_from_freq_m**2           # [1/m]

    # Sort by ascending wavelength for np.interp
    order     = np.argsort(wl_from_freq_nm)
    wl_sorted = wl_from_freq_nm[order]
    ls_sorted = ls_wl[order]

    ls_out = np.interp(wavelengths_nm, wl_sorted, ls_sorted, left=0., right=0.)

    # Area-normalise
    area = np.trapz(ls_out, wavelengths_nm * 1e-9)
    if area > 0:
        ls_out /= area
    return ls_out
