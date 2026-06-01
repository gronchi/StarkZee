"""
Rosato Stark-Zeeman-Doppler lineshape.

Based on pystark package (https://github.com/jsallcock/pystark).
"""

import os
import numpy as np
from scipy.constants import c as C, e as E, h as H
from scipy.io import netcdf_file

# ── Parameter grids (Rosato et al. database) ──────────────────────────────────
density_val     = np.array([1e13, 2.15e13, 4.64e13, 1e14, 2.15e14, 4.64e14,
                             1e15, 2.15e15, 4.64e15, 1e16])          # cm⁻³
temperature_val = np.array([0.316, 1., 3.16, 10., 31.6])             # eV
B_val           = np.array([0., 1., 2., 2.5, 3., 5.])                # T

# ── NetCDF database ────────────────────────────────────────────────────────────
_NC_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'rosato_tables.nc')
_NC_PATH = os.path.abspath(_NC_PATH)
_NC_DB   = None
try:
    if os.path.exists(_NC_PATH):
        _NC_DB = netcdf_file(_NC_PATH, 'r', mmap=True)
except Exception:
    pass

_FILE_CACHE = {}


def _load_file(filepath):
    if filepath not in _FILE_CACHE:
        data = np.loadtxt(filepath)
        _FILE_CACHE[filepath] = (data[:, 0], data[:, 1])
    return _FILE_CACHE[filepath]


def _set_bounds(dens_cm3, temp_ev, bfield):
    d = int(np.searchsorted(density_val, dens_cm3, side='right')) + 1
    t = int(np.searchsorted(temperature_val, temp_ev, side='right')) + 1
    b = int(np.searchsorted(B_val, bfield, side='right')) + 1
    if d <= 1 or d >= 11 or t <= 1 or t >= 6 or b <= 1 or b >= 7:
        raise ValueError(
            f"Rosato: parameters out of range "
            f"(Ne={dens_cm3:.2g} cm⁻³, T={temp_ev:.2g} eV, B={bfield:.2g} T)"
        )
    return d, t, b


def _set_name_file(n_upper, d_idx, t_idx, b_idx, angle_idx):
    names = []
    for d in range(2):
        for t in range(2):
            if d * t == 1:
                continue
            for b in range(2):
                if d_idx == 10:
                    names.append(f"ls10{t_idx-1+t}{b_idx-1+b}{angle_idx}.txt")
                else:
                    names.append(f"ls0{d_idx-1+d}{t_idx-1+t}{b_idx-1+b}{angle_idx}.txt")
    return "".join(names)


_LINE_NAMES = ['D_alpha', 'D_beta', 'D_gamma', 'D_delta', 'D_epsilon']


def _read_file(dir_path, names_str):
    names = [names_str[i:i+11] for i in range(0, 66, 11)]
    w_arr  = np.zeros((2, 2, 2, 1000))
    ls_arr = np.zeros((2, 2, 2, 1000))

    if _NC_DB is not None:
        try:
            subfolder = os.path.basename(os.path.normpath(dir_path))
            t_idx = _LINE_NAMES.index(subfolder)
            idx = 0
            for d in range(2):
                for t in range(2):
                    if d * t == 1:
                        break
                    for b in range(2):
                        fn = names[idx]; idx += 1
                        di = int(fn[2:4]) - 1
                        ti = int(fn[4])   - 1
                        bi = int(fn[5])   - 1
                        ai = int(fn[6])
                        w_arr [d, t, b, :] = _NC_DB.variables['detunings'] [t_idx, di, ti, bi, ai, :]
                        ls_arr[d, t, b, :] = _NC_DB.variables['intensities'][t_idx, di, ti, bi, ai, :]
            return w_arr, ls_arr
        except Exception:
            pass

    idx = 0
    for d in range(2):
        for t in range(2):
            if d * t == 1:
                break
            for b in range(2):
                w_arr[d, t, b, :], ls_arr[d, t, b, :] = _load_file(
                    os.path.join(dir_path, names[idx]))
                idx += 1
    return w_arr, ls_arr


def _ls_interpol(dens_cm3, temp_ev, bfield, wmax, npts,
                 w_arr, ls_arr, d_idx, t_idx, b_idx):
    det = np.linspace(-wmax, wmax, npts)
    arr2 = np.zeros((2, 2, 2, npts))
    arr3 = np.zeros((2, 2, npts))

    if b_idx != 2:  # B >= 1 T
        for d in range(2):
            for t in range(2):
                if d * t == 1:
                    continue
                for b in range(2):
                    sc = B_val[b_idx - 2 + b] / bfield
                    arr2[d, t, b] = np.interp(det * sc, w_arr[d, t, b], ls_arr[d, t, b],
                                               left=0., right=0.)
        denom = B_val[b_idx - 1] - B_val[b_idx - 2]
        w1 = ((bfield - B_val[b_idx - 2]) / denom) * (B_val[b_idx - 1] / bfield)
        w0 = ((B_val[b_idx - 1] - bfield) / denom) * (B_val[b_idx - 2] / bfield)
        for d in range(2):
            for t in range(2):
                if d * t == 1:
                    continue
                arr3[d, t] = w1 * arr2[d, t, 1] + w0 * arr2[d, t, 0]

    elif bfield == 0.:
        for d in range(2):
            for t in range(2):
                if d * t == 1:
                    continue
                arr3[d, t] = np.interp(det, w_arr[d, t, 0], ls_arr[d, t, 0],
                                        left=0., right=0.)
    else:  # 0 < B < 1 T
        for d in range(2):
            for t in range(2):
                if d * t == 1:
                    continue
                arr2[d, t, 0] = np.interp(det, w_arr[d, t, 0], ls_arr[d, t, 0],
                                            left=0., right=0.)
                arr2[d, t, 1] = np.interp(det * (B_val[1] / bfield), w_arr[d, t, 1],
                                            ls_arr[d, t, 1], left=0., right=0.)
        for d in range(2):
            for t in range(2):
                if d * t == 1:
                    continue
                arr3[d, t] = arr2[d, t, 1] + (1. - bfield) * arr2[d, t, 0]

    u = 3. * np.log10(dens_cm3  / density_val    [d_idx - 2])
    v = 2. * np.log10(temp_ev   / temperature_val[t_idx - 2])
    ls = u * arr3[1, 0] + v * arr3[0, 1] + (1. - u - v) * arr3[0, 0]

    # Zero out points where any interpolation corner leaves its table range
    wmax_safe = wmax
    for d, t in [(0, 0), (1, 0), (0, 1)]:
        if b_idx != 2:
            for b in range(2):
                sc = B_val[b_idx - 2 + b] / bfield
                wmax_safe = min(wmax_safe, float(w_arr[d, t, b, -1]) / sc)
        elif bfield == 0.:
            wmax_safe = min(wmax_safe, float(w_arr[d, t, 0, -1]))
        else:
            wmax_safe = min(wmax_safe, float(w_arr[d, t, 0, -1]))
            wmax_safe = min(wmax_safe, float(w_arr[d, t, 1, -1]) / (B_val[1] / bfield))
    ls[np.abs(det) > wmax_safe] = 0.

    return ls


def rosato(wavelengths_nm, n_u, n_l, B, Ne_m3, Te_ev, Ti_ev,
           view_angle_deg=90.0, species='H'):
    """Rosato Stark-Zeeman-Doppler profile using local tabulated data."""
    from starkzee.utils import species_to_ZA
    from starkzee.convolutions import apply_doppler_broadening
    from scipy.signal import fftconvolve

    _, A = species_to_ZA(species)
    # Isotope fudge: Rosato tables are for D; doubling T approximates halving mass (H)
    cc = 2 if A == 1 else 1
    dens_cm3 = Ne_m3 * 1e-6
    temp_eff = cc * Ti_ev

    d_idx, t_idx, b_idx = _set_bounds(dens_cm3, temp_eff, B)

    # Detuning range: cover the full wavelength grid plus 10 %
    lambda0_m  = np.mean(wavelengths_nm) * 1e-9
    freq_ctr   = C / lambda0_m
    max_dfreq  = max(abs(C / (wavelengths_nm.min() * 1e-9) - freq_ctr),
                     abs(C / (wavelengths_nm.max() * 1e-9) - freq_ctr))
    wmax_ev    = H * max_dfreq / E * 1.1
    npts       = max(2001, len(wavelengths_nm) * 2)

    # Balmer lines only; n_l == 2 assumed (Rosato is Balmer-only)
    line_name  = _LINE_NAMES[n_u - 3]
    data_dir   = os.path.join(os.path.dirname(_NC_PATH), '..', 'rosato_database',
                              line_name)  # not used when NC_DB is available

    lss = np.zeros((npts, 2))
    for angle_idx in range(2):
        names_str = _set_name_file(n_u, d_idx, t_idx, b_idx, angle_idx)
        w_arr, ls_arr = _read_file(data_dir, names_str)
        lss[:, angle_idx] = _ls_interpol(dens_cm3, temp_eff, B, wmax_ev, npts,
                                          w_arr, ls_arr, d_idx, t_idx, b_idx)

    theta = np.deg2rad(view_angle_deg)
    ls_sz = lss[:, 0] * np.sin(theta)**2 + lss[:, 1] * np.cos(theta)**2

    # Convert detuning axis (eV) to absolute frequency [Hz] then to wavelength [nm]
    det_ev     = np.linspace(-wmax_ev, wmax_ev, npts)
    freqs      = E * det_ev / H + freq_ctr           # Hz
    wl_nm      = C / freqs * 1e9                     # nm (non-uniform)

    # Convert intensity from 1/eV to 1/m (wavelength)
    ls_sz_hz = ls_sz * E / H                          # 1/Hz
    # Normalise in frequency space
    ls_sz_hz /= np.trapz(ls_sz_hz, freqs)

    # Sort by ascending wavelength for interp
    order     = np.argsort(wl_nm)
    wl_sorted = wl_nm[order]
    ls_sorted = ls_sz_hz[order]

    # Interpolate onto uniform output wavelength grid (zero outside Rosato range)
    rosato_support = ls_sz > 0
    wl_support_sorted = wl_nm[order][rosato_support[order]]
    wl_min = wl_support_sorted.min() if wl_support_sorted.size else wl_nm.min()
    wl_max = wl_support_sorted.max() if wl_support_sorted.size else wl_nm.max()

    # dλ = -λ² dν / c  →  I(λ) = I(ν) × c/λ²
    ls_wl_sorted = ls_sorted * C / (wl_sorted * 1e-9)**2
    ls_out = np.interp(wavelengths_nm, wl_sorted, ls_wl_sorted, left=0., right=0.)
    ls_out[(wavelengths_nm < wl_min) | (wavelengths_nm > wl_max)] = 0.

    # Doppler broadening
    support_mask = ls_out > 0
    from scipy.signal import fftconvolve as _fftconv
    from starkzee.models.analytical import _fwhm_doppler_nm, _SIGMA2FWHM
    fwhm_d = _fwhm_doppler_nm(np.mean(wavelengths_nm), Ti_ev, A)
    sigma  = fwhm_d / _SIGMA2FWHM
    d      = wavelengths_nm - np.mean(wavelengths_nm)
    kernel = np.exp(-0.5 * (d / sigma)**2)
    ls_out = _fftconv(ls_out, kernel / kernel.sum(), 'same')
    ls_out[~support_mask] = 0.

    # Area-normalise in wavelength [m]
    area = np.trapz(ls_out, wavelengths_nm * 1e-9)
    if area > 0:
        ls_out /= area

    return ls_out
