"""
Analytical Stark-Zeeman Doppler lineshape models.

Implements lomanowski, stehle_param, and voigt directly.

All functions share the same signature::

    profile = <model>(wavelengths_nm, n_u, n_l, B, Ne_m3, Te_ev, Ti_ev,
                      view_angle_deg=90.0, species='H')

and return an area-normalized lineshape (1/m) on *wavelengths_nm*.

Based on pystark package (https://github.com/jsallcock/pystark).
"""

import numpy as np
from scipy.constants import c as C, e as E, m_e as M_E, atomic_mass as _U
from scipy.special import wofz

from scipy.signal import fftconvolve

from starkzee.utils import species_to_ZA

try:
    from numpy import trapezoid as trapz
except ImportError:
    from numpy import trapz

_SIGMA2FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))

# NIST air wavelengths [nm] — intensity-weighted mean of fine-structure components.
# Source: NIST Atomic Spectra Database (ASD), same values used by pystark.
# Key: (species_initial, n_u, n_l)
_NIST_CENTER_AIR_NM = {
    # H Balmer
    ('H', 3, 2): 656.279,
    ('H', 4, 2): 486.135,
    ('H', 5, 2): 434.047,
    ('H', 6, 2): 410.173,
    ('H', 7, 2): 397.007,
    ('H', 8, 2): 388.905,
    ('H', 9, 2): 383.539,
    # H Paschen
    ('H', 4, 3): 1875.10,
    ('H', 5, 3): 1281.81,
    ('H', 6, 3): 1093.80,
    ('H', 7, 3): 1004.94,
    ('H', 8, 3):  954.62,
    ('H', 9, 3):  922.90,
    # D Balmer
    # (3,2): corrected from pystark's 656.107 to NIST ASD's directly published
    # blended-line Ritz wavelength (physics.nist.gov Lines query, D I, ~656 nm;
    # NIST has no separate "Observed" entry for this blended line, only Ritz,
    # unc. 0.0009 nm). 656.107 does not match any NIST-published value for D-alpha.
    ('D', 3, 2): 656.1012,
    ('D', 4, 2): 486.000,
    ('D', 5, 2): 433.928,
    ('D', 6, 2): 410.062,
    ('D', 7, 2): 396.899,
    # T Balmer
    ('T', 3, 2): 656.042,
    ('T', 4, 2): 485.970,
}


def _nist_center_air_nm(n_u, n_l, species, wavelengths_nm):
    """Return the NIST air line center [nm], falling back to the grid midpoint."""
    key = (species.strip().upper()[0], n_u, n_l)
    if key in _NIST_CENTER_AIR_NM:
        return _NIST_CENTER_AIR_NM[key]
    return float(np.mean(wavelengths_nm))


# Emitter masses for Doppler broadening, keyed by mass number A.
# These are the standard atomic weights used by pystark's get_species_mass
# (atom mass, i.e. nucleus + electrons), NOT bare nuclear masses — keeping
# them identical guarantees the Doppler width matches the reference exactly.
_DOPPLER_MASS_KG = {
    1: 1.00794        * _U,   # H
    2: 2.01410178     * _U,   # D
    3: 3.01604928199  * _U,   # T
}

# Lomanowski (2015) fitting coefficients [a_ij, b_ij, c_ij]
# delta_lambda_12 = c * ne^a / Te^b  [nm]
_LOMAN_COEFFS = {
    '32': (0.7665, 0.064, 3.710e-18),
    '42': (0.7803, 0.050, 8.425e-18),
    '52': (0.6796, 0.030, 1.310e-15),
    '62': (0.7149, 0.028, 3.954e-16),
    '72': (0.7120, 0.029, 6.258e-16),
    '82': (0.7159, 0.032, 7.378e-16),
    '92': (0.7177, 0.033, 8.947e-16),
    '43': (0.7449, 0.045, 1.330e-16),
    '53': (0.7356, 0.044, 6.640e-16),
    '63': (0.7118, 0.016, 2.481e-15),
    '73': (0.7137, 0.029, 3.270e-15),
    '83': (0.7133, 0.032, 4.343e-15),
    '93': (0.7165, 0.033, 5.588e-15),
}

# Griem alpha_12 coefficients for Stark FWHM (voigt model)
_GRIEM_A12 = {3: 0.05, 4: 0.08, 5: 0.0922747860122222,
              6: 0.17, 7: 0.22, 8: 0.28, 9: 0.36, 10: 0.46}


def _loman_coeffs(n_u, n_l):
    key = f'{n_u}{n_l}'
    if key not in _LOMAN_COEFFS:
        raise ValueError(
            f"Lomanowski coefficients not available for {n_u}→{n_l}. "
            f"Supported transitions: {sorted(_LOMAN_COEFFS.keys())}"
        )
    return _LOMAN_COEFFS[key]


def _fwhm_stark_loman_nm(n_u, n_l, Ne_m3, Te_ev):
    a, b, c = _loman_coeffs(n_u, n_l)
    return c * (Ne_m3 ** a) / (Te_ev ** b)


def _fwhm_stark_griem_nm(n_u, Ne_m3):
    if n_u not in _GRIEM_A12:
        raise ValueError(f"Griem α₁₂ not available for n_u={n_u}. Supported: {sorted(_GRIEM_A12)}")
    alpha12 = _GRIEM_A12[n_u]
    return 0.53860867250797 * alpha12 * (Ne_m3 * 1e-20) ** (2.0 / 3.0)


def _fwhm_doppler_nm(lambda0_nm, Ti_ev, A):
    mass_kg = _DOPPLER_MASS_KG.get(A, A * _U)
    sigma_nm = lambda0_nm * np.sqrt(E * Ti_ev / (mass_kg * C**2))
    return _SIGMA2FWHM * sigma_nm


def _build_freq_axis(wavelengths_nm, freq_center, npts=2001, margin=1.05):
    """Uniform internal frequency axis covering the output grid (+`margin`).

    Mirrors pystark's freq_axis: half-width = (max detuning across the grid) × 1.05,
    with pystark's default of 2001 points.
    """
    max_df = max(abs(C / (wavelengths_nm.min() * 1e-9) - freq_center),
                 abs(C / (wavelengths_nm.max() * 1e-9) - freq_center))
    half = max_df * margin
    return np.linspace(freq_center - half, freq_center + half, npts)


def _zeeman_split_freq(freq_axis, profile, B, view_angle_deg):
    """First-order Zeeman splitting in frequency space."""
    if B == 0.0:
        return profile
    theta = np.deg2rad(view_angle_deg)
    rel_pi    = np.sin(theta)**2 / 2.0
    rel_sigma = (1.0 + np.cos(theta)**2) / 4.0
    shift = E / (4.0 * np.pi * M_E) * B
    sigma_minus = rel_sigma * np.interp(freq_axis + shift, freq_axis, profile, left=0., right=0.)
    sigma_plus  = rel_sigma * np.interp(freq_axis - shift, freq_axis, profile, left=0., right=0.)
    return rel_pi * profile + sigma_minus + sigma_plus


def _freq_to_wl_norm(freq_axis, profile_freq, wavelengths_nm):
    """Convert a frequency-space profile to wavelength, interpolate onto the output
    grid and area-normalize to 1/m.  I(λ) = I(ν) · c/λ²."""
    wlf_nm = C / freq_axis * 1e9
    ls_wl  = profile_freq * C / (C / freq_axis)**2
    order  = np.argsort(wlf_nm)
    out    = np.interp(wavelengths_nm, wlf_nm[order], ls_wl[order], left=0., right=0.)
    area   = trapz(out, wavelengths_nm * 1e-9)
    return out / area if area > 0 else out


def lomanowski(wavelengths_nm, n_u, n_l, B, Ne_m3, Te_ev, Ti_ev,
               view_angle_deg=90.0, species='H'):
    """Lomanowski pseudo-Voigt Stark-Zeeman-Doppler profile."""
    _, A = species_to_ZA(species)
    lambda0_nm = _nist_center_air_nm(n_u, n_l, species, wavelengths_nm)

    fwhm_l = _fwhm_stark_loman_nm(n_u, n_l, Ne_m3, Te_ev)
    fwhm_g = _fwhm_doppler_nm(lambda0_nm, Ti_ev, A)

    # Pseudo-Voigt FWHM combining formula (Lomanowski 2015)
    if fwhm_g <= fwhm_l:
        r = fwhm_g / fwhm_l
        cf = [1., 0., 0.57575, 0.37902, -0.42519, -0.31525, 0.31718]
        fwhm = fwhm_l * sum(ci * r**i for i, ci in enumerate(cf))
    else:
        r = fwhm_l / fwhm_g
        cf = [1., 0.15882, 1.04388, -1.38281, 0.46251, 0.82325, -0.58026]
        fwhm = fwhm_g * sum(ci * r**i for i, ci in enumerate(cf))

    # Lorentzian weight
    rl = fwhm_l / fwhm
    if rl < 0.01:
        eta_l, eta_g = 0.0, 1.0
    elif rl > 0.999:
        eta_l, eta_g = 1.0, 0.0
    else:
        lc = [5.14820e-04, 1.38821e+00, -9.60424e-02,
              -3.83995e-02, -7.40042e-03, -5.47626e-04]
        eta_l = np.exp(sum(lci * np.log(rl)**i for i, lci in enumerate(lc)))
        eta_g = 1.0 - eta_l

    # Build the pseudo-Voigt in frequency space (pystark.make_lomanowski): the FWHM
    # combination and Lorentzian weight above are unitless/in nm, but the profile itself
    # must be evaluated on the frequency grid to match the reference.
    freq_center = C / (lambda0_nm * 1e-9)
    fwhm_hz = fwhm * 1e-9 * C / (lambda0_nm * 1e-9)**2
    freq_axis = _build_freq_axis(wavelengths_nm, freq_center)
    df = freq_axis - freq_center

    _stark_norm = 2.641279471021934
    hwhm = fwhm_hz / 2.0
    ls_l = (hwhm**1.5 / _stark_norm) / (np.abs(df)**2.5 + hwhm**2.5)

    sigma = fwhm_hz / _SIGMA2FWHM
    ls_g = (np.exp(-0.5 * (df / sigma)**2) / (sigma * np.sqrt(2.0 * np.pi))
            if sigma > 0 else np.zeros_like(df))

    profile = eta_l * ls_l + eta_g * ls_g
    profile = _zeeman_split_freq(freq_axis, profile, B, view_angle_deg)
    return _freq_to_wl_norm(freq_axis, profile, wavelengths_nm)


def stehle_param(wavelengths_nm, n_u, n_l, B, Ne_m3, Te_ev, Ti_ev,
                 view_angle_deg=90.0, species='H'):
    """Parameterized Stehle Stark-Zeeman-Doppler profile (FFT convolution).

    Mirrors pystark.make_stehle_param: the modified-Lorentzian Stark profile is
    built in wavelength, converted to the internal frequency axis, then convolved
    with the Doppler Gaussian *in frequency space* (a frequency-symmetric Gaussian
    is asymmetric in wavelength, so the convolution domain matters).
    """
    _, A = species_to_ZA(species)
    lambda0_nm  = _nist_center_air_nm(n_u, n_l, species, wavelengths_nm)
    freq_center = C / (lambda0_nm * 1e-9)

    # Stark profile on the output wavelength grid, area-normalised in m (mirrors pystark).
    delta_wl12 = _fwhm_stark_loman_nm(n_u, n_l, Ne_m3, Te_ev)
    ls_s_wl    = 1.0 / (np.abs(wavelengths_nm - lambda0_nm)**2.5 + (delta_wl12 / 2.0)**2.5)
    ls_s_wl   /= trapz(ls_s_wl, wavelengths_nm * 1e-9)

    # Convert to freq_axis: interpolate with boundary extension (no zero-fill at edges)
    # then apply λ→ν Jacobian I(ν) = I(λ) · c/ν².
    freq_axis     = _build_freq_axis(wavelengths_nm, freq_center)
    wl_at_freq_nm = C / freq_axis * 1e9                              # nm, descending
    ls_s          = np.interp(wl_at_freq_nm, wavelengths_nm, ls_s_wl) * C / freq_axis**2

    # Analytically-normalised Doppler Gaussian on the extended axis (mirrors pystark.doppler_lineshape).
    mass_kg   = _DOPPLER_MASS_KG.get(A, A * _U)
    v_th      = np.sqrt(2.0 * E * Ti_ev / mass_kg)
    sigma_hz  = v_th * freq_center / (np.sqrt(2.0) * C)
    extra     = 1000
    dfreq     = (freq_axis[-1] - freq_axis[0]) / (len(freq_axis) - 1)
    freq_conv = np.linspace(freq_axis[0]  - extra // 2 * dfreq,
                             freq_axis[-1] + extra // 2 * dfreq,
                             len(freq_axis) + extra)
    ls_d = ((freq_center**-1) * np.sqrt((C / v_th)**2 / np.pi)
            * np.exp(-0.5 * ((freq_conv - freq_center) / sigma_hz)**2))

    ls_sd  = fftconvolve(ls_s, ls_d, 'same')
    ls_sd /= trapz(ls_sd, freq_axis)

    profile = _zeeman_split_freq(freq_axis, ls_sd, B, view_angle_deg)
    return _freq_to_wl_norm(freq_axis, profile, wavelengths_nm)


def griem(wavelengths_nm, n_u, n_l, B, Ne_m3, Te_ev, Ti_ev,
          view_angle_deg=90.0, species='H'):
    """Griem α₁₂ Stark profile — identical to ``voigt``.

    Uses Griem's tabulated α₁₂ coefficient for the Stark HWHM and produces a
    Voigt profile (Lorentzian Stark core + Gaussian Doppler).  Named separately
    so comparison plots can label the curve by its physical origin.
    """
    return voigt(wavelengths_nm, n_u, n_l, B, Ne_m3, Te_ev, Ti_ev,
                 view_angle_deg=view_angle_deg, species=species)


def voigt(wavelengths_nm, n_u, n_l, B, Ne_m3, Te_ev, Ti_ev,
          view_angle_deg=90.0, species='H'):
    """Voigt (Griem Stark Lorentzian + Doppler Gaussian) Zeeman-split profile.

    Mirrors pystark.make_voigt: the Faddeeva function is evaluated in frequency
    space with widths in Hz.
    """
    _, A = species_to_ZA(species)
    lambda0_nm = _nist_center_air_nm(n_u, n_l, species, wavelengths_nm)
    freq_center = C / (lambda0_nm * 1e-9)
    c_over_l2   = C / (lambda0_nm * 1e-9)**2

    hwhm_l_hz  = (_fwhm_stark_griem_nm(n_u, Ne_m3) * 1e-9 * c_over_l2) / 2.0
    fwhm_d_hz  = _fwhm_doppler_nm(lambda0_nm, Ti_ev, A) * 1e-9 * c_over_l2
    sigma_d_hz = (fwhm_d_hz / 2.0) / np.sqrt(2.0 * np.log(2.0))

    freq_axis = _build_freq_axis(wavelengths_nm, freq_center)
    df = freq_axis - freq_center
    z = (df + 1j * hwhm_l_hz) / (sigma_d_hz * np.sqrt(2.0))
    profile = np.real(wofz(z)) / (sigma_d_hz * np.sqrt(2.0 * np.pi))

    profile = _zeeman_split_freq(freq_axis, profile, B, view_angle_deg)
    return _freq_to_wl_norm(freq_axis, profile, wavelengths_nm)
