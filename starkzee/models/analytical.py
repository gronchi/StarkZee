"""
Analytical Stark-Zeeman Doppler lineshape models.

Implements lomanowski, stehle_param, and voigt directly.

All functions share the same signature::

    profile = <model>(wavelengths_nm, n_u, n_l, B, Ne_m3, Te_ev, Ti_ev,
                      view_angle_deg=90.0, species='H')

and return an area-normalised lineshape (1/m) on *wavelengths_nm*.

Based on pystark package (https://github.com/jsallcock/pystark).
"""

import numpy as np
from scipy.constants import c as C, e as E, m_e as M_E, m_p as M_P
from scipy.special import wofz

from starkzee.utils import species_to_ZA
from starkzee.convolutions import convolve_fft

_SIGMA2FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))

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
    sigma_nm = lambda0_nm * np.sqrt(E * Ti_ev / (A * M_P * C**2))
    return _SIGMA2FWHM * sigma_nm


def _zeeman_split(wavelengths_nm, lambda0_nm, profile, B, view_angle_deg):
    """First-order Zeeman splitting in wavelength space."""
    if B == 0.0:
        return profile
    theta = np.deg2rad(view_angle_deg)
    rel_pi    = np.sin(theta)**2 / 2.0
    rel_sigma = (1.0 + np.cos(theta)**2) / 4.0
    # σ shift in nm: Δλ = λ₀² eB / (4π m_e c)
    dlambda_nm = (lambda0_nm * 1e-9)**2 * E * B / (4.0 * np.pi * M_E * C) * 1e9
    # σ- (red) and σ+ (blue) components
    sigma_minus = rel_sigma * np.interp(
        wavelengths_nm - dlambda_nm, wavelengths_nm, profile, left=0.0, right=0.0)
    sigma_plus  = rel_sigma * np.interp(
        wavelengths_nm + dlambda_nm, wavelengths_nm, profile, left=0.0, right=0.0)
    return rel_pi * profile + sigma_minus + sigma_plus


def _area_norm_m(profile, wavelengths_nm):
    """Area-normalise to 1/m (∫ profile dλ_m = 1)."""
    area = np.trapz(profile, wavelengths_nm * 1e-9)
    return profile / area if area > 0 else profile


def lomanowski(wavelengths_nm, n_u, n_l, B, Ne_m3, Te_ev, Ti_ev,
               view_angle_deg=90.0, species='H'):
    """Lomanowski pseudo-Voigt Stark-Zeeman-Doppler profile."""
    _, A = species_to_ZA(species)
    lambda0_nm = np.mean(wavelengths_nm)

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

    d = wavelengths_nm - lambda0_nm
    hwhm = fwhm / 2.0

    _stark_norm = 2.641279471021934
    ls_l = (hwhm**1.5 / _stark_norm) / (np.abs(d)**2.5 + hwhm**2.5)

    sigma = fwhm / _SIGMA2FWHM
    ls_g = (np.exp(-0.5 * (d / sigma)**2) / (sigma * np.sqrt(2.0 * np.pi))
            if sigma > 0 else np.zeros_like(d))

    profile = eta_l * ls_l + eta_g * ls_g
    profile = _zeeman_split(wavelengths_nm, lambda0_nm, profile, B, view_angle_deg)
    return _area_norm_m(profile, wavelengths_nm)


def stehle_param(wavelengths_nm, n_u, n_l, B, Ne_m3, Te_ev, Ti_ev,
                 view_angle_deg=90.0, species='H'):
    """Parameterised Stehle Stark-Zeeman-Doppler profile (FFT convolution)."""
    _, A = species_to_ZA(species)
    lambda0_nm = np.mean(wavelengths_nm)

    delta_wl12 = _fwhm_stark_loman_nm(n_u, n_l, Ne_m3, Te_ev)
    d = wavelengths_nm - lambda0_nm
    ls_s = 1.0 / (np.abs(d)**2.5 + (delta_wl12 / 2.0)**2.5)
    ls_s /= np.trapz(ls_s, wavelengths_nm)

    fwhm_g = _fwhm_doppler_nm(lambda0_nm, Ti_ev, A)
    sigma_g = fwhm_g / _SIGMA2FWHM
    kernel = np.exp(-0.5 * (d / sigma_g)**2) if sigma_g > 0 else np.ones(1)

    profile = convolve_fft(wavelengths_nm, ls_s, kernel)
    profile = _zeeman_split(wavelengths_nm, lambda0_nm, profile, B, view_angle_deg)
    return _area_norm_m(profile, wavelengths_nm)


def voigt(wavelengths_nm, n_u, n_l, B, Ne_m3, Te_ev, Ti_ev,
          view_angle_deg=90.0, species='H'):
    """Voigt (Griem Stark Lorentzian + Doppler Gaussian) Zeeman-split profile."""
    _, A = species_to_ZA(species)
    lambda0_nm = np.mean(wavelengths_nm)

    hwhm_l_nm = _fwhm_stark_griem_nm(n_u, Ne_m3) / 2.0
    fwhm_g_nm = _fwhm_doppler_nm(lambda0_nm, Ti_ev, A)
    sigma_g_nm = fwhm_g_nm / _SIGMA2FWHM

    d = wavelengths_nm - lambda0_nm
    z = (d + 1j * hwhm_l_nm) / (sigma_g_nm * np.sqrt(2.0))
    profile = np.real(wofz(z)) / (sigma_g_nm * np.sqrt(2.0 * np.pi))

    profile = _zeeman_split(wavelengths_nm, lambda0_nm, profile, B, view_angle_deg)
    return _area_norm_m(profile, wavelengths_nm)
