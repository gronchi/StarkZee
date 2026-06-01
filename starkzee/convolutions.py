# Convolutions module for Doppler and instrument broadening in starkzee

import numpy as np
from scipy.fft import fft, ifft, fftshift
from scipy.constants import c as C_LIGHT, e as E_CHARGE, m_p as _M_P
from starkzee.utils import energy_ev_to_wavelength_nm


def calculate_doppler_width_ev(E0_ev, Ti_ev, A_emitter):
    """Return the thermal Doppler 1/e half-width ΔE_D [eV].

    For a Maxwellian ion distribution at temperature T_i the emission line is
    Doppler broadened into a Gaussian with 1/e half-width:

        ΔE_D = E₀ × v_th / c

    where the thermal velocity is

        v_th = √(2 k_B T_i / m)  →  v_th/c = √(2 T_i / (m c² / e))

    and m c² is the emitter rest energy in eV.  The resulting FWHM is

        FWHM = 2 √(ln 2) × ΔE_D

    Parameters
    ----------
    E0_ev : float
        Line-centre photon energy [eV].
    Ti_ev : float
        Ion temperature [eV].
    A_emitter : float
        Atomic mass number of the emitting species (e.g. 1 for H, 2 for D,
        12 for C).  Used to compute the emitter mass m = A × m_p.

    Returns
    -------
    float
        Gaussian 1/e half-width ΔE_D [eV].  The full Gaussian profile is
        exp(−(E − E₀)² / ΔE_D²).

    Notes
    -----
    The formula uses the non-relativistic approximation v_th ≪ c, valid for
    all plasma temperatures relevant to magnetic fusion diagnostics.
    """
    m_emitter = A_emitter * _M_P
    mc2 = m_emitter * (C_LIGHT**2) / E_CHARGE
    v_th_over_c = np.sqrt(2.0 * Ti_ev / mc2)
    delta_E_D = E0_ev * v_th_over_c
    return delta_E_D


def convolve_fft(grid, profile, kernel):
    """Convolve *profile* with *kernel* on a uniform *grid* using FFT.

    Both ``profile`` and ``kernel`` must be sampled on the same uniform grid.
    The convolution is computed via the convolution theorem:

        (f * g)[n] = IFFT(FFT(f) × FFT(g))

    The input arrays are zero-padded by half their length on each side to
    suppress the wrap-around artefacts of the circular FFT convolution.  The
    kernel is area-normalised before application so that the integrated
    intensity of the profile is conserved.

    Parameters
    ----------
    grid : array-like, shape (N,)
        Uniform coordinate grid (wavelengths or energies).  Used only to
        determine array length; the actual coordinate values are not used.
    profile : array-like, shape (N,)
        Spectral profile to be convolved.
    kernel : array-like, shape (N,)
        Broadening kernel (not necessarily normalised).  Its centre must
        coincide with the centre of ``grid``; ``fftshift`` is applied
        internally to place the peak at the origin for correct phase.

    Returns
    -------
    ndarray, shape (N,)
        Convolved profile, trimmed back to length N.

    Notes
    -----
    Edge-padding (``mode='edge'``) is used for the profile to reduce ringing
    at the spectrum boundaries.  Zero-padding is used for the kernel because
    the kernel is assumed to be compact relative to the grid.
    """
    n = len(grid)
    pad_len = n // 2
    profile_padded = np.pad(profile, pad_len, mode='edge')
    kernel_padded = np.pad(kernel, pad_len, mode='constant', constant_values=0.0)

    total_area = np.sum(kernel_padded)
    if total_area > 0:
        kernel_padded /= total_area

    F_prof = fft(profile_padded)
    F_kern = fft(fftshift(kernel_padded))

    convoluted_padded = np.real(ifft(F_prof * F_kern))

    return convoluted_padded[pad_len:-pad_len]


def apply_doppler_broadening(wavelengths_nm, profile, Ti_ev, species='H'):
    """Apply thermal Doppler broadening to a spectrum on a uniform wavelength grid.

    Constructs a Gaussian kernel with 1/e width

        Δλ_D = λ₀ × v_th / c,   v_th = √(2 T_i / m c²)

    centred at the grid midpoint λ₀ = mean(wavelengths_nm), and convolves it
    with ``profile`` via :func:`convolve_fft`.

    Parameters
    ----------
    wavelengths_nm : ndarray, shape (N,)
        Uniform wavelength grid [nm].
    profile : ndarray, shape (N,)
        Input spectral profile (arbitrary units).
    Ti_ev : float
        Ion temperature [eV].
    species : str, optional
        Emitting species: ``'H'`` / ``'hydrogen'``, ``'D'`` / ``'deuterium'``,
        or ``'T'`` / ``'tritium'``.  Default is ``'H'``.

    Returns
    -------
    ndarray, shape (N,)
        Doppler-broadened profile (same units as input).

    Notes
    -----
    The wavelength grid must be uniform (constant spacing).  If the grid is
    non-uniform, resample before calling this function.
    """
    from starkzee.utils import species_to_ZA
    _, A = species_to_ZA(species)
    mc2_ev = (A * _M_P) * (C_LIGHT ** 2) / E_CHARGE
    v_th_over_c = np.sqrt(2.0 * Ti_ev / mc2_ev)

    lambda0_nm = np.mean(wavelengths_nm)
    w_doppler_nm = lambda0_nm * v_th_over_c

    x = wavelengths_nm - lambda0_nm
    kernel = np.exp(-x**2 / (w_doppler_nm**2))

    return convolve_fft(wavelengths_nm, profile, kernel)


def apply_instrument_broadening(wavelengths_nm, profile, fwhm_nm):
    """Apply Gaussian instrumental slit broadening to a spectrum.

    Models the finite spectral resolution of the spectrometer as a Gaussian
    instrumental profile with the given FWHM.  The standard deviation is

        σ = FWHM / (2 √(2 ln 2))

    and the kernel is exp(−(λ − λ̄)² / (2σ²)).

    Parameters
    ----------
    wavelengths_nm : ndarray, shape (N,)
        Uniform wavelength grid [nm].
    profile : ndarray, shape (N,)
        Input spectral profile (arbitrary units).
    fwhm_nm : float
        Instrumental FWHM [nm].  If ≤ 0 the profile is returned unchanged.

    Returns
    -------
    ndarray, shape (N,)
        Instrument-broadened profile (same units as input).
    """
    if fwhm_nm <= 0:
        return profile

    sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    x = wavelengths_nm - np.mean(wavelengths_nm)
    kernel = np.exp(-x**2 / (2.0 * sigma**2))

    return convolve_fft(wavelengths_nm, profile, kernel)
