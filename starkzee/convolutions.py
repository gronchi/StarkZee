# Convolutions module for Doppler and instrument broadening in starkzee

import numpy as np
from scipy.fft import fft, ifft, ifftshift
from scipy.constants import c as C_LIGHT, e as E_CHARGE, m_p as _M_P


def uniform_energy_grid(grid, resample=True):
    """Return an increasing uniform work grid and an output interpolation function.

    Nonuniform inputs are evaluated at no worse than their smallest spacing.
    Interpolation returns energy densities, not densities in the input coordinate.
    """
    original = np.asarray(grid, dtype=float)
    if original.ndim != 1 or len(original) < 2 or not np.all(np.isfinite(original)):
        raise ValueError("Energy grid must contain at least two finite points.")
    delta = np.diff(original)
    if not (np.all(delta > 0) or np.all(delta < 0)):
        raise ValueError("Energy grid must be strictly monotonic.")
    ordered = original if delta[0] > 0 else original[::-1]
    steps = np.diff(ordered)
    if not resample or np.allclose(steps, steps[0], rtol=1e-8, atol=0):
        work = ordered
    else:
        count = int(np.ceil((ordered[-1] - ordered[0]) / steps.min())) + 1
        if count > 1000000:
            raise ValueError("Nonuniform grid requires over one million work points; resample explicitly.")
        work = np.linspace(ordered[0], ordered[-1], count)
    return work, lambda values: np.interp(original, work, values)



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
        Line-center photon energy [eV].
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
    kernel is discretely normalized. Cropping to the output window can lose
    area; enlarge the window to check convergence. No renormalization of the
    returned spectrum is performed.

    Parameters
    ----------
    grid : array-like, shape (N,)
        Uniform coordinate grid (wavelengths or energies).  Used only to
        validate spacing; ascending and descending grids are accepted.
    profile : array-like, shape (N,)
        Spectral profile to be convolved.
    kernel : array-like, shape (N,)
        Broadening kernel (not necessarily normalized).  Its center must
        coincide with index ``N//2`` (zero lag, not the physical line center); ``ifftshift`` is applied
        internally to place the peak at index 0 (the origin) for correct
        phase -- this is the inverse of the padding/centering ``fftshift``
        would undo, and for odd-length arrays the two differ by one sample.

    Returns
    -------
    ndarray, shape (N,)
        Convolved profile, trimmed back to length N.

    Notes
    -----
    Both arrays are zero-extended. Boundary intensity is not extended into
    an artificial pedestal outside the observation window.
    """
    grid = np.asarray(grid, dtype=float)
    profile = np.asarray(profile, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    n = len(grid)
    if n < 2 or grid.ndim != 1 or profile.shape != grid.shape or kernel.shape != grid.shape:
        raise ValueError("grid, profile and kernel must be one-dimensional arrays of equal length >= 2.")
    steps = np.diff(grid)
    if not np.all(np.isfinite(grid)) or steps[0] == 0 or not np.allclose(steps, steps[0], rtol=1e-8, atol=0):
        raise ValueError("Convolution requires a uniform, strictly monotonic grid.")
    if not np.all(np.isfinite(profile)) or not np.all(np.isfinite(kernel)) or kernel.sum() <= 0:
        raise ValueError("Profile and kernel must be finite and kernel must have positive mass.")
    pad_len = n // 2
    profile_padded = np.pad(profile, pad_len, mode='constant')
    kernel_padded = np.pad(kernel, pad_len, mode='constant', constant_values=0.0)

    total_area = np.sum(kernel_padded)
    if total_area > 0:
        kernel_padded /= total_area

    F_prof = fft(profile_padded)
    F_kern = fft(ifftshift(kernel_padded))

    convoluted_padded = np.real(ifft(F_prof * F_kern))

    return convoluted_padded[pad_len:-pad_len]


def apply_doppler_broadening(wavelengths_nm, profile, Ti_ev, species='H', lambda0_nm=None):
    """Apply thermal Doppler broadening to a spectrum on a uniform wavelength grid.

    Constructs a Gaussian kernel with 1/e width

        Δλ_D = λ₀ × v_th / c,   v_th = √(2 T_i / m c²)

    centered at zero lag (index N//2), and convolves it with ``profile`` via
    :func:`convolve_fft`.

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
    lambda0_nm : float, optional
        The physical line-center wavelength [nm] to use for the Doppler-width
        scale. When omitted (default), falls back to ``mean(wavelengths_nm)``,
        which is only a good proxy for the true line center when the grid is
        narrow and centered on the line; for a wide or off-center window,
        pass the actual transition wavelength explicitly so the modeled
        Doppler width doesn't silently depend on the observation window
        instead of the emitter.

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
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    if not np.isfinite(Ti_ev) or Ti_ev < 0:
        raise ValueError("Ti_ev must be finite and nonnegative.")
    if Ti_ev == 0:
        return np.array(profile, dtype=float, copy=True)
    _, A = species_to_ZA(species)
    mc2_ev = (A * _M_P) * (C_LIGHT ** 2) / E_CHARGE
    v_th_over_c = np.sqrt(2.0 * Ti_ev / mc2_ev)

    lambda0_nm = np.mean(wavelengths_nm) if lambda0_nm is None else lambda0_nm
    w_doppler_nm = lambda0_nm * v_th_over_c

    if not np.isfinite(lambda0_nm) or lambda0_nm <= 0:
        raise ValueError("lambda0_nm must be finite and positive.")
    x = wavelengths_nm - wavelengths_nm[len(wavelengths_nm) // 2]
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

    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    x = wavelengths_nm - wavelengths_nm[len(wavelengths_nm) // 2]
    kernel = np.exp(-x**2 / (2.0 * sigma**2))

    return convolve_fft(wavelengths_nm, profile, kernel)
