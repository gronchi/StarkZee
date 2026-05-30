# Physical constants and conversion utility functions for starkzee
#
# Simple scipy.constants attributes (hbar, m_e, e, k, c, epsilon_0,
# fine_structure) are imported directly by callers.  Only the verbose
# physical_constants[...][0] lookups and unit-conversion functions live here.

import numpy as np
from scipy import constants as _c

A0                 = _c.physical_constants['Bohr radius'][0]               # m
RYDBERG_EV         = _c.physical_constants['Rydberg constant times hc in eV'][0]  # eV
BOHR_MAGNETON_EV_T = _c.physical_constants['Bohr magneton in eV/T'][0]            # eV/T


def temp_ev_to_joules(t_ev):
    """Convert a thermal energy from eV to Joules.

    Uses the exact SI definition e = 1.602176634e-19 C so that
    1 eV = e × 1 J exactly.

    Parameters
    ----------
    t_ev : float or array-like
        Energy in electronvolts [eV].

    Returns
    -------
    float or ndarray
        Energy in Joules [J].
    """
    return t_ev * _c.e


def temp_ev_to_kelvin(t_ev):
    """Convert a thermal energy from eV to the equivalent temperature in Kelvin.

    Uses k_B T = e × T_eV, so T [K] = e / k_B × T_eV.

    Parameters
    ----------
    t_ev : float or array-like
        Thermal energy in electronvolts [eV].

    Returns
    -------
    float or ndarray
        Equivalent temperature in Kelvin [K].
    """
    return t_ev * _c.e / _c.k


def energy_ev_to_wavelength_nm(energy_ev):
    """Convert photon energy in eV to vacuum wavelength in nm.

    Uses the relation E = hc / λ, giving λ [nm] = hc [eV·nm] / E [eV].

    Parameters
    ----------
    energy_ev : float or array-like
        Photon energy [eV]. Zero-valued elements are mapped to zero wavelength
        rather than triggering a division-by-zero error.

    Returns
    -------
    float or ndarray
        Vacuum wavelength [nm].  For an increasing energy array the returned
        wavelengths are in *decreasing* order.

    Notes
    -----
    hc = 2π ħ c ≈ 1239.842 eV·nm.  The code derives this from CODATA values
    of ħ and c so there is no hardcoded conversion constant.
    """
    if np.any(energy_ev == 0):
        return np.zeros_like(energy_ev)
    h_ev_s = _c.hbar * 2.0 * np.pi / _c.e
    return (h_ev_s * _c.c / energy_ev) * 1e9


def wavelength_nm_to_energy_ev(wavelength_nm):
    """Convert vacuum wavelength in nm to photon energy in eV.

    Uses E = hc / λ.

    Parameters
    ----------
    wavelength_nm : float or array-like
        Vacuum wavelength [nm]. Must be non-zero.

    Returns
    -------
    float or ndarray
        Photon energy [eV].  For an increasing wavelength array the returned
        energies are in *decreasing* order.
    """
    h_ev_s = _c.hbar * 2.0 * np.pi / _c.e
    return (h_ev_s * _c.c) / (wavelength_nm * 1e-9)


def frequency_thz_to_energy_ev(frequency_thz):
    """Convert photon frequency in THz to energy in eV.

    Uses E = h·f.

    Parameters
    ----------
    frequency_thz : float or array-like
        Photon frequency [THz].

    Returns
    -------
    float or ndarray
        Photon energy [eV].
    """
    h_ev_s = _c.h / _c.e
    return h_ev_s * np.asarray(frequency_thz) * 1e12


def energy_ev_to_frequency_thz(energy_ev):
    """Convert photon energy in eV to frequency in THz.

    Uses f = E / h.

    Parameters
    ----------
    energy_ev : float or array-like
        Photon energy [eV].

    Returns
    -------
    float or ndarray
        Photon frequency [THz].
    """
    h_ev_s = _c.h / _c.e
    return np.asarray(energy_ev) / h_ev_s / 1e12


def wavenumber_cm_to_energy_ev(wavenumber_cm):
    """Convert photon wavenumber in cm⁻¹ to energy in eV.

    Uses E = hc·ν̃.

    Parameters
    ----------
    wavenumber_cm : float or array-like
        Photon wavenumber [cm⁻¹].

    Returns
    -------
    float or ndarray
        Photon energy [eV].
    """
    hc_ev_cm = _c.h * _c.c / _c.e * 100.0   # eV·cm
    return hc_ev_cm * np.asarray(wavenumber_cm)


def energy_ev_to_wavenumber_cm(energy_ev):
    """Convert photon energy in eV to wavenumber in cm⁻¹.

    Uses ν̃ = E / (hc).

    Parameters
    ----------
    energy_ev : float or array-like
        Photon energy [eV].

    Returns
    -------
    float or ndarray
        Photon wavenumber [cm⁻¹].
    """
    hc_ev_cm = _c.h * _c.c / _c.e * 100.0
    return np.asarray(energy_ev) / hc_ev_cm
