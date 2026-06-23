# Physical constants and conversion utility functions for starkzee
#
# Simple scipy.constants attributes (hbar, m_e, e, k, c, epsilon_0,
# fine_structure) are imported directly by callers.  Only the verbose
# physical_constants[...][0] lookups and unit-conversion functions live here.

import numpy as np
from scipy import constants as _c

A0                 = _c.physical_constants['Bohr radius'][0]               # m
RYDBERG_EV         = _c.physical_constants['Rydberg constant times hc in eV'][0]  # eV  Ryd = 13.6057 eV
HARTREE_EV         = 2.0 * RYDBERG_EV                                             # eV  E_H = 27.2114 eV
BOHR_MAGNETON_EV_T = _c.physical_constants['Bohr magneton in eV/T'][0]            # eV/T
G_S                = abs(_c.physical_constants['electron g factor'][0])   # g_s
HBAR               = _c.hbar
M_E                = _c.m_e
E_CHARGE           = _c.e
FINE_STRUCTURE     = _c.fine_structure

# Nuclear masses for common hydrogenic species (most abundant isotope).
# Used for the reduced-mass Rydberg correction R_atom = R_∞ / (1 + m_e/M_nucleus).
_NUCLEAR_MASSES_KG = {
    1: _c.physical_constants['proton mass'][0],          # H-1
    2: _c.physical_constants['alpha particle mass'][0],  # He-4
}

# Isotope-specific nuclear masses keyed by (Z, A).
# Takes priority over _NUCLEAR_MASSES_KG when A is known.
_ISOTOPE_MASSES_KG = {
    (1, 1): _c.physical_constants['proton mass'][0],
    (1, 2): _c.physical_constants['deuteron mass'][0],
    (1, 3): _c.physical_constants['triton mass'][0],
    (2, 4): _c.physical_constants['alpha particle mass'][0],
}


_SPECIES_TABLE = {
    'h':         (1, 1),
    'hydrogen':  (1, 1),
    'd':         (1, 2),
    'deuterium': (1, 2),
    't':         (1, 3),
    'tritium':   (1, 3),
}


def species_to_ZA(species):
    """Return (Z, A) for a species name string.

    Parameters
    ----------
    species : str
        Species identifier, case-insensitive.  Accepted values:
        ``'H'`` / ``'hydrogen'``, ``'D'`` / ``'deuterium'``,
        ``'T'`` / ``'tritium'``.

    Returns
    -------
    Z : int
        Nuclear charge number.
    A : int
        Atomic mass number (most abundant isotope).

    Raises
    ------
    ValueError
        If *species* is not in the lookup table.
    """
    key = species.strip().lower()
    if key not in _SPECIES_TABLE:
        raise ValueError(
            f"Unknown species '{species}'. "
            f"Supported: {sorted({k.upper() for k in _SPECIES_TABLE if len(k) <= 2})}."
        )
    return _SPECIES_TABLE[key]


def nuclear_mass_kg(Z, A=None):
    """Return the nuclear mass [kg] for nuclear charge *Z* and mass number *A*.

    Parameters
    ----------
    Z : int
        Nuclear charge number.
    A : int, optional
        Atomic mass number.  When supplied the isotope-specific CODATA value
        is used (H, D, T, ⁴He); otherwise falls back to the most-abundant
        isotope for that Z.

    Returns
    -------
    float
        Nuclear mass in kg.
    """
    if A is not None and (Z, A) in _ISOTOPE_MASSES_KG:
        return _ISOTOPE_MASSES_KG[(Z, A)]
    if A is not None:
        return A * _c.atomic_mass
    if Z in _NUCLEAR_MASSES_KG:
        return _NUCLEAR_MASSES_KG[Z]
    return 2.0 * Z * _c.atomic_mass


def reduced_mass_rydberg_ev(Z, A=None):
    """Return the reduced-mass-corrected Rydberg energy for nuclear charge *Z* [eV].

    Replaces the infinite-nuclear-mass value ``RYDBERG_EV`` with the
    atom-specific value:

        R_atom = R_∞ × μ / m_e = R_∞ / (1 + m_e / M_nucleus)

    where M_nucleus is the nuclear mass from :func:`nuclear_mass_kg`.

    Parameters
    ----------
    Z : int
        Nuclear charge number (1 = hydrogen, 2 = helium, etc.).

    Returns
    -------
    float
        Reduced-mass-corrected Rydberg energy [eV].
    """
    m_nucleus = nuclear_mass_kg(Z, A)
    return RYDBERG_EV / (1.0 + _c.m_e / m_nucleus)


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


def vacuum_to_air_wavelength_nm(wavelength_vac_nm):
    """Convert vacuum wavelength(s) to air wavelength(s) using the Edlén (1966) formula.

    Valid for standard dry air (15 °C, 101 325 Pa) over approximately 200–2000 nm.
    The formula is the one used by NIST ASD for all tabulated air wavelengths.

    Parameters
    ----------
    wavelength_vac_nm : float or array-like
        Vacuum wavelength [nm].

    Returns
    -------
    float or ndarray
        Air wavelength [nm].

    Notes
    -----
    Edlén (1966) dispersion formula:

        (n − 1) × 10⁸ = 8342.13 + 2406030 / (130 − σ²) + 15997 / (38.9 − σ²)

    where σ = 1/λ_vac [µm⁻¹] = 1000 / λ_vac [nm].

    References
    ----------
    Edlén, B., Metrologia 2, 71 (1966).
    """
    sigma2 = (1e3 / np.asarray(wavelength_vac_nm, dtype=float)) ** 2
    n_air = 1.0 + 1e-8 * (8342.13 + 2406030.0 / (130.0 - sigma2)
                                    + 15997.0  / (38.9  - sigma2))
    return np.asarray(wavelength_vac_nm, dtype=float) / n_air


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
