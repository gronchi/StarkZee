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

# CODATA nuclear masses of the hydrogen isotopes and 4He, keyed by (Z, A).
_ISOTOPE_MASSES_KG = {
    (1, 1): _c.physical_constants['proton mass'][0],
    (1, 2): _c.physical_constants['deuteron mass'][0],
    (1, 3): _c.physical_constants['triton mass'][0],
    (2, 4): _c.physical_constants['alpha particle mass'][0],
}

# Most abundant isotope of each element: Z -> (A, neutral atomic mass [u]) (AME2020).
_MOST_ABUNDANT = {
    1: (1, 1.00782503223), 2: (4, 4.00260325413), 3: (7, 7.0160034366), 4: (9, 9.012183065),
    5: (11, 11.00930536), 6: (12, 12.0), 7: (14, 14.00307400443), 8: (16, 15.99491461957),
    9: (19, 18.99840316273), 10: (20, 19.9924401762), 11: (23, 22.989769282), 12: (24, 23.985041697),
    13: (27, 26.98153853), 14: (28, 27.97692653465), 15: (31, 30.97376199842), 16: (32, 31.9720711744),
    17: (35, 34.968852682), 18: (40, 39.9623831237), 19: (39, 38.9637064864), 20: (40, 39.962590863),
    26: (56, 55.93493633), 28: (58, 57.93534241), 36: (84, 83.9114977282), 42: (98, 97.90540482),
    54: (132, 131.9041550856), 74: (184, 183.95093092),
}

_ATOM_OF_HYDROGEN_ISOTOPE = {1: 'H', 2: 'D', 3: 'T'}

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


def mass_number(Z, A=None):
    """A if given, else the mass number of the most abundant isotope of Z."""
    if A is not None:
        return A
    if Z not in _MOST_ABUNDANT:
        raise ValueError(f"No default isotope for Z={Z}; pass the mass number A.")
    return _MOST_ABUNDANT[Z][0]


def nuclear_mass_kg(Z, A=None):
    """Return the nuclear mass [kg] of the isotope (Z, A); A defaults to the most abundant isotope.

    CODATA values for H, D, T and 4He; otherwise the neutral atomic mass minus
    Z electron masses (electron binding energies, below 1e-6 u for Z <= 30, are
    neglected), or A u - Z m_e for an isotope without a tabulated mass.
    """
    A = mass_number(Z, A)
    if (Z, A) in _ISOTOPE_MASSES_KG:
        return _ISOTOPE_MASSES_KG[(Z, A)]
    tab = _MOST_ABUNDANT.get(Z)
    atomic_u = tab[1] if tab is not None and tab[0] == A else float(A)
    return atomic_u * _c.atomic_mass - Z * _c.m_e


def emitter_mass_kg(Z, A=None, bound_electrons=1):
    """Mass [kg] of the radiating ion: nucleus plus its bound electrons (1 for hydrogen-like)."""
    return nuclear_mass_kg(Z, A) + bound_electrons * _c.m_e


def emitter_mass_amu(Z, A=None, bound_electrons=1):
    """:func:`emitter_mass_kg` in atomic mass units [u]."""
    return emitter_mass_kg(Z, A, bound_electrons) / _c.atomic_mass


def perturber_mass_kg(A_perturber):
    """Mass [kg] of a perturbing ion of mass number A: CODATA p, d, t, alpha for A = 1-4, else A u."""
    A = float(A_perturber)
    k = int(round(A))
    if abs(A - k) < 1e-9 and k in (1, 2, 3):
        return _ISOTOPE_MASSES_KG[(1, k)]
    if abs(A - k) < 1e-9 and k == 4:
        return _ISOTOPE_MASSES_KG[(2, 4)]
    return A * _c.atomic_mass


def reduced_mass_factor(Z, A=None):
    """mu / m_e = 1 / (1 + m_e / M_nucleus)."""
    return 1.0 / (1.0 + _c.m_e / nuclear_mass_kg(Z, A))


def orbital_g_factor(Z, A=None):
    """Orbital g factor of the electron with nuclear motion, g_L = 1 - m_e / M_nucleus."""
    return 1.0 - _c.m_e / nuclear_mass_kg(Z, A)


def core_mass_factors(mass_amu):
    """(mu / m_e, g_L) for an electron bound to the rest of an atom of mass mass_amu [u].

    The core (nucleus + other electrons) has mass M = mass_amu u - m_e;
    mu / m_e = 1 / (1 + m_e / M) and g_L = 1 - m_e / M. None: infinite mass, (1, 1).
    """
    if mass_amu is None:
        return 1.0, 1.0
    M = mass_amu * _c.atomic_mass - _c.m_e
    return 1.0 / (1.0 + _c.m_e / M), 1.0 - _c.m_e / M


def resolve_atom(Z, A=None, atom=None):
    """Element key of the measured-level table for a hydrogen-like emitter.

    For Z = 1 the key follows the isotope: 'H', 'D' or 'T' for A = 1, 2, 3; an
    explicit ``atom`` that names a different isotope is an error (it would put
    the line at the wrong isotope's position). For Z > 1 ``atom`` is returned
    unchanged (the bundled measured levels cover only Z = 1).
    """
    if Z != 1:
        return atom
    A = mass_number(Z, A)
    expected = _ATOM_OF_HYDROGEN_ISOTOPE.get(int(round(A))) if np.isfinite(A) else None
    if atom is None:
        if expected is None:
            raise ValueError(f"No measured levels for a hydrogen isotope with A={A}.")
        return expected
    if expected is not None and atom != expected:
        raise ValueError(f"atom={atom!r} does not match the isotope A={A} (expected {expected!r}).")
    return atom


def reduced_mass_rydberg_ev(Z, A=None):
    """Return the reduced-mass Rydberg energy R_inf mu / m_e [eV] for the isotope (Z, A).

    A defaults to the most abundant isotope of Z (see :func:`nuclear_mass_kg`).
    """
    return RYDBERG_EV * reduced_mass_factor(Z, A)


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
        Photon energy [eV]. Zero-valued elements are mapped elementwise to zero
        wavelength rather than triggering a division-by-zero error.

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
    h_ev_s = _c.hbar * 2.0 * np.pi / _c.e
    energy = np.asarray(energy_ev, dtype=float)
    zero = energy == 0.0
    wl = np.where(zero, 0.0, (h_ev_s * _c.c * 1e9) / np.where(zero, 1.0, energy))
    if np.ndim(energy_ev) == 0:
        return float(wl)
    return wl


def vacuum_to_air_wavelength_nm(wavelength_vac_nm):
    """Convert vacuum wavelength(s) to air wavelength(s) with the Peck & Reeder (1972) formula of NIST ASD.

    Standard air (15 °C, 101 325 Pa, 0.033 % CO2); the formula was fitted over
    185–1700 nm. The 200 nm lower limit is a convention, not the edge of the fit:
    as in NIST ASD, the vacuum wavelength is returned unchanged for wavenumbers
    σ ≥ 50 000 cm⁻¹ (λ ≤ 200 nm) and σ ≤ 5000 cm⁻¹ (λ ≥ 2000 nm).

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
    Peck & Reeder (1972) dispersion formula:

        (n − 1) × 10⁸ = 8060.51 + 2480990 / (132.274 − σ²) + 17455.7 / (39.32957 − σ²)

    where σ = 1/λ_vac [µm⁻¹] = 1000 / λ_vac [nm].

    References
    ----------
    Peck, E. R. & Reeder, K., J. Opt. Soc. Am. 62, 958 (1972).
    """
    lam = np.asarray(wavelength_vac_nm, dtype=float)
    sigma2 = (1e3 / np.clip(lam, 200.0, 2000.0)) ** 2
    n_air = 1.0 + 1e-8 * (8060.51 + 2480990.0 / (132.274 - sigma2)
                                    + 17455.7 / (39.32957 - sigma2))
    in_air = (lam > 200.0) & (lam < 2000.0)          # 5000 < σ < 50 000 cm⁻¹ (NIST ASD)
    out = np.where(in_air, lam / n_air, lam)
    return float(out) if out.ndim == 0 else out


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
