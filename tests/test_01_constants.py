"""
test_01_constants.py — Physical constants and unit conversion tests.

Verifies that all physical constants in starkzee/utils.py are correct
to the required precision, and that wavelength <-> energy conversions
give known spectroscopic results.
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scipy.constants import (
    hbar as HBAR, m_e as M_E, e as E_CHARGE, k as K_B,
    c as C_LIGHT, epsilon_0 as EPSILON_0, fine_structure as FINE_STRUCTURE,
)
from starkzee.utils import (
    A0, RYDBERG_EV, BOHR_MAGNETON_EV_T,
    energy_ev_to_wavelength_nm, wavelength_nm_to_energy_ev,
)


# ── Helper ────────────────────────────────────────────────────────────────────
def relerr(got, ref):
    return abs(got - ref) / abs(ref)


# ── 1. Physical constants vs. CODATA 2018 ────────────────────────────────────

def test_hbar_value():
    """CODATA 2018: ħ = 1.054571817e-34 J·s"""
    assert relerr(HBAR, 1.054571817e-34) < 1e-9, f"HBAR = {HBAR}"

def test_electron_mass():
    """CODATA 2022: me = 9.1093837139e-31 kg"""
    assert relerr(M_E, 9.1093837139e-31) < 1e-9, f"M_E = {M_E}"

def test_electron_charge():
    """CODATA 2018: e = 1.602176634e-19 C (exact)"""
    assert relerr(E_CHARGE, 1.602176634e-19) < 1e-12, f"E_CHARGE = {E_CHARGE}"

def test_bohr_radius():
    """CODATA 2018: a0 = 5.29177210903e-11 m"""
    assert relerr(A0, 5.29177210903e-11) < 1e-9, f"A0 = {A0}"

def test_fine_structure():
    """CODATA 2018: α = 1/137.035999084"""
    assert relerr(FINE_STRUCTURE, 1.0/137.035999084) < 1e-9, f"alpha = {FINE_STRUCTURE}"

def test_rydberg_eV():
    """Rydberg energy = 13.605693122994 eV."""
    assert relerr(RYDBERG_EV, 13.605693122994) < 1e-10, f"RYDBERG_EV = {RYDBERG_EV}"

def test_bohr_magneton_eV_T():
    """Bohr magneton μ_B = eħ/(2me) = 5.7883818... × 10⁻⁵ eV/T."""
    mu_B_calc = E_CHARGE * HBAR / (2.0 * M_E * E_CHARGE)  # in eV/T
    assert relerr(BOHR_MAGNETON_EV_T, mu_B_calc) < 1e-6, (
        f"BOHR_MAGNETON_EV_T={BOHR_MAGNETON_EV_T}, calc={mu_B_calc}"
    )

def test_hc_product():
    """hc = 1239.84193... eV·nm — the standard photon energy conversion."""
    h_ev_s = HBAR * 2.0 * np.pi / E_CHARGE
    hc_eV_nm = h_ev_s * C_LIGHT * 1e9
    assert relerr(hc_eV_nm, 1239.84193) < 1e-5, f"hc = {hc_eV_nm} eV·nm"


# ── 2. Derived spectroscopic quantities ───────────────────────────────────────



def test_H_lyman_alpha_energy():
    """H Lyman-α: n=2→1 gives (3/4)×13.6057 eV.
    
    Note: the commonly cited 10.1993 eV is the experimental value with
    relativistic/QED corrections. The non-relativistic Bohr formula gives
    (3/4) × RYDBERG_EV = 10.2043 eV.  We test the Bohr formula value.
    """
    E_lya_H = (3.0 / 4.0) * RYDBERG_EV
    assert relerr(E_lya_H, 10.2043) < 1e-4, f"H Ly-α = {E_lya_H} eV"

def test_H_lyman_alpha_wavelength():
    """H Lyman-α wavelength = 121.567 nm (vacuum)."""
    E_lya_H = (3.0 / 4.0) * RYDBERG_EV
    lam = energy_ev_to_wavelength_nm(E_lya_H)
    assert relerr(lam, 121.567) < 1e-3, f"H Ly-α λ = {lam} nm"

def test_CVI_lyman_alpha_energy():
    """C VI (Z=6) Lyman-α: (3/4)×36×13.6057 = 367.354 eV."""
    E0 = (3.0 / 4.0) * 36.0 * RYDBERG_EV
    assert relerr(E0, 367.354) < 1e-4, f"C VI Ly-α = {E0} eV"

def test_CVI_lyman_alpha_wavelength():
    """C VI Lyman-α: ~3.375 nm."""
    E0 = (3.0 / 4.0) * 36.0 * RYDBERG_EV
    lam = energy_ev_to_wavelength_nm(E0)
    assert relerr(lam, 3.375) < 1e-3, f"C VI Ly-α λ = {lam} nm"

# NOTE: test_CIV_n5_n4_energy removed — C IV is lithium-like (3 electrons),
# not a hydrogen-like Z=4 ion. A proper C IV treatment requires a quantum-defect
# model; see FUTURE_MULTIELECTRON_CIV.md for the implementation roadmap.


# ── 3. Roundtrip conversion ───────────────────────────────────────────────────

def test_energy_wavelength_roundtrip():
    """energy -> wavelength -> energy roundtrip must be exact."""
    E_in = np.array([1.0, 5.0, 100.0, 367.354])
    lam = energy_ev_to_wavelength_nm(E_in)
    E_out = wavelength_nm_to_energy_ev(lam)
    np.testing.assert_allclose(E_out, E_in, rtol=1e-10,
                               err_msg="energy↔wavelength roundtrip failed")
