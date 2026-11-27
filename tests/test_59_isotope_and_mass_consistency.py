"""Isotope selection, nuclear-motion (reduced-mass) terms, masses and unit conventions."""
import math
import warnings

import numpy as np
import pytest

from starkzee.microfield import (calculate_holtsmark_field, calculate_normal_field, holtsmark_distribution,
                                 microfield_quadrature, potekhin_distribution)
from starkzee.radiator import build_basis, build_hamiltonian, einstein_a_substate_rates
from starkzee.radiators import HydrogenicTransition
from starkzee.static_profile import calculate_static_profile
from starkzee.utils import (emitter_mass_amu, energy_ev_to_wavelength_nm, nuclear_mass_kg, orbital_g_factor,
                            reduced_mass_factor, resolve_atom, vacuum_to_air_wavelength_nm)


# ---------------------------------------------------------------- isotopes and measured levels
def test_measured_levels_follow_the_isotope():
    assert resolve_atom(1, 2) == "D"
    assert resolve_atom(1, None) == "H"
    with pytest.raises(ValueError, match="does not match"):
        resolve_atom(1, 2, "H")
    d = HydrogenicTransition(3, 2, 1, 0.0, A=2)
    # NIST D I H-alpha (vacuum) 656.2824 nm; the H position is 656.4610 nm
    assert energy_ev_to_wavelength_nm(d.E0_line) == pytest.approx(656.2824, abs=2e-4)
    with pytest.raises(ValueError, match="does not match"):
        HydrogenicTransition(3, 2, 1, 0.0, A=2, atom="H")


def test_missing_measured_levels_use_the_dirac_energy_below_the_limit():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = HydrogenicTransition(7, 2, 1, 0.0, A=2)             # D levels are tabulated to n = 6
        h = HydrogenicTransition(7, 2, 1, 0.0, A=1)             # H to n = 8 (measured)
    # the isotope shift of the line is the reduced-mass ratio to ~1e-7
    ratio = reduced_mass_factor(1, 2) / reduced_mass_factor(1, 1)
    assert d.E0_line / h.E0_line == pytest.approx(ratio, rel=2e-7)
    with pytest.warns(UserWarning, match="Dirac"):
        build_hamiltonian(9, 1, 0.0, A=3, use_empirical_data=True)


# ---------------------------------------------------------------- reduced-mass terms
def test_nuclear_motion_scales_fine_structure_zeeman_and_diamagnetic_terms():
    n, B = 3, 300.0
    mu = reduced_mass_factor(1, 1)
    basis = build_basis(n)

    def parts(A):
        h0 = build_hamiltonian(n, 1, 0.0, False, True, A)
        hz = build_hamiltonian(n, 1, B, False, True, A) - h0
        hq = build_hamiltonian(n, 1, B, True, True, A) - build_hamiltonian(n, 1, B, False, True, A)
        return np.linalg.eigvalsh(h0), hz, hq

    e1, z1, q1 = parts(1)
    ei, zi, qi = parts(math.inf)
    np.testing.assert_allclose(np.ptp(e1), mu * np.ptp(ei), rtol=1e-9)              # fine structure ∝ mu
    for k, s in enumerate(basis):                                                     # g_L = 1 - m_e/M
        expect = zi[k, k].real + (orbital_g_factor(1, 1) - 1) * 5.7883818060e-5 * B * s.ml
        assert z1[k, k].real == pytest.approx(expect, rel=1e-8, abs=1e-15)
    np.testing.assert_allclose(q1, qi / mu**3, rtol=1e-12, atol=1e-20)              # e^2 B^2 r^2 / 8 mu


def test_hydrogen_rates_include_the_reduced_mass():
    # Wiese & Fuhr (2009): A(2p-1s) = 6.2648e8 s^-1, A(3d-2p) = 6.4651e7 s^-1
    def term_rate(n_u, n_l, l):
        rates, basis = einstein_a_substate_rates(n_u, n_l, 1), build_basis(n_u)
        return np.mean([rates[i] for i, s in enumerate(basis) if s.l == l])
    assert term_rate(2, 1, 1) == pytest.approx(6.2648e8, rel=1e-4)
    assert term_rate(3, 2, 2) == pytest.approx(6.4651e7, rel=1e-4)
    assert einstein_a_substate_rates(2, 1, 1, A=math.inf)[2] / einstein_a_substate_rates(2, 1, 1)[2] \
        == pytest.approx(1 / reduced_mass_factor(1, 1), rel=1e-12)


# ---------------------------------------------------------------- masses
def test_default_isotope_and_masses():
    assert HydrogenicTransition(2, 1, 6, 0.0, use_empirical_data=False).A == 12
    assert nuclear_mass_kg(6) == pytest.approx(12 * 1.66053906660e-27 - 6 * 9.1093837015e-31, rel=1e-9)
    assert emitter_mass_amu(1, 2) == pytest.approx(2.014101778, rel=1e-8)          # D atom


def test_hydrogen_like_doppler_width_uses_the_emitter_mass():
    E0 = 36 * 13.6057 * (1 / 4 - 1 / 9)
    grid = np.linspace(E0 - 0.6, E0 + 0.6, 6001)
    kw = dict(n_u=3, n_l=2, Z=6, B=0.0, Ne_m3=1e18, Te_ev=10.0, energies_ev=grid, Ti_ev=1000.0,
              use_empirical_data=False, num_f=4, num_mu=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sum(calculate_static_profile(**kw))
        b = sum(calculate_static_profile(A=12, **kw))
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="has Z=1"):
        calculate_static_profile(species="D", **kw)


# ---------------------------------------------------------------- microfield units
def test_microfield_beta_is_in_the_holtsmark_normal_field():
    F_H = calculate_holtsmark_field(1e22)
    F0, _ = calculate_normal_field(1e22)
    assert F_H / F0 == pytest.approx(2 * np.pi * (4 / 15) ** (2 / 3) / (4 * np.pi / 3) ** (2 / 3), rel=1e-14)
    fields, _ = microfield_quadrature(1e22, 5.0, num_points=11, max_beta=10.0, microfield_model="holtsmark")
    np.testing.assert_allclose(fields, np.linspace(0, 10, 11) * F_H, rtol=1e-14)
    # both the closed form and the Potekhin fit have W -> 4 beta^2 / 3 pi in this unit
    b = 0.02
    assert holtsmark_distribution(b) / b**2 == pytest.approx(4 / (3 * np.pi), rel=1e-3)
    assert potekhin_distribution(b, 0.0, 0.0, False) / b**2 == pytest.approx(4 / (3 * np.pi), rel=1e-3)


def test_air_wavelengths_stay_vacuum_below_200_nm():
    assert vacuum_to_air_wavelength_nm(121.567) == 121.567
    # standard dry air at H-alpha: n - 1 = 2.762e-4 (Peck & Reeder 1972)
    assert 656.4610 / vacuum_to_air_wavelength_nm(656.4610) - 1 == pytest.approx(2.762e-4, rel=1e-3)
