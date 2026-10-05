"""Hydrogen-like emitter/background separation and FFM rate conventions."""

import numpy as np
import pytest
from scipy.constants import e, hbar, m_p

from starkzee.ffm import calculate_ion_fluctuation_rate
from starkzee.microfield import microfield_quadrature


def _rate_from_mass(ne_m3, ti_ev, charge, mass_kg, speed_factor):
    ion_density = ne_m3 / charge
    sphere_radius = (3.0 / (4.0 * np.pi * ion_density)) ** (1.0 / 3.0)
    speed = np.sqrt(speed_factor * ti_ev * e / mass_kg)
    return speed / sphere_radius * hbar / e


def test_zest_rate_retains_historical_perturber_mass_formula():
    actual = calculate_ion_fluctuation_rate(1e23, 5.0, 1.0, 2.0)
    expected = _rate_from_mass(1e23, 5.0, 1.0, 2.0 * m_p, 2.0)
    assert actual == pytest.approx(expected, rel=1e-15)


def test_ppp_reduced_mass_rate_for_heavy_emitter_in_proton_plasma():
    actual = calculate_ion_fluctuation_rate(
        1e30, 861.733, 1.0, 1.0, A_emitter=40.0, model='ppp')
    reduced_mass = (40.0 / 41.0) * m_p
    expected = _rate_from_mass(1e30, 861.733, 1.0, reduced_mass, 1.0)
    assert actual == pytest.approx(expected, rel=1e-15)


def test_equal_mass_zest_and_ppp_rate_conventions_coincide():
    zest = calculate_ion_fluctuation_rate(1e24, 10.0, 1.0, 2.0)
    ppp = calculate_ion_fluctuation_rate(
        1e24, 10.0, 1.0, 2.0, A_emitter=2.0, model='ppp')
    assert ppp == pytest.approx(zest, rel=1e-15)


def test_ppp_rate_requires_explicit_emitter_mass():
    with pytest.raises(ValueError, match="A_emitter"):
        calculate_ion_fluctuation_rate(1e23, 1.0, 1.0, 1.0, model='ppp')


def test_emitter_charge_derives_charged_microfield_selector():
    common = dict(
        Ne_m3=1e23, Te_ev=10.0, Ti_ev=10.0, num_points=20,
        max_beta=5.0, microfield_model='potekhin')
    inferred = microfield_quadrature(**common, emitter_charge=17.0)
    explicit = microfield_quadrature(**common, charged=True)
    for actual, expected in zip(inferred, explicit):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_current_potekhin_fit_uses_charge_status_not_magnitude():
    common = dict(
        Ne_m3=1e23, Te_ev=10.0, Ti_ev=10.0, num_points=20,
        max_beta=5.0, microfield_model='potekhin')
    singly_charged = microfield_quadrature(**common, emitter_charge=1.0)
    ar17 = microfield_quadrature(**common, emitter_charge=17.0)
    for actual, expected in zip(ar17, singly_charged):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("emitter_charge", [-1.0, np.nan])
def test_invalid_emitter_charge_is_rejected(emitter_charge):
    with pytest.raises(ValueError, match="emitter_charge"):
        microfield_quadrature(
            1e23, 10.0, use_screening=False,
            emitter_charge=emitter_charge)
