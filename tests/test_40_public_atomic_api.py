"""Public atomic helpers and LineProfile convenience API."""

import numpy as np
import pytest

from starkzee import atomic
from starkzee.line_profile import LineProfile
from starkzee.radiator import einstein_a as radiator_einstein_a


def test_atomic_facade_matches_existing_field_free_implementation():
    radii = np.linspace(0.0, 20.0, 41)
    lp = LineProfile(3, 2, B=2.0, Ne_m3=1e20, Te_ev=1.0, species='D')

    np.testing.assert_allclose(
        lp.radial_wavefunction(radii, l=1, shell='upper'),
        atomic.radial_wavefunction(radii, 3, 1, Z=1),
    )
    assert lp.radial_dipole(1, 0) == pytest.approx(
        atomic.radial_dipole(3, 1, 2, 0, Z=1), rel=0.0, abs=0.0)
    assert lp.field_free_einstein_a_s == pytest.approx(
        radiator_einstein_a(3, 2, 1, A=2), rel=0.0, abs=0.0)


def test_line_profile_radial_wavefunction_rejects_unknown_shell():
    lp = LineProfile(3, 2, B=0.0, Ne_m3=1e20, Te_ev=1.0)
    with pytest.raises(ValueError, match="upper.*lower"):
        lp.radial_wavefunction(1.0, l=0, shell='middle')


def test_einstein_a_from_strength_broadcasts_and_validates():
    rates = atomic.einstein_a_from_strength(
        np.array([1.0, 2.0]), np.array([3.0, 3.0]))
    assert rates.shape == (2,)
    assert rates[1] == pytest.approx(8.0 * rates[0])
    assert atomic.einstein_a_from_strength(0.0, 2.0) == 0.0

    with pytest.raises(ValueError, match="non-negative"):
        atomic.einstein_a_from_strength(-1.0, 2.0)
    with pytest.raises(ValueError, match="non-negative"):
        atomic.einstein_a_from_strength(1.0, -2.0)


def test_discrete_dressed_rates_recover_field_free_shell_average():
    lp = LineProfile(3, 2, B=0.0, Ne_m3=1e20, Te_ev=1.0)
    lp.compute_discrete(
        Fz=0.0, Fx=0.0, fine_structure=False,
        quadratic_zeeman=False, use_empirical_data=False)

    assert lp.discrete._einstein_a_s is None
    assert lp.discrete._upper_partial_decay_rate_s is None
    assert lp.discrete.einstein_a_s.shape == lp.discrete.energy_ev.shape
    assert np.all(lp.discrete.einstein_a_s >= 0.0)
    assert lp.discrete.upper_partial_decay_rate_s.shape == (2 * lp.n_u**2,)
    assert np.mean(lp.discrete.upper_partial_decay_rate_s) == pytest.approx(
        lp.field_free_einstein_a_s, rel=2e-5)
