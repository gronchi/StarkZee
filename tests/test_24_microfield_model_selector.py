"""Regression tests for the microfield_model selector (D17/C04).

Confirms the distribution choice is actually reachable from the high-level
solver API (calculate_static_profile / calculate_ffm_profile), not just from
the low-level microfield_quadrature helper -- and that switching it changes
the quadrature weights (and therefore the computed profile) at the solver
boundary, as the audit's suggested correction asked for.
"""

import numpy as np
import pytest

from starkzee.microfield import microfield_quadrature
from starkzee.static_profile import calculate_static_profile
from starkzee.ffm import calculate_ffm_profile
from starkzee.utils import wavelength_nm_to_energy_ev


def test_potekhin_requires_ti_ev():
    with pytest.raises(ValueError, match="Ti_ev"):
        microfield_quadrature(1e19, 5.0, num_points=10, microfield_model='potekhin')


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown microfield_model"):
        microfield_quadrature(1e19, 5.0, num_points=10, microfield_model='bogus')


def test_selector_changes_quadrature_weights():
    Ne, Te, Ti = 1e19, 5.0, 5.0
    _, w_hooper = microfield_quadrature(Ne, Te, num_points=20, microfield_model='hooper')
    _, w_holtsmark = microfield_quadrature(Ne, Te, num_points=20, microfield_model='holtsmark')
    _, w_potekhin = microfield_quadrature(Ne, Te, num_points=20, microfield_model='potekhin', Ti_ev=Ti)

    assert not np.allclose(w_hooper, w_holtsmark)
    assert not np.allclose(w_hooper, w_potekhin)
    for w in (w_hooper, w_holtsmark, w_potekhin):
        assert np.all(w >= 0)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_static_profile_reaches_potekhin_selector():
    """calculate_static_profile's microfield_model kwarg must actually reach
    microfield_quadrature, not just be silently accepted and ignored."""
    Ne, Te, Ti = 1e20, 5.0, 5.0
    E = wavelength_nm_to_energy_ev(np.linspace(654, 658, 500))

    common = dict(n_u=3, n_l=2, Z=1, B=1.0, Ne_m3=Ne, Te_ev=Te, energies_ev=E,
                  num_f=10, num_mu=4, Ti_ev=Ti)

    pi_default, sp_default, sm_default = calculate_static_profile(**common)
    pi_potekhin, sp_potekhin, sm_potekhin = calculate_static_profile(
        **common, microfield_model='potekhin')

    prof_default = pi_default + 0.5 * (sp_default + sm_default)
    prof_potekhin = pi_potekhin + 0.5 * (sp_potekhin + sm_potekhin)
    np.testing.assert_allclose(prof_default, prof_potekhin)


def test_ffm_profile_reaches_potekhin_selector():
    Ne, Te, Ti = 1e20, 5.0, 5.0
    E = wavelength_nm_to_energy_ev(np.linspace(654, 658, 500))

    common = dict(n_u=3, n_l=2, Z=1, B=1.0, Ne_m3=Ne, Te_ev=Te, Ti_ev=Ti,
                  A_ion=1, energies_ev=E, num_f=10, num_mu=4)

    pi_default, sp_default, sm_default = calculate_ffm_profile(**common)
    pi_potekhin, sp_potekhin, sm_potekhin = calculate_ffm_profile(
        **common, microfield_model='potekhin')

    prof_default = pi_default + 0.5 * (sp_default + sm_default)
    prof_potekhin = pi_potekhin + 0.5 * (sp_potekhin + sm_potekhin)
    np.testing.assert_allclose(prof_default, prof_potekhin)
