"""Regression tests for force_frequency_dependent_width (D21/C08).

Confirms that ``frequency_dependent_width=True`` is always honored and that
the legacy ``force_frequency_dependent_width`` argument is a compatibility
no-op for both frequency-dependent and constant-width calculations.
"""

import warnings

import numpy as np

from starkzee.static_profile import calculate_static_profile
from starkzee.utils import wavelength_nm_to_energy_ev


def _common(Ne=1e19, Te=5.0, Ti=5.0):
    E = wavelength_nm_to_energy_ev(np.linspace(654, 658, 2000))
    return dict(n_u=3, n_l=2, Z=1, B=1.0, Ne_m3=Ne, Te_ev=Te, energies_ev=E,
                num_f=10, num_mu=4, Ti_ev=Ti)


def test_default_honors_requested_width_without_override_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calculate_static_profile(**_common(), frequency_dependent_width=True)
    assert not any("frequency_dependent_width" in str(w.message) for w in caught)


def test_legacy_force_flag_matches_now_correct_default():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pi_f, sp_f, sm_f = calculate_static_profile(
            **_common(), frequency_dependent_width=True,
            force_frequency_dependent_width=True)
    assert not any(
        "Gaussian-accumulation path is used" in str(w.message) for w in caught)

    pi_d, sp_d, sm_d = calculate_static_profile(
        **_common(), frequency_dependent_width=True)

    prof_forced = pi_f + 0.5 * (sp_f + sm_f)
    prof_default = pi_d + 0.5 * (sp_d + sm_d)
    np.testing.assert_allclose(prof_forced, prof_default)


def test_force_flag_is_noop_when_width_not_frequency_dependent():
    """No warning, no behavior change, when frequency_dependent_width=False."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pi1, sp1, sm1 = calculate_static_profile(
            **_common(), frequency_dependent_width=False)
        pi2, sp2, sm2 = calculate_static_profile(
            **_common(), frequency_dependent_width=False,
            force_frequency_dependent_width=True)
    assert not any("frequency_dependent_width=True was requested" in str(w.message)
                   for w in caught)
    assert np.allclose(pi1, pi2) and np.allclose(sp1, sp2) and np.allclose(sm1, sm2)
