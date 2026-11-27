"""FFM ion dynamics for radiator objects (hydrogenic, LS / single-valence)."""
import warnings

import numpy as np
import pytest

import starkzee.ffm as ffm
from starkzee.ffm import calculate_ffm_profile
from starkzee.radiators import HydrogenicTransition
from starkzee.single_valence import c_iv_5_4
from starkzee.static_profile import calculate_static_profile


@pytest.mark.parametrize("B", [5.0, 0.0])
def test_hydrogenic_radiator_reproduces_the_built_in_ffm(B):
    E0 = 2.5497
    E = np.linspace(E0 - 0.01, E0 + 0.01, 401)
    kw = dict(num_f=6, num_mu=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = calculate_ffm_profile(4, 2, 1, B, 1e22, 2.0, 2.0, 1.0, E, **kw)
        r = HydrogenicTransition(4, 2, 1, B, True, True, 1.0, use_empirical_data=True, atom="H")
        got = calculate_ffm_profile(None, None, None, B, 1e22, 2.0, 2.0, 1.0, E, radiator=r, **kw)
    for a, b in zip(ref, got):
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-10 * a.max())


@pytest.mark.parametrize("B, operator", [(300.0, True), (0.0, False)])
def test_zero_fluctuation_rate_gives_the_static_profile(monkeypatch, B, operator):
    monkeypatch.setattr(ffm, "calculate_ion_fluctuation_rate", lambda *a, **k: 0.0)
    r = c_iv_5_4(B)
    E0 = r.line_energy_ev
    E = np.linspace(E0 - 0.1, E0 + 0.1, 801)
    kw = dict(num_f=6, num_mu=2, radiator=r, apply_doppler=False, electron_operator=operator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        static = calculate_static_profile(5, 4, 4, B, 2e24, 10.0, E, Ti_ev=10.0, frequency_dependent_width=False,
                                          **kw)
        dynamic = calculate_ffm_profile(None, None, None, B, 2e24, 10.0, 10.0, None, E, **kw)
    for a, b in zip(static, dynamic):
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-10 * a.max())


def test_c_iv_ion_dynamics_broadens_the_central_peak():
    # 2e18 cm^-3, 10 eV, proton perturbers: FFM fills the narrow Stark core (FWHM 1.07 -> 1.38 nm
    # with the full quadrature; Ferri et al. 2022 Fig. 6 and the measured profile: 1.38 nm)
    r = c_iv_5_4(0.0)
    E0 = r.line_energy_ev
    E = np.linspace(E0 - 0.15, E0 + 0.15, 1201)
    kw = dict(num_f=10, num_mu=2, radiator=r, electron_operator=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = calculate_static_profile(5, 4, 4, 0.0, 2e24, 10.0, E, Ti_ev=10.0, frequency_dependent_width=False, **kw)[0]
        d = calculate_ffm_profile(None, None, None, 0.0, 2e24, 10.0, 10.0, None, E, fluctuation_rate_model="ppp",
                                  **kw)[0]
    fwhm = lambda y: np.ptp(E[y >= y.max() / 2])                                  # noqa: E731
    assert np.trapezoid(d, E) == pytest.approx(np.trapezoid(s, E), rel=0.02)
    assert 1.15 < fwhm(d) / fwhm(s) < 1.45


def test_radiator_path_rejects_unsupported_options():
    r = c_iv_5_4(0.0)
    E = np.linspace(r.line_energy_ev - 0.05, r.line_energy_ev + 0.05, 11)
    with pytest.raises(NotImplementedError):
        calculate_ffm_profile(None, None, None, 0.0, 2e24, 10.0, 10.0, None, E, radiator=r,
                              sdt_frequency_dependent_width=True)
    with pytest.raises(ValueError, match="radiator"):
        calculate_ffm_profile(4, 2, 1, 0.0, 1e22, 2.0, 2.0, 1.0, E, electron_operator=True)
