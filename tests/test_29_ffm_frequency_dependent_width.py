"""Regression tests for the optional per-SDT FFM electron width."""

import numpy as np

import starkzee.ffm as ffm


def _kwargs():
    return dict(
        n_u=3, n_l=2, Z=1, B=1.0, Ne_m3=1e22, Te_ev=2.0,
        Ti_ev=1.0, A_ion=1.0, energies_ev=np.linspace(1.86, 1.92, 80),
        num_f=3, num_mu=2, apply_doppler=False,
        microfield_model='holtsmark',
    )


def test_sdt_frequency_dependent_width_is_off_by_default():
    default = ffm.calculate_ffm_profile(**_kwargs())
    explicit = ffm.calculate_ffm_profile(
        **_kwargs(), sdt_frequency_dependent_width=False)
    for actual, expected in zip(default, explicit):
        np.testing.assert_array_equal(actual, expected)


def test_sdt_frequency_dependent_width_uses_sdt_detunings(monkeypatch):
    seen = []

    def varying_width(detuning, *args, **kwargs):
        values = np.asarray(detuning, dtype=float)
        seen.append(values.copy())
        result = 2e-5 + 0.2 * np.abs(values)
        return float(result) if result.ndim == 0 else result

    monkeypatch.setattr(ffm, 'electron_impact_width_model', varying_width)
    constant = ffm.calculate_ffm_profile(**_kwargs())
    seen.clear()
    varying = ffm.calculate_ffm_profile(
        **_kwargs(), sdt_frequency_dependent_width=True)

    assert seen
    assert any(values.ndim == 1 and np.any(np.abs(values) > 0) for values in seen)
    for profile in varying:
        assert np.all(np.isfinite(profile))
        assert np.all(profile >= 0)
    assert any(not np.allclose(a, b) for a, b in zip(constant, varying))


def test_sdt_width_vector_matches_full_inversion():
    kwargs = _kwargs()
    kwargs.update(energies_ev=np.linspace(1.88, 1.90, 30), num_f=2, num_mu=2,
                  sdt_frequency_dependent_width=True)
    analytical = ffm.calculate_ffm_profile(**kwargs, numerical_inversion=False)
    numerical = ffm.calculate_ffm_profile(**kwargs, numerical_inversion=True)
    for actual, expected in zip(analytical, numerical):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
