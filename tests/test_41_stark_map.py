"""Tests for the opt-in multi-shell Stark-map utility."""

import numpy as np
import pytest

from starkzee.stark_map import StarkMap


def test_parallel_map_uses_target_mj_sector_and_normalized_highlights():
    result = StarkMap(shells=(2, 3), Z=1, A=1, B=0.0).compute(
        [0.0, 1.0e6],
        angle_deg=0.0,
        target_state=(3, 2, 2.5, 0.5),
    )

    assert result.mj == 0.5
    assert result.energies_ev.shape == (2, 8)
    assert result.highlights.shape == result.energies_ev.shape
    np.testing.assert_allclose(result.highlights.sum(axis=1), 1.0, atol=2e-15)
    assert np.all(np.isfinite(result.dominant_target_energy_ev))


def test_zero_field_spectrum_is_rotation_invariant_without_mj_restriction():
    stark_map = StarkMap(shells=(2,), Z=1, A=1, B=0.0)
    parallel = stark_map.compute([2.0e6], angle_deg=0.0)
    perpendicular = stark_map.compute([2.0e6], angle_deg=90.0)

    np.testing.assert_allclose(
        parallel.energies_ev, perpendicular.energies_ev,
        rtol=0.0, atol=2e-13,
    )


def test_crossed_field_rejects_fixed_mj_sector():
    stark_map = StarkMap(shells=(2,), Z=1, A=1, B=1.0)
    with pytest.raises(ValueError, match="fixed-mj"):
        stark_map.compute(
            [0.0, 1.0e5], angle_deg=45.0,
            target_state=(2, 1, 1.5, 0.5), mj=0.5,
        )


def test_negative_magnetic_field_magnitude_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        StarkMap(shells=(2,), B=-1.0)


def test_shifted_energy_units_and_reference():
    result = StarkMap(shells=(2,), Z=1, A=1).compute(
        [0.0], target_state=(2, 1, 1.5, 0.5)
    )
    energy_ev = result.shifted_energies(unit="eV")
    energy_mev = result.shifted_energies(unit="meV")
    np.testing.assert_allclose(energy_mev, 1.0e3 * energy_ev)

    with pytest.raises(ValueError, match="unit"):
        result.shifted_energies(unit="joule")
