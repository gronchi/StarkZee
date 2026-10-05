"""Regression tests for interpolation of the sparse Stehle common grid."""

import numpy as np
import pytest
from scipy.constants import c as C, e as E, k as K

from starkzee.models.stehle_impl import (
    _AVAILABLE_TRANSITIONS,
    _NC,
    _axis_weights,
    _compute_stehle_stark,
    _fintrp_grid,
    _interpolate_grid,
    _log_log_grid,
    stehle,
)


def test_fintrp_grid_reproduces_nodes_and_holtsmark_wing():
    detuning = np.array([0.0, 1.0, 2.0, 4.0])
    profile = np.array([1.0, 0.5, 0.2, 0.05])
    targets = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0])
    wing_constant = 3.0

    result = _fintrp_grid(
        detuning, profile, targets, wing_constant=wing_constant
    )

    assert result[:4] == pytest.approx(profile)
    assert result[4] == pytest.approx(wing_constant / 8.0**2.5)
    assert result[5] / result[4] == pytest.approx(2.0**-2.5)


def test_log_log_grid_reproduces_nodes_and_holtsmark_wing():
    detuning = np.array([0.0, 1.0, 2.0, 4.0])
    profile = np.array([1.0, 0.5, 0.2, 0.05])
    targets = np.array([0.0, 0.5, 1.0, np.sqrt(2.0), 2.0, 4.0, 8.0])

    result = _log_log_grid(
        detuning, profile, targets, wing_constant=3.0
    )

    assert result[[0, 2, 4, 5]] == pytest.approx(profile)
    assert result[1] == pytest.approx(0.75)
    assert result[3] == pytest.approx(np.sqrt(0.5 * 0.2))
    assert result[6] == pytest.approx(3.0 / 8.0**2.5)


def test_halpha_has_no_linear_interpolation_shelf():
    center_nm = 656.279
    wavelength_nm = np.linspace(center_nm - 1.5, center_nm + 1.5, 6001)
    temperature_ev = 5000.0 / 11604.518121550082
    profile = stehle(
        wavelength_nm,
        3,
        2,
        0.0,
        1.0e19,
        temperature_ev,
        temperature_ev,
        species='H',
    )
    profile /= np.max(profile)

    center = len(wavelength_nm) // 2
    offset_02 = center + int(round(0.2 / (wavelength_nm[1] - wavelength_nm[0])))
    offset_05 = center + int(round(0.5 / (wavelength_nm[1] - wavelength_nm[0])))
    assert profile[offset_05] < 0.2 * profile[offset_02]


def test_log_log_is_default_and_fintrp_remains_available():
    center_nm = 656.279
    wavelength_nm = np.linspace(center_nm - 1.5, center_nm + 1.5, 1001)
    temperature_ev = 5000.0 / 11604.518121550082
    arguments = (
        wavelength_nm, 3, 2, 0.0, 1.0e19, temperature_ev, temperature_ev
    )

    default = stehle(*arguments, species='H')
    log_log = stehle(*arguments, species='H', interpolation='log-log')
    fintrp = stehle(*arguments, species='H', interpolation='fintrp')

    assert default == pytest.approx(log_log, rel=0.0, abs=0.0)
    assert not np.allclose(log_log, fintrp, rtol=1.0e-6, atol=0.0)


def test_unknown_interpolation_is_rejected():
    wavelength_nm = np.linspace(655.0, 657.0, 101)
    temperature_ev = 5000.0 / 11604.518121550082

    with pytest.raises(ValueError, match="interpolation"):
        stehle(
            wavelength_nm, 3, 2, 0.0, 1.0e19,
            temperature_ev, temperature_ev, interpolation='linear',
        )


def test_bundled_transition_inventory_is_complete_and_explicit():
    expected = {
        (upper, lower)
        for lower in (1, 2, 3)
        for upper in range(lower + 1, 31)
    }

    assert len(_AVAILABLE_TRANSITIONS) == 84
    assert set(_AVAILABLE_TRANSITIONS) == expected


def test_bundled_table_has_no_embedded_provenance_metadata():
    assert _NC._attributes == {}
    assert all(variable._attributes == {} for variable in _NC.variables.values())


def test_all_transition_coordinate_metadata_are_consistent():
    expected_temperatures_k = np.array(
        [2500, 5000, 10000, 19950, 39810, 79430,
         158500, 316200, 631000, 1259600],
        dtype=float,
    )
    for upper, lower in _AVAILABLE_TRANSITIONS:
        prefix = f'n_{upper}_{lower}_'
        temperatures_k = np.asarray(_NC.variables[prefix + 'tempe'].data)
        densities_cm3 = np.asarray(_NC.variables[prefix + 'dense'].data)
        final_density_index = int(
            np.asarray(_NC.variables[prefix + 'id_max'].data).item()
        )
        density_allocation = int(
            np.asarray(_NC.variables[prefix + 'id_maxi'].data).item()
        )

        assert temperatures_k == pytest.approx(expected_temperatures_k)
        active_density_count = final_density_index + 1
        active_densities_cm3 = densities_cm3[:active_density_count]
        assert np.all(np.diff(active_densities_cm3) > 0.0)
        assert density_allocation == len(densities_cm3)
        assert np.all(densities_cm3[active_density_count:] == 0.0)
        assert _NC.variables[prefix + 'jtot'].data.shape == (
            len(densities_cm3), len(expected_temperatures_k)
        )


def test_axis_weights_recover_endpoints_and_do_not_extrapolate():
    nodes = np.array([1.0, 2.0, 4.0])

    assert _axis_weights(nodes, 1.0, 'test') == ((0, 1.0),)
    assert _axis_weights(nodes, 4.0, 'test') == ((2, 1.0),)
    assert _axis_weights(nodes, 4.0 * (1.0 + 5.0e-13), 'test') == ((2, 1.0),)
    weights = _axis_weights(nodes, 3.0, 'test')
    assert tuple(index for index, _ in weights) == (1, 2)
    assert tuple(weight for _, weight in weights) == pytest.approx((0.5, 0.5))
    with pytest.raises(ValueError, match='outside table range'):
        _axis_weights(nodes, 0.99, 'test')
    with pytest.raises(ValueError, match='outside table range'):
        _axis_weights(nodes, 4.01, 'test')


def test_exact_density_temperature_node_recovers_source_profile():
    prefix = 'n_3_2_'
    density_index = 4
    temperature_index = 4
    density_cm3 = float(_NC.variables[prefix + 'dense'].data[density_index])
    temperature_k = float(
        _NC.variables[prefix + 'tempe'].data[temperature_index]
    )
    point_count = int(
        _NC.variables[prefix + 'jtot'].data[
            density_index, temperature_index
        ]
    )
    center_m = 656.279e-9
    center_hz = C / center_m
    frequency_hz = center_hz + np.linspace(-2.0e12, 2.0e12, 101)

    actual = _compute_stehle_stark(
        3,
        2,
        density_cm3 * 1.0e6,
        temperature_k * K / E,
        center_m,
        frequency_hz,
    )

    normal_hf = 1.25e-9 * density_cm3 ** (2.0 / 3.0)
    requested_dom = 2.0 * np.pi * np.abs(frequency_hz - center_hz) / normal_hf
    wavelength_angstrom = center_m * 1.0e10
    otrans = -2.0 * np.pi * (C * 1.0e10) / wavelength_angstrom**2
    source_detuning = _NC.variables[prefix + 'dom'].data[
        density_index, temperature_index, :point_count
    ]
    source_profile = _NC.variables[prefix + 'o1lines'].data[
        density_index, temperature_index, :point_count
    ] / abs(otrans)
    fainom = float(np.asarray(_NC.variables[prefix + 'fainom'].data).item())
    expected = _interpolate_grid(
        source_detuning,
        source_profile,
        requested_dom,
        fainom,
        'log-log',
    ) * (2.0 * np.pi / normal_hf)

    assert actual == pytest.approx(expected, rel=2.0e-14, abs=0.0)


@pytest.mark.parametrize('density_index, temperature_index', [(0, 0), (-1, -1)])
def test_halpha_table_endpoints_are_accepted(density_index, temperature_index):
    prefix = 'n_3_2_'
    density_cm3 = float(_NC.variables[prefix + 'dense'].data[density_index])
    temperature_k = float(
        _NC.variables[prefix + 'tempe'].data[temperature_index]
    )
    center_m = 656.279e-9
    center_hz = C / center_m

    profile = _compute_stehle_stark(
        3,
        2,
        density_cm3 * 1.0e6,
        temperature_k * K / E,
        center_m,
        center_hz + np.linspace(-1.0e12, 1.0e12, 31),
    )

    assert np.all(np.isfinite(profile))
    assert np.all(profile > 0.0)


def test_padded_density_coordinate_uses_only_active_nodes():
    prefix = 'n_4_1_'
    final_density_index = int(
        np.asarray(_NC.variables[prefix + 'id_max'].data).item()
    )
    density_cm3 = float(
        _NC.variables[prefix + 'dense'].data[final_density_index]
    )
    temperature_k = float(_NC.variables[prefix + 'tempe'].data[-1])
    center_m = 100.0e-9
    center_hz = C / center_m

    profile = _compute_stehle_stark(
        4,
        1,
        density_cm3 * 1.0e6,
        temperature_k * K / E,
        center_m,
        center_hz + np.linspace(-1.0e12, 1.0e12, 31),
    )

    assert np.all(np.isfinite(profile))
    assert np.all(profile > 0.0)


def test_stehle_is_grid_reversal_invariant_and_zero_ti_is_supported():
    wavelength_nm = np.linspace(655.0, 657.5, 801)
    temperature_ev = 5000.0 / 11604.518121550082
    arguments = (3, 2, 0.0, 1.0e19, temperature_ev, 0.0)

    increasing = stehle(wavelength_nm, *arguments, species='H')
    decreasing = stehle(wavelength_nm[::-1], *arguments, species='H')

    assert decreasing[::-1] == pytest.approx(increasing, rel=2.0e-13, abs=0.0)
    assert np.trapezoid(increasing, wavelength_nm * 1.0e-9) == pytest.approx(1.0)


@pytest.mark.parametrize(
    'wavelength_nm, message',
    [
        (np.array([655.0, 656.0, 655.5]), 'strictly monotonic'),
        (np.array([655.0, np.nan]), 'finite positive'),
    ],
)
def test_stehle_rejects_invalid_wavelength_grids(wavelength_nm, message):
    temperature_ev = 5000.0 / 11604.518121550082
    with pytest.raises(ValueError, match=message):
        stehle(
            wavelength_nm, 3, 2, 0.0, 1.0e19,
            temperature_ev, temperature_ev,
        )


def test_stehle_rejects_transition_absent_from_table():
    wavelength_nm = np.linspace(655.0, 657.0, 101)
    temperature_ev = 5000.0 / 11604.518121550082

    with pytest.raises(ValueError, match='not in the bundled table'):
        stehle(
            wavelength_nm, 5, 4, 0.0, 1.0e19,
            temperature_ev, temperature_ev,
        )
