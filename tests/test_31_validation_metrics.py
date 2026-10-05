"""Tests for convergence-report metrics."""

import numpy as np
import pytest

from scripts.profile_convergence import _matched_solver_report, _solver_report
from starkzee.validation import (
    compare_profiles,
    coordinate_jacobian_metrics,
    profile_metrics,
)


def _profiles(grid):
    base = np.exp(-((grid - 2.0) / 0.1)**2)
    return np.vstack((base, 0.5 * base, 0.25 * base))


def test_metrics_are_reversal_invariant():
    grid = np.linspace(1.5, 2.5, 501)
    forward = profile_metrics(grid, _profiles(grid))
    reverse = profile_metrics(grid[::-1], _profiles(grid)[..., ::-1])
    assert forward == reverse
    assert forward["centroid_ev"] == pytest.approx(2.0, abs=1e-14)


def test_identical_profiles_have_zero_comparison_error():
    grid = np.linspace(1.5, 2.5, 501)
    comparison = compare_profiles(grid, _profiles(grid), grid, _profiles(grid))
    assert comparison["area_relative_error"] == pytest.approx(0.0, abs=1e-15)
    assert comparison["centroid_delta_ev"] == pytest.approx(0.0, abs=1e-15)
    assert comparison["shape_l1_relative_on_overlap"] == pytest.approx(0.0, abs=1e-15)
    assert comparison["shape_linf_relative_on_overlap"] == pytest.approx(0.0, abs=1e-15)
    assert comparison["reference_overlap_fraction"] == pytest.approx(1.0)


def test_selected_core_and_wing_intensities_are_reported():
    grid = np.linspace(1.5, 2.5, 501)
    samples = np.array([1.8, 2.0, 2.2])
    comparison = compare_profiles(
        grid, _profiles(grid), grid, 1.1 * _profiles(grid),
        sample_energies_ev=samples)

    assert [item["energy_ev"] for item in comparison["sampled_intensities"]] == (
        pytest.approx(samples)
    )
    for item in comparison["sampled_intensities"]:
        assert item["relative_error"] == pytest.approx(0.1)
        assert item["candidate_components"] == pytest.approx(
            1.1 * np.asarray(item["reference_components"])
        )


def test_sampled_intensities_are_reversal_invariant():
    grid = np.linspace(1.5, 2.5, 501)
    forward = compare_profiles(
        grid, _profiles(grid), grid, _profiles(grid),
        sample_energies_ev=[1.8, 2.0, 2.2])
    reverse = compare_profiles(
        grid[::-1], _profiles(grid)[:, ::-1],
        grid[::-1], _profiles(grid)[:, ::-1],
        sample_energies_ev=[1.8, 2.0, 2.2])

    assert reverse["sampled_intensities"] == forward["sampled_intensities"]


def test_sampled_intensities_must_lie_in_overlap():
    reference_grid = np.linspace(1.5, 2.5, 501)
    candidate_grid = np.linspace(1.8, 2.2, 201)
    with pytest.raises(ValueError, match="profile overlap"):
        compare_profiles(
            reference_grid, _profiles(reference_grid),
            candidate_grid, _profiles(candidate_grid),
            sample_energies_ev=[1.7, 2.0])


def test_different_grid_is_interpolated_and_window_loss_is_reported():
    reference_grid = np.linspace(1.5, 2.5, 1001)
    candidate_grid = np.linspace(1.9, 2.1, 101)
    comparison = compare_profiles(
        reference_grid, _profiles(reference_grid),
        candidate_grid, _profiles(candidate_grid))
    assert comparison["reference_overlap_fraction"] < 1.0
    assert comparison["reference_overlap_fraction"] > 0.8
    assert comparison["area_relative_error"] < 0.0
    assert comparison["shape_l1_relative_on_overlap"] < 2e-4


def test_coordinate_jacobians_preserve_component_and_total_areas():
    grid = np.linspace(1.5, 2.5, 2001)
    metrics = coordinate_jacobian_metrics(grid, _profiles(grid))

    assert metrics["wavelength_nm"]["coordinate_order_from_energy"] == "decreasing"
    for coordinate in ("wavelength_nm", "frequency_thz", "wavenumber_cm"):
        transformed = metrics[coordinate]
        assert transformed["area_relative_error"] == pytest.approx(0.0, abs=2e-7)
        assert transformed["component_relative_errors"] == pytest.approx(
            [0.0, 0.0, 0.0], abs=2e-7)


def test_coordinate_jacobian_metrics_are_reversal_invariant():
    grid = np.linspace(1.5, 2.5, 501)
    forward = coordinate_jacobian_metrics(grid, _profiles(grid))
    reverse = coordinate_jacobian_metrics(
        grid[::-1], _profiles(grid)[:, ::-1])

    assert reverse == forward


def test_coordinate_jacobians_reject_nonpositive_photon_energy():
    grid = np.linspace(-1.0, 1.0, 101)
    with pytest.raises(ValueError, match="positive photon energies"):
        coordinate_jacobian_metrics(grid, _profiles(grid))


@pytest.mark.parametrize("grid,profiles,match", [
    (np.array([1.0]), np.zeros((3, 1)), "at least two"),
    (np.array([1.0, 1.0]), np.zeros((3, 2)), "strictly monotonic"),
    (np.array([1.0, 2.0]), np.zeros((2, 2)), "pi, sigma-plus"),
])
def test_invalid_inputs_are_rejected(grid, profiles, match):
    with pytest.raises(ValueError, match=match):
        profile_metrics(grid, profiles)


def test_convergence_report_includes_supported_doppler_path():
    report = _solver_report("static", quick=True)
    doppler = report["solver_doppler_effect"]

    assert report["reference_settings"]["apply_doppler"] is False
    assert [point["label"] for point in report["sample_points"]] == [
        "blue_wing", "blue_core", "line_center", "red_core", "red_wing"
    ]
    assert doppler["settings"]["apply_doppler"] is True
    assert doppler["comparison"]["shape_l1_relative_on_overlap"] > 0.0
    assert "not a numerical convergence error" in doppler["scope"]


@pytest.mark.parametrize("apply_doppler", [False, True])
def test_matched_static_ffm_report_uses_same_exposed_controls(apply_doppler):
    report = _matched_solver_report(quick=True, apply_doppler=apply_doppler)

    assert report["settings"]["apply_doppler"] is apply_doppler
    assert report["comparison"]["reference_overlap_fraction"] == pytest.approx(1.0)
    assert "physical FFM ion-dynamics" in report["scope"]

