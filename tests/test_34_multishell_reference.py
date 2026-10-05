"""Tests for the standalone multi-shell configuration-interaction reference."""

import json
from pathlib import Path

import numpy as np
import pytest

from starkzee.multishell import (
    build_multishell_basis,
    build_multishell_diamagnetic_matrix,
    build_multishell_dipole_matrices,
    build_multishell_hamiltonian,
    build_multishell_stark_matrix,
    diagonalize_multishell_hamiltonian,
    multishell_convergence_report,
    multishell_transition_diagnostics,
    oscillator_strength_breakdown,
    radial_r2_between_shells,
    transition_energy_mask,
    transition_strength_breakdown,
)
from starkzee.radiator import (
    build_hamiltonian, line_strength, oscillator_strength, radial_r2_element,
)
from starkzee.static_profile import build_stark_matrix
from starkzee.utils import RYDBERG_EV, reduced_mass_rydberg_ev
from starkzee.microfield import calculate_normal_field


_CASES_PATH = (Path(__file__).parents[1] / "starkzee" / "data" /
               "multishell_validation_cases.json")


def test_multishell_basis_has_global_indices_and_shell_order():
    basis = build_multishell_basis((2, 3, 4))
    assert len(basis) == 2 * (2**2 + 3**2 + 4**2)
    assert [state.index for state in basis] == list(range(len(basis)))
    assert [state.n for state in basis[:8]] == [2] * 8
    assert [state.n for state in basis[-32:]] == [4] * 32


@pytest.mark.parametrize("shells", [(), (2, 2), (3, 2), (0, 2), (1.5, 2)])
def test_multishell_basis_rejects_ambiguous_shell_sets(shells):
    with pytest.raises(ValueError, match="strictly increasing"):
        build_multishell_basis(shells)


@pytest.mark.parametrize("Z", [0, -1, 1.5])
def test_multishell_operators_reject_invalid_nuclear_charge(Z):
    with pytest.raises(ValueError, match="positive integer"):
        build_multishell_stark_matrix((2, 3), Z, 0.0, 0.0)


@pytest.mark.parametrize("states", [
    (2, 0, 3, 0, 1),
    (2, 1, 3, 1, 1),
    (3, 0, 4, 2, 1),
    (3, 1, 5, 3, 2),
])
def test_inter_shell_r2_is_symmetric_and_scales_as_inverse_z_squared(states):
    n1, l1, n2, l2, Z = states
    forward = radial_r2_between_shells(n1, l1, n2, l2, Z)
    reverse = radial_r2_between_shells(n2, l2, n1, l1, Z)
    assert forward == pytest.approx(reverse, rel=2e-13, abs=1e-12)
    at_one = radial_r2_between_shells(n1, l1, n2, l2, 1)
    assert forward == pytest.approx(at_one / Z**2, rel=2e-12, abs=1e-12)


def test_general_r2_reuses_same_shell_result():
    assert radial_r2_between_shells(4, 1, 4, 3, 1) == pytest.approx(
        radial_r2_element(4, 1, 3, 1), rel=0.0, abs=0.0)


def test_zero_field_multishell_hamiltonian_is_exact_block_diagonal():
    shells = (2, 3)
    combined = build_multishell_hamiltonian(
        shells, 1, B=0.0, Fz=0.0, Fx=0.0,
        quadratic_zeeman=False, fine_structure=True)
    split = 2 * shells[0]**2
    expected = np.zeros_like(combined)
    expected[:split, :split] = build_hamiltonian(
        2, 1, 0.0, quadratic_zeeman=False, fine_structure=True)
    expected[split:, split:] = build_hamiltonian(
        3, 1, 0.0, quadratic_zeeman=False, fine_structure=True)
    np.testing.assert_array_equal(combined, expected)


@pytest.mark.parametrize("Fz,Fx", [(2e7, 0.0), (0.0, 2e7), (2e7, -3e7)])
def test_multishell_stark_matrix_is_hermitian_and_has_inter_shell_coupling(Fz, Fx):
    basis = build_multishell_basis((2, 3))
    matrix = build_multishell_stark_matrix((2, 3), 1, Fz, Fx)
    np.testing.assert_allclose(matrix, matrix.conj().T, rtol=0.0, atol=1e-15)
    inter = np.array([[matrix[i, j] for j, ket in enumerate(basis) if ket.n == 3]
                      for i, bra in enumerate(basis) if bra.n == 2])
    assert np.max(np.abs(inter)) > 0.0


def test_single_shell_stark_spectrum_matches_production_convention():
    reference = build_multishell_stark_matrix((3,), 1, 2e7, -3e7)
    production = build_stark_matrix(3, 1, 2e7, -3e7)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(reference), np.linalg.eigvalsh(production),
        rtol=0.0, atol=2e-17)


def test_single_shell_full_spectrum_matches_production_hamiltonian():
    reference = build_multishell_hamiltonian(
        (3,), 1, B=500.0, Fz=2e7, Fx=-3e7,
        quadratic_zeeman=True, fine_structure=True)
    production = (build_hamiltonian(
        3, 1, 500.0, quadratic_zeeman=True, fine_structure=True)
        + build_stark_matrix(3, 1, 2e7, -3e7))
    np.testing.assert_allclose(
        np.linalg.eigvalsh(reference), np.linalg.eigvalsh(production),
        rtol=0.0, atol=2e-15)


def test_diamagnetic_matrix_is_hermitian_positive_and_inter_shell():
    basis = build_multishell_basis((2, 3, 4))
    matrix = build_multishell_diamagnetic_matrix((2, 3, 4), 1, 500.0)
    np.testing.assert_allclose(matrix, matrix.conj().T, rtol=0.0, atol=1e-15)
    assert np.min(np.linalg.eigvalsh(matrix)) >= -1e-14
    inter_values = [matrix[i, j] for i, bra in enumerate(basis)
                    for j, ket in enumerate(basis) if bra.n != ket.n]
    assert np.max(np.abs(inter_values)) > 0.0


def test_multishell_field_scaling_and_full_hermiticity():
    qz_200 = build_multishell_diamagnetic_matrix((2, 3), 1, 200.0)
    qz_400 = build_multishell_diamagnetic_matrix((2, 3), 1, 400.0)
    np.testing.assert_allclose(qz_400, 4.0 * qz_200, rtol=2e-15, atol=1e-18)

    stark_1 = build_multishell_stark_matrix((2, 3), 1, 1e7, -2e7)
    stark_2 = build_multishell_stark_matrix((2, 3), 1, 2e7, -4e7)
    np.testing.assert_allclose(stark_2, 2.0 * stark_1, rtol=2e-15, atol=1e-18)

    full = build_multishell_hamiltonian(
        (2, 3, 4), 1, B=500.0, Fz=1e7, Fx=2e7,
        quadratic_zeeman=True, fine_structure=True)
    np.testing.assert_allclose(full, full.conj().T, rtol=0.0, atol=1e-13)


@pytest.mark.parametrize("n_u,n_l", [(2, 1), (3, 2), (4, 2)])
def test_single_pair_multishell_dipole_strength_matches_production(n_u, n_l):
    matrices = build_multishell_dipole_matrices((n_u,), (n_l,), 1)
    strength = sum(np.vdot(matrix, matrix).real
                   for matrix in matrices.values())
    assert strength == pytest.approx(line_strength(n_u, n_l, 1), rel=2e-15)


def test_dipole_strength_is_invariant_under_ci_basis_rotation():
    upper_energies, V_u = diagonalize_multishell_hamiltonian(
        (3, 4), 1, B=500.0, Fz=1e7, Fx=2e7)
    lower_energies, V_l = diagonalize_multishell_hamiltonian(
        (1, 2), 1, B=500.0, Fz=1e7, Fx=2e7)
    assert len(upper_energies) == V_u.shape[0]
    assert len(lower_energies) == V_l.shape[0]
    original = build_multishell_dipole_matrices((3, 4), (1, 2), 1)
    rotated = build_multishell_dipole_matrices(
        (3, 4), (1, 2), 1, V_u, V_l)
    for q in (0, 1, -1):
        assert np.vdot(rotated[q], rotated[q]).real == pytest.approx(
            np.vdot(original[q], original[q]).real, rel=3e-15)


def test_strength_breakdown_closes_globally_and_in_spectral_window():
    upper_energies, V_u = diagonalize_multishell_hamiltonian(
        (3, 4, 5), 1, B=500.0, Fz=1e7, Fx=2e7)
    lower_energies, V_l = diagonalize_multishell_hamiltonian(
        (2,), 1, B=500.0, Fz=1e7, Fx=2e7)
    total = build_multishell_dipole_matrices(
        (3, 4, 5), (2,), 1, V_u, V_l)
    target = build_multishell_dipole_matrices(
        (3, 4, 5), (2,), 1, V_u, V_l, shell_pair=(4, 2))

    global_result = transition_strength_breakdown(total, target)
    assert global_result["target"] == pytest.approx(
        line_strength(4, 2, 1), rel=3e-15)
    assert global_result["neighboring"] > 0.0
    assert abs(global_result["interference"]) < 1e-12
    assert abs(global_result["closure_error"]) < 1e-12

    center = reduced_mass_rydberg_ev(1, 1) * (1 / 2**2 - 1 / 4**2)
    mask = transition_energy_mask(
        upper_energies, lower_energies, center, half_width_ev=0.2)
    window_result = transition_strength_breakdown(total, target, mask)
    assert window_result["total"] > 0.0
    assert abs(window_result["closure_error"]) < 1e-12
    assert window_result["target"] <= global_result["target"]

    gf_result = oscillator_strength_breakdown(
        total, target, upper_energies, lower_energies, mask)
    assert gf_result["total"] > 0.0
    assert abs(gf_result["closure_error"]) < 1e-12


@pytest.mark.parametrize("n_u,n_l", [(2, 1), (3, 2), (4, 2)])
def test_single_pair_oscillator_strength_matches_production(n_u, n_l):
    matrices = build_multishell_dipole_matrices((n_u,), (n_l,), 1)
    upper = np.full(2 * n_u**2, -RYDBERG_EV / n_u**2)
    lower = np.full(2 * n_l**2, -RYDBERG_EV / n_l**2)
    result = oscillator_strength_breakdown(
        matrices, matrices, upper, lower)
    assert result["total"] == pytest.approx(
        oscillator_strength(n_u, n_l, 1), rel=2e-15)
    assert result["neighboring"] == 0.0
    assert result["interference"] == 0.0


def test_transition_energy_mask_excludes_negative_and_outside_transitions():
    mask = transition_energy_mask(
        np.array([1.0, 3.0]), np.array([0.0, 2.0]),
        center_ev=1.0, half_width_ev=0.01)
    np.testing.assert_array_equal(mask, [[True, False], [False, True]])


def test_transition_diagnostic_rejects_overlapping_radiating_spaces():
    with pytest.raises(ValueError, match="lower_n not in shells"):
        multishell_transition_diagnostics(
            (2, 3, 4), target_n=4, lower_n=2, Z=1, B=500.0,
            field=1e7, angle_deg=0.0, half_width_ev=0.2)


def test_convergence_report_uses_maximal_disjoint_basis_per_angle():
    report = multishell_convergence_report(
        target_n=4, lower_n=2, Z=1, B=500.0, field=1e7,
        half_width_ev=0.2, max_lower_padding=3, max_upper_padding=1,
        angles_deg=(0.0, 90.0), energy_tolerance_ev=1e-12,
        gf_relative_tolerance=1e-12)
    assert report["case"]["effective_max_lower_padding"] == 1
    assert len(report["rows"]) == 2 * 2 * 2
    references = [
        row for row in report["rows"]
        if row["lower_padding"] == 1 and row["upper_padding"] == 1
    ]
    assert len(references) == 2
    assert all(row["convergence"]["passed"] for row in references)
    assert all(2 not in row["shells"] for row in report["rows"])
    assert any(not row["convergence"]["passed"] for row in report["rows"])


@pytest.mark.parametrize("case_index", range(5))
def test_registered_hbeta_cutoff_meets_declared_case_specific_tolerances(
        case_index):
    registry = json.loads(_CASES_PATH.read_text(encoding="utf-8"))
    assert len(registry["cases"]) == 5
    case = registry["cases"][case_index]
    normal_field, _ = calculate_normal_field(case["Ne_m3"])
    field = case["beta_F0"] * normal_field
    assert field == pytest.approx(case["F_V_per_m"], rel=1e-15)
    reference = case["reference_padding"]
    tolerances = case["tolerances"]
    report = multishell_convergence_report(
        target_n=case["target_n"], lower_n=case["lower_n"], Z=case["Z"],
        B=case["B_T"], field=field, half_width_ev=case["half_window_ev"],
        max_lower_padding=reference["lower"],
        max_upper_padding=reference["upper"],
        angles_deg=case["angles_deg"],
        energy_tolerance_ev=tolerances["tracked_energy_abs_ev"],
        gf_relative_tolerance=tolerances["gf_target_and_total_relative"],
    )
    passing = case["minimum_tested_passing_padding"]
    selected = [
        row for row in report["rows"]
        if (row["lower_padding"] == passing["lower"]
            and row["upper_padding"] == passing["upper"])
    ]
    assert len(selected) == len(case["angles_deg"])
    assert all(row["shells"] == passing["shells"] for row in selected)
    assert all(row["convergence"]["passed"] for row in selected)

    smaller = [
        row for row in report["rows"]
        if (row["lower_padding"] < passing["lower"]
            or row["upper_padding"] < passing["upper"])
    ]
    if smaller:
        assert not all(row["convergence"]["passed"] for row in smaller)
