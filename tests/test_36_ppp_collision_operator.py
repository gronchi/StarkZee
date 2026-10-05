"""PPP Appendix-B collision operator and complex-SDT regressions."""

import numpy as np
import pytest

from starkzee.broadening import (
    electron_impact_collision_coefficient,
    electron_impact_width,
)
from starkzee.collision import (
    build_ppp_collision_operator,
    generalized_lorentzian,
    generalized_lorentzian_components,
    intrashell_cartesian_dipoles,
    ppp_complex_sdts,
    rephase_transition_dipoles,
)
from starkzee.ffm import (
    _complex_ffm_profile_analytical,
    calculate_ffm_profile,
    group_complex_sdts,
    summarize_complex_sdt_probabilities,
)
from starkzee.radiator import _uncoupled_dipole_matrices
from starkzee.static_profile import calculate_static_profile, solve_starkzee


def _h_alpha_system():
    energies_u, vectors_u = solve_starkzee(3, 1, 100.0, 1.0e6, 2.0e6)
    energies_l, vectors_l = solve_starkzee(2, 1, 100.0, 1.0e6, 2.0e6)
    transition_energies = energies_u[np.newaxis, :] - energies_l[:, np.newaxis]
    dipoles = _uncoupled_dipole_matrices(3, 2, 1)
    return transition_energies, vectors_u, vectors_l, dipoles


@pytest.mark.parametrize("n", [1, 2, 3])
def test_intrashell_cartesian_dipoles_are_hermitian(n):
    matrices = intrashell_cartesian_dipoles(n, 1)
    assert len(matrices) == 3
    for matrix in matrices:
        np.testing.assert_allclose(matrix, matrix.conj().T, rtol=0, atol=1e-13)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_manual_impact_coefficient_matches_intrashell_scalar_width(n):
    conditions = dict(Ne_m3=1e20, Te_ev=2.0, B=3.0, Z=1, n=n)
    coefficient = electron_impact_collision_coefficient(**conditions)
    r2_intra_average = 9.0 * n**2 * (n**2 - 1) / 8.0
    expected_width = electron_impact_width(
        0.0, **conditions, r2_form='intra')
    assert coefficient * r2_intra_average == pytest.approx(expected_width)

    # With a 1s lower manifold, Appendix B has no lower intra-shell dipoles;
    # the mean no-interference coherence width is exactly the upper width.
    phi = build_ppp_collision_operator(
        np.eye(2 * n**2), np.eye(2), n, 1, 1, coefficient,
        electron_interference=False,
    )
    assert np.trace(phi).real / phi.shape[0] == pytest.approx(expected_width)


def test_appendix_b_interference_changes_only_cross_manifold_term():
    _, vectors_u, vectors_l, _ = _h_alpha_system()
    without = build_ppp_collision_operator(
        vectors_u, vectors_l, 3, 2, 1, 2e-6,
        electron_interference=False,
    )
    with_interference = build_ppp_collision_operator(
        vectors_u, vectors_l, 3, 2, 1, 2e-6,
        electron_interference=True,
    )
    np.testing.assert_allclose(without, without.conj().T, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        with_interference, with_interference.conj().T, rtol=0, atol=1e-14)
    assert np.linalg.eigvalsh(with_interference).min() >= -1e-14
    assert np.linalg.norm(with_interference - without) > 0
    # Dipole operators are traceless, so the interference term redistributes
    # damping without changing the collision superoperator trace.
    assert np.trace(with_interference).real == pytest.approx(np.trace(without).real)


def test_complex_sdts_reproduce_direct_nonhermitian_resolvent():
    d_e, vectors_u, vectors_l, dipoles = _h_alpha_system()
    natural = np.full_like(d_e, 2e-7)
    omega, gamma, strengths = ppp_complex_sdts(
        d_e, vectors_u, vectors_l, dipoles, 3, 2, 1,
        2e-6, natural, electron_interference=True,
    )

    phi = build_ppp_collision_operator(
        vectors_u, vectors_l, 3, 2, 1, 2e-6,
        electron_interference=True,
    ) + np.diag(natural.T.ravel())
    generator = np.diag(d_e.T.ravel()) - 1j * phi
    rephased = rephase_transition_dipoles(dipoles, 3, 2)
    dressed = vectors_l.conj().T @ rephased[0] @ vectors_u
    vector = dressed.conj().T.ravel()
    grid = np.linspace(d_e.min() - 2e-4, d_e.max() + 2e-4, 17)
    direct = np.array([
        -np.imag(vector.conj() @ np.linalg.solve(
            energy * np.eye(generator.shape[0]) - generator, vector)) / np.pi
        for energy in grid
    ])
    decomposed = generalized_lorentzian(grid, omega, gamma, strengths[0])
    np.testing.assert_allclose(decomposed, direct, rtol=2e-8, atol=1e-8)
    assert strengths[0].real.sum() == pytest.approx(np.vdot(vector, vector).real)
    assert strengths[0].imag.sum() == pytest.approx(0.0, abs=1e-10)

    combined = generalized_lorentzian_components(
        grid, omega, gamma, strengths)
    for q in (0, 1, -1):
        separate = generalized_lorentzian(
            grid, omega, gamma, strengths[q])
        np.testing.assert_allclose(
            combined[q], separate, rtol=2e-14, atol=1e-8)


def test_complex_profile_is_invariant_to_dressed_state_phases_and_ordering():
    d_e, vectors_u, vectors_l, dipoles = _h_alpha_system()
    natural = np.full_like(d_e, 2e-7)
    baseline = ppp_complex_sdts(
        d_e, vectors_u, vectors_l, dipoles, 3, 2, 1,
        2e-6, natural, electron_interference=True,
    )

    rng = np.random.default_rng(1942)
    order_u = rng.permutation(vectors_u.shape[1])
    order_l = rng.permutation(vectors_l.shape[1])
    phase_u = np.exp(1j * rng.uniform(-np.pi, np.pi, vectors_u.shape[1]))
    phase_l = np.exp(1j * rng.uniform(-np.pi, np.pi, vectors_l.shape[1]))
    changed_vectors_u = vectors_u[:, order_u] * phase_u
    changed_vectors_l = vectors_l[:, order_l] * phase_l
    changed_energies = d_e[np.ix_(order_l, order_u)]
    changed_natural = natural[np.ix_(order_l, order_u)]
    changed = ppp_complex_sdts(
        changed_energies, changed_vectors_u, changed_vectors_l, dipoles,
        3, 2, 1, 2e-6, changed_natural, electron_interference=True,
    )

    grid = np.linspace(d_e.min() - 2e-4, d_e.max() + 2e-4, 19)
    for q in (0, 1, -1):
        reference = generalized_lorentzian(grid, *baseline[:2], baseline[2][q])
        actual = generalized_lorentzian(grid, *changed[:2], changed[2][q])
        np.testing.assert_allclose(actual, reference, rtol=2e-8, atol=1e-8)


def test_public_solvers_reject_incompatible_interference_options():
    common = dict(
        n_u=2, n_l=1, Z=1, B=1.0, Ne_m3=1e20, Te_ev=2.0,
        energies_ev=np.linspace(10.1, 10.3, 31), num_f=2, num_mu=2,
        microfield_model="holtsmark", apply_doppler=False,
    )
    with pytest.raises(ValueError, match="frequency_dependent_width=False"):
        calculate_static_profile(**common, electron_interference=True)

    ffm = dict(common)
    ffm.update(Ti_ev=1.0, A_ion=1.0)
    with pytest.raises(ValueError, match="sdt_bin_tol=None"):
        calculate_ffm_profile(
            **ffm, electron_interference=True, sdt_bin_tol=1e-5)
    with pytest.raises(ValueError, match="requires interference_group_tolerance_ev"):
        calculate_ffm_profile(
            **ffm, electron_interference=True,
            interference_group_width_tolerance_ev=1e-6)


def test_ffm_interference_path_is_finite_for_lyman_alpha():
    profiles = calculate_ffm_profile(
        n_u=2, n_l=1, Z=1, B=10.0, Ne_m3=1e20, Te_ev=5.0,
        Ti_ev=1.0, A_ion=1.0, energies_ev=np.linspace(10.15, 10.25, 101),
        num_f=2, num_mu=2, microfield_model="holtsmark",
        apply_doppler=False, electron_interference=True,
    )
    for profile in profiles:
        assert np.all(np.isfinite(profile))
        assert np.min(profile) >= -1e-10
        assert np.max(profile) > 0


@pytest.mark.parametrize("solver", ["static", "ffm"])
def test_full_operator_is_independent_of_scalar_radius_selector(solver):
    center = 10.198715448212068
    common = dict(
        n_u=2, n_l=1, Z=1, B=3.0, Ne_m3=1e20, Te_ev=2.0,
        energies_ev=np.linspace(center - 2e-6, center + 2e-6, 31),
        num_f=2, num_mu=1,
        microfield_model="holtsmark", apply_doppler=False,
        electron_interference=True,
    )
    if solver == "static":
        common["frequency_dependent_width"] = False
        function = calculate_static_profile
    else:
        common.update(Ti_ev=1.0, A_ion=1.0)
        function = calculate_ffm_profile

    full_scalar = function(**common, electron_model="pppb")
    intra_scalar = function(**common, electron_model="pppb-intra")
    for full, intra in zip(full_scalar, intra_scalar):
        np.testing.assert_allclose(full, intra, rtol=0, atol=0)


def test_ffm_accepts_negative_residues_with_modulus_probabilities():
    diagnostics = {}
    profiles = calculate_ffm_profile(
        n_u=3, n_l=2, Z=1, B=3.0, Ne_m3=1e20, Te_ev=1.0,
        Ti_ev=1.0, A_ion=2.0,
        energies_ev=np.linspace(1.88, 1.90, 21),
        num_f=4, num_mu=3, use_empirical_data=True, atom="D",
        electron_model="pppb-intra", apply_doppler=False,
        electron_interference=True, interference_diagnostics=diagnostics,
    )
    for profile in profiles:
        assert np.all(np.isfinite(profile))
        assert np.max(np.abs(profile)) > 0
    assert set(diagnostics) == {0, 1, -1}
    assert sum(item["negative_mode_count"] for item in diagnostics.values()) > 0
    assert all(0 <= item["negative_absolute_fraction"] < 1
               for item in diagnostics.values())


def test_complex_sdt_probability_diagnostics_report_signed_modes():
    diagnostics = summarize_complex_sdt_probabilities(
        frequencies=np.array([2.0, 1.0, 1.5]),
        gamma_k=np.array([0.2, 0.1, 0.15]),
        strengths=np.array([2.0 + 0.1j, -0.25 - 0.2j, 1.0 + 0.1j]),
    )
    assert diagnostics["mode_count"] == 3
    assert diagnostics["negative_mode_count"] == 1
    assert diagnostics["signed_strength_sum"] == pytest.approx(2.75)
    assert diagnostics["absolute_strength_sum"] == pytest.approx(3.25)
    assert diagnostics["negative_absolute_fraction"] == pytest.approx(0.25 / 3.25)
    mode = diagnostics["reported_negative_modes"][0]
    assert mode["index"] == 1
    assert mode["a_k"] == pytest.approx(-0.25)
    assert mode["nearest_frequency_neighbor_pole_distance_ev"] == pytest.approx(
        np.hypot(0.5, 0.05))


def test_complex_sdt_grouping_preserves_residue_and_first_pole_moment():
    frequencies = np.array([0.0, 1e-6, 0.1])
    widths = np.array([1e-3, 1.1e-3, 2e-3])
    strengths = np.array([-0.2 + 0.1j, 1.2 - 0.1j, 2.0 + 0.0j])
    grouped_frequency, grouped_width, grouped_strength, sizes = (
        group_complex_sdts(
            frequencies, widths, strengths,
            frequency_tolerance_ev=2e-6,
            width_tolerance_ev=2e-4,
        )
    )
    assert sizes.tolist() == [2, 1]
    np.testing.assert_allclose(np.sum(grouped_strength), np.sum(strengths))
    original_moment = np.sum(
        strengths * (frequencies - 1j * widths))
    grouped_moment = np.sum(
        grouped_strength * (grouped_frequency - 1j * grouped_width))
    np.testing.assert_allclose(grouped_moment, original_moment)
    assert np.all(grouped_strength.real >= 0)
    assert np.all(grouped_width > 0)


def test_complex_sdt_grouping_reports_unresolved_negative_residues():
    with pytest.raises(ValueError, match="negative grouped residues"):
        calculate_ffm_profile(
            n_u=3, n_l=2, Z=1, B=3.0, Ne_m3=1e20, Te_ev=1.0,
            Ti_ev=1.0, A_ion=2.0,
            energies_ev=np.linspace(1.88, 1.90, 21),
            num_f=2, num_mu=2, use_empirical_data=True, atom="D",
            electron_model="pppb-intra", apply_doppler=False,
            electron_interference=True,
            interference_group_tolerance_ev=1e-15,
        )


def test_complex_sdt_grouping_accepts_validated_lyman_case():
    diagnostics = {}
    profiles = calculate_ffm_profile(
        n_u=2, n_l=1, Z=1, B=10.0, Ne_m3=1e20, Te_ev=5.0,
        Ti_ev=1.0, A_ion=1.0,
        energies_ev=np.linspace(10.15, 10.25, 101),
        num_f=2, num_mu=2, microfield_model="holtsmark",
        apply_doppler=False, electron_interference=True,
        interference_diagnostics=diagnostics,
        interference_group_tolerance_ev=1e-6,
        interference_group_profile_rtol=1e-3,
    )
    for profile in profiles:
        assert np.all(np.isfinite(profile))
    for item in diagnostics.values():
        grouping = item["grouping"]
        assert grouping["grouped_mode_count"] < grouping["original_mode_count"]
        assert grouping["largest_group_size"] >= 2
        assert grouping["static_profile_relative_max_error"] < 1e-3


def test_zero_fluctuation_ffm_recovers_full_operator_static_profile():
    energies = np.linspace(1.87, 1.91, 61)
    common = dict(
        n_u=3, n_l=2, Z=1, B=100.0, Ne_m3=1e18, Te_ev=5.0,
        energies_ev=energies, num_f=2, num_mu=1,
        microfield_model="holtsmark", apply_doppler=False,
        electron_interference=True,
    )
    with pytest.warns(UserWarning, match="grid spacing"):
        static = calculate_static_profile(
            **common, frequency_dependent_width=False)
    dynamic = calculate_ffm_profile(
        **common, Ti_ev=0.0, A_ion=1.0)
    for actual, expected in zip(dynamic, static):
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-8)


def test_complex_ffm_analytical_matches_large_matrix_inversion():
    """The modulus-probability closed form must match a dense Markov solve."""
    rng = np.random.default_rng(51_1918)
    mode_count = 64
    frequencies = np.linspace(-0.035, 0.041, mode_count)
    frequencies += rng.normal(scale=2e-4, size=mode_count)
    gamma = rng.uniform(3e-4, 2e-3, size=mode_count)
    real_strength = rng.lognormal(mean=-0.2, sigma=0.7, size=mode_count)
    # Exercise destructive-interference residues while retaining a positive
    # total oscillator strength.
    real_strength[::7] *= -0.35
    dispersion = rng.normal(scale=0.15, size=mode_count) * np.abs(real_strength)
    # The complete resolvent has zero summed dispersive strength. Enforce that
    # physical sum rule without making the individual c_k vanish.
    dispersion -= np.sum(dispersion) * real_strength / np.sum(real_strength)
    strengths = real_strength + 1j * dispersion
    fluctuation_rate = 4e-3
    energies = np.linspace(-0.055, 0.061, 23)

    analytical = _complex_ffm_profile_analytical(
        energies, frequencies, gamma, strengths, fluctuation_rate,
        max_chunk_elements=3 * mode_count,
    )

    total_strength = np.sum(real_strength)
    probability = np.abs(real_strength) / np.sum(np.abs(real_strength))
    complex_weight = strengths / total_strength
    ones = np.ones(mode_count)
    explicit = np.empty_like(energies)
    for index, energy in enumerate(energies):
        diagonal = fluctuation_rate + gamma + 1j * (energy - frequencies)
        markov_resolvent = (
            np.diag(diagonal)
            - fluctuation_rate * np.outer(probability, ones)
        )
        response = np.linalg.solve(markov_resolvent, complex_weight)
        explicit[index] = total_strength / np.pi * np.real(np.sum(response))

    np.testing.assert_allclose(analytical, explicit, rtol=2e-12, atol=2e-12)
