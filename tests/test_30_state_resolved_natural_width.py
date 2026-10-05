"""Regression coverage for state-resolved radiative lifetimes (C11)."""

import json
from pathlib import Path

import warnings

import numpy as np
import pytest

from starkzee.ffm import calculate_ffm_profile
from starkzee.radiator import (
    _uncoupled_dipole_matrices, build_basis, einstein_a,
    einstein_a_substate_rates, natural_decay_rates,
)
from starkzee.static_profile import calculate_static_profile
from starkzee.utils import reduced_mass_rydberg_ev


_BENCHMARK_PATH = (Path(__file__).parents[1] / "starkzee" / "data" /
                   "radiative_benchmarks.json")


def _radiative_benchmarks():
    return json.loads(_BENCHMARK_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("n_u,n_l,Z", [(2, 1, 1), (3, 1, 1), (3, 2, 1),
                                        (4, 2, 2)])
def test_substate_rate_average_recovers_shell_einstein_a(n_u, n_l, Z):
    rates = einstein_a_substate_rates(n_u, n_l, Z)
    assert rates.shape == (2 * n_u**2,)
    assert np.all(rates >= 0.0)
    assert np.mean(rates) == pytest.approx(einstein_a(n_u, n_l, Z), rel=2e-15)


def test_metastable_2s_and_radiating_2p_are_distinguished():
    basis = build_basis(2)
    rates = natural_decay_rates(2, 1)
    rates_2s = rates[[i for i, state in enumerate(basis) if state.l == 0]]
    rates_2p = rates[[i for i, state in enumerate(basis) if state.l == 1]]

    assert np.array_equal(rates_2s, np.zeros(2))
    assert np.all(rates_2p > 0.0)
    assert np.allclose(rates_2p, rates_2p[0], rtol=0.0, atol=0.0)
    assert np.mean(rates) == pytest.approx(einstein_a(2, 1, 1), rel=2e-15)


def test_external_radiative_benchmark_provenance_is_explicit():
    registry = _radiative_benchmarks()
    assert registry["source"]["doi"] == "10.1063/1.3077727"
    assert registry["source"]["location"] == "Table 6"
    assert registry["source"]["accuracy_grade"] == "AAA"
    assert registry["comparison"]["relative_tolerance"] == 0.001
    for benchmark in registry["benchmarks"]:
        assert sum(channel["rate_s_inv"]
                   for channel in benchmark["channels"]) == pytest.approx(
                       benchmark["total_rate_s_inv"], rel=0.0, abs=0.0)


@pytest.mark.parametrize("benchmark", _radiative_benchmarks()["benchmarks"],
                         ids=lambda item: item["state"])
def test_state_resolved_e1_rates_match_nist_hydrogen_data(benchmark):
    """Validate nl-term rates against Wiese & Fuhr (2009), NIST Table 6."""
    n = benchmark["n"]
    l = benchmark["l"]
    basis = build_basis(n)
    term_rates = natural_decay_rates(n, 1)[
        [i for i, state in enumerate(basis) if state.l == l]
    ]

    # With gross-structure energies and no external fields, all m_l/m_s
    # substates belonging to one nl term have the same isotropic E1 rate.
    np.testing.assert_allclose(term_rates, term_rates[0], rtol=2e-15, atol=0.0)
    assert term_rates[0] == pytest.approx(
        benchmark["total_rate_s_inv"],
        rel=_radiative_benchmarks()["comparison"]["relative_tolerance"],
    )


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_isotropic_decay_operator_is_diagonal_in_uncoupled_basis(n):
    """The state-rate rotation may discard no radiative coherences."""
    operator = np.zeros((2 * n**2, 2 * n**2), dtype=complex)
    for n_l in range(1, n):
        # Distinct positive weights ensure cancellation is not occurring only
        # between lower shells; physical energy prefactors are also scalar per
        # shell and therefore cannot change this diagonality result.
        for D in _uncoupled_dipole_matrices(n, n_l, 1).values():
            operator += n_l * (D.conj().T @ D)
    off_diagonal = operator - np.diag(np.diag(operator))
    assert np.max(np.abs(off_diagonal)) < 1e-12


def test_hydrogenic_substate_rates_scale_as_z_fourth_power():
    rates_h = einstein_a_substate_rates(3, 2, 1)
    rates_he = einstein_a_substate_rates(3, 2, 2)
    np.testing.assert_allclose(rates_he, 2**4 * rates_h, rtol=2e-14, atol=1e-6)


def _profile_kwargs():
    e0 = reduced_mass_rydberg_ev(1, 1) * (1 / 2**2 - 1 / 3**2)
    return dict(
        n_u=3, n_l=2, Z=1, B=10.0, Ne_m3=1e14, Te_ev=1.0,
        energies_ev=e0 + np.linspace(-2e-4, 2e-4, 201),
        num_f=2, num_mu=2, use_screening=False,
        quadratic_zeeman=False, fine_structure=False, apply_doppler=False,
    )


def test_static_default_is_state_resolved_and_compatibility_mode_differs():
    kwargs = _profile_kwargs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        default = np.array(calculate_static_profile(
            **kwargs, frequency_dependent_width=False))
        explicit = np.array(calculate_static_profile(
            **kwargs, frequency_dependent_width=False,
            natural_width_mode='state_resolved'))
        averaged = np.array(calculate_static_profile(
            **kwargs, frequency_dependent_width=False,
            natural_width_mode='shell_average'))

    assert np.array_equal(default, explicit)
    assert np.all(np.isfinite(default))
    assert not np.array_equal(default, averaged)


def test_ffm_default_is_state_resolved_and_compatibility_mode_differs():
    kwargs = _profile_kwargs()
    kwargs.update(Ti_ev=1.0, A_ion=1.0)
    default = np.array(calculate_ffm_profile(**kwargs))
    explicit = np.array(calculate_ffm_profile(
        **kwargs, natural_width_mode='state_resolved'))
    averaged = np.array(calculate_ffm_profile(
        **kwargs, natural_width_mode='shell_average'))

    assert np.array_equal(default, explicit)
    assert np.all(np.isfinite(default))
    assert not np.array_equal(default, averaged)


def test_invalid_natural_width_mode_is_rejected():
    with pytest.raises(ValueError, match="natural_width_mode"):
        calculate_static_profile(**_profile_kwargs(), natural_width_mode='invalid')

