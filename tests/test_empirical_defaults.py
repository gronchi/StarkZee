"""Default empirical energies and explicit analytical compatibility."""
import numpy as np
import pytest
from starkzee.line_profile import LineProfile
from starkzee.radiator import build_hamiltonian
from starkzee.static_profile import calculate_static_profile, discrete_transitions
from starkzee.ffm import calculate_ffm_profile
from starkzee.atomic_data import load_levels


@pytest.mark.parametrize('species', ['H', 'D', 'T'])
@pytest.mark.parametrize('method', ['compute_static_profile', 'compute_ffm_profile'])
def test_default_profile_matches_explicit_empirical(species, method):
    lp = LineProfile(3, 2, B=1, Ne_m3=1e20, Te_ev=5, Ti_ev=1, species=species)
    grid = np.linspace(lp.E0 - 0.01, lp.E0 + 0.01, 101)
    kw = dict(num_f=3, num_mu=2, use_screening=False, apply_doppler=False)
    getattr(lp, method)(grid, **kw)
    default = lp.profile.copy()
    assert lp.result_metadata['use_empirical_data'] is True
    assert lp.E0_empirical is not None
    getattr(lp, method)(grid, use_empirical_data=True, **kw)
    np.testing.assert_array_equal(lp.profile, default)
    getattr(lp, method)(grid, use_empirical_data=False, **kw)
    assert lp.E0_empirical is None
    assert lp.reference_energy_ev == lp.E0
    assert np.max(np.abs(lp.profile - default)) > 0


@pytest.mark.parametrize('solver', [calculate_static_profile, calculate_ffm_profile])
def test_direct_solver_default_matches_empirical(solver):
    kw = dict(n_u=3, n_l=2, Z=1, B=1, Ne_m3=1e20, Te_ev=5,
              energies_ev=np.linspace(1.88, 1.90, 101), num_f=3, num_mu=2,
              use_screening=False, apply_doppler=False)
    if solver is calculate_ffm_profile:
        kw.update(Ti_ev=1, A_ion=1)
    np.testing.assert_array_equal(solver(**kw), solver(**kw, use_empirical_data=True))


def test_default_discrete_transitions_match_explicit_empirical():
    lp = LineProfile(2, 1, B=0, Ne_m3=1e20, Te_ev=5)
    lp.compute_discrete()
    explicit = discrete_transitions(2, 1, 1, 0, use_empirical_data=True)
    np.testing.assert_array_equal(lp.discrete.energy_ev, explicit['energy_ev'])


@pytest.mark.parametrize('atom', ['H', 'D', 'T'])
def test_empirical_without_fine_structure_uses_shell_average(atom):
    from starkzee.atomic_data import empirical_shell_energy_cm
    matrix = build_hamiltonian(2, 1, 0, fine_structure=False,
                               use_empirical_data=True, atom=atom)
    expected = empirical_shell_energy_cm(atom, 2)
    np.testing.assert_allclose(matrix, np.eye(len(matrix)) * expected, atol=1e-8)


def test_tritium_shell_average_uses_measured_degeneracy_weights():
    from starkzee.atomic_data import empirical_shell_energy_cm
    # Independent sum of all n=2 magnetic substates: 2s1/2, 2p1/2, 2p3/2.
    expected = (2 * 82288.78325 + 2 * 82288.74801 + 4 * 82289.11417) / 8
    assert empirical_shell_energy_cm('T', 2) == pytest.approx(expected, abs=1e-8)


def test_empirical_high_z_requires_analytical_opt_out():
    with pytest.raises(ValueError, match='use_empirical_data=False'):
        discrete_transitions(2, 1, 2, 0)
    result = discrete_transitions(2, 1, 2, 0, use_empirical_data=False)
    assert np.all(result['energy_ev'] > 40)
