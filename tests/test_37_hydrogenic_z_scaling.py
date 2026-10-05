"""Exact nonrelativistic hydrogenic Z-scaling regressions."""

import numpy as np
import pytest

from starkzee.radiator import _uncoupled_dipole_matrices
from starkzee.static_profile import build_stark_matrix


@pytest.mark.parametrize("n,Z", [(2, 2), (3, 6), (4, 18)])
def test_intrashell_stark_matrix_obeys_exact_z_scaling(n, Z):
    """H(Z,F) = Z^2 H(1,F/Z^3) for the linear Stark contribution."""
    field_z = 2.3e8
    field_x = -1.1e8
    actual = build_stark_matrix(n, Z, field_z, field_x)
    scaled_hydrogen = Z**2 * build_stark_matrix(
        n, 1, field_z / Z**3, field_x / Z**3)
    np.testing.assert_allclose(actual, scaled_hydrogen, rtol=2e-15, atol=1e-20)


@pytest.mark.parametrize("n,Z", [(2, 2), (3, 6), (4, 18)])
def test_linear_stark_spectrum_obeys_exact_z_scaling(n, Z):
    """The eigenvalue spectrum of the linear-Stark term scales exactly."""
    field_z = 2.3e8
    field_x = -1.1e8
    energies_z = np.linalg.eigvalsh(
        build_stark_matrix(n, Z, field_z, field_x))
    energies_h = Z**2 * np.linalg.eigvalsh(
        build_stark_matrix(n, 1, field_z / Z**3, field_x / Z**3))
    np.testing.assert_allclose(energies_z, energies_h, rtol=2e-13, atol=2e-17)


@pytest.mark.parametrize("transition,Z", [((2, 1), 2), ((3, 2), 6), ((5, 2), 18)])
def test_radiative_dipole_matrices_scale_as_inverse_z(transition, Z):
    n_u, n_l = transition
    hydrogen = _uncoupled_dipole_matrices(n_u, n_l, 1)
    ion = _uncoupled_dipole_matrices(n_u, n_l, Z)
    for q in (-1, 0, 1):
        np.testing.assert_allclose(ion[q], hydrogen[q] / Z, rtol=2e-14, atol=1e-14)
