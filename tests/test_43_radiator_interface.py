"""The static solver takes all atomic input from a radiator object (starkzee.radiators)."""
import warnings

import numpy as np
import pytest

from starkzee.radiators import HydrogenicTransition, Manifold
from starkzee.static_profile import calculate_static_profile

GRID = np.linspace(1.8890, 1.8905, 201)
PLASMA = dict(B=2.0, Ne_m3=1e20, Te_ev=5.0, num_f=4, num_mu=2)


def _profile(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return calculate_static_profile(energies_ev=GRID, **PLASMA, **kw)


@pytest.mark.parametrize("options", [{}, dict(electron_operator=True),
                                     dict(electron_interference=True, frequency_dependent_width=False)])
def test_explicit_hydrogenic_radiator_equals_default(options):
    radiator = HydrogenicTransition(3, 2, 1, PLASMA["B"])
    default = _profile(n_u=3, n_l=2, Z=1, **options)
    explicit = _profile(n_u=3, n_l=2, Z=1, radiator=radiator, **options)
    for a, b in zip(default, explicit):
        np.testing.assert_array_equal(a, b)


def test_solver_uses_only_the_radiator():
    # n_u, n_l and Z are ignored when a radiator is given.
    radiator = HydrogenicTransition(3, 2, 1, PLASMA["B"])
    reference = _profile(n_u=3, n_l=2, Z=1)
    via_radiator = _profile(n_u=9, n_l=8, Z=5, radiator=radiator, emitter_charge=0)
    for a, b in zip(reference, via_radiator):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("empirical", [True, False])
def test_manifold_shapes_and_units(empirical):
    r = HydrogenicTransition(3, 2, 1, 1.0, use_empirical_data=empirical)
    for m, n in ((r.upper, 3), (r.lower, 2)):
        assert isinstance(m, Manifold)
        dim = 2 * n * n
        assert m.H0.shape == m.M_z.shape == m.M_x.shape == (dim, dim)
        assert m.decay_rates.shape == (dim,)
        # Hermitian up to the ~1e-11 relative asymmetry of the numerical <r^2> elements of
        # the quadratic Zeeman term (eigh reads one triangle only).
        np.testing.assert_allclose(m.H0, m.H0.conj().T, atol=1e-10 * np.abs(m.H0).max())
    assert {q: r.dipoles[q].shape for q in (-1, 0, 1)} == {q: (8, 18) for q in (-1, 0, 1)}   # lower x upper
    assert r.unit_to_ev == (pytest.approx(1.2398419843e-4, rel=1e-9) if empirical else 1.0)
    assert r.E0_line == pytest.approx(r.E0_physical, rel=1e-4)       # Balmer-alpha, ~1.89 eV
