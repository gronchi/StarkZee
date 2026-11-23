"""B = 0 shortcuts of the static solver: one field direction, and Delta M blocks of the PPP generator."""
import dataclasses
import warnings

import numpy as np
import pytest

from starkzee.he_i import he_i_4471
from starkzee.radiators import HydrogenicTransition
from starkzee.static_profile import _eigh_by_m, calculate_static_profile

PPP = dict(electron_operator=True, electron_interference=True, frequency_dependent_width=False)


def _profile(*args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return calculate_static_profile(*args, **kw)


@pytest.mark.parametrize("options", [{}, dict(electron_operator=True), PPP], ids=["scalar", "operator", "ppp"])
def test_b0_single_direction_equals_angle_average(options):
    # B = 1e-12 T takes the angle quadrature; Zeeman effects are ~1e-16 of the widths
    grid = np.linspace(1.8877, 1.8897, 401)
    iso = _profile(3, 2, 1, 0.0, 1e21, 5.0, grid, num_f=6, num_mu=4, Ti_ev=2.0, **options)
    avg = _profile(3, 2, 1, 1e-12, 1e21, 5.0, grid, num_f=6, num_mu=4, Ti_ev=2.0, **options)
    scale = max(np.max(p) for p in avg)
    for a, b in zip(iso, avg):
        assert np.max(np.abs(a - b)) < 1e-8 * scale
    assert np.array_equal(iso[0], iso[1]) and np.array_equal(iso[1], iso[2])


def test_eigh_by_m_gives_pure_m_eigenvectors():
    r = HydrogenicTransition(3, 2, 1, 0.0)
    H = r.upper.H0 + 1e5 * r.upper.M_z
    E, V, M = _eigh_by_m(H, r.upper.m_values)
    assert np.allclose(E, np.linalg.eigvalsh(H), atol=1e-9 * np.max(np.abs(E)))
    assert np.allclose(H @ V, V * E, atol=1e-9 * np.max(np.abs(E)))
    for k in range(len(E)):                         # every eigenvector lives in one M block
        assert np.all(V[r.upper.m_values != M[k], k] == 0)


def _without_m(r):
    r.upper = dataclasses.replace(r.upper, m_values=None)
    r.lower = dataclasses.replace(r.lower, m_values=None)
    return r


@pytest.mark.parametrize("make, nnz, ne, center, half", [
    (lambda: HydrogenicTransition(3, 2, 1, 0.0), (3, 2, 1), 1e21, 1.8887, 0.01),
    (lambda: he_i_4471(0.0), (4, 2, 1), 1e21, None, 4e-3),
], ids=["H-alpha", "He I 447.1"])
def test_delta_m_blocks_match_the_full_generator(make, nnz, ne, center, half):
    out = []
    for r in (make(), _without_m(make())):
        E0 = center if center is not None else r.line_energy_ev
        grid = np.linspace(E0 - half, E0 + half, 601)
        out.append(_profile(*nnz, 0.0, ne, 5.0, grid, num_f=4, num_mu=2, radiator=r, Ti_ev=5.0, **PPP))
    scale = max(np.max(p) for p in out[1])
    for a, b in zip(*out):
        assert np.max(np.abs(a - b)) < 1e-9 * scale
