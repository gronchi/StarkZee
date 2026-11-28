"""StaticProfileTable: dressed states cached on a fixed (geometric) field grid, reused for
every (Ne, Te, Ti); and microfield_weights, the quadrature weights on given fields."""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from starkzee.he_i import he_i_4471                                       # noqa: E402
from starkzee.microfield import (calculate_holtsmark_field, microfield_quadrature,  # noqa: E402
                                 microfield_weights)
from starkzee.profile_table import StaticProfileTable                     # noqa: E402
from starkzee.static_profile import calculate_static_profile, line_reference_energy  # noqa: E402

E0 = line_reference_energy(3, 2, 1)
GRID = np.linspace(E0 - 4e-3, E0 + 4e-3, 300)


def _err(a, b):
    a, b = np.array(a), np.array(b)
    return np.max(np.abs(a - b)) / b.max()


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ---------------------------------------------------------------- microfield_weights

@pytest.mark.parametrize("kw", [dict(), dict(Ti_ev=3.0), dict(microfield_model="holtsmark"),
                                dict(microfield_model="hooper", charged=True)])
def test_uniform_weights_equal_the_quadrature(kw):
    fields, weights = microfield_quadrature(1e20, 5.0, num_points=20, **kw)
    np.testing.assert_allclose(microfield_weights(fields, 1e20, 5.0, **kw), weights, rtol=0, atol=1e-12)


def test_log_weights_converge_to_the_field_integral():
    """Moments of W(beta) on geometric grids against a fine uniform quadrature."""
    F_H = calculate_holtsmark_field(1e20)
    fields, weights = microfield_quadrature(1e20, 5.0, num_points=4001, Ti_ev=5.0)
    beta = fields / F_H
    ref = [np.sum(weights * beta ** k) for k in (0.5, 1, 2)]
    errors = []
    for n in (20, 40, 80):
        log_fields = 0.01 * F_H * (1000.0 ** (np.arange(n) / (n - 1)))      # beta 0.01 .. 10
        w = microfield_weights(log_fields, 1e20, 5.0, Ti_ev=5.0, rule="log", beta_min=0.01)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        errors.append(max(abs(np.sum(w * (log_fields / F_H) ** k) - r) / r for k, r in zip((0.5, 1, 2), ref)))
    assert errors[-1] < 2e-3 and errors[-1] < errors[0]


def test_weights_reject_bad_grids():
    with pytest.raises(ValueError):
        microfield_weights(np.array([1.0, 2.0, 4.0]), 1e20, 5.0, Ti_ev=5.0)               # not uniform from 0
    with pytest.raises(ValueError):
        microfield_weights(np.linspace(0, 1e9, 10), 1e20, 5.0, Ti_ev=5.0, rule="log")   # not geometric
    with pytest.raises(ValueError):
        microfield_weights(np.linspace(0, 1e9, 10), 1e20, 5.0, Ti_ev=5.0, rule="trapezoid")


# ---------------------------------------------------------------- StaticProfileTable

@pytest.fixture(scope="module")
def table():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return StaticProfileTable(3, 2, 1, 2.0, (1e19, 1e21), num_f=20, num_mu=3)


def test_grid_gives_num_f_nodes_at_every_density(table):
    ratios = table.fields[1:] / table.fields[:-1]
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-12)
    for ne in (1e19, 2.7e19, 1e20, 5.5e20, 1e21):
        assert np.count_nonzero(table.weights(ne, 5.0, 5.0)) in (20, 21)


@pytest.mark.parametrize("kw", [dict(Ti_ev=5.0), dict(Ti_ev=5.0, frequency_dependent_width=False),
                                dict(), dict(Ti_ev=5.0, method="fft")])
def test_cached_profile_equals_the_uncached_calculation(table, kw):
    """The cache changes nothing: same nodes and weights through calculate_static_profile."""
    for ne, te in ((3e19, 4.0), (2e20, 8.0)):
        out = table.profile(GRID, ne, te, **kw)
        weights = table.weights(ne, te, kw.get("Ti_ev"))
        direct = calculate_static_profile(3, 2, 1, 2.0, ne, te, GRID, num_mu=3, radiator=table.radiator,
                                          _quadrature=(table.fields, weights), **kw)
        assert _err(out, direct) < 1e-13


def test_cache_is_reused_and_order_independent():
    a = StaticProfileTable(3, 2, 1, 2.0, (1e19, 1e21), num_f=20, num_mu=3)
    b = StaticProfileTable(3, 2, 1, 2.0, (1e19, 1e21), num_f=20, num_mu=3)
    a.profile(GRID, 1e21, 5.0, Ti_ev=5.0)
    first = a.cached_points
    out_a = a.profile(GRID, 2e19, 5.0, Ti_ev=5.0)
    assert a.cached_points > first
    out_b = b.profile(GRID, 2e19, 5.0, Ti_ev=5.0)
    assert _err(out_a, out_b) < 1e-13
    again = a.cached_points
    a.profile(GRID, 2e19, 9.0, Ti_ev=2.0)                 # new temperatures: no new nodes
    assert a.cached_points == again


def test_converges_to_the_standard_profile():
    """Fine table and fine standard quadrature agree; both rules converge to one integral."""
    fine = StaticProfileTable(3, 2, 1, 2.0, (1e20, 1e20), num_f=80, num_mu=3)
    ref = calculate_static_profile(3, 2, 1, 2.0, 1e20, 5.0, GRID, num_f=400, num_mu=3, Ti_ev=5.0)
    assert _err(fine.profile(GRID, 1e20, 5.0, Ti_ev=5.0), ref) < 2e-3


def test_options_and_ranges(table):
    with pytest.raises(ValueError):
        table.profile(GRID, 5e18, 5.0, Ti_ev=5.0)          # below the range
    with pytest.raises(ValueError):
        table.profile(GRID, 2e21, 5.0, Ti_ev=5.0)          # above the range
    with pytest.raises(TypeError):
        table.profile(GRID, 1e20, 5.0, electron_model="zest")
    with pytest.raises(TypeError):
        table.profile(GRID, 1e20, 5.0, electron_interference=True)
    with pytest.raises(TypeError):
        StaticProfileTable(3, 2, 1, 2.0, (1e19, 1e21), Ti_ev=5.0)
    operator = StaticProfileTable(3, 2, 1, 2.0, (1e19, 1e21), num_mu=3, electron_operator=True,
                                  electron_model="zest")
    weights = operator.weights(1e20, 5.0, 5.0)
    direct = calculate_static_profile(3, 2, 1, 2.0, 1e20, 5.0, GRID, num_mu=3, radiator=operator.radiator,
                                      electron_operator=True, electron_model="zest", Ti_ev=5.0,
                                      _quadrature=(operator.fields, weights))
    assert _err(operator.profile(GRID, 1e20, 5.0, Ti_ev=5.0), direct) < 1e-13


def test_multi_electron_radiator():
    radiator = he_i_4471(2.0)
    grid = np.linspace(radiator.line_energy_ev - 3e-3, radiator.line_energy_ev + 4e-3, 300)
    t = StaticProfileTable(4, 2, 1, 2.0, (1e20, 3e21), num_f=16, num_mu=3, radiator=radiator)
    for ne in (1e20, 1e21):
        weights = t.weights(ne, 5.0, 2.0)
        direct = calculate_static_profile(4, 2, 1, 2.0, ne, 5.0, grid, num_mu=3, radiator=radiator,
                                          Ti_ev=2.0, _quadrature=(t.fields, weights))
        assert _err(t.profile(grid, ne, 5.0, Ti_ev=2.0), direct) < 1e-13
