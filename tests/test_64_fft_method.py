"""method='fft' of calculate_static_profile and the FFT sums of starkzee.lorentzian_fft.

The FFT method is approximate with a target error fft_tol relative to the peak; the
default method='exact' is covered by test_62 (frozen reference profiles).
"""
import inspect
import math
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from starkzee import lorentzian_fft as lf                                # noqa: E402
from starkzee.static_profile import calculate_static_profile, line_reference_energy  # noqa: E402
from tests import test_62_static_profile_regression as regression        # noqa: E402

GRID = np.linspace(0.0, 1.0, 601)
DX = GRID[1] - GRID[0]


def _sticks(K, width_range, centre_range=(-0.2, 1.2), seed=0):
    rng = np.random.default_rng(seed)
    return (rng.uniform(*centre_range, K), rng.random((K, 3)), rng.uniform(*width_range, K))


def _err(out, ref):
    return np.max(np.abs(out - ref)) / ref.max()


# ---------------------------------------------------------------- fixed widths

@pytest.mark.parametrize("tol", [1e-6, 1e-8, 1e-10])
@pytest.mark.parametrize("width_range", [(0.02 * DX, 0.05 * DX), (0.5 * DX, 2 * DX),
                                         (3 * DX, 40 * DX), (0.1 * DX, 100 * DX)],
                         ids=["narrow", "about dx", "wide", "spread"])
def test_lorentzian_sum(width_range, tol):
    c, a, w = _sticks(3000, width_range)
    assert _err(lf.lorentzian_sum(GRID, c, a, w, None, tol), lf._direct(GRID, c, a, w, None)) < tol


@pytest.mark.parametrize("tol", [1e-6, 1e-8, 1e-10])
@pytest.mark.parametrize("sigma, width_range", [(10 * DX, (0.01 * DX, DX)), (3 * DX, (0.1 * DX, 20 * DX)),
                                                (0.7 * DX, (0.1 * DX, 2 * DX))],
                         ids=["doppler dominated", "mixed", "sigma below dx"])
def test_voigt_sum(sigma, width_range, tol):
    c, a, w = _sticks(2000, width_range)
    assert _err(lf.lorentzian_sum(GRID, c, a, w, sigma, tol), lf._direct(GRID, c, a, w, sigma)) < tol


def test_far_components_and_edges():
    """Components far outside the grid (multipole) and on its edges."""
    c, a, w = _sticks(3000, (DX, 3 * DX), centre_range=(-20.0, 21.0))
    c[:10] = GRID[0]
    c[10:20] = GRID[-1]
    assert _err(lf.lorentzian_sum(GRID, c, a, w, None, 1e-9), lf._direct(GRID, c, a, w, None)) < 1e-9
    assert _err(lf.lorentzian_sum(GRID, c, a, w, 5 * DX, 1e-9), lf._direct(GRID, c, a, w, 5 * DX)) < 1e-9


def test_empty_and_single_component():
    assert lf.lorentzian_sum(GRID, np.array([]), np.zeros((0, 3)), np.array([])).shape == (len(GRID), 3)
    c, a, w = np.array([0.5]), np.array([[1.0, 2.0, 3.0]]), np.array([DX])
    assert _err(lf.lorentzian_sum(GRID, c, a, w), lf._direct(GRID, c, a, w, None)) < 1e-8


def test_faddeeva_derivatives_match_finite_differences():
    """Recurrence (|z| < ASYMPTOTIC_Z) and asymptotic branch of w^(p)(z)/p!."""
    from scipy.special import wofz
    z = np.array([0.3 + 0.2j, -2.0 + 1.0j, 4.0 + 0.05j, 12.0 + 0.5j, -30.0 + 3.0j])
    q = lf._faddeeva_taylor(z, 6)
    h = 1e-3
    for p in range(1, 4):                       # complex-step style central differences
        approx = sum((-1) ** (p - k) * math.comb(p, k) * wofz(z + (k - p / 2) * h)
                     for k in range(p + 1)) / h ** p / math.factorial(p)
        np.testing.assert_allclose(q[p], approx, rtol=1e-4 * 10 ** p, atol=1e-12)
    np.testing.assert_allclose(q[0], wofz(z), rtol=1e-13)


# ---------------------------------------------------------------- frequency-dependent widths

def _fd_reference(c, a, nat, u, l, su, sl, floor=None):
    w = u[:, None] * (1 if su is None else su) + l[:, None] * (1 if sl is None else sl) + nat
    if floor is not None:
        w = np.maximum(w, floor)
    d = GRID[:, None] - c
    return ((w / np.pi) / (d ** 2 + w ** 2)) @ a


@pytest.mark.parametrize("tol", [1e-6, 1e-8])
@pytest.mark.parametrize("variation", [0.03, 0.3])
@pytest.mark.parametrize("scales", ["none", "both", "upper only"])
def test_frequency_dependent_sum(scales, variation, tol):
    rng = np.random.default_rng(1)
    x = np.linspace(0, 3, len(GRID))
    u = 2 * DX * (1 + variation * np.cos(x))
    l = 0.3 * u * (1 + 0.04 * np.sin(x))
    c, a, nat = rng.uniform(-3, 4, 3000), rng.random((3000, 3)), rng.uniform(0.01, 0.1, 3000) * DX
    su = None if scales == "none" else rng.uniform(0.5, 2, 3000)
    sl = rng.uniform(0.5, 2, 3000) if scales == "both" else None
    out = lf.frequency_dependent_sum(GRID, c, a, nat, u, l, su, sl, None, tol)
    assert _err(out, _fd_reference(c, a, nat, u, l, su, sl)) < tol


def test_frequency_dependent_floor_and_constant_widths():
    rng = np.random.default_rng(2)
    c, a = rng.uniform(0, 1, 500), rng.random((500, 3))
    nat = rng.uniform(0.0, 0.5, 500) * DX
    zeros = np.zeros(len(GRID))
    out = lf.frequency_dependent_sum(GRID, c, a, nat, zeros, zeros, None, None, DX, 1e-8)
    assert _err(out, _fd_reference(c, a, nat, zeros, zeros, None, None, DX)) < 1e-8


# ---------------------------------------------------------------- calculate_static_profile

def test_exact_is_the_default():
    params = inspect.signature(calculate_static_profile).parameters
    assert params["method"].default == "exact"


FFT_CASES = [name for name in regression.CASES if "interference" not in name]


@pytest.fixture(scope="module")
def reference():
    return np.load(regression.REFERENCE)


@pytest.mark.parametrize("name", FFT_CASES)
def test_fft_method_matches_exact(reference, monkeypatch, name):
    """Every non-interference branch with method='fft' agrees with the exact profiles."""
    import starkzee.static_profile as sp
    original = sp.calculate_static_profile
    monkeypatch.setattr(regression, "calculate_static_profile",
                        lambda *args, **kw: original(*args, method="fft", fft_tol=1e-8, **kw))
    ref = reference[name]
    assert _err(regression._evaluate(name), ref) < 1e-8


def test_fft_method_on_nonuniform_grid():
    E0 = line_reference_energy(3, 2, 1)
    grid = E0 + np.sort(np.random.default_rng(3).uniform(-3e-3, 3e-3, 400))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kw = dict(Ti_ev=5.0, num_f=8, num_mu=3)
        exact = np.array(calculate_static_profile(3, 2, 1, 5.0, 1e20, 5.0, grid, **kw))
        fft = np.array(calculate_static_profile(3, 2, 1, 5.0, 1e20, 5.0, grid, method="fft", **kw))
    assert _err(fft, exact) < 1e-6          # both interpolate from uniform work grids


def test_invalid_method_and_interference():
    E0 = line_reference_energy(3, 2, 1)
    grid = np.linspace(E0 - 3e-3, E0 + 3e-3, 100)
    with pytest.raises(ValueError):
        calculate_static_profile(3, 2, 1, 5.0, 1e20, 5.0, grid, method="fast")
    with pytest.raises(ValueError):
        calculate_static_profile(3, 2, 1, 5.0, 1e20, 5.0, grid, method="fft",
                                 frequency_dependent_width=False, electron_interference=True)
