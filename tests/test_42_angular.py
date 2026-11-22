"""starkzee.angular: Wigner back ends (pywigxjpf vs Racah fallback) and closed-form elements."""
import importlib.util
import itertools

import numpy as np
import pytest

from starkzee import angular
from starkzee.angular import wigner_3j, wigner_6j

HALVES = [x / 2 for x in range(0, 7)]          # 0 .. 3 in steps of 1/2


def _symbols_3j():
    for j1, j2, j3 in itertools.product(HALVES, repeat=3):
        for m1 in np.arange(-j1, j1 + 1):
            for m2 in np.arange(-j2, j2 + 1):
                yield j1, j2, j3, m1, m2, -m1 - m2


def _symbols_6j():
    return itertools.product(HALVES[:5], repeat=6)


def _values(backend, symbols, fn):
    angular.set_backend(backend)
    try:
        return np.array([fn(*s) for s in symbols])
    finally:
        angular.set_backend("auto")


def test_fallback_is_used_without_pywigxjpf():
    angular.set_backend("racah")
    try:
        assert angular.BACKEND == "racah"
        assert wigner_3j(1, 1, 0, 1, -1, 0) == pytest.approx(1 / np.sqrt(3))
    finally:
        angular.set_backend("auto")


def test_invalid_arguments_return_zero():
    for backend in ("auto", "racah"):
        angular.set_backend(backend)
        assert wigner_3j(1, 1, 1, 0.3, -0.3, 0) == 0.0          # not half-integer
        assert wigner_3j(1, 1, 3, 0, 0, 0) == 0.0               # triangle
        assert wigner_3j(1, 1, 1, 1, 1, -1) == 0.0              # m sum
        assert wigner_3j(0.5, 0.5, 0.5, -0.5, 0.5, 0) == 0.0    # j1 + j2 + j3 not integer
        assert wigner_6j(-1, 1, 1, 1, 1, 1) == 0.0
        assert wigner_6j(0, 0, 0, 0, 0, 1) == 0.0
    angular.set_backend("auto")


def test_backend_selection_errors():
    with pytest.raises(ValueError):
        angular.set_backend("sympy")


needs_pywigxjpf = pytest.mark.skipif(importlib.util.find_spec("pywigxjpf") is None,
                                     reason="pywigxjpf not installed")


@needs_pywigxjpf
def test_backends_agree_3j():
    symbols = list(_symbols_3j())
    fast = _values("pywigxjpf", symbols, wigner_3j)
    slow = _values("racah", symbols, wigner_3j)
    assert np.count_nonzero(fast) > 1000
    np.testing.assert_allclose(fast, slow, rtol=1e-13, atol=1e-15)
    assert np.array_equal(fast == 0, np.abs(slow) < 1e-15)


@needs_pywigxjpf
def test_backends_agree_6j():
    symbols = list(_symbols_6j())
    fast = _values("pywigxjpf", symbols, wigner_6j)
    slow = _values("racah", symbols, wigner_6j)
    assert np.count_nonzero(fast) > 500
    np.testing.assert_allclose(fast, slow, rtol=1e-13, atol=1e-15)


def _ck(l1, m1, k, q, l2, m2):
    """<l1 m1| C^k_q |l2 m2> from 3-j symbols (Condon-Shortley)."""
    return ((-1) ** m1 * np.sqrt((2 * l1 + 1) * (2 * l2 + 1))
            * wigner_3j(l1, k, l2, -m1, q, m2) * wigner_3j(l1, k, l2, 0, 0, 0))


def test_closed_forms_match_wigner_formulas():
    for l1 in range(6):
        for l2 in range(6):
            for m1 in range(-l1, l1 + 1):
                for m2 in range(-l2, l2 + 1):
                    # T_0 = C_0, T_{+1} = -C_{+1}, T_{-1} = +C_{-1}
                    for q, sign in ((0, 1), (1, -1), (-1, 1)):
                        assert angular.angular_dipole_element(l1, m1, l2, m2, q) == pytest.approx(
                            sign * _ck(l1, m1, 1, q, l2, m2), abs=1e-14)
                    # sin^2 = 2/3 (1 - C^2_0)
                    expected = (2 / 3) * ((l1 == l2 and m1 == m2) - _ck(l1, m1, 2, 0, l2, m2))
                    assert angular.sin2_theta_element(l1, m1, l2, m2) == pytest.approx(expected, abs=1e-14)


def test_spin_orbit_eigenvalues():
    for l in range(1, 6):
        states = [(ml, ms) for ml in range(-l, l + 1) for ms in (-0.5, 0.5)]
        ls = np.array([[angular.spin_orbit_element(l, *a, *b) for b in states] for a in states])
        expected = sorted([(j * (j + 1) - l * (l + 1) - 0.75) / 2
                           for j in (l - 0.5, l + 0.5) for _ in range(int(2 * j + 1))])
        np.testing.assert_allclose(np.linalg.eigvalsh(ls), expected, atol=1e-13)


@needs_pywigxjpf
def test_large_j_beyond_racah_float_range():
    # The double-precision Racah sum raises OverflowError here; pywigxjpf is exact.
    angular.set_backend("pywigxjpf")
    value = wigner_3j(150, 150, 200, 10, -10, 0)
    assert np.isfinite(value) and value != 0.0
    angular.set_backend("auto")
