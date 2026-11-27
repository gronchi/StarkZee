"""Unit tests of the static-profile kernels: fused Lorentzian kernel and batched eigh."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from starkzee import static_profile as sp                                # noqa: E402

N, P = 400, 90
rng = np.random.default_rng(3)
E = np.linspace(1.88, 1.90, N)
C = rng.uniform(1.885, 1.895, P)


def _plain(w_u, w_l, w_n, s_u=None, s_l=None, floor=None):
    """The original expression: (w/pi) / (detuning**2 + w**2)."""
    u = w_u if s_u is None else w_u * s_u
    l = w_l if s_l is None else w_l * s_l
    w = u + l + w_n
    if floor is not None:
        w = np.maximum(w, floor)
    d = E[:, np.newaxis] - C[np.newaxis, :]
    return np.broadcast_to((w / np.pi) / (d**2 + w**2), (N, P))


WIDTHS = {               # (w_u, w_l, w_natural, scale_u, scale_l, floor)
    "scalars": (2e-5, 1e-5, 3e-7, None, None, None),
    "per transition natural": (2e-5, 1e-5, rng.uniform(1e-7, 1e-6, P), None, None, None),
    "frequency dependent": (rng.uniform(1e-5, 3e-5, (N, 1)), rng.uniform(1e-6, 1e-5, (N, 1)),
                            rng.uniform(1e-7, 1e-6, P), None, None, None),
    "operator scales": (rng.uniform(1e-5, 3e-5, (N, 1)), rng.uniform(1e-6, 1e-5, (N, 1)),
                        rng.uniform(1e-7, 1e-6, P), rng.uniform(0.5, 2, P), rng.uniform(0.5, 2, P), None),
    "upper scale only": (2e-5, 0.0, 3e-7, rng.uniform(0.5, 2, P), None, None),
    "floored": (0.0, 0.0, rng.uniform(1e-8, 1e-4, P), None, None, 2e-5),
}


@pytest.mark.parametrize("use_numexpr", [True, False])
@pytest.mark.parametrize("widths", WIDTHS)
def test_lorentzian_kernel_matches_plain_expression(monkeypatch, use_numexpr, widths):
    if use_numexpr:
        pytest.importorskip("numexpr")
        monkeypatch.setattr(sp, "NUMEXPR_MIN_ELEMENTS", 1)
    else:
        monkeypatch.setattr(sp, "_numexpr", None)
    args = WIDTHS[widths]
    ref = _plain(*args)
    out = sp._lorentzian_kernel(E, C, *args)
    assert out.shape == (N, P)
    np.testing.assert_allclose(out, ref, rtol=1e-14, atol=0)


def test_floor_applies_to_every_width():
    w_n = np.array([0.0, 1e-9, 5e-5])
    w = sp._transition_widths(0.0, 0.0, w_n, floor=1e-5)
    np.testing.assert_array_equal(w, [1e-5, 1e-5, 5e-5])


@pytest.mark.parametrize("budget", [sp.EIGH_BATCH_BYTES, 1])
def test_batched_eigh_is_identical_to_single_calls(monkeypatch, budget):
    """Stacked eigh (in one chunk, or one matrix per chunk) equals eigh per point bitwise."""
    monkeypatch.setattr(sp, "EIGH_BATCH_BYTES", budget)
    n = 18
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H0 = a + a.conj().T
    Mz, Mx = (lambda b: b + b.T)(rng.normal(size=(n, n))), (lambda b: b + b.T)(rng.normal(size=(n, n)))
    Fz, Fx = rng.uniform(0, 1, 7), rng.uniform(0, 1, 7)
    results = list(sp._batched_eigh(H0, Mz, Mx, Fz, Fx))
    assert len(results) == 7
    for k, (energies, vectors) in enumerate(results):
        e, v = np.linalg.eigh(H0 + Fz[k] * Mz + Fx[k] * Mx)
        np.testing.assert_array_equal(energies, e)
        np.testing.assert_array_equal(vectors, v)
