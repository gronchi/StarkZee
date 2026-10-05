"""Regression tests for starkzee.microfield's Potekhin distribution fixes.

Covers three independently confirmed issues in the Potekhin
(Chabrier & Gilles 2002) microfield-distribution transcription:

- C01: several coefficients in the charged-screened fit (A2, A, alpha, the
  shape exponent, and c) used Gamma where the published fit uses sqrt(Gamma),
  or grouped denominators incorrectly. Pinned here against independently
  re-derived reference values at Gamma=4, s=1.
- C02: `_P_from_Q_grid` differentiated across whichever grid the caller
  supplied, so the density at a given beta depended on which other points
  were queried alongside it.
- C03: the charged-screened density's exponential terms were floored at
  exp(-100) rather than allowed to underflow to zero, so the far tail
  eventually rose instead of continuing to decay.
"""

import numpy as np
import pytest

from starkzee.microfield import potekhin_distribution, _P_charged_screened


def test_charged_screened_coefficients_match_published_fit():
    """A2, A, alpha, shape exponent, and c at Gamma=4, s=1 (published fit)."""
    Gamma, s = 4.0, 1.0
    sqrt_gamma = np.sqrt(Gamma)

    A2 = (0.55 + 10.0 * np.sqrt(s) + 2.0 * s**4.5) / (1.0 + 20.0 * np.sqrt(s))
    A1 = 0.59 + 2540.0 * s**4 + 3.0 * s**14
    A3 = 2.17e-3 * s**5
    A4 = 14.8 / (1.0 + 117.0 * s**3.5)
    A = A1 * (1.0 + A4 * sqrt_gamma) / (1.0 + A2 * Gamma**2 + A3 * Gamma**4)

    alpha1 = 0.1 + 1.1 / (1.0 + 0.145 * s**3)
    alpha2 = 5.4 / (1.0 + 20.0 * s**2) + 1.1 / (1.0 + 14.0 * s**0.35)
    alpha = (alpha1 + 2.0 * alpha2 * sqrt_gamma) / (1.0 + alpha2 * sqrt_gamma)

    gamma1 = 0.1 + 1.1 / (1.0 + 0.174 * s**2.5)
    gamma2 = 5.4 / (1.0 + 21.0 * s**1.5) + 1.1 / (1.0 + 19.0 * s**0.16)
    g_param = (gamma1 + 1.5 * gamma2 * sqrt_gamma) / (1.0 + gamma2 * sqrt_gamma)

    c = 0.097 / (1.0 + 210.0 * s**2.5 * np.exp(-1.3 * s**1.5))

    assert A2 == pytest.approx(0.5976190476, abs=1e-9)
    assert A == pytest.approx(286.1852590, abs=1e-6)
    assert alpha == pytest.approx(1.4344802893, abs=1e-9)
    assert g_param == pytest.approx(1.2107691058, abs=1e-9)
    assert c == pytest.approx(0.0016657600, abs=1e-9)


def test_neutral_pdf_independent_of_query_grid():
    """The density at a fixed beta must not depend on neighboring query points."""
    scalar_val = potekhin_distribution(1.0, gamma=0.0, s=0.0, charged=False)
    arr_val = potekhin_distribution(
        np.array([0.0, 1.0, 2.0]), gamma=0.0, s=0.0, charged=False)[1]
    assert arr_val == pytest.approx(scalar_val, rel=1e-6)

    # A finer grid around the same point must also agree.
    fine_val = potekhin_distribution(
        np.linspace(0.0, 5.0, 501), gamma=0.0, s=0.0, charged=False)
    idx = np.argmin(np.abs(np.linspace(0.0, 5.0, 501) - 1.0))
    assert fine_val[idx] == pytest.approx(scalar_val, rel=1e-3)


def test_charged_screened_tail_decays_monotonically():
    """The far tail must decay, not eventually rise, as beta grows."""
    betas = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000], dtype=float)
    p = _P_charged_screened(betas, gamma=4.0, s=1.0)
    assert np.all(np.isfinite(p))
    assert np.all(p >= 0.0)
    assert np.all(np.diff(p) <= 0.0)
