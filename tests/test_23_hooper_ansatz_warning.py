"""Regression test for the Hooper-like screened-microfield ansatz warning.

D14/D15 (StarkZee_original_audit.md): the analytic screened characteristic-
function ansatz behind `hooper_distribution` produces a negative raw density
for a range of beta once the screening parameter a is large enough, and (per
a Fatou's-lemma argument) cannot be the characteristic function of any
non-degenerate field distribution for any a > 0. The returned values are
silently clipped to zero and renormalized downstream, which is not a
validated correction. This is flagged to callers via a UserWarning rather
than fixed by replacing the ansatz (tracked separately in TODO item 2).
"""

import warnings

import pytest

from starkzee.microfield import hooper_distribution


def test_warns_above_screening_threshold():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hooper_distribution(2.0, a=1.0, charged=True)
    assert any(issubclass(w.category, UserWarning) for w in caught)
    assert any("negative" in str(w.message) for w in caught)


def test_no_warning_for_small_screening():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hooper_distribution(2.0, a=0.1, charged=True)
    assert not any(
        "screening parameter" in str(w.message) for w in caught
    )


@pytest.mark.parametrize("charged", [True, False])
def test_raw_density_actually_goes_negative_above_threshold(charged):
    """Confirm the documented failure mode is real, not just a warning label."""
    import numpy as np
    from scipy.integrate import quad
    import math

    def raw_value(beta, a, charged):
        fac = 1.5 if charged else 1.0

        def integrand(y, beta, a, charged):
            if y == 0:
                return 0.0
            screening = (1.0 + fac * (a**2) / (y**2 + 1e-8)) ** (-0.75)
            return y * math.sin(beta * y) * math.exp(-(y**1.5) * screening)

        val, _ = quad(integrand, 0, 30, args=(beta, a, charged), limit=200)
        return (2.0 * beta / np.pi) * val

    betas = np.linspace(0.5, 8, 40)
    vals = [raw_value(b, 1.0, charged) for b in betas]
    assert min(vals) < 0.0
