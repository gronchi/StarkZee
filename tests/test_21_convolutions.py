"""Regression tests for starkzee.convolutions.convolve_fft.

Covers the odd-length one-sample shift defect: convolving a centered delta
profile with a centered delta kernel must return the profile unchanged
(peak at the same index), for both odd- and even-length grids.  Before the
fix, ``fftshift`` (rather than ``ifftshift``) was used to move the kernel to
the origin, which differs from the correct inverse for odd-length arrays and
silently shifted every convolved profile by one sample.
"""

import numpy as np
import pytest

from starkzee.convolutions import convolve_fft


@pytest.mark.parametrize("n", [5, 6, 7, 8, 9, 21, 100, 101])
def test_delta_convolution_preserves_center(n):
    grid = np.arange(n, dtype=float)
    center = n // 2

    profile = np.zeros(n)
    profile[center] = 1.0
    kernel = np.zeros(n)
    kernel[center] = 1.0

    out = convolve_fft(grid, profile, kernel)

    assert np.argmax(out) == center
    assert out[center] == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("n", [5, 6, 7, 21])
def test_asymmetric_kernel_shift_matches_offset(n):
    """A kernel delta offset by +1 from center shifts the profile by +1."""
    grid = np.arange(n, dtype=float)
    center = n // 2

    profile = np.zeros(n)
    profile[center] = 1.0
    kernel = np.zeros(n)
    kernel[center + 1] = 1.0

    out = convolve_fft(grid, profile, kernel)

    assert np.argmax(out) == center + 1
