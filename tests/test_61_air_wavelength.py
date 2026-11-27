"""Vacuum-to-air conversion: Peck & Reeder (1972) with the NIST ASD ranges."""
import numpy as np

from starkzee.utils import vacuum_to_air_wavelength_nm


def test_nist_standard_air_example():
    # NIST ASD: 200.0648 nm in vacuum is 200.0000 nm in standard air.
    assert abs(vacuum_to_air_wavelength_nm(200.0648) - 200.0000) < 1e-4


def test_vacuum_outside_5000_to_50000_per_cm():
    lam = np.array([121.567, 200.0, 2000.0, 4051.2])
    np.testing.assert_array_equal(vacuum_to_air_wavelength_nm(lam), lam)


def test_air_inside_range():
    lam = np.array([300.0, 656.4614, 1999.0])
    air = vacuum_to_air_wavelength_nm(lam)
    assert np.all(air < lam)
    assert abs(vacuum_to_air_wavelength_nm(656.4614) - 656.2801) < 1e-4
