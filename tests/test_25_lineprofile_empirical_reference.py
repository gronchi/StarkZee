"""Regression tests for LineProfile.E0_empirical (D08/C05).

Confirms LineProfile exposes a proper empirical (NIST) counterpart of E0 when
use_empirical_data=True is passed to compute_profile/compute_static_profile/
compute_ffm_profile, without changing self.E0 itself (backward compatible).
"""

import numpy as np
import pytest

from starkzee.line_profile import LineProfile


def test_e0_empirical_none_with_analytical_opt_out():
    lp = LineProfile(n_u=3, n_l=2, B=1.0, Ne_m3=1e19, Te_ev=5.0, species='H')
    assert lp.E0_empirical is None
    lp.compute_profile(np.linspace(650, 665, 200), grid_type='wavelength_nm',
                       use_empirical_data=False)
    assert lp.E0_empirical is None
    assert lp.E0_empirical_wavelength_nm is None


def test_e0_empirical_populated_and_close_to_analytic():
    lp = LineProfile(n_u=3, n_l=2, B=1.0, Ne_m3=1e19, Te_ev=5.0, species='H')
    lp.compute_profile(np.linspace(650, 665, 200), grid_type='wavelength_nm',
                        use_empirical_data=True, atom='H')

    assert lp.E0_empirical is not None
    # E0 (analytic) is untouched, still the pre-computed constructor value.
    assert lp.E0 == pytest.approx(
        (1.0**2) * 13.6 * (1.0 / 4 - 1.0 / 9), rel=0.02)
    # Empirical and analytic line centers should be close (H-alpha), not identical.
    assert lp.E0_empirical == pytest.approx(lp.E0, rel=1e-3)
    assert lp.E0_empirical != lp.E0
    # Derived units are self-consistent.
    assert lp.E0_empirical_wavelength_nm == pytest.approx(656.4632, abs=0.01)
    # Air wavelength should land near the textbook H-alpha air wavelength.
    assert lp.E0_empirical_wavelength_air_nm == pytest.approx(656.28, abs=0.01)


def test_e0_empirical_unknown_shell_raises():
    """n=20 is beyond the bundled fine_structure=False (shell-averaged) H
    data (n<=12), so _empirical_gross_structure_energy_ev's own lookup must
    raise -- distinct from calculate_static_profile's separate fine_structure=True
    lookup, which has a narrower range (n<=8) and would raise first for a
    smaller out-of-range n."""
    from starkzee.line_profile import _empirical_gross_structure_energy_ev
    with pytest.raises(ValueError, match="No empirical"):
        _empirical_gross_structure_energy_ev(20, 2, 'H')


def test_e0_empirical_via_ffm():
    lp = LineProfile(n_u=3, n_l=2, B=1.0, Ne_m3=1e19, Te_ev=5.0, Ti_ev=5.0, species='H')
    lp.compute_ffm_profile(np.linspace(650, 665, 200), grid_type='wavelength_nm',
                            use_empirical_data=True, atom='H')
    assert lp.E0_empirical is not None
