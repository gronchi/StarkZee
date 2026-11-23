"""electron_model='none': ion quasi-static (+ natural, Doppler) profiles for any radiator."""
import warnings

import numpy as np
import pytest

from starkzee.he_i import he_i_4471
from starkzee.static_profile import calculate_static_profile


def _profile(*args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return calculate_static_profile(*args, **kw)


def _central_width(grid, p):
    c = np.cumsum(p)
    c /= c[-1]
    return np.interp(0.75, c, grid) - np.interp(0.25, c, grid)


def test_none_rejects_the_ppp_operator():
    with pytest.raises(ValueError, match="needs an electron model"):
        _profile(3, 2, 1, 0.0, 1e21, 5.0, np.linspace(1.88, 1.90, 51), electron_model="none",
                 electron_interference=True, frequency_dependent_width=False)


@pytest.mark.parametrize("radiator", [None, "he_i"])
def test_ion_only_profile_keeps_the_area_and_is_narrower(radiator):
    if radiator is None:
        nnz, E0, r = (3, 2, 1), 1.8887, None
    else:
        r = he_i_4471(0.0)
        nnz, E0 = (4, 2, 1), r.line_energy_ev
    grid = np.linspace(E0 - 0.02, E0 + 0.02, 8001)
    ions = sum(_profile(*nnz, 0.0, 1e21, 5.0, grid, num_f=20, radiator=r, apply_doppler=False,
                        electron_model="none"))
    both = sum(_profile(*nnz, 0.0, 1e21, 5.0, grid, num_f=20, radiator=r, apply_doppler=False))
    # the Lorentzian electron wings reach beyond the window; the ion-only profile does not
    assert np.trapezoid(ions, grid) == pytest.approx(np.trapezoid(both, grid), rel=0.05)
    assert np.trapezoid(ions, grid) >= np.trapezoid(both, grid)
    assert _central_width(grid, ions) < _central_width(grid, both)


def test_none_works_with_the_operator_flag_and_doppler():
    grid = np.linspace(1.8867, 1.8907, 801)
    a = _profile(3, 2, 1, 1.0, 1e20, 5.0, grid, num_f=8, Ti_ev=2.0, electron_model="none")
    b = _profile(3, 2, 1, 1.0, 1e20, 5.0, grid, num_f=8, Ti_ev=2.0, electron_model="none", electron_operator=True)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)               # the operator flag has nothing to scale
