"""J-level radiator: equals the LS radiator through ls_level_data; data checks for FAC-type input."""
import warnings

import numpy as np
import pytest

from starkzee.jlevel_radiator import JLevelTransition
from starkzee.level_data import E1Element, JLevel, LevelData
from starkzee.ls_radiator import LSTransition, ls_level_data
from starkzee.static_profile import calculate_static_profile
from starkzee.utils import G_S, wavenumber_cm_to_energy_ev
from tests.test_44_ls_radiator import hydrogen_terms

CM_TO_EV = float(wavenumber_cm_to_energy_ev(1.0))


def hydrogen_couplings(up, lo, n_u, n_l):
    from starkzee.ls_radiator import hydrogenic_radial, one_electron_reduced
    c = {}
    for n, terms in ((n_u, up), (n_l, lo)):
        for l in range(n - 1):
            c[(terms[l].label, terms[l + 1].label)] = one_electron_reduced(
                l, l, l + 1, l + 1, 0, hydrogenic_radial(n, l, n, l + 1, 1))
    for lu in range(n_u):
        for ll in range(n_l):
            if abs(lu - ll) == 1:
                c[(up[lu].label, lo[ll].label)] = one_electron_reduced(
                    lu, lu, ll, ll, 0, hydrogenic_radial(n_u, lu, n_l, ll, 1))
    return c


@pytest.fixture(scope="module")
def pair():
    up, lo = hydrogen_terms(3), hydrogen_terms(2)
    c = hydrogen_couplings(up, lo, 3, 2)
    return up, lo, c


@pytest.mark.parametrize("B", [0.0, 2.0, 8.0])
def test_matrices_equal_ls_radiator(pair, B):
    up, lo, c = pair
    ls = LSTransition(up, lo, c, B, g_L=1.0, g_s=G_S)
    data, iu, il = ls_level_data(up, lo, c, g_L=1.0, g_s=G_S)
    jl = JLevelTransition(data, iu, il, B)
    for a, b in ((ls.upper, jl.upper), (ls.lower, jl.lower)):
        np.testing.assert_allclose(a.H0 * CM_TO_EV, b.H0, atol=1e-15)
        np.testing.assert_allclose(a.M_z * CM_TO_EV, b.M_z, atol=1e-25)
        np.testing.assert_allclose(a.M_x * CM_TO_EV, b.M_x, atol=1e-25)
        np.testing.assert_array_equal(a.decay_rates, b.decay_rates)
    for q in (-1, 0, 1):
        np.testing.assert_allclose(ls.dipoles[q], jl.dipoles[q], atol=1e-13)
    assert jl.E0_line == pytest.approx(ls.E0_line, rel=1e-13)


def test_profiles_equal_ls_radiator(pair):
    up, lo, c = pair
    B, widths = 2.0, (2e-5, 1e-5)
    ls = LSTransition(up, lo, c, B, g_L=1.0, g_s=G_S, electron_widths_ev=widths)
    data, iu, il = ls_level_data(up, lo, c, g_L=1.0, g_s=G_S)
    jl = JLevelTransition(data, iu, il, B, electron_widths_ev=widths)
    grid = np.linspace(1.8885, 1.8910, 301)
    kw = dict(Ne_m3=1e20, Te_ev=5.0, energies_ev=grid, num_f=5, num_mu=3, frequency_dependent_width=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = calculate_static_profile(3, 2, 1, B, radiator=ls, **kw)
        got = calculate_static_profile(3, 2, 1, B, radiator=jl, **kw)
    for a, b in zip(ref, got):
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-9 * np.max(a))


def _fac_like(g=None, signed=True):
    levels = {0: JLevel(0, 0.0, 2, -1, g_J=g), 1: JLevel(1, 20.0, 2, 1, g_J=g),
              2: JLevel(2, 20.001, 4, -1, g_J=g)}
    e1 = [E1Element(0, 1, 0.5, True, rate_s=1e8), E1Element(1, 2, 3.0, signed)]
    return LevelData(levels, e1)


def test_level_data_checks():
    data = _fac_like()
    JLevelTransition(data, [1, 2], [0], 0.0)                       # B = 0 needs no g_J
    with pytest.raises(ValueError, match="g_J"):
        JLevelTransition(data, [1, 2], [0], 1.0)
    with pytest.raises(ValueError, match="no sign"):
        JLevelTransition(_fac_like(g=1.0, signed=False), [1, 2], [0], 1.0)
    with pytest.warns(UserWarning, match="without sign"):
        JLevelTransition(_fac_like(g=1.0, signed=False), [1], [2], 1.0)   # unsigned radiative pair
    with pytest.raises(ValueError):
        JLevelTransition(data, [1], [1], 0.0)


def test_decay_rates_from_dataset():
    data = _fac_like(g=1.0)
    jl = JLevelTransition(data, [1, 2], [0], 1.0)
    assert np.all(jl.upper.decay_rates[:3] == 1e8)                 # level 1: sum of its E1 rates
    assert np.all(jl.upper.decay_rates[3:] == 0.0)                 # level 2: no rate given


def test_weak_field_zeeman_from_g_J():
    data = _fac_like(g=1.5)
    jl = JLevelTransition(data, [2], [0], 2.0)
    from starkzee.utils import BOHR_MAGNETON_EV_T
    shifts = np.sort(np.linalg.eigvalsh(jl.upper.H0))
    expected = np.sort(1.5 * BOHR_MAGNETON_EV_T * 2.0 * np.arange(-2, 3)) + (20.001 - jl.upper.E_ref)
    np.testing.assert_allclose(shifts, expected, atol=1e-12)
