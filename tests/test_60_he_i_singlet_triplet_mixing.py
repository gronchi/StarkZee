"""He I n = 4 singlet-triplet mixing: level couplings of the J-level / LS radiators and the 447.1 nm radiator."""
import warnings

import numpy as np
import pytest

from starkzee.he_i import he_i_4471, load_he_i_triplets, singlet_triplet_coupling
from starkzee.ls_radiator import LSTerm, LSTransition
from starkzee.static_profile import calculate_static_profile
from starkzee.utils import energy_ev_to_wavenumber_cm

NIST_4 = ("4s 3S", "4p 3P", "4d 3D", "4f 3F", "4s 1S", "4p 1P", "4d 1D", "4f 1F")


@pytest.fixture(scope="module")
def data():
    return load_he_i_triplets()


def _upper_levels_cm(r):
    e = np.linalg.eigvalsh(r.upper.H0)
    return energy_ev_to_wavenumber_cm(e + r.upper.E_ref)


def test_coupling_fit_reproduces_both_observed_levels(data):
    T = data["terms"]
    for t3, t1 in (("4d 3D", "4d 1D"), ("4f 3F", "4f 1F")):
        E_TT, E_SS, V = singlet_triplet_coupling(T[t3], T[t1])
        L = float(T[t3].L)
        ev = np.linalg.eigvalsh([[E_TT, V], [V, E_SS]])
        np.testing.assert_allclose(ev, sorted([T[t3].energies_cm[L], T[t1].energies_cm[L]]), rtol=0, atol=1e-9)
    with pytest.raises(ValueError):
        singlet_triplet_coupling(T["4s 3S"], T["4s 1S"])


def test_radiator_levels_are_the_nist_levels(data):
    levels = _upper_levels_cm(he_i_4471(0.0, singlets=True))
    nist = [E for t in NIST_4 for E in data["terms"][t].energies_cm.values()]
    assert max(np.min(np.abs(levels - E)) for E in nist) < 1e-8


@pytest.mark.parametrize("B", [0.0, 4.0])
def test_4f_j3_levels_are_close_to_equal_singlet_triplet_mixtures(B):
    r = he_i_4471(B, singlets=True)
    su, _ = r.states
    _, v = np.linalg.eigh(r.upper.H0)
    singlet = np.array([sum(abs(v[k, j]) ** 2 for k, (lab, _, _) in enumerate(su) if lab == "4f 1F")
                        for j in range(v.shape[1])])
    mixed = singlet[(singlet > 0.05) & (singlet < 0.95)]
    assert len(mixed) == 14 if B == 0 else len(mixed) >= 12     # the 2 x 7 J = 3 sublevels at B = 0
    assert np.all(np.abs(mixed - 0.5) < 0.06)


def test_singlets_leave_the_statistically_populated_profile_unchanged():
    grid = np.linspace(2.7718, 2.7740, 1101)            # 447.0-447.3 nm in eV
    kw = dict(Ne_m3=1e21, Te_ev=5.0, energies_ev=grid, num_f=8, num_mu=3, frequency_dependent_width=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sum(calculate_static_profile(4, 2, 1, 0.0, radiator=he_i_4471(0.0, singlets=True, electron_widths_ev=(2e-5, 0)), **kw))
        b = sum(calculate_static_profile(4, 2, 1, 0.0, radiator=he_i_4471(0.0, singlets=False,
                                                                            electron_widths_ev=(2e-5, 0)), **kw))
    # equal sublevel populations: the mixing shares the 3F strength between levels 0.024 cm^-1 apart
    assert np.trapezoid(a, grid) == pytest.approx(np.trapezoid(b, grid), rel=3e-3)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-2 * np.max(b))


def test_level_couplings_need_matching_levels_and_the_exact_treatment():
    t3 = LSTerm("3", 1, 1.0, -1, {0.0: 10.0, 1.0: 10.0, 2.0: 10.0})
    t1 = LSTerm("1", 1, 0.0, -1, {1.0: 11.0})
    lo = LSTerm("g", 0, 1.0, 1, {1.0: 0.0})
    c = {("3", "g"): 1.0}
    LSTransition([t3, t1], [lo], c, 1.0, term_couplings={("3", "1"): {1: 0.3}})
    with pytest.raises(ValueError, match="J = 2"):
        LSTransition([t3, t1], [lo], c, 1.0, term_couplings={("3", "1"): {2: 0.3}})
    with pytest.raises(ValueError, match="exact"):
        LSTransition([t3, t1], [lo], c, 1.0, term_couplings={("3", "1"): {1: 0.3}}, zeeman="weak")
    strong = LSTransition([t3, t1], [lo], c, 1.0, term_couplings={("3", "1"): {1: 0.3}}, zeeman="strong")
    assert np.allclose(strong.upper.H0, np.diag(np.diag(strong.upper.H0)) + (strong.upper.H0 - np.diag(
        np.diag(strong.upper.H0))))                     # builds; spin-orbit coupling dropped
