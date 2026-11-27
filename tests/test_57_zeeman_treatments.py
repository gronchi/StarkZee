"""Zeeman treatments: exact (intermediate field), weak (g_J only) and strong (no spin-orbit).

C VI n = 2 is built twice, as the hydrogenic radiator (uncoupled |l m_l m_s>
basis, projectors onto the (l, j) levels) and as LS terms (|term J M> basis):
every treatment must give the same levels in both.
"""
import math

import numpy as np
import pytest
from scipy.constants import physical_constants

from starkzee.jlevel_radiator import JLevelTransition
from starkzee.ls_radiator import ActiveElectron, LSTerm, LSTransition, ls_level_data, reduced_element
from starkzee.radiators import HydrogenicTransition
from starkzee.utils import BOHR_MAGNETON_EV_T, FINE_STRUCTURE, G_S

Z = 6
RY_CM = physical_constants["Rydberg constant"][0] / 100     # infinite mass, as the hydrogenic radiator
CM_TO_EV = 1 / 8065.543937


def dirac_cm(n, j):
    return -Z**2 * RY_CM / n**2 - Z**4 * FINE_STRUCTURE**2 * RY_CM / n**3 * (1 / (j + 0.5) - 3 / (4 * n))


T1s = LSTerm("1s 2S", 0, 0.5, 1, {0.5: dirac_cm(1, 0.5)})
T2s = LSTerm("2s 2S", 0, 0.5, 1, {0.5: dirac_cm(2, 0.5)})
T2p = LSTerm("2p 2P", 1, 0.5, -1, {0.5: dirac_cm(2, 0.5), 1.5: dirac_cm(2, 1.5)})
_E = {"1s 2S": ActiveElectron(1, 0), "2s 2S": ActiveElectron(2, 0), "2p 2P": ActiveElectron(2, 1)}
_T = {"1s 2S": T1s, "2s 2S": T2s, "2p 2P": T2p}
COUPLINGS = {(a, b): reduced_element(_T[a], _E[a], _T[b], _E[b], Z_core=Z, radial="hydrogenic")
             for a, b in (("2s 2S", "2p 2P"), ("2p 2P", "1s 2S"))}


def levels(H):
    e = np.linalg.eigvalsh(H)
    return np.sort(e - e.mean())


def hydrogenic(B, zeeman):
    # infinite nuclear mass, as the Dirac levels and g_L = 1 of the LS reference below
    return HydrogenicTransition(2, 1, Z, B, quadratic_zeeman=False, use_empirical_data=False, zeeman=zeeman,
                                A=math.inf)


def ls(B, zeeman):
    return LSTransition([T2s, T2p], [T1s], COUPLINGS, B, g_L=1.0, emitter_charge=Z - 1, zeeman=zeeman)


@pytest.mark.parametrize("zeeman", ["exact", "weak", "strong"])
@pytest.mark.parametrize("B", [100.0, 1000.0])
def test_hydrogenic_and_ls_treatments_agree(B, zeeman):
    a, b = hydrogenic(B, zeeman), ls(B, zeeman)
    for side in ("upper", "lower"):
        np.testing.assert_allclose(levels(getattr(a, side).H0), levels(getattr(b, side).H0), rtol=0, atol=1e-9)


def test_weak_field_is_the_anomalous_zeeman_effect():
    B = 1000.0
    mu = BOHR_MAGNETON_EV_T * B
    g = {0.5: 1 - (G_S - 1) / 3, 1.5: 1 + (G_S - 1) / 3}            # 2p 1/2, 3/2 (g_L = 1)
    expect = ([dirac_cm(2, 0.5) * CM_TO_EV + G_S * mu * M for M in (-0.5, 0.5)]
              + [dirac_cm(2, J) * CM_TO_EV + g[J] * mu * M for J in (0.5, 1.5)
                 for M in np.arange(-J, J + 1)])
    r = ls(B, "weak")
    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(r.upper.H0) + r.upper.E_ref), np.sort(expect),
                               rtol=0, atol=1e-8)


def test_limits():
    # low field: weak = exact to first order in B
    d = np.abs(levels(hydrogenic(0.1, "exact").upper.H0) - levels(hydrogenic(0.1, "weak").upper.H0)).max()
    assert d < 1e-4 * BOHR_MAGNETON_EV_T * 0.1
    # strong: the 2p states split as mu_B B (m_l + g_s m_s), with no spin-orbit term
    mu = BOHR_MAGNETON_EV_T * 1000.0
    H = hydrogenic(1000.0, "strong").upper.H0
    p = np.arange(2, 8)                                               # the six 2p states (basis order s, p)
    e = np.sort(np.linalg.eigvalsh(H[np.ix_(p, p)]))
    expected = np.sort([mu * (ml + G_S * ms) for ml in (-1, 0, 1) for ms in (0.5, -0.5)])
    np.testing.assert_allclose(e - e.mean(), expected - expected.mean(), rtol=0, atol=1e-10)


def test_jlevel_options():
    data, up, lo = ls_level_data([T2s, T2p], [T1s], COUPLINGS, g_L=1.0)
    weak = JLevelTransition(data, up, lo, 500.0, emitter_charge=Z - 1, zeeman="weak")
    np.testing.assert_allclose(levels(weak.upper.H0), levels(ls(500.0, "weak").upper.H0), rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="strong"):
        JLevelTransition(data, up, lo, 500.0, zeeman="strong")
    with pytest.raises(ValueError):
        HydrogenicTransition(2, 1, Z, 1.0, zeeman="paschen")
