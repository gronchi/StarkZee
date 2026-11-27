"""First-moment sum rule of the sigma and pi components.

With every upper sublevel equally populated, the intensity-weighted mean
frequency of polarization q is

    <nu>_q = [Tr(D_q H_u D_q^+) - Tr(D_q^+ H_l D_q)] / Tr(D_q D_q^+),

linear in the Hamiltonians and independent of the eigenvectors. Scalar terms
do not depend on q, the linear Zeeman term (rank 1) gives a shift proportional
to q, and the Stark term (odd parity) drops out. The mean of sigma+ and sigma-
therefore sits on pi, except for the rank-2 part of the diamagnetic term, and
that offset does not depend on the electric field. The check runs through the
eigenvectors of H0 + Fz*M_z + Fx*M_x, so it catches inconsistent bookkeeping
between the Hamiltonian, the Stark templates and the dipole matrices.
"""
import warnings

import numpy as np
import pytest

from starkzee.he_i import he_i_4471
from starkzee.jlevel_radiator import JLevelTransition
from starkzee.ls_radiator import ls_level_data
from starkzee.radiators import HydrogenicTransition
from starkzee.single_valence import c_iv_5_4, load_single_valence, shell
from starkzee.utils import BOHR_MAGNETON_EV_T

B = 300.0
DIRECTIONS = (1.0, 0.6, 0.0)              # cos(angle) between the electric and magnetic fields


def centroids(r, Fz, Fx):
    """Intensity-weighted mean frequency [eV] of each polarization q for one field."""
    eu, vu = np.linalg.eigh(r.upper.H0 + Fz * r.upper.M_z + Fx * r.upper.M_x)
    el, vl = np.linalg.eigh(r.lower.H0 + Fz * r.lower.M_z + Fx * r.lower.M_x)
    nu = (eu[None, :] - el[:, None]) * r.unit_to_ev
    out = {}
    for q in (-1, 0, 1):
        s = np.abs(vl.conj().T @ r.dipoles[q] @ vu) ** 2
        out[q] = (s * nu).sum() / s.sum()
    return out


def sigma_minus_pi(r, Fz=0.0, Fx=0.0):
    c = centroids(r, Fz, Fx)
    return (c[1] + c[-1]) / 2 - c[0]


def fields(r):
    """(Fz, Fx) pairs from below to well above the Stark/Zeeman crossover of the upper manifold."""
    spread = np.ptp(np.linalg.eigvalsh(r.upper.H0))
    scale = spread / np.abs(r.upper.M_z).max()
    return [(f * mu, f * np.sqrt(1 - mu * mu)) for f in (0.3 * scale, 3 * scale) for mu in DIRECTIONS]


def hydrogenic(zeeman, diamagnetic):
    return HydrogenicTransition(3, 2, 6, B, quadratic_zeeman=diamagnetic, use_empirical_data=False,
                                zeeman=zeeman)


def c_iv(zeeman, diamagnetic):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return c_iv_5_4(B, diamagnetic=diamagnetic, zeeman=zeeman)


def he_i(zeeman, diamagnetic):
    assert not diamagnetic
    return he_i_4471(B, zeeman=zeeman)


def jlevel(zeeman, diamagnetic):
    d = load_single_valence("c_iv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        couplings = c_iv_5_4(0.0, diamagnetic=False).couplings
    up, lo = [d["terms"][k] for k in shell(d, 5)], [d["terms"][k] for k in shell(d, 4)]
    data, iu, il = ls_level_data(up, lo, couplings)
    dia = None
    if diamagnetic:                       # any rank-0/2 operator inside each manifold obeys the rule
        dia = {}
        for group in (iu, il):
            for i, a in enumerate(group):
                for b in group[i:]:
                    la, lb = data.levels[a], data.levels[b]
                    if la.parity == lb.parity and abs(la.two_j - lb.two_j) <= 4:
                        # [a0^2], of the size of <r^2> for C IV n = 4-5
                        dia[(a, b)] = {0: 300.0 + 10 * a, 2: 100.0 + 5 * b} if a == b else {2: 50.0}
    return JLevelTransition(data, iu, il, B, diamagnetic=dia, zeeman=zeeman)


CASES = [(hydrogenic, z) for z in ("exact", "weak", "strong")] \
    + [(c_iv, z) for z in ("exact", "weak", "strong")] \
    + [(he_i, z) for z in ("exact", "weak", "strong")] \
    + [(jlevel, z) for z in ("exact", "weak")]
IDS = [f"{f.__name__}-{z}" for f, z in CASES]


@pytest.mark.parametrize("build, zeeman", CASES, ids=IDS)
def test_sigma_mean_sits_on_pi_without_the_diamagnetic_term(build, zeeman):
    r = build(zeeman, diamagnetic=False)
    zeeman_ev = BOHR_MAGNETON_EV_T * B
    for Fz, Fx in [(0.0, 0.0)] + fields(r):
        c = centroids(r, Fz, Fx)
        assert abs((c[1] + c[-1]) / 2 - c[0]) < 1e-9 * zeeman_ev, (Fz, Fx)
        assert abs(c[1] - c[-1]) > 0.1 * zeeman_ev                    # the components are split


@pytest.mark.parametrize("build, zeeman", [c for c in CASES if c[0] is not he_i],
                         ids=[i for c, i in zip(CASES, IDS) if c[0] is not he_i])
def test_diamagnetic_offset_does_not_depend_on_the_electric_field(build, zeeman):
    r = build(zeeman, diamagnetic=True)
    zeeman_ev = BOHR_MAGNETON_EV_T * B
    reference = sigma_minus_pi(r)
    assert abs(reference) > 1e-4 * zeeman_ev
    for Fz, Fx in fields(r):
        assert sigma_minus_pi(r, Fz, Fx) == pytest.approx(reference, rel=0, abs=1e-9 * zeeman_ev), (Fz, Fx)


def test_c_iv_diamagnetic_offset_is_small_and_blue():
    # Scale for comparison with spectra: +0.07 meV at 300 T against a 17 meV Zeeman shift.
    offset = sigma_minus_pi(c_iv("exact", diamagnetic=True))
    assert 0.0 < offset < 0.01 * BOHR_MAGNETON_EV_T * B
