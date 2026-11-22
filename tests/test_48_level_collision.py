"""Generalized impact operator: hydrogenic limit, level-gap suppression, perturbers."""
import warnings

import numpy as np
import pytest

from starkzee.level_collision import (
    build_level_collision_operator, cutoff_frequency_ev, impact_factors, line_center_factor,
)
from starkzee.ls_radiator import LSTerm, LSTransition, hydrogenic_radial, one_electron_reduced
from starkzee.radiators import HydrogenicTransition
from starkzee.static_profile import calculate_static_profile
from starkzee.utils import G_S
from tests.test_44_ls_radiator import hydrogen_terms
from tests.test_47_jlevel_radiator import hydrogen_couplings

NE, TE = 1e20, 5.0
GRID = np.linspace(1.8885, 1.8910, 301)


def hydrogen_ls(B):
    up, lo = hydrogen_terms(3), hydrogen_terms(2)
    hyd = HydrogenicTransition(3, 2, 1, B, quadratic_zeeman=False, use_empirical_data=False)
    ls = LSTransition(up, lo, hydrogen_couplings(up, lo, 3, 2), B, g_L=1.0, g_s=G_S,
                      collision_n=3, collision_Z=1, collision_n_lower=2,
                      reference_energy_ev=hyd.E0_physical)
    return hyd, ls


@pytest.mark.parametrize("options", [
    dict(electron_interference=True, frequency_dependent_width=False),
    dict(electron_operator=True, electron_model="pppb-intra"),
    dict(electron_model="pppb-intra"),
    dict(electron_operator=True, electron_model="pppb-intra", frequency_dependent_width=False, Ti_ev=2.0),
])
def test_hydrogen_from_ls_terms_reproduces_hydrogenic_electron_broadening(options):
    B = 2.0
    hyd, ls = hydrogen_ls(B)
    assert max(abs(e1 - e2) for e1 in ls._collision_u.energies_ev for e2 in ls._collision_u.energies_ev) \
        < cutoff_frequency_ev(NE, TE, B)                    # fine-structure gaps below the cutoff
    kw = dict(Ne_m3=NE, Te_ev=TE, energies_ev=GRID, num_f=5, num_mu=3, **options)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = calculate_static_profile(3, 2, 1, B, radiator=hyd, **kw)
        got = calculate_static_profile(3, 2, 1, B, radiator=ls, **kw)
    # the non-Hermitian solve of the interference case is done in different bases: ~1e-9 rounding
    tol = 1e-8 if options.get("electron_interference") else 1e-9
    for a, b in zip(ref, got):
        np.testing.assert_allclose(b, a, rtol=0, atol=tol * np.max(np.abs(a)))


def test_impact_factor_gap_suppression():
    wc = cutoff_frequency_ev(NE, TE, 0.0)
    W = impact_factors([0.0, 0.1 * wc, 30 * wc, 300 * wc], NE, TE, 0.0, 4, 1)
    W0 = line_center_factor(NE, TE, 0.0, 4, 1)
    assert W[0, 0] == pytest.approx(W0) and W[0, 1] == pytest.approx(W0)     # below the cutoff
    assert W0 > W[0, 2] > W[0, 3] > 0                                         # G decreases with the gap
    np.testing.assert_array_equal(W, W.T)


def _two_terms(gap_cm, perturber=False):
    d = LSTerm("4d", 2, 1.0, 1, {1: 0.0, 2: 0.0, 3: 0.0})
    f = LSTerm("4f", 3, 1.0, -1, {2: gap_cm, 3: gap_cm, 4: gap_cm})
    p = LSTerm("2p", 1, 1.0, -1, {0: -20000.0, 1: -20000.0, 2: -20000.0})
    c = {("4d", "4f"): one_electron_reduced(2, 2, 3, 3, 0, hydrogenic_radial(4, 2, 4, 3, 1)),
         ("4d", "2p"): one_electron_reduced(2, 2, 1, 1, 0, hydrogenic_radial(4, 2, 2, 1, 1))}
    if perturber:
        return LSTransition([d], [p], c, 0.0, collision_n=4, upper_perturbers=[f])
    return LSTransition([d, f], [p], c, 0.0, collision_n=4)


def test_perturbing_level_adds_to_the_self_term():
    alone = LSTransition([LSTerm("4d", 2, 1.0, 1, {1: 0.0, 2: 0.0, 3: 0.0})],
                         [LSTerm("2p", 1, 1.0, -1, {0: -20000.0, 1: -20000.0, 2: -20000.0})],
                         {("4d", "2p"): 1.0}, 0.0, collision_n=4)
    with_f = _two_terms(3.0, perturber=True)
    assert alone._collision_u.r2_mean == 0.0
    # 4d manifold perturbed by 4f: sum over all |J M> of both terms gives (2S+1)|<L||r||L'>|^2,
    # so the mean per 4d state is |<4d||r||4f>|^2 / (2L+1)
    red = one_electron_reduced(2, 2, 3, 3, 0, hydrogenic_radial(4, 2, 4, 3, 1))
    assert with_f._collision_u.r2_mean == pytest.approx(red**2 / 5, rel=1e-12)
    w_u, _ = with_f.electron_widths(0.0, NE, TE, 0.0, "pppb")
    assert w_u > 0


def test_operator_is_hermitian_and_dissipative():
    r = _two_terms(3.0)
    du, dl = len(r.states[0]), len(r.states[1])
    rng = np.random.default_rng(1)
    Vu = np.linalg.qr(rng.normal(size=(du, du)) + 1j * rng.normal(size=(du, du)))[0]
    Vl = np.linalg.qr(rng.normal(size=(dl, dl)) + 1j * rng.normal(size=(dl, dl)))[0]
    phi = build_level_collision_operator(Vu, Vl, r._collision_u, r._collision_l, NE, TE, 0.0, 4, 1)
    np.testing.assert_allclose(phi, phi.conj().T, atol=1e-14 * np.abs(phi).max())
    assert np.linalg.eigvalsh(phi).min() > -1e-12 * np.abs(phi).max()


def test_missing_collision_model_raises():
    up, lo = hydrogen_terms(3), hydrogen_terms(2)
    ls = LSTransition(up, lo, hydrogen_couplings(up, lo, 3, 2), 1.0)
    with pytest.raises(NotImplementedError, match="collision_n"):
        ls.ppp_impact_coefficient(NE, TE, 1.0)
    with pytest.raises(NotImplementedError, match="PPPB"):
        _two_terms(3.0).electron_widths(0.0, NE, TE, 0.0, "zest")
