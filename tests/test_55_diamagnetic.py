"""Diamagnetic (quadratic Zeeman) term of the J-level / LS radiators, on the one-valence-electron path.

(e^2 B^2 / 8 m_e) r^2 sin^2(theta) = (e^2 B^2 / 12 m_e) r^2 (C^0 - C^2_0), with the
reduced elements of r^2 C^k coupled to |(L S) J M>. With hydrogenic radial
integrals it must have the spectrum of the hydrogenic radiator's term in the
uncoupled |n l m_l m_s> basis.
"""
import warnings

import numpy as np
import pytest

from starkzee.coulomb_approximation import coulomb_r2
from starkzee.radiator import build_hamiltonian, radial_r2_element
from starkzee.single_valence import c_iv_5_4, load_single_valence, single_valence_transition
from starkzee.static_profile import calculate_static_profile

EV_TO_CM = 8065.543937


@pytest.fixture(scope="module")
def c_iv():
    return load_single_valence("c_iv")


def _diamagnetic_part(data, B, radial="coulomb"):
    a = single_valence_transition(data, 5, 4, B, radial=radial)
    b = single_valence_transition(data, 5, 4, B, radial=radial, diamagnetic=False)
    return a.upper.H0 - b.upper.H0, a.lower.H0 - b.lower.H0, a


@pytest.mark.parametrize("n, l1, l2, Z", [(5, 0, 0, 4), (5, 1, 3, 4), (5, 2, 4, 4), (4, 3, 3, 4), (3, 0, 2, 1)])
def test_coulomb_r2_is_hydrogenic_for_integer_n(n, l1, l2, Z):
    assert coulomb_r2(float(n), l1, float(n), l2, float(Z)) == pytest.approx(radial_r2_element(n, l1, l2, Z),
                                                                             rel=1e-10)


def test_coupled_operator_matches_the_hydrogenic_radiator(c_iv):
    B = 500.0
    up, lo, _ = _diamagnetic_part(c_iv, B, radial="hydrogenic")
    for H, n in ((up, 5), (lo, 4)):
        assert np.allclose(H, H.conj().T, atol=1e-15)
        # the C IV core (charge 4, mass ~12 u) as a hydrogen-like nucleus of the same mass
        A = c_iv["mass_amu"]
        ref = (build_hamiltonian(n, 4, B, quadratic_zeeman=True, A=A)
               - build_hamiltonian(n, 4, B, quadratic_zeeman=False, A=A))
        # core mass 12 u - m_e (LS) vs nucleus 12 u - 4 m_e (hydrogen-like): ~2e-8 through mu^3
        np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(H)), np.sort(np.linalg.eigvalsh(ref)),
                                   rtol=1e-7, atol=1e-15)


def test_scaling_m_conservation_and_zero_field(c_iv):
    up5, lo5, r = _diamagnetic_part(c_iv, 500.0)
    up1, lo1, _ = _diamagnetic_part(c_iv, 100.0)
    np.testing.assert_allclose(up5, 25.0 * up1, rtol=0, atol=1e-9 * np.abs(up5).max())
    M = np.array([m for _, _, m in r.states[0]])
    assert np.all(np.abs(up5[np.abs(M[:, None] - M[None, :]) > 1e-9]) < 1e-18)
    # C IV n = 5 at 500 T: shifts of a few cm^-1 (positive operator); the Coulomb approximation
    # is within a few per cent of the hydrogenic values
    ev = np.linalg.eigvalsh(up5) * EV_TO_CM
    assert 1.0 < ev.min() and ev.max() < 11.0
    hyd = np.linalg.eigvalsh(_diamagnetic_part(c_iv, 500.0, "hydrogenic")[0]) * EV_TO_CM
    assert np.abs(ev - hyd).max() < 0.06 * hyd.max()
    a = single_valence_transition(c_iv, 5, 4, 0.0)
    b = single_valence_transition(c_iv, 5, 4, 0.0, diamagnetic=False)
    np.testing.assert_array_equal(a.upper.H0, b.upper.H0)


def test_profile_shift_at_500_tesla():
    B = 500.0
    profiles = {}
    for dia in (True, False):
        r = c_iv_5_4(B, diamagnetic=dia)
        E0 = r.line_energy_ev
        grid = np.linspace(E0 - 0.15, E0 + 0.15, 1501)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pi, sp, sm = calculate_static_profile(5, 4, 4, B, 2e24, 10.0, grid, num_f=8, num_mu=2, radiator=r,
                                                  Ti_ev=10.0, electron_operator=True)
        profiles[dia] = pi + 0.5 * (sp + sm)
    a, b = profiles[True], profiles[False]
    shift = (np.sum(grid * a) / np.sum(a) - np.sum(grid * b) / np.sum(b)) * EV_TO_CM
    assert 1.0 < shift < 4.5                             # blue shift ~ mean upper - lower shift (3.4 cm^-1)
    assert 0.005 < np.abs(a - b).max() / b.max() < 0.1
