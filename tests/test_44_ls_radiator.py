"""LS-term radiator: Zeeman limits, and hydrogen rebuilt from LS terms equals the hydrogenic radiator."""
import warnings

import numpy as np
import pytest

from starkzee.ls_radiator import (
    LSTerm, LSTransition, _Basis, _zeeman_operator, hydrogenic_radial, one_electron_reduced,
)
from starkzee.radiator import build_basis, build_hamiltonian, natural_decay_rates
from starkzee.radiators import HydrogenicTransition
from starkzee.static_profile import calculate_static_profile
from starkzee.utils import G_S, energy_ev_to_wavenumber_cm

EV_TO_CM = float(energy_ev_to_wavenumber_cm(1.0))


# ---------------------------------------------------------------- Zeeman operator
@pytest.mark.parametrize("L,S", [(2, 1.0), (1, 0.5), (3, 1.5), (0, 1.0), (2, 0.0)])
def test_zeeman_weak_field_g_J_and_paschen_back_limit(L, S):
    g_L, g_s = 0.9998, G_S
    Js = [abs(L - S) + k for k in range(int(round(2 * min(L, S))) + 1)]
    basis = _Basis([LSTerm("t", L, S, 1, {J: 0.0 for J in Js})])
    Z = _zeeman_operator(basis, g_L, g_s)
    for i, (_, J, M) in enumerate(basis.states):
        g_J = 0.0 if J == 0 else (g_L * (J*(J+1) - S*(S+1) + L*(L+1))
                                  + g_s * (J*(J+1) + S*(S+1) - L*(L+1))) / (2*J*(J+1))
        assert Z[i, i] == pytest.approx(g_J * M, abs=1e-14)
    # degenerate J levels: eigenvalues are exactly g_L m_L + g_s m_S
    m_S = [-S + k for k in range(int(round(2 * S)) + 1)]
    expected = sorted(g_L * mL + g_s * mS for mL in range(-L, L + 1) for mS in m_S)
    np.testing.assert_allclose(np.linalg.eigvalsh(Z), expected, atol=1e-13)


def test_pair_validation():
    s = LSTerm("s", 0, 1.0, 1, {1: 0.0})
    d = LSTerm("d", 2, 1.0, 1, {1: 10.0, 2: 10.0, 3: 10.0})
    with pytest.raises(ValueError):
        LSTransition([d], [s], {("d", "s"): 1.0}, 1.0)          # same parity, dL = 2
    with pytest.raises(ValueError):
        LSTerm("bad", 1, 1.0, -1, {3: 0.0})                     # J = 3 not allowed for 3P


# ---------------------------------------------------------------- hydrogen through LS terms
def hydrogen_terms(n, Z=1):
    """Hydrogenic (n l) terms with the fine-structure energies of build_hamiltonian at B = 0."""
    H = build_hamiltonian(n, Z, 0.0, False, True, 1, use_empirical_data=False)
    basis = build_basis(n)
    rates = natural_decay_rates(n, Z)
    terms = []
    for l in range(n):
        idx = [k for k, s in enumerate(basis) if s.l == l]
        ev = np.linalg.eigvalsh(H[np.ix_(idx, idx)])
        energies = {0.5: ev[0] * EV_TO_CM} if l == 0 else {l - 0.5: ev[0] * EV_TO_CM,
                                                            l + 0.5: ev[-1] * EV_TO_CM}
        terms.append(LSTerm(f"{n}{'spdfghi'[l]}", l, 0.5, (-1) ** l, energies, rates[idx[0]]))
    return terms


def hydrogen_ls_radiator(n_u, n_l, B, electron_widths, Z=1, reference_energy_ev=None):
    up, lo = hydrogen_terms(n_u, Z), hydrogen_terms(n_l, Z)
    couplings = {}
    for n, terms in ((n_u, up), (n_l, lo)):
        for l in range(n - 1):
            r = hydrogenic_radial(n, l, n, l + 1, Z)
            couplings[(terms[l].label, terms[l + 1].label)] = one_electron_reduced(l, l, l + 1, l + 1, 0, r)
    for lu in range(n_u):
        for ll in range(n_l):
            if abs(lu - ll) == 1:
                r = hydrogenic_radial(n_u, lu, n_l, ll, Z)
                couplings[(up[lu].label, lo[ll].label)] = one_electron_reduced(lu, lu, ll, ll, 0, r)
    return LSTransition(up, lo, couplings, B, g_L=1.0, g_s=G_S, emitter_charge=Z - 1,
                        electron_widths_ev=electron_widths, reference_energy_ev=reference_energy_ev)


PLASMA = dict(Ne_m3=1e20, Te_ev=5.0, num_f=5, num_mu=3, frequency_dependent_width=False)


@pytest.mark.parametrize("B,Ti", [(2.0, None), (0.5, 2.0), (6.0, None)])
def test_hydrogen_balmer_alpha_from_ls_terms_equals_hydrogenic(B, Ti):
    hyd = HydrogenicTransition(3, 2, 1, B, quadratic_zeeman=False, use_empirical_data=False)
    widths = tuple(float(w) for w in hyd.electron_widths(0.0, PLASMA["Ne_m3"], PLASMA["Te_ev"], B, "pppb"))
    # same Doppler/detuning reference (the hydrogenic one is the gross-structure energy)
    ls = hydrogen_ls_radiator(3, 2, B, widths, reference_energy_ev=hyd.E0_physical)
    grid = np.linspace(1.8885, 1.8910, 401)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = calculate_static_profile(3, 2, 1, B, energies_ev=grid, radiator=hyd, Ti_ev=Ti, **PLASMA)
        got = calculate_static_profile(3, 2, 1, B, energies_ev=grid, radiator=ls, Ti_ev=Ti, **PLASMA)
    for a, b in zip(ref, got):
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-9 * np.max(np.abs(a)))


def test_operator_options_are_not_available_yet():
    ls = hydrogen_ls_radiator(3, 2, 1.0, (1e-5, 1e-5))
    with pytest.raises(NotImplementedError):
        calculate_static_profile(3, 2, 1, 1.0, Ne_m3=1e20, Te_ev=5.0, energies_ev=np.linspace(1.888, 1.890, 11),
                                 num_f=2, num_mu=1, radiator=ls, electron_operator=True)
