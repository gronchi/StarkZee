"""LS-term radiator: multi-electron lines from LS terms and signed reduced E1 elements.

Each manifold (upper or lower) is a set of LS terms; the basis is |term, J, M>.
For every manifold the radiator builds

* H0(B) = E_J (zero-field level energies, any number of terms) + the exact Zeeman
  operator mu_B B (g_L L_z + g_s S_z), including the elements between the J levels
  of a term (Paschen-Back mixing);
* the Stark templates M_z, M_x from the E1 couplings between terms of opposite
  parity inside the manifold (e.g. 4d-4f, 4d-4p for He I 447.1 nm), so that the
  electric interaction is Fz*M_z + Fx*M_x;

and the dipole matrices D_q between the manifolds. It implements the radiator
interface of :mod:`starkzee.radiators`, so :func:`calculate_static_profile` can
use it unchanged.

Conventions (Edmonds / Condon-Shortley):

    <J M| T^1_q |J' M'> = (-1)^(J-M) (J 1 J'; -M q M') <J||T||J'>
    <(L S) J || d || (L' S) J'> = (-1)^(L+S+J'+1) sqrt((2J+1)(2J'+1)) {L J S; J' L' 1} <L||d||L'>
    <(Lp l) L || d || (Lp l') L'> = (-1)^(Lp+l+L'+1) sqrt((2L+1)(2L'+1)) {l L Lp; L' l' 1} <l||r||l'>
    <l||r||l'> = (-1)^l sqrt((2l+1)(2l'+1)) (l 1 l'; 0 0 0) <n l|r|n' l'>

Reduced elements are real; <b||d||a> = (-1)^(L_b - L_a) <a||d||b>. The electric
interaction of the bound electron is V = e F . r. Energies are in cm^-1 inside the
radiator; ``unit_to_ev`` converts them.

Electron broadening: this first version takes constant impact widths per manifold
(``electron_widths_ev``); the per-state operator and the full PPP operator need
the generalized collision operator (plan B4) and raise NotImplementedError.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.constants import atomic_mass as _U, hbar as _HBAR, e as _E_CHARGE

from starkzee.angular import wigner_3j, wigner_6j
from starkzee.radiator import intrashell_radial_dipole, radial_dipole
from starkzee.radiators import Manifold
from starkzee.utils import (
    A0, BOHR_MAGNETON_EV_T, G_S, M_E, energy_ev_to_wavenumber_cm, wavenumber_cm_to_energy_ev,
)

_EV_TO_CM = float(energy_ev_to_wavenumber_cm(1.0))
_CM_TO_EV = float(wavenumber_cm_to_energy_ev(1.0))


def _phase(x):
    """(-1)^x for an integer-valued x (given as float for half-integer arithmetic)."""
    k = int(round(x))
    if abs(x - k) > 1e-9:
        raise ValueError(f"phase exponent {x} is not an integer")
    return -1.0 if k % 2 else 1.0


@dataclass(frozen=True)
class LSTerm:
    """An LS term: L, S, parity (+1/-1) and the energy [cm^-1] of each J level.

    decay_rates: total E1 spontaneous decay rate [s^-1] of each J level (to all
    lower levels), a dict {J: rate} or one value for every J; used for the
    natural width.
    """
    label: str
    L: int
    S: float
    parity: int
    energies_cm: dict
    decay_rates: object = 0.0

    def __post_init__(self):
        if self.parity not in (1, -1):
            raise ValueError("parity must be +1 or -1")
        allowed = {abs(self.L - self.S) + k for k in range(int(round(2 * min(self.L, self.S))) + 1)}
        for J in self.energies_cm:
            if not any(abs(J - a) < 1e-9 for a in allowed):
                raise ValueError(f"term {self.label}: J = {J} is not allowed for L = {self.L}, S = {self.S}")

    def rate(self, J):
        if isinstance(self.decay_rates, dict):
            return float(self.decay_rates.get(J, 0.0))
        return float(self.decay_rates)


# ---------------------------------------------------------------- reduced elements
def orbital_reduced_r(l_a, l_b, radial):
    """<l_a||r||l_b> = (-1)^l_a sqrt((2l_a+1)(2l_b+1)) (l_a 1 l_b; 0 0 0) R  [a0]."""
    return (_phase(l_a) * np.sqrt((2 * l_a + 1) * (2 * l_b + 1))
            * wigner_3j(l_a, 1, l_b, 0, 0, 0) * radial)


def one_electron_reduced(l_a, L_a, l_b, L_b, L_parent, radial):
    """<(Lp l_a) L_a || d || (Lp l_b) L_b> for one active electron on a parent Lp [e a0].

    radial is the signed radial integral <n_a l_a| r |n_b l_b> [a0].
    """
    return (_phase(L_parent + l_a + L_b + 1) * np.sqrt((2 * L_a + 1) * (2 * L_b + 1))
            * wigner_6j(l_a, L_a, L_parent, L_b, l_b, 1) * orbital_reduced_r(l_a, l_b, radial))


def hydrogenic_radial(n_a, l_a, n_b, l_b, Z_core):
    """Signed hydrogenic <n_a l_a| r |n_b l_b> [a0] (positive-near-origin radial functions).

    Gordon's closed form between shells; exact intra-shell value -(3n/2Z) sqrt(n^2 - l^2).
    """
    if n_a == n_b:
        return -intrashell_radial_dipole(n_a, l_a, l_b, Z_core)
    return radial_dipole(n_a, l_a, n_b, l_b, Z_core)


# ---------------------------------------------------------------- radiator
@dataclass
class _Basis:
    terms: list
    states: list = field(default_factory=list)        # (term index, J, M)
    energies: np.ndarray = None

    def __post_init__(self):
        for t, term in enumerate(self.terms):
            for J in sorted(self.terms[t].energies_cm):
                for k in range(int(round(2 * J)) + 1):
                    self.states.append((t, J, -J + k))
        self.energies = np.array([self.terms[t].energies_cm[J] for t, J, _ in self.states])


def _zeeman_operator(basis, g_L, g_s):
    """g_L L_z + g_s S_z in the |term J M> basis (dimensionless, units of mu_B B)."""
    dim = len(basis.states)
    Z = np.zeros((dim, dim))
    for i, (t, J, M) in enumerate(basis.states):
        term = basis.terms[t]
        L, S = term.L, term.S
        for j, (t2, J2, M2) in enumerate(basis.states):
            if t2 != t or abs(M2 - M) > 1e-9 or abs(J2 - J) > 1 + 1e-9:
                continue
            w = _phase(J - M) * wigner_3j(J, 1, J2, -M, 0, M)
            if w == 0.0:
                continue
            root = np.sqrt((2 * J + 1) * (2 * J2 + 1))
            red_L = (_phase(L + S + J2 + 1) * root * wigner_6j(L, J, S, J2, L, 1)
                     * np.sqrt(L * (L + 1) * (2 * L + 1)))
            red_S = (_phase(L + S + J + 1) * root * wigner_6j(S, J, L, J2, S, 1)
                     * np.sqrt(S * (S + 1) * (2 * S + 1)))
            Z[i, j] = w * (g_L * red_L + g_s * red_S)
    return Z


def _dipole_element(bra, ket, reduced_LS, q):
    """<bra| r_q |ket> [e a0] for LS states bra = (term, J, M), ket = (term', J', M')."""
    (ta, J, M), (tb, J2, M2) = bra, ket
    L, S, L2 = ta.L, ta.S, tb.L
    red_J = (_phase(L + S + J2 + 1) * np.sqrt((2 * J + 1) * (2 * J2 + 1))
             * wigner_6j(L, J, S, J2, L2, 1) * reduced_LS)
    return _phase(J - M) * wigner_3j(J, 1, J2, -M, q, M2) * red_J


class LSTransition:
    """Radiator for a line between two manifolds of LS terms.

    Parameters
    ----------
    upper_terms, lower_terms : sequences of LSTerm
    couplings : dict {(label_a, label_b): <a||d||b>}
        Signed LS reduced E1 elements [e a0] (e.g. from :func:`one_electron_reduced`).
        Pairs inside one manifold give the Stark mixing; pairs between the
        manifolds give the radiative dipoles. Each unordered pair once.
    B : float
        Magnetic field [T].
    mass_amu : float, optional
        Emitter mass [u]; sets g_L = 1 - m_e/M unless g_L is given.
    g_L, g_s : float, optional
        Orbital and spin g factors (defaults: finite-mass g_L, CODATA g_s).
    emitter_charge : float
        Net charge of the radiator (microfield selection), default 0.
    electron_widths_ev : (float, float)
        Constant electron impact HWHM of the upper and lower manifold [eV].
    reference_energy_ev : float, optional
        Fixed line reference [eV] for the Doppler width and the electron detuning;
        default: difference of the mean (unweighted over |J M>) manifold energies.
    """

    def __init__(self, upper_terms, lower_terms, couplings, B, mass_amu=None, g_L=None, g_s=G_S,
                 emitter_charge=0.0, electron_widths_ev=(0.0, 0.0), reference_energy_ev=None):
        self.default_emitter_charge = emitter_charge
        self._electron = tuple(float(w) for w in electron_widths_ev)
        if g_L is None:
            g_L = 1.0 if mass_amu is None else 1.0 - M_E / (mass_amu * _U)
        self.g_L, self.g_s = g_L, g_s
        labels = [t.label for t in (*upper_terms, *lower_terms)]
        if len(set(labels)) != len(labels):
            raise ValueError("term labels must be unique")
        self._upper = _Basis(list(upper_terms))
        self._lower = _Basis(list(lower_terms))
        where = {t.label: ("u", i) for i, t in enumerate(upper_terms)}
        where.update({t.label: ("l", i) for i, t in enumerate(lower_terms)})
        inner = {"u": [], "l": []}
        across = []
        for (a, b), value in couplings.items():
            (sa, ia), (sb, ib) = where[a], where[b]
            ta = (self._upper if sa == "u" else self._lower).terms[ia]
            tb = (self._upper if sb == "u" else self._lower).terms[ib]
            if ta.parity == tb.parity or abs(ta.S - tb.S) > 1e-9 or abs(ta.L - tb.L) > 1 or ta.L == tb.L == 0:
                raise ValueError(f"{a} - {b} is not an LS E1 pair (parity, S, |dL| <= 1, not S-S)")
            if sa == sb:
                inner[sa].append((ia, ib, float(value)))
            else:
                # store as <lower||d||upper>; <b||d||a> = (-1)^(L_b - L_a) <a||d||b>
                if sa == "u":
                    across.append((ib, ia, _phase(tb.L - ta.L) * float(value)))
                else:
                    across.append((ia, ib, float(value)))

        mu_cm = BOHR_MAGNETON_EV_T * _EV_TO_CM * B
        self.upper = self._manifold(self._upper, inner["u"], mu_cm)
        self.lower = self._manifold(self._lower, inner["l"], mu_cm)
        self.dipoles = self._dipoles(across)
        self.unit_to_ev = _CM_TO_EV
        self.E0_line = (self.upper.E_ref - self.lower.E_ref) * _CM_TO_EV
        self.E0_physical = self.E0_line if reference_energy_ev is None else float(reference_energy_ev)

    # ---------------------------------------------------------------- construction
    def _manifold(self, basis, pairs, mu_cm):
        H = np.diag(basis.energies.astype(complex))
        H = H + mu_cm * _zeeman_operator(basis, self.g_L, self.g_s)
        E_ref = float(basis.energies.mean())
        H = H - E_ref * np.eye(len(H))
        dim = len(basis.states)
        r_z = np.zeros((dim, dim), dtype=complex)
        r_x = np.zeros((dim, dim), dtype=complex)
        for ia, ib, reduced in pairs:
            ta, tb = basis.terms[ia], basis.terms[ib]
            for i, (t, J, M) in enumerate(basis.states):
                if t != ia:
                    continue
                for j, (t2, J2, M2) in enumerate(basis.states):
                    if t2 != ib:
                        continue
                    bra, ket = (ta, J, M), (tb, J2, M2)
                    z = _dipole_element(bra, ket, reduced, 0)
                    # x = (r_-1 - r_+1)/sqrt(2)
                    x = (_dipole_element(bra, ket, reduced, -1) - _dipole_element(bra, ket, reduced, 1)) / np.sqrt(2.0)
                    r_z[i, j] += z
                    r_z[j, i] += np.conj(z)
                    r_x[i, j] += x
                    r_x[j, i] += np.conj(x)
        scale = A0 * _EV_TO_CM                     # V = e F r: [V/m] * [a0] -> cm^-1
        rates = np.array([basis.terms[t].rate(J) for t, J, _ in basis.states])
        return Manifold(H, E_ref, r_z * scale, r_x * scale, rates)

    def _dipoles(self, across):
        lo, up = self._lower, self._upper
        D = {q: np.zeros((len(lo.states), len(up.states)), dtype=complex) for q in (-1, 0, 1)}
        for il_term, iu_term, reduced in across:
            tl, tu = lo.terms[il_term], up.terms[iu_term]
            for i, (t, J, M) in enumerate(lo.states):
                if t != il_term:
                    continue
                for j, (t2, J2, M2) in enumerate(up.states):
                    if t2 != iu_term:
                        continue
                    q = int(round(M - M2))
                    if abs(q) <= 1:
                        D[q][i, j] = _dipole_element((tl, J, M), (tu, J2, M2), reduced, q)
        return D

    # ---------------------------------------------------------------- natural width
    def shell_natural_width_ev(self):
        """Natural HWHM [eV] from the statistically averaged decay rates of both manifolds."""
        return _HBAR * (self.upper.decay_rates.mean() + self.lower.decay_rates.mean()) / (2.0 * _E_CHARGE)

    # ---------------------------------------------------------------- electron broadening
    def electron_widths(self, detuning_ev, Ne_m3, Te_ev, B, electron_model):
        """Constant impact widths (upper, lower) [eV], broadcast to the detuning grid."""
        shape = np.shape(detuning_ev)
        w_u, w_l = self._electron
        return (np.full(shape, w_u) if shape else w_u), (np.full(shape, w_l) if shape else w_l)

    def electron_r2_scaling(self, vectors_u, vectors_l, r2_form):
        raise NotImplementedError("electron_operator needs the generalized collision operator "
                                  "(plan B4); not yet available for LS radiators")

    def ppp_impact_coefficient(self, Ne_m3, Te_ev, B):
        raise NotImplementedError("electron_interference needs the generalized collision operator "
                                  "(plan B4); not yet available for LS radiators")

    ppp_self_term_r2 = ppp_sdts = ppp_impact_coefficient

    @property
    def states(self):
        """(upper, lower) basis labels as (term label, J, M)."""
        return ([(self._upper.terms[t].label, J, M) for t, J, M in self._upper.states],
                [(self._lower.terms[t].label, J, M) for t, J, M in self._lower.states])
