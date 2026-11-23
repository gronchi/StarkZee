"""LS-term radiator: multi-electron lines from LS terms and signed reduced E1 elements.

Each manifold (upper or lower) is a set of LS terms; the basis is |term, J, M>.
For every manifold the radiator builds

* H0(B) = E_J (zero-field level energies, any number of terms) + the exact Zeeman
  operator mu_B B (g_L L_z + g_s S_z), including the elements between the J levels
  of a term (Paschen-Back mixing), and optionally the diamagnetic term from the
  term elements of r^2 C^0 and r^2 C^2 (``diamagnetic``);
* the Stark templates M_z, M_x from the E1 couplings between terms of opposite
  parity inside the manifold (e.g. 4d-4f, 4d-4p for He I 447.1 nm), so that the
  electric interaction is Fz*M_z + Fx*M_x;

and the dipole matrices D_q between the manifolds. :class:`LSTransition` expresses
the terms as J levels (:func:`ls_level_data`) and runs them through
:class:`~starkzee.jlevel_radiator.JLevelTransition`, which implements the radiator
interface of :mod:`starkzee.radiators`; :func:`_zeeman_operator` and
:func:`_dipole_element` build the same matrices directly in the |term J M> basis
as an independent reference.

Conventions (Edmonds / Condon-Shortley):

    <J M| T^1_q |J' M'> = (-1)^(J-M) (J 1 J'; -M q M') <J||T||J'>
    <(L S) J || d || (L' S) J'> = (-1)^(L+S+J'+1) sqrt((2J+1)(2J'+1)) {L J S; J' L' 1} <L||d||L'>
    <(Lp l) L || d || (Lp l') L'> = (-1)^(Lp+l+L'+1) sqrt((2L+1)(2L'+1)) {l L Lp; L' l' 1} <l||r||l'>
    <l||r||l'> = (-1)^l sqrt((2l+1)(2l'+1)) (l 1 l'; 0 0 0) <n l|r|n' l'>

Reduced elements are real; <b||d||a> = (-1)^(L_b - L_a) <a||d||b>. The electric
interaction of the bound electron is V = e F . r. Term energies are in cm^-1; the
radiator works in eV.

Electron broadening: constant widths per manifold (``electron_widths_ev``), or the
generalized impact operator of :mod:`starkzee.level_collision` with ``collision_n``
(see :class:`~starkzee.jlevel_radiator.JLevelTransition`).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.constants import atomic_mass as _U

from starkzee.angular import wigner_3j, wigner_6j
from starkzee.coulomb_approximation import coulomb_radial, effective_n
from starkzee.radiator import intrashell_radial_dipole, radial_dipole
from starkzee.jlevel_radiator import JLevelTransition
from starkzee.utils import G_S, M_E, energy_ev_to_wavenumber_cm, wavenumber_cm_to_energy_ev

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

    @property
    def mean_energy_cm(self):
        """Statistically weighted (2J+1) mean energy of the term [cm^-1]."""
        w = {J: 2 * J + 1 for J in self.energies_cm}
        return sum(w[J] * E for J, E in self.energies_cm.items()) / sum(w.values())


@dataclass(frozen=True)
class ActiveElectron:
    """The radiating electron of a term: n l on a parent term L_parent.

    limit_cm is the series limit (ionization energy to the parent term) on the
    same energy scale as the term energies; it is needed for the Coulomb
    approximation.
    """
    n: int
    l: int
    L_parent: int = 0
    limit_cm: float = None


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


def one_electron_r2_reduced(l_a, L_a, l_b, L_b, L_parent, k, radial_r2):
    """<(Lp l_a) L_a || r^2 C^k || (Lp l_b) L_b> [a0^2] for one active electron on a parent Lp.

    radial_r2 is <n_a l_a| r^2 |n_b l_b> [a0^2]; k = 0 or 2 (the diamagnetic term).
    """
    orbital = (_phase(l_a) * np.sqrt((2 * l_a + 1) * (2 * l_b + 1))
               * wigner_3j(l_a, k, l_b, 0, 0, 0))                                 # <l_a||C^k||l_b>
    return (_phase(L_parent + l_a + L_b + k) * np.sqrt((2 * L_a + 1) * (2 * L_b + 1))
            * wigner_6j(l_a, L_a, L_parent, L_b, l_b, k) * orbital * radial_r2)


def ls_diamagnetic_elements(terms, index, diamagnetic):
    """J-level elements {(i, j): {k: <(L S)J||r^2 C^k||(L' S)J'>}} from term elements.

    terms: {label: LSTerm}; index: {(label, J): level index}; diamagnetic:
    {(label_a, label_b): {k: <L_a||r^2 C^k||L_b>}} (each unordered pair once).
    """
    out = {}
    for (a, b), by_rank in diamagnetic.items():
        ta, tb = terms[a], terms[b]
        if abs(ta.S - tb.S) > 1e-9 or ta.parity != tb.parity:
            raise ValueError(f"{a} - {b}: the diamagnetic term needs equal S and parity")
        for J in sorted(ta.energies_cm):
            for J2 in sorted(tb.energies_cm):
                if a == b and J2 < J:                  # the reverse follows from hermiticity
                    continue
                values = {}
                for k, red in by_rank.items():
                    v = (_phase(ta.L + ta.S + J2 + k) * np.sqrt((2 * J + 1) * (2 * J2 + 1))
                         * wigner_6j(ta.L, J, ta.S, J2, tb.L, k) * red)
                    if v:
                        values[k] = float(v)
                if values:
                    out[(index[(a, J)], index[(b, J2)])] = values
    return out


def hydrogenic_radial(n_a, l_a, n_b, l_b, Z_core):
    """Signed hydrogenic <n_a l_a| r |n_b l_b> [a0] (positive-near-origin radial functions).

    Gordon's closed form between shells; exact intra-shell value -(3n/2Z) sqrt(n^2 - l^2).
    """
    if n_a == n_b:
        return -intrashell_radial_dipole(n_a, l_a, l_b, Z_core)
    return radial_dipole(n_a, l_a, n_b, l_b, Z_core)


def reduced_element(term_a, electron_a, term_b, electron_b, Z_core=1.0, mass_amu=None, radial="coulomb"):
    """Signed LS reduced E1 element <a||d||b> [e a0] for one active electron.

    radial : "coulomb" (default) uses the Coulomb approximation with n* from the
        weighted term energies and the series limits of ``electron_a/b``;
        "hydrogenic" uses Gordon's hydrogenic integrals with the integer n and
        Z_core.
    Both terms must share the parent L (the spectator core is unchanged).
    """
    if electron_a.L_parent != electron_b.L_parent:
        raise ValueError("one-electron E1 requires the same parent term")
    if radial == "coulomb":
        if electron_a.limit_cm is None or electron_b.limit_cm is None:
            raise ValueError("the Coulomb approximation needs the series limits (limit_cm)")
        na = effective_n(term_a.mean_energy_cm, electron_a.limit_cm, Z_core, mass_amu)
        nb = effective_n(term_b.mean_energy_cm, electron_b.limit_cm, Z_core, mass_amu)
        r = coulomb_radial(na, electron_a.l, nb, electron_b.l, Z_core)
    elif radial == "hydrogenic":
        r = hydrogenic_radial(electron_a.n, electron_a.l, electron_b.n, electron_b.l, Z_core)
    else:
        raise ValueError("radial must be 'coulomb' or 'hydrogenic'")
    return one_electron_reduced(electron_a.l, term_a.L, electron_b.l, term_b.L, electron_a.L_parent, r)


def ls_level_data(upper_terms, lower_terms, couplings, g_L=1.0, g_s=G_S):
    """Express LS terms as a J-level :class:`~starkzee.level_data.LevelData`.

    Every (term, J) becomes a level (energy in eV) with its LS g_J; the elements of
    g_L L + g_s S between the J levels of a term become magnetic elements, and the
    LS couplings give the E1 elements <(L S)J||d||(L' S)J'>. Returns
    (data, upper indices, lower indices); with these,
    :class:`~starkzee.jlevel_radiator.JLevelTransition` reproduces
    :class:`LSTransition`.
    """
    from starkzee.level_data import E1Element, JLevel, LevelData, MagneticElement

    levels, index, terms = {}, {}, {}
    upper, lower = [], []
    for side, group in (("u", upper_terms), ("l", lower_terms)):
        for term in group:
            terms[term.label] = term
            for J in sorted(term.energies_cm):
                k = len(levels)
                index[(term.label, J)] = k
                L, S = term.L, term.S
                g = 0.0 if J == 0 else (g_L * (J*(J+1) - S*(S+1) + L*(L+1))
                                        + g_s * (J*(J+1) + S*(S+1) - L*(L+1))) / (2*J*(J+1))
                levels[k] = JLevel(k, term.energies_cm[J] * _CM_TO_EV, int(round(2 * J)), term.parity,
                                   configuration=term.label, label=f"{term.label} J={J:g}", g_J=g,
                                   decay_rate_s=term.rate(J))
                (upper if side == "u" else lower).append(k)
    magnetic = []
    for term in terms.values():
        Js = sorted(term.energies_cm)
        L, S = term.L, term.S
        for J, J2 in zip(Js, Js[1:]):
            root = np.sqrt((2 * J + 1) * (2 * J2 + 1))
            red_L = (_phase(L + S + J2 + 1) * root * wigner_6j(L, J, S, J2, L, 1)
                     * np.sqrt(L * (L + 1) * (2 * L + 1)))
            red_S = (_phase(L + S + J + 1) * root * wigner_6j(S, J, L, J2, S, 1)
                     * np.sqrt(S * (S + 1) * (2 * S + 1)))
            magnetic.append(MagneticElement(index[(term.label, J)], index[(term.label, J2)],
                                            g_L * red_L + g_s * red_S))
    e1 = []
    for (a, b), reduced_LS in couplings.items():
        ta, tb = terms[a], terms[b]
        for J in sorted(ta.energies_cm):
            for J2 in sorted(tb.energies_cm):
                red = (_phase(ta.L + ta.S + J2 + 1) * np.sqrt((2 * J + 1) * (2 * J2 + 1))
                       * wigner_6j(ta.L, J, ta.S, J2, tb.L, 1) * reduced_LS)       # <a J||d||b J2>
                if red == 0.0:
                    continue
                ia, ib = index[(a, J)], index[(b, J2)]
                lo, up = (ia, ib) if levels[ia].energy_ev <= levels[ib].energy_ev else (ib, ia)
                if (lo, up) != (ia, ib):                  # store <lower||d||upper>
                    red *= _phase(J - J2)
                e1.append(E1Element(lo, up, float(red), True))
    return LevelData(levels, e1, magnetic, source="LS terms"), upper, lower


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


class LSTransition(JLevelTransition):
    """Radiator for a line between two manifolds of LS terms.

    Each (term, J) becomes a level of a :class:`~starkzee.level_data.LevelData`
    (:func:`ls_level_data`) and the line is a :class:`JLevelTransition`, which
    provides the Hamiltonian, Stark templates, dipoles and electron broadening.

    Parameters
    ----------
    upper_terms, lower_terms : sequences of LSTerm
    couplings : dict {(label_a, label_b): <a||d||b>}
        Signed LS reduced E1 elements [e a0] (e.g. from :func:`one_electron_reduced`
        or :func:`reduced_element`). Pairs inside one manifold give the Stark
        mixing, pairs between the manifolds the radiative dipoles, pairs with a
        perturbing term only the collision self terms. Each unordered pair once.
    B : float
        Magnetic field [T].
    mass_amu : float, optional
        Emitter mass [u]; sets g_L = 1 - m_e/M unless g_L is given, and the Doppler mass.
    g_L, g_s : float, optional
        Orbital and spin g factors (defaults: finite-mass g_L, CODATA g_s).
    emitter_charge, electron_widths_ev, reference_energy_ev, collision_n, collision_Z, collision_n_lower :
        see :class:`~starkzee.jlevel_radiator.JLevelTransition`.
    upper_perturbers, lower_perturbers : sequences of LSTerm
        Terms outside the manifolds that perturb them through electron collisions.
    diamagnetic : dict, optional
        {(label_a, label_b): {0: <a||r^2 C^0||b>, 2: <a||r^2 C^2||b>}} [a0^2],
        term elements (e.g. :func:`one_electron_r2_reduced`) between terms of
        one manifold, diagonal pairs included, each unordered pair once. Adds the
        diamagnetic (quadratic Zeeman) term; default: linear Zeeman term only.
    """

    def __init__(self, upper_terms, lower_terms, couplings, B, mass_amu=None, g_L=None, g_s=G_S,
                 emitter_charge=0.0, electron_widths_ev=None, reference_energy_ev=None,
                 collision_n=None, collision_Z=None, upper_perturbers=(), lower_perturbers=(),
                 collision_n_lower=None, diamagnetic=None):
        if g_L is None:
            g_L = 1.0 if mass_amu is None else 1.0 - M_E / (mass_amu * _U)
        self.g_L, self.g_s = g_L, g_s
        groups = [list(upper_terms), list(lower_terms), list(upper_perturbers), list(lower_perturbers)]
        terms = {}
        for group in groups:
            for t in group:
                if t.label in terms:
                    raise ValueError("term labels must be unique")
                terms[t.label] = t
        for (a, b) in couplings:
            ta, tb = terms[a], terms[b]
            if ta.parity == tb.parity or abs(ta.S - tb.S) > 1e-9 or abs(ta.L - tb.L) > 1 or ta.L == tb.L == 0:
                raise ValueError(f"{a} - {b} is not an LS E1 pair (parity, S, |dL| <= 1, not S-S)")
        data, _, _ = ls_level_data(groups[0] + groups[2], groups[1] + groups[3], couplings, g_L, g_s)
        index = lambda group: [k for t in group for k, lv in data.levels.items()   # noqa: E731
                               if lv.configuration == t.label]
        self._term_of = {k: lv.configuration for k, lv in data.levels.items()}
        level_index = {(lv.configuration, lv.J): k for k, lv in data.levels.items()}
        dia = ls_diamagnetic_elements(terms, level_index, diamagnetic) if diamagnetic else None
        super().__init__(data, index(groups[0]), index(groups[1]), B, emitter_charge=emitter_charge,
                         electron_widths_ev=electron_widths_ev, reference_energy_ev=reference_energy_ev,
                         collision_n=collision_n, collision_Z=collision_Z,
                         upper_perturbers=index(groups[2]), lower_perturbers=index(groups[3]),
                         collision_n_lower=collision_n_lower, mass_amu=mass_amu, diamagnetic=dia)

    @property
    def states(self):
        """(upper, lower) basis labels as (term label, J, M)."""
        su, sl = super().states
        return ([(self._term_of[i], J, M) for i, J, M in su], [(self._term_of[i], J, M) for i, J, M in sl])
