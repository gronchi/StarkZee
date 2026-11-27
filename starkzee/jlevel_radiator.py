"""J-level radiator: lines built directly from :class:`~starkzee.level_data.LevelData`.

For data without LS labels (FAC, Cowan, PPP files), and the common engine of the
LS radiator. The basis of each manifold is ``|level, J, M>``. With the Edmonds
Wigner-Eckart theorem,

* ``H0(B) = E_level + mu_B B <a J M| T_0 |b J' M>``, ``T = L + g_s S``, from the levels'
  g_J (diagonal) and the dataset's magnetic elements between levels (Paschen-Back
  mixing; optional), plus, when the reduced elements of r^2 C^0 and r^2 C^2 are
  given (``diamagnetic``), the diamagnetic term
  (e^2 B^2 / 8 m_e) r^2 sin^2(theta) = (e^2 B^2 / 12 m_e) r^2 (C^0 - C^2_0);
* the Stark templates (V = e F . r) come from the E1 elements between levels of
  the same manifold, the dipole matrices ``D_q[lower, upper] = <lower| r_q |upper>``
  from the elements between the manifolds.

Energies are in eV (``unit_to_ev = 1``). Stark mixing and the interference of
radiative paths need signed elements; unsigned ones raise for Stark pairs and
warn for radiative pairs.

Electron broadening: with ``collision_n`` (effective principal quantum number of
the upper manifold) the generalized impact operator of
:mod:`starkzee.level_collision` is used for every electron option of the solver
(scalar widths, per-state operator, full PPP interference). Perturbing levels
outside the manifolds (``upper_perturbers``, ``lower_perturbers``) add to the
self terms. Without ``collision_n``, constant widths ``electron_widths_ev``.
"""
import warnings

import numpy as np
from scipy.constants import hbar as _HBAR, e as _E_CHARGE

from starkzee.angular import wigner_3j
from starkzee.collision import complex_sdts_from_operator
from starkzee.level_collision import (
    ManifoldCollisionData, build_level_collision_operator, line_center_factor, scalar_width_ev,
)
from starkzee.radiators import Manifold
from starkzee.utils import A0, BOHR_MAGNETON_EV_T, E_CHARGE, M_E, core_mass_factors

_PPPB_MODELS = ("pppb", "ferri", "pppb-intra", "ferri-intra")


def _phase(x):
    k = int(round(x))
    return -1.0 if k % 2 else 1.0


def _we(J, M, J2, M2, q, reduced, k=1):
    """<J M| T^k_q |J2 M2> = (-1)^(J-M) (J k J2; -M q M2) <J||T^k||J2>."""
    return _phase(J - M) * wigner_3j(J, k, J2, -M, q, M2) * reduced


class JLevelTransition:
    """Radiator for a line between two sets of levels of a LevelData.

    Parameters
    ----------
    data : LevelData
    upper, lower : sequences of level indices of the two manifolds
    B : float
        Magnetic field [T]. For B != 0 every manifold level needs g_J.
    emitter_charge : float
        Net charge of the radiator (microfield selection), default 0.
    electron_widths_ev : (float, float), optional
        Constant electron impact HWHM of the upper and lower manifold [eV]
        (used when collision_n is None; default 0).
    reference_energy_ev : float, optional
        Line reference [eV] for the Doppler width and electron detuning; default:
        difference of the mean (over ``|J M>``) manifold energies.
    collision_n : float, optional
        Effective principal quantum number of the upper manifold for the impact
        operator (C_n and GBK argument). Enables the collision operator.
    collision_Z : float, optional
        Charge in the GBK argument (default emitter_charge + 1, the core charge
        seen by the active electron).
    upper_perturbers, lower_perturbers : sequences of level indices
        Levels outside the manifolds whose E1 couplings to them add to the
        collision self terms (e.g. 4p/4f for a 4d manifold, or Delta n != 0 levels).
    mass_amu : float, optional
        Emitter mass [u]: Doppler width and the reduced mass of the diamagnetic term.
    collision_n_lower : float, optional
        Effective n of the lower manifold for its scalar and per-state widths
        (default collision_n). The full PPP operator uses collision_n for all terms.
    diamagnetic : dict, optional
        {(a, b): {0: <a||r^2 C^0||b>, 2: <a||r^2 C^2||b>}} [a0^2], the reduced
        elements of the radiating electron between levels of one manifold
        (each unordered pair once, diagonal pairs included; missing ranks are
        zero). Adds the diamagnetic term to H0 for B != 0; without it the
        Zeeman term is linear only.
    zeeman : {"exact", "weak"}
        "exact" (default): all magnetic elements of the dataset, including those
        between J levels (intermediate field / Paschen-Back mixing). "weak": the
        field acts inside each J level only (g_J, anomalous Zeeman effect; the
        diamagnetic term likewise). The strong-field limit needs term structure:
        see :class:`~starkzee.ls_radiator.LSTransition`.
    level_couplings : dict, optional
        {(a, b): V} [eV], field-free couplings between levels of one manifold
        with equal J and parity (e.g. spin-orbit mixing of singlet and triplet
        levels), added to H0 for every M. The level energies are then the
        diagonal (unmixed) energies. Needs zeeman="exact".
    """

    def __init__(self, data, upper, lower, B, emitter_charge=0.0, electron_widths_ev=None,
                 reference_energy_ev=None, collision_n=None, collision_Z=None,
                 upper_perturbers=(), lower_perturbers=(), collision_n_lower=None, mass_amu=None,
                 diamagnetic=None, zeeman="exact", level_couplings=None):
        groups = [set(upper), set(lower), set(upper_perturbers), set(lower_perturbers)]
        if groups[0] & groups[1] or (groups[0] | groups[1]) & (groups[2] | groups[3]):
            raise ValueError("a level cannot be in two manifolds or be both radiating and perturbing")
        if zeeman not in ("exact", "weak"):
            raise ValueError("JLevelTransition supports zeeman='exact' or 'weak' (strong: LSTransition)")
        self.zeeman = zeeman
        self.data = data
        self.level_couplings = dict(level_couplings or {})
        for (a, b) in self.level_couplings:
            la, lb = data.levels[a], data.levels[b]
            if a == b or la.two_j != lb.two_j or la.parity != lb.parity:
                raise ValueError(f"level coupling {a} - {b} needs two levels of equal J and parity")
            if not ((a in groups[0] and b in groups[0]) or (a in groups[1] and b in groups[1])):
                raise ValueError(f"level coupling {a} - {b} must join two levels of one manifold")
        if self.level_couplings and zeeman != "exact":
            raise ValueError("level_couplings need zeeman='exact' (the weak treatment projects on the "
                             "unmixed J levels)")
        self.diamagnetic = dict(diamagnetic or {})
        self.mass_amu = mass_amu
        self.default_emitter_charge = emitter_charge
        self._electron = (0.0, 0.0) if electron_widths_ev is None else tuple(float(w) for w in electron_widths_ev)
        self.collision_n = collision_n
        self.collision_n_lower = collision_n if collision_n_lower is None else collision_n_lower
        self.collision_Z = (emitter_charge + 1.0) if collision_Z is None else float(collision_Z)
        self._states_u = self._states(upper)
        self._states_l = self._states(lower)
        self.upper = self._manifold(self._states_u, B)
        self.lower = self._manifold(self._states_l, B)
        self.dipoles = self._dipoles()
        self.unit_to_ev = 1.0
        self.E0_line = self.upper.E_ref - self.lower.E_ref
        self.E0_physical = self.E0_line if reference_energy_ev is None else float(reference_energy_ev)
        self._collision_u = self._collision(self._states_u, self._states(upper_perturbers))
        self._collision_l = self._collision(self._states_l, self._states(lower_perturbers))
        self._plasma = None

    # ---------------------------------------------------------------- construction
    def _states(self, indices):
        states = []
        for i in indices:
            lv = self.data.levels[i]
            for k in range(lv.two_j + 1):
                states.append((i, lv.J, -lv.J + k))
        return states

    def _warn_weak_field(self, states, mu):
        """Without off-diagonal magnetic elements only g_J enters: warn if B mixes J levels."""
        data = self.data
        idx = sorted({i for i, _, _ in states})
        for k, a in enumerate(idx):
            for b in idx[k + 1:]:
                la, lb = data.levels[a], data.levels[b]
                if (la.parity == lb.parity and abs(la.two_j - lb.two_j) <= 2
                        and abs(la.energy_ev - lb.energy_ev) < 10.0 * mu):
                    warnings.warn(
                        f"levels {a} and {b} are {abs(la.energy_ev - lb.energy_ev):.2e} eV apart, not large "
                        f"next to mu_B B = {mu:.2e} eV, but the dataset has no off-diagonal magnetic "
                        "elements (e.g. FAC tables): the g_J-only Zeeman term misses their B mixing",
                        UserWarning, stacklevel=4)
                    return

    def _manifold(self, states, B):
        data = self.data
        E = np.array([data.levels[i].energy_ev for i, _, _ in states])
        H = np.diag(E).astype(complex)
        if self.level_couplings:
            position = {s: k for k, s in enumerate(states)}
            for (a, b), V in self.level_couplings.items():
                for (i, J, M), k in position.items():
                    if i != a:
                        continue
                    k2 = position.get((b, J, M))
                    if k2 is not None:
                        H[k, k2] += V
                        H[k2, k] += np.conj(V)
        if B != 0.0:
            mu = BOHR_MAGNETON_EV_T * B
            weak = self.zeeman == "weak"
            if not data.magnetic and not weak:
                self._warn_weak_field(states, mu)
            for i, (a, J, M) in enumerate(states):
                for j, (b, J2, M2) in enumerate(states):
                    if abs(M - M2) > 1e-9 or abs(J - J2) > 1 + 1e-9 or (weak and a != b):
                        continue
                    red = data.magnetic_element(a, b)
                    if red is None:
                        raise ValueError(f"level {a} has no g_J; needed for B != 0")
                    if red:
                        H[i, j] += mu * _we(J, M, J2, M2, 0, red)
            if self.diamagnetic:
                H += self._diamagnetic(states, B)
        E_ref = float(E.mean())
        H -= E_ref * np.eye(len(H))
        r_x, r_y, r_z = self._cartesian(states, states, require_sign=True)
        rates = np.array([data.decay_rate(i) for i, _, _ in states])
        return Manifold(H, E_ref, r_z * A0, r_x * A0, rates, np.array([M for _, _, M in states]))

    def _diamagnetic_reduced(self, a, b, k):
        """<a||r^2 C^k||b> [a0^2] in either orientation (Hermitian tensor: (-1)^(J_a - J_b))."""
        if (a, b) in self.diamagnetic:
            return self.diamagnetic[(a, b)].get(k, 0.0)
        if (b, a) in self.diamagnetic:
            la, lb = self.data.levels[a], self.data.levels[b]
            return _phase((la.two_j - lb.two_j) / 2) * self.diamagnetic[(b, a)].get(k, 0.0)
        return 0.0

    def _diamagnetic(self, states, B):
        """(e^2 B^2 / 12 mu) <i| r^2 (C^0 - C^2_0) |j> [eV] in the |level J M> basis.

        mu is the reduced mass of the active electron for the atom mass mass_amu
        (m_e without it); the r^2 elements are used as given [a0^2].
        """
        mu = core_mass_factors(self.mass_amu)[0]
        coefficient = E_CHARGE * B**2 * A0**2 / (12.0 * M_E * mu)     # e^2 B^2 a0^2 / (12 mu), in eV
        dim = len(states)
        H = np.zeros((dim, dim))
        for i, (a, J, M) in enumerate(states):
            for j, (b, J2, M2) in enumerate(states):
                if abs(M - M2) > 1e-9 or (self.zeeman == "weak" and a != b):
                    continue
                value = 0.0
                if abs(J - J2) < 1e-9:
                    value += _we(J, M, J2, M2, 0, self._diamagnetic_reduced(a, b, 0), k=0)
                if abs(J - J2) <= 2 + 1e-9:
                    value -= _we(J, M, J2, M2, 0, self._diamagnetic_reduced(a, b, 2), k=2)
                H[i, j] = coefficient * value
        return H

    def _cartesian(self, bra_states, ket_states, require_sign=False):
        """x, y, z matrices <bra| r_c |ket> [a0] from the dataset's E1 elements."""
        nb, nk = len(bra_states), len(ket_states)
        sph = {q: np.zeros((nb, nk), dtype=complex) for q in (-1, 0, 1)}
        bra_levels = {i for i, _, _ in bra_states}
        ket_levels = {i for i, _, _ in ket_states}
        for t in self.data.e1:
            for a, b in ((t.lower, t.upper), (t.upper, t.lower)):
                if a not in bra_levels or b not in ket_levels:
                    continue
                if require_sign and not t.signed:
                    raise ValueError(f"Stark pair {t.lower}-{t.upper} has no sign (unsigned element)")
                red = self.data.element(a, b)
                for i, (s, J, M) in enumerate(bra_states):
                    if s != a:
                        continue
                    for j, (s2, J2, M2) in enumerate(ket_states):
                        if s2 == b:
                            for q in (-1, 0, 1):
                                sph[q][i, j] = _we(J, M, J2, M2, q, red)
        x = (sph[-1] - sph[1]) / np.sqrt(2.0)
        y = 1j * (sph[-1] + sph[1]) / np.sqrt(2.0)
        return x, y, sph[0]

    def _dipoles(self):
        su, sl = self._states_u, self._states_l
        up, lo = {i for i, _, _ in su}, {i for i, _, _ in sl}
        D = {q: np.zeros((len(sl), len(su)), dtype=complex) for q in (-1, 0, 1)}
        unsigned = False
        for t in self.data.e1:
            if (t.lower in lo and t.upper in up) or (t.lower in up and t.upper in lo):
                a, b = (t.lower, t.upper) if t.lower in lo else (t.upper, t.lower)
                red = self.data.element(a, b)              # <lower-manifold level||d||upper-manifold level>
                unsigned |= not t.signed
                for i, (s, J, M) in enumerate(sl):
                    if s != a:
                        continue
                    for j, (s2, J2, M2) in enumerate(su):
                        if s2 != b:
                            continue
                        q = int(round(M - M2))
                        if abs(q) <= 1:
                            D[q][i, j] = _we(J, M, J2, M2, q, red)
        if unsigned:
            warnings.warn("radiative E1 elements without sign: interference between radiative "
                          "paths of mixed states is not represented", UserWarning, stacklevel=3)
        return D

    def _collision(self, states, perturbers):
        ext = states + perturbers
        r = self._cartesian(ext, ext)
        E = np.array([self.data.levels[i].energy_ev for i, _, _ in ext])
        return ManifoldCollisionData(r, E, len(states))

    # ---------------------------------------------------------------- widths
    def shell_natural_width_ev(self):
        return _HBAR * (self.upper.decay_rates.mean() + self.lower.decay_rates.mean()) / (2.0 * _E_CHARGE)

    def _require_collision(self, what):
        if self.collision_n is None:
            raise NotImplementedError(f"{what} needs the collision operator: give collision_n "
                                      "(effective n of the upper manifold)")

    def electron_widths(self, detuning_ev, Ne_m3, Te_ev, B, electron_model):
        """Impact widths (upper, lower) [eV] at the detuning from the line reference."""
        if self.collision_n is None:
            shape = np.shape(detuning_ev)
            w_u, w_l = self._electron
            return (np.full(shape, w_u) if shape else w_u), (np.full(shape, w_l) if shape else w_l)
        if electron_model.lower() not in _PPPB_MODELS:
            raise NotImplementedError("level radiators support the PPPB/Ferri electron models only")
        Z = self.collision_Z
        return (scalar_width_ev(detuning_ev, self._collision_u.r2_mean, Ne_m3, Te_ev, B, self.collision_n, Z),
                scalar_width_ev(detuning_ev, self._collision_l.r2_mean, Ne_m3, Te_ev, B, self.collision_n_lower, Z))

    def electron_r2_scaling(self, vectors_u, vectors_l, r2_form):
        """Per-dressed-state <k|r.r|k>/<r^2> (closure over the manifold and its perturbers)."""
        self._require_collision("electron_operator")
        scale_l = self._collision_l.r2_scaling(vectors_l) if self._collision_l.r2_mean else None
        return self._collision_u.r2_scaling(vectors_u), scale_l

    def ppp_impact_coefficient(self, Ne_m3, Te_ev, B):
        """Line-centre impact factor; also records the plasma for the full operator."""
        self._require_collision("electron_interference")
        self._plasma = (Ne_m3, Te_ev, B)
        return line_center_factor(Ne_m3, Te_ev, B, self.collision_n, self.collision_Z)

    def ppp_self_term_r2(self):
        return self._collision_u.r2_mean + self._collision_l.r2_mean

    def ppp_sdts(self, dE, vectors_u, vectors_l, coefficient, natural, m_values=None):
        """Complex generalized SDTs with the generalized impact operator (m_values: see HydrogenicTransition)."""
        if self._plasma is None:
            raise RuntimeError("ppp_impact_coefficient must be called first")
        Ne_m3, Te_ev, B = self._plasma
        phi = build_level_collision_operator(vectors_u, vectors_l, self._collision_u, self._collision_l,
                                             Ne_m3, Te_ev, B, self.collision_n, self.collision_Z)
        return complex_sdts_from_operator(phi, dE, vectors_u, vectors_l, self.dipoles, natural,
                                          m_values=m_values)

    @property
    def states(self):
        """(upper, lower) basis labels as (level index, J, M)."""
        return list(self._states_u), list(self._states_l)
