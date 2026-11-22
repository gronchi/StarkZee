"""J-level radiator: lines built directly from :class:`~starkzee.level_data.LevelData`.

For data without LS labels (FAC, Cowan, PPP files). The basis of each manifold
is |level, J, M>. With the Edmonds Wigner-Eckart theorem,

* H0(B) = E_level + mu_B B <a J M| T_0 |b J' M>, T = L + g_s S, from the levels'
  g_J (diagonal) and the dataset's magnetic elements between levels (Paschen-Back
  mixing; optional);
* the Stark templates (V = e F . r) come from the E1 elements between levels of
  the same manifold, the dipole matrices D_q[lower, upper] = <lower| r_q |upper>
  from the elements between the manifolds.

Energies are in eV (``unit_to_ev = 1``). Stark mixing and the interference of
radiative paths need signed elements; unsigned ones raise for Stark pairs and
warn for radiative pairs. Electron broadening: constant widths per manifold, as
for :class:`~starkzee.ls_radiator.LSTransition`.
"""
import warnings

import numpy as np
from scipy.constants import hbar as _HBAR, e as _E_CHARGE

from starkzee.angular import wigner_3j
from starkzee.radiators import Manifold
from starkzee.utils import A0, BOHR_MAGNETON_EV_T


def _phase(x):
    k = int(round(x))
    return -1.0 if k % 2 else 1.0


def _we(J, M, J2, M2, q, reduced):
    """<J M| T_q |J2 M2> = (-1)^(J-M) (J 1 J2; -M q M2) <J||T||J2>."""
    return _phase(J - M) * wigner_3j(J, 1, J2, -M, q, M2) * reduced


class JLevelTransition:
    """Radiator for a line between two sets of levels of a LevelData.

    Parameters
    ----------
    data : LevelData
    upper, lower : sequences of level indices of the two manifolds
    B : float
        Magnetic field [T]. For B != 0 every level needs g_J.
    emitter_charge : float
        Net charge of the radiator (microfield selection), default 0.
    electron_widths_ev : (float, float)
        Constant electron impact HWHM of the upper and lower manifold [eV].
    reference_energy_ev : float, optional
        Line reference [eV] for the Doppler width and electron detuning; default:
        difference of the mean (over |J M>) manifold energies.
    """

    def __init__(self, data, upper, lower, B, emitter_charge=0.0, electron_widths_ev=(0.0, 0.0),
                 reference_energy_ev=None):
        if set(upper) & set(lower):
            raise ValueError("a level cannot be in both manifolds")
        self.data = data
        self.default_emitter_charge = emitter_charge
        self._electron = tuple(float(w) for w in electron_widths_ev)
        self._states_u = self._states(upper)
        self._states_l = self._states(lower)
        self.upper = self._manifold(self._states_u, B)
        self.lower = self._manifold(self._states_l, B)
        self.dipoles = self._dipoles()
        self.unit_to_ev = 1.0
        self.E0_line = self.upper.E_ref - self.lower.E_ref
        self.E0_physical = self.E0_line if reference_energy_ev is None else float(reference_energy_ev)

    def _states(self, indices):
        states = []
        for i in indices:
            lv = self.data.levels[i]
            for k in range(lv.two_j + 1):
                states.append((i, lv.J, -lv.J + k))
        return states

    def _manifold(self, states, B):
        data = self.data
        E = np.array([data.levels[i].energy_ev for i, _, _ in states])
        H = np.diag(E).astype(complex)
        if B != 0.0:
            mu = BOHR_MAGNETON_EV_T * B
            for i, (a, J, M) in enumerate(states):
                for j, (b, J2, M2) in enumerate(states):
                    if abs(M - M2) > 1e-9 or abs(J - J2) > 1 + 1e-9:
                        continue
                    red = data.magnetic_element(a, b)
                    if red is None:
                        raise ValueError(f"level {a} has no g_J; needed for B != 0")
                    if red:
                        H[i, j] += mu * _we(J, M, J2, M2, 0, red)
        E_ref = float(E.mean())
        H -= E_ref * np.eye(len(H))
        dim = len(states)
        r_z = np.zeros((dim, dim), dtype=complex)
        r_x = np.zeros((dim, dim), dtype=complex)
        levels = {i for i, _, _ in states}
        for t in data.e1:
            if t.lower in levels and t.upper in levels:
                if not t.signed:
                    raise ValueError(f"Stark pair {t.lower}-{t.upper} has no sign (unsigned element)")
                self._fill_stark(states, t.lower, t.upper, t.reduced, r_z, r_x)
        rates = np.array([data.decay_rate(i) for i, _, _ in states])
        return Manifold(H, E_ref, r_z * A0, r_x * A0, rates)

    @staticmethod
    def _fill_stark(states, a, b, reduced, r_z, r_x):
        for i, (s, J, M) in enumerate(states):
            if s != a:
                continue
            for j, (s2, J2, M2) in enumerate(states):
                if s2 != b:
                    continue
                z = _we(J, M, J2, M2, 0, reduced)
                x = (_we(J, M, J2, M2, -1, reduced) - _we(J, M, J2, M2, 1, reduced)) / np.sqrt(2.0)
                r_z[i, j] += z
                r_z[j, i] += np.conj(z)
                r_x[i, j] += x
                r_x[j, i] += np.conj(x)

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

    # ---------------------------------------------------------------- widths
    def shell_natural_width_ev(self):
        return _HBAR * (self.upper.decay_rates.mean() + self.lower.decay_rates.mean()) / (2.0 * _E_CHARGE)

    def electron_widths(self, detuning_ev, Ne_m3, Te_ev, B, electron_model):
        shape = np.shape(detuning_ev)
        w_u, w_l = self._electron
        return (np.full(shape, w_u) if shape else w_u), (np.full(shape, w_l) if shape else w_l)

    def electron_r2_scaling(self, vectors_u, vectors_l, r2_form):
        raise NotImplementedError("electron_operator needs the generalized collision operator "
                                  "(plan B4); not yet available for J-level radiators")

    def ppp_impact_coefficient(self, Ne_m3, Te_ev, B):
        raise NotImplementedError("electron_interference needs the generalized collision operator "
                                  "(plan B4); not yet available for J-level radiators")

    ppp_self_term_r2 = ppp_sdts = ppp_impact_coefficient

    @property
    def states(self):
        """(upper, lower) basis labels as (level index, J, M)."""
        return list(self._states_u), list(self._states_l)
