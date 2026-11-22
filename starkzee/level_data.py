"""Neutral J-level atomic data filled by the file readers (FAC now; Cowan, PPP, ADAS later).

A dataset holds levels (energy, J, parity, labels) and E1 reduced elements between
them. Reduced elements follow the Edmonds / Wigner-Eckart convention used in
:mod:`starkzee.ls_radiator`,

    <J M| d_q |J' M'> = (-1)^(J-M) (J 1 J'; -M q M') <J||d||J'>,
    S = |<J||d||J'>|^2 [(e a0)^2],   gf = (2/3) dE[Hartree] S,

stored with the orientation <lower||d||upper>; the other one follows from
<upper||d||lower> = (-1)^(J_upper - J_lower) <lower||d||upper>.

A sign is meaningful only relative to the other elements of the same dataset
(level phases are a gauge freedom; products around closed loops are not), so
``signed`` records whether the source gave it. Unsigned elements carry |<J||d||J'>|
from gf and must not be used where interference between paths matters (Stark
mixing) without a sign from elsewhere.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from starkzee.utils import HARTREE_EV


@dataclass(frozen=True)
class JLevel:
    """One level. energy_ev is relative to the dataset's reference (usually the ground level)."""
    index: int
    energy_ev: float
    two_j: int
    parity: int                         # +1 even, -1 odd
    configuration: str = ""
    label: str = ""
    n_valence: Optional[int] = None
    l_valence: Optional[int] = None
    nele: Optional[int] = None
    g_J: Optional[float] = None          # Lande factor (diagonal Zeeman element)
    decay_rate_s: Optional[float] = None  # total E1 decay rate to all lower levels [s^-1]

    @property
    def J(self):
        return self.two_j / 2


@dataclass(frozen=True)
class MagneticElement:
    """Off-diagonal reduced element <a||L + g_s S||b> [mu_B] between levels of equal parity.

    The diagonal elements come from the levels' g_J: <a||..||a> = g_J sqrt(J(J+1)(2J+1)).
    """
    a: int
    b: int
    reduced: float


@dataclass(frozen=True)
class E1Element:
    """E1 reduced element <lower||d||upper> [e a0] between two levels of the dataset."""
    lower: int
    upper: int
    reduced: float
    signed: bool
    gf: Optional[float] = None
    rate_s: Optional[float] = None       # spontaneous rate upper -> lower [s^-1]
    energy_ev: Optional[float] = None


@dataclass
class LevelData:
    levels: dict                        # index -> JLevel
    e1: list = field(default_factory=list)
    magnetic: list = field(default_factory=list)   # MagneticElement, off-diagonal only
    source: str = ""
    Z: Optional[float] = None
    reference_energy_ev: Optional[float] = None   # absolute energy of the zero of energy_ev, if known
    notes: list = field(default_factory=list)

    def element(self, a, b):
        """<a||d||b> for any orientation (None if the pair is absent)."""
        for t in self.e1:
            if (t.lower, t.upper) == (a, b):
                return t.reduced
            if (t.lower, t.upper) == (b, a):
                la, lb = self.levels[a], self.levels[b]
                return (-1.0 if ((la.two_j - lb.two_j) // 2) % 2 else 1.0) * t.reduced
        return None

    def magnetic_element(self, a, b):
        """<a||L + g_s S||b> [mu_B] in either orientation; diagonal from g_J (None if unknown)."""
        if a == b:
            lv = self.levels[a]
            if lv.g_J is None:
                return None
            J = lv.J
            return lv.g_J * float(np.sqrt(J * (J + 1) * (2 * J + 1)))
        for t in self.magnetic:
            if (t.a, t.b) == (a, b):
                return t.reduced
            if (t.a, t.b) == (b, a):
                la, lb = self.levels[a], self.levels[b]
                return (-1.0 if ((la.two_j - lb.two_j) // 2) % 2 else 1.0) * t.reduced
        return 0.0

    def decay_rate(self, index):
        """Level decay rate [s^-1]: the level's own value, else the sum of the dataset's E1 rates."""
        lv = self.levels[index]
        if lv.decay_rate_s is not None:
            return lv.decay_rate_s
        return sum(t.rate_s or 0.0 for t in self.e1 if t.upper == index)

    @property
    def all_signed(self):
        return all(t.signed for t in self.e1)

    def validate(self, energy_rtol=1e-5):
        """Selection rules and energy consistency; raises ValueError on the first problem."""
        for t in self.e1:
            lo, up = self.levels[t.lower], self.levels[t.upper]
            if lo.parity == up.parity:
                raise ValueError(f"E1 {t.lower}-{t.upper}: same parity")
            if abs(lo.two_j - up.two_j) > 2 or (lo.two_j == 0 and up.two_j == 0):
                raise ValueError(f"E1 {t.lower}-{t.upper}: J selection rule violated")
            if t.energy_ev is not None:
                de = up.energy_ev - lo.energy_ev
                if not np.isclose(t.energy_ev, de, rtol=energy_rtol, atol=1e-9):
                    raise ValueError(f"E1 {t.lower}-{t.upper}: transition energy {t.energy_ev} "
                                     f"differs from the level difference {de}")
        return self


def reduced_from_gf(gf, energy_ev):
    """|<J||d||J'>| [e a0] from gf and the transition energy [eV]: S = 3 gf / (2 dE[Ha])."""
    return float(np.sqrt(1.5 * gf / (energy_ev / HARTREE_EV)))
