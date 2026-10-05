# multielectron.py — Unfinished stubs for multi-electron (quantum-defect) systems.
#
# NOT connected to the main profile pipeline.  See FUTURE_MULTIELECTRON_CIV.md
# for the implementation roadmap (Option C).

import math
from dataclasses import dataclass
import numpy as np
from starkzee.utils import BOHR_MAGNETON_EV_T


def wigner_3j(j1, j2, j3, m1, m2, m3):
    """Return the Wigner 3-j symbol using the Racah sum formula.

    The 3-j symbol encodes the coupling of two angular momenta and is related
    to the Clebsch-Gordan coefficients by:

        ⟨j₁ m₁ j₂ m₂ | j₃ −m₃⟩ = (−1)^{j₁−j₂+m₃} √(2j₃+1) ×
                                    (j₁  j₂  j₃ )
                                    (m₁  m₂  m₃ )

    Selection rules enforced before the Racah sum:
    - m₁ + m₂ + m₃ = 0
    - Triangle condition: abs(j₁ − j₂) ≤ j₃ ≤ j₁ + j₂
    - abs(mᵢ) ≤ jᵢ for each i

    Parameters
    ----------
    j1, j2, j3 : float
        Angular momentum quantum numbers (integer or half-integer).
    m1, m2, m3 : float
        Magnetic quantum numbers (integer or half-integer).

    Returns
    -------
    float
        Wigner 3-j symbol value.  Returns 0 if any selection rule is violated
        or if the arguments are not valid (half-)integers.

    References
    ----------
    Racah, G., Phys. Rev. 62, 438 (1942).
    """
    if m1 + m2 + m3 != 0:
        return 0.0
    if not (abs(j1 - j2) <= j3 <= j1 + j2):
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0

    def is_valid(x):
        return (2 * x) == int(2 * x)
    if not (is_valid(j1) and is_valid(j2) and is_valid(j3)
            and is_valid(m1) and is_valid(m2) and is_valid(m3)):
        return 0.0

    def tri_coeff(a, b, c):
        num = (math.factorial(int(a + b - c))
               * math.factorial(int(a - b + c))
               * math.factorial(int(-a + b + c)))
        den = math.factorial(int(a + b + c + 1))
        return num / den

    try:
        t_min = max(0, int(j2 - j3 - m1), int(j1 - j3 + m2))
        t_max = min(int(j1 + j2 - j3), int(j1 - m1), int(j2 + m2))

        sum_val = 0.0
        for t in range(t_min, t_max + 1):
            term = ((-1)**t) / (
                math.factorial(t)
                * math.factorial(int(j1 + j2 - j3 - t))
                * math.factorial(int(j1 - m1 - t))
                * math.factorial(int(j2 + m2 - t))
                * math.factorial(int(j3 - j2 + m1 + t))
                * math.factorial(int(j3 - j1 - m2 + t))
            )
            sum_val += term

        prefactor = ((-1)**int(j1 - j2 - m3)) * math.sqrt(tri_coeff(j1, j2, j3)) * math.sqrt(
            math.factorial(int(j1 + m1)) * math.factorial(int(j1 - m1))
            * math.factorial(int(j2 + m2)) * math.factorial(int(j2 - m2))
            * math.factorial(int(j3 + m3)) * math.factorial(int(j3 - m3))
        )
        return prefactor * sum_val
    except ValueError:
        return 0.0


def wigner_6j(j1, j2, j3, j4, j5, j6):
    """Return a Wigner 6-j symbol using the Racah sum formula.

    Integer and half-integer arguments are supported.  Invalid angular-
    momentum triangles return zero, following the convention used by
    :func:`wigner_3j`.  The implementation is intentionally dependency-free so
    the J-coupled hydrogenic and future multi-electron paths use the same
    Condon--Shortley angular convention.
    """
    values = (j1, j2, j3, j4, j5, j6)
    if any(value < 0 or not math.isclose(2.0 * value, round(2.0 * value),
                                         abs_tol=1e-12)
           for value in values):
        return 0.0

    def triangle(a, b, c):
        return (a + b >= c and a + c >= b and b + c >= a
                and math.isclose(a + b + c, round(a + b + c),
                                 abs_tol=1e-12))

    triangles = ((j1, j2, j3), (j1, j5, j6),
                 (j4, j2, j6), (j4, j5, j3))
    if not all(triangle(*triple) for triple in triangles):
        return 0.0

    def factorial(value):
        nearest = round(value)
        if value < -1e-12 or not math.isclose(value, nearest, abs_tol=1e-12):
            raise ValueError("factorial argument is not a non-negative integer")
        return math.factorial(int(nearest))

    def delta(a, b, c):
        return math.sqrt(
            factorial(a + b - c)
            * factorial(a - b + c)
            * factorial(-a + b + c)
            / factorial(a + b + c + 1.0)
        )

    lower = max(j1 + j2 + j3, j1 + j5 + j6,
                j4 + j2 + j6, j4 + j5 + j3)
    upper = min(j1 + j2 + j4 + j5, j1 + j3 + j4 + j6,
                j2 + j3 + j5 + j6)
    z_min = int(round(lower))
    z_max = int(round(upper))
    if z_min > z_max:
        return 0.0

    total = 0.0
    for z in range(z_min, z_max + 1):
        denominator = (
            factorial(z - j1 - j2 - j3)
            * factorial(z - j1 - j5 - j6)
            * factorial(z - j4 - j2 - j6)
            * factorial(z - j4 - j5 - j3)
            * factorial(j1 + j2 + j4 + j5 - z)
            * factorial(j1 + j3 + j4 + j6 - z)
            * factorial(j2 + j3 + j5 + j6 - z)
        )
        total += (-1.0)**z * factorial(z + 1.0) / denominator

    return math.prod(delta(*triple) for triple in triangles) * total


@dataclass(frozen=True)
class MultiElectronState:
    """Basis state |level_idx, J, M_J⟩ for a multi-electron (quantum-defect) system."""
    index:     int
    level_idx: int
    J:         float
    mj:        float

    def __repr__(self):
        return f"|level={self.level_idx}, J={self.J}, M_J={self.mj:.1f}>"


def build_multielectron_basis(levels):
    """Return the full |level_idx, J, M_J⟩ basis generated from loaded levels.

    Parameters
    ----------
    levels : dict {int: ParsedAtomicLevel}
        Levels loaded by :func:`~starkzee.atomic_loader.load_atomic_database`.

    Returns
    -------
    list of MultiElectronState
    """
    basis = []
    idx = 0
    for l_idx, lvl in levels.items():
        J   = 0.5 * (lvl.g - 1)          # g = 2J + 1
        mjs = np.arange(-J, J + 1)
        for mj in mjs:
            basis.append(MultiElectronState(idx, l_idx, J, mj))
            idx += 1
    return basis


def build_multielectron_hamiltonian(levels, B):
    """Build the Hamiltonian matrix for multi-electron levels in magnetic field B.

    Parameters
    ----------
    levels : dict {int: ParsedAtomicLevel}
    B : float
        Magnetic field [T].

    Returns
    -------
    H : ndarray, complex
        Hamiltonian matrix in eV.
    basis : list of MultiElectronState
    """
    basis = build_multielectron_basis(levels)
    dim   = len(basis)
    H     = np.zeros((dim, dim), dtype=complex)

    for i, state in enumerate(basis):
        H[i, i] += levels[state.level_idx].energy_ev

    g_J = 1.0   # placeholder; replace with Landé g-factor from atomic data
    for i, state in enumerate(basis):
        H[i, i] += g_J * state.mj * BOHR_MAGNETON_EV_T * B

    return H, basis
