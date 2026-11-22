# multielectron.py — Unfinished stubs for multi-electron (quantum-defect) systems.
#
# NOT connected to the main profile pipeline.  See FUTURE_MULTIELECTRON_CIV.md
# for the implementation roadmap (Option C).

from dataclasses import dataclass
import numpy as np
from starkzee.utils import BOHR_MAGNETON_EV_T



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
