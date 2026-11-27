import json
import os
import warnings
from functools import lru_cache
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class AtomicState:
    """Explicitly represents an atomic quantum state."""
    energy: float
    n: int
    l: Optional[int] = None
    j: Optional[float] = None
    spin: Optional[float] = None

    def __repr__(self):
        parts = [f"n={self.n}"]
        if self.l is not None:
            parts.append(f"l={self.l}")
        if self.j is not None:
            parts.append(f"j={self.j}")
        if self.spin is not None:
            parts.append(f"s={self.spin}")
        parts_str = ", ".join(parts)
        return f"AtomicState({parts_str}, energy={self.energy})"


def get_data_path() -> str:
    """Returns the absolute path to the data directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "data", "atomic_levels.json")


def load_levels(atom: str, fine_structure: bool) -> List[AtomicState]:
    r"""Load atomic states from the JSON database.

    Parameters
    ----------
    atom : str
        Chemical symbol of the element (e.g. ``'H'``, ``'He'``).
    fine_structure : bool
        If ``True``, load :math:`(n, l, j)`-resolved levels
        (``fine_structure_true`` section); otherwise load shell-averaged
        levels (``fine_structure_false`` section).

    Returns
    -------
    list of AtomicState
        Atomic states ordered as stored in the JSON file, with energies
        in :math:`\mathrm{cm}^{-1}` above the ground state.

    Raises
    ------
    ValueError
        If *atom* or the requested fine-structure section is absent from
        the database.
    """
    data_path = get_data_path()
    
    with open(data_path, 'r') as f:
        database = json.load(f)
        
    if atom not in database:
        raise ValueError(f"Atom '{atom}' not found in database.")
        
    config_key = "fine_structure_true" if fine_structure else "fine_structure_false"
    
    if config_key not in database[atom]:
        raise ValueError(f"Configuration '{config_key}' not found for atom '{atom}'.")
        
    raw_levels = database[atom][config_key]
    
    states = []
    for level_data in raw_levels:
        state = AtomicState(
            energy=level_data["energy"],
            n=level_data["n"],
            l=level_data.get("l"),
            j=level_data.get("j"),
            spin=level_data.get("spin")
        )
        states.append(state)
        
    return states


def empirical_shell_energy_cm(atom: str, n: int) -> float:
    """Tabulated shell energy, or a degeneracy-weighted complete fine shell."""
    for state in load_levels(atom, fine_structure=False):
        if state.n == n:
            return state.energy
    states = {(s.l, s.j): s for s in load_levels(atom, fine_structure=True)
              if s.n == n}
    expected = {(0, 0.5)} | {(l, j) for l in range(1, n)
                             for j in (l - 0.5, l + 0.5)}
    if set(states) != expected:
        raise ValueError(f"No empirical complete shell for n={n} under atom={atom!r}.")
    return sum((2 * j + 1) * states[l, j].energy for l, j in expected) / (2 * n**2)


def calculate_wavenumber(upper_state: AtomicState, lower_state: AtomicState, lambda_shift: float = 0.0) -> float:
    r"""Compute the transition wavenumber between two atomic states.

    Parameters
    ----------
    upper_state : AtomicState
        Upper energy level.
    lower_state : AtomicState
        Lower energy level.
    lambda_shift : float, optional
        Additional shift added to the computed wavenumber (default 0.0).

    Returns
    -------
    float
        Transition wavenumber :math:`E_u - E_l + \lambda_\mathrm{shift}`
        in :math:`\mathrm{cm}^{-1}`.

    Raises
    ------
    ValueError
        If *upper_state* has lower energy than *lower_state*.
    """
    base_wavenumber = upper_state.energy - lower_state.energy
    if base_wavenumber < 0:
        raise ValueError("Upper state must have higher energy than lower state.")
        
    return base_wavenumber + lambda_shift


def dirac_binding_cm(n, j, rydberg_ev, Z=1):
    """Dirac level energy (negative, from the ionization limit) [cm^-1], reduced-mass Rydberg, no Lamb shift."""
    from starkzee.utils import FINE_STRUCTURE, energy_ev_to_wavenumber_cm
    e = -Z**2 * rydberg_ev / n**2 - Z**4 * FINE_STRUCTURE**2 * rydberg_ev / n**3 * (1.0 / (j + 0.5) - 0.75 / n)
    return float(energy_ev_to_wavenumber_cm(e))


@lru_cache(maxsize=None)
def ionization_limit_cm(atom, rydberg_ev):
    """Ionization limit [cm^-1 above the ground state] implied by the highest measured level.

    The measured level with the largest n (then largest l) has the smallest Lamb
    shift; its energy plus its Dirac binding energy gives the limit.
    """
    states = [s for s in load_levels(atom, fine_structure=True) if s.l is not None and s.j is not None]
    top = max(states, key=lambda s: (s.n, s.l))
    return top.energy - dirac_binding_cm(top.n, top.j, rydberg_ev)


def level_energy_cm(atom, n, l, j, rydberg_ev):
    """Measured (n, l, j) level [cm^-1 above the ground state], or ionization limit + Dirac energy.

    Levels missing from the table (e.g. D above n = 6, T above n = 3) fall back
    to the Dirac energy below the ionization limit of :func:`ionization_limit_cm`;
    the omitted Lamb shift is below 0.001 cm^-1 for n >= 7.
    """
    for s in load_levels(atom, fine_structure=True):
        if s.n == n and s.l == l and s.j == j:
            return s.energy
    warnings.warn(f"{atom}: no measured level n={n}, l={l}, j={j}; using the Dirac energy below the "
                  "ionization limit (no Lamb shift).", UserWarning, stacklevel=3)
    return ionization_limit_cm(atom, rydberg_ev) + dirac_binding_cm(n, j, rydberg_ev)


def shell_energy_cm(atom, n, rydberg_ev):
    """Tabulated shell energy, the (2j+1)-weighted mean of the measured fine levels, or the Dirac fallback."""
    try:
        return empirical_shell_energy_cm(atom, n)
    except ValueError:
        levels = [(0, 0.5)] + [(l, j) for l in range(1, n) for j in (l - 0.5, l + 0.5)]
        return sum((2 * j + 1) * level_energy_cm(atom, n, l, j, rydberg_ev) for l, j in levels) / (2 * n**2)
