import json
import os
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
    """
    Loads atomic states from the JSON database.
    
    Args:
        atom: The chemical symbol (e.g., 'H', 'He').
        fine_structure: Whether to load the fine structure sublevels.
        
    Returns:
        A list of AtomicState objects.
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


def calculate_wavenumber(upper_state: AtomicState, lower_state: AtomicState, lambda_shift: float = 0.0) -> float:
    """
    Computes the transition wavenumber, optionally including a lambda shift.
    
    Args:
        upper_state: The upper energy level.
        lower_state: The lower energy level.
        lambda_shift: An optional shift to apply to the resulting wavenumber.
        
    Returns:
        The transition wavenumber.
    """
    base_wavenumber = upper_state.energy - lower_state.energy
    if base_wavenumber < 0:
        raise ValueError("Upper state must have higher energy than lower state.")
        
    return base_wavenumber + lambda_shift
