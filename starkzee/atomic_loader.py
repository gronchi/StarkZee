# Loader module for arbitrary multi-electron atomic data for starkzee

from dataclasses import dataclass
import numpy as np


@dataclass
class ParsedAtomicLevel:
    """Atomic level loaded from an external file/database."""
    index:     int
    name:      str
    energy_ev: float
    g:         int    # statistical weight 2J+1
    n:         int    # principal quantum number
    l:         int    # orbital angular momentum

    def __post_init__(self):
        self.index     = int(self.index)
        self.name      = str(self.name)
        self.energy_ev = float(self.energy_ev)
        self.g         = int(self.g)
        self.n         = int(self.n)
        self.l         = int(self.l)

    def __repr__(self):
        return (f"Level(idx={self.index}, {self.name}, "
                f"E={self.energy_ev:.3f} eV, g={self.g}, n={self.n}, l={self.l})")


@dataclass
class ParsedDipoleTransition:
    """Dipole transition matrix element between two atomic levels."""
    upper_idx:          int
    lower_idx:          int
    dipole_strength_a0: float   # transition dipole element [a₀]

    def __post_init__(self):
        self.upper_idx          = int(self.upper_idx)
        self.lower_idx          = int(self.lower_idx)
        self.dipole_strength_a0 = float(self.dipole_strength_a0)

    def __repr__(self):
        return (f"Transition({self.upper_idx} -> {self.lower_idx}, "
                f"d={self.dipole_strength_a0:.4f} a₀)")

def load_atomic_database(levels_file_path, transitions_file_path):
    """Loads and parses levels and transition dipoles from tabular text files.
    
    Format of levels file:
    # index  name  energy_ev  g  n  l
    1        1s    0.0        2  1  0
    2        2s    10.2       2  2  0
    3        2p    10.201     6  2  1
    
    Format of transitions file:
    # upper_idx  lower_idx  dipole_strength_a0
    3            1          0.745
    """
    levels = {}
    transitions = []
    
    # 1. Parse levels
    with open(levels_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 6:
                idx = int(parts[0])
                level = ParsedAtomicLevel(
                    index=idx,
                    name=parts[1],
                    energy_ev=parts[2],
                    g=parts[3],
                    n=parts[4],
                    l=parts[5]
                )
                levels[idx] = level
                
    # 2. Parse transition dipoles
    with open(transitions_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                u_idx = int(parts[0])
                l_idx = int(parts[1])
                strength = float(parts[2])
                trans = ParsedDipoleTransition(
                    upper_idx=u_idx,
                    lower_idx=l_idx,
                    dipole_strength_a0=strength
                )
                transitions.append(trans)
                
    print(f"Successfully loaded {len(levels)} levels and {len(transitions)} dipole transitions.")
    return levels, transitions
