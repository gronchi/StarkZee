# starkzee: Stark-Zeeman Line-Shape Model
# Atomic Hamiltonian + static profile solver + FFM ion dynamics

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("starkzee")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

from starkzee.line_profile import LineProfile, DiscreteTransitions
