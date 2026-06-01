"""
Tabulated Stark-Zeeman-Doppler lineshape models.

Reads from starkzee/data/{rosato,stehle}_tables.nc — no pystark required.

stehle  — Stehle (MMM) tabulated Stark profile, Doppler + Zeeman applied here.
rosato  — Rosato Stark-Zeeman tables (Zeeman already included), Doppler applied here.

"""

from starkzee.models.rosato_impl import rosato  # noqa: F401
from starkzee.models.stehle_impl  import stehle  # noqa: F401
