"""
Comparison lineshape models for use alongside StarkZee.

All functions share the same signature::

    profile = <model>(wavelengths_nm, n_u, n_l, B, Ne_m3, Te_ev, Ti_ev,
                      view_angle_deg=90.0, species='H')

Database models:
    stehle
    rosato

Native analytical models:
    lomanowski
    stehle_param
    voigt

"""

from starkzee.models.analytical import (  # noqa: F401
    lomanowski,
    stehle_param,
    voigt,
)
from starkzee.models.tabulated import (  # noqa: F401
    stehle,
    rosato,
)
