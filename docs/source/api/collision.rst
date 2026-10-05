collision
=========

Opt-in PPP/PPPB electron-collision operator in the optical-coherence basis.
The implementation follows Section 2.2 and Appendix B of Calisti, Ferri,
Mossé and Talin, *The PPP Code -- User Manual* (2024),
`HAL hal-04501367v1 <https://hal.science/hal-04501367v1>`_. It constructs the
three terms of Eq. (B1), forms :math:`L_f-i\Phi`, and converts its general
complex eigensystem into the manual's generalized intensities
:math:`a_k+i c_k`, center frequencies :math:`\omega_k`, and damping rates
:math:`\gamma_k` (generator eigenvalues :math:`\omega_k-i\gamma_k` in the
retarded-resolvent convention used here).

This path is selected through ``electron_interference=True`` in the static or
FFM solver. It intentionally supports the manual's impact-limit
:math:`G(0)` construction first; observation-frequency-dependent collision
matrices remain outside the implemented scope.

The common coefficient :math:`W_0[C_{n_u}+G_{n_u}(0)]` is evaluated directly
from the PPP/Ferri kernel. It is not inferred by dividing a scalar full-shell
or intra-shell width by a radius average; the selected-shell Cartesian dipoles
in this module define the finite operator closure.

The implementation uses dense LAPACK eigensolves in a coherence space of
dimension :math:`(2n_u^2)(2n_l^2)`. It solves only the three radiative-dipole
right-hand sides instead of forming a full eigenvector inverse, shares the pole
kernel across polarizations, and caches field-independent within-shell dipole
operators. These optimizations preserve the direct resolvent; JIT compilation
does not accelerate the dominant LAPACK operation.

.. automodule:: starkzee.collision
   :members:
   :show-inheritance:
