radiator
========

Atomic basis, Hamiltonian, and transition dipole construction.  The radiator
Hamiltonian assembled in the uncoupled hydrogenic basis is

.. math::

   H_A = H_0 + H_\mathrm{SO} + H_\mathrm{FS} + H_Z^{(1)} + H_Z^{(2)}.

The linear Zeeman term is diagonal,

.. math::

   H_Z^{(1)} = \mu_B B\,(m_l + g_s m_s),

while the diamagnetic term is

.. math::

   H_Z^{(2)} = \frac{e^2B^2}{8m_e}\,r^2\sin^2\theta.

Transition dipole matrices obey

.. math::

   \Delta l = \pm 1,\qquad \Delta m_l = q,\qquad \Delta m_s = 0,
   \qquad q\in\{0,\pm1\}.

.. automodule:: starkzee.radiator
   :members:
   :show-inheritance:
