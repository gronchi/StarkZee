static\_profile
===============

Core quasi-static Stark-Zeeman solver.  For each electric microfield magnitude
:math:`F` and orientation :math:`\mu=\cos\theta`, the solver diagonalizes the
full field-dependent Hamiltonian and accumulates the three polarization profiles.

.. math::

   H(F,\mu) = H_A + V_E(F,\mu),
   \qquad
   V_E(F,\mu) = F\mu\,M_z + F\sqrt{1-\mu^2}\,M_x.

The static profile is the microfield and angle average

.. math::

   I_q(\omega) = \int_0^\infty \int_0^1
   W(F)\,I_q(\omega,F,\mu)\,d\mu\,dF,
   \qquad q \in \{\pi,\sigma^+,\sigma^-\}.

.. automodule:: starkzee.static_profile
   :members:
   :show-inheritance:
