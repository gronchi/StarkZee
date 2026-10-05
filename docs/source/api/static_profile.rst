static\_profile
===============

``calculate_static_profile`` and ``discrete_transitions`` default to
``use_empirical_data=True``. Supply ``atom='D'`` or ``'T'`` for isotope levels;
the direct API defaults to H. Use ``False`` for analytical energies, ``Z > 1``,
or shells beyond the bundled data. Low-level ``solve_starkzee`` retains its
analytical default.

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

Natural damping is state resolved by default. The uncoupled E1 decay-rate
operator includes every lower principal shell and is rotated into each
field-dependent eigenbasis, giving transition half-width
:math:`\hbar(\Gamma_{u,i}+\Gamma_{l,j})/2`. Set
``natural_width_mode='shell_average'`` only to reproduce the historical common
shell width.

Set ``electron_interference=True`` together with
``frequency_dependent_width=False`` to use the PPP Appendix-B impact-limit
operator. This opt-in path retains the upper--lower interference term,
diagonalizes :math:`L_f-i\Phi`, and evaluates the generalized components with
complex strengths :math:`a_k+i c_k`. The default remains the established
scalar/diagonal solver path.

.. automodule:: starkzee.static_profile
   :members:
   :show-inheritance:
