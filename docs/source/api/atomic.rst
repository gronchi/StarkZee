atomic
======

Public field-free atomic helpers
--------------------------------

The functions in this module expose hydrogenic radial functions, dipoles,
line strengths, and radiative rates without requiring a line-profile
calculation. Field-dressed transition properties are obtained from
``LineProfile.compute_discrete()``.

See :doc:`../atomic_api` for runnable examples, plots, and a table of units.
``line_strength`` sums squared dipoles over both complete shells and all
polarizations without population weights. ``oscillator_strength`` returns
the weighted absorption quantity ``gf``; divide by ``2*n_l**2`` for ``f``.
``einstein_a`` averages over ``2*n_u**2`` upper substates and includes only
the specified lower shell. Pass ``A`` for a reduced-mass photon energy;
the omitted value preserves the infinite-nuclear-mass convention. Radial
functions and dipoles are field-free basis quantities, including when
accessed through a ``LineProfile`` with nonzero B.

.. automodule:: starkzee.atomic
   :members:
   :imported-members:
   :show-inheritance:
