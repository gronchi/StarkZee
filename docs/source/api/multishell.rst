multishell
==========

Experimental multi-shell configuration-interaction matrices. This module
provides signed inter-shell radial integrals and Hermitian linear-Stark and
diamagnetic blocks in a truncated hydrogenic basis. It is a standalone
reference implementation and is not yet used by the static or FFM solvers.

The multi-shell E1 matrices can be rotated into CI eigenbases while preserving
their full Frobenius strength. For a selected source shell pair,
``transition_strength_breakdown`` and ``oscillator_strength_breakdown`` retain
the target, neighboring-channel, and coherent interference terms separately.
They close exactly both globally and inside a transition-energy window; the
interference term must not be assigned to either channel as an independent
intensity.

Use ``python scripts/multishell_ci_reference.py`` to report a tracked-state
diagnostic while independently varying the lower and upper shell cutoffs and
the electric-field angle. It also reports the coherent oscillator-strength
breakdown in a selected target-line window. That diagnostic is not a
line-profile benchmark. Numerical tolerances are mandatory command-line
arguments; an optional JSON output preserves all cutoff rows and their errors.

``starkzee/data/multishell_validation_cases.json`` records one deliberately
narrow regression case: H-beta at :math:`B=500` T and one Holtsmark normal
field for :math:`N_e=10^{23}` m\ :sup:`-3`. At 0, 45, and 90 degrees, the
:math:`n=3\ldots6` upper space agrees with :math:`n=3\ldots7` within
:math:`10^{-5}` eV for the tracked energy and 0.1% for target and total
windowed ``gf``. This finite-reference result is not an infinite-basis proof
or a cutoff for other transitions and conditions.

.. automodule:: starkzee.multishell
   :members:
   :show-inheritance:
