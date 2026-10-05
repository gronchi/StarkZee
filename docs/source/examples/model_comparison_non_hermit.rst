Full PPP collision-operator comparison
======================================

**Script:** ``examples/model_comparison_non-hermit.py``

This is the full non-Hermitian counterpart of
:doc:`model_comparison`. It compares the same D-α and D-γ conditions and
reference models, but calls both StarkZee solvers with
``electron_interference=True``. The static calculation constructs all three
terms of the PPP manual's Appendix-B impact-limit collision matrix,
diagonalizes :math:`L_f-i\Phi`, and evaluates the generalized strengths
:math:`a_k+i c_k`.

Run it from the repository root:

.. code-block:: bash

   python examples/model_comparison_non-hermit.py output.png

For a faster exploratory calculation, use:

.. code-block:: bash

   python examples/model_comparison_non-hermit.py output.png --quick

This selects ``num_f=12`` and ``num_mu=5`` (60 field configurations instead
of 660). Use ``--static-only`` to skip the FFM attempt, or set explicit
``--num-f`` and ``--num-mu`` values. Preview quadrature is not a substitute for
convergence testing.

The requested path receives the D-α figure; D-γ is saved beside it with a
``_dgamma`` suffix. The calculation uses ``num_f=60`` and ``num_mu=11`` for
parity with the default comparison. A general complex eigensolve is performed
in a coherence space of dimension :math:`(2n_u^2)(2n_l^2)` at every quadrature
point, so D-γ is substantially more expensive. Reduce the two quadrature
values for exploratory runs and converge them before quantitative use.

FFM probability closures
------------------------

The PPPB FFM identifies :math:`p_k=a_k/\sum_j a_j` as instantaneous Markov
probabilities. A finite-shell non-Hermitian decomposition can yield negative
real :math:`a_k`, which cannot be signed transition probabilities. StarkZee's
default experimental closure uses :math:`p_k=|a_k|/\sum_j|a_j|`, but retains
the original signed :math:`a_k+i c_k` in the radiative numerator. The example
reports the affected mode count and fraction of :math:`\sum|a_k|` for every
polarization. Sparse open-circle markers distinguish the FFM curve from the
dashed static curve even if they overlap.

Passing ``--group-tolerance-ev T`` instead tests an experimental radiative-
channel grouping closure. ``--group-width-tolerance-ev`` can set a separate
width span, and ``--group-profile-rtol`` sets the required static-profile
fidelity. The calculation stops with an explicit error if grouping leaves a
negative residue, creates a nonpositive width, or exceeds that profile error;
it does not silently fall back to modulus probabilities.

The static generalized-Lorentzian profile does not interpret individual
:math:`a_k` as probabilities and remains available. External PPP/PPPB component
data and finite-shell closure convergence are still needed before either
finite-ion-dynamics closure can be regarded as validated PPP parity.

As in the default comparison, the Rosato :math:`10^{-6}` cutoff affects only a
normalized plotting copy; it does not alter the model or source tables.

Code
----

.. literalinclude:: ../../../examples/model_comparison_non-hermit.py
   :language: python
   :linenos:
