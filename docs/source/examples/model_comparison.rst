StarkZee vs. built-in reference models
=========================================

**Script:** ``examples/model_comparison.py``

Compares StarkZee's static and FFM (dynamic-ion) solvers against all five
built-in reference models (Voigt, Stehlé, Stehlé-parameterized, Rosato; here
Lomanowski is disabled since it has no magnetic-field treatment) for D-α
(n=3→2) at :math:`N_e = 10^{20}` m\ :sup:`-3`, :math:`T_i = T_e = 1` eV,
:math:`B = 3` T, :math:`\theta = 90^\circ`. Uses
``use_empirical_data=True`` so both StarkZee solvers land on the measured
NIST line center (see :ref:`sec:empirical`) for a fair comparison against the
reference models, which are anchored to their own tabulated NIST air
wavelength. Can save the figure to a file when given a path on the command line, which is
how the figure at the end of the :doc:`../manual_reference_models` page was
produced (the same figure is shown below).

Code
----

.. literalinclude:: ../../../examples/model_comparison.py
   :language: python
   :linenos:

Result
------

.. figure:: ../../figures/model_comparison.png
   :width: 100%
   :alt: StarkZee static and FFM compared against Voigt, Stehle, Stehle-parameterized, and Rosato reference models for D-alpha

   StarkZee (static and FFM) against the field-treating (Rosato) and
   field-free (Voigt, Stehlé, Stehlé-parameterized) reference models. Linear
   scale (left) and log scale (right).
