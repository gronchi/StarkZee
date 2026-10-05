Physics Explanation of Use Cases and Operational Limits
===============================================================

Magnetic fields
---------------

The normal-Zeeman component shift is 0.05788 meV per tesla. Its importance
relative to fine structure and Stark mixing depends on the chosen transition,
charge and microfield distribution. A field or density range alone does not
establish a weak-Stark or isolated-triplet regime.

Quadratic Zeeman terms are retained within each principal shell only. Inter-shell
configuration interaction and higher relativistic terms are absent. Numerical
diagonalization of that finite matrix does not validate white-dwarf or neutron-star
spectra. Compare neglected couplings to shell gaps and converge an extended
atomic basis before claiming such applicability. See :doc:`manual_approximations`.

Density and temperature
-----------------------

At fixed temperature and charge, lowering density collapses the physical field
scale toward zero; the reduced-field distribution need not become a delta.
Doppler dominance additionally depends on ion temperature and radiative lifetime.
At high density, compare Stark/diamagnetic couplings with shell gaps and atomic
extent with perturber spacing. Separately compare ion fluctuation energy with
component widths and splittings: high density alone does not imply breakdown
of the quasi-static approximation.

Use matched static/FFM inputs and independently refine spectral spacing,
window, microfield cutoff, field resolution, angular quadrature and FFM binning.
No universal safe defaults or physical error bounds are established.




