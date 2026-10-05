# Changelog

## 0.5.0 — 4 October 2026

- Add the opt-in full PPP impact-limit collision operator, complex transition
  residues, interference diagnostics, and analytical FFM solver benchmarks.
- Default static, FFM, and discrete-transition calculations to empirical
  H/D/T level energies. Preserve analytical energies through
  `use_empirical_data=False`; require that choice for `Z > 1` and uncovered
  shells. Derive missing shell averages from complete fine-structure tables.
- Separate emitter and perturber masses and charges, and support both ZEST
  and PPP reduced-mass ion fluctuation rates.
- Correct profile references, convolution centering, grid handling, screened
  microfields, state-resolved natural damping, and scalar collision closures.
- Harden Rosato and Stehle table handling and archive reference-data,
  bibliography, and historical implementation provenance.
- Add numerical convergence tools, an experimental multishell CI reference,
  hydrogenic scaling checks, and Ar XVII manual benchmarks.
- Update the examples, notes, API documentation, and manual.

Validation: 599 tests pass with 90 model/grid warnings; strict Sphinx builds
pass. The full PPP operator and its finite-ion-dynamics probability closure
remain experimental. External PPP/PPPB validation, APEX microfields,
high-Z QED energies, and production multishell profiles remain open.
