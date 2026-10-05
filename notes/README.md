# StarkZee notes index

Updated 4 October 2026. These notes supplement the public documentation; the
live backlog remains `TODO.md` and completed repair evidence remains in
`StarkZee_repair_status.md`.

## Current implementation records

Static, FFM, and discrete-transition APIs now default to
`use_empirical_data=True`. `LineProfile` selects H/D/T automatically. Use
`False` for analytical energies, uncovered shells, or `Z > 1`; direct solvers
require the matching `atom`. Low-level Hamiltonian APIs retain their analytical
default and eV units. Missing shell-average entries are derived only from
complete fine-structure tables using degeneracy weights.

- [`starkzee_equations.md`](starkzee_equations.md): equations and conventions
  implemented by the hydrogenic static and FFM solvers.
- [`FFM_PPPB.md`](FFM_PPPB.md): relationship between the ZEST fast FFM, PPPB,
  and the selectable ZEST/PPP-manual fluctuation-rate conventions.
- [`FFM_implementation_plan.md`](FFM_implementation_plan.md): implemented
  impact-limit non-Hermitian PPP collision operator and remaining acceptance
  work.
- [`PPP_FFM_complex_amplitudes.md`](PPP_FFM_complex_amplitudes.md): signed
  complex residues, Markov probabilities, grouping experiment, and analytical
  versus dense-inversion benchmark.
- [`note_electron_impact.md`](note_electron_impact.md): scalar GBK/ZEST width
  conventions and their distinction from the full PPP operator.

## Active plans

- [`Z_gt1_and_multielectron_plan.md`](Z_gt1_and_multielectron_plan.md): staged
  hydrogen-like and multi-electron plan. Hydrogen-like work has priority.
  Work package 1 is complete; work package 2 has the public charge/mass API and
  both rate formulae, while structured multispecies plasma data and a
  charge-magnitude-dependent APEX/MD microfield remain pending. The next atomic
  task is exact Dirac plus sourced Lamb/QED energies.

## Standalone proposal

- [`report_TMSE.md`](report_TMSE.md): assessment of thermal motional Stark
  broadening. TMSE is not implemented and this report is not a current solver
  specification.

Current regression baseline: 599 tests passed with 90 model/grid warnings on
4 October 2026. A green internal suite does not replace external physical
validation.
