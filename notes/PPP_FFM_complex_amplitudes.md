# PPP complex amplitudes and FFM compatibility

Implementation snapshot updated 3 October 2026. The separate emitter and
perturber masses and selectable ZEST/PPP-manual fluctuation rates affect
`nu_i`, but do not alter the complex-residue algebra documented here.

## Purpose

This note records the audit of how the original frequency-fluctuation model
(FFM), the PPP code, and the later fast FFM formulation use the generalized
Stark-dressed-transition amplitudes

\[
A_k=a_k+i c_k.
\]

The sources checked were:

- B. Talin et al., *Phys. Rev. A* **51**, 1918 (1995), available locally as
  `scratch/papers/PhysRevA.51.1918.pdf`;
- S. Ferri, HDR synthesis, Chapter 4 and Annexes F and G, available locally as
  `scratch/papers/Ferri_HDR_tel-01178604.pdf`;
- the transcribed 2024 PPP manual in
  `scratch/papers/ppp_tex/transcription.tex`.

## How PPP obtains `a_k` and `c_k`

For a fixed ionic microfield configuration, let the non-Hermitian optical
generator be diagonalized by a similarity transformation,

\[
K_f M_f=M_f Z_f.
\]

If `b = |d rho_0>>` is the radiative source and `u = <<d*|` is the observing
dipole, the residue of pole `k` is

\[
A_{f k}=(uM_f)_k(M_f^{-1}b)_k=a_{f k}+i c_{f k}.
\]

The microfield quadrature weight may be included in `A_fk` or applied during
the later field sum. Neither the 1995 paper nor the later sources replace the
real part by its modulus. The fixed-field spectrum is the generalized
Lorentzian sum

\[
I_f(\omega)=\frac{1}{\pi}\sum_k
\frac{a_{f k}\gamma_{f k}+c_{f k}(\omega-\omega_{f k})}
     {(\omega-\omega_{f k})^2+\gamma_{f k}^2}.
\]

StarkZee uses the opposite retarded-generator sign convention, so its raw
resolvent residue is conjugated when it is returned as the manual's
`a_k+i c_k`. A direct-resolvent regression verifies that this convention
reproduces the same generalized Lorentzian.

## How the historical FFM handles `a_k`

The FFM defines the stationary Markov probability by

\[
p_k=\frac{a_k}{r^2},\qquad r^2=\sum_j a_j.
\]

The 1995 paper calls `a_k` the real part of a radiative-channel intensity and
assumes that the normalized values are probabilities. It gives no rule for a
negative `a_k`: there is no absolute value, clipping, or signed-measure Markov
process.

The original FFM did not necessarily mix every primitive pole. Components
whose frequencies and homogeneous widths were observationally
indistinguishable were statistically grouped into radiative channels. Ferri's
Chapter 4 states that this compression generated new `(omega, gamma, a, c)`
parameters and was constrained to leave the static profile unchanged, or nearly
unchanged. The sources do not specify a reproducible grouping metric or
tolerance. They also identify this renormalization as an additional
approximation that could fail for complex cases.

The fast formulation in Calisti et al. (2010), reproduced in Ferri's Annex F,
removes the need for the large matrix inversion and therefore the computational
motivation for channel compression. It retains the complete complex numerator:

\[
I(\omega)=\frac{r^2}{\pi}\operatorname{Re}
\frac{\displaystyle\sum_k
 (a_k+i c_k)/r^2\,[\nu+\gamma_k+i(\omega-\omega_k)]^{-1}}
{\displaystyle1-\nu\sum_k
 (a_k/r^2)\,[\nu+\gamma_k+i(\omega-\omega_k)]^{-1}}.
\]

This formula still assumes that `p_k=a_k/r^2` is a probability. Annex F does
not discuss negative real residues. Its further replacement of the discrete
weights by a normalized static distribution `W(omega)` is stated only for the
additional regime `|c_k| << a_k` and approximately mode-independent
homogeneous widths.

Annex G explicitly states that PPP relies on the FFM and again uses
`p_j=a_j/sum(a)`. Thus it is incorrect to characterize traditional PPP as a
code without FFM. What is unresolved is the compatibility of a particular
full non-Hermitian collision operator and finite atomic closure with the
probability assumption.

## Consequences for StarkZee

Taking `abs(a_k)` or `abs(a_k+i c_k)` is not prescribed by these sources.
StarkZee now makes the explicit project choice

\[
p_k=\frac{|a_k|}{\sum_j |a_j|}
\]

for the stationary Markov probabilities only. The radiative numerator keeps
the signed complex residue `a_k+i c_k`, so destructive-interference terms are
not replaced by their moduli. This changes the finite-`nu_i` stochastic
closure, but it does not change the `nu_i -> 0` static-resolvent limit because
the probability-dependent term is proportional to `nu_i`.

Before this modulus-probability choice, StarkZee rejected materially negative
real weights. For the reduced
D-alpha diagnostic at `B=3 T`, `Ne=1e20 m^-3`, and `Te=Ti=1 eV`, the most
negative local normalized weight was:

| collision normalization | minimum `a_k / sum(a)` |
|---|---:|
| former `pppb` full-shell scalar calibration | `-2.013e-2` |
| direct PPP coefficient / `pppb-intra`-equivalent calibration | `-1.102e-3` |

The selected-shell normalization greatly reduces, but does not eliminate, the
problem. StarkZee now evaluates the coefficient directly as
`W0 [C_nu + G_nu(0)]`; it no longer rescales the collision tensor from either
scalar radius average. The remaining signed modes therefore point to the
finite collision closure and/or radiative-channel definition rather than a
full-shell normalization artifact or numerical roundoff.

## Change plan

1. **Lock down the present algebra.** Keep regression tests for the direct
   non-Hermitian static resolvent, the generalized-Lorentzian decomposition,
   the zero-fluctuation limit, and the equality of the explicit FFM matrix
   inversion and the analytical rank-one expression for complex strengths.

2. **Completed internally: separate collision closure from scalar-width
   calibration.** The impact coefficient is evaluated directly as
   `W0 [C_nu + G_nu(0)]`, and the selected upper/lower-shell intermediate-state
   closure is declared. The operator is no longer rescaled from a full-closure
   scalar width. External PPP parity remains required.

3. **Retain diagnostics for the approximation.** Report the number, total
   signed weight, minimum normalized weight, frequency, width, and nearby pole
   separations of negative-residue modes so the size of the modulus-probability
   correction remains visible.

4. **Prototype radiative-channel grouping as opt-in.** Group only modes that
   are close in both frequency and homogeneous width, sum their complex
   residues, and define an effective pole using a documented rule. Do not use
   `abs(a)`. Because the historical sources do not specify the algorithm, mark
   this path experimental and require a user-supplied tolerance. Compare it
   against the modulus-probability closure now used by the opt-in path.

5. **Validate every grouping.** Require the grouped generalized-Lorentzian
   static profile to agree with the ungrouped direct resolvent over the actual
   observation grid within a configurable error bound. Then require all
   grouped real weights to be nonnegative and verify that the `nu -> 0` FFM
   profile reproduces the grouped static profile.

6. **Do not hide unresolved cases.** If no grouping satisfying both profile
   fidelity and probability positivity exists, retain the static full-operator
   result and report that the FFM Markov reduction is invalid for that closure
   and plasma condition.

7. **Validate externally.** Compare component frequencies, widths, and complex
   amplitudes against an immutable PPP/PPPB output before making the new path a
   default. Converge the atomic closure, microfield quadrature, and grouping
   tolerance independently.

## Plan status

- Items 1 and 2 are implemented by the direct-resolvent, decomposition,
  analytical-versus-dense FFM tests and the explicit `pppb-intra` model.
- Item 3 is implemented by `interference_diagnostics` and
  `scripts/scan_ffm_interference_residues.py`.
- Items 4--6 are implemented as the opt-in
  `interference_group_tolerance_ev` closure. Groups are restricted in both
  frequency and width, preserve summed complex residue and first complex pole
  moment, and are rejected unless the grouped static profile meets
  `interference_group_profile_rtol`, every real group residue is nonnegative,
  and every width is positive. No fallback is performed.
- In a coarse D-alpha `2x2` diagnostic, tolerances from `1e-8` through
  `3e-4` eV did not satisfy positivity and 10% static-profile fidelity
  simultaneously. This negative result is retained: it demonstrates why the
  grouping path must remain opt-in and validated per case.
- Item 7 remains externally blocked. The workspace contains the PPP manual and
  papers but no archived PPP/PPPB component output containing matched
  `omega_k`, `gamma_k`, `a_k`, and `c_k` values. Internal StarkZee output cannot
  serve as its own external validation.

No source examined explicitly prescribes modulus probabilities; this behavior
is therefore documented as a StarkZee modeling choice rather than a recovered
historical PPP rule.

## Solver performance benchmark

`scripts/benchmark_ffm_solvers.py` compares the analytical complex-strength FFM
expression with a batched explicit inversion of the same rank-one Markov matrix.
It verifies numerical agreement before reporting timings. Run it from the
repository root:

```text
python scripts/benchmark_ffm_solvers.py
```

The benchmark is kept outside pytest because timing assertions are sensitive to
the processor, BLAS library, concurrent load, and memory pressure. The expected
costs for `M` observation energies and `N` SDTs are:

| method | time | dense temporary storage |
|---|---:|---:|
| explicit matrix inversion | `O(M N^3)` | `O(M N^2)` |
| analytical rank-one form | `O(M N)` | chunk-bounded `O(M N)` |

Benchmark numbers are diagnostic rather than acceptance criteria. Numerical
equivalence is enforced separately by the deterministic regression test
`test_complex_ffm_analytical_matches_large_matrix_inversion`.

An illustrative run on the development environment on 2026-10-01, using 23
observation energies and the best of three repetitions, gave:

| SDT modes | dense inversion | analytical | speedup | maximum difference |
|---:|---:|---:|---:|---:|
| 16 | 0.000187 s | 0.000055 s | 3.4x | `5.68e-14` |
| 32 | 0.000487 s | 0.000061 s | 8.0x | `1.71e-13` |
| 64 | 0.002444 s | 0.000074 s | 32.9x | `3.41e-13` |
| 96 | 0.005599 s | 0.000079 s | 70.9x | `9.10e-13` |

These small synthetic cases already show the expected rapidly increasing
advantage. Production quadrature can generate tens or hundreds of thousands of
SDTs, for which constructing the dense matrix is not practical at all.
