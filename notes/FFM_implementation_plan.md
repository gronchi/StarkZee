# FFM electron-collision implementation record

Implementation snapshot updated 3 October 2026.

This note records the implemented ZEST-diagonal and PPP full-operator paths.
The source derivation is Calisti, Ferri, Mossé, and Talin, *The PPP code - User
Manual -* (2024), HAL hal-04501367v1, especially Section 2.2 and Appendix B.
The earlier comparison with the 2010 FFM paper is retained in
[`FFM_PPPB.md`](FFM_PPPB.md).

## Public modes

The default remains the established scalar/diagonal approximation:

- `electron_model` chooses the PPPB/Ferri or ZEST scalar impact kernel.
- `electron_operator=True` uses the diagonal dressed-state `r²` factors for
  both shells. It is the ZEST-style `c_k = 0` approximation.
- `electron_interference=False` preserves existing behavior.

The opt-in `electron_interference=True` mode constructs the complete
impact-limit collision operator described below. It is currently supported for
the PPPB/Ferri model aliases only and requires a constant impact kernel
(`frequency_dependent_width=False` in the static solver and
`sdt_frequency_dependent_width=False` in the FFM).

## Full PPP impact-limit operator

For an upper/lower optical coherence `|u><l|`, Appendix B Eq. (B1) has the
structure

    Phi = A sum_c [R_u,c² ⊗ I + I ⊗ (R_l,c²)^T
                   - 2 R_u,c ⊗ R_l,c^T].

`starkzee.collision.build_ppp_collision_operator` builds this matrix after
rotating the Cartesian within-shell dipoles into the Stark-Zeeman eigenbases.
One coefficient `A = W0 [C_nu + G_nu(0)]` is evaluated directly from the PPP
impact kernel, independently of the scalar full-shell/intra-shell radius
average. Using one coefficient for all three terms preserves the positive
`sum_c (R_u,c - R_l,c)²` structure; independently normalizing the two shell
terms would not represent Eq. (B1).

The generator

    K_f = L_f - i Phi

is diagonalized with a general complex similarity transformation. Left/right
residues give the manual's generalized strengths `a_k + i c_k`, while the
complex eigenvalues give `omega_k - i gamma_k`. For each polarization, the
static profile is

    I(E) = (1/pi) sum_k
           [a_k gamma_k + c_k (E - omega_k)]
           / [(E - omega_k)² + gamma_k²].

The FFM keeps `a_k + i c_k` in its numerator. For the Markov denominator,
StarkZee makes the explicit closure choice `p_k=|a_k|/sum(|a|)` so destructive
residues are not interpreted as negative probabilities. This modulus rule is
not stated in the PPP manual; it changes finite-ion-dynamics profiles but not
the zero-fluctuation static limit. Optional structured diagnostics report the
size and location of the signed-residue correction.

## Basis and numerical conventions

- Coherences are flattened in `(upper, lower)` order.
- Signed radiative dipoles are rephased by `(-1)^l` to match the production
  Stark-Hamiltonian convention.
- Natural widths are added to the diagonal of `Phi` before diagonalization.
- Arbitrary dressed-state phase and ordering changes leave the reconstructed
  profile invariant.
- The full path does not use SDT binning or the FFM numerical-inversion route;
  neither operation has a justified complex-residue reduction.
- Doppler broadening is applied after sampling the generalized static profile,
  because the `c_k` contribution is dispersive and is not an ordinary Voigt
  component.
- `A_ion` is the emitter mass. Optional `A_perturber` separates a light
  background ion, and `fluctuation_rate_model` selects the ZEST or PPP-manual
  rate. The collision operator itself is unchanged by that kinematic choice.
- `emitter_charge` is distinct from background `Z_bar`; current analytic
  microfield fits use only neutral versus charged-point status.

## Validation completed

`tests/test_36_ppp_collision_operator.py` checks:

1. Hermiticity of the Cartesian within-shell dipoles.
2. Hermiticity and positive semidefiniteness of Appendix-B `Phi`.
3. Trace preservation when the interference term is enabled.
4. Equality of the generalized-Lorentzian expansion and a direct resolvent.
5. The complex-strength sum rule and zero summed dispersion.
6. Invariance to dressed-state phases and permutations.
7. Public-option compatibility checks.
8. A finite nonnegative Lyman-alpha FFM result.
9. The zero-ion-fluctuation FFM/static reduction.

`tests/test_39_hydrogenlike_plasma.py` separately verifies emitter/perturber
mass handling, both fluctuation-rate equations, their equal-mass identity, and
the current Potekhin charge-magnitude limitation.

## Remaining acceptance work

This is an implementation of the manual's finite-shell, impact-limit operator,
not yet a claim of externally validated PPP parity. Before treating it as an
accepted production model:

1. Compare `omega_k`, `gamma_k`, `a_k`, and `c_k` against archived PPP/PPPB
   output for identical plasma and field configurations.
2. Converge the finite intermediate-state closure used by the Cartesian
   within-shell dipoles, especially for higher Balmer members.
3. Map regimes that yield near-zero or negative real `a_k` and record the
   modulus-probability diagnostics across the intended application domain.
4. Implement and validate the full detuning-dependent Appendix-B kernel before
   allowing frequency-dependent widths in this mode.

### Signed-residue mapping infrastructure

`scripts/scan_ffm_interference_residues.py` produces a JSON map over selected
transitions, magnetic fields, densities, temperatures, and `(num_f, num_mu)`
quadratures. It reports negative-mode counts, their absolute-strength fraction,
the most negative normalized residue, and representative complex-pole
separations for each polarization. For example:

```text
python scripts/scan_ffm_interference_residues.py --quadrature 2x2 4x3 8x5 12x7 --output residue_map.json
```

An initial D-alpha/D-gamma run at `B=3 T`, `Ne=1e20 m^-3`, and
`Te=Ti=1 eV` gave:

| line | quadrature | negative modes / retained modes | negative fraction of `sum(|a_k|)` |
|---|---:|---:|---:|
| D-alpha | 2x2 | 43 / 972 | 0.0781% |
| D-alpha | 4x3 | 240 / 4,042 | 0.4264% |
| D-alpha | 8x5 | 981 / 15,001 | 0.7775% |
| D-alpha | 12x7 | 2,021 / 32,911 | 0.7390% |
| D-gamma | 2x2 | 137 / 2,604 | 0.0122% |
| D-gamma | 4x3 | 1,452 / 11,088 | 4.076% |
| D-gamma | 8x5 | 6,379 / 41,636 | 8.015% |
| D-gamma | 12x7 | 14,959 / 91,162 | 9.625% |

These values show that a tiny quadrature materially understates the modulus
correction, especially for D-gamma. They are diagnostic samples, not converged
acceptance values: higher quadrature and independent variation of the
microfield cutoff are still required.

At `12x7`, varying `max_beta` confirmed material cutoff sensitivity:

| line | `max_beta=6` | `max_beta=10` | `max_beta=15` |
|---|---:|---:|---:|
| D-alpha negative fraction | 0.8382% | 0.7390% | 0.7855% |
| D-gamma negative fraction | 9.047% | 9.625% | 6.733% |

The D-gamma correction is therefore not converged in either quadrature or
cutoff at these settings. The scanner exposes `--max-beta` and
`--microfield-model` so those axes can be varied explicitly rather than folded
into a single apparent quadrature trend.

### Experimental radiative-channel grouping

The opt-in `interference_group_tolerance_ev` path implements the grouping
experiment without changing the default modulus-probability closure. It
clusters only modes whose total frequency and width spans fit the requested
tolerances, sums their complex residues, and defines the effective complex pole
by the residue-weighted first moment. It then evaluates both grouped and
ungrouped zero-fluctuation profiles on the caller's actual energy grid.

The grouped result is used only if its peak-normalized maximum profile error is
within `interference_group_profile_rtol`, all grouped real residues are
nonnegative, and all effective widths are positive. Otherwise the solver raises
an explicit error and identifies the failed criteria; it never silently falls
back to the modulus closure. Diagnostics also compare an accepted grouped FFM
profile with the ungrouped modulus-probability profile.

A coarse D-alpha `2x2` tolerance scan from `1e-8` to `3e-4` eV found no setting
that simultaneously removed all negative groups and preserved the static
profile within 10%. At `3e-5` eV the first polarization had no negative groups,
but its static-profile error was 64.7%. This is evidence against treating
frequency/width grouping as an automatic repair for this case.
The scan is reproducible with:

```text
python scripts/scan_ffm_grouping.py --profile-rtol 0.1 --output grouping_scan.json
```

The general complex eigensolve acts in a space of dimension
`(2 n_u²)(2 n_l²)` for every microfield quadrature point, so the opt-in path is
substantially more expensive than the default diagonal approximation.
