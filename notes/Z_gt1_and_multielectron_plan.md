# Plan: hydrogen-like ions (Z > 1) and multi-electron emitters

Status updated 3 October 2026. Hydrogen and hydrogen-like work has priority.
WP1 is complete; WP2 has explicit charge/mass APIs and both fluctuation-rate
formulae, with structured multispecies plasma data and APEX/MD
charge-magnitude dependence still open. The next atomic task is WP3: exact
Dirac plus sourced Lamb/QED energies.

This is an implementation plan plus audit record, not a claim that every work
package is complete. Sources:
PPP manual (March 2024, transcribed in `scratch/papers/ppp_tex/`), Talin et al.
PRA 51, 1918 (1995), Ferri HDR
(`scratch/papers/Ferri_HDR_tel-01178604.pdf`), and the current code. Numerical
constants and sign conventions marked **[verify]** were written from memory and
must be checked against a primary source before use.

Related notes: `PPP_FFM_complex_amplitudes.md`, `note_electron_impact.md`,
`TODO.md` items 2-4, memory notes on the ZEST/PPPB differences.

---------------------------------------------------------------------------

## 0. What the code does today (audit)

| Area | Current state | Gap for Z > 1 / multi-electron |
|---|---|---|
| Atomic structure | `radiator.build_hamiltonian`: shell-by-shell, uncoupled `|n l m_l m_s>`, E_n = -Z^2 Ry/n^2, spin-orbit xi ~ Z^4 alpha^2, mass-velocity + Darwin to O((Z alpha)^2) | Fine for light Z; no Lamb shift, no higher-order Dirac; `use_empirical_data` supports only H, D, T (`atomic_levels.json`) |
| Stark | `_stark_templates`: <n l|r|n l-1> = (3n/2Z) sqrt(n^2-l^2), one shell at a time | Correct Z scaling inside a shell; no inter-shell coupling (experimental `multishell.py` exists, not integrated) |
| Dipoles | hydrogenic radial integrals with Z (`radial_dipole`, Gordon formula) | non-relativistic radial functions only |
| Microfield | explicit `emitter_charge`, charged-point selector, and background `Z_bar` | Potekhin uses only neutral/charged status; no z_e/z_p-dependent APEX/MD distribution (see A5) |
| Electron collisions | `broadening`/`collision`: GBK-type G(Delta omega) with y = (n^2/2Z)^2 (...), intra-shell dipoles | no Delta n != 0 perturbing levels, no inelastic rates, hydrogenic n and Z only |
| FFM | separate emitter/perturber masses; selectable ZEST or PPP-manual rate | external validation of the rate for coupled plasmas remains open |
| Multi-electron | `multielectron.py` (stubs, g_J = 1 placeholder), `atomic_loader.py` (unsigned dipoles in a0), `scripts/ppp_atomic_preprocess.py` (new) | no solver connection; no signed reduced matrix elements |

Two features of the PPP manual drive the design: (i) the quantum system is a set
of *J-levels* with reduced dipole matrix elements (Wigner-Eckart), not an
n-shell; (ii) the ground **and** excited manifolds are both Stark-mixed and the
electron operator couples them.

---------------------------------------------------------------------------

## PART A. Hydrogen-like ions with Z > 1

### A1. Exact scaling identity (free validation)

In atomic units, H(Z,F) = p^2/2 - Z/r + F z. With r = rho/Z,

    H(Z,F) = Z^2 [ p_rho^2/2 - 1/rho + (F/Z^3) rho_z ] = Z^2 H(1, F/Z^3).

So without fine structure, every energy of (Z, F) equals Z^2 times the Z = 1
energy at field F/Z^3, and dipole elements scale as 1/Z. This is now enforced
by `tests/test_37_hydrogenic_z_scaling.py` for the production Stark matrix,
its linear-Stark eigenvalue spectrum, and full polarization-resolved radiative
dipole matrices at Z = 2, 6, and 18. The complete shell Hamiltonian is not used
for this identity because its `fine_structure=False` option suppresses the
mass-velocity/Darwin terms but deliberately retains spin-orbit coupling.
**Test T-A1: implemented for the exact nonrelativistic operators.**

### A2. Regimes: where Z changes the physics

Stark shift in shell n: Delta E_S ~ (3/2) n (n1 - n2) e a0 F / Z.
Fine-structure scale: Delta E_FS ~ Z^4 alpha^2 Ry / n^3.
Setting these equal gives F_c ∝ Z^5, and since F ≈ F0 ∝ Ne^(2/3),
N_crit ∝ Z^(15/2). Low Z / high density: Stark dominates (hydrogen-like pattern,
Lyman-alpha without fine structure). High Z / low density: fine-structure levels
dominate and Stark acts as a perturbation mixing 2s1/2-2p1/2-2p3/2 (the Ar17+
case of Talin 1995). The full (2 n^2) diagonalization already covers the
crossover, provided the field-free energies are accurate.

### A3. Field-free energies at high Z

Replace the perturbative fine structure by the Dirac eigenvalue, then add QED:

    E_nj = m c^2 { [1 + (Z alpha)^2 / (n - delta_j)^2]^(-1/2) - 1 },
    delta_j = (j + 1/2) - sqrt( (j + 1/2)^2 - (Z alpha)^2 ),

plus recoil (reduced mass; already used) and the Lamb shift

    Delta E^L_{nl} = (alpha/pi) (Z alpha)^4 m c^2 F_{nl}(Z alpha) / n^3 **[verify]**.

Why it matters: the 2s1/2 - 2p1/2 splitting is zero in Dirac theory but is
nonzero in H-like Ar. The manual's Ar17 example gives 2p1/2 = 3318.220703 eV,
2s1/2 = 3318.388428 eV, 2p3/2 = 3323.036377 eV, i.e. 2s sits 0.1677 eV above
2p1/2. That splitting competes with the Stark matrix element (~eV at
N_e = 1e24 cm^-3 for Ar17) and controls the 2s-2p1/2 mixing.

Recommended implementation: an energy table `E_{n l j}(Z)` for H-like ions
(tabulate from Yerokhin and Shabaev, J. Phys. Chem. Ref. Data 44, 033103 (2015)
or NIST ASD **[verify]**), loaded through the existing `use_empirical_data` path.
Extend `atomic_levels.json` / `atomic_data.load_levels` from {H, D, T} to
H-like ions of the elements of interest, with a documented source per entry.
Fallback when no table exists: Dirac formula, with the Lamb shift flagged as
missing.

`tests/test_38_ar17_manual_benchmark.py` now quantifies the current fallback at
Z = 18 using the PPP manual's four-level example.  With A = 40, the retained
O((Z alpha)^4) Hamiltonian places both excitation groups about 0.9--1.0 eV
above the manual values, while reproducing the 2p3/2--2p1/2 interval within 2%.
It also explicitly verifies that the model leaves 2s1/2 and 2p1/2 degenerate,
whereas the manual gives a 0.167725 eV Lamb/QED separation.  **Test T-A2 is
implemented as a quantified limitation; the tabulated/Dirac+QED repair remains
WP3.**

### A4. Dipole matrix elements at high Z

Non-relativistic radial integrals are good to O((Z alpha)^2) (about 1.5% at
Z = 18). Check against the reduced matrix elements printed in the manual's
`miscel.` example for Ar17 (J-coupled, a.u.):

| pair | manual | hydrogenic estimate |
|---|---|---|
| 1s1/2 - 2p1/2 | -0.057735 | 0.8165 * (1.2903/Z) = 0.0585 |
| 1s1/2 - 2p3/2 | -0.082023 | ratio to the above ≈ sqrt(2) |
| 2p1/2 - 2s1/2 | -0.23400 | 0.8165 * (3 sqrt(3)/Z) = 0.2357 |
| 2s1/2 - 2p3/2 | -0.33237 | ratio to the above ≈ sqrt(2) |

J-coupled reduced element used for the estimate:

    <(l s) j || r || (l' s) j'> = (-1)^(l+s+j'+1) sqrt((2j+1)(2j'+1))
                                    { l  j  s ; j' l' 1 } <l||C^1||l'> R_{nl,n'l'},
    <l||C^1||l'> = (-1)^l sqrt((2l+1)(2l'+1)) (l 1 l' ; 0 0 0).

`reduced_hydrogenic_dipole_j` implements this Cowan/Edmonds recoupling using a
dependency-free Wigner 6j calculation.  All four nonrelativistic magnitudes
agree with the manual within 2%.  In the positive-near-origin radial phase
convention their signs are (-,+,-,+); rephasing only the 2s1/2 level maps them
to the manual's four negative signs.  Individual signs are therefore not an
absolute convention.  The four-edge loop product, which is phase invariant,
is positive in both representations.  Do not add relativistic radial
integrals unless the residual matters for the application. **Test T-A4 is
implemented, including the convention mapping and loop invariant.**

### A5. Microfield with a charged emitter

`emitter_charge` is now explicit in both profile solvers and in
`microfield_quadrature`; for H-like ions it defaults to Z-1. `Z_bar` remains the
separate background perturber charge. The current Potekhin backend can use only
the neutral-versus-charged-point distinction, so emitter charges +1 and +17
currently select the same dimensionless fit. This limitation is regression
tested rather than hidden. Physically (APEX, Iglesias et al. 1983; manual Sec.
3.3.2):

    P(F) = (2F/pi) ∫_0^∞ k sin(kF) T(k) dk,   T(k) = < exp(i k·F) >,

with T(k) computed for a *test charge* z_e in an ion + screening-electron
plasma, so P(F) depends on the ion-ion coupling and on z_e:

    F0 = 2.603 e z_p N_i^(2/3),
    Gamma_ii = z_p^2 e^2 / (r_i k T_i),   r_i = (3 / 4 pi N_i)^(1/3),
    emitter-ion repulsion ~ exp(-z_e z_p e^2 / (r k T_i)) in the pair correlation,
    screening: electron Debye length lambda_e, parameter s = r_i / lambda_e.

Remaining plan: introduce a structured multi-species plasma specification with
quasi-neutrality N_e = sum z_p N_i, then pass z_e/z_p into a traceable APEX/MD
distribution. Potekhin's charged-point fit remains a stand-in (TODO item 4),
to be quantified when those tables are available. For Z > 1 at high density
this is likely the dominant inaccuracy in the ion field.

### A6. Electron collision operator for Z > 1

Manual Appendix B (impact limit, G evaluated at the energy to the perturbing
level):

    G(Δω) = -(4π/3) (2m/(π k_B T_e))^(1/2) n_e (ħ/m)^2 ( C + (1/2) E1(y) ),
    y = (ħ n^2 / 2z)^2 (ω^2 + ω_p^2 + ω_{αα''}^2) / (E_H k_B T) **[verify the
    dimensional form against the code]**.

Items for Z > 1:

1. **z in y**: manual says "charge of the ionic core"; the code and ZEST use the
   nuclear Z (kappa_geo = Z / (n^2 a0)). Audit which one is intended for a
   Z-electron ion with one bound electron, and record it next to the existing
   r^2-scheme ambiguity (memory note "PPPB R·R subspace").
2. **Delta n != 0 perturbers**: B1 sums over *all* alpha'' and beta''. Today
   only the intra-shell dipoles are kept. For Z > 1 the gaps scale as Z^2, so the
   omega_{αα''}^2 term in y suppresses these terms strongly; implement them with
   `radial_dipole` (Gordon formula, valid for any n, n') and verify that they are
   small, instead of assuming it.
3. **Strong-collision constant C and Born validity**: Born/semiclassical G is
   questionable for highly charged ions (Coulomb-Born, Gaunt factor). Record
   the validity criterion rho_min vs rho_max and keep `C` as the documented
   GBK 1979 constants.
4. **Natural width** scales as Z^4 (E1 `einstein_a`); two-photon (about 8.2 Z^6
   s^-1) and M1 are negligible next to eV-scale widths and stay excluded, as in
   TODO item 1.

### A7. Ion dynamics (FFM)

    nu = v_th / r_s,  v_th = sqrt(k_B T / m_mu),  m_mu = emitter-perturber reduced mass,
    r_s = (3 / 4 pi N_i)^(1/3)    (manual Appendix C, Eq. C5).

`calculate_ion_fluctuation_rate` now exposes both conventions. `model='zest'`
uses sqrt(2 kT/m_pert), while `model='ppp'` uses the emitter--perturber reduced
mass and sqrt(kT/m_mu) from manual Eq. C5. They coincide exactly for equal
masses but differ for a heavy radiator in a proton plasma. The FFM API therefore
separates `A_ion` (emitter) from optional `A_perturber`; the historical
same-species default is unchanged. Compare with the published
values: Talin 1995 used nu = 2.78, 5.45, 9.75 eV for N_e = 1.5e23, 1e24,
5e24 cm^-3 (T = 1e7 K, Ar17+ in a proton plasma), and the manual's `miscel.`
example prints nu = 2.937 eV. A back-of-envelope evaluation of Eq. C5 for
N_i = 1e24 cm^-3 gives about 3.1 eV, below the 5.45 eV in the paper, so the
paper's values seem to come from an MD-fitted correlation time; reproduce the
definition before relying on either number. Unit regressions cover both
formulae and their equal-mass identity.

### A8. Validation ladder for Part A

| ID | Test | Reference |
|---|---|---|
| T-A1 | Z-scaling identity, no fine structure | exact (A1) |
| T-A2 | field-free energies of Ar17+ 2p1/2, 2s1/2, 2p3/2 (**quantified; QED repair pending**) | manual `miscel.`, 3318.2207 / 3318.3884 / 3323.0364 eV |
| T-A3 | tabulated vs Dirac+Lamb energies, Z = 2..18 | Yerokhin-Shabaev tables **[verify]** |
| T-A4 | four reduced matrix elements, phase mapping and loop sign (**complete**) | manual `miscel.`, within 1-2% |
| T-A5 | electron width of the 1s-2p3/2 line at zero field | manual `outd` sample: width 0.3172 eV for the conditions of the default `in_new.txt` (density units and exact conditions to confirm) |
| T-A6 | Ar17+ Lyman-alpha profile at N_e = 1.5e23, 1e24, 5e24 cm^-3, T_e = T_i = 1e7 K, with the paper's nu and Gaussian widths | Talin 1995 Figs. 2-4 (digitize); MD curves are the external reference |
| T-A7 | He+ (Z = 2) Lyman-alpha / Paschen-alpha vs published Stark profiles | to select |

T-A5/T-A6 are the first external, non-StarkZee references for Z > 1.

---------------------------------------------------------------------------

## PART B. Multi-electron emitters

### B0. Strategy: three tiers sharing one solver

A direct jump to arbitrary FAC/Cowan output hits an unsolved data problem
(signed reduced matrix elements, B6). To get a validated product early, build
the solver once on a generic J-level interface and feed it from three sources of
increasing generality:

* **Tier 0 (regression)**: hydrogenic Z, expressed through the generic interface.
  Must reproduce the existing production profile (T-B0). This proves the
  abstraction.
* **Tier 1**: single-active-electron ions and neutrals (alkali-like, Li-like
  C IV, Na-like) with quantum-defect energies and Coulomb-approximation radial
  integrals. Signed matrix elements come from angular algebra, so no external
  phase problem. Matches the existing stub `multielectron.py` (Option C).
* **Tier 2**: arbitrary level and dipole data from FAC/Cowan/MCDF (PPP's
  `base` file).

### B1. Generic radiator interface

Introduce a data model independent of hydrogenic shells (new module, e.g.
`starkzee/jbasis.py`):

    Level:   index, E (eV), J, parity, n_eff, label, config
    Dipole:  signed reduced matrix elements <i||d||j> in a.u. (Edmonds convention)
    Manifold: set of levels (lower / upper), populations p_i
    Basis:   |i J M>, M = -J..J, global index

Required properties, enforced by validation at load time:

* Hermiticity of the rank-1 operator: D_q^† = (-1)^q D_{-q} when D_q is built
  from Wigner-Eckart; equivalently the reduced elements satisfy
  <i||d||j>^* = (-1)^(J_i - J_j) <j||d||i> **[verify convention]**.
* Parity selection (E1 connects opposite parities, |ΔJ| ≤ 1, not 0-0).
* A single phase convention for all levels (see B6).

The solver-facing object is an abstract `Radiator` providing, for given
(B, F, mu): `H_upper`, `H_lower`, `D_q` matrices, `natural_widths`,
`effective_n` for each level, and the perturber list for the electron operator.
The hydrogenic code becomes one implementation of this interface.

### B2. Hamiltonian in the J-level basis

Field-free: H_0 = sum_i E_i |i J M><i J M| (from the atomic code; diagonal in
(i, J, M)). Zeeman (for PPPB-type use): H_Z = g_i mu_B B M with the Lande factor

    g_J = 1 + [J(J+1) + S(S+1) - L(L+1)] / [2 J(J+1)]   (pure LS),

or, in intermediate coupling, g_i = sum_LS |c_{i,LS}|^2 g_J(L,S,J) (needs the
mixing coefficients from the atomic code; if unavailable, flag as LS
approximation). Stark: V_F = -F d_0 for F ∥ z, using

    <i J M | d_alpha | i' J' M'> = (-1)^(J-M) ( J  1  J' ; -M alpha M' ) <i J || d || i' J'>   (manual Eq. 33).

Because V couples opposite-parity levels within a manifold, the relevant
matrix elements are the "Stark couplings" (xstrs = 1) between all pairs of
levels kept in the same manifold. With F ∥ z and B = 0, M is conserved; the
Liouville space splits into blocks by q = M_u - M_l ∈ {0, ±1}, giving the
pi/sigma decomposition already used for hydrogenic ions.

### B3. Liouville generator and line profile

In the ground x excited product space (manual Appendix A, A6):

    <e_i g_j | L | e_i g_j> = <e_i|H|e_i> - <g_j|H|g_j>^*,
    <e_i g_j | L | e_i g_k> = <g_j|H|g_k>,
    <e_i g_j | L | e_k g_j> = -<e_i|H|e_k>^*.

Generator K_F = L_F - i Φ (Φ: electron + natural broadening), diagonalize
K_F M = M Z, and obtain the Stark dressed transitions (checked numerically in
this session against the exact resolvent):

    a_k + i c_k = (d† M)_k (M^-1 ρ0 d)_k,   z_k = ω_k + i γ_k  (γ_k = -Im z_k in the code),
    I_F(ω) = (1/π) Σ_k [ a_k γ_k + c_k (ω - ω_k) ] / [ (ω - ω_k)^2 + γ_k^2 ]
    (sign of c_k tied to the conjugate in the code, `collision.py:211`),
    I(ω) = Σ_F P_F I_F(ω).

Dimension: dim = (Σ_u (2J+1)) (Σ_l (2J+1)); with q-blocks and, for ric = n, the
block structure found by the manual's `blod` routine, the solve is per block.
Keep the existing dense `eig` first; add block detection later (performance).

Emission vs absorption: ρ0 must be the population of the *radiating* manifold
(upper levels for emission). Use ETL populations as the default,
p_i = (g_i / g_0) exp(-(E_i - E_0)/k T_e), which reproduces the `popu.` column of
the manual's example, with an optional population file (`pfi`).

### B4. Electron collision operator for non-degenerate multi-level systems

Φ_{αα'ββ'} = Σ_{α''} δ_{ββ'} d_{αα''}·d_{α''α'} G(Δω_{α''β})
           + Σ_{β''} δ_{αα'} d_{β'β''}·d_{β''β} G(-Δω_{αβ''})
           - d_{αα'}·d_{β'β} [ G(Δω_{αβ'}) + G(-Δω_{α'β}) ]    (manual B1, **[verify indices]**)

Generalizations needed beyond the hydrogenic intra-shell case:

1. The sums over α'', β'' run over every level connected by a dipole matrix
   element, including levels *outside* the Stark-mixed set. The dipole list for
   the collision operator is therefore larger than the one used in H_F.
2. G is evaluated with the actual frequency to the perturbing level
   (impact limit at line center gives Δω_{α''β} = ω_{αα''}); y contains ω_{αα''}^2.
3. n in y is the **effective** principal quantum number,
   n_eff = z_c sqrt(Ry / (E_ion - E_i)), and z is the core charge seen by the
   perturbing electron; for highly stripped ions this differs from the
   hydrogenic assignment. State the choice explicitly.
4. Inelastic (non-degenerate) collisions: levels far from the radiating
   manifold contribute width through excitation/de-excitation. Use Van Regemorter
   **[verify constants]**:

       Ω_ij = (8π/√3) g_i f_ij (Ry/ΔE) ḡ(E/ΔE),
       C_{i→j} = (8.63e-6 / (g_i √T_K)) Ω_ij exp(-ΔE / k T) cm^3 s^-1,
       γ_i^{inel} = (ħ/2) N_e Σ_j C_{i→j}   (HWHM),

   as an optional addition, flagged as a model choice distinct from PPP's pure
   dipole operator.
5. Interference terms (third line): optional (`ric`), subject to the positivity
   issue below.

Unit audit: the manual's prefactor writes (ħ/m)^2 multiplying d·d; confirm the
combination that `build_ppp_collision_operator` uses, in particular the factor
that converts dipoles in a.u. to energy widths.

### B5. Positivity and the FFM reduction

With interference, a_k can be negative (modes of a non-Hermitian generator are
not orthogonal). The static profile remains valid (exact resolvent). The
current hydrogenic prototype uses the explicit StarkZee closure
p_k = |a_k| / Σ_j |a_j| for finite-ion dynamics while retaining signed residues
in the radiative numerator; an opt-in grouping closure is separately validated
against the static resolvent. Neither choice has been externally validated for
multi-electron systems, where off-diagonal terms may be larger. Plan:

* default to `ric = n`-equivalent closure for FFM (all a_k ≥ 0), as in PPP;
* run the interference closure for static profiles and report the signed-
  residue diagnostics and modulus correction;
* treat a complex-weight Liouvillian as a separate research item (TODO #3), not
  as clipping.

### B6. The atomic-data problem (blocking for Tier 2)

PPP needs signed reduced matrix elements with a consistent phase convention for
all levels, including those between levels *of the same manifold*. The real FAC
reference table we examined (`ne.tr`) gives gf and A but its last column merely
repeats gf, so signs are lost. Why signs matter: level phases are a gauge
freedom, so d_ij → e^{i(φ_i - φ_j)} d_ij is unobservable; but the dipole graph
is bipartite (parity), and products around closed loops of length 4 or more are
gauge invariant. Those loop products control interference between Stark paths
and cannot be recovered from unsigned values.

Plan (spike first, before writing code):

1. Determine whether FAC (binary `.tr` through `rfac.py`, or `PrintTable`
   options) or Cowan RCG/MCDF (GRASP) output exposes signed reduced matrix
   elements or eigenvector mixing coefficients.
2. If yes: write readers that output the signed `Dipole` table (extend
   `scripts/ppp_atomic_preprocess.py` and the CSV format with a `redu` column).
3. If no: for single-valence-electron ions use Tier 1; for general ions
   compute the reduced elements ourselves from a model-potential or CI wave
   function. This is a large project and should be scoped separately.
4. Add a loop-invariant check in the loader (product of signed elements around
   every 4-cycle against an independent recomputation) so that wrong phase
   conventions are detected.

### B7. Tier 1 specifics (quantum defect, single active electron)

    E_{n l j} = - z_c^2 Ry / (n - δ_{l j}(n))^2,
    δ(n) = δ_0 + δ_2 / (n - δ_0)^2 + ...   (Ritz expansion, fitted to NIST levels),
    n_eff = n - δ.

Radial dipole by the Coulomb approximation (Bates-Damgaard) or numerical
integration (Numerov) in a model potential; angular part from the J-coupled
formula in A4, which carries correct signs. Fine structure from the measured
j-dependent energies. Validation: with δ = 0 it reduces to hydrogenic matrix
elements (T-B1); compare Li-like and Na-like transitions with published Stark
widths and shifts.

### B8. Level selection ("pim" algorithm)

Manual Eqs. 31-32: coupling strength between a and b through intermediates,

    u(a,b) = s(a, i_1) m(i_1, i_2) ... m(i_n, b),   m(i,j) = 1/√2 for degenerate pair,

where s is the mean Stark shift (a function of N_e), `lay` is the number of
neighbor shells and `xset` the cutoff, `u(a,b) ≥ q / xset` with q the largest
first-neighbor shift. Implement as `select_levels(levels, dipoles, ne, lay, xset)`
returning the lower and upper subsets and the transitions marked radiative
(2) or Stark (1). Expose the selection report, and make convergence in
(`lay`, `xset`) part of `profile_convergence.py`.

### B9. Remaining pieces (shared with the hydrogenic code)

* Doppler: convolution, emitter mass A (existing).
* Microfield: as A5, z_e = ion net charge (not nuclear Z).
* Ion dynamics: FFM as A7. Channel count can exceed 10^4-10^5, so use the
  inversion-free formulation (Ferri HDR Annex F; PPP manual Eq. 26) already
  implemented in `_complex_ffm_profile_analytical`, and consider the 1995
  paper's channel renormalization (grouping modes of similar frequency and
  width) only if memory requires it.
* Autoionizing levels (dielectronic satellites) would add an autoionization rate
  to the diagonal of Φ; defer.

### B10. Validation ladder for Part B

| ID | Test | Reference |
|---|---|---|
| T-B0 | hydrogenic Z through the generic interface equals the production profile | internal (strict) |
| T-B1 | quantum defect → 0 gives hydrogenic matrix elements | exact |
| T-B2 | Hermiticity and phase loop invariants of the dipole table | exact |
| T-B3 | the manual's 4-level Ar17 `miscel.` example round trip (preprocessor output equals the manual's table to printed precision) | manual |
| T-B4 | Li-like and Be-like Ar spectra (PPP 1990 PRA conditions) | Calisti et al. 1990 |
| T-B5 | He-like Ar (e.g. He-beta, ICF conditions) Stark profile | published PPP / MD comparisons **[to select]** |
| T-B6 | convergence in `lay`, `xset`, basis size, `num_f`, `num_mu` | `profile_convergence.py` |

---------------------------------------------------------------------------

## C. Work packages and order

| WP | Content | Depends on | Acceptance |
|---|---|---|---|
| 1 | Tests T-A1, T-A2, T-A4 on the existing code (**complete: T-A2 records the missing-QED limitation**) | none | pass or quantified failures |
| 2 | Plasma/emitter parameters (A5) and reduced-mass nu (A7) (**API/rate complete; structured multispecies spec and APEX magnitude dependence pending**) | 1 | unit tests; documented differences |
| 3 | H-like energy table / Dirac + Lamb (A3) | 1 | T-A2, T-A3 |
| 4 | Δn ≠ 0 perturbers and z/n audit in the electron operator (A6) | 1 | T-A5 |
| 5 | Ar17 external comparison (T-A6) | 2-4 | declared tolerance vs Talin 1995 figures |
| 6 | `Radiator` interface + J-basis + Wigner-Eckart + Hermiticity checks (B1-B2); hydrogenic as Tier 0 | 1 | T-B0 |
| 7 | Generalized electron operator and level selection (B4, B8) | 6 | T-B3 |
| 8 | Data spike: signed matrix elements from FAC/Cowan (B6) | none | decision documented |
| 9 | Tier 1 quantum-defect module (B7) | 6 | T-B1, one published comparison |
| 10 | Tier 2 database reader and validation (B6) | 6-8 | T-B4, T-B5 |
| 11 | Docs: manual sections, TODO items 2-4, validation table | each WP | doc build |

WP 8 should start early and in parallel because it decides the shape of WP 10.

## D. Risks and open questions

1. **Signed dipoles** may not be obtainable from standard FAC output (B6).
2. **Microfield**: Potekhin's charged-point fit is not APEX; its error for large
   z_e / z_p is unknown (TODO item 4). Obtain the APEX/MD tables or a
   reference before quoting accuracy.
3. **z and n assignments** in the electron operator are ambiguous in the
   sources (A6.1, B4.3) and can change widths by tens of percent.
4. **Dimension growth**: dense eigendecomposition scales as dim^3; for
   multi-electron cases of a few hundred states, use q-blocks and the manual's
   block detection first.
5. **Domain claims**: no fixed `lay`, `xset` or quadrature count is universally
   safe; every claim needs a declared plasma domain and tolerance, per TODO
   item 1.
6. Questions for you: which ions and lines are the first target (He-like,
   Li-like, other)? Is a quantum-defect Tier 1 acceptable as a milestone, or is
   FAC-based Tier 2 required first? Should the H-like energy table include only
   the elements of interest, or all Z ≤ 20?

---------------------------------------------------------------------------

## E. Review addendum: AtomSpect (Nofs and Loch) cross-check

Sources: L. Nofs and S.D. Loch, JQSRT 362, 110015 (2026), read in full from
`scratch/papers/AtomSpect.pdf`, and the repository Lnofs/AtomSpect (local clone
in `Documents/AtomSpect`; `AtomSpect.py` read in part: `dipolestr`,
`polarization`, `stateprocess`, `HZeeman`). AtomSpect is a Zeeman and hyperfine synthetic-spectrum tool built
from tabulated level energies and angular-momentum algebra, with no radial
integrals; it has no Stark broadening, no microfield, no electron-collision
operator and no ion dynamics (DC Stark is listed as future work). It therefore
does not overlap with the broadening core of this plan, but it exposes the
following omissions in Part B.

1. **Zeeman operator is not diagonal in J (B2 is wrong at intermediate field).**
   B2 gave H_Z = g_i mu_B B M, diagonal in the level index. That is valid only
   when the Zeeman energy is small compared with the spacing between levels of
   the same M (weak field). Between J and J +- 1 levels of the same term,
   mu_B (L + g_s S)·B has nonzero reduced elements, so at intermediate fields
   (PPPB targets large B) levels mix (Paschen-Back transition). AtomSpect handles
   this by building H_FS from the J-level energies in the |J M_J> basis,
   rotating to |L m_L S m_S> with Clebsch-Gordan coefficients, and adding
   mu_B B (g_L m_L + g_s m_S) exactly. Fix: treat the Zeeman operator as a
   rank-1 tensor with a *reduced-element table including off-diagonal J*
   (pure LS: analytic 6j expressions; intermediate coupling: M1 reduced elements
   from the atomic code, which FAC can supply). Add a test: the weak-field limit
   recovers g_J, and the strong-field limit recovers E = mu_B B (m_L + 2 m_S).
   Scope note: AtomSpect mixes only the J levels of *one* LS term (paper Eqs.
   6, 10-12). Mixing between terms of the same J and parity (for example He I
   n d 1D2 / 3D2, or Ar I levels where the paper itself attributes a residual
   sigma-splitting mismatch to "a breakdown in the purity of the LS coupling
   notation") is already present at zero field through intermediate coupling.
   Total spin S is conserved by L + 2S, so the Zeeman operator cannot create it.
   It must come from the atomic data (IC mixing coefficients or
   level-to-level M1 reduced elements), so the reduced-element table has to
   include *different-term* pairs, not only J +- 1 within a term.
2. **Pure-LS "term model" tier with signed dipoles.** With only term energies,
   J-level energies and one reduced element <L||d||L'> per multiplet,

       <L S J || d || L' S J'> = (-1)^(L+S+J'+1) sqrt((2J+1)(2J'+1))
                                  { L  J  S ; J'  L'  1 } <L||d||L'>,

   which AtomSpect's `dipolestr` also uses (3j x 6j form). This gives signed,
   gauge-consistent dipoles without any atomic-structure code, resolving the
   B6 sign problem for LS-coupled systems. Validity: LS purity; intercombination
   lines (Delta S != 0) and strong configuration mixing need Tier 2. Add as
   Tier 1b, next to the quantum-defect tier (B7).
   Limit of the "no radial integrals" idea: AtomSpect can normalize relative
   intensities within one multiplet because <L||d||L'> cancels. Stark mixing
   cannot work that way: it needs the *magnitude* of <L||d||L''> to every
   perturbing term (for He I 447.1 nm, 4d-4f and 4d-4p), plus consistent
   signs. Tier 1b must take magnitudes from multiplet f-values/A-values (NIST)
   and signs from a one-electron model (Coulomb approximation), and still meets
   the loop-sign problem of B6 when more than one perturbing path exists. This
   is probably why Stark is still listed as future work in the paper.
3. **Observation geometry and polarization.** The plan decomposes by q = Delta M
   but does not state the line-of-sight weights. For the angle theta between
   B and the line of sight AtomSpect uses amplitudes eps_0 = sin(theta) (pi) and
   eps_(+-1) = sqrt((cos^2(theta) + 1)/2) (sigma), plus an optional polarizing
   filter. StarkZee already carries `view_angle_deg`; the multi-electron path must
   use the same projection and add a third angle (E-field direction relative to
   B) when both fields are present.
4. **Hyperfine structure.** Absent from the plan. H_hfs = A I·J + B quadrupole,
   basis |J M_J I M_I>, F = I + J, hyperfine Lande factor g_F. Scale check: it
   is micro- to milli-eV, so negligible against Stark/electron widths at high
   density, but relevant for low-density Stark-Zeeman diagnostics with nonzero
   nuclear spin. Keep it as an optional extension with an explicit
   applicability criterion (compare A, B to the Stark width), not as a default.
5. **State identification after diagonalization.** AtomSpect needs low-field
   energy sorting (`sortE`) to label eigenstates. Our FFM does not need labels,
   but diagnostics, Zeeman-fan validation and convergence studies do: add
   overlap-based adiabatic tracking of eigenvectors along B (and F).
6. **Data-handling details.** (a) Tabulated NIST levels are in cm^-1: fix the
   unit conversions and record the source. (b) Air-versus-vacuum wavelengths
   when comparing with spectrometer data (AtomSpect ships `Vac_to_air`); use a
   documented dispersion formula. (c) Doppler: bulk ion drift (mono and
   bidirectional) and instrument functions (Gaussian, Voigt, skewed Lorentzian)
   as options on the convolution stage.
7. **License.** AtomSpect is GPL-3.0. Do not copy its code into StarkZee unless
   StarkZee's license is compatible; re-implement from the published equations
   (Wigner-Eckart, Isler 1997 polarization, standard Lande formulas).
8. **Quadratic Zeeman for multi-electron levels** needs rank-0 and rank-2 parts
   of the r^2 sin^2(theta) operator between levels; not available from a J-level
   table alone. Include only if the diamagnetic shift is needed (state the
   field threshold) and mark it unsupported otherwise.

9. **Polarization from the full dipole vector, not from Delta M labels.** The
   paper (Fig. 7, Sec. 4.2) reports components with |Delta M_J| > 1 and argues
   they are needed to fit He I 447 nm. With B along z as the only field,
   J_z commutes with H_FS + H_B, so every eigenstate has a definite M_J, and an
   E1 operator gives only Delta M = 0, +-1. Real |Delta M| > 1 E1 components
   are therefore impossible; the feature must come from something else (state
   labels assigned by low-field energy sorting, intermediate-coupling or
   inter-term mixing absent from the model, or an electric field mixing in the
   4f "forbidden" component, a known Stark-sensitive feature of He I 447.1 nm).
   For our implementation: (a) compute each component's intensity as
   |e_obs* · <f|d|i>|^2 with the full vector of spherical components after
   diagonalization, rotated to the observation frame with Wigner D-matrices when
   F is not parallel to B; (b) add an assertion that no |Delta M| > 1 intensity
   appears when F = 0 or F ∥ B.
10. **External Zeeman data set.** The repository includes spectrometer data for
    He I 447 nm (B = 1.5, 2.5, 3.2 T, with and without a polarizer), Ar I
    750.4/751.4 nm (B = 2.2, 3.2 T) from MDPX, and C III 464.9 nm from W7-X
    (public Aurora data set). These are low-density plasmas, so they test only
    the Zeeman, polarization and Doppler layers of a multi-electron radiator,
    but they are measured, non-StarkZee references. The He I 447.1 nm line is
    also a natural first multi-electron Stark-Zeeman target (4d-4f mixing).
    Check the data licensing terms before copying them into the repo.
11. **Instrument polarization.** The fits need beam-splitter transmissions
    (T_alpha, T_beta) that change sigma/pi ratios (paper Eqs. 13-14). Provide an
    optional instrument Mueller-type weight on (pi, sigma) after the line shape,
    separate from the physics.
12. **Points in the paper not to adopt.** (a) "Levels avoid crossing" (Figs. 9,
    13) holds only within the same M_J. (b) Normalizing the relative strengths
    to 1 is presented as following the Thomas-Reiche-Kuhn sum rule, which is a
    different statement (sum of f over all final states). (c) Populations are
    assumed equal across M and levels (no alignment, no absolute intensities);
    PPP-type profiles need ρ0 from populations (B3). (d) The air-wavelength
    formula (Morton 2000, Eq. 25) is fine but its pressure, temperature and CO2
    conditions must be recorded with each comparison.

What AtomSpect does *not* validate: any Stark, collision or ion-dynamics step.
Its examples (He, Ar, C III spectra) test Zeeman patterns, polarization and
Doppler/instrument convolution only.
