# TODO

---


## 1. Stark-dressed states and quadratic Zeeman: missing inter-n (inter-level) configuration interaction

**Goal:** reproduce the red shift of the high-n Balmer lines (Hβ, Hδ) seen in Ferri, Peyrusse & Calisti (2022) Fig. 1(c) at B = 1 kT, which the current code cannot produce, and give the Stark-dressed-state Hamiltonian the inter-level coupling its own formulation already calls for.

**This is not only a quadratic-Zeeman issue.** The Liouvillian, Eq. :eq:`eq:liouvillian` in `manual_formulation.rst` (Section "Liouville Space Representation"), is written generally as :math:`L = \hbar^{-1}(H_u \otimes \hat I^d - \hat I \otimes H_l^d)`, with the accompanying text stating its off-diagonal elements "encode the Stark and Zeeman couplings between dressed states" — nothing in that formula itself restricts the coupling to a single shell. In practice, though, `H_u` and `H_l` are only ever built per-shell (`build_hamiltonian`/`build_stark_matrix` each take one `n`), so the "dressed states" produced (Section "Full Electron-Radiator Hamiltonian", `manual_formulation.rst:287`) are only ever intra-shell dressed states. The Stark selection rule is stated explicitly right below that (`manual_formulation.rst:292`): "the selection rule for the within-shell Stark coupling is :math:`\Delta n = 0` ..." — i.e. the documented formulation already flags Δn = 0 as the operative restriction, it's just never lifted. Fixing the inter-n gap therefore means extending *both* the Stark and the quadratic-Zeeman matrix elements to couple different `n`'s, not just the diamagnetic term in isolation.

**Where the Stark breakdown condition is documented.** Unlike quadratic Zeeman (triggered by high B, per Ferri), the Stark Δn = 0 restriction's documented failure mode is triggered by high *density*: `manual_validity.rst:12-15` (domain-of-validity table, row "**Within-Shell Isolation (Δn = 0)**": form `V_E = diag_n(V_E)`, no `n → n±1` coupling; validity "Stark shift ≪ shell spacing `2Z²Ry/n³`"; failure mode "Quadratic Stark; Inglis-Teller merging at `Ne ≳ 10²⁴ m⁻³`") and `manual_operational_limits.rst:24` (Density Limits section, same statement in prose). Both approximations — Stark and quadratic Zeeman — share the identical code-level restriction (single-shell `H_u`/`H_l`), just approached from opposite regimes (density vs. field), so a multi-shell CI basis fixes both at once. Note this is distinct from the `Δn = 0` restriction discussed in `manual_formulation.rst:695-704` for the *electron-impact broadening operator* (ZEST model) — that one is separately, physically justified via the GBK dynamical factor suppressing inter-shell contributions exponentially, and is unrelated to this item.

**Symptom.** At B = 1 kT (Ne = 1e23 m⁻³, Te = 5 eV, transverse), Ferri Fig. 1 shows the with-QZ profiles shifting *red* relative to the no-QZ profiles, with the shift growing with principal quantum number (PQN). StarkZee instead shifts these lines *blue* (verified: Hβ peak −3.3 nm, Hδ peak −15.5 nm).

**Root cause.** The diamagnetic operator H_dia = (e²B²/8mₑ) r²sin²θ is positive-definite, so to first order (the diabatic bright-state energy) it raises every level and blue-shifts the line — which is all StarkZee computes. But the *observed* line center is the dipole-intensity-weighted centroid. At 1 kT the inter-n diamagnetic coupling is non-perturbative for the upper Balmer levels: for n=6 the diamagnetic shift (~90 meV) is comparable to the n=6→7 spacing (~100 meV). The bright state hybridizes with the n+1, n+2, … manifold just above it, oscillator strength fragments toward lower-energy admixtures, and the centroid is dragged *red*. The effect grows with PQN (for n=4 the ratio is only ~0.06 → still near the perturbative blue), matching Ferri's "global shift that increases with the PQN." Ferri states this explicitly: the quadratic term *"introduces mixing among states of different n … performed in a configuration-interaction mode … this point is crucial."*

**What the code is missing.** StarkZee keeps only the intra-shell (Δn = 0) diamagnetic block:
- `build_hamiltonian(n, …)` builds one shell at a time — no Δn block exists.
- `radial_r2_element` (`radiator.py:129`) is same-n only.
- `dipole_matrix_elements` diagonalizes n_u and n_l in separate Hamiltonians.
- `build_stark_matrix(n, …)` (`static_profile.py:55`) has the identical restriction, so no Stark coupling between upper levels of different PQNs is included either — the same architectural gap, not unique to the quadratic term.

**Verification done.** A crude hydrogenic model (multi-n basis with the off-diagonal-in-n diamagnetic elements, dipole-weighted line center, per-mₗ blocks, no fine structure) reproduces the red direction: Hδ moves +2 nm red, stable across Nmax = 9–11. The Hβ number is band-contaminated by neighboring-line strength in the model but trends red as the basis grows. Direction and mechanism are solid; magnitudes need a converged, truncation-controlled treatment.

**Proposed fix.** Build a truncated multi-n (configuration-interaction) Hamiltonian: basis spanning n = n_min … n_max per (mₗ, m_s), including both the inter-n diamagnetic elements ⟨n l mₗ | r²sin²θ | n′ l′ mₗ⟩ (Δl ∈ {0, ±2}) *and* the inter-n Stark elements ⟨n l mₗ | z F_z + x F_x | n′ l′ mₗ⟩ (Δl = ±1, the same operator as the within-shell matrix element in `manual_formulation.rst:296`, just no longer restricted to n′ = n), for all n, n′ in the truncated set; diagonalize once; take dipole matrix elements between the mixed eigenstates. This is an architectural change — the per-shell assumption is wired through `build_hamiltonian`/`build_stark_matrix` → `diagonalize_hamiltonian` → `dipole_matrix_elements` → `static_profile`/`ffm`. Suggested first step: a standalone `build_multishell_hamiltonian(n_list, Z, B, F, …)` (both operators) plus a dipole-weighted line-center check against Ferri Fig. 1(c), before threading it through the profile pipeline.

**Documentation status.** Flagged as a known, documented limitation (not yet fixed) in `manual_approximations.rst`, `manual_operational_limits.rst`, and `docs/source/examples/reproduce_fig1.rst` — the `reproduce_fig1*.py` examples are qualitative illustrations of the intra-shell quadratic Zeeman effect only, not a quantitative reproduction of Ferri Fig. 1.

---


## 2. Electric quadrupole (E2) transitions

**Goal:** assess whether adding the E2 transition operator is necessary for any realistic use case of this code.

**Current situation.** The code uses only the E1 dipole operator (Δl = ±1, q ∈ {−1, 0, +1}). Observable "forbidden" satellite lines in dense plasmas are already produced by Stark-induced E1 transitions — the microfield mixes l-states within the same n-shell, so formally E1-forbidden transitions gain amplitude. This mechanism is implemented and tested (`test_13_starkzee_satellites.py`).

**Why E2 is unlikely to matter here.** The E2/E1 intensity ratio scales as (r_atomic/λ)² ≈ (a₀/λ)². For H Balmer-α at λ = 656 nm this is ~5×10⁻⁹ — below any practical detection threshold. The ratio only becomes relevant in the X-ray regime (e.g., Fe XXVI at 0.18 nm, where E2/E1 ~ 10⁻³).

**The case that could matter.** Quadrupole Stark coupling — the interaction of the microfield *gradient* (not the field itself) with the atomic quadrupole moment — modifies `build_stark_matrix_general` rather than the transition operators. This is a distinct effect from E2 radiation and could be relevant at ICF or white-dwarf densities (Ne > 10²⁸ m⁻³). It is not the same as adding E2 emission.

**Conclusion.** Adding E2 transition operators is not justified for the visible/UV plasma conditions this code targets. Quadrupole Stark coupling is a plausible future extension if very-high-density regimes become a priority.

---


## 3. FFM: per-SDT frequency-dependent electron width (optional PPPB mode)

**Current state.** `calculate_ffm_profile` evaluates `electron_impact_width_model(0.0, …)` once at line center and uses that single γ for every SDT. This matches the ZEST fast-FFM approximation G(Δω) ≈ G(0) (ZEST Eqs. 24–25).

**What PPPB does.** PPPB Eq. 18 keeps a per-SDT width γ_{q,k} evaluated at each SDT's detuning from line center via the full G(Δω) function (PPPB Eq. 20). StarkZee's *static* path already supports this (`frequency_dependent_width=True` in `calculate_static_profile`).

**Impact.** The single-γ approximation underestimates the width in the far wings where G(Δω) < G(0). The error is small at line center but grows with detuning; it matters most for the extended wing profiles used in density diagnostics.

**Proposed fix.** Compute `gamma_k` per SDT inside the SDT accumulation loop, storing `gamma_k[i]` alongside `frequencies[i]` and `intensities[i]`. The Sherman-Morrison sum then becomes `Σ pₖ/(νᵢ + γₖ + iΔₖ)` with per-SDT denominators — still O(N) per frequency point.

---


## 4. FFM: complex SDT weights / line asymmetry (optional PPPB mode)

**Current state.** StarkZee uses real `intensities = |mixed_D|²` (aₖ only, cₖ = 0). This matches ZEST's construction of J from the real quasi-static profile and is consistent with the static path.

**What PPPB does.** PPPB Eq. 18 retains complex SDT weights `(aₖ + i cₖ)` where `cₖ` arises from the off-diagonal part of the electron-broadening operator in the SDT basis (the `⟨k|r²|k′⟩` terms with k ≠ k′). These produce a dispersion-shaped (antisymmetric) contribution to each SDT lineshape — the physical line-asymmetry that accumulates differently in the π and σ components.

**Impact.** Estimated ~1–2 % intensity asymmetry for Hβ at B = 1 kT. Negligible for most diagnostics but relevant if the asymmetry itself is the observable.

**Prerequisites.** Implementing cₖ requires the full off-diagonal electron-broadening operator in the SDT basis, which in turn requires a non-Hermitian Liouvillian. This is a more substantial change than item 3. Item 3 should be implemented first.
