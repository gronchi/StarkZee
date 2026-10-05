# StarkZee FFM vs. Ferri (PPPB) and ZEST

Comparison of `starkzee/ffm.py` against the Frequency Fluctuation Model (FFM) as
formulated in two reference papers:

- **PPPB** — Ferri, Peyrusse & Calisti, *Matter Radiat. Extremes* **7**, 015901
  (2022), Sec. II B, Eqs. (15)–(20).
- **ZEST** — the ZEST code description, Sec. 2.4, Eqs. (20)–(25).

Both cite the *same* underlying model: Calisti's fast FFM (ZEST Ref. [11] = PPPB
Ref. [56]). StarkZee's **default** FFM follows the ZEST fast approximation. The
opt-in `electron_interference=True` path now implements the PPP manual's full
impact-limit collision operator and carries its complex strengths into the
PPPB FFM numerator. See [`FFM_implementation_plan.md`](FFM_implementation_plan.md)
for its scope and validation status.

---

## 1. The two formulations are the same model

    PPPB (Eq. 18):
        I_{d,q}(ω) = (r_q²/π) Re[ S / (1 − νᵢ S) ],
        S = Σ_k (a_{q,k} + i c_{q,k})/r_q² / (νᵢ + γ_{q,k} + i(ω − ω_{q,k}))

    ZEST (Eqs. 21–25):
        I_dyn(ω) = (1/π) Re[ J(ω) / (1 − γ J(ω)) ],
        J(ω) = ∫ I_qs(ω′)/(γ + i(ω−ω′)) dω′ = Σ_k f_k / (γ + a_k + i(ω−ω_k))

Same Sherman–Morrison expression under the dictionary:

| quantity            | PPPB        | ZEST   |
|---------------------|-------------|--------|
| SDT intensity       | a_{q,k}     | f_k    |
| Lorentz half-width  | γ_{q,k}     | a_k    |
| fluctuation rate    | νᵢ          | γ      |
| total weight        | r_q²=Σa_{q,k} | Σf_k (normalized) |

With Σf_k normalized, ZEST's `J ≡ PPPB's S` and the two coincide. The rank-1
transition-rate matrix W = νᵢ p_{q,k} (PPPB Eqs. 16–17) is what produces the
closed form and "avoids matrix inversion."

They differ in only two deliberate choices:

- **Rate model.** ZEST Eq. 20 uses the **most-probable perturber** speed
  `(2k_BT_i/M_i)^½`; PPP manual Appendix C uses the relative speed
  `(kT/mu)^½` with emitter--perturber reduced mass `mu`.
- **SDT weight / asymmetry.** PPPB Eq. 18 keeps the complex weight `(a_k + i c_k)`
  (from `D_{q,j}=r_q√(1+i c_j/a_j)`); ZEST builds J from the **real** quasi-static
  profile `I_qs` (intensities only, c_k = 0), valid in the impact approximation
  `G(Δω)≈G(0)`.

---

## 2. What StarkZee implements correctly ✓

- **Sherman–Morrison structure.** `ffm.py` computes `S = Σ pₖ/(νᵢ+γ+iΔ)` and
  `profile = (r²/π)·Re[S/(1−νᵢ S)]`, matching both formulas above term-for-term.
  Both the default analytical path and the optional `numerical_inversion` path use
  the rank-1 W = νᵢ pₖ structure.
- **SDT pooling.** All dressed transitions are accumulated across the (F, μ)
  microfield quadrature grid with weights `f_weight·mu_weight`, then a single
  Markov mixing is run over the pool — the correct stationary-distribution FFM.
- **νᵢ forms.** The default
  `νᵢ = √(2kT_i/m_pert) · (4πN_i/3)^⅓`, with `N_i=N_e/Z_bar`, is ZEST
  Eq. 20. The opt-in PPP-manual form replaces the speed by
  `√(kT_i/mu)` with emitter--perturber reduced mass.

---

## 3. Conventions where StarkZee follows ZEST, not PPPB (NOT bugs)

These three were initially flagged as "discrepancies vs PPPB"; on cross-checking
ZEST they are exactly the ZEST fast-FFM conventions. Listed here so the choice is
explicit and revisitable, but none is an error.

**C1. Two explicit rate conventions.**
The default `fluctuation_rate_model='zest'` uses
`v_th = √(2kT_i/m_pert)`, matching ZEST Eq. 20. The opt-in `'ppp'` model uses
`v_th = √(kT_i/mu)` with emitter--perturber reduced mass, matching PPP manual
Appendix C Eq. C5. They are identical for equal masses (`mu=m/2`) and diverge
for, e.g., Ar XVII in a proton plasma. `A_ion` now unambiguously denotes the
emitter mass in the profile solver; `A_perturber` overrides the historical
same-species assumption.

**C2. Electron width γ frozen on resonance.**
`ffm.py` evaluates `electron_impact_width_model(0.0, …)` once and uses that single γ
for every SDT. This **matches ZEST's impact-approximation fast FFM**, which
explicitly assumes `G(Δω)≈G(0)` to obtain the analytic Eqs. 24–25. PPPB Eq. 18
keeps the per-SDT frequency-dependent `γ_{q,k}` via G(Δω) (Eq. 20). Note that
StarkZee's *static* path is already more general here
(`frequency_dependent_width=True`, `static_profile.py`).

**C3. Real |d|² intensities in the default mode.**
The default uses `intensities = |mixed_D|²` (real aₖ only), matching ZEST's
construction of J from the real quasi-static profile (`c_k = 0`). With
`electron_interference=True`, StarkZee instead diagonalizes the non-Hermitian
PPP generator and retains `a_k + i c_k` in the numerator of PPPB Eq. (18).

**C4. Modulus probabilities for signed non-Hermitian residues.**
In the opt-in interference path, StarkZee uses
`p_k = |a_k| / sum_j |a_j|` for the nonnegative stationary Markov
distribution, but retains the original signed `a_k + i c_k` in the radiative
numerator. This is a deliberate StarkZee closure for destructive-interference
residues; it is not stated in the PPP/PPPB sources. Because the probabilities
enter multiplied by the ion fluctuation rate, it leaves the `nu_i -> 0` static
limit unchanged but changes the finite-fluctuation FFM profile.

---

## 4. Resolved issues ✓ (were D1–D2)

**D1. Doppler broadening in the FFM path — fixed.**
`calculate_ffm_profile` now applies a thermal Doppler Gaussian convolution after
the Markov accumulation loop, using the same zero-padded FFT strategy as
`calculate_static_profile`.  The 1/e half-width is
`σ_D = E₀ √(T_i / m_ion c²)` computed from `Ti_ev` and `A_ion`.  A keyword
`apply_doppler=True` (default) controls it; set `False` to recover the
purely Stark-Zeeman FFM output.

**D2. Width floor replaced by physical natural linewidth — fixed.**
The ad-hoc `gamma_k += 1e-4` (0.1 meV fudge) is replaced by
`ħ(Γ_u + Γ_l)/2` summed over all Einstein-A decay channels, exactly as
`static_profile.py` computes `w_natural_ev`.  The two paths are now consistent.

---

## 5. Formalism note: not Floquet-Liouville

StarkZee uses the standard **Liouville-space resolvent** formalism
(Anderson-Talman-Baranger / impact approximation), not the Floquet-Liouville
extension.  Floquet-Liouville applies when the Hamiltonian is explicitly
time-periodic (laser-dressed or rf-modulated plasmas); it expands states into
harmonics of the drive frequency in an extended Liouville space.  No such
periodic driving is present in StarkZee's physical model (quasi-static ions,
stochastic FFM, perturbative electron broadening), so the standard resolvent
without Floquet is correct.

---

## 6. Bottom line

The default FFM reproduces the **ZEST fast FFM** (ZEST rate, frozen γ, real
intensities). Doppler broadening and physical natural widths are implemented.
The optional full-operator mode retains the impact-limit PPP `c_k` asymmetry
term, but it intentionally does not yet combine that operator with a
detuning-dependent `G(Δω)`. Its common coefficient is evaluated directly as
`W0 [C_nu + G_nu(0)]`, independently of the scalar radius-average selector.
External PPP/PPPB parity and finite-shell closure convergence remain acceptance
tasks.

The ZEST and PPP-manual rate choices are explicit; comparison with MD-fitted
rates from Calisti et al., *Phys. Rev. A* **42**, 5433 (1990), remains an
external validation task.
