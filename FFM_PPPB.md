# StarkZee FFM vs. Ferri (PPPB) and ZEST

Comparison of `starkzee/ffm.py` against the Frequency Fluctuation Model (FFM) as
formulated in two reference papers:

- **PPPB** — Ferri, Peyrusse & Calisti, *Matter Radiat. Extremes* **7**, 015901
  (2022), Sec. II B, Eqs. (15)–(20).
- **ZEST** — the ZEST code description, Sec. 2.4, Eqs. (20)–(25).

Both cite the *same* underlying model: Calisti's fast FFM (ZEST Ref. [11] = PPPB
Ref. [56]). The headline result of this note: **StarkZee's FFM is a faithful
implementation of the ZEST fast FFM. It is *not* PPPB Eq. (18); the differences
from PPPB are deliberate ZEST conventions, not bugs.**

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

- **Rate prefactor.** ZEST Eq. 20 uses the **most-probable** ion speed
  `(2k_BT_i/M_i)^½`; PPPB's text uses its "thermal velocity" `(kT/m)^½` (no √2).
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
- **νᵢ form.** `νᵢ = √(2kT_i/m_i) · (4πN_i/3)^⅓`, with N_i = N_e/Z. This is **ZEST
  Eq. 20 verbatim.**

---

## 3. Conventions where StarkZee follows ZEST, not PPPB (NOT bugs)

These three were initially flagged as "discrepancies vs PPPB"; on cross-checking
ZEST they are exactly the ZEST fast-FFM conventions. Listed here so the choice is
explicit and revisitable, but none is an error.

**C1. Rate uses the √2 (most-probable) ion velocity.**
`calculate_ion_fluctuation_rate` uses `v_th = √(2kT_i/m_i)`. This **matches ZEST
Eq. 20 exactly**. PPPB's text would give `√(kT_i/m_i)` (1/√2 smaller). Numerically,
at Nₑ=1e23 m⁻³, Tᵢ=5 eV: νᵢ(ZEST/StarkZee)=1.52e-3 eV vs νᵢ(PPPB-conv)=1.08e-3 eV
(for context ωₚ=1.17e-2 eV). A larger νᵢ pushes toward motional narrowing and
affects central dip depth — but the larger value is the published ZEST one.

**C2. Electron width γ frozen on resonance.**
`ffm.py` evaluates `electron_impact_width_model(0.0, …)` once and uses that single γ
for every SDT. This **matches ZEST's impact-approximation fast FFM**, which
explicitly assumes `G(Δω)≈G(0)` to obtain the analytic Eqs. 24–25. PPPB Eq. 18
keeps the per-SDT frequency-dependent `γ_{q,k}` via G(Δω) (Eq. 20). Note that
StarkZee's *static* path is already more general here
(`frequency_dependent_width=True`, `static_profile.py`).

**C3. Real |d|² intensities (no c_k asymmetry term).**
StarkZee uses `intensities = |mixed_D|²` (real aₖ only). This **matches ZEST's
construction of J from the real quasi-static profile** (c_k = 0). PPPB Eq. 18
retains the imaginary `c_{q,k}` — the SDT-interference / line-asymmetry
(dispersion-shaped) contribution. Dropping it is also consistent with StarkZee's
static profile, which sums `|d|²` Lorentzians.

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

The FFM machinery (Sherman–Morrison algebra, SDT pooling, Markov mixing) is
faithfully implemented and is equivalent to both the ZEST and PPPB formulations,
which are themselves the same model. StarkZee specifically reproduces the **ZEST
fast FFM** (√2 rate, frozen γ, real intensities — items C1–C3). Doppler
broadening (D1) and the natural-linewidth floor (D2) are now implemented.

Optional higher-fidelity *PPPB mode* (a choice, not a correction):

- per-SDT frequency-dependent `γ_{q,k}` (C2) — the static path already supports it;
- the `c_k` asymmetry term (C3) — would capture line asymmetry PPPB Eq. 18 retains.

The √2 rate (C1) is a ZEST-vs-PPPB convention; settle against Calisti et al.,
*Phys. Rev. A* **42**, 5433 (1990) if a definitive value is wanted.
