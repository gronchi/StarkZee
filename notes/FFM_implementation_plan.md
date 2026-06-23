# FFM implementation plan — selectable ZEST and PPPB models

Goal: offer **both** FFM variants in `starkzee/ffm.py` behind a `model` switch, so
they can be compared and selected. Background and formula cross-check are in
[`FFM_PPPB.md`](FFM_PPPB.md).

## STATUS

**DONE — scalar electron-impact model selection (the two published width
prescriptions).** Added `electron_impact_width_model(...)` dispatcher in
`broadening.py` (`ELECTRON_MODELS = ferri/pppb, zest, zest-gbk, zest-lee,
zest-dufty`) and threaded an `electron_model='ferri'` parameter through
`calculate_static_profile`, `calculate_ffm_profile`, and (via `**kwargs`)
`LineProfile.compute_profile`. Default `'ferri'` is byte-identical to the old
behavior (verified). `'zest*'` selects the ZEST κ_m/ω_p prescription with the
GBK/Lee/Dufty G-functions. Tests: `tests/test_19_electron_model.py` (9), full
suite 355 passed. This makes both the **Ferri/PPPB** and **ZEST** electron-impact
*width* models accessible and selectable — the literal "have both models
available" goal, at the level StarkZee's broadening actually operates (a scalar
width per transition).

**DONE — electron-impact OPERATOR diagonal (ZEST operator, `c_k = 0`).**
Added `electron_impact_r2_scaling(eigenvectors, n, Z)` in `broadening.py`: the
per-Stark-Zeeman-state operator diagonal `⟨k|r²|k⟩/⟨r²⟩_avg` (mean = 1, trace
preserved). Threaded an `electron_operator=False` flag through
`calculate_static_profile` (and `LineProfile.compute_profile` via `**kwargs`).
When `True`, each dressed state's electron width is rescaled by its own ⟨r²⟩
instead of the shell average — i.e. **the diagonal of the broadening operator**,
which is exactly the ZEST operator with the off-diagonal/`c_k` dropped. Default
`False` is byte-identical to the scalar path. Composes with `electron_model`
(`zest`+operator, `ferri`+operator). Tests: 5 added in
`tests/test_19_electron_model.py` (trace preserved, profile changes, norm
conserved to <0.5%, composes with ZEST); full suite **360 passed**.

**REVIEW / NOT done** (carry forward):
- **PPPB `c_k`** — the operator *off-diagonal* `⟨k|r²|k'⟩` → complex intensity
  `a_k + i c_k` via the non-Hermitian Liouvillian. The `electron_impact_r2_scaling`
  docstring + a code comment in `static_profile.py` mark where it plugs in.
  Estimate ~1–2 % for Hβ at 1 kT (see `c_k` section).
- **Lower-manifold `d†·d`** — the operator currently rescales the **upper** state
  only (matching the scalar model, which is upper-only). ZEST also has the
  lower-state diagonal; sub-dominant for Balmer (`⟨r²⟩∝n⁴`) but real.
- **FFM operator** — `calculate_ffm_profile` still applies one resonance width to
  all SDTs; the operator diagonal is implemented in the **static** path only.
  Per-SDT widths in the FFM (storing `r2_scale_u` alongside each SDT) is the next
  step and the natural foundation for `c_k` there.
- **Doppler path** — operator scaling is applied in the in-loop Lorentzian; the
  post-FFT Lorentzian (Doppler-dominant grids) uses the scalar resonance width.

## Formulas (re-checked)

Both reduce to the same Sherman–Morrison skeleton. With
`L_k(ω) = 1/(νᵢ + γ_k + i(ω − ω_k))`:

    I_q(ω) = (r_q²/π) Re[ Σ_k (w_k/r_q²) L_k(ω) / (1 − νᵢ Σ_k (a_k/r_q²) L_k(ω)) ],
    r_q² = Σ_k a_k,   a_k = |d_k|²

The **denominator is identical** in both models (real weight `a_k`). Only three
knobs differ:

| knob                  | ZEST (= current code)        | PPPB (Eq. 18)            |
|-----------------------|------------------------------|--------------------------|
| numerator weight `w_k`| `a_k` (real)                 | `a_k + i c_k` (complex)  |
| electron width `γ_k`  | frozen `γ(0)`, same ∀k       | per-SDT `γ(Δω_k)`        |
| rate `νᵢ` velocity    | `√(2kT_i/M)` (most-probable) | `√(kT_i/M)` (thermal)    |

`νᵢ`, the SDT frequencies `ω_k`, and the complex amplitudes `d_k` are **shared**,
so a single collection loop feeds both models.

## `c_k` — deferred (architecturally out of reach), magnitude UNVERIFIED

Source: **Calisti et al., Phys. Rev. E 81, 016406 (2010)**, Eqs. (16)–(25).
(Note: an earlier draft mis-claimed these terms weren't in the paper — that was a
unicode-grep failure; Eq. (23) is verbatim below.)

**What the paper actually says.** With electron impact included, the Liouville
operator has a **non-Hermitian homogeneous electron-impact broadening
contribution**, so the SDTs have a complex frequency `ω_k − i γ_k` *and* a complex
intensity `a_k + i c_k`. The full FFM line shape (Eq. 23):

    I(ω) = (r²/π) Re { Σ_k (a_k + i c_k)/r² / [i(ω−ω_k) + γ_k + Γ] }
                     / { 1 − Γ Σ_k (a_k/r²) / [i(ω−ω_k) + γ_k + Γ] }

The collapse to the convolution / ZEST form (Eqs. 24–25) requires **two**
conditions stated as a bare conditional: "**If** `c_k ⪡ a_k` is fulfilled **and**
`γ_k` is weakly k-dependent (`γ_k = γ`)."

**Critical caveats (do not overstate).**
- The paper does **not** claim `c_k ≪ a_k` holds generally or "in most practical
  conditions." It only gives the conditional. Negligibility for *our* regime
  (H Balmer, Nₑ~10²³ m⁻³, kilotesla B) is **untested**. PPPB carries the full
  Eq. (23) precisely for strong-field, overlapping-SDT regimes.
- The paper gives **no explicit formula** for `c_k` (the bilinear
  `⟨⟨d†|k⟩⟩⟨⟨k̃|dρ₀⟩⟩` floated in summaries is a plausible reconstruction, not in
  the text). `c_k` simply emerges from the non-Hermitian Liouvillian.
- `Γ = v_th/d` with `v_th` just "thermal velocity" — **no √2 specified.** The
  ZEST(√2)-vs-PPPB-text convention is not resolvable from Calisti 2010 either.

**Why deferred anyway (corrected reasoning — NOT "it's negligible"):**
`c_k` is the imaginary part of the **non-Hermitian electron-impact operator Φ
projected on the SDT basis**. StarkZee has no Φ operator — only a scalar GBK
*width* per transition (`electron_impact_width`). You cannot extract `c_k` from a
scalar width; it needs Φ built as a full matrix (with off-diagonal Stark-component
couplings) plus a bi-orthogonal non-Hermitian diagonalization. That is a large
broadening-operator overhaul, out of scope here. So `c_k` is deferred because it
is **not computable in the current architecture**, and its size for our case is
**unverified** — not because the paper proves it small.

If wanted, estimate the asymmetry magnitude empirically before committing to the
Φ-operator work.

### PPPB DOES have an electron-impact operator (key finding)

PPPB **Eq. (19)** defines the broadening as an *operator*, not a scalar:

    Φ(Δω) = −(4π/3) Ne √(2mₑ/πk_BTe) (ℏ/mₑ)² (R⃗·R⃗) [Cₙ + G(Δω)]

with R⃗ "the electron position operator operating in the subspace of PQN n." So
`Φ ∝ R⃗·R⃗ = r²` is a **matrix in the n-subspace**. `r²` is diagonal in `|nlm⟩`
but with strongly l-dependent values (`⟨r²⟩₄ₗ = 648, 600, 504, 360` for l=0–3);
in the Stark–Zeeman eigenbasis (which mixes l) it acquires **off-diagonal SDT
couplings**. Through the non-Hermitian Liouvillian those off-diagonals produce the
complex intensity `a_k + i c_k`.

**StarkZee collapses this operator to its scalar trace.**
`electron_impact_width` uses `r2_avg = (1/n²) Σ(2l+1)⟨r²⟩ₙₗ` — a single number —
discarding both the per-SDT diagonal variation and all off-diagonals → `c_k ≡ 0`.

**Revised tractability** (walks back the earlier "out of reach"): the operator is
just l-resolved `r²`, whose closed form StarkZee already has. Implementing PPPB
mode = (a) use `r²` as an operator (diag(⟨r²⟩ₙₗ) in `|nlm⟩`, rotated into the SZ
eigenbasis) instead of averaging — this alone gives genuine per-SDT widths; plus
(b) the non-Hermitian / tetradic (upper⊕lower + interference) treatment to extract
`c_k`. Moderate, not a wall.

### Estimated `c_k` magnitude — Hβ at 1 kT, Nₑ=1e23 m⁻³, Te=5 eV (order-of-magnitude)

- Overlap: `γ_e / W_static = 3.9 meV / 49 meV ≈ 0.08` (electron HWHM vs the
  strength-weighted RMS Stark+Zeeman envelope).
- Operator coupling: off-diag/diag of `r²` in the SZ eigenbasis ≈ **0.21** at the
  mean microfield (the l-mixing; `⟨r²⟩ₙₗ` swings ±29% across l).
- First-order, upper-state-only: **c_k/a_k ~ R2 × f_off ≈ 0.08 × 0.21 ≈ 1–2 %**,
  plausibly a few % in the wings.

So `c_k ≪ a_k` holds here — as a *computed* result, not an assumption — making it
a ~1–2 % asymmetry (matters for precision B-field diagnostics, not gross shape).
**Caveats:** heuristic — first-order, upper-state only, `r²` as proxy for the full
tetradic Φ, and `R2×f_off` is a scaling argument, not a derived formula; true `c_k`
could differ by a factor of a few. Not a substitute for building the operator.

**RESOLVED — ZEST is also diagonal-per-q (no cross-q).** Confirmed from ZEST.html:

- After diagonalizing the (real, symmetric) Liouville operator and **neglecting
  the off-diagonal elements of φ′** and **"the interference terms between upper
  and lower states,"** ZEST's field profile is
  `I(ω,F) = (C/π) Σ_{i→f} ⟨i|ρ|i⟩ |⟨i|d|f⟩|² · φ_if/[(ω−ω_if)²+φ_if²]` —
  modulus-squared real weight, **no `c_k`**.
- Polarizations combine as independent scalar intensities (ZEST Eq. 18):
  `I_∥ = I_+ + I_-`, `I_⊥ = ½(I_+ + I_-) + I_0`, each `I_q` (q = M−M′) built
  separately from `|⟨i|d_q|f⟩|²`. **No `D_{d,k} D_{q,k}*` cross terms.**
  (`I_⊥` = `I_π + ½(I_σ+ + I_σ−)` = StarkZee's transverse `pi + 0.5(sp+sm)`.)

So the current code already matches ZEST exactly on this point, and ZEST *says*
it neglects these interference terms ("to be relaxed in future versions").
**Consequence:** the cross-q `c_k` machinery is purely additive PPPB-only — it
does not touch the ZEST path. The `c_k` definition is still pending the Calisti
source (user checking).

## How ZEST handles the electron-impact operator (and why ours is cruder)

All three codes define the **same** broadening *operator* (ZEST Eq. 8 / PPPB Eq. 19):

    φ(Δω) = (4π/3) Ne √(2m/πk_BTe) (e²a₀/4πε₀ℏ)² (d·d†) G(Δω)        [ZEST Eq. 8]
    Φ(Δω) = (4π/3) Ne √(2mₑ/πk_BTe) (ℏ/mₑ)² (R⃗·R⃗) [Cₙ + G(Δω)]       [PPPB Eq. 19]

`d·d†` and `R⃗·R⃗` are the same operator family (`d = −eR`) — neither is a scalar.
The codes differ only in **what they do with it**:

| code              | electron-impact broadening                        | per-SDT width            | c_k |
|-------------------|---------------------------------------------------|--------------------------|-----|
| **StarkZee (now)**| scalar shell-average `⟨r²⟩ₙ`, one number ∀ SDT     | no (only via G(Δω))      | no  |
| **ZEST**          | operator `d·d†`, **diagonal** in SZ eigenbasis     | **yes** `φ_if=⟨i\|d·d†\|i⟩…` | no  |
| **PPPB**          | operator `d·d†`, **full** (off-diagonal kept)      | yes                      | **yes** |

ZEST diagonalizes the (real, symmetric) Liouville operator (eigvecs `P`),
transforms `φ′ = PᵀφP`, then *"neglects the off-diagonal elements of φ′ and
retains only the diagonal `φ_if`"* — each Stark component gets its own width from
its actual dipole/l content. Dropping the off-diagonal **is** the `c_k = 0`
approximation. **StarkZee is cruder than ZEST**: it averages the operator down to
a scalar, losing the per-component diagonal as well.

**Upper AND lower manifolds.** ZEST's text: *"we neglect the interference terms
between upper and lower states in φ(ω)"* — which means φ keeps **both** the
upper-state (`d·d†`) and lower-state (`d†·d`) contributions and drops only the
upper↔lower cross (interference) term (and the imaginary part / static shifts).
So `φ_if = φ^upper_ii + φ^lower_ff`. The κ_m cutoff (Eq. 11) is on the upper
shell n, but that is just the momentum cutoff, not the manifold structure.

**StarkZee is more approximate than ZEST here too:** `electron_impact_width`
docstring — *"the width refers to the upper-level broadening only; the lower-level
contribution is neglected."* So StarkZee drops the lower-state piece entirely.

| contribution            | ZEST                  | StarkZee (now)        |
|-------------------------|-----------------------|-----------------------|
| upper-state `d·d†`      | ✓ (operator diagonal) | ✓ but scalar-averaged |
| lower-state `d†·d`      | ✓                     | ✗ dropped             |
| upper↔lower interference| ✗                     | ✗                     |

For Hβ (4→2) the lower (n=2) piece is sub-dominant (`⟨r²⟩∝n⁴`) but nonzero.

**Revised recommendation for the electron operator** (supersedes "Phase 1 =
frequency-dependent G on the scalar"): adopt the **ZEST diagonal** approach —
build the broadening operator as `diag of d·d† (upper) + diag of d†·d (lower)` in
the SZ eigenbasis, take the per-SDT diagonal as the width. Hermitian, no `c_k`, no
non-Hermitian diagonalization, and it restores the lower-state contribution
StarkZee currently drops. Same first step as the `c_k` groundwork (build + rotate
the operator); ZEST stops at the diagonal, PPPB keeps the off-diagonal.

## Proposed implementation — phased

**API.** Add `model="zest"` (default, preserves current behavior) → `"pppb"` to
`calculate_ffm_profile`. Optionally `ion_velocity="most_probable"|"thermal"`
(default keeps √2 / ZEST).

**Shared refactor (both models).** The collection loop currently stores
`weight·|mixed_D|²`. Change it to store, per SDT: `ω_k` (=`dE`), the **complex**
amplitude `d_k` (=`mixed_D`), and `weight`. Profile assembly becomes one helper:

```python
def _ffm_lineshape(omega, w_k, a_k, omega_k, gamma_k, nu_i):
    L = 1/(nu_i + gamma_k + 1j*(omega[:, None] - omega_k[None, :]))
    r2 = a_k.sum()
    S_num = (w_k/r2 * L).sum(1)
    S_den = (a_k/r2 * L).sum(1)
    return (r2/np.pi) * np.real(S_num / (1 - nu_i*S_den))
```

- `a_k` = `weight·|d_k|²` (both models).
- `γ_k`: ZEST → `electron_impact_width(0.0, …)` scalar; PPPB → per-SDT
  `electron_impact_width(Δω_k, …)` vector (the static path already makes this exact
  call, `static_profile.py:417`).
- `w_k`: ZEST → `a_k`; PPPB → `a_k + i·c_k`.

**Phase 1 (the whole job):** `model` param + per-SDT `γ_k` + velocity
convention. Real, defensible PPPB-vs-ZEST comparison (frozen-vs-dispersive
electron width is visible in the wings) with **no theoretical uncertainty**. Ship
with a test asserting `model="pppb"` with frozen γ reproduces `model="zest"`
bit-for-bit. With `c_k` descoped (see above), the `pppb` model differs from
`zest` *only* by the per-SDT frequency-dependent width `γ(Δω_k)` and the optional
νᵢ velocity convention — both already supported by existing code.

**`c_k` (deferred):** not implemented — **not** because it is proven negligible
(it isn't, for our regime; see above), but because it requires the full
non-Hermitian electron-impact operator Φ as a matrix, which StarkZee lacks. Its
asymmetry magnitude for H Balmer at our conditions is untested; estimate it
empirically before committing to the Φ-operator overhaul.

## Validation harness (both phases)

1. **Reduction test:** `model="pppb"` with γ frozen == `model="zest"` (numerical
   identity), confirming the only active difference is the per-SDT width.
2. **Norm check:** `∫ I dω` conserved across models.
3. **Reproduce PPPB Fig.** (Balmer, 3 B values); inspect how the per-SDT
   frequency-dependent width reshapes the wings vs the frozen-γ ZEST profile.

## Notes / out of scope

- **D1 (Doppler):** neither FFM path folds Doppler yet; the static profile's Voigt
  machinery can be reused, but keep it orthogonal to the `model` switch.
- **D2 (width floor):** `ffm.py:215` `gamma_k += 1e-4` fudge → replace with the
  physical natural linewidth `w_natural_ev` (as `static_profile.py:369`).
