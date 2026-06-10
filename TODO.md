# TODO

---


## 1. Electric quadrupole (E2) transitions

**Goal:** assess whether adding the E2 transition operator is necessary for any realistic use case of this code.

**Current situation.** The code uses only the E1 dipole operator (Δl = ±1, q ∈ {−1, 0, +1}). Observable "forbidden" satellite lines in dense plasmas are already produced by Stark-induced E1 transitions — the microfield mixes l-states within the same n-shell, so formally E1-forbidden transitions gain amplitude. This mechanism is implemented and tested (`test_13_starkzee_satellites.py`).

**Why E2 is unlikely to matter here.** The E2/E1 intensity ratio scales as (r_atomic/λ)² ≈ (a₀/λ)². For H Balmer-α at λ = 656 nm this is ~5×10⁻⁹ — below any practical detection threshold. The ratio only becomes relevant in the X-ray regime (e.g., Fe XXVI at 0.18 nm, where E2/E1 ~ 10⁻³).

**The case that could matter.** Quadrupole Stark coupling — the interaction of the microfield *gradient* (not the field itself) with the atomic quadrupole moment — modifies `build_stark_matrix_general` rather than the transition operators. This is a distinct effect from E2 radiation and could be relevant at ICF or white-dwarf densities (Ne > 10²⁸ m⁻³). It is not the same as adding E2 emission.

**Conclusion.** Adding E2 transition operators is not justified for the visible/UV plasma conditions this code targets. Quadrupole Stark coupling is a plausible future extension if very-high-density regimes become a priority.
