# TODO

---

## 1. Quantum-Defect Model for multi-electron ions

**Goal:** correctly compute Stark-Zeeman profiles for lithium-like ions such as C IV (n=5→4), which appear in Ferri et al. 2022 Figure 2 but cannot be reproduced with the current code.

**Why the current code is wrong for C IV.** The code assumes complete l-degeneracy within each n-shell (hydrogenic approximation). In C IV the outer electron orbits a C⁴⁺ core, and the resulting quantum defects lift that degeneracy significantly — for n=4 the s–p energy gap is ~0.35 eV, far larger than the typical Holtsmark field Stark shift. As a result, s and p states do not mix freely under the microfield, the π component is narrower and more structured, and the σ± wings are weaker than the hydrogenic model predicts. Running the code with `Z=4` produces output that is qualitatively wrong.

**What the fix looks like.** Replace the degenerate diagonal energies in `build_hamiltonian_general` with quantum-defect corrected values `E_{n,l} = -Z²Ry/(n − δ_l)²`, where δ_l is a per-species, per-l parameter. The radial matrix elements and angular couplings need no change — the suppression of s–p Stark mixing emerges naturally from the diagonalization once the energy gaps are present. The kwarg `quantum_defects={l: δ_l}` should be threaded through `solve_mascb_general`, `solve_starkzee_general`, and `calculate_static_profile_general` (separate dicts for upper and lower shells).

**C IV parameters (from NIST, n=4–6):**

| n | δ_0 (s) | δ_1 (p) | δ_2 (d) |
|---|---------|---------|---------|
| 4 | 0.757   | 0.278   | 0.040   |
| 5 | 0.756   | 0.277   | 0.039   |

Target: reproduce Ferri et al. Figure 2 (C IV n=5→4, Ne=2×10²⁵ m⁻³, Te=10 eV, B=0–500 T).

**Accuracy of the quantum-defect approximation.** It uses hydrogenic radial wavefunctions and ignores core polarization and configuration interaction. For n ≥ 4 these corrections are typically a few percent and the QDM is adequate for line-shape comparison. For low-n states (n=2,3) or precision work a model-potential calculation would be needed.

**Files to change:**

| File | What changes |
|------|-------------|
| `starkzee/atomic_hamiltonian.py` | `build_hamiltonian_general`, `solve_mascb_general`, `get_dipole_matrix_elements_general` — add `quantum_defects` kwarg |
| `starkzee/static_profile.py` | `calculate_static_profile_general`, `solve_starkzee_general` — add `quantum_defects_u/l` kwargs |
| `starkzee/utils.py` | Add a `SPECIES` dict with Z, Z_core, and quantum defects for common ions (C VI, C IV, …) |
| `tests/test_19_quantum_defect_civ.py` | New: check energy gaps match NIST, s–p mixing suppressed vs. hydrogenic, Zeeman splitting preserved |
| `examples/` | New reproduction script for Ferri et al. Figure 2 |

---

## 2. Electric quadrupole (E2) transitions

**Goal:** assess whether adding the E2 transition operator is necessary for any realistic use case of this code.

**Current situation.** The code uses only the E1 dipole operator (Δl = ±1, q ∈ {−1, 0, +1}). Observable "forbidden" satellite lines in dense plasmas are already produced by Stark-induced E1 transitions — the microfield mixes l-states within the same n-shell, so formally E1-forbidden transitions gain amplitude. This mechanism is implemented and tested (`test_13_starkzee_satellites.py`).

**Why E2 is unlikely to matter here.** The E2/E1 intensity ratio scales as (r_atomic/λ)² ≈ (a₀/λ)². For H Balmer-α at λ = 656 nm this is ~5×10⁻⁹ — below any practical detection threshold. The ratio only becomes relevant in the X-ray regime (e.g., Fe XXVI at 0.18 nm, where E2/E1 ~ 10⁻³).

**The case that could matter.** Quadrupole Stark coupling — the interaction of the microfield *gradient* (not the field itself) with the atomic quadrupole moment — modifies `build_stark_matrix_general` rather than the transition operators. This is a distinct effect from E2 radiation and could be relevant at ICF or white-dwarf densities (Ne > 10²⁸ m⁻³). It is not the same as adding E2 emission.

**Conclusion.** Adding E2 transition operators is not justified for the visible/UV plasma conditions this code targets. Quadrupole Stark coupling is a plausible future extension if very-high-density regimes become a priority.
