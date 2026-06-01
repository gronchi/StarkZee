"""
test_05_stark.py — Stark effect tests (B=0).

Verifies:
  - Stark matrix is Hermitian
  - Linear Stark shift for H n=2 matches the analytic first-order result
  - Zero-field limit: Stark matrix vanishes
  - Within-n radial element formula correctness
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.static_profile import build_stark_matrix, solve_starkzee
from starkzee.atomic_hamiltonian import build_hamiltonian, build_basis
from scipy.constants import e as E_CHARGE
from starkzee.utils import reduced_mass_rydberg_ev, A0


def relerr(got, ref):
    return abs(got - ref) / abs(ref)


# ── 1. Stark matrix is Hermitian ─────────────────────────────────────────────

@pytest.mark.parametrize("n,Z,Fz,Fx", [
    (2, 1,  1e9,  0.0),
    (2, 6,  5e9,  1e9),
    (3, 1,  2e9,  2e9),
    (4, 4,  1e8,  0.0),
])
def test_stark_matrix_hermitian(n, Z, Fz, Fx):
    """Stark perturbation matrix V_E must be Hermitian."""
    V = build_stark_matrix(n, Z, Fz, Fx)
    diff = np.max(np.abs(V - V.conj().T))
    assert diff < 1e-20, f"n={n},Z={Z}: max|V-V†| = {diff}"


def test_stark_matrix_zero_at_zero_field():
    """At F=0, Stark matrix must be identically zero."""
    for n in [1, 2, 3]:
        V = build_stark_matrix(n, Z=1, Fz=0.0, Fx=0.0)
        assert np.all(V == 0), f"n={n}: Stark matrix non-zero at F=0"


def test_stark_matrix_scales_linearly_with_field():
    """Stark shift is linear in F (first-order perturbation theory)."""
    n, Z = 2, 1
    F1, F2 = 1e9, 2e9
    V1 = build_stark_matrix(n, Z, F1, 0.0)
    V2 = build_stark_matrix(n, Z, F2, 0.0)
    ratio = np.max(np.abs(V2)) / np.max(np.abs(V1))
    assert relerr(ratio, 2.0) < 1e-10, f"Stark not linear in F: ratio={ratio}"


# ── 2. Within-n radial element formula ───────────────────────────────────────

def test_within_n_radial_element_n2_l1():
    """
    For the n=2 manifold, the within-n radial element:
    ⟨n=2, l=1|r|n=2, l=0⟩ = (3n/2Z)√(n²-l²) where l = max(l1,l2) = 1
    = (3×2/2Z) × √(4-1) = (3/Z) × √3

    For Z=1: r_val = 3√3 ≈ 5.196 a0.

    This is the formula used in build_stark_matrix.
    We verify it by looking at the matrix element between states
    |n=2,l=0,ml=0⟩ and |n=2,l=1,ml=0⟩ with Fz only (q=0).
    """
    from starkzee.atomic_hamiltonian import angular_dipole_element
    n, Z = 2, 1
    Fz = 1.0   # V/m (arbitrary for this test)

    # Expected radial: (3n/2Z)√(n²-1²) = 3√3 for Z=1,n=2
    r_expected = (3.0 * n / (2.0 * Z)) * np.sqrt(n**2 - 1**2)  # in a0

    # Angular factor for q=0, l1=1,ml1=0 → l2=0,ml2=0
    ang = angular_dipole_element(1, 0, 0, 0, 0)   # = 1/√3

    # Expected matrix element |V_E[state_10, state_00]| = Fz × r_expected × ang × A0 (in eV)
    V_exp = abs(Fz * r_expected * ang * A0)

    V = build_stark_matrix(n, Z, Fz=Fz, Fx=0.0)
    basis = build_basis(n)

    # Find the l=0,ml=0,ms=+0.5 and l=1,ml=0,ms=+0.5 indices
    i_s, i_p = None, None
    for i, s in enumerate(basis):
        if s.l == 0 and s.ml == 0 and s.ms == 0.5:
            i_s = i
        if s.l == 1 and s.ml == 0 and s.ms == 0.5:
            i_p = i

    assert i_s is not None and i_p is not None
    V_got = abs(V[i_p, i_s].real)

    print(f"n={n},Z={Z}: |V[2p₀|2s]| = {V_got:.6e} eV, expected {V_exp:.6e} eV")
    assert relerr(V_got, V_exp) < 1e-8, (
        f"Stark matrix element mismatch: got {V_got:.6e}, expected {V_exp:.6e}"
    )


# ── 3. Linear Stark effect for H n=2 ─────────────────────────────────────────

def test_linear_stark_energy_H_n2():
    """
    For H (Z=1) n=2 in a field Fz along z (B=0), the first-order
    Stark splitting follows:
      ΔE = ±(3/2) n e a0 F / Z = ±3 e a0 F  (for n=2, Z=1)

    The eigenvalues of the n=2 manifold at small F should split as:
      E = E0 ± 3 e a0 F  (for the m=0 states that mix in first order)
      plus two unshifted states (m=±1 don't mix in first order with z-field only)

    Reference: Eq. from Bethe & Salpeter / textbook Stark effect.
    """
    Z, n = 1, 2
    E0 = -(Z**2) * reduced_mass_rydberg_ev(Z, 1) / n**2

    # Use a small field where first-order perturbation theory is valid
    # F0 (normal field for Ne~1e18) ~ 1e8 V/m; use F = 1e7 V/m
    F = 1e7   # V/m

    evals, _ = solve_starkzee(n, Z, B=0.0, Fz=F, Fx=0.0,
                                           include_quadratic=False)
    deviations = np.sort((evals.real - E0))

    # First-order prediction: ΔE = ±3 × e × a0 × F
    # In eV: ΔE = ±3 × A0 × F (since A0 in meters, F in V/m → eV directly)
    dE_first_order = 3.0 * A0 * F  # in eV
    print(f"H n=2 linear Stark: 1st-order ΔE = ±{dE_first_order:.4e} eV")
    print(f"Eigenvalue deviations from E0: {deviations}")

    # The maximum deviation should match the first-order prediction within ~10%
    # (higher-order corrections are small for F = 1e7 V/m)
    max_dev = np.max(np.abs(deviations))
    assert relerr(max_dev, dE_first_order) < 0.20, (
        f"Max Stark deviation {max_dev:.4e} differs from 1st-order {dE_first_order:.4e}"
    )


# ── 4. Stark + Zeeman combined Hermitian ─────────────────────────────────────

@pytest.mark.parametrize("n,Z,B,Fz,Fx", [
    (2, 1,  100.0, 1e9, 0.0),
    (2, 6,  100.0, 5e9, 1e9),
    (3, 1,    0.0, 2e9, 2e9),
    (4, 4,  200.0, 1e8, 5e7),
])
def test_combined_hamiltonian_hermitian(n, Z, B, Fz, Fx):
    """Total Hamiltonian H_atom + V_Stark must be Hermitian."""
    from starkzee.static_profile import build_stark_matrix
    from starkzee.atomic_hamiltonian import build_hamiltonian
    H_atom = build_hamiltonian(n, Z, B, include_quadratic=True)
    V_E    = build_stark_matrix(n, Z, Fz, Fx)
    H_tot  = H_atom + V_E
    diff   = np.max(np.abs(H_tot - H_tot.conj().T))
    assert diff < 1e-18, f"n={n},Z={Z},B={B}: |H+V - (H+V)†| = {diff}"


def test_stark_selection_rules_no_ms_mixing():
    """
    The Stark matrix must NOT mix states with different ms (spin is not
    coupled to the electric field). Off-diagonal elements with Δms ≠ 0
    must be zero.
    """
    n, Z = 2, 1
    V = build_stark_matrix(n, Z, Fz=1e9, Fx=0.0)
    basis = build_basis(n)

    for i, si in enumerate(basis):
        for j, sj in enumerate(basis):
            if si.ms != sj.ms and abs(V[i, j]) > 1e-30:
                pytest.fail(
                    f"Stark matrix mixes ms={si.ms} and ms={sj.ms}: "
                    f"V[{i},{j}] = {V[i,j]}"
                )
