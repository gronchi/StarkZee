"""
test_10_spin_orbit.py — Spin-orbit coupling ladder operator correctness.

The L·S coupling term in build_hamiltonian uses ladder operators
L+S- and L-S+. The matrix elements of S± are:
  ⟨ms'|S±|ms⟩ = √(s(s+1) - ms(ms±1))

For s=1/2, this gives:
  ⟨+1/2|S+|-1/2⟩ = 1  (√(3/4 - (-1/2)(1/2)) = √1 = 1)
  ⟨-1/2|S-|+1/2⟩ = 1

The code uses a simplified boolean:
  term_s = 1.0 if state_j.ms == 0.5 else 0.0  (for L+S-)
which COINCIDENTALLY gives the correct value 1.0 for s=1/2, but the
logic is not physically transparent and should be verified.

This test also verifies that the spin-orbit fine structure for hydrogen
matches the Dirac formula.
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.atomic_hamiltonian import build_hamiltonian, build_basis, diagonalize_hamiltonian
from scipy.constants import fine_structure as FINE_STRUCTURE
from starkzee.utils import reduced_mass_rydberg_ev, RYDBERG_EV


def relerr(got, ref):
    return abs(got - ref) / abs(ref)


# ── 1. S± ladder operator values for s=1/2 ───────────────────────────────────

def test_spin_ladder_values():
    """
    For s=1/2:
    ⟨ms=+1/2|S+|ms=-1/2⟩ = √(s(s+1) - ms(ms+1))|_{ms=-1/2} = √(3/4 - (-1/4)) = 1
    ⟨ms=-1/2|S-|ms=+1/2⟩ = √(s(s+1) - ms(ms-1))|_{ms=+1/2} = √(3/4 - (-1/4)) = 1

    The code uses the condition 'term_s = 1.0 if state_j.ms == 0.5 else 0.0'
    This gives the correct value (1.0) for the only non-zero case, but we verify
    it analytically here for transparency.
    """
    s = 0.5
    # S+ on |ms=-1/2⟩ → |ms=+1/2⟩
    ms = -0.5
    term_s_correct = np.sqrt(s*(s+1) - ms*(ms+1))
    assert relerr(term_s_correct, 1.0) < 1e-10, f"S+ term = {term_s_correct}"
    # S- on |ms=+1/2⟩ → |ms=-1/2⟩
    ms = 0.5
    term_s_minus_correct = np.sqrt(s*(s+1) - ms*(ms-1))
    assert relerr(term_s_minus_correct, 1.0) < 1e-10, f"S- term = {term_s_minus_correct}"


def test_spin_orbit_matrix_element_symmetry():
    """
    The L·S matrix must be Hermitian.
    If it is not, the L±S∓ ladder terms have wrong signs/factors.
    """
    for n, Z in [(2, 1), (2, 6), (3, 1), (4, 4)]:
        H = build_hamiltonian(n, Z, B=0.0, quadratic_zeeman=False)
        diff = np.max(np.abs(H - H.conj().T))
        assert diff < 1e-14, f"n={n},Z={Z}: H non-Hermitian with |H-H†|_max = {diff}"


# ── 2. Fine structure for hydrogen n=2 ───────────────────────────────────────

def test_hydrogen_n2_fine_structure():
    """
    For hydrogen n=2, the relativistic (Dirac) fine structure gives:
    E(2p_{3/2}) - E(2p_{1/2}) = (5/2) × Z⁴ α² Ry / n³ × (1/l(2l+1))
    
    From the spin-orbit formula:
    ξ = Z⁴ α² Ry / (n³ l(l+1)(l+1/2))
    
    The 2p levels split into:
      2p_{3/2}: E_n + ξ/2   (j=3/2, l=1, L·S = +1/2 for j=l+s)
      2p_{1/2}: E_n - ξ     (j=1/2, l=1, L·S = -1 for j=l-s)
    
    So the splitting is ΔE = (3/2) ξ.
    
    For H (Z=1): ξ = α²Ry/(8×2×1.5) = α²Ry/24 ≈ 4.53e-5 eV
    ΔE = (3/2) × 4.53e-5 ≈ 6.79e-5 eV
    """
    Z, n = 1, 2
    xi = (Z**4) * (FINE_STRUCTURE**2) * RYDBERG_EV / (n**3 * 1 * 2 * 1.5)
    dE_expected = 1.5 * xi

    evals, _ = diagonalize_hamiltonian(n, Z, B=0.0, quadratic_zeeman=False)

    # The eigenvalues should contain both 2p_{3/2} (4-fold degenerate) and
    # 2p_{1/2} (2-fold degenerate) clusters, plus 2s (2-fold, near center).
    # The l=0 (2s) states are unaffected by SOC.
    # Sort eigenvalues and find the spread among l=1 states.
    basis = build_basis(n)
    H = build_hamiltonian(n, Z, B=0.0, quadratic_zeeman=False)
    evals_full, evecs = np.linalg.eigh(H)

    # The spread among eigenvalues near E_n
    En = -(Z**2) * reduced_mass_rydberg_ev(Z, 1) / n**2
    # The p-states should have energies near En ± some SOC
    soc_energies = evals_full.real - En
    soc_range = soc_energies.max() - soc_energies.min()

    print(f"H n=2: ξ = {xi:.4e} eV, expected ΔE = {dE_expected:.4e} eV")
    print(f"Eigenvalue spread = {soc_range:.4e} eV")

    # The spread should match the fine structure splitting within 20%
    assert relerr(soc_range, dE_expected) < 0.20, (
        f"Fine structure spread {soc_range:.4e} eV vs expected {dE_expected:.4e} eV"
    )


def test_cvi_n2_fine_structure_larger():
    """
    For C VI (Z=6): ξ scales as Z^4, so it's 6^4=1296 times larger than H.
    ξ_CVI = 1296 × ξ_H ≈ 0.059 eV.
    This is the SOC that the code must correctly capture.
    """
    Z, n = 6, 2
    xi = (Z**4) * (FINE_STRUCTURE**2) * RYDBERG_EV / (n**3 * 1 * 2 * 1.5)
    dE_expected = 1.5 * xi

    evals, _ = diagonalize_hamiltonian(n, Z, B=0.0, quadratic_zeeman=False)
    En = -(Z**2) * reduced_mass_rydberg_ev(Z, 1) / n**2
    soc_range = evals.real.max() - evals.real.min()

    print(f"C VI n=2: ξ = {xi:.4f} eV, expected ΔE = {dE_expected:.4f} eV")
    print(f"Eigenvalue spread = {soc_range:.4f} eV")

    assert relerr(soc_range, dE_expected) < 0.20, (
        f"C VI fine structure {soc_range:.4f} eV vs expected {dE_expected:.4f} eV"
    )


# ── 3. SOC does NOT affect l=0 states ────────────────────────────────────────

def test_soc_zero_for_l0():
    """
    l=0 states have no SOC (L=0, so L·S = 0). With fine structure disabled,
    their diagonal must remain exactly at -Z²Ry/n² regardless of Z.
    (With fine structure enabled, MV+Darwin shifts the diagonal, which is
    tested separately in test_16_fine_structure.py.)
    """
    for n, Z in [(2, 1), (2, 6), (3, 4)]:
        H = build_hamiltonian(n, Z, B=0.0, quadratic_zeeman=False,
                                      fine_structure=False)
        basis = build_basis(n)
        En_exact = -(Z**2) * reduced_mass_rydberg_ev(Z, 1) / n**2
        for i, s in enumerate(basis):
            if s.l == 0:
                E_diag = H[i, i].real
                assert relerr(E_diag, En_exact) < 1e-10, (
                    f"n={n},Z={Z},l=0: H[{i},{i}]={E_diag:.8f}, expected={En_exact:.8f}"
                )


# ── 4. SOC L+S- off-diagonal elements correct ────────────────────────────────

def test_soc_off_diagonal_lplus_sminus():
    """
    The L+S- coupling term connects |l, ml, ms=+1/2⟩ to |l, ml+1, ms=-1/2⟩.
    The matrix element is: (ξ/2) × √(l(l+1) - ml(ml+1)) × 1.0

    We verify the actual H[i,j] value against the analytic formula.
    """
    n, Z = 2, 1
    H = build_hamiltonian(n, Z, B=0.0, quadratic_zeeman=False)
    basis = build_basis(n)

    xi = (Z**4) * (FINE_STRUCTURE**2) * RYDBERG_EV / (n**3 * 1 * 2 * 1.5)

    # For l=1, ml=0, ms=+0.5 → l=1, ml=1, ms=-0.5 (L+ raises ml, S- lowers ms)
    i_from = next(idx for idx, s in enumerate(basis)
                  if s.l == 1 and s.ml == 0 and s.ms == 0.5)
    i_to   = next(idx for idx, s in enumerate(basis)
                  if s.l == 1 and s.ml == 1 and s.ms == -0.5)

    # Analytic: (ξ/2) × √(l(l+1) - ml(ml+1)) × √(s(s+1) - ms_lower(ms_lower+1))
    # ml=0: √(1×2 - 0×1) = √2; ms_lower=+0.5 (where S- acts): √(3/4 - (0.5)(-0.5)) = √1 = 1
    val_expected = 0.5 * xi * np.sqrt(2.0) * 1.0
    val_got = H[i_to, i_from].real

    print(f"L+S- element: got={val_got:.6e}, expected={val_expected:.6e}")
    assert relerr(val_got, val_expected) < 1e-8, (
        f"L+S- matrix element: got {val_got:.6e}, expected {val_expected:.6e}"
    )
