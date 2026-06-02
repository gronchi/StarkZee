"""
test_03_hamiltonian_B0.py — Hamiltonian at B=0 (hydrogenic limit).

At B=0 and no electric field, the 2n² states within a given n must all
be degenerate at −Z²Ry/n², with only spin-orbit coupling splitting the
l > 0 sub-levels.  This test verifies those analytic limits.
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.atomic_hamiltonian import build_hamiltonian, diagonalize_hamiltonian, build_basis
from scipy.constants import fine_structure as FINE_STRUCTURE
from starkzee.utils import reduced_mass_rydberg_ev


def relerr(got, ref):
    return abs(got - ref) / abs(ref)


# ── 1. Hamiltonian is Hermitian ───────────────────────────────────────────────

@pytest.mark.parametrize("n,Z", [(1,1), (2,1), (2,6), (3,1), (4,4), (5,4)])
def test_hamiltonian_hermitian(n, Z):
    """H must be Hermitian at B=0."""
    H = build_hamiltonian(n, Z, B=0.0, quadratic_zeeman=False)
    diff = np.max(np.abs(H - H.conj().T))
    assert diff < 1e-15, f"n={n},Z={Z}: max|H-H†| = {diff}"


# ── 2. Correct size ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_hamiltonian_size(n):
    """H must be (2n²) × (2n²)."""
    H = build_hamiltonian(n, Z=1, B=0.0)
    assert H.shape == (2*n**2, 2*n**2), f"n={n}: shape {H.shape}"


# ── 3. Unperturbed energy (no SOC, no B) ─────────────────────────────────────

@pytest.mark.parametrize("n,Z", [(1,1), (2,1), (2,6), (3,1), (4,4), (5,4)])
def test_unperturbed_energy_n1_l0(n, Z):
    """
    For l=0 states at B=0, the diagonal of H is exactly -Z²Ry/n²
    (no SOC contribution for l=0). Fine structure disabled so only the
    unperturbed Coulomb energy appears on the diagonal.
    """
    H = build_hamiltonian(n, Z, B=0.0, quadratic_zeeman=False,
                                  fine_structure=False)
    basis = build_basis(n)
    En_exact = -(Z**2) * reduced_mass_rydberg_ev(Z, 1) / n**2

    for i, state in enumerate(basis):
        if state.l == 0:
            diag_val = H[i, i].real
            assert relerr(diag_val, En_exact) < 1e-10, (
                f"n={n},Z={Z},l=0 state: diag={diag_val:.6f}, expected={En_exact:.6f}"
            )


# ── 4. Eigenvalue centroid = unperturbed energy ───────────────────────────────

@pytest.mark.parametrize("n,Z", [(1,1), (2,1), (2,6), (3,2), (4,4)])
def test_eigenvalue_centroid(n, Z):
    """
    SOC only shifts levels within n; the mean eigenvalue must still equal
    the unperturbed energy -Z²Ry/n² (to first order, and exactly for l=0).
    Fine structure disabled — MV+Darwin legitimately shifts the centroid.
    """
    eigenvalues, _ = diagonalize_hamiltonian(n, Z, B=0.0, quadratic_zeeman=False,
                                         fine_structure=False)
    En_exact = -(Z**2) * reduced_mass_rydberg_ev(Z, 1) / n**2
    mean_E = np.mean(eigenvalues.real)
    assert relerr(mean_E, En_exact) < 1e-4, (
        f"n={n},Z={Z}: mean eigenvalue={mean_E:.6f}, expected={En_exact:.6f}"
    )


# ── 5. n=1 is purely degenerate (no l>0, no SOC) ─────────────────────────────

def test_n1_degenerate():
    """
    n=1 has only l=0 states. At B=0 with fine structure disabled, all 2
    eigenvalues must equal -Z²Ry exactly. (With fine structure enabled, both
    states shift by the same MV+Darwin amount but remain degenerate.)
    """
    for Z in [1, 4, 6]:
        evals, _ = diagonalize_hamiltonian(1, Z, B=0.0, quadratic_zeeman=False,
                                       fine_structure=False)
        En = -(Z**2) * reduced_mass_rydberg_ev(Z, 1)
        np.testing.assert_allclose(evals.real, En, rtol=1e-10,
                                   err_msg=f"n=1, Z={Z}: n=1 levels not degenerate")


# ── 6. n=2 SOC splitting magnitude ───────────────────────────────────────────

def test_n2_soc_splitting_hydrogen():
    """
    For H (Z=1), n=2 l=1 spin-orbit splitting:
    ξ = Z⁴ α² Ry / (n³ l(l+1)(l+½)) = α² Ry / (8 × 2 × 1.5) = α²Ry/24
    The 2p levels split into j=3/2 and j=1/2 with ΔE = (3/2)ξ.

    Reference value: ΔE_SOC(n=2, Z=1) ≈ 4.53e-5 eV
    """
    evals, _ = diagonalize_hamiltonian(n=2, Z=1, B=0.0, quadratic_zeeman=False)
    # Eigenvalues should cluster near -3.4014 eV with small SOC spread
    E_spread = evals.real.max() - evals.real.min()
    xi = (1**4) * (FINE_STRUCTURE**2) * reduced_mass_rydberg_ev(1, 1) / (2**3 * 1 * 2 * 1.5)
    # Maximum SOC spread is ~1.5 * xi (j=3/2 minus j=1/2)
    expected_spread_approx = 1.5 * xi
    print(f"H n=2 SOC: ξ={xi:.3e} eV, eigenvalue spread={E_spread:.3e} eV")
    assert E_spread < 1e-3, (
        f"n=2 Z=1 eigenvalue spread {E_spread:.3e} eV is too large (expect ~{expected_spread_approx:.3e} eV)"
    )
    assert E_spread > 1e-7, "SOC splitting is suspiciously zero"

def test_n2_soc_splitting_CVI():
    """
    For C VI (Z=6), n=2 SOC is larger: ξ = Z⁴α²Ry/(n³ l(l+1)(l+½))
    = 1296 × α² × Ry / 24 ≈ 1296 × 5.32e-5 × 13.6 / 24 ≈ 0.039 eV.
    This is significant but still << 367 eV.
    """
    evals, _ = diagonalize_hamiltonian(n=2, Z=6, B=0.0, quadratic_zeeman=False)
    E_spread = evals.real.max() - evals.real.min()
    xi = (6**4) * (FINE_STRUCTURE**2) * reduced_mass_rydberg_ev(6, 1) / (2**3 * 1 * 2 * 1.5)
    print(f"C VI n=2 SOC: ξ={xi:.4f} eV, eigenvalue spread={E_spread:.4f} eV")
    # Spread should be ~1.5 ξ
    assert relerr(E_spread, 1.5 * xi) < 0.5, (
        f"C VI n=2 SOC spread {E_spread:.4f} eV, expected ~{1.5*xi:.4f} eV"
    )


# ── 7. Eigenvalues are real ───────────────────────────────────────────────────

@pytest.mark.parametrize("n,Z", [(2,1), (2,6), (4,4), (5,4)])
def test_eigenvalues_real(n, Z):
    """np.linalg.eigh guarantees real eigenvalues for Hermitian H."""
    evals, _ = diagonalize_hamiltonian(n, Z, B=0.0, quadratic_zeeman=False)
    max_imag = np.max(np.abs(evals.imag))
    assert max_imag < 1e-15, f"n={n},Z={Z}: max imaginary part = {max_imag}"


# ── 8. Eigenvectors form a unitary matrix ────────────────────────────────────

@pytest.mark.parametrize("n,Z", [(2,1), (3,1), (4,4)])
def test_eigenvectors_unitary(n, Z):
    """Eigenvectors from np.linalg.eigh must satisfy V†V = I."""
    _, evecs = diagonalize_hamiltonian(n, Z, B=0.0, quadratic_zeeman=False)
    VdV = evecs.conj().T @ evecs
    I = np.eye(len(evecs))
    diff = np.max(np.abs(VdV - I))
    assert diff < 1e-12, f"n={n},Z={Z}: V†V deviation from I: {diff}"
