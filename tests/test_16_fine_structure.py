"""
test_16_fine_structure.py — Mass-velocity and Darwin correction correctness.

The combined MV + Darwin + SO terms reproduce the exact Dirac fine structure.
Key property: 2s_{1/2} and 2p_{1/2} must be exactly degenerate (to the precision
of first-order perturbation theory within a single n-shell).

Tests:
  1. Diagonal shift for l=0 states (MV + Darwin formula)
  2. Diagonal shift for l>0 states (MV only formula)
  3. 2s_{1/2} == 2p_{1/2} degeneracy for H and C VI
  4. 2p_{3/2} - 2p_{1/2} gap unchanged from SO-only
  5. n=1 eigenvalues shift by the correct MV+Darwin amount
  6. Centroid of n-shell shifts by the correct mean value
  7. Hamiltonian remains Hermitian with fine structure enabled
  8. Dirac eigenvalue formula verified for n=2 H and C VI
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.atomic_hamiltonian import (
    build_hamiltonian, diagonalize_hamiltonian, build_basis
)
from scipy.constants import fine_structure as FINE_STRUCTURE
from starkzee.utils import RYDBERG_EV


def relerr(got, ref):
    return abs(got - ref) / abs(ref)


def A(n, Z):
    """A = Z^4 * alpha^2 * Ry / n^4 (common pre-factor)."""
    return (Z**4) * (FINE_STRUCTURE**2) * RYDBERG_EV / n**4


# ── 1. Diagonal shift for l=0 states ─────────────────────────────────────────

@pytest.mark.parametrize("n,Z", [(1,1), (2,1), (2,6), (3,1), (3,6)])
def test_mvdarwin_diagonal_l0(n, Z):
    """For l=0 states: H_diag = En + (-A*(n - 3/4))."""
    H_fs  = build_hamiltonian(n, Z, B=0.0, include_quadratic=False,
                                      include_fine_structure=True)
    H_nofs = build_hamiltonian(n, Z, B=0.0, include_quadratic=False,
                                       include_fine_structure=False)
    basis = build_basis(n)
    expected_shift = -A(n, Z) * (n - 0.75)
    for i, s in enumerate(basis):
        if s.l == 0:
            got = (H_fs[i, i] - H_nofs[i, i]).real
            assert relerr(got, expected_shift) < 1e-10, (
                f"n={n},Z={Z},l=0: shift={got:.6e}, expected={expected_shift:.6e}"
            )


# ── 2. Diagonal shift for l>0 states ─────────────────────────────────────────

@pytest.mark.parametrize("n,Z,l", [
    (2, 1, 1), (2, 6, 1),
    (3, 1, 1), (3, 1, 2),
    (4, 6, 1), (4, 6, 2), (4, 6, 3),
])
def test_mvdarwin_diagonal_l_gt0(n, Z, l):
    """For l>0 states: H_diag shift = -A*(n/(l+1/2) - 3/4)."""
    H_fs  = build_hamiltonian(n, Z, B=0.0, include_quadratic=False,
                                      include_fine_structure=True)
    H_nofs = build_hamiltonian(n, Z, B=0.0, include_quadratic=False,
                                       include_fine_structure=False)
    basis = build_basis(n)
    expected_shift = -A(n, Z) * (n / (l + 0.5) - 0.75)
    for i, s in enumerate(basis):
        if s.l == l:
            got = (H_fs[i, i] - H_nofs[i, i]).real
            assert relerr(got, expected_shift) < 1e-10, (
                f"n={n},Z={Z},l={l}: shift={got:.6e}, expected={expected_shift:.6e}"
            )


# ── 3. 2s_{1/2} == 2p_{1/2} degeneracy ──────────────────────────────────────

@pytest.mark.parametrize("Z", [1, 6])
def test_2s_2p_half_degenerate(Z):
    """With fine structure, 2s_{1/2} and 2p_{1/2} are exactly degenerate.

    n=2 has 8 states: 4 with j=1/2 (2s_{1/2} × 2 + 2p_{1/2} × 2)
    all at the same energy, and 4 with j=3/2 (2p_{3/2}) at a higher energy.
    """
    evals, _ = diagonalize_hamiltonian(n=2, Z=Z, B=0.0, include_quadratic=False,
                                    include_fine_structure=True)
    evals_sorted = np.sort(evals.real)

    # Lower 4: j=1/2 group; upper 4: j=3/2 group
    lower = evals_sorted[:4]
    upper = evals_sorted[4:]

    # Lower group must be degenerate (spread < 1e-10 eV)
    spread_lower = lower.max() - lower.min()
    assert spread_lower < 1e-10, (
        f"Z={Z}: j=1/2 group not degenerate: spread = {spread_lower:.3e} eV"
    )

    # Upper group must also be degenerate
    spread_upper = upper.max() - upper.min()
    assert spread_upper < 1e-10, (
        f"Z={Z}: j=3/2 group not degenerate: spread = {spread_upper:.3e} eV"
    )


# ── 4. 2p_{3/2} - 2p_{1/2} gap unchanged from SO-only ───────────────────────

@pytest.mark.parametrize("Z", [1, 6])
def test_fine_structure_gap_unchanged(Z):
    """The 2p_{3/2} - 2p_{1/2} energy gap equals 3xi/2 with or without MV+Darwin.

    Without FS the sorted order is: 2p_{1/2}(×2), 2s(×2), 2p_{3/2}(×4) for H,
    or the same ordering for C VI (2p_{1/2} is always lowest since xi > 0).
    The gap is correctly measured as sorted[4] - sorted[0], spanning both groups.
    With FS all j=1/2 states collapse to the same level, giving the same gap.
    """
    xi = (Z**4) * (FINE_STRUCTURE**2) * RYDBERG_EV / (2**3 * 1 * 2 * 1.5)
    expected_gap = 1.5 * xi  # = A = Z^4 alpha^2 Ry / n^4

    for fs in [True, False]:
        evals, _ = diagonalize_hamiltonian(n=2, Z=Z, B=0.0, include_quadratic=False,
                                        include_fine_structure=fs)
        evals_sorted = np.sort(evals.real)
        # First 4 sorted eigenvalues contain j=1/2 states (possibly mixed with 2s
        # when fs=False); last 4 are always 2p_{3/2}. The gap min(upper) - min(all)
        # equals 3xi/2 in both cases.
        gap = evals_sorted[4] - evals_sorted[0]
        assert relerr(gap, expected_gap) < 1e-6, (
            f"Z={Z}, fs={fs}: gap={gap:.6e} eV, expected={expected_gap:.6e} eV"
        )


# ── 5. n=1 eigenvalues shift by MV+Darwin amount ─────────────────────────────

@pytest.mark.parametrize("Z", [1, 4, 6])
def test_n1_mvdarwin_shift(Z):
    """n=1 has only l=0, so both eigenvalues shift by -A*(1 - 3/4) = -A/4."""
    expected_shift = -A(n=1, Z=Z) * (1.0 - 0.75)
    evals_fs,   _ = diagonalize_hamiltonian(n=1, Z=Z, B=0.0, include_fine_structure=True)
    evals_nofs, _ = diagonalize_hamiltonian(n=1, Z=Z, B=0.0, include_fine_structure=False)
    shifts = evals_fs.real - evals_nofs.real
    for s in shifts:
        assert relerr(s, expected_shift) < 1e-10, (
            f"Z={Z}: n=1 shift={s:.6e}, expected={expected_shift:.6e}"
        )


# ── 6. Shell centroid shifts by correct mean value ───────────────────────────

@pytest.mark.parametrize("n,Z", [(2, 1), (2, 6), (3, 1)])
def test_shell_centroid_shift(n, Z):
    """Mean eigenvalue shift equals the degeneracy-weighted MV+Darwin average."""
    evals_fs,   _ = diagonalize_hamiltonian(n, Z, B=0.0, include_fine_structure=True)
    evals_nofs, _ = diagonalize_hamiltonian(n, Z, B=0.0, include_fine_structure=False)
    mean_shift_got = np.mean(evals_fs.real) - np.mean(evals_nofs.real)

    # Weighted mean: (2 * delta_l0 + sum_{l>0} (2*(2l+1)) * delta_l) / 2n^2
    basis = build_basis(n)
    expected = 0.0
    for s in basis:
        l = s.l
        if l == 0:
            expected += -A(n, Z) * (n - 0.75)
        else:
            expected += -A(n, Z) * (n / (l + 0.5) - 0.75)
    expected /= len(basis)

    assert relerr(mean_shift_got, expected) < 1e-8, (
        f"n={n},Z={Z}: mean shift={mean_shift_got:.6e}, expected={expected:.6e}"
    )


# ── 7. Hermiticity preserved ──────────────────────────────────────────────────

@pytest.mark.parametrize("n,Z,B", [
    (2, 1, 0.0), (2, 6, 0.0), (3, 1, 15.0), (4, 6, 5.0)
])
def test_hamiltonian_hermitian_with_fs(n, Z, B):
    """H must remain Hermitian when fine structure is enabled."""
    H = build_hamiltonian(n, Z, B=B, include_quadratic=True,
                                  include_fine_structure=True)
    diff = np.max(np.abs(H - H.conj().T))
    assert diff < 1e-14, f"n={n},Z={Z},B={B}T: |H-H†|_max = {diff:.3e}"


# ── 8. Eigenvalues match the Dirac fine structure formula ────────────────────

@pytest.mark.parametrize("n,Z", [(2, 1), (2, 6)])
def test_dirac_eigenvalue_formula(n, Z):
    """Eigenvalues with fine structure match the Dirac formula E_nj = En * [1 + (alphaZ)^2/n^2 * (n/(j+1/2) - 3/4)].

    For n=2: two groups j=1/2 and j=3/2.
    """
    En = -(Z**2) * RYDBERG_EV / n**2

    # Dirac corrections
    dE_j12 = -A(n, Z) * (n / (0.5 + 0.5) - 0.75)   # j = 1/2: n/(j+1/2)=n/1
    dE_j32 = -A(n, Z) * (n / (1.5 + 0.5) - 0.75)   # j = 3/2: n/(j+1/2)=n/2

    evals, _ = diagonalize_hamiltonian(n=n, Z=Z, B=0.0, include_quadratic=False,
                                    include_fine_structure=True)
    evals_sorted = np.sort(evals.real)

    E_j12_got = np.mean(evals_sorted[:4])
    E_j32_got = np.mean(evals_sorted[4:])

    assert relerr(E_j12_got, En + dE_j12) < 1e-8, (
        f"Z={Z} j=1/2: E={E_j12_got:.8f}, expected={En + dE_j12:.8f}"
    )
    assert relerr(E_j32_got, En + dE_j32) < 1e-8, (
        f"Z={Z} j=3/2: E={E_j32_got:.8f}, expected={En + dE_j32:.8f}"
    )
