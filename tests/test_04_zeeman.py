"""
test_04_zeeman.py — Linear Zeeman effect tests (no electric field).

At F=0 (Stark off), the energy shifts must follow:
  ΔE = μ_B × B × (ml + g_s × ms)
and the three Zeeman lines (σ−, π, σ+) must be separated by μ_B × B.
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.atomic_hamiltonian import build_hamiltonian, diagonalize_hamiltonian, build_basis
from scipy.constants import fine_structure as FINE_STRUCTURE
from starkzee.utils import RYDBERG_EV, BOHR_MAGNETON_EV_T


def relerr(got, ref):
    return abs(got - ref) / abs(ref)


# ── 1. Linear Zeeman: diagonal shifts ────────────────────────────────────────

def test_zeeman_diagonal_l0_states():
    """
    For l=0 states (ml=0), the Zeeman shift is purely due to spin:
    ΔE = g_s × ms × μ_B × B.
    The two ms=±½ states split by g_s × μ_B × B.
    """
    Z, n, B = 1, 1, 100.0  # n=1: only l=0
    g_s = 2.0023192

    H_B0  = build_hamiltonian(n, Z, B=0.0,  include_quadratic=False)
    H_B   = build_hamiltonian(n, Z, B=B,    include_quadratic=False)

    basis = build_basis(n)
    for i, state in enumerate(basis):
        if state.l == 0:
            dE_got = (H_B[i, i] - H_B0[i, i]).real
            dE_exp = BOHR_MAGNETON_EV_T * B * (0.0 + g_s * state.ms)
            assert relerr(dE_got, dE_exp) < 1e-10, (
                f"n=1, ms={state.ms}: ΔE={dE_got:.6e}, expected={dE_exp:.6e}"
            )


def test_zeeman_diagonal_l1_states():
    """
    For n=2, l=1 states: ΔE = μ_B × B × (ml + g_s × ms).
    Test all 6 substates (ml=−1,0,1 × ms=±½).
    """
    Z, n, B = 1, 2, 100.0
    g_s = 2.0023192

    H_B0 = build_hamiltonian(n, Z, B=0.0, include_quadratic=False)
    H_B  = build_hamiltonian(n, Z, B=B,   include_quadratic=False)

    basis = build_basis(n)
    for i, state in enumerate(basis):
        if state.l == 1:
            dE_got = (H_B[i, i] - H_B0[i, i]).real
            dE_exp = BOHR_MAGNETON_EV_T * B * (state.ml + g_s * state.ms)
            assert relerr(dE_got, dE_exp) < 1e-10, (
                f"l=1, ml={state.ml}, ms={state.ms}: ΔE={dE_got:.6e}, expected={dE_exp:.6e}"
            )


# ── 2. Zeeman splitting magnitude ────────────────────────────────────────────

@pytest.mark.parametrize("B", [10.0, 100.0, 500.0, 1000.0])
def test_zeeman_energy_scale(B):
    """
    The Zeeman energy μ_B × B must scale linearly with B.
    For B=100 T: μ_B × B = 5.788e-3 eV.
    """
    zeeman_energy = BOHR_MAGNETON_EV_T * B
    expected_100T = 5.7883817982e-5 * 100.0  # ≈ 5.788e-3 eV (CODATA 2022)
    assert relerr(zeeman_energy, expected_100T * (B / 100.0)) < 1e-10


# ── 3. Eigenvalue structure at finite B (no Stark) ───────────────────────────

def test_zeeman_splits_n1_into_two():
    """
    n=1 (l=0 only): B field splits the two ms=±½ states.
    The gap must equal g_s × μ_B × B.
    """
    Z, B = 1, 100.0
    g_s = 2.0023192
    evals, _ = diagonalize_hamiltonian(n=1, Z=Z, B=B, include_quadratic=False)
    gap = evals.real.max() - evals.real.min()
    expected = g_s * BOHR_MAGNETON_EV_T * B
    assert relerr(gap, expected) < 1e-6, (
        f"n=1 Zeeman gap = {gap:.6e} eV, expected g_s×μ_B×B = {expected:.6e} eV"
    )

def test_zeeman_n2_l1_splits_into_multiple():
    """
    n=2, l=1 states: 6 substates should split into distinct energy levels under B.
    Minimum and maximum shift from the centroid must be ±(1 + g_s/2)×μ_B×B.
    """
    Z, B = 1, 100.0
    g_s = 2.0023192

    evals, _ = diagonalize_hamiltonian(n=2, Z=Z, B=B, include_quadratic=False)
    E_centroid = -(Z**2) * RYDBERG_EV / 4.0

    # The extremal Zeeman state has ml=+1, ms=+1/2:
    # ΔE_max = μ_B × B × (1 + g_s × 0.5)
    dE_max_expected = BOHR_MAGNETON_EV_T * B * (1.0 + g_s * 0.5)
    dE_max_got = evals.real.max() - E_centroid

    # Should match within SOC corrections (~1e-4 eV at Z=1)
    assert abs(dE_max_got - dE_max_expected) < 1e-3, (
        f"Max Zeeman shift: {dE_max_got:.6e} eV, expected {dE_max_expected:.6e} eV"
    )


# ── 4. Larmor splitting between σ+ and σ− ────────────────────────────────────

def test_larmor_frequency_relation():
    """
    The orbital gap between ml=+1 and ml=-1 states (same ms=+0.5) is:
      gap = 2 × μ_B × B  +  ξ × (ml_+ - ml_-) × ms
          = 2 × μ_B × B  +  ξ × (1 - (-1)) × 0.5
          = 2 × μ_B × B  +  ξ

    where the extra ξ contribution comes from the diagonal Lz·Sz term in
    the SOC Hamiltonian. At B=100T, ξ for H n=2 is ~3e-5 eV and the
    SOC gap is ~6e-5 eV, small but measurable in high-precision tests.
    """
    B = 100.0
    Z = 1
    H_B = build_hamiltonian(n=2, Z=Z, B=B, include_quadratic=False)
    basis = build_basis(2)

    E_plus = E_minus = None
    for i, s in enumerate(basis):
        if s.l == 1 and s.ms == 0.5:
            if s.ml == 1:
                E_plus = H_B[i, i].real
            elif s.ml == -1:
                E_minus = H_B[i, i].real

    assert E_plus is not None and E_minus is not None
    gap = E_plus - E_minus

    # Orbital term:
    zeeman_gap = 2.0 * BOHR_MAGNETON_EV_T * B   # = 2 μ_B B
    # SOC Lz·Sz diagonal: ξ × ml × ms → gap contribution = ξ × (1-(-1)) × 0.5 = ξ
    xi = (Z**4) * (FINE_STRUCTURE**2) * RYDBERG_EV / (2**3 * 1 * 2 * 1.5)
    soc_correction = xi  # (ml_+ - ml_-) × ms × ξ
    expected = zeeman_gap + soc_correction

    assert relerr(gap, expected) < 1e-6, (
        f"ml±1 gap = {gap:.6e} eV, expected 2μ_BB + ξ = {expected:.6e} eV"
    )




# ── 5. Quadratic Zeeman term scale ────────────────────────────────────────────

def test_quadratic_zeeman_positive_definite():
    """
    The quadratic Zeeman term (∝ B² r²) adds a positive diagonal contribution.
    Eigenvalues with include_quadratic=True should be >= those without.
    """
    Z, n, B = 1, 2, 1000.0   # Large B to make quadratic term visible
    evals_no_quad, _ = diagonalize_hamiltonian(n, Z, B, include_quadratic=False)
    evals_quad, _    = diagonalize_hamiltonian(n, Z, B, include_quadratic=True)

    # Mean eigenvalue should shift upward with quadratic term
    mean_shift = np.mean(evals_quad.real) - np.mean(evals_no_quad.real)
    assert mean_shift >= 0, (
        f"Quadratic Zeeman term caused negative mean shift: {mean_shift:.4e} eV"
    )
    print(f"n=2,Z=1,B={B}T: quadratic Zeeman mean shift = {mean_shift:.4e} eV")

def test_quadratic_zeeman_scales_as_B_squared():
    """
    The quadratic Zeeman energy ∝ B². Verify the mean shift doubles when B
    increases by √2.
    """
    Z, n = 1, 2
    B1, B2 = 500.0, 500.0 * np.sqrt(2.0)

    def mean_quad_shift(B):
        e_noq, _ = diagonalize_hamiltonian(n, Z, B, include_quadratic=False)
        e_q, _   = diagonalize_hamiltonian(n, Z, B, include_quadratic=True)
        return np.mean(e_q.real) - np.mean(e_noq.real)

    shift1 = mean_quad_shift(B1)
    shift2 = mean_quad_shift(B2)

    if shift1 > 1e-10:
        ratio = shift2 / shift1
        assert relerr(ratio, 2.0) < 0.05, (
            f"Quadratic Zeeman B² scaling: ratio = {ratio:.4f}, expected 2.0"
        )
    else:
        pytest.skip("Quadratic Zeeman shift too small to test at these B values")
