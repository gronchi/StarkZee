"""
test_11_quadratic_zeeman.py — Quadratic Zeeman term correctness.

The diamagnetic (quadratic Zeeman) term is
  H_QZ = (e²B²/8mₑ) r² sin²θ

Its matrix elements in the hydrogenic basis split into a radial part
  ⟨n,l₁|r²|n,l₂⟩  (a₀²)
and an angular part
  ⟨l₁,mₗ|sin²θ|l₂,mₗ⟩

This file tests:
  1. Off-diagonal radial element ⟨n,l|r²|n,l±2⟩ via numerical integration
  2. Diagonal radial element ⟨n,l|r²|n,l⟩ vs. analytic formula
  3. Angular diagonal element ⟨l,mₗ|sin²θ|l,mₗ⟩ vs. analytic values
  4. Angular off-diagonal element ⟨l,mₗ|cos²θ|l+2,mₗ⟩ (correct formula)
  5. Hermiticity of the full Hamiltonian with quadratic_zeeman=True
  6. QZ shifts all eigenvalues upward (positive semi-definite contribution)
  7. QZ magnitude scales as B²
  8. QZ shifts are small vs. linear Zeeman at moderate B for Z=6
"""
import numpy as np
import pytest
from scipy.integrate import quad
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.radiator import (
    build_hamiltonian, build_basis, diagonalize_hamiltonian,
    radial_wavefunction, radial_r2_element
)
from scipy.constants import e as E_CHARGE, m_e as M_E
from starkzee.utils import RYDBERG_EV, BOHR_MAGNETON_EV_T, A0


def relerr(got, ref):
    return abs(got - ref) / abs(ref)


def numerical_r2(n, l1, l2, Z):
    """Reference: ∫ R_{nl1}(r) R_{nl2}(r) r⁴ dr in a₀²."""
    val, _ = quad(
        lambda r: radial_wavefunction(r, n, l1, Z) *
                  radial_wavefunction(r, n, l2, Z) * r**4,
        0, 300, limit=300
    )
    return val


def analytic_r2_diagonal(n, l, Z):
    """Closed-form ⟨n,l|r²|n,l⟩ = (n²/2Z²)(5n²+1−3l(l+1)) in a₀²."""
    return (n**2 / (2.0 * Z**2)) * (5.0 * n**2 + 1.0 - 3.0 * l * (l + 1.0))


# ── 1. Off-diagonal radial element ───────────────────────────────────────────

@pytest.mark.parametrize("n,l1,l2,Z", [
    (3, 0, 2, 1),
    (3, 2, 0, 1),
    (3, 0, 2, 6),
    (4, 0, 2, 1),
    (4, 1, 3, 1),
    (4, 0, 2, 6),
    (5, 0, 2, 1),
    (5, 1, 3, 1),
    (5, 2, 4, 1),
])
def test_radial_r2_off_diagonal_vs_numerical(n, l1, l2, Z):
    """radial_r2_element() must match direct numerical integration within 1%."""
    ref = numerical_r2(n, l1, l2, Z)
    got = radial_r2_element(n, l1, l2, Z)
    # For very small elements allow absolute tolerance
    if abs(ref) < 1e-8:
        assert abs(got - ref) < 1e-6, (
            f"n={n},l1={l1},l2={l2},Z={Z}: got={got:.6e}, ref={ref:.6e}"
        )
    else:
        assert relerr(got, ref) < 0.01, (
            f"n={n},l1={l1},l2={l2},Z={Z}: got={got:.4f}, ref={ref:.4f}, "
            f"relerr={relerr(got,ref):.4f}"
        )


def test_radial_r2_symmetric():
    """⟨n,l₁|r²|n,l₂⟩ = ⟨n,l₂|r²|n,l₁⟩  (real symmetric)."""
    for n, Z in [(3, 1), (4, 1), (5, 6)]:
        for l1 in range(n - 2):
            l2 = l1 + 2
            assert abs(radial_r2_element(n, l1, l2, Z) - radial_r2_element(n, l2, l1, Z)) < 1e-10, (
                f"n={n},l1={l1},l2={l2}: r2({l1},{l2})≠r2({l2},{l1})"
            )


def test_radial_r2_scales_as_inv_Z2():
    """⟨n,l₁|r²|n,l₂⟩ ∝ 1/Z² (hydrogenic scaling)."""
    n, l1, l2 = 3, 0, 2
    r2_Z1 = radial_r2_element(n, l1, l2, Z=1)
    r2_Z6 = radial_r2_element(n, l1, l2, Z=6)
    expected = r2_Z1 / 36.0   # 1/6²
    assert relerr(r2_Z6, expected) < 1e-4, (
        f"Z-scaling: r2(Z=6)={r2_Z6:.6f}, expected={expected:.6f}"
    )


# ── 2. Diagonal radial element ───────────────────────────────────────────────

@pytest.mark.parametrize("n,l,Z", [
    (2, 0, 1), (2, 1, 1), (3, 0, 1), (3, 1, 1), (3, 2, 1),
    (2, 0, 6), (3, 2, 6),
])
def test_radial_r2_diagonal_analytic(n, l, Z):
    """⟨n,l|r²|n,l⟩ from numerical integration matches the closed-form formula."""
    ref_analytic = analytic_r2_diagonal(n, l, Z)
    got_numerical = numerical_r2(n, l, l, Z)
    assert relerr(got_numerical, ref_analytic) < 1e-4, (
        f"n={n},l={l},Z={Z}: numerical={got_numerical:.4f}, analytic={ref_analytic:.4f}"
    )


# ── 3. Angular diagonal ⟨l,mₗ|sin²θ|l,mₗ⟩ ──────────────────────────────────

def analytic_sin2_diagonal(l, ml):
    """⟨l,mₗ|sin²θ|l,mₗ⟩ = 1 − ⟨cos²θ⟩ from the closed-form formula."""
    if l == 0:
        return 2.0 / 3.0
    cos2 = (2.0*l**2 + 2.0*l - 1.0 - 2.0*ml**2) / ((2.0*l - 1.0) * (2.0*l + 3.0))
    return 1.0 - cos2


def test_sin2_diagonal_sum_rule():
    """
    ∑_{mₗ=-l}^{l} (2l+1)⁻¹ ⟨l,mₗ|sin²θ|l,mₗ⟩ must equal 2/3
    (spherical average of sin²θ).
    """
    for l in range(0, 5):
        avg = sum(analytic_sin2_diagonal(l, ml) for ml in range(-l, l+1)) / (2*l + 1)
        assert relerr(avg, 2.0/3.0) < 1e-10, (
            f"l={l}: spherical average of sin²θ = {avg:.8f}, expected 2/3"
        )


@pytest.mark.parametrize("l,ml,expected", [
    (0,  0, 2.0/3.0),
    (1,  0, 2.0/5.0),   # cos2 = 3/5, sin2 = 2/5
    (1,  1, 4.0/5.0),   # cos2 = 1/5, sin2 = 4/5
    (1, -1, 4.0/5.0),
    (2,  0, 10.0/21.0), # cos2 = 11/21
    (2,  2, 6.0/7.0),   # cos2 = 1/7 = 3/21, sin2 = 6/7
])
def test_sin2_diagonal_spot_values(l, ml, expected):
    """Spot-check analytic sin²θ diagonal values."""
    got = analytic_sin2_diagonal(l, ml)
    assert relerr(got, expected) < 1e-10, (
        f"l={l},ml={ml}: got={got:.8f}, expected={expected:.8f}"
    )


# ── 4. Angular off-diagonal ⟨l,mₗ|cos²θ|l+2,mₗ⟩ ────────────────────────────

def analytic_cos2_offdiag(l_low, ml):
    """⟨l_low,mₗ|cos²θ|l_low+2,mₗ⟩ from Wigner-Eckart / Racah algebra."""
    num = ((l_low + 1.0)**2 - ml**2) * ((l_low + 2.0)**2 - ml**2)
    den = (2.0*l_low + 1.0) * (2.0*l_low + 3.0)**2 * (2.0*l_low + 5.0)
    return np.sqrt(num / den)


def test_cos2_offdiag_sign_pattern():
    """cos²θ off-diagonal element is always non-negative."""
    for l in range(0, 5):
        for ml in range(-l, l+1):
            val = analytic_cos2_offdiag(l, ml)
            assert val >= 0.0, f"l={l},ml={ml}: negative cos² off-diag {val}"


@pytest.mark.parametrize("l_low,ml,expected", [
    # ⟨l_low,ml|cos²θ|l_low+2,ml⟩ = √[((l+1)²-ml²)((l+2)²-ml²) / ((2l+1)(2l+3)²(2l+5))]
    # The off-diagonal element is ALWAYS positive for |ml| ≤ l_low (valid basis range).
    (0, 0, np.sqrt(1.0 * 4.0   / (1.0 * 9.0  * 5.0))),   # √(4/45)   ≈ 0.2981
    (1, 0, np.sqrt(4.0 * 9.0   / (3.0 * 25.0 * 7.0))),   # √(36/525) ≈ 0.2619
    (1, 1, np.sqrt(3.0 * 8.0   / (3.0 * 25.0 * 7.0))),   # √(24/525) ≈ 0.2138
    (2, 2, np.sqrt(5.0 * 12.0  / (5.0 * 49.0 * 9.0))),   # √(60/2205)≈ 0.1650
])
def test_cos2_offdiag_spot_values(l_low, ml, expected):
    got = analytic_cos2_offdiag(l_low, ml)
    assert relerr(got, expected) < 1e-10, (
        f"l_low={l_low},ml={ml}: got={got:.8e}, expected={expected:.8e}"
    )


# ── 5. Full Hamiltonian Hermiticity with quadratic Zeeman ────────────────────

@pytest.mark.parametrize("n,Z,B", [
    (2, 1,  100.0),
    (2, 6,  100.0),
    (3, 1,  500.0),
    (3, 6,  100.0),
    (4, 4, 1000.0),
])
def test_hamiltonian_hermitian_with_qz(n, Z, B):
    """H (including quadratic Zeeman) must be Hermitian."""
    H = build_hamiltonian(n, Z, B, quadratic_zeeman=True)
    diff = np.max(np.abs(H - H.conj().T))
    assert diff < 1e-12, (
        f"n={n},Z={Z},B={B}T: |H-H†|_max = {diff:.3e}"
    )


# ── 6. QZ shifts all eigenvalues upward ──────────────────────────────────────

@pytest.mark.parametrize("n,Z,B", [
    (2, 1, 100.),
    (3, 1, 500.),
    (2, 6, 500.),
])
def test_qz_shifts_eigenvalues_up(n, Z, B):
    """
    H_QZ = e²B²r²sin²θ / 8mₑ ≥ 0.
    Every eigenvalue must shift upward (or stay the same) when QZ is added.
    """
    evals_no,  _ = diagonalize_hamiltonian(n, Z, B, quadratic_zeeman=False)
    evals_yes, _ = diagonalize_hamiltonian(n, Z, B, quadratic_zeeman=True)
    # Each eigenvalue should increase (QZ is positive semi-definite)
    diffs = evals_yes - evals_no
    assert np.all(diffs >= -1e-12), (
        f"n={n},Z={Z},B={B}T: some eigenvalues shifted DOWN: min(ΔE)={diffs.min():.4e}"
    )


# ── 7. QZ magnitude scales as B² ─────────────────────────────────────────────

def test_qz_scales_as_B_squared():
    """
    The QZ contribution ΔE_QZ ∝ B².
    Compare the average eigenvalue shift (which = Tr[H_QZ]/dim) at B and 2B.
    """
    n, Z = 3, 1
    B1, B2 = 100.0, 200.0

    def avg_shift(B):
        ev_no,  _ = diagonalize_hamiltonian(n, Z, B, quadratic_zeeman=False)
        ev_yes, _ = diagonalize_hamiltonian(n, Z, B, quadratic_zeeman=True)
        return np.mean(ev_yes - ev_no)

    shift1 = avg_shift(B1)
    shift2 = avg_shift(B2)

    # Expect shift2 / shift1 ≈ (B2/B1)² = 4
    ratio = shift2 / shift1
    assert relerr(ratio, 4.0) < 0.01, (
        f"QZ scaling: shift(B={B2})/shift(B={B1}) = {ratio:.4f}, expected 4.0"
    )


# ── 8. QZ is small vs. linear Zeeman at B ≤ 500 T for C VI ──────────────────

def test_qz_small_vs_linear_zeeman_cvi():
    """
    For C VI (Z=6), the QZ contribution should remain < 10% of linear Zeeman
    at B=500 T (verifying it is a perturbative correction).

    Linear Zeeman scale: μ_B × B_max × ml_max = 5.788e-5 × 500 × 2 ≈ 0.058 eV
    QZ scale: (eB²a₀²/8mₑ) × <r²>_max × (sin²θ)_max
    """
    n, Z, B = 3, 6, 500.0

    ev_no,  _ = diagonalize_hamiltonian(n, Z, B, quadratic_zeeman=False)
    ev_yes, _ = diagonalize_hamiltonian(n, Z, B, quadratic_zeeman=True)
    max_qz_shift = np.max(ev_yes - ev_no)

    linear_scale = BOHR_MAGNETON_EV_T * B * (n - 1)  # max ml = n-1
    ratio = max_qz_shift / linear_scale

    print(f"C VI n=3, B=500T: max QZ shift={max_qz_shift:.4e} eV, "
          f"linear Zeeman scale={linear_scale:.4e} eV, ratio={ratio:.4f}")
    assert ratio < 0.10, (
        f"QZ/linear-Zeeman = {ratio:.4f} > 10% — QZ is not perturbative at B=500T, Z=6"
    )
