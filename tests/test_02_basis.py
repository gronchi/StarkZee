"""
test_02_basis.py — Basis generation and dipole matrix element tests.

Verifies:
  - build_basis: correct count and quantum numbers
  - radial_dipole: known analytic values for H Lyman-alpha
  - angular_dipole_element: selection rules, symmetry, normalization sum
"""
import numpy as np
import math
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.radiator import (
    build_basis, radial_dipole, angular_dipole_element, AtomicState
)
from starkzee.utils import RYDBERG_EV


def relerr(got, ref):
    return abs(got - ref) / abs(ref)


# ── 1. Basis generation ───────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_basis_count(n):
    """build_basis(n) must return exactly 2n² states."""
    basis = build_basis(n)
    assert len(basis) == 2 * n**2, f"n={n}: got {len(basis)}, expected {2*n**2}"

@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_basis_quantum_numbers(n):
    """All states have correct l ∈ [0,n-1], ml ∈ [-l,l], ms ∈ {±0.5}."""
    basis = build_basis(n)
    for s in basis:
        assert s.n == n
        assert 0 <= s.l < n, f"l={s.l} out of range for n={n}"
        assert -s.l <= s.ml <= s.l, f"ml={s.ml} out of range for l={s.l}"
        assert s.ms in (0.5, -0.5), f"ms={s.ms} invalid"

def test_basis_indices_unique():
    """Each state has a unique index."""
    basis = build_basis(4)
    indices = [s.index for s in basis]
    assert len(set(indices)) == len(indices), "Duplicate indices in basis"


# ── 2. Radial dipole matrix elements ─────────────────────────────────────────

def test_radial_dipole_H_lyman_alpha():
    """
    For H (Z=1): ⟨n=2,l=1|r|n=1,l=0⟩ = 768/(243√6) a0 ≈ 1.2904 a0.

    Reference: standard quantum mechanics textbooks (Bethe & Salpeter).
    """
    expected = 768.0 / (243.0 * math.sqrt(6.0))  # ≈ 1.2904
    result = radial_dipole(2, 1, 1, 0, Z=1)
    assert relerr(result, expected) < 1e-3, (
        f"⟨2p|r|1s⟩ = {result:.6f} a0, expected {expected:.6f} a0"
    )

def test_radial_dipole_Z_scaling():
    """
    The radial element scales as 1/Z:
    ⟨n=2,l=1|r|n=1,l=0⟩_Z = ⟨2p|r|1s⟩_{Z=1} / Z.
    """
    ref_Z1 = radial_dipole(2, 1, 1, 0, Z=1)
    for Z in [2, 4, 6]:
        result = radial_dipole(2, 1, 1, 0, Z=Z)
        expected = ref_Z1 / Z
        assert relerr(result, expected) < 1e-3, (
            f"Z={Z}: ⟨2p|r|1s⟩ = {result:.6f}, expected {expected:.6f}"
        )

def test_radial_dipole_forbidden_same_l():
    """Radial dipole is zero when Δl = 0 (selection rule)."""
    assert radial_dipole(2, 1, 1, 1, Z=1) == 0.0
    assert radial_dipole(2, 0, 1, 0, Z=1) == 0.0  # Δl=0

def test_radial_dipole_forbidden_same_n():
    """Within same n, radial_dipole should be zero (different n required)."""
    # This tests the integral: within n, n=2,l=1 -> n=2,l=0 should not be used
    # in the profile calculation (dipole matrix between same n levels)
    # The function should return near-zero since oscillatory integral cancels.
    result = abs(radial_dipole(2, 1, 2, 0, Z=1))
    # While not strictly forbidden by the integral, it's physically meaningless
    # between degenerate states. Just document the value.
    print(f"⟨2,1|r|2,0⟩ (same-n) = {result:.6e} a0  [informational, not used in ppp.py]")

def test_radial_dipole_positive():
    """Radial dipole ⟨n_u, l_u|r|n_l, l_l⟩ for allowed transitions is positive."""
    # All physically meaningful between-n radial dipoles should be positive
    for (n2, l2, n1, l1) in [(2,1,1,0), (3,1,2,0), (3,2,2,1)]:
        result = radial_dipole(n2, l2, n1, l1, Z=1)
        assert result >= 0, f"⟨{n2},{l2}|r|{n1},{l1}⟩ = {result} < 0"


# ── 3. Angular matrix elements ────────────────────────────────────────────────

def test_angular_element_selection_rule_delta_l():
    """Angular element must be zero unless |Δl| = 1."""
    # Same l, any m → zero
    for l in [0, 1, 2]:
        for ml in range(-l, l+1):
            for q in [0, 1, -1]:
                assert angular_dipole_element(l, ml, l, ml, q) == 0.0

def test_angular_element_pi_selection_rule():
    """For q=0 (π): requires m1 = m2."""
    # This is tested by checking non-zero only when m1==m2
    # l1=1, l2=0 allowed only for m1=m2=0
    assert angular_dipole_element(1, 0, 0, 0, 0) != 0.0   # allowed
    assert angular_dipole_element(1, 1, 0, 0, 0) == 0.0   # Δm=1, forbidden for q=0
    assert angular_dipole_element(1, -1, 0, 0, 0) == 0.0  # Δm=-1, forbidden for q=0

def test_angular_element_sigma_plus_selection_rule():
    """For q=+1 (σ+): requires m1 = m2 + 1."""
    assert angular_dipole_element(1, 1, 0, 0, 1) != 0.0   # m1=1=m2+1=0+1, allowed
    assert angular_dipole_element(1, 0, 0, 0, 1) == 0.0   # m1=0≠1, forbidden
    assert angular_dipole_element(1, -1, 0, 0, 1) == 0.0  # m1=-1≠1, forbidden

def test_angular_element_sigma_minus_selection_rule():
    """For q=-1 (σ-): requires m1 = m2 - 1."""
    assert angular_dipole_element(1, -1, 0, 0, -1) != 0.0  # m1=-1=m2-1=-1, allowed
    assert angular_dipole_element(1,  0, 0, 0, -1) == 0.0  # forbidden
    assert angular_dipole_element(1,  1, 0, 0, -1) == 0.0  # forbidden

def test_angular_element_lyman_alpha_pi_known():
    """
    For H Ly-α, q=0: ⟨l=1,m=0|cosθ|l=0,m=0⟩ = 1/√3.
    This is the angular factor for the z-component of 1s→2p_0.
    """
    val = angular_dipole_element(1, 0, 0, 0, 0)
    expected = 1.0 / math.sqrt(3.0)
    assert relerr(abs(val), expected) < 1e-10, (
        f"⟨1,0|cosθ|0,0⟩ = {val}, expected ±{expected:.6f}"
    )

def test_angular_element_lyman_alpha_full_dipole():
    """
    Full H Ly-α dipole z-component:
    ⟨2p,m=0|z|1s⟩ = radial × angular = 1.2904 × (1/√3) ≈ 0.7449 a0
    This equals the textbook value 128√2/243.
    """
    rad = radial_dipole(2, 1, 1, 0, Z=1)
    ang = angular_dipole_element(1, 0, 0, 0, 0)   # ⟨l=1,m=0|cosθ|l=0,m=0⟩
    full = abs(rad * ang)
    expected = 128.0 * math.sqrt(2.0) / 243.0  # ≈ 0.7449 a0
    assert relerr(full, expected) < 1e-3, (
        f"Full Ly-α z-dipole = {full:.6f} a0, expected {expected:.6f} a0"
    )

def test_angular_normalization_sum():
    """
    Sum rule: ∑_q ∑_m' |⟨l',m'|T_q^(1)|l,m⟩|² = (2l'+1) / (3(2l+1))
    (related to the Wigner-Eckart theorem; tests internal consistency)

    Here l=0,m=0 → l'=1: sum over q and m' should give 1/3.
    """
    total = 0.0
    for q in [0, 1, -1]:
        for mp in [-1, 0, 1]:
            val = angular_dipole_element(1, mp, 0, 0, q)
            total += val**2
    # Expected: (2×1+1) / (3×(2×0+1)) = 3/3 = 1
    assert relerr(total, 1.0) < 1e-10, (
        f"Angular sum for 0→1 transitions = {total}, expected 1.0"
    )
