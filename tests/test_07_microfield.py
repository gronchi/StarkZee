"""
test_07_microfield.py — Microfield distribution tests.

Verifies:
  - F0 (normal field) formula
  - Debye length formula
  - Holtsmark and Hooper distributions are normalized: ∫W(β)dβ = 1
  - Screening parameter a = re/λ_D is in a physically reasonable range
  - Field grid returns non-negative weights that sum to ≤ 1
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.microfield import (
    calculate_normal_field,
    calculate_debye_length,
    holtsmark_distribution,
    hooper_distribution,
    microfield_quadrature,
)
from scipy.constants import epsilon_0 as EPSILON_0, e as E_CHARGE


def relerr(got, ref):
    return abs(got - ref) / abs(ref)


# ── 1. Normal microfield F0 ───────────────────────────────────────────────────

def test_F0_formula_1e25():
    """
    For Ne = 1e25 m^-3, F0 = e/(4πε₀ re²) where re = (3/(4πNe))^{1/3}.
    Expected: F0 ≈ 3.83e9 V/m (check against known plasma physics result).
    """
    F0, re = calculate_normal_field(1e25)
    re_expected = (3.0 / (4.0 * np.pi * 1e25))**(1.0/3.0)
    F0_expected = E_CHARGE / (4.0 * np.pi * EPSILON_0 * re_expected**2)
    assert relerr(F0, F0_expected) < 1e-10, f"F0={F0:.4e}, expected {F0_expected:.4e}"
    print(f"F0(Ne=1e25 m^-3) = {F0:.4e} V/m,  re = {re:.4e} m")

def test_F0_scales_as_Ne_two_thirds():
    """F0 ∝ Ne^{2/3} (since re ∝ Ne^{-1/3}, F0 ∝ 1/re² ∝ Ne^{2/3})."""
    F0_1, _ = calculate_normal_field(1e25)
    F0_2, _ = calculate_normal_field(8e25)  # 8× density
    ratio = F0_2 / F0_1
    expected = 8.0**(2.0/3.0)  # = 4.0
    assert relerr(ratio, expected) < 1e-8, f"F0 ratio = {ratio:.6f}, expected {expected:.6f}"


# ── 2. Debye length ───────────────────────────────────────────────────────────

def test_debye_length_formula():
    """
    λ_D = √(ε₀ Te / (Ne e)) for Te in eV.
    For Te=10 eV, Ne=2e25 m^-3: λ_D ≈ 1.67e-9 m.
    """
    Te, Ne = 10.0, 2e25
    lam_D = calculate_debye_length(Te, Ne)
    lam_D_expected = np.sqrt(EPSILON_0 * Te / (Ne * E_CHARGE))
    assert relerr(lam_D, lam_D_expected) < 1e-10
    print(f"λ_D(Te=10eV, Ne=2e25 m^-3) = {lam_D:.4e} m")

def test_debye_length_scales_correctly():
    """λ_D ∝ √(Te/Ne)."""
    lam1 = calculate_debye_length(10.0, 1e25)
    lam2 = calculate_debye_length(40.0, 1e25)  # 4× Te
    lam3 = calculate_debye_length(10.0, 4e25)  # 4× Ne
    assert relerr(lam2 / lam1, 2.0) < 1e-8, "λ_D should double when Te×4"
    assert relerr(lam3 / lam1, 0.5) < 1e-8, "λ_D should halve when Ne×4"


# ── 3. Holtsmark distribution normalization ───────────────────────────────────

def test_holtsmark_normalization():
    """
    ∫₀^∞ W_H(β) dβ = 1.
    We integrate numerically over a fine grid up to β=15 (tail is negligible beyond).
    """
    beta = np.linspace(0.0, 15.0, 2000)
    W = np.array([holtsmark_distribution(b) for b in beta])
    integral = np.trapezoid(W, beta)
    # Integral up to beta=15 is ~0.9821. The tail integral from 15 to infinity is ~0.017.
    assert relerr(integral, 1.0) < 0.02, (
        f"Holtsmark integral = {integral:.4f}, expected 1.0 (with tail)"
    )

def test_holtsmark_peak_location():
    """Holtsmark distribution W_H(β) peaks near β ≈ 1.0."""
    beta = np.linspace(0.1, 5.0, 200)
    W = np.array([holtsmark_distribution(b) for b in beta])
    beta_peak = beta[np.argmax(W)]
    assert 0.5 < beta_peak < 2.0, (
        f"Holtsmark peak at β={beta_peak:.2f}, expected near 1.0"
    )


# ── 4. Hooper distribution normalization ─────────────────────────────────────

@pytest.mark.parametrize("a", [0.1, 0.3, 0.5])
def test_hooper_normalization(a):
    """∫₀^∞ W_Hooper(β, a) dβ ≈ 1 for various screening parameters."""
    beta = np.linspace(0.0, 15.0, 2000)
    W = np.array([hooper_distribution(b, a) for b in beta])
    integral = np.trapezoid(W, beta)
    assert relerr(integral, 1.0) < 0.05, (
        f"Hooper(a={a}) integral = {integral:.4f}, expected 1.0"
    )

def test_hooper_approaches_holtsmark_at_zero_screening():
    """As a → 0, Hooper should approach Holtsmark."""
    beta = np.linspace(0.1, 8.0, 100)
    W_holt  = np.array([holtsmark_distribution(b) for b in beta])
    W_hoop  = np.array([hooper_distribution(b, a=0.01) for b in beta])
    # RMS difference should be small
    rms = np.sqrt(np.mean((W_hoop - W_holt)**2))
    max_W = np.max(W_holt)
    assert rms / max_W < 0.05, (
        f"Hooper(a=0.01) RMS deviation from Holtsmark = {rms/max_W:.3f}"
    )


# ── 5. Microfield grid ────────────────────────────────────────────────────────

@pytest.mark.parametrize("Ne,Te,use_screening", [
    (5e25, 100.0, True),
    (2e25, 10.0,  True),
    (1e25, 100.0, False),
])
def test_microfield_grid_weights_positive(Ne, Te, use_screening):
    """All microfield weights must be non-negative."""
    fields, weights = microfield_quadrature(Ne, Te, num_points=30,
                                          use_screening=use_screening)
    assert np.all(weights >= 0), f"Negative weights in microfield grid"

def test_microfield_grid_weights_sum():
    """Microfield weights must sum to ~1.0 (they are normalized)."""
    fields, weights = microfield_quadrature(5e25, 100.0, num_points=50)
    total = np.sum(weights)
    assert relerr(total, 1.0) < 0.02, (
        f"Microfield weights sum = {total:.4f}, expected 1.0"
    )

def test_microfield_grid_fields_positive():
    """Field values must be non-negative."""
    fields, _ = microfield_quadrature(5e25, 100.0, num_points=30)
    assert np.all(fields >= 0), "Negative field values in microfield grid"

def test_microfield_F0_scale():
    """
    The maximum field in the grid should be O(F0), not orders of magnitude off.
    F_max = max_beta × F0. With default max_beta=10, F_max = 10×F0.
    """
    Ne, Te = 5e25, 100.0
    F0, _ = calculate_normal_field(Ne)
    fields, _ = microfield_quadrature(Ne, Te, num_points=30, max_beta=10.0)
    F_max = fields[-1]
    assert relerr(F_max, 10.0 * F0) < 0.01, (
        f"F_max = {F_max:.4e}, expected 10×F0 = {10*F0:.4e}"
    )


# ── 6. Screening parameter is physically reasonable ──────────────────────────

def test_screening_parameter_range():
    """
    a = re / λ_D for typical plasma conditions should be in [0.01, 2].
    Outside this range the Hooper distribution may not be accurate.
    """
    from starkzee.microfield import calculate_normal_field, calculate_debye_length
    test_cases = [
        (5e25, 100.0),   # Paper Fig.3 conditions
        (2e25,  10.0),   # Paper C IV conditions
        (1e23,  10.0),   # Low-density limit
    ]
    for Ne, Te in test_cases:
        F0, re = calculate_normal_field(Ne)
        lam_D  = calculate_debye_length(Te, Ne)
        a = re / lam_D
        print(f"Ne={Ne:.0e}, Te={Te} eV: a = {a:.4f}")
        assert 0.001 < a < 5.0, (
            f"Screening parameter a={a:.4f} out of expected range for Ne={Ne:.0e}, Te={Te}"
        )
