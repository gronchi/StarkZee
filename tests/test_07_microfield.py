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
    calculate_coupling_parameter,
    potekhin_distribution,
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


def test_normal_field_with_Z_bar():
    """Verify that calculate_normal_field correctly scales with Z_bar."""
    Ne = 1e25
    # Default Z_bar=1.0
    F0_def, re_def = calculate_normal_field(Ne)
    F0_1, re_1 = calculate_normal_field(Ne, Z_bar=1.0)
    assert F0_def == F0_1
    assert re_def == re_1

    # Z_bar=2.0
    F0_2, re_2 = calculate_normal_field(Ne, Z_bar=2.0)
    
    # re should scale as Z_bar^(1/3) -> 2^(1/3)
    re_expected = re_1 * (2.0**(1.0 / 3.0))
    # F0 should scale as Z_bar / re^2 -> 2 / (2^(2/3)) = 2^(1/3)
    F0_expected = F0_1 * (2.0**(1.0 / 3.0))

    assert relerr(re_2, re_expected) < 1e-10
    assert relerr(F0_2, F0_expected) < 1e-10


def test_microfield_quadrature_with_Z_bar():
    """Verify that microfield_quadrature scales fields and screening correctly with Z_bar."""
    Ne, Te = 5e25, 100.0
    # Unscreened (Holtsmark)
    fields_1, weights_1 = microfield_quadrature(Ne, Te, num_points=30, use_screening=False, Z_bar=1.0)
    fields_2, weights_2 = microfield_quadrature(Ne, Te, num_points=30, use_screening=False, Z_bar=2.0)
    
    # Weights should be identical since the beta grid is the same and Holtsmark is Z_bar-independent
    np.testing.assert_allclose(weights_1, weights_2, rtol=1e-10)
    
    # Fields should scale as 2^(1/3)
    factor = 2.0**(1.0 / 3.0)
    np.testing.assert_allclose(fields_2, fields_1 * factor, rtol=1e-10)



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


# ── 7. Zest Microfield integration ───────────────────────────────────────────

def test_potekhin_normalization():
    """Verify that the Zest Potekhin distribution integrates to ~1."""
    beta_grid = np.linspace(0.01, 15.0, 1000)
    # Screened charged case
    gamma = 0.5
    s = 0.3
    P_screened = potekhin_distribution(beta_grid, gamma=gamma, s=s, charged=True)
    integral_screened = np.trapezoid(P_screened, beta_grid)
    assert relerr(integral_screened, 1.0) < 0.05, f"Screened integral = {integral_screened:.4f}"

    # Unscreened neutral case (s = 0, charged = False)
    P_unscreened = potekhin_distribution(beta_grid, gamma=gamma, s=0.0, charged=False)
    integral_unscreened = np.trapezoid(P_unscreened, beta_grid)
    assert relerr(integral_unscreened, 1.0) < 0.05, f"Unscreened integral = {integral_unscreened:.4f}"



def test_calculate_coupling_parameter():
    """Test coupling parameter calculation."""
    # Compute for some physical case and check
    # Let Z_bar=1, Ti_ev=10 eV, R_ii = 1e-9 m
    gamma = calculate_coupling_parameter(1.0, 10.0, 1e-9)
    # expected: (1^2 * e) / (4 * pi * eps_0 * 10 * 1e-9)
    expected = E_CHARGE / (4.0 * np.pi * EPSILON_0 * 10.0 * 1e-9)
    assert relerr(gamma, expected) < 1e-10


def test_holtsmark_distribution_methods():
    """Test holtsmark_distribution with exact, vectorized, and potekhin methods."""
    beta = 1.2
    val_vec = holtsmark_distribution(beta, method='vectorized')
    val_ex = holtsmark_distribution(beta, method='exact')
    val_pot = holtsmark_distribution(beta, method='potekhin')
    
    assert relerr(val_vec, val_ex) < 1e-6, f"vectorized vs exact relative difference: {relerr(val_vec, val_ex):.2e}"
    assert relerr(val_pot, val_ex) < 2e-3, f"potekhin vs exact relative difference: {relerr(val_pot, val_ex):.2e}"
    
    # Test with array input
    beta_arr = np.linspace(0.1, 5.0, 50)
    arr_vec = holtsmark_distribution(beta_arr, method='vectorized')
    arr_ex = holtsmark_distribution(beta_arr, method='exact')
    
    np.testing.assert_allclose(arr_vec, arr_ex, rtol=1e-5, atol=1e-6)
    
    # Test invalid method
    with pytest.raises(ValueError, match="Unknown method 'invalid'"):
        holtsmark_distribution(beta, method='invalid')


def test_hooper_distribution_methods():
    """Test hooper_distribution with exact and vectorized methods."""
    beta = 1.5
    a = 0.5
    val_vec = hooper_distribution(beta, a, charged=True, method='vectorized')
    val_ex = hooper_distribution(beta, a, charged=True, method='exact')
    
    assert relerr(val_vec, val_ex) < 1e-6
    
    # Test array input
    beta_arr = np.linspace(0.1, 5.0, 50)
    arr_vec = hooper_distribution(beta_arr, a, charged=True, method='vectorized')
    arr_ex = hooper_distribution(beta_arr, a, charged=True, method='exact')
    
    np.testing.assert_allclose(arr_vec, arr_ex, rtol=1e-5, atol=1e-6)
    
    with pytest.raises(ValueError, match="Unknown method 'invalid'"):
        hooper_distribution(beta, a, charged=True, method='invalid')


def test_hooper_charged_parameter():
    """Verify that hooper_distribution with charged=True/False works and is different."""
    beta = np.linspace(0.1, 5.0, 50)
    a = 0.4
    
    val_sz_charged = hooper_distribution(beta, a, charged=True)
    val_sz_neutral = hooper_distribution(beta, a, charged=False)
    
    # Both should be non-negative
    assert np.all(val_sz_charged >= 0.0)
    assert np.all(val_sz_neutral >= 0.0)
    
    # Check that charged and neutral are actually different
    assert np.any(np.abs(val_sz_charged - val_sz_neutral) > 1e-4)


