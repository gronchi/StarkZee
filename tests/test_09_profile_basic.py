"""
test_09_profile_basic.py — Full profile integration: basic physical properties.

Tests the complete pipeline (microfield × Stark-Zeeman × broadening) for:
  1. Profile is non-negative
  2. Profile is symmetric at B~0 (π = σ)
  3. Profile integrates to a finite positive value
  4. Peak is at the correct transition energy
  5. Increasing Ne broadens the profile (Stark broadening increases)
  6. Increasing B splits the peak (Zeeman splitting visible)
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.static_profile import calculate_static_profile
from starkzee.utils import RYDBERG_EV, energy_ev_to_wavelength_nm


def make_profile(n_u, n_l, Z, B, Ne, Te, detuning_range=0.2, npts=300,
                 num_f=20, num_mu=6):
    """Helper: compute pi, sigma profiles on a detuning grid.

    Fine structure disabled so the energy grid (centered on the non-relativistic
    Bohr formula E0) matches the actual profile center. Fine structure correctness
    is tested separately in test_16_fine_structure.py.
    """
    E0 = (Z**2) * RYDBERG_EV * (1.0/n_l**2 - 1.0/n_u**2)
    det = np.linspace(-detuning_range, detuning_range, npts)
    energies = E0 + det
    pi, sp, sm = calculate_static_profile(
        n_u=n_u, n_l=n_l, Z=Z, B=B, Ne_m3=Ne, Te_ev=Te,
        energies_ev=energies, num_f=num_f, num_mu=num_mu,
        use_screening=True, include_quadratic=False, include_fine_structure=False,
        frequency_dependent_width=False
    )
    sigma = sp + sm
    return det, pi, sigma


# ── 1. Non-negative profiles ──────────────────────────────────────────────────

@pytest.mark.parametrize("Z,n_u,n_l,B,Ne,Te", [
    (1, 2, 1, 1e-3, 1e25, 10.0),   # H Ly-α, near B=0
    (6, 2, 1, 100., 5e25, 100.0),  # C VI Ly-α, Fig.3 conditions
    (4, 5, 4, 100., 2e25, 10.0),   # C IV n=5→4
])
def test_profile_nonnegative(Z, n_u, n_l, B, Ne, Te):
    """Profile values must be ≥ 0 everywhere."""
    det, pi, sigma = make_profile(n_u, n_l, Z, B, Ne, Te)
    assert np.all(pi    >= -1e-15), f"π profile has negative values: min={pi.min():.4e}"
    assert np.all(sigma >= -1e-15), f"σ profile has negative values: min={sigma.min():.4e}"


# ── 2. Symmetry at B~0 ────────────────────────────────────────────────────────

def test_pi_equals_sigma_at_zero_B():
    """
    At B=0 (using B=1e-3 T as proxy), σ+ + σ- should equal π
    (all polarizations degenerate).
    The ratio total_sigma / total_pi should be ~2 (2 sigma components vs 1 pi).
    """
    Z, n_u, n_l = 1, 2, 1
    det, pi, sigma = make_profile(n_u, n_l, Z, B=1e-3, Ne=1e22, Te=1.0,
                                   detuning_range=0.05, npts=100, num_f=10, num_mu=4)
    total_pi    = np.trapezoid(pi,    det)
    total_sigma = np.trapezoid(sigma, det)

    print(f"H Ly-α B~0: ∫π={total_pi:.4e}, ∫σ={total_sigma:.4e}, ratio={total_sigma/total_pi:.3f}")

    # σ = σ+ + σ- (two components), π = just π
    # For isotropic atom, σ+/σ- each equal π in intensity per component
    # So ratio total_sigma/total_pi ≈ 2
    ratio = total_sigma / total_pi
    assert 1.5 < ratio < 2.5, (
        f"∫σ/∫π = {ratio:.3f}, expected ~2.0 at B~0"
    )


# ── 3. Profile integrates to finite positive value ───────────────────────────

@pytest.mark.parametrize("Z,n_u,n_l,B,Ne,Te", [
    (1, 2, 1, 1e-3, 1e25, 10.0),
    (6, 2, 1, 100., 5e25, 100.0),
])
def test_profile_integrates_finite_positive(Z, n_u, n_l, B, Ne, Te):
    """∫profile dE > 0 and finite."""
    det, pi, sigma = make_profile(n_u, n_l, Z, B, Ne, Te, detuning_range=0.5)
    for name, arr in [("pi", pi), ("sigma", sigma)]:
        integral = np.trapezoid(arr, det)
        assert np.isfinite(integral) and integral > 0, (
            f"{name} profile integral = {integral:.4e}"
        )


# ── 4. Peak at correct transition energy ─────────────────────────────────────

@pytest.mark.parametrize("Z,n_u,n_l,B,Ne,Te,tol_eV", [
    (1, 2, 1,  1e-3, 1e22, 1.0,  5e-3),   # H, tiny B+Ne: very narrow, tight tol
    (6, 2, 1,  100., 5e25, 100., 0.05),   # C VI: broadened, looser tol
])
def test_sigma_peak_near_E0(Z, n_u, n_l, B, Ne, Te, tol_eV):
    """
    The peak of the σ (or combined) profile must be within tol_eV of E0.
    At B~0 and low Ne, the peak should be essentially at E0.
    """
    det, pi, sigma = make_profile(n_u, n_l, Z, B, Ne, Te,
                                   detuning_range=max(0.2, 3*tol_eV), npts=400)
    peak_detuning = det[np.argmax(sigma)]
    print(f"Z={Z},n={n_u}→{n_l}: sigma peak at detuning={peak_detuning:.4f} eV (tol={tol_eV})")
    assert abs(peak_detuning) < tol_eV, (
        f"Peak detuning {peak_detuning:.4f} eV exceeds tolerance {tol_eV} eV"
    )


# ── 5. Stark broadening increases with Ne ────────────────────────────────────

def test_stark_broadening_increases_with_Ne():
    """
    Higher Ne should produce a broader σ profile (larger Stark broadening).
    Measure FWHM by finding half-maximum points.
    """
    Z, n_u, n_l, B, Te = 1, 2, 1, 1e-3, 10.0

    def get_fwhm(Ne):
        det, _, sigma = make_profile(n_u, n_l, Z, B=B, Ne=Ne, Te=Te,
                                     detuning_range=0.1, npts=300, num_f=15, num_mu=4)
        peak = sigma.max()
        above_half = det[sigma >= peak / 2]
        return above_half[-1] - above_half[0] if len(above_half) > 1 else 0.0

    fwhm_low  = get_fwhm(1e24)
    fwhm_high = get_fwhm(1e25)

    print(f"FWHM(Ne=1e24)={fwhm_low:.4f} eV, FWHM(Ne=1e25)={fwhm_high:.4f} eV")
    assert fwhm_high > fwhm_low, (
        f"Higher Ne should broaden profile but FWHM decreased: {fwhm_high:.4f} < {fwhm_low:.4f}"
    )


# ── 6. Zeeman splitting visible at large B ────────────────────────────────────

def test_zeeman_splitting_at_large_B():
    """
    At B=500 T and low Ne (pure Zeeman), the σ+ and σ− components must be
    shifted away from E0 by approximately μ_B × B.
    """
    from starkzee.utils import BOHR_MAGNETON_EV_T
    from starkzee.static_profile import calculate_static_profile

    Z, n_u, n_l = 1, 2, 1
    B = 500.0
    Ne, Te = 1e21, 1.0   # very low density: minimal Stark
    E0 = (Z**2) * RYDBERG_EV * (1.0/n_l**2 - 1.0/n_u**2)

    det = np.linspace(-0.2, 0.2, 500)
    energies = E0 + det

    pi, sp, sm = calculate_static_profile(
        n_u=n_u, n_l=n_l, Z=Z, B=B, Ne_m3=Ne, Te_ev=Te,
        energies_ev=energies, num_f=5, num_mu=4,
        use_screening=False, include_quadratic=False, include_fine_structure=False,
        frequency_dependent_width=False
    )

    # σ+ peak should be blue-shifted, σ- red-shifted by ~μ_B × B
    peak_sp = det[np.argmax(sp)]
    peak_sm = det[np.argmax(sm)]
    expected_shift = BOHR_MAGNETON_EV_T * B  # ≈ 0.029 eV at B=500T

    print(f"B=500T: σ+ peak at {peak_sp:.4f} eV, σ- peak at {peak_sm:.4f} eV")
    print(f"Expected Zeeman shift: ±{expected_shift:.4f} eV")

    # Allow 50% tolerance due to SOC and finite Stark mixing at Ne=1e15
    assert peak_sp > 0, f"σ+ should be blue-shifted but peak at {peak_sp:.4f}"
    assert peak_sm < 0, f"σ- should be red-shifted but peak at {peak_sm:.4f}"
    assert relerr(abs(peak_sp), expected_shift) < 0.5, (
        f"σ+ shift {peak_sp:.4f} eV, expected ~{expected_shift:.4f} eV"
    )


def relerr(got, ref):
    return abs(got - ref) / abs(ref)
