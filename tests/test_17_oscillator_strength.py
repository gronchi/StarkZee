"""
test_17_oscillator_strength.py — Oscillator strengths, Einstein A coefficients,
and transition wavelengths against NIST data.

Tests:
  1. Line strength S_ul > 0 for all allowed hydrogenic transitions
  2. gf against NIST for Lyman, Balmer, and Paschen series (within 0.5%)
  3. gf is independent of Z for hydrogen-like ions (H vs C VI)
  4. Einstein A for Balmer and Paschen lines against NIST (within 1%)
  5. Einstein A for Lyman lines rescaled to NIST g convention (within 1%)
  6. Self-consistency: gf ↔ S_ul via the oscillator-strength formula
  7. Self-consistency: einstein_a ↔ gf via the quantum-mechanical relation
  8. Profile integral at low density ≈ S_ul (within 5%)
  9. Transition wavelengths against NIST vacuum values (within 0.1%)
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.atomic_hamiltonian import (
    line_strength, oscillator_strength, einstein_a
)
from scipy.constants import fine_structure as FINE_STRUCTURE, hbar as HBAR, e as E_CHARGE
from starkzee.utils import RYDBERG_EV, energy_ev_to_wavelength_nm, reduced_mass_rydberg_ev


def relerr(got, ref):
    return abs(got - ref) / abs(ref)


E_HARTREE = 2.0 * RYDBERG_EV       # 27.2114 eV
HBAR_EV_S = HBAR / E_CHARGE        # ħ in eV·s
TAU_AU    = HBAR_EV_S / E_HARTREE  # atomic unit of time ≈ 2.419e-17 s


# Reference values from Wiese & Fuhr (2009) NIST Atomic Transition Probabilities
# (J. Phys. Chem. Ref. Data 38, 565).
#
#  gf = g_l × f_lu  (g_l = 2 n_l²)
#  A_NIST: spontaneous emission rate at the NIST g_k convention.
#  g_NIST: statistical weight NIST uses for the upper level.
#
#  For Balmer and Paschen lines g_NIST = 2 n_u² (full shell), so A can be
#  compared directly to einstein_a().
#  For Lyman lines NIST uses g_k = 6 (2p sublevel only), requiring rescaling.
#
REFERENCE = {
    # (n_u, n_l) : (gf,      A_NIST,    g_NIST)
    # Lyman series (→ n=1)
    (2, 1):  (0.8325,  6.265e8,   6),   # Ly-α
    (3, 1):  (0.1582,  1.672e8,   6),   # Ly-β
    # Balmer series (→ n=2)
    (3, 2):  (5.1263,  4.410e7,  18),   # Hα
    (4, 2):  (0.9521,  8.419e6,  32),   # Hβ
    (5, 2):  (0.3574,  2.530e6,  50),   # Hγ
    (6, 2):  (0.1771,  9.732e5,  72),   # Hδ
    # Paschen series (→ n=3)
    (4, 3):  (15.16,   8.986e6,  32),   # Pα
    (5, 3):  (2.711,   2.201e6,  50),   # Pβ
    (6, 3):  (1.004,   7.783e5,  72),   # Pγ
}


# NIST vacuum wavelengths used in test 9.
#
# The code uses the infinite-nuclear-mass Rydberg (Ry_∞), which underestimates
# H wavelengths by ~0.05% relative to NIST observed values (reduced-mass effect:
# Ry_H / Ry_∞ ≈ 1 − m_e/m_p ≈ 0.99945). For C VI the reduced-mass shift is
# ~0.005% but the relativistic correction adds ~0.04%. Tolerance is set to 0.1%
# to cover both effects while still catching formula or unit-conversion errors.
#
# Sources:
#   H Ly-α: NIST ASD vacuum measurement, 121.5670 nm.
#   H Balmer/Paschen: NIST ASD air wavelengths converted to vacuum via Edlén (1966).
#   C VI Ly-α: NIST ASD, 3.3736 nm.
#
WAVELENGTH_NIST_NM = {
    # (n_u, n_l, Z): lambda_vacuum_nm
    (2, 1, 1): 121.567,   # H Ly-α
    (3, 2, 1): 656.461,   # H Hα  (NIST air 656.279 → vacuum)
    (4, 2, 1): 486.269,   # H Hβ  (NIST air 486.133 → vacuum)
    (5, 2, 1): 434.169,   # H Hγ  (NIST air 434.047 → vacuum)
    (4, 3, 1): 1875.60,   # H Pα  (NIST air 1875.091 → vacuum)
    (5, 3, 1): 1281.97,   # H Pβ  (NIST air 1281.807 → vacuum)
    (2, 1, 6): 3.3736,    # C VI Ly-α (NIST ASD vacuum)
}


# ── 1. Line strength positive for allowed transitions ────────────────────────

@pytest.mark.parametrize("n_u,n_l", [
    (2,1),(3,1),(3,2),(4,2),(5,2),(6,2),(4,3),(5,3),(6,3)
])
def test_line_strength_positive(n_u, n_l):
    """S_ul must be strictly positive for any allowed transition."""
    S = line_strength(n_u, n_l, Z=1)
    assert S > 0, f"n={n_u}→{n_l}: S_ul = {S}"


# ── 2. gf against NIST tabulated values ─────────────────────────────────────

@pytest.mark.parametrize("n_u,n_l", [
    (2,1),(3,1),(3,2),(4,2),(5,2),(6,2),(4,3),(5,3),(6,3)
])
def test_gf_vs_nist(n_u, n_l):
    """gf must agree with NIST to within 0.5%."""
    gf_ref, _, _ = REFERENCE[(n_u, n_l)]
    gf_got = oscillator_strength(n_u, n_l, Z=1)
    assert relerr(gf_got, gf_ref) < 0.005, (
        f"n={n_u}→{n_l}: gf={gf_got:.5f}, NIST={gf_ref:.5f}, "
        f"err={relerr(gf_got, gf_ref):.4f}"
    )


# ── 3. gf is Z-independent for hydrogenic ions ───────────────────────────────

@pytest.mark.parametrize("n_u,n_l", [(2,1),(3,2),(4,2),(5,2),(6,2),(4,3)])
def test_gf_z_independent(n_u, n_l):
    """gf(Z=1) must equal gf(Z=6) within 0.5% (Z cancels analytically)."""
    gf_H   = oscillator_strength(n_u, n_l, Z=1)
    gf_CVI = oscillator_strength(n_u, n_l, Z=6)
    assert relerr(gf_CVI, gf_H) < 0.005, (
        f"n={n_u}→{n_l}: gf(Z=1)={gf_H:.5f}, gf(Z=6)={gf_CVI:.5f}"
    )


# ── 4. Einstein A for Balmer and Paschen lines against NIST ─────────────────
# For these series g_NIST = 2 n_u² = g_code, so A is directly comparable.

@pytest.mark.parametrize("n_u,n_l", [
    (3,2),(4,2),(5,2),(6,2),(4,3),(5,3),(6,3)
])
def test_einstein_a_vs_nist_series(n_u, n_l):
    """A_ul must agree with NIST to within 1% for Balmer and Paschen lines."""
    _, A_ref, g_nist = REFERENCE[(n_u, n_l)]
    A_got = einstein_a(n_u, n_l, Z=1)
    g_code = 2 * n_u**2
    assert g_nist == g_code, (
        f"n={n_u}→{n_l}: g_NIST={g_nist} ≠ g_code={g_code}"
    )
    assert relerr(A_got, A_ref) < 0.01, (
        f"n={n_u}→{n_l}: A={A_got:.4e} s⁻¹, NIST={A_ref:.4e} s⁻¹, "
        f"err={relerr(A_got, A_ref):.4f}"
    )


# ── 5. Einstein A for Lyman lines: rescale to NIST g convention ──────────────
# NIST uses g_k = 6 (2p sublevel), our code uses g_u = 2n_u².

@pytest.mark.parametrize("n_u,n_l", [(2,1),(3,1)])
def test_einstein_a_vs_nist_lyman(n_u, n_l):
    """A_ul rescaled to NIST g_k must agree within 1% for Lyman lines."""
    _, A_ref, g_nist = REFERENCE[(n_u, n_l)]
    A_got = einstein_a(n_u, n_l, Z=1)
    g_code = 2 * n_u**2
    A_rescaled = A_got * g_code / g_nist
    assert relerr(A_rescaled, A_ref) < 0.01, (
        f"n={n_u}→{n_l}: A_rescaled={A_rescaled:.4e} s⁻¹, NIST={A_ref:.4e} s⁻¹"
    )


# ── 6. S_ul self-consistent with gf via the oscillator-strength formula ──────

@pytest.mark.parametrize("n_u,n_l,Z", [
    (2,1,1),(3,1,1),(3,2,1),(4,2,1),(5,2,1),(6,2,1),
    (4,3,1),(5,3,1),(6,3,1),(2,1,6),(3,2,6)
])
def test_sline_consistent_with_gf(n_u, n_l, Z):
    """gf = (2/3) × (ΔE/E_hartree) × S_ul must hold to machine precision."""
    S  = line_strength(n_u, n_l, Z)
    gf = oscillator_strength(n_u, n_l, Z)
    delta_E_hartree = (Z**2) * RYDBERG_EV * (1.0/n_l**2 - 1.0/n_u**2) / E_HARTREE
    gf_from_S = (2.0/3.0) * delta_E_hartree * S
    assert relerr(gf, gf_from_S) < 1e-12, (
        f"n={n_u}→{n_l},Z={Z}: gf={gf:.8e}, gf_from_S={gf_from_S:.8e}"
    )


# ── 7. einstein_a self-consistent with gf via quantum mechanics ──────────────

@pytest.mark.parametrize("n_u,n_l,Z", [
    (2,1,1),(3,2,1),(4,2,1),(5,2,1),(6,2,1),
    (4,3,1),(5,3,1),(6,3,1),(3,2,6)
])
def test_einstein_a_consistent_with_gf(n_u, n_l, Z):
    """A_ul = α³ × (ΔE/E_h)² × gf / n_u² / τ_au to machine precision."""
    gf       = oscillator_strength(n_u, n_l, Z)
    A_direct = einstein_a(n_u, n_l, Z)
    delta_E_hartree = (Z**2) * RYDBERG_EV * (1.0/n_l**2 - 1.0/n_u**2) / E_HARTREE
    A_from_gf = FINE_STRUCTURE**3 * delta_E_hartree**2 * gf / n_u**2 / TAU_AU
    assert relerr(A_direct, A_from_gf) < 1e-12, (
        f"n={n_u}→{n_l},Z={Z}: A={A_direct:.8e}, A_from_gf={A_from_gf:.8e}"
    )


# ── 8. Profile integral ≈ S_ul at low density ────────────────────────────────

@pytest.mark.parametrize("n_u,n_l,Z", [(2,1,1),(3,2,1)])
def test_profile_integral_vs_line_strength(n_u, n_l, Z):
    """∫(π+σ+σ-)dE ≈ S_ul at B≈0 and very low Ne (within 5%)."""
    from starkzee.static_profile import calculate_static_profile
    Ne, Te = 1e20, 1.0
    E0 = (Z**2) * reduced_mass_rydberg_ev(Z, 1) * (1.0/n_l**2 - 1.0/n_u**2)
    energies = E0 + np.linspace(-0.005, 0.005, 3000)
    pi, sp, sm = calculate_static_profile(
        n_u=n_u, n_l=n_l, Z=Z, B=1e-6, Ne_m3=Ne, Te_ev=Te,
        energies_ev=energies, num_f=20, num_mu=8,
        include_fine_structure=False, include_quadratic=False,
        frequency_dependent_width=False
    )
    integral = np.trapezoid(pi + sp + sm, energies)
    S = line_strength(n_u, n_l, Z)
    assert relerr(integral, S) < 0.05, (
        f"n={n_u}→{n_l},Z={Z}: ∫profile={integral:.4e}, S_ul={S:.4e}, "
        f"err={relerr(integral, S):.4f}"
    )


# ── 9. Transition wavelengths against NIST vacuum values ────────────────────

@pytest.mark.parametrize("n_u,n_l,Z", [
    (2,1,1),(3,2,1),(4,2,1),(5,2,1),(4,3,1),(5,3,1),(2,1,6)
])
def test_transition_wavelength_vs_nist(n_u, n_l, Z):
    """
    Transition wavelength from the Bohr formula must agree with NIST vacuum
    wavelengths within 0.1%.

    The code uses the infinite-nuclear-mass Rydberg (Ry_∞). NIST observed
    wavelengths include the reduced-mass correction (Ry_H/Ry_∞ − 1 ≈ −5×10⁻⁴
    for H) and relativistic/QED effects (~4×10⁻⁵ for H, ~4×10⁻⁴ for C VI).
    The 0.1% tolerance covers these known systematic offsets.
    """
    E0       = (Z**2) * RYDBERG_EV * (1.0/n_l**2 - 1.0/n_u**2)
    lam_code = energy_ev_to_wavelength_nm(E0)
    lam_nist = WAVELENGTH_NIST_NM[(n_u, n_l, Z)]
    err = relerr(lam_code, lam_nist)
    assert err < 1e-3, (
        f"n={n_u}→{n_l}, Z={Z}: λ_code={lam_code:.4f} nm, "
        f"λ_NIST={lam_nist:.4f} nm, err={err:.2e}"
    )
