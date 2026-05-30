"""
test_12_halpha_qz_wings.py — Quadratic-Zeeman polarization wings on Hα.

At B ≥ 500 T, the diamagnetic (QZ) term splits the n=3 Hα σ±/π components
because different (n=3, l, ml) states receive different diagonal QZ shifts:

  3p(ml=±1): +8.87 meV at B=1000 T  ← largest in n=3
  3d(ml=±2): +6.65 meV
  …
  2p(ml=0):  +0.74 meV

The dominant transitions are 3d→2p (large radial dipole), but the 3p→2s
transitions (~21 % relative oscillator strength) are pushed to a distinct
"wing" position ≈5–7 meV beyond the 3d cluster at B=1000 T.

Tests verify:
  1. The wing position predicted analytically matches the profile peak.
  2. The wing intensity (relative to no-QZ baseline) is large and grows with B.
  3. Without QZ all σ+ components collapse to a single peak at E0 + μ_B B.
  4. π profile shifts toward higher energy with QZ (all transitions blue-shifted).
  5. The σ+ wing is absent at B=100 T (wings only emerge for B ≥ ~500 T).
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.static_profile import calculate_static_profile
from scipy.constants import e as E_CHARGE, m_e as M_E
from starkzee.utils import RYDBERG_EV, BOHR_MAGNETON_EV_T, A0


Z, n_u, n_l = 1, 3, 2
Ne, Te       = 1e23, 5.0
E0           = RYDBERG_EV * (1.0/n_l**2 - 1.0/n_u**2)   # ≈ 1.889 eV


def qz_coeff(B):
    """Quadratic Zeeman coefficient (e B² a₀²/8mₑ) in eV."""
    return E_CHARGE * B**2 * A0**2 / (8.0 * M_E)


def qz_diag(n, l, ml, B):
    """Diagonal QZ shift ⟨n,l,ml|H_QZ|n,l,ml⟩ in eV."""
    r2 = (n**2 / (2.0)) * (5*n**2 + 1 - 3*l*(l+1))   # Z=1 diagonal
    sin2 = 2.0/3.0 if l == 0 else (
        1.0 - (2.0*l**2 + 2.0*l - 1.0 - 2.0*ml**2) / ((2*l-1.0)*(2*l+3.0))
    )
    return qz_coeff(B) * r2 * sin2


def sigma_plus_profiles(B, det_rel, num_f=20, num_mu=6):
    """Compute σ+ profile (with and without QZ) on det_rel grid centered at E0+muB_B."""
    center = E0 + BOHR_MAGNETON_EV_T * B
    en     = center + det_rel
    _, sp_nq, _ = calculate_static_profile(
        n_u, n_l, Z, B, Ne, Te, en, num_f=num_f, num_mu=num_mu,
        use_screening=True, include_quadratic=False, frequency_dependent_width=False)
    _, sp_yq, _ = calculate_static_profile(
        n_u, n_l, Z, B, Ne, Te, en, num_f=num_f, num_mu=num_mu,
        use_screening=True, include_quadratic=True, frequency_dependent_width=False)
    return sp_nq, sp_yq


# ── 1. Wing predicted position ────────────────────────────────────────────────

def test_sigma_plus_wing_position_at_1000T():
    """
    At B=1000 T, the σ+ wing peak should appear at approximately
    E0 + μ_B·B + (QZ(3p,ml=1) − QZ(2s)) ≈ E0 + μ_B·B + 7.1 meV.

    We probe a ±20 meV window around the cluster center and look for
    a local maximum in the WITH-QZ profile at det > +4 meV.
    """
    B = 1000.0
    det = np.linspace(-0.020, 0.020, 800)   # ±20 meV
    sp_nq, sp_yq = sigma_plus_profiles(B, det)

    det_meV = det * 1e3

    # Expected wing position relative to σ+ cluster center
    expected_wing_meV = (qz_diag(3, 1, 1, B) - qz_diag(2, 0, 0, B)) * 1e3  # ≈ +7.1 meV

    # Find the secondary peak in the wing region (det > +4 meV)
    mask = det_meV > 4.0
    assert sp_yq[mask].max() > 0.05 * sp_yq.max(), (
        "σ+ wing peak should have ≥ 5 % of main peak intensity at B=1000 T"
    )

    wing_peak_meV = det_meV[mask][np.argmax(sp_yq[mask])]

    # Require the wing peak within ±3 meV of prediction
    err = abs(wing_peak_meV - expected_wing_meV)
    assert err < 3.0, (
        f"σ+ wing at {wing_peak_meV:.2f} meV, expected ~{expected_wing_meV:.2f} meV "
        f"(error={err:.2f} meV)"
    )


# ── 2. Wing intensity grows with B (scales as B²) ────────────────────────────

def test_sigma_plus_wing_intensity_grows_with_B():
    """
    The QZ-induced σ+ wing intensity (relative to without-QZ baseline)
    should be larger at B=1000 T than at B=500 T.
    """
    wing_ratios = {}
    for B in [500.0, 1000.0]:
        det = np.linspace(-0.015, 0.015, 600)
        det_meV = det * 1e3
        sp_nq, sp_yq = sigma_plus_profiles(B, det)

        # Wing region: det_meV > expected_separation/2
        sep = (qz_diag(3, 1, 1, B) - qz_diag(3, 2, 2, B)) * 1e3  # meV
        threshold = max(sep / 2.0, 2.0)
        mask = det_meV > threshold
        base = sp_nq[mask].sum()
        wing = sp_yq[mask].sum()
        wing_ratios[B] = wing / (base + 1e-30)

    print(f"  Wing ratio B=500T:  {wing_ratios[500.0]:.3f}")
    print(f"  Wing ratio B=1000T: {wing_ratios[1000.0]:.3f}")
    assert wing_ratios[1000.0] > wing_ratios[500.0], (
        f"Wing ratio should be larger at 1000 T: "
        f"got {wing_ratios[1000.0]:.3f} ≤ {wing_ratios[500.0]:.3f}"
    )


# ── 3. No-QZ: all σ+ components degenerate → single peak near E0 + μ_B B ────

def test_sigma_plus_noquad_single_peak():
    """
    Without QZ (and at low Ne so Stark broadening is tiny), all Hα σ+
    components are degenerate at E0 + μ_B·B.  The no-QZ profile peak
    should be within ±5 meV of that energy.
    """
    B    = 1000.0
    det  = np.linspace(-0.015, 0.015, 600)
    sp_nq, _ = sigma_plus_profiles(B, det)

    peak_meV = det[np.argmax(sp_nq)] * 1e3
    # With no QZ, peak should be very close to 0 (the cluster center IS E0+muB_B)
    assert abs(peak_meV) < 5.0, (
        f"No-QZ σ+ peak at {peak_meV:.2f} meV from cluster center "
        f"(expected ≤ 5 meV)"
    )


# ── 4. QZ shifts the π profile toward higher energy ──────────────────────────

def test_pi_profile_blueshifts_with_qz():
    """
    All QZ diagonal shifts are positive, and n=3 shifts exceed n=2 shifts.
    Hence every Hα π transition energy increases with QZ.
    The centroid of the π profile (first moment) must shift to higher energy.
    """
    B    = 1000.0
    det  = np.linspace(-0.030, 0.030, 1200)
    en   = E0 + det

    pi_nq, _, _ = calculate_static_profile(
        n_u, n_l, Z, B, Ne, Te, en, num_f=20, num_mu=6,
        use_screening=True, include_quadratic=False, frequency_dependent_width=False)
    pi_yq, _, _ = calculate_static_profile(
        n_u, n_l, Z, B, Ne, Te, en, num_f=20, num_mu=6,
        use_screening=True, include_quadratic=True, frequency_dependent_width=False)

    centroid_nq = np.sum(det * pi_nq) / (np.sum(pi_nq) + 1e-30)
    centroid_yq = np.sum(det * pi_yq) / (np.sum(pi_yq) + 1e-30)
    shift_meV   = (centroid_yq - centroid_nq) * 1e3

    print(f"  π centroid shift at B=1000T: {shift_meV:.3f} meV (should be > 0)")
    assert shift_meV > 0, (
        f"π profile centroid should shift blue with QZ, got {shift_meV:.3f} meV"
    )


# ── 5. No wing at B=100 T (QZ too small) ─────────────────────────────────────

def test_no_visible_wing_at_100T():
    """
    At B=100 T, the QZ shifts are 100× smaller than at 1000 T (scales as B²).
    The wing separation is ~0.07 meV, far smaller than the Stark width.
    The with-QZ and without-QZ σ+ profiles should be nearly identical
    (relative difference < 2 % everywhere).
    """
    B    = 100.0
    det  = np.linspace(-0.010, 0.010, 400)
    sp_nq, sp_yq = sigma_plus_profiles(B, det, num_f=15, num_mu=4)

    peak = sp_nq.max() + sp_yq.max()
    max_diff = np.max(np.abs(sp_yq - sp_nq)) / (peak / 2 + 1e-30)
    assert max_diff < 0.02, (
        f"At B=100 T QZ should be invisible: max relative diff = {max_diff:.4f} > 2 %"
    )
