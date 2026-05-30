"""
test_14_b0_profile.py — Full pipeline at B=0.

At B=0 there is no preferred quantisation axis. By spherical symmetry,
after integrating over all ion microfield orientations:
    profile_pi == profile_sig_plus == profile_sig_minus

The total profile is rotationally invariant and physically correct.

Tests:
  1. pi = sigma+ = sigma- at B=0 (symmetry restoration)
  2. All profile components non-negative at B=0
  3. Total profile has finite positive integrated intensity
  4. FFM profile runs without error and is non-negative at B=0
  5. Continuity: total profile at B=0 matches B=1e-4 T within 1%
  6. Broadening is finite (non-zero FWHM) at B=0
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.static_profile import calculate_static_profile
from starkzee.ffm import calculate_ffm_profile
from starkzee.utils import RYDBERG_EV


def transition_energy(n_u, n_l, Z):
    return (Z**2) * RYDBERG_EV * (1.0/n_l**2 - 1.0/n_u**2)


def static_profile_b0(n_u, n_l, Z, Ne, Te, detuning_range=0.05, npts=200,
                       num_f=20, num_mu=8, B=0.0):
    E0 = transition_energy(n_u, n_l, Z)
    energies = E0 + np.linspace(-detuning_range, detuning_range, npts)
    return energies, calculate_static_profile(
        n_u=n_u, n_l=n_l, Z=Z, B=B, Ne_m3=Ne, Te_ev=Te,
        energies_ev=energies, num_f=num_f, num_mu=num_mu,
        use_screening=True, include_quadratic=False, include_fine_structure=True,
        frequency_dependent_width=False
    )


# ── 1. pi = sigma+ = sigma- at B=0 ───────────────────────────────────────────

@pytest.mark.parametrize("Z,n_u,n_l,Ne,Te", [
    (1, 2, 1, 1e25, 10.0),   # H Ly-α
    (1, 3, 2, 1e25, 10.0),   # H Balmer-α
    (6, 2, 1, 5e25, 100.0),  # C VI Ly-α
])
def test_b0_pi_equals_sigma_symmetry(Z, n_u, n_l, Ne, Te):
    """At B=0 all three polarization components must be equal (spherical symmetry)."""
    _, (pi, sp, sm) = static_profile_b0(n_u, n_l, Z, Ne, Te)
    total = pi + sp + sm
    norm = np.max(total)
    assert norm > 0, "Total profile is zero"

    # Each component should be total/3 within 2% (numerical quadrature tolerance)
    for name, comp in [("pi", pi), ("sigma+", sp), ("sigma-", sm)]:
        diff = np.max(np.abs(comp - total / 3.0)) / norm
        assert diff < 0.02, (
            f"Z={Z},n={n_u}→{n_l}: {name} deviates from total/3 by {diff:.4f} (>2%)"
        )


# ── 2. Non-negative at B=0 ────────────────────────────────────────────────────

@pytest.mark.parametrize("Z,n_u,n_l,Ne,Te", [
    (1, 2, 1, 1e25, 10.0),
    (1, 3, 2, 1e24, 5.0),
    (6, 2, 1, 5e25, 100.0),
])
def test_b0_profile_nonnegative(Z, n_u, n_l, Ne, Te):
    """All profile components must be >= 0 at B=0."""
    _, (pi, sp, sm) = static_profile_b0(n_u, n_l, Z, Ne, Te)
    for name, arr in [("pi", pi), ("sigma+", sp), ("sigma-", sm)]:
        assert np.all(arr >= -1e-15), (
            f"Z={Z},n={n_u}→{n_l}: {name} has negative values, min={arr.min():.4e}"
        )


# ── 3. Finite positive integrated intensity ──────────────────────────────────

@pytest.mark.parametrize("Z,n_u,n_l,Ne,Te", [
    (1, 2, 1, 1e25, 10.0),
    (6, 2, 1, 5e25, 100.0),
])
def test_b0_profile_integrates_positive(Z, n_u, n_l, Ne, Te):
    """Total profile integrates to a finite positive value at B=0."""
    energies, (pi, sp, sm) = static_profile_b0(n_u, n_l, Z, Ne, Te,
                                                detuning_range=0.2)
    total = pi + sp + sm
    integral = np.trapezoid(total, energies)
    assert np.isfinite(integral), f"Z={Z}: integral is not finite"
    assert integral > 0, f"Z={Z}: integral = {integral:.4e} <= 0"


# ── 4. FFM profile at B=0 ─────────────────────────────────────────────────────

def test_b0_ffm_profile_runs():
    """FFM must run without error at B=0 and return a non-negative profile."""
    Z, n_u, n_l = 1, 3, 2
    Ne, Te, Ti = 1e24, 5.0, 0.1
    E0 = transition_energy(n_u, n_l, Z)
    energies = E0 + np.linspace(-0.05, 0.05, 100)

    pi, sp, sm = calculate_ffm_profile(
        n_u=n_u, n_l=n_l, Z=Z, B=0.0, Ne_m3=Ne, Te_ev=Te, Ti_ev=Ti,
        A_ion=1, energies_ev=energies, num_f=15, num_mu=6,
        include_fine_structure=True
    )
    total = pi + sp + sm
    assert np.all(pi >= -1e-15), f"FFM pi has negatives: min={pi.min():.4e}"
    assert np.all(sp >= -1e-15), f"FFM sigma+ has negatives: min={sp.min():.4e}"
    assert np.trapezoid(total, energies) > 0, "FFM total profile integral <= 0"


# ── 5. Continuity: B=0 vs B=1e-4 T ──────────────────────────────────────────

@pytest.mark.parametrize("Z,n_u,n_l,Ne,Te", [
    (1, 2, 1, 1e25, 10.0),
    (1, 3, 2, 1e25, 10.0),
])
def test_b0_continuity(Z, n_u, n_l, Ne, Te):
    """Total profile must be continuous across B=0: profile(B=0) ≈ profile(B=1e-4 T)."""
    _, (pi0, sp0, sm0) = static_profile_b0(n_u, n_l, Z, Ne, Te, B=0.0)
    _, (piB, spB, smB) = static_profile_b0(n_u, n_l, Z, Ne, Te, B=1e-4)

    total0 = pi0 + sp0 + sm0
    totalB = piB + spB + smB
    norm = np.max(total0)
    assert norm > 0

    max_diff = np.max(np.abs(total0 - totalB)) / norm
    assert max_diff < 0.01, (
        f"Z={Z},n={n_u}→{n_l}: B=0 vs B=1e-4T max deviation = {max_diff:.4f} (>1%)"
    )


# ── 6. Finite broadening at B=0 ──────────────────────────────────────────────

def test_b0_nonzero_fwhm():
    """Electron impact broadening must give a finite FWHM at B=0 (not a delta function)."""
    energies, (pi, sp, sm) = static_profile_b0(
        n_u=2, n_l=1, Z=1, Ne=1e25, Te=10.0, detuning_range=0.05, npts=300
    )
    total = pi + sp + sm
    peak = total.max()
    assert peak > 0
    above_half = energies[total >= peak / 2]
    fwhm = above_half[-1] - above_half[0] if len(above_half) > 1 else 0.0
    assert fwhm > 1e-5, f"FWHM = {fwhm:.4e} eV is suspiciously small (near zero)"
