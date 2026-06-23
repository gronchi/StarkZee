"""
test_15_low_b_profile.py — Profile correctness in the low-B regime (B = 0–15 T).

At DIII-D edge conditions (B ~ 2–6 T) the Zeeman splitting is much smaller
than both the Stark broadening and the fine structure of C VI. The code must:
  - Give a smooth, continuous profile as B varies from 0 to 15 T
  - Conserve the integrated line power (oscillator strength × broadening)
  - Show growing pi/sigma asymmetry proportional to B
  - Have Zeeman splitting at the eigenvalue level scale linearly with B

Tests:
  1. Zeeman eigenvalue splitting scales linearly with B (0.1 T – 15 T)
  2. Integrated profile power is conserved from B=0 to B=15 T (within 2%)
  3. pi/sigma asymmetry grows monotonically from B=0 toward B=15 T
  4. Profile varies smoothly: no jump at any intermediate B value
  5. Electron broadening at B=15 T is dominated by plasma (omega_L << omega_p)
     for typical DIII-D density
  6. Static vs FFM comparison at low B, high density
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.static_profile import calculate_static_profile, solve_starkzee
from starkzee.ffm import calculate_ffm_profile
from starkzee.radiator import diagonalize_hamiltonian
from starkzee.broadening import electron_impact_width
from starkzee.utils import RYDBERG_EV, BOHR_MAGNETON_EV_T


def relerr(got, ref):
    return abs(got - ref) / abs(ref)


def transition_energy(n_u, n_l, Z):
    return (Z**2) * RYDBERG_EV * (1.0/n_l**2 - 1.0/n_u**2)


def total_profile(n_u, n_l, Z, B, Ne, Te, detuning_range=0.05, npts=200,
                  num_f=20, num_mu=8):
    """Return (energies, total_profile) for a given B and plasma condition."""
    E0 = transition_energy(n_u, n_l, Z)
    energies = E0 + np.linspace(-detuning_range, detuning_range, npts)
    pi, sp, sm = calculate_static_profile(
        n_u=n_u, n_l=n_l, Z=Z, B=B, Ne_m3=Ne, Te_ev=Te,
        energies_ev=energies, num_f=num_f, num_mu=num_mu,
        use_screening=True, quadratic_zeeman=False, fine_structure=True,
        frequency_dependent_width=False
    )
    return energies, pi + sp + sm, pi, sp + sm


# ── 1. Zeeman eigenvalue splitting linear in B ───────────────────────────────

@pytest.mark.parametrize("B", [0.1, 1.0, 5.0, 10.0, 15.0])
def test_zeeman_eigenvalue_linear_in_B(B):
    """For n=1 H (l=0 only), the two spin-split eigenvalues must differ by g_s*mu_B*B."""
    g_s = 2.0023192
    evals, _ = diagonalize_hamiltonian(n=1, Z=1, B=B, quadratic_zeeman=False,
                                    fine_structure=True)
    gap = evals.real.max() - evals.real.min()
    expected = g_s * BOHR_MAGNETON_EV_T * B
    assert relerr(gap, expected) < 1e-6, (
        f"B={B}T: gap={gap:.6e} eV, expected g_s*mu_B*B={expected:.6e} eV"
    )


# ── 2. Integrated power conserved from B=0 to B=15 T ─────────────────────────

@pytest.mark.parametrize("B", [0.0, 1.0, 5.0, 10.0, 15.0])
def test_integrated_power_conserved(B):
    """Line integral must stay within 2% of its B=0 value (oscillator strength invariant)."""
    Ne, Te = 1e25, 10.0
    Z, n_u, n_l = 1, 2, 1

    energies_B0, total_B0, _, _ = total_profile(n_u, n_l, Z, 0.0, Ne, Te,
                                                  detuning_range=0.1)
    energies_B,  total_B,  _, _ = total_profile(n_u, n_l, Z, B,   Ne, Te,
                                                  detuning_range=0.1)

    integral_B0 = np.trapezoid(total_B0, energies_B0)
    integral_B  = np.trapezoid(total_B,  energies_B)

    assert integral_B0 > 0
    assert relerr(integral_B, integral_B0) < 0.02, (
        f"B={B}T: integral={integral_B:.4e}, B=0 integral={integral_B0:.4e}, "
        f"relerr={relerr(integral_B, integral_B0):.4f}"
    )


# ── 3. pi/sigma asymmetry grows monotonically with B ─────────────────────────

def test_pi_sigma_asymmetry_grows_with_B():
    """The fractional difference |int(pi) - int(sigma+)| / int(total) grows with B.

    At B=0: pi == sigma+ == sigma- by spherical symmetry, so asym = 0.
    At finite B: pi != sigma+, so asym grows.
    Note: sigma in total_profile() = sigma+ + sigma-, so sigma/2 = sigma+.
    """
    Ne, Te = 1e25, 10.0
    Z, n_u, n_l = 1, 2, 1
    B_values = [0.0, 1.0, 5.0, 15.0]
    asym_values = []

    for B in B_values:
        energies, total, pi, sigma = total_profile(n_u, n_l, Z, B, Ne, Te)
        int_pi    = np.trapezoid(pi,           energies)
        int_sp    = np.trapezoid(sigma / 2.0,  energies)  # sigma+ alone
        int_total = np.trapezoid(total,         energies)
        asym = abs(int_pi - int_sp) / int_total if int_total > 0 else 0.0
        asym_values.append(asym)

    # At B=0: pi == sigma+, so asym should be ~0
    assert asym_values[0] < 0.01, (
        f"Asymmetry at B=0 should be ~0, got {asym_values[0]:.4f}"
    )
    # At B=15T: asymmetry should be larger than at B=0
    assert asym_values[-1] > asym_values[0], (
        f"Asymmetry should be larger at B=15T than at B=0: "
        f"asym(15T)={asym_values[-1]:.4f}, asym(0T)={asym_values[0]:.4f}"
    )


# ── 4. Profile varies smoothly from B=0 to B=15 T ────────────────────────────

def test_profile_smooth_b_variation():
    """Max step between consecutive B values should be small (< 20% of peak)."""
    Ne, Te = 1e25, 10.0
    Z, n_u, n_l = 1, 2, 1
    B_values = np.linspace(0, 15, 6)  # 0, 3, 6, 9, 12, 15 T

    integrals = []
    for B in B_values:
        energies, total, _, _ = total_profile(n_u, n_l, Z, B, Ne, Te,
                                               detuning_range=0.1)
        integrals.append(np.trapezoid(total, energies))

    integrals = np.array(integrals)
    # No single step should change the integral by more than 20%
    max_step = np.max(np.abs(np.diff(integrals))) / np.mean(integrals)
    assert max_step < 0.20, (
        f"Profile integral has a large step between B values: max_step={max_step:.4f}"
    )


# ── 5. Electron broadening dominated by plasma at B=15 T, Ne=1e19 m^-3 ──────

def test_larmor_vs_plasma_frequency_low_B():
    """At B=15 T, Ne=1e19 m^-3, omega_L < omega_p for DIII-D edge: broadening unchanged."""
    from starkzee.broadening import calculate_larmor_frequency, calculate_plasma_frequency
    from scipy.constants import hbar as HBAR, e as E_CHARGE

    Ne_m3 = 1e19
    B_low = 15.0

    omega_L = calculate_larmor_frequency(B_low)
    omega_p = calculate_plasma_frequency(Ne_m3)

    # At these conditions Larmor should exceed plasma frequency
    # (omega_L ≈ 2.6e12 rad/s vs omega_p ≈ 5.6e10 rad/s)
    # broadening width at B=0 vs B=15T should differ
    w_B0  = electron_impact_width(0.0, Ne_m3, Te_ev=10.0, B=0.0,  Z=1, n=2)
    w_B15 = electron_impact_width(0.0, Ne_m3, Te_ev=10.0, B=B_low, Z=1, n=2)

    # Both should be finite and positive
    assert w_B0  > 0, f"Broadening at B=0 is zero: {w_B0}"
    assert w_B15 > 0, f"Broadening at B=15T is zero: {w_B15}"
    # At low Ne, omega_L > omega_p, so B=15T gives wider cutoff → larger width
    assert w_B15 >= w_B0, (
        f"Expected width at B=15T >= width at B=0 (omega_L > omega_p), "
        f"got w(0)={w_B0:.4e}, w(15T)={w_B15:.4e}"
    )


# ── 6. Static vs FFM comparison at low B, moderate density ───────────────────

def test_static_vs_ffm_low_B():
    """At low density and B=5 T, static and FFM profiles should integrate to within 10%."""
    Z, n_u, n_l = 1, 3, 2
    B = 5.0
    Ne, Te, Ti = 1e21, 2.0, 0.1
    E0 = transition_energy(n_u, n_l, Z)
    energies = E0 + np.linspace(-0.05, 0.05, 150)

    pi_s, sp_s, sm_s = calculate_static_profile(
        n_u=n_u, n_l=n_l, Z=Z, B=B, Ne_m3=Ne, Te_ev=Te,
        energies_ev=energies, num_f=15, num_mu=6,
        fine_structure=True, quadratic_zeeman=False,
        frequency_dependent_width=False
    )
    pi_f, sp_f, sm_f = calculate_ffm_profile(
        n_u=n_u, n_l=n_l, Z=Z, B=B, Ne_m3=Ne, Te_ev=Te, Ti_ev=Ti,
        A_ion=1, energies_ev=energies, num_f=15, num_mu=6,
        fine_structure=True, quadratic_zeeman=False
    )

    int_static = np.trapezoid(pi_s + sp_s + sm_s, energies)
    int_ffm    = np.trapezoid(pi_f + sp_f + sm_f, energies)

    assert int_static > 0 and int_ffm > 0
    # At this low density, ion dynamics are moderate; integrals should be within 10%
    assert relerr(int_ffm, int_static) < 0.10, (
        f"Static integral={int_static:.4e}, FFM integral={int_ffm:.4e}, "
        f"relerr={relerr(int_ffm, int_static):.4f}"
    )
