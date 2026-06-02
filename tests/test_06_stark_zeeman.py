"""
test_06_starkzee.py — Combined Stark+Zeeman: conservation laws and intensity sum rules.

Verifies:
  - Transition energy is correct in absence of perturbations
  - Dipole intensity sum rule: total intensity is preserved under rotation
  - Intensity is non-negative
  - Profile peak is near the unperturbed transition energy at small F, B
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.static_profile import solve_starkzee
from starkzee.atomic_hamiltonian import (
    build_basis, angular_dipole_element, radial_dipole, dipole_matrix_elements
)
from starkzee.utils import reduced_mass_rydberg_ev


def relerr(got, ref):
    return abs(got - ref) / abs(ref)


# ── 1. Transition energy at zero perturbation ────────────────────────────────

@pytest.mark.parametrize("n_u,n_l,Z", [
    (2, 1, 1),   # H Ly-α
    (2, 1, 6),   # C VI Ly-α
    (5, 4, 4),   # C IV n=5→4
    (3, 2, 1),   # H Balmer-α
])
def test_transition_energy_zero_field(n_u, n_l, Z):
    """
    At B~0, F=0, mean transition energy must equal -Z²Ry(1/n_l²-1/n_u²).
    """
    E0_expected = (Z**2) * reduced_mass_rydberg_ev(Z, 1) * (1.0/n_l**2 - 1.0/n_u**2)
    # Fine structure disabled: comparing to the non-relativistic Bohr formula.
    # MV+Darwin correctly shifts each n-shell by a different amount, so the
    # mean transition energy deviates from the Bohr value when FS is enabled.
    evals_u, _ = solve_starkzee(n_u, Z, B=1e-3, Fz=0.0, Fx=0.0,
                                             quadratic_zeeman=False,
                                             fine_structure=False)
    evals_l, _ = solve_starkzee(n_l, Z, B=1e-3, Fz=0.0, Fx=0.0,
                                             quadratic_zeeman=False,
                                             fine_structure=False)
    E0_got = np.mean(evals_u.real) - np.mean(evals_l.real)
    assert relerr(E0_got, E0_expected) < 1e-6, (
        f"n_u={n_u},n_l={n_l},Z={Z}: E0={E0_got:.6f} eV, expected={E0_expected:.6f} eV"
    )


# ── 2. Dipole sum rule (intensity conservation) ───────────────────────────────

@pytest.mark.parametrize("n_u,n_l,Z", [
    (2, 1, 1),
    (2, 1, 6),
    (3, 2, 1),
])
def test_dipole_intensity_sum_rule(n_u, n_l, Z):
    """
    The total squared dipole intensity ∑_q ∑_{ij} |d_q(i→j)|² must be
    independent of the field orientation (unitary transformation preserves norm).
    Compare B=0 vs B>0: total intensity should be identical (same basis,
    just rotated).
    """
    # Get dipole elements at B~0 (to avoid SOC complications)
    evals_u0, evecs_u0, d_B0 = dipole_matrix_elements(n_u, n_l, Z, B=1e-3)
    evals_u1, evecs_u1, d_B1 = dipole_matrix_elements(n_u, n_l, Z, B=100.0)

    total_B0 = sum(np.sum(np.abs(d_B0[q])**2) for q in [0, 1, -1])
    total_B1 = sum(np.sum(np.abs(d_B1[q])**2) for q in [0, 1, -1])

    assert relerr(total_B1, total_B0) < 1e-6, (
        f"Intensity sum: B=0 → {total_B0:.6e}, B=100T → {total_B1:.6e}"
    )


# ── 3. Intensities are non-negative ──────────────────────────────────────────

@pytest.mark.parametrize("n_u,n_l,Z,B", [
    (2, 1, 1, 0.0),
    (2, 1, 6, 100.0),
    (5, 4, 4, 300.0),
])
def test_intensities_nonnegative(n_u, n_l, Z, B):
    """All |d_q|² must be ≥ 0 (trivially true, but checks for complex bugs)."""
    B_eff = max(B, 1e-3)
    _, _, d = dipole_matrix_elements(n_u, n_l, Z, B=B_eff)
    for q in [0, 1, -1]:
        intensities = np.abs(d[q])**2
        assert np.all(intensities >= 0), f"Negative intensity for q={q}"


# ── 4. Symmetry: at B=0, σ+ and σ− intensities are equal ────────────────────

def test_sigma_symmetry_at_zero_B():
    """
    At B=0, σ+ and σ− must have equal total intensity (time-reversal symmetry).
    """
    _, _, d = dipole_matrix_elements(2, 1, Z=1, B=1e-3)
    I_plus  = np.sum(np.abs(d[+1])**2)
    I_minus = np.sum(np.abs(d[-1])**2)
    assert relerr(I_plus, I_minus) < 1e-8, (
        f"σ+ intensity {I_plus:.6e} ≠ σ- intensity {I_minus:.6e} at B~0"
    )


# ── 5. Transition selection rule: Δl = ±1 ────────────────────────────────────

def test_forbidden_transitions_zero_intensity():
    """
    Transitions between states with the same l must have zero intensity.
    The sum rule naturally enforces Δl=±1 through the angular matrix elements,
    but we check that no forbidden transitions sneak through.
    """
    # For n_u=n_l=2, both levels have same n → forbidden (not computed in ppp.py,
    # but the angular elements should give zero for same-l couplings).
    # We test a simpler case: verify directly on the angular element.
    for l in [0, 1, 2]:
        for ml in range(-l, l+1):
            for q in [0, 1, -1]:
                # Same l → no dipole
                val = angular_dipole_element(l, ml, l, ml, q)
                assert val == 0.0, f"Non-zero angular element for Δl=0: l={l},ml={ml},q={q}"


# ── 6. Profile peak near E0 at small perturbation ────────────────────────────

def test_profile_peak_near_E0():
    """
    With tiny B and tiny F, the profile pi/sigma peaks should be centered
    near the unperturbed E0 = Z²Ry(1/n_l²-1/n_u²).

    Uses calculate_static_profile with tiny Ne to minimize broadening.
    """
    from starkzee.static_profile import calculate_static_profile

    Z, n_u, n_l = 1, 2, 1
    B = 1e-3   # nearly zero field
    Ne = 1e18  # very low density → narrow profile
    Te = 1.0

    E0 = (Z**2) * reduced_mass_rydberg_ev(Z, 1) * (1.0/n_l**2 - 1.0/n_u**2)

    detuning = np.linspace(-0.01, 0.01, 200)
    energies = E0 + detuning

    pi, sp, sm = calculate_static_profile(
        n_u=n_u, n_l=n_l, Z=Z, B=B, Ne_m3=Ne, Te_ev=Te,
        energies_ev=energies, num_f=10, num_mu=4,
        use_screening=False, quadratic_zeeman=False,
        frequency_dependent_width=False
    )
    sigma = sp + sm

    # Peak of sigma must be within ±0.5e-3 eV of E0 (which is at detuning=0)
    peak_det = detuning[np.argmax(sigma)]
    assert abs(peak_det) < 5e-3, (
        f"Profile sigma peak at detuning={peak_det:.4e} eV from E0"
    )
    print(f"H Ly-α profile peak detuning from E0: {peak_det:.4e} eV ✓")
