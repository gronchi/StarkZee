import pytest
import numpy as np
from starkzee.atomic_hamiltonian import build_hamiltonian, build_basis
from starkzee.utils import BOHR_MAGNETON_EV_T, G_S, energy_ev_to_wavenumber_cm

def test_empirical_hamiltonian_b0():
    """Test that B=0 Hamiltonian exactly reproduces empirical energies on its diagonal."""
    n = 2
    Z = 1
    B = 0.0

    # Get Hamiltonian with empirical data
    H_emp = build_hamiltonian(n, Z, B, quadratic_zeeman=False, fine_structure=True, use_empirical_data=True, atom="H")

    # Because B=0, H_emp should be easily diagonalized to yield EXACTLY the empirical energies
    evals, _ = np.linalg.eigh(H_emp)

    # We expect 2s_1/2, 2p_1/2, 2p_3/2 energies from the JSON (in cm⁻¹)
    # n=2, l=0, j=0.5: 82258.9543992821
    # n=2, l=1, j=0.5: 82258.919113
    # n=2, l=1, j=1.5: 82259.2850014

    expected_energies = [
        82258.919113,     # 2p_1/2 (degenerate x2)
        82258.919113,
        82258.9543992821, # 2s_1/2 (degenerate x2)
        82258.9543992821,
        82259.2850014,    # 2p_3/2 (degenerate x4)
        82259.2850014,
        82259.2850014,
        82259.2850014
    ]

    expected_sorted = np.sort(expected_energies)
    evals_sorted = np.sort(evals)

    np.testing.assert_allclose(evals_sorted, expected_sorted, rtol=1e-12)

def test_empirical_hamiltonian_zeeman():
    """Test that Zeeman terms are added correctly on top of empirical Hamiltonian."""
    n = 2
    Z = 1
    B = 1.0 # 1 Tesla

    # The Zeeman shift is mu_B * B. At 1T, mu_B * B is about 5.788e-5 eV.
    # We just ensure it runs without errors and produces different eigenvalues than B=0.
    H_emp_b1 = build_hamiltonian(n, Z, B, quadratic_zeeman=True, fine_structure=True, use_empirical_data=True, atom="H")
    evals_b1, _ = np.linalg.eigh(H_emp_b1)

    H_emp_b0 = build_hamiltonian(n, Z, 0.0, quadratic_zeeman=False, fine_structure=True, use_empirical_data=True, atom="H")
    evals_b0, _ = np.linalg.eigh(H_emp_b0)

    # Degeneracy should be lifted
    assert len(np.unique(np.round(evals_b1, 8))) > len(np.unique(np.round(evals_b0, 8)))


def test_empirical_zeeman_unit_scale():
    """Zeeman eigenvalue spread is in cm⁻¹, not eV, when use_empirical_data=True.

    This guards against a latent bug where the Zeeman terms (computed in eV)
    were added directly onto the empirical Hamiltonian (in cm⁻¹) without unit
    conversion, making them ~8066x too small and effectively invisible.

    At B=1 T the Zeeman span across all n=2 states is ~1.87 cm⁻¹.
    The buggy code would produce a span of ~2.3e-4 (the raw eV value),
    leaving the eigenvalue spread unchanged from the B=0 fine-structure
    value of ~0.37 cm⁻¹.
    """
    n, Z, B = 2, 1, 1.0

    evals_0 = np.linalg.eigvalsh(
        build_hamiltonian(n, Z, 0.0, quadratic_zeeman=False, fine_structure=True,
                          use_empirical_data=True, atom="H")
    )
    evals_B = np.linalg.eigvalsh(
        build_hamiltonian(n, Z, B, quadratic_zeeman=False, fine_structure=True,
                          use_empirical_data=True, atom="H")
    )

    # Expected Zeeman span in cm⁻¹: range of (m_l + g_s * m_s) × μ_B × B
    basis = build_basis(n)
    m_vals = [s.ml + G_S * s.ms for s in basis]
    zeeman_span_cm = float(energy_ev_to_wavenumber_cm(
        (max(m_vals) - min(m_vals)) * BOHR_MAGNETON_EV_T * B
    ))  # ≈ 1.87 cm⁻¹ at B=1 T

    spread_B = evals_B.max() - evals_B.min()

    # Correct behaviour: spread matches the Zeeman span (fine-structure contributions
    # cancel between the extreme states, both drawn from the 2p level).
    # Buggy behaviour: spread ≈ 0.37 cm⁻¹ (fine structure only; Zeeman negligible).
    np.testing.assert_allclose(spread_B, zeeman_span_cm, rtol=1e-4,
        err_msg=f"Eigenvalue spread {spread_B:.4f} cm⁻¹ should equal Zeeman span "
                f"{zeeman_span_cm:.4f} cm⁻¹; got {spread_B:.4e} — "
                f"Zeeman may have been added in eV ({zeeman_span_cm/8065:.2e}).")
