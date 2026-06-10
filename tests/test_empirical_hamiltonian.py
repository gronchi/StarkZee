import pytest
import numpy as np
from starkzee.atomic_hamiltonian import build_hamiltonian

def test_empirical_hamiltonian_b0():
    """Test that B=0 Hamiltonian exactly reproduces empirical energies on its diagonal."""
    n = 2
    Z = 1
    B = 0.0
    
    # Get Hamiltonian with empirical data
    H_emp = build_hamiltonian(n, Z, B, quadratic_zeeman=False, fine_structure=True, use_empirical_data=True, atom="H")
    
    # Because B=0, H_emp should be easily diagonalized to yield EXACTLY the empirical energies
    evals, _ = np.linalg.eigh(H_emp)
    
    # We expect 2s_1/2, 2p_1/2, 2p_3/2 energies from the JSON
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
