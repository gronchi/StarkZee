import pytest
from starkzee.atomic_data import load_levels, calculate_wavenumber, AtomicState

def test_load_levels_success():
    """Test loading atomic states successfully from the JSON."""
    # Test without fine structure
    levels_no_fs = load_levels("H", fine_structure=False)
    assert len(levels_no_fs) == 12
    assert levels_no_fs[0].n == 1
    assert levels_no_fs[0].energy == 0.0
    
    assert levels_no_fs[1].n == 2
    assert levels_no_fs[1].energy == 82259.158
    
    # Test with fine structure
    levels_fs = load_levels("H", fine_structure=True)
    assert len(levels_fs) == 64
    
    # Check 2p 3/2 state
    state_2p_3_2 = levels_fs[3]
    assert state_2p_3_2.n == 2
    assert state_2p_3_2.l == 1
    assert state_2p_3_2.j == 1.5
    assert state_2p_3_2.energy == 82259.2850014

def test_load_levels_errors():
    """Test error handling when loading non-existent atoms."""
    with pytest.raises(ValueError, match="Atom 'Li' not found"):
        load_levels("Li", fine_structure=False)

def test_calculate_wavenumber():
    """Test wavenumber calculation logic."""
    upper = AtomicState(energy=10.0, n=2)
    lower = AtomicState(energy=2.0, n=1)
    
    # Basic difference
    wavenumber = calculate_wavenumber(upper, lower)
    assert wavenumber == 8.0
    
    # With lambda shift
    wavenumber_shifted = calculate_wavenumber(upper, lower, lambda_shift=0.5)
    assert wavenumber_shifted == 8.5
    
    # Negative lambda shift
    wavenumber_shifted_neg = calculate_wavenumber(upper, lower, lambda_shift=-0.5)
    assert wavenumber_shifted_neg == 7.5

def test_calculate_wavenumber_errors():
    """Test error handling for invalid transition energies."""
    upper = AtomicState(energy=2.0, n=1)
    lower = AtomicState(energy=10.0, n=2)
    
    with pytest.raises(ValueError, match="Upper state must have higher energy"):
        calculate_wavenumber(upper, lower)
