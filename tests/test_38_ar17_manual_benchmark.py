"""Hydrogen-like Ar XVII checks against the PPP manual ``miscel.`` example.

The source values are transcribed in ``scratch/papers/Readme-PPP-2024.tex``,
lines 375--383.  They quantify both the useful accuracy and the known QED
limit of StarkZee's present perturbative Dirac Hamiltonian.
"""

import numpy as np

from starkzee.multielectron import wigner_6j
from starkzee.radiator import (
    diagonalize_hamiltonian,
    reduced_hydrogenic_dipole_j,
)


AR17_MANUAL_ENERGIES_EV = {
    "2p1/2": 3318.220703,
    "2s1/2": 3318.388428,
    "2p3/2": 3323.036377,
}


def _ar17_dirac_excitation_groups():
    ground, _ = diagonalize_hamiltonian(
        1, 18, 0.0, quadratic_zeeman=False, fine_structure=True, A=40)
    excited, _ = diagonalize_hamiltonian(
        2, 18, 0.0, quadratic_zeeman=False, fine_structure=True, A=40)
    transitions = np.sort(excited.real - ground[0].real)
    return transitions[:4], transitions[4:]


def test_ar17_fine_structure_is_quantitatively_close_to_manual():
    """The O((Z alpha)^4) Hamiltonian resolves the manual's p splitting."""
    j12, j32 = _ar17_dirac_excitation_groups()
    assert np.ptp(j12) < 1e-10
    assert np.ptp(j32) < 1e-10

    predicted_p12 = float(np.mean(j12))
    predicted_p32 = float(np.mean(j32))
    manual_p12 = AR17_MANUAL_ENERGIES_EV["2p1/2"]
    manual_p32 = AR17_MANUAL_ENERGIES_EV["2p3/2"]

    # The current perturbative Dirac expansion is about 1 eV high in absolute
    # excitation energy at Z=18, while the physically important fine-structure
    # interval is already reproduced within 2 percent.
    assert abs(predicted_p12 - manual_p12) < 1.1
    assert abs(predicted_p32 - manual_p32) < 1.1
    predicted_interval = predicted_p32 - predicted_p12
    manual_interval = manual_p32 - manual_p12
    np.testing.assert_allclose(predicted_interval, manual_interval, rtol=0.02)


def test_ar17_current_dirac_model_explicitly_lacks_lamb_splitting():
    """Record, rather than conceal, the missing 2s1/2--2p1/2 QED shift."""
    j12, _ = _ar17_dirac_excitation_groups()
    predicted_2s_minus_2p = float(np.ptp(j12))
    manual_2s_minus_2p = (
        AR17_MANUAL_ENERGIES_EV["2s1/2"]
        - AR17_MANUAL_ENERGIES_EV["2p1/2"]
    )
    assert predicted_2s_minus_2p < 1e-10
    np.testing.assert_allclose(manual_2s_minus_2p, 0.167725, atol=1e-6)


def test_wigner_6j_values_used_by_ar17_recoupling():
    np.testing.assert_allclose(
        wigner_6j(0, 0.5, 0.5, 0.5, 1, 1), np.sqrt(6) / 6)
    np.testing.assert_allclose(
        wigner_6j(0, 0.5, 0.5, 1.5, 1, 1), -np.sqrt(6) / 6)
    assert wigner_6j(0, 0, 0, 0, 0, 1) == 0.0


def test_ar17_reduced_dipoles_match_manual_after_consistent_rephasing():
    """Nonrelativistic magnitudes agree and one level phase maps all signs."""
    # PPP manual level order: 1=1s1/2, 2=2p1/2, 3=2s1/2, 4=2p3/2.
    calculated = np.array([
        reduced_hydrogenic_dipole_j(1, 0, 0.5, 2, 1, 0.5, 18),
        reduced_hydrogenic_dipole_j(2, 1, 0.5, 2, 0, 0.5, 18),
        reduced_hydrogenic_dipole_j(1, 0, 0.5, 2, 1, 1.5, 18),
        reduced_hydrogenic_dipole_j(2, 0, 0.5, 2, 1, 1.5, 18),
    ])
    manual = np.array([-0.057735, -0.23400, -0.082023, -0.33237])

    # Positive-near-origin radial phases give signs (-,+,-,+).  Rephasing only
    # manual level 3 (2s1/2) by -1 maps all four edges to the PPP convention.
    level_phases = {1: 1.0, 2: 1.0, 3: -1.0, 4: 1.0}
    edges = ((1, 2), (2, 3), (1, 4), (3, 4))
    rephased = np.array([
        level_phases[i] * level_phases[j] * value
        for value, (i, j) in zip(calculated, edges)
    ])

    np.testing.assert_allclose(rephased, manual, rtol=0.02, atol=0.0)
    assert np.prod(np.sign(calculated)) == np.prod(np.sign(manual)) == 1.0
