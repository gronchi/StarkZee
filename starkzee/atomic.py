"""Public field-free atomic helpers for hydrogen-like radiators.

This module is a stable, user-facing facade over the implementations in
``starkzee.radiator``. Field-dressed transition energies, strengths, and
Einstein coefficients are available through
``LineProfile.compute_discrete()`` and its ``discrete`` result.
"""

from starkzee.radiator import (
    einstein_a as _einstein_a,
    einstein_a_from_strength,
    einstein_a_substate_rates as _einstein_a_substate_rates,
    line_strength as _line_strength,
    natural_decay_rates as _natural_decay_rates,
    oscillator_strength as _oscillator_strength,
    radial_dipole as _radial_dipole,
    radial_wavefunction as _radial_wavefunction,
    reduced_hydrogenic_dipole_j as _reduced_hydrogenic_dipole_j,
)

__all__ = [
    "radial_wavefunction",
    "radial_dipole",
    "reduced_hydrogenic_dipole_j",
    "line_strength",
    "oscillator_strength",
    "einstein_a",
    "einstein_a_from_strength",
    "einstein_a_substate_rates",
    "natural_decay_rates",
]


def radial_wavefunction(r_a0, n, l, Z=1):
    r"""Return the field-free Coulomb radial function ``R_nl`` [a0^-3/2].

    ``r_a0`` is a scalar or array of radii in Bohr radii. The function is a
    basis function; a field-dressed physical state is generally a linear
    combination of several such functions.
    """
    return _radial_wavefunction(r_a0, n, l, Z)


def radial_dipole(n1, l1, n2, l2, Z=1, method=None):
    r"""Return the signed field-free radial dipole integral [a0]."""
    return _radial_dipole(n1, l1, n2, l2, Z, method=method)


def reduced_hydrogenic_dipole_j(n_bra, l_bra, j_bra,
                                n_ket, l_ket, j_ket, Z=1, method=None):
    r"""Return the field-free J-reduced hydrogenic dipole [a0]."""
    return _reduced_hydrogenic_dipole_j(
        n_bra, l_bra, j_bra, n_ket, l_ket, j_ket, Z, method=method)


def line_strength(n_u, n_l, Z=1):
    r"""Return the field-free shell-to-shell line strength [a0^2]."""
    return _line_strength(n_u, n_l, Z)


def oscillator_strength(n_u, n_l, Z=1):
    """Return the weighted field-free absorption oscillator strength ``gf``."""
    return _oscillator_strength(n_u, n_l, Z)


def einstein_a(n_u, n_l, Z=1, A=None):
    r"""Return the field-free shell-averaged Einstein A coefficient [s^-1].

    Pass the emitter mass number as ``A`` for a reduced-mass-corrected photon
    energy. Omitting it preserves the historical infinite-mass convention.
    """
    return _einstein_a(n_u, n_l, Z, A=A)


def einstein_a_substate_rates(n_u, n_l, Z=1):
    r"""Return field-free E1 rates from each upper-shell basis state [s^-1]."""
    return _einstein_a_substate_rates(n_u, n_l, Z)


def natural_decay_rates(n, Z=1):
    r"""Return total field-free E1 decay rates of shell-``n`` basis states."""
    return _natural_decay_rates(n, Z)
