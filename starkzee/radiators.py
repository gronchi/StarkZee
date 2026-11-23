"""Radiator interface for the profile solvers.

A *radiator* supplies everything the Stark-Zeeman profile solvers need about one
emission line, independently of how the atomic structure is described:

* the upper and lower manifolds (:class:`Manifold`): field-free + magnetic
  Hamiltonian H0(B), the Stark templates M_z and M_x (so that the electric
  interaction is Fz*M_z + Fx*M_x), and the E1 decay rate of every basis state;
* the dipole matrices D_q between the two manifolds;
* the line reference energies and the unit of the Hamiltonian matrices;
* the electron-broadening hooks (impact widths, per-state operator scaling and
  the full PPP collision operator);
* optionally ``mass_amu``, the emitter mass [u] for the Doppler width (the
  hydrogenic radiator leaves it to the solver's ``species``).

:class:`HydrogenicTransition` implements it for the hydrogenic n -> n' shells
of :mod:`starkzee.radiator`. Multi-electron radiators implement the same
attributes and methods (see ``notes/Z_gt1_and_multielectron_plan.md``, B1).
"""
from dataclasses import dataclass

import numpy as np
from scipy.constants import hbar as _HBAR, e as _E_CHARGE

from starkzee.broadening import (
    electron_impact_collision_coefficient, electron_impact_r2_scaling,
    electron_impact_width_model,
)
from starkzee.collision import ppp_complex_sdts
from starkzee.radiator import (
    _uncoupled_dipole_matrices, build_hamiltonian, einstein_a, natural_decay_rates,
)
from starkzee.utils import (
    energy_ev_to_wavenumber_cm, reduced_mass_rydberg_ev, wavenumber_cm_to_energy_ev,
)


@dataclass
class Manifold:
    """One manifold (upper or lower) in the units of its Hamiltonian.

    H0 is H_atom(B) with the reference energy E_ref removed from the diagonal,
    so that it is well conditioned; the eigenvalues of H0 + Fz*M_z + Fx*M_x are
    energies relative to E_ref. decay_rates are E1 spontaneous rates [s^-1] of
    the basis states.
    """
    H0: np.ndarray
    E_ref: float
    M_z: np.ndarray
    M_x: np.ndarray
    decay_rates: np.ndarray


class HydrogenicTransition:
    """Hydrogenic n_u -> n_l line in the uncoupled |n l m_l m_s> basis.

    Parameters follow :func:`~starkzee.static_profile.calculate_static_profile`.
    The Hamiltonians are in cm^-1 with ``use_empirical_data=True`` and in eV
    otherwise; ``unit_to_ev`` converts eigenvalue differences to eV.
    """

    def __init__(self, n_u, n_l, Z, B, quadratic_zeeman=True, fine_structure=True, A=1,
                 use_empirical_data=True, atom="H"):
        from starkzee.static_profile import _stark_templates, line_reference_energy

        self.n_u, self.n_l, self.Z = n_u, n_l, Z
        self.default_emitter_charge = Z - 1
        self.dipoles = _uncoupled_dipole_matrices(n_u, n_l, Z)

        H_u = build_hamiltonian(n_u, Z, B, quadratic_zeeman, fine_structure, A,
                                use_empirical_data=use_empirical_data, atom=atom).copy()
        H_l = build_hamiltonian(n_l, Z, B, quadratic_zeeman, fine_structure, A,
                                use_empirical_data=use_empirical_data, atom=atom).copy()
        M_z_u, M_x_u = _stark_templates(n_u, Z)
        M_z_l, M_x_l = _stark_templates(n_l, Z)
        if use_empirical_data:
            # H is in cm^-1 (NIST levels, incl. Lamb shift); the Stark templates are in
            # eV/(V/m) and are scaled to cm^-1/(V/m). Centre on the mean level energy.
            ev_to_cm = energy_ev_to_wavenumber_cm(1.0)
            M_z_u, M_x_u = M_z_u * ev_to_cm, M_x_u * ev_to_cm
            M_z_l, M_x_l = M_z_l * ev_to_cm, M_x_l * ev_to_cm
            E_u = H_u.diagonal().real.mean()
            E_l = H_l.diagonal().real.mean()
            self.E0_line = float(wavenumber_cm_to_energy_ev(E_u - E_l))
            self.unit_to_ev = float(wavenumber_cm_to_energy_ev(1.0))
        else:
            E_u = -(Z**2) * reduced_mass_rydberg_ev(Z, A) / (n_u**2)
            E_l = -(Z**2) * reduced_mass_rydberg_ev(Z, A) / (n_l**2)
            self.E0_line = (Z**2) * reduced_mass_rydberg_ev(Z, A) * (1.0/n_l**2 - 1.0/n_u**2)
            self.unit_to_ev = 1.0
        # Remove the shell energies from the diagonal (norm ~1e-3 eV instead of ~eV).
        for H, E in ((H_u, E_u), (H_l, E_l)):
            for i in range(H.shape[0]):
                H[i, i] -= E
        self.upper = Manifold(H_u, E_u, M_z_u, M_x_u, natural_decay_rates(n_u, Z))
        self.lower = Manifold(H_l, E_l, M_z_l, M_x_l, natural_decay_rates(n_l, Z))
        # Fixed physical line reference (Doppler width, electron detuning origin).
        self.E0_physical = line_reference_energy(n_u, n_l, Z, A, use_empirical_data, atom)

    # ---------------------------------------------------------------- natural width
    def shell_natural_width_ev(self):
        """Historical shell-averaged natural HWHM [eV] (natural_width_mode='shell_average')."""
        gamma_upper = sum(einstein_a(self.n_u, k, self.Z) for k in range(1, self.n_u))
        gamma_lower = (sum(einstein_a(self.n_l, k, self.Z) for k in range(1, self.n_l))
                       if self.n_l > 1 else 0.0)
        return _HBAR * (gamma_upper + gamma_lower) / (2.0 * _E_CHARGE)

    # ---------------------------------------------------------------- electron broadening
    def electron_widths(self, detuning_ev, Ne_m3, Te_ev, B, electron_model):
        """Impact widths (upper, lower) [eV] at the detuning from the line reference.

        The lower contribution is 0 for n_l = 1 (no intra-shell dipole channel).
        """
        w_u = electron_impact_width_model(detuning_ev, Ne_m3, Te_ev, B, self.Z, n=self.n_u,
                                          electron_model=electron_model)
        w_l = (electron_impact_width_model(detuning_ev, Ne_m3, Te_ev, B, self.Z, n=self.n_l,
                                           electron_model=electron_model)
               if self.n_l > 1 else 0.0)
        return w_u, w_l

    def electron_r2_scaling(self, vectors_u, vectors_l, r2_form):
        """Per-eigenstate <k|r^2|k>/<r^2> scaling (upper, lower or None) of the operator widths."""
        scale_u = electron_impact_r2_scaling(vectors_u, self.n_u, self.Z, r2_form=r2_form)
        scale_l = (electron_impact_r2_scaling(vectors_l, self.n_l, self.Z, r2_form=r2_form)
                   if self.n_l > 1 else None)
        return scale_u, scale_l

    def ppp_impact_coefficient(self, Ne_m3, Te_ev, B):
        """Appendix-B impact coefficient of the full PPP collision operator."""
        return electron_impact_collision_coefficient(Ne_m3, Te_ev, B, self.Z, n=self.n_u)

    def ppp_self_term_r2(self):
        """Mean intra-shell <r^2> of the self terms (upper + lower), for grid diagnostics."""
        n_u, n_l, Z = self.n_u, self.n_l, self.Z
        r2_u = 9.0 * n_u**2 * (n_u**2 - 1.0) / (8.0 * Z**2)
        r2_l = 9.0 * n_l**2 * (n_l**2 - 1.0) / (8.0 * Z**2) if n_l > 1 else 0.0
        return r2_u + r2_l

    def ppp_sdts(self, dE, vectors_u, vectors_l, coefficient, natural):
        """Complex generalized SDTs (frequencies, widths, strengths) with the full PPP operator."""
        return ppp_complex_sdts(dE, vectors_u, vectors_l, self.dipoles, self.n_u, self.n_l, self.Z,
                                coefficient, natural, electron_interference=True)
