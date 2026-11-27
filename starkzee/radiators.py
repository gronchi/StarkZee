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
from starkzee.angular import spin_orbit_element
from starkzee.radiator import (
    _uncoupled_dipole_matrices, build_basis, build_hamiltonian, einstein_a, natural_decay_rates,
)
from starkzee.utils import (
    emitter_mass_amu, energy_ev_to_wavenumber_cm, mass_number, reduced_mass_factor, reduced_mass_rydberg_ev,
    resolve_atom,
    wavenumber_cm_to_energy_ev,
)


@dataclass
class Manifold:
    """One manifold (upper or lower) in the units of its Hamiltonian.

    H0 is H_atom(B) with the reference energy E_ref removed from the diagonal,
    so that it is well conditioned; the eigenvalues of H0 + Fz*M_z + Fx*M_x are
    energies relative to E_ref. decay_rates are E1 spontaneous rates [s^-1] of
    the basis states. m_values (optional) is the magnetic quantum number M of
    each basis state; with it the solver uses the Delta M block structure of the
    PPP generator at B = 0.
    """
    H0: np.ndarray
    E_ref: float
    M_z: np.ndarray
    M_x: np.ndarray
    decay_rates: np.ndarray
    m_values: np.ndarray = None


ZEEMAN_TREATMENTS = ("exact", "weak", "strong")


def check_zeeman(zeeman):
    if zeeman not in ZEEMAN_TREATMENTS:
        raise ValueError(f"zeeman must be one of {ZEEMAN_TREATMENTS}, got {zeeman!r}")


def _fine_structure_projectors(n):
    """Projectors onto the (l, j) fine-structure levels of shell n (uncoupled basis)."""
    basis = build_basis(n)
    projectors = []
    for l in range(n):
        idx = [i for i, s in enumerate(basis) if s.l == l]
        ls = np.array([[spin_orbit_element(l, basis[i].ml, basis[i].ms, basis[j].ml, basis[j].ms)
                        for j in idx] for i in idx])
        values, vectors = np.linalg.eigh(ls)            # l/2 (j = l + 1/2), -(l + 1)/2 (j = l - 1/2)
        for value in sorted(set(np.round(values, 9))):
            v = vectors[:, np.abs(values - value) < 1e-6]
            P = np.zeros((len(basis), len(basis)))
            P[np.ix_(idx, idx)] = v @ v.T
            projectors.append((l, P))
    return projectors


def zeeman_treatment(H_field, H_zero, n, zeeman):
    """Shell Hamiltonian under a Zeeman treatment, from H(B) and H(0) in the uncoupled basis.

    Treatments::

        exact  : H(B) (Zeeman and fine structure diagonalized together; intermediate field);
        weak   : H(0) + sum_(l,j) P_lj [H(B) - H(0)] P_lj, the field terms inside each
                 fine-structure level only (g_j, anomalous Zeeman effect);
        strong : H(B) with the j splitting of every l replaced by its mean, i.e. without
                 spin-orbit coupling (Paschen-Back limit); l-dependent shifts are kept.
    """
    check_zeeman(zeeman)
    if zeeman == "exact":
        return H_field
    projectors = _fine_structure_projectors(n)
    if zeeman == "weak":
        field = H_field - H_zero
        return H_zero + sum(P @ field @ P for _, P in projectors)
    H = H_field - H_zero
    for l in range(n):
        P = sum(P for k, P in projectors if k == l)
        H = H + P * (np.trace(P @ H_zero).real / np.trace(P))
    return H


def hydrogenic_m_values(n):
    """M = m_l + m_s of the uncoupled basis of :func:`starkzee.radiator.build_basis` (same order)."""
    return np.array([ml + ms for l in range(n) for ml in range(-l, l + 1) for ms in (0.5, -0.5)])


class HydrogenicTransition:
    """Hydrogenic n_u -> n_l line in the uncoupled ``|n l m_l m_s>`` basis.

    Parameters follow :func:`~starkzee.static_profile.calculate_static_profile`.
    The Hamiltonians are in cm^-1 with ``use_empirical_data=True`` and in eV
    otherwise; ``unit_to_ev`` converts eigenvalue differences to eV.
    ``zeeman`` selects the treatment of the magnetic field (:func:`zeeman_treatment`):
    "exact" (default, intermediate field), "weak" or "strong".
    ``A`` is the mass number (default: the most abundant isotope of Z;
    ``math.inf`` for an infinite nuclear mass); ``atom`` the measured-level table,
    by default the one of that isotope ('H', 'D', 'T'). ``mass_amu`` is the
    emitter mass (nucleus + electron) used for the Doppler width.
    """

    def __init__(self, n_u, n_l, Z, B, quadratic_zeeman=True, fine_structure=True, A=None,
                 use_empirical_data=True, atom=None, zeeman="exact"):
        from starkzee.static_profile import _stark_templates, line_reference_energy

        A = mass_number(Z, A)
        if use_empirical_data:
            atom = resolve_atom(Z, A, atom)
        self.n_u, self.n_l, self.Z, self.A, self.atom = n_u, n_l, Z, A, atom
        self.mass_amu = emitter_mass_amu(Z, A)
        self.default_emitter_charge = Z - 1
        a_mu = 1.0 / reduced_mass_factor(Z, A)                 # radial lengths in the reduced Bohr radius
        self.dipoles = {q: D * a_mu for q, D in _uncoupled_dipole_matrices(n_u, n_l, Z).items()}
        # electron-impact widths scale with <r^2>: (a_mu / a0)^2 on the a0-based broadening model
        self._r2_scale = a_mu**2

        check_zeeman(zeeman)
        self.zeeman = zeeman

        def shell(n):
            H = build_hamiltonian(n, Z, B, quadratic_zeeman, fine_structure, A,
                                  use_empirical_data=use_empirical_data, atom=atom).copy()
            if zeeman == "exact":
                return H
            H0 = build_hamiltonian(n, Z, 0.0, quadratic_zeeman, fine_structure, A,
                                   use_empirical_data=use_empirical_data, atom=atom)
            return zeeman_treatment(H, H0, n, zeeman).astype(complex)

        H_u, H_l = shell(n_u), shell(n_l)
        M_z_u, M_x_u = _stark_templates(n_u, Z, A)
        M_z_l, M_x_l = _stark_templates(n_l, Z, A)
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
        self.upper = Manifold(H_u, E_u, M_z_u, M_x_u, natural_decay_rates(n_u, Z, A), hydrogenic_m_values(n_u))
        self.lower = Manifold(H_l, E_l, M_z_l, M_x_l, natural_decay_rates(n_l, Z, A), hydrogenic_m_values(n_l))
        # Fixed physical line reference (Doppler width, electron detuning origin).
        self.E0_physical = line_reference_energy(n_u, n_l, Z, A, use_empirical_data, atom)

    # ---------------------------------------------------------------- natural width
    def shell_natural_width_ev(self):
        """Historical shell-averaged natural HWHM [eV] (natural_width_mode='shell_average')."""
        gamma_upper = sum(einstein_a(self.n_u, k, self.Z, self.A) for k in range(1, self.n_u))
        gamma_lower = (sum(einstein_a(self.n_l, k, self.Z, self.A) for k in range(1, self.n_l))
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
        return w_u * self._r2_scale, w_l * self._r2_scale

    def electron_r2_scaling(self, vectors_u, vectors_l, r2_form):
        """Per-eigenstate <k|r^2|k>/<r^2> scaling (upper, lower or None) of the operator widths."""
        scale_u = electron_impact_r2_scaling(vectors_u, self.n_u, self.Z, r2_form=r2_form)
        scale_l = (electron_impact_r2_scaling(vectors_l, self.n_l, self.Z, r2_form=r2_form)
                   if self.n_l > 1 else None)
        return scale_u, scale_l

    def ppp_impact_coefficient(self, Ne_m3, Te_ev, B):
        """Appendix-B impact coefficient of the full PPP collision operator."""
        return electron_impact_collision_coefficient(Ne_m3, Te_ev, B, self.Z, n=self.n_u) * self._r2_scale

    def ppp_self_term_r2(self):
        """Mean intra-shell <r^2> of the self terms (upper + lower), for grid diagnostics."""
        n_u, n_l, Z = self.n_u, self.n_l, self.Z
        r2_u = 9.0 * n_u**2 * (n_u**2 - 1.0) / (8.0 * Z**2)
        r2_l = 9.0 * n_l**2 * (n_l**2 - 1.0) / (8.0 * Z**2) if n_l > 1 else 0.0
        return r2_u + r2_l          # [a0^2]; ppp_impact_coefficient carries the (a_mu/a0)^2 factor

    def ppp_sdts(self, dE, vectors_u, vectors_l, coefficient, natural, m_values=None):
        """Complex generalized SDTs (frequencies, widths, strengths) with the full PPP operator.

        m_values = (M of the upper, M of the lower dressed states) enables the
        Delta M block diagonalization (field along z, B = 0).
        """
        return ppp_complex_sdts(dE, vectors_u, vectors_l, self.dipoles, self.n_u, self.n_l, self.Z,
                                coefficient, natural, electron_interference=True, m_values=m_values)
