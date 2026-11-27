"""Impact-limit electron-collision operator for non-degenerate (multi-electron) manifolds.

Generalizes :func:`starkzee.collision.build_ppp_collision_operator` (PPP manual,
Appendix B, Eq. B1) from one degenerate hydrogenic shell to arbitrary levels.
Every d.d product carries the impact factor of its own pair of field-free
levels::

    W(a, b) = W0 [C_n + G(0)],  G evaluated with the cutoff
    omega_c(a, b) = max(omega_p, omega_L, omega_e, |E_a - E_b|)

(Ferri, Peyrusse & Calisti 2022: the state-to-state frequency omega_aa' enters
the maximum; the hydrogenic code keeps it at zero). For each manifold
(upper u, lower l) the operator in the coherence space ``|u><l|`` is::

    Phi = A_u (x) 1 + 1 (x) A_l^T - sum_c [ (r_u,c o W_u) (x) r_l,c^T + r_u,c (x) (r_l,c o W_l)^T ],
    A[a, a'] = sum_c sum_a'' r_c[a, a''] r_c[a'', a'] (W[a, a''] + W[a', a'']) / 2,

with o the elementwise product. The intermediate states a'' run over the
manifold and over additional perturbing levels (self terms only). With all gaps
below the plasma cutoff it reduces exactly to the hydrogenic operator,
A (r.r (x) 1 + 1 (x) r.r^T - 2 r (x) r^T).

As in the hydrogenic code, n (in C_n and in the GBK argument) and the charge Z
are those of the upper manifold for all three terms; for multi-electron levels n
is an effective principal quantum number (C_n uses its rounded value, at least
2). Cartesian dipoles are in a0, W0 in eV/a0^2.
"""
import numpy as np

from starkzee.broadening import (
    _strong_collision_constant, calculate_configuration_frequency, calculate_electron_impact_prefactor,
    calculate_larmor_frequency, calculate_plasma_frequency, gbk_model,
)
from starkzee.utils import E_CHARGE, HBAR


def cutoff_frequency_ev(Ne_m3, Te_ev, B):
    """omega_c0 = max(omega_p, omega_L, omega_e) [eV], the level-independent part of the cutoff."""
    rad = max(calculate_plasma_frequency(Ne_m3), calculate_larmor_frequency(B),
              calculate_configuration_frequency(Ne_m3, Te_ev))
    return rad * HBAR / E_CHARGE


def impact_factors(energies_ev, Ne_m3, Te_ev, B, n, Z):
    """``W[a, b] = W0 [C_n + G(0, max(omega_c0, |E_a - E_b|))]`` [eV/a0^2] for the given level energies."""
    E = np.asarray(energies_ev, dtype=float)
    omega_c = np.maximum(cutoff_frequency_ev(Ne_m3, Te_ev, B), np.abs(E[:, None] - E[None, :]))
    C = _strong_collision_constant(max(2, int(round(n))))
    return calculate_electron_impact_prefactor(Ne_m3, Te_ev) * (C + gbk_model(0.0, omega_c, Te_ev, Z, n=n))


def line_center_factor(Ne_m3, Te_ev, B, n, Z):
    """W0 [C_n + G(0, omega_c0)]: the common factor of degenerate levels [eV/a0^2]."""
    C = _strong_collision_constant(max(2, int(round(n))))
    return calculate_electron_impact_prefactor(Ne_m3, Te_ev) * (
        C + gbk_model(0.0, cutoff_frequency_ev(Ne_m3, Te_ev, B), Te_ev, Z, n=n))


def scalar_width_ev(detuning_ev, r2_mean, Ne_m3, Te_ev, B, n, Z):
    """Scalar impact width W0 <r^2> [C_n + G(detuning, omega_c0)] [eV] (frequency-dependent form)."""
    C = _strong_collision_constant(max(2, int(round(n))))
    return calculate_electron_impact_prefactor(Ne_m3, Te_ev) * r2_mean * (
        C + gbk_model(detuning_ev, cutoff_frequency_ev(Ne_m3, Te_ev, B), Te_ev, Z, n=n))


class ManifoldCollisionData:
    """Cartesian dipoles of one manifold, extended by its perturbing levels.

    r : three (dim_ext, dim_ext) Hermitian matrices (x, y, z) [a0]; the first
        ``dim`` states are the manifold, the rest the perturbers.

    energies_ev : field-free level energy of each extended state.
    """

    def __init__(self, r, energies_ev, dim):
        self.r = [np.asarray(c, dtype=complex) for c in r]
        self.energies_ev = np.asarray(energies_ev, dtype=float)
        self.dim = int(dim)
        m = self.dim
        self.r2 = sum(c[:m] @ c[:, :m] for c in self.r)          # unweighted closure sum_c r_c r_c
        self.r2_mean = float(np.real(np.trace(self.r2))) / m if m else 0.0

    def self_term(self, W):
        m = self.dim
        return 0.5 * sum((c * W)[:m] @ c[:, :m] + c[:m] @ (W * c)[:, :m] for c in self.r)

    def blocks(self, W):
        """(r_c, r_c o W) restricted to the manifold, for c = x, y, z."""
        m = self.dim
        return [c[:m, :m] for c in self.r], [(c * W)[:m, :m] for c in self.r]

    def r2_scaling(self, vectors):
        """<k|sum_c r_c r_c|k> / <r^2>_mean for the dressed states (columns of vectors)."""
        if not self.r2_mean:
            return np.ones(vectors.shape[1])
        V = np.asarray(vectors, dtype=complex)
        return np.real(np.einsum("ik,ij,jk->k", V.conj(), self.r2, V)) / self.r2_mean


def build_level_collision_operator(vectors_u, vectors_l, upper, lower, Ne_m3, Te_ev, B, n, Z,
                                   electron_interference=True):
    """Phi [eV] in the coherence basis of the dressed states (columns of vectors_u/l)."""
    Vu = np.asarray(vectors_u, dtype=complex)
    Vl = np.asarray(vectors_l, dtype=complex)
    W_u = impact_factors(upper.energies_ev, Ne_m3, Te_ev, B, n, Z)
    W_l = impact_factors(lower.energies_ev, Ne_m3, Te_ev, B, n, Z)
    A_u = Vu.conj().T @ upper.self_term(W_u) @ Vu
    A_l = Vl.conj().T @ lower.self_term(W_l) @ Vl
    du, dl = Vu.shape[0], Vl.shape[0]
    phi = np.kron(A_u, np.eye(dl)) + np.kron(np.eye(du), A_l.T)
    if electron_interference:
        r_u, rw_u = upper.blocks(W_u)
        r_l, rw_l = lower.blocks(W_l)
        rot = lambda V, ms: np.asarray([V.conj().T @ m @ V for m in ms])     # noqa: E731
        Ru, RWu, Rl, RWl = rot(Vu, r_u), rot(Vu, rw_u), rot(Vl, r_l), rot(Vl, rw_l)
        cross = (np.einsum("cik,cjl->ijkl", RWu, np.transpose(Rl, (0, 2, 1)), optimize=True)
                 + np.einsum("cik,cjl->ijkl", Ru, np.transpose(RWl, (0, 2, 1)), optimize=True))
        phi = phi - cross.reshape(phi.shape)
    return (phi + phi.conj().T) / 2.0
