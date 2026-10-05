"""Experimental multi-shell hydrogenic configuration-interaction matrices.

This module supplies the standalone reference calculation required before
multi-shell states are threaded through the static and FFM profile solvers.
It deliberately does not change either solver's public behavior.
"""

from dataclasses import replace
from functools import lru_cache

import numpy as np
from scipy.integrate import quad

from starkzee.radiator import (
    angular_dipole_element,
    build_basis,
    build_hamiltonian,
    radial_dipole,
    radial_r2_element,
    radial_wavefunction,
)
from starkzee.utils import (
    A0, E_CHARGE, M_E, RYDBERG_EV, reduced_mass_rydberg_ev,
)


def _shell_tuple(shells):
    try:
        values = tuple(shells)
    except TypeError as exc:
        raise ValueError("shells must be a strictly increasing iterable of integers") from exc
    if (not values or any(not isinstance(n, (int, np.integer)) or n < 1
                          for n in values)
            or any(a >= b for a, b in zip(values, values[1:]))):
        raise ValueError("shells must be a strictly increasing iterable of positive integers")
    return tuple(int(n) for n in values)


def _nuclear_charge(Z):
    if not isinstance(Z, (int, np.integer)) or Z < 1:
        raise ValueError("Z must be a positive integer")
    return int(Z)


@lru_cache(maxsize=None)
def _multishell_basis_cached(shells):
    basis = []
    for n in shells:
        for state in build_basis(n):
            basis.append(replace(state, index=len(basis)))
    return tuple(basis)


def build_multishell_basis(shells):
    """Return the canonical, globally indexed basis for several shells.

    Shells must be positive, unique, and strictly increasing. Within each shell
    the ordering is exactly that of :func:`starkzee.radiator.build_basis`.
    """
    return _multishell_basis_cached(_shell_tuple(shells))


@lru_cache(maxsize=None)
def radial_r2_between_shells(n1, l1, n2, l2, Z):
    """Return signed ``<n1,l1|r^2|n2,l2>`` in squared Bohr radii.

    Same-shell elements use the existing exact formulas where available.
    Inter-shell elements use the standard positive-near-origin radial phases
    and adaptive quadrature in a coordinate scaled by ``max(n1,n2)^2/Z``.
    """
    quantum_numbers = (n1, l1, n2, l2, Z)
    if any(not isinstance(value, (int, np.integer)) for value in quantum_numbers):
        raise ValueError("n, l, and Z quantum numbers must be integers")
    if n1 < 1 or n2 < 1 or Z < 1 or not (0 <= l1 < n1) or not (0 <= l2 < n2):
        raise ValueError("require n >= 1, Z >= 1, and 0 <= l < n")
    if n1 == n2:
        return radial_r2_element(int(n1), int(l1), int(l2), int(Z))

    scale = max(n1, n2)**2 / Z
    value, _ = quad(
        lambda x: radial_wavefunction(x * scale, n1, l1, Z)
        * radial_wavefunction(x * scale, n2, l2, Z)
        * (x * scale)**4 * scale,
        0.0,
        np.inf,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=500,
    )
    return value


@lru_cache(maxsize=None)
def _multishell_stark_templates(shells, Z):
    basis = build_multishell_basis(shells)
    dim = len(basis)
    z_matrix = np.zeros((dim, dim), dtype=complex)
    x_matrix = np.zeros((dim, dim), dtype=complex)
    for i, bra in enumerate(basis):
        for j, ket in enumerate(basis):
            if bra.ms != ket.ms or abs(bra.l - ket.l) != 1:
                continue
            radial = radial_dipole(bra.n, bra.l, ket.n, ket.l, Z)
            z_angular = angular_dipole_element(
                bra.l, bra.ml, ket.l, ket.ml, 0)
            x_angular = (
                angular_dipole_element(bra.l, bra.ml, ket.l, ket.ml, -1)
                + angular_dipole_element(bra.l, bra.ml, ket.l, ket.ml, 1)
            ) / np.sqrt(2.0)
            # Electron interaction in eV: -F * <r> [m]. The signed radial
            # convention is used consistently both within and between shells.
            z_matrix[i, j] = -radial * z_angular * A0
            x_matrix[i, j] = -radial * x_angular * A0
    return z_matrix, x_matrix


def build_multishell_stark_matrix(shells, Z, Fz, Fx):
    """Return the Hermitian multi-shell linear-Stark matrix in eV."""
    shell_key = _shell_tuple(shells)
    Z = _nuclear_charge(Z)
    if not np.isfinite(Fz) or not np.isfinite(Fx):
        raise ValueError("Fz and Fx must be finite")
    z_matrix, x_matrix = _multishell_stark_templates(shell_key, Z)
    result = Fz * z_matrix + Fx * x_matrix
    if not np.allclose(result, result.conj().T, rtol=0.0, atol=1e-13):
        raise RuntimeError("constructed multi-shell Stark matrix is not Hermitian")
    return result.copy()


def _sin2_angular(l1, ml1, l2, ml2):
    if ml1 != ml2:
        return 0.0
    if l1 == l2:
        if l1 == 0:
            return 2.0 / 3.0
        cos2 = (2.0 * l1**2 + 2.0 * l1 - 1.0 - 2.0 * ml1**2) / (
            (2.0 * l1 - 1.0) * (2.0 * l1 + 3.0))
        return 1.0 - cos2
    if abs(l1 - l2) == 2:
        lower = min(l1, l2)
        numerator = (((lower + 1.0)**2 - ml1**2)
                     * ((lower + 2.0)**2 - ml1**2))
        denominator = ((2.0 * lower + 1.0) * (2.0 * lower + 3.0)**2
                       * (2.0 * lower + 5.0))
        return -np.sqrt(numerator / denominator)
    return 0.0


@lru_cache(maxsize=None)
def _multishell_diamagnetic_template(shells, Z):
    basis = build_multishell_basis(shells)
    matrix = np.zeros((len(basis), len(basis)), dtype=complex)
    for i, bra in enumerate(basis):
        for j, ket in enumerate(basis):
            if bra.ms != ket.ms:
                continue
            angular = _sin2_angular(bra.l, bra.ml, ket.l, ket.ml)
            if angular:
                matrix[i, j] = angular * radial_r2_between_shells(
                    bra.n, bra.l, ket.n, ket.l, Z)
    return matrix


def build_multishell_diamagnetic_matrix(shells, Z, B):
    """Return ``e^2 B^2 r^2 sin^2(theta)/(8 m_e)`` in eV."""
    shell_key = _shell_tuple(shells)
    Z = _nuclear_charge(Z)
    if not np.isfinite(B):
        raise ValueError("B must be finite")
    coefficient_ev = E_CHARGE * B**2 * A0**2 / (8.0 * M_E)
    result = coefficient_ev * _multishell_diamagnetic_template(shell_key, Z)
    if not np.allclose(result, result.conj().T, rtol=0.0, atol=1e-13):
        raise RuntimeError("constructed multi-shell diamagnetic matrix is not Hermitian")
    return result.copy()


def build_multishell_hamiltonian(shells, Z, B, Fz=0.0, Fx=0.0,
                                 quadratic_zeeman=True, fine_structure=True,
                                 A=1):
    """Build an experimental multi-shell hydrogenic CI Hamiltonian in eV.

    Atomic, fine-structure, and linear-Zeeman terms are assembled shell by
    shell. Linear Stark and diamagnetic terms span the full truncated basis.
    Empirical level injection is intentionally unsupported at this stage.
    """
    shell_key = _shell_tuple(shells)
    Z = _nuclear_charge(Z)
    if not np.isfinite(B):
        raise ValueError("B must be finite")
    basis = build_multishell_basis(shell_key)
    matrix = np.zeros((len(basis), len(basis)), dtype=complex)
    offset = 0
    for n in shell_key:
        block = build_hamiltonian(
            n, Z, B, quadratic_zeeman=False,
            fine_structure=fine_structure, A=A,
            use_empirical_data=False,
        )
        size = block.shape[0]
        matrix[offset:offset + size, offset:offset + size] = block
        offset += size
    matrix += build_multishell_stark_matrix(shell_key, Z, Fz, Fx)
    if quadratic_zeeman:
        matrix += build_multishell_diamagnetic_matrix(shell_key, Z, B)
    return matrix


def diagonalize_multishell_hamiltonian(*args, **kwargs):
    """Diagonalize :func:`build_multishell_hamiltonian` with ``eigh``."""
    return np.linalg.eigh(build_multishell_hamiltonian(*args, **kwargs))


@lru_cache(maxsize=None)
def _multishell_dipole_matrices_cached(upper_shells, lower_shells, Z,
                                       shell_pair):
    basis_u = build_multishell_basis(upper_shells)
    basis_l = build_multishell_basis(lower_shells)
    matrices = {q: np.zeros((len(basis_l), len(basis_u)), dtype=complex)
                for q in (0, 1, -1)}
    radial_cache = {}
    for lower in basis_l:
        for upper in basis_u:
            if (shell_pair is not None
                    and (upper.n, lower.n) != shell_pair):
                continue
            if upper.ms != lower.ms or abs(upper.l - lower.l) != 1:
                continue
            radial_key = (upper.n, upper.l, lower.n, lower.l)
            if radial_key not in radial_cache:
                radial_cache[radial_key] = radial_dipole(
                    upper.n, upper.l, lower.n, lower.l, Z)
            for q in matrices:
                angular = angular_dipole_element(
                    lower.l, lower.ml, upper.l, upper.ml, q)
                matrices[q][lower.index, upper.index] = (
                    -radial_cache[radial_key] * angular)
    return matrices


def _eigenvector_matrix(vectors, dimension, name):
    if vectors is None:
        return np.eye(dimension, dtype=complex)
    vectors = np.asarray(vectors, dtype=complex)
    if vectors.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension})")
    if not np.allclose(vectors.conj().T @ vectors, np.eye(dimension),
                       rtol=1e-11, atol=1e-12):
        raise ValueError(f"{name} must be unitary")
    return vectors


def build_multishell_dipole_matrices(upper_shells, lower_shells, Z,
                                     upper_eigenvectors=None,
                                     lower_eigenvectors=None,
                                     shell_pair=None):
    """Return E1 matrices between two truncated multi-shell spaces.

    The returned arrays have shape ``(n_lower_states, n_upper_states)``.
    Supplying eigensystem matrices rotates the operators coherently as
    ``V_lower^dagger D V_upper``. ``shell_pair=(n_upper, n_lower)`` restricts
    the source amplitude to one unmixed shell pair; omitting it includes every
    pair represented by the two spaces.
    """
    upper_key = _shell_tuple(upper_shells)
    lower_key = _shell_tuple(lower_shells)
    Z = _nuclear_charge(Z)
    if shell_pair is not None:
        if (not isinstance(shell_pair, tuple) or len(shell_pair) != 2
                or shell_pair[0] not in upper_key
                or shell_pair[1] not in lower_key):
            raise ValueError(
                "shell_pair must be an (upper, lower) tuple present in the bases")
        shell_pair = tuple(int(n) for n in shell_pair)
    uncoupled = _multishell_dipole_matrices_cached(
        upper_key, lower_key, Z, shell_pair)
    V_u = _eigenvector_matrix(
        upper_eigenvectors, len(build_multishell_basis(upper_key)),
        "upper_eigenvectors")
    V_l = _eigenvector_matrix(
        lower_eigenvectors, len(build_multishell_basis(lower_key)),
        "lower_eigenvectors")
    return {q: V_l.conj().T @ matrix @ V_u
            for q, matrix in uncoupled.items()}


def transition_energy_mask(upper_energies, lower_energies, center_ev,
                           half_width_ev):
    """Return a lower-by-upper mask for a positive transition-energy window."""
    upper = np.asarray(upper_energies, dtype=float)
    lower = np.asarray(lower_energies, dtype=float)
    if upper.ndim != 1 or lower.ndim != 1:
        raise ValueError("upper and lower energies must be one-dimensional")
    if (not np.all(np.isfinite(upper)) or not np.all(np.isfinite(lower))
            or not np.isfinite(center_ev) or not np.isfinite(half_width_ev)
            or half_width_ev < 0):
        raise ValueError("energies and window parameters must be finite")
    differences = upper[np.newaxis, :] - lower[:, np.newaxis]
    return ((differences > 0.0)
            & (np.abs(differences - center_ev) <= half_width_ev))


def _coherent_strength_breakdown(total_matrices, target_matrices, mask,
                                 weights):
    first_shape = np.asarray(total_matrices[0]).shape
    if mask is None:
        selected = np.ones(first_shape, dtype=bool)
    else:
        selected = np.asarray(mask)
        if selected.dtype != np.bool_ or selected.shape != first_shape:
            raise ValueError("mask must be Boolean with the dipole-matrix shape")
    if weights is None:
        weight = np.ones(first_shape, dtype=float)
    else:
        weight = np.asarray(weights, dtype=float)
        if (weight.shape != first_shape or not np.all(np.isfinite(weight))
                or np.any(weight < 0.0)):
            raise ValueError("weights must be finite, non-negative, and match the matrices")

    target_strength = 0.0
    neighboring_strength = 0.0
    interference = 0.0
    total_strength = 0.0
    w = weight[selected]
    for q in (0, 1, -1):
        total = np.asarray(total_matrices[q], dtype=complex)
        target = np.asarray(target_matrices[q], dtype=complex)
        if total.shape != first_shape or target.shape != first_shape:
            raise ValueError("all dipole matrices must have the same shape")
        neighboring = total - target
        t = target[selected]
        n = neighboring[selected]
        a = total[selected]
        target_strength += float(np.sum(w * np.abs(t)**2))
        neighboring_strength += float(np.sum(w * np.abs(n)**2))
        interference += float(np.sum(w * 2.0 * np.real(np.conj(t) * n)))
        total_strength += float(np.sum(w * np.abs(a)**2))
    return {
        "target": target_strength,
        "neighboring": neighboring_strength,
        "interference": interference,
        "total": total_strength,
        "closure_error": total_strength - (
            target_strength + neighboring_strength + interference),
    }


def transition_strength_breakdown(total_matrices, target_matrices, mask=None):
    """Split coherent E1 strength into target, neighboring, and interference.

    ``target_matrices`` identify one source shell pair after the same basis
    rotations as ``total_matrices``. The neighboring amplitude is their
    difference. The returned dictionary satisfies
    ``total = target + neighboring + interference`` up to roundoff, including
    when a lower-by-upper Boolean spectral-window mask is supplied.
    """
    if set(total_matrices) != {0, 1, -1} or set(target_matrices) != {0, 1, -1}:
        raise ValueError("dipole mappings must contain q = 0, +1, and -1")
    return _coherent_strength_breakdown(
        total_matrices, target_matrices, mask, weights=None)


def oscillator_strength_breakdown(total_matrices, target_matrices,
                                  upper_energies, lower_energies, mask=None):
    """Return a coherent weighted-absorption-oscillator-strength breakdown.

    Every positive transition is weighted by
    ``(2/3) * (E_upper-E_lower)/E_hartree``. The four strength fields use the
    same dimensionless ``gf`` convention as :func:`starkzee.radiator.oscillator_strength`.
    Negative-energy matrix elements receive zero weight. Target, neighboring,
    and interference terms therefore reconstruct the total in any selected
    spectral window without treating coherent channels as independent lines.
    """
    if set(total_matrices) != {0, 1, -1} or set(target_matrices) != {0, 1, -1}:
        raise ValueError("dipole mappings must contain q = 0, +1, and -1")
    upper = np.asarray(upper_energies, dtype=float)
    lower = np.asarray(lower_energies, dtype=float)
    shape = np.asarray(total_matrices[0]).shape
    if (upper.ndim != 1 or lower.ndim != 1
            or shape != (len(lower), len(upper))
            or not np.all(np.isfinite(upper))
            or not np.all(np.isfinite(lower))):
        raise ValueError("energy arrays must be finite and match the dipole dimensions")
    delta = upper[np.newaxis, :] - lower[:, np.newaxis]
    weights = (2.0 / 3.0) * np.maximum(delta, 0.0) / (2.0 * RYDBERG_EV)
    return _coherent_strength_breakdown(
        total_matrices, target_matrices, mask, weights=weights)


def multishell_transition_diagnostics(shells, target_n, lower_n, Z, B,
                                      field, angle_deg, half_width_ev):
    """Evaluate one disjoint-upper-space CI transition diagnostic.

    The lower radiating shell is diagonalized separately and is forbidden from
    the upper CI space. This avoids duplicating one physical shell in both
    manifolds while production integration remains unimplemented.
    """
    shell_key = _shell_tuple(shells)
    if target_n not in shell_key or lower_n in shell_key or target_n <= lower_n:
        raise ValueError(
            "require target_n in shells, target_n > lower_n, and lower_n not in shells")
    if (not np.isfinite(field) or field < 0.0
            or not np.isfinite(angle_deg) or not np.isfinite(half_width_ev)
            or half_width_ev <= 0.0):
        raise ValueError("field and window must be positive and angle must be finite")
    angle = np.deg2rad(angle_deg)
    kwargs = dict(
        Z=Z, B=B, Fz=field * np.cos(angle), Fx=field * np.sin(angle),
        quadratic_zeeman=True, fine_structure=True,
    )
    upper_energies, V_u = diagonalize_multishell_hamiltonian(shell_key, **kwargs)
    lower_energies, V_l = diagonalize_multishell_hamiltonian((lower_n,), **kwargs)
    basis = build_multishell_basis(shell_key)
    target_index = next(
        state.index for state in basis
        if (state.n, state.l, state.ml, state.ms) == (target_n, 0, 0, 0.5)
    )
    overlaps = np.abs(V_u[target_index, :])**2
    branch = int(np.argmax(overlaps))

    total = build_multishell_dipole_matrices(
        shell_key, (lower_n,), Z, V_u, V_l)
    target = build_multishell_dipole_matrices(
        shell_key, (lower_n,), Z, V_u, V_l,
        shell_pair=(target_n, lower_n))
    center = Z**2 * reduced_mass_rydberg_ev(Z, 1) * (
        1 / lower_n**2 - 1 / target_n**2)
    mask = transition_energy_mask(
        upper_energies, lower_energies, center, half_width_ev)
    return {
        "shells": list(shell_key),
        "dimension": len(basis),
        "angle_deg": float(angle_deg),
        "tracked_energy_ev": float(upper_energies[branch]),
        "tracked_overlap": float(overlaps[branch]),
        "selected_transition_count": int(np.count_nonzero(mask)),
        "line_strength": transition_strength_breakdown(total, target, mask),
        "oscillator_strength": oscillator_strength_breakdown(
            total, target, upper_energies, lower_energies, mask),
    }


def multishell_convergence_report(target_n, lower_n, Z, B, field,
                                  half_width_ev, max_lower_padding,
                                  max_upper_padding, angles_deg,
                                  energy_tolerance_ev,
                                  gf_relative_tolerance):
    """Compare cutoff pairs with the largest requested basis at each angle.

    Tolerances are mandatory and case-specific. A row passes only when its
    tracked energy and both target and total windowed ``gf`` agree with the
    maximal-cutoff reference at the requested levels. This is a numerical
    truncation report, not evidence of physical accuracy.
    """
    integer_args = (target_n, lower_n, max_lower_padding, max_upper_padding)
    if any(not isinstance(value, (int, np.integer)) for value in integer_args):
        raise ValueError("shell and padding arguments must be integers")
    if (target_n <= lower_n or lower_n < 1 or max_lower_padding < 0
            or max_upper_padding < 0):
        raise ValueError("require target_n > lower_n >= 1 and non-negative padding")
    angles = tuple(float(angle) for angle in angles_deg)
    if not angles or not np.all(np.isfinite(angles)):
        raise ValueError("angles_deg must contain finite values")
    if (not np.isfinite(energy_tolerance_ev) or energy_tolerance_ev <= 0.0
            or not np.isfinite(gf_relative_tolerance)
            or gf_relative_tolerance <= 0.0):
        raise ValueError("convergence tolerances must be finite and positive")

    effective_lower_max = min(max_lower_padding, target_n - lower_n - 1)
    rows = []
    for lower_padding in range(effective_lower_max + 1):
        n_min = target_n - lower_padding
        for upper_padding in range(max_upper_padding + 1):
            shells = tuple(range(n_min, target_n + upper_padding + 1))
            for angle in angles:
                row = multishell_transition_diagnostics(
                    shells, target_n, lower_n, Z, B, field, angle,
                    half_width_ev)
                row.update({
                    "lower_padding": lower_padding,
                    "upper_padding": upper_padding,
                })
                rows.append(row)

    references = {
        row["angle_deg"]: row for row in rows
        if (row["lower_padding"] == effective_lower_max
            and row["upper_padding"] == max_upper_padding)
    }
    for row in rows:
        reference = references[row["angle_deg"]]
        energy_error = abs(
            row["tracked_energy_ev"] - reference["tracked_energy_ev"])
        gf_errors = {}
        for key in ("target", "total"):
            value = row["oscillator_strength"][key]
            expected = reference["oscillator_strength"][key]
            gf_errors[key] = (abs(value - expected) / abs(expected)
                              if expected != 0.0 else abs(value - expected))
        row["convergence"] = {
            "tracked_energy_abs_error_ev": energy_error,
            "gf_target_relative_error": gf_errors["target"],
            "gf_total_relative_error": gf_errors["total"],
            "passed": bool(
                energy_error <= energy_tolerance_ev
                and gf_errors["target"] <= gf_relative_tolerance
                and gf_errors["total"] <= gf_relative_tolerance),
        }
    return {
        "scope": (
            "case-specific numerical cutoff convergence; not physical or "
            "external validation, and not a universal safe basis"
        ),
        "case": {
            "target_n": int(target_n), "lower_n": int(lower_n), "Z": int(Z),
            "B_T": float(B), "F_V_per_m": float(field),
            "half_window_ev": float(half_width_ev),
            "angles_deg": list(angles),
            "requested_max_lower_padding": int(max_lower_padding),
            "effective_max_lower_padding": int(effective_lower_max),
            "max_upper_padding": int(max_upper_padding),
        },
        "tolerances": {
            "tracked_energy_abs_ev": float(energy_tolerance_ev),
            "gf_target_and_total_relative": float(gf_relative_tolerance),
        },
        "all_rows_passed": all(row["convergence"]["passed"] for row in rows),
        "rows": rows,
    }
