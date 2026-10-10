"""Opt-in DC Stark-map calculations for hydrogenic atoms.

The map solver is intentionally separate from the static and FFM profile
solvers. Importing or using :class:`~starkzee.line_profile.LineProfile` does
not perform any Stark-map work.
"""

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
from scipy.constants import e as _E_CHARGE, h as _PLANCK

from starkzee.multielectron import wigner_3j
from starkzee.multishell import (
    build_multishell_basis,
    build_multishell_hamiltonian,
    build_multishell_stark_matrix,
)


__all__ = ["StarkMap", "StarkMapResult"]


def _field_array(fields_v_m):
    fields = np.asarray(fields_v_m, dtype=float)
    if fields.ndim != 1 or fields.size == 0:
        raise ValueError("fields_v_m must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(fields)) or np.any(fields < 0.0):
        raise ValueError("electric-field magnitudes must be finite and non-negative")
    return fields


def _coupled_target_vector(basis, target_state):
    """Return ``|n,l,j,mj>`` in the uncoupled ``|n,l,ml,ms>`` basis."""
    if not isinstance(target_state, (tuple, list)) or len(target_state) != 4:
        raise ValueError("target_state must be (n, l, j, mj)")
    n, l, j, mj = target_state
    if (not isinstance(n, (int, np.integer))
            or not isinstance(l, (int, np.integer))
            or n < 1 or not 0 <= l < n):
        raise ValueError("target_state requires integer n >= 1 and 0 <= l < n")
    n, l, j, mj = int(n), int(l), float(j), float(mj)
    if (not np.isfinite(j) or not np.isfinite(mj)
            or not math.isclose(j, l - 0.5, abs_tol=1e-12)
            and not math.isclose(j, l + 0.5, abs_tol=1e-12)):
        raise ValueError("target j must equal l-1/2 or l+1/2")
    if l == 0 and not math.isclose(j, 0.5, abs_tol=1e-12):
        raise ValueError("an l=0 target must have j=1/2")
    if abs(mj) > j + 1e-12 or not math.isclose(2.0 * mj, round(2.0 * mj), abs_tol=1e-12):
        raise ValueError("target mj must be an allowed half-integer projection")

    vector = np.zeros(len(basis), dtype=complex)
    phase_exponent = l - 0.5 + mj
    if not math.isclose(phase_exponent, round(phase_exponent), abs_tol=1e-12):
        raise ValueError("target quantum numbers do not define a coupled state")
    phase = (-1.0) ** int(round(phase_exponent))
    for index, state in enumerate(basis):
        if state.n != n or state.l != l:
            continue
        if not math.isclose(state.ml + state.ms, mj, abs_tol=1e-12):
            continue
        vector[index] = (
            phase * math.sqrt(2.0 * j + 1.0)
            * wigner_3j(l, 0.5, j, state.ml, state.ms, -mj)
        )
    norm = np.linalg.norm(vector)
    if norm < 1e-14:
        raise ValueError("target_state is absent from the selected basis")
    return vector / norm, (n, l, j, mj)


@dataclass
class StarkMapResult:
    """Eigensystems sampled over a one-dimensional electric-field sweep."""

    fields_v_m: np.ndarray
    energies_ev: np.ndarray
    zero_field_energies_ev: np.ndarray
    highlights: np.ndarray
    reference_energy_ev: float
    shells: tuple
    basis: tuple
    angle_deg: float
    B: float
    Z: int
    A: int
    target_state: Optional[tuple]
    mj: Optional[float]

    @property
    def dominant_target_indices(self):
        """Index carrying the largest target-state weight at each field."""
        if self.target_state is None:
            raise ValueError("this map has no target_state")
        return np.argmax(self.highlights, axis=1)

    @property
    def dominant_target_energy_ev(self):
        """Energy carrying the largest target-state weight at each field."""
        indices = self.dominant_target_indices
        return self.energies_ev[np.arange(len(indices)), indices]

    def shifted_energies(self, unit="GHz", reference="target_zero_field"):
        """Return map energies in the requested unit and reference convention."""
        if reference == "target_zero_field":
            if self.target_state is None:
                raise ValueError("target_zero_field requires a target_state")
            reference_ev = self.reference_energy_ev
        elif reference == "ionization":
            reference_ev = 0.0
        elif reference == "lowest_zero_field":
            reference_ev = float(np.min(self.zero_field_energies_ev))
        elif isinstance(reference, (int, float, np.integer, np.floating)):
            reference_ev = float(reference)
        else:
            raise ValueError(
                "reference must be 'target_zero_field', 'ionization', "
                "'lowest_zero_field', or an energy in eV")

        shifted = self.energies_ev - reference_ev
        normalized_unit = str(unit).lower().replace(" ", "")
        if normalized_unit == "ev":
            return shifted
        if normalized_unit == "mev":
            return shifted * 1.0e3
        if normalized_unit == "ghz":
            return shifted * _E_CHARGE / _PLANCK / 1.0e9
        if normalized_unit in ("cm^-1", "cm-1", "wavenumber"):
            return shifted * _E_CHARGE / (_PLANCK * 299792458.0 * 100.0)
        raise ValueError("unit must be 'eV', 'meV', 'GHz', or 'cm^-1'")

    def plot(self, unit="GHz", reference="target_zero_field",
             field_unit="V/cm", ax=None, colorbar=True,
             point_size=5.0, cmap=None):
        """Plot an ARC-style Stark map, colored by target-state overlap."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, Normalize

        if ax is None:
            _, ax = plt.subplots(figsize=(10.5, 6.0))
        fig = ax.figure

        normalized_field_unit = str(field_unit).lower().replace(" ", "")
        if normalized_field_unit in ("v/cm", "vcm"):
            x = self.fields_v_m / 100.0
            x_label = "Electric field (V/cm)"
        elif normalized_field_unit in ("v/m", "vm"):
            x = self.fields_v_m
            x_label = "Electric field (V/m)"
        else:
            raise ValueError("field_unit must be 'V/cm' or 'V/m'")

        y = self.shifted_energies(unit=unit, reference=reference)
        x_flat = np.repeat(x, y.shape[1])
        y_flat = y.ravel()
        weights = self.highlights.ravel()
        order = np.argsort(weights, kind="stable")
        if cmap is None:
            cmap = LinearSegmentedColormap.from_list(
                "starkzee_target", ["0.82", "#d73027", "black"])
        scatter = ax.scatter(
            x_flat[order], y_flat[order], c=weights[order],
            s=point_size, cmap=cmap, norm=Normalize(0.0, 1.0),
            linewidths=0.0, rasterized=True,
        )
        ax.set_xlabel(x_label)
        unit_label = {"ghz": "GHz", "ev": "eV", "mev": "meV"}.get(
            str(unit).lower(), r"cm$^{-1}$")
        if reference == "ionization":
            ax.set_ylabel(f"Energy relative to ionization ({unit_label})")
        else:
            ax.set_ylabel(f"Energy shift ({unit_label})")
        ax.grid(alpha=0.2)
        if colorbar:
            label = "Target-state overlap" if self.target_state is None else (
                r"$|\langle nljm_j|\psi\rangle|^2$")
            fig.colorbar(scatter, ax=ax, label=label, pad=0.02)
        return fig, ax


class StarkMap:
    """Compute DC Stark maps using a truncated multi-shell hydrogenic basis.

    Parameters
    ----------
    shells : iterable of int
        Principal shells retained in the configuration-interaction basis.
    Z, A : int, optional
        Nuclear charge and emitter mass number.
    B : float, optional
        Magnetic-field magnitude [T], defining the z-axis.
    max_l : int, optional
        Maximum orbital angular momentum retained, useful for matching an ARC
        basis. By default every allowed l in each shell is included.
    """

    def __init__(self, shells, Z=1, A=1, B=0.0, max_l=None,
                 quadratic_zeeman=True, fine_structure=True):
        self.shells = tuple(shells)
        self.Z = Z
        self.A = A
        self.B = float(B)
        self.max_l = max_l
        self.quadratic_zeeman = bool(quadratic_zeeman)
        self.fine_structure = bool(fine_structure)
        if not np.isfinite(self.B) or self.B < 0.0:
            raise ValueError("B must be a finite, non-negative magnitude")
        if max_l is not None and (
                not isinstance(max_l, (int, np.integer)) or max_l < 0):
            raise ValueError("max_l must be a non-negative integer or None")

        self._basis_full = tuple(build_multishell_basis(self.shells))
        self.shells = tuple(sorted({state.n for state in self._basis_full}))
        self._h0_full = build_multishell_hamiltonian(
            self.shells, self.Z, self.B,
            Fz=0.0, Fx=0.0,
            quadratic_zeeman=self.quadratic_zeeman,
            fine_structure=self.fine_structure,
            A=self.A,
        )
        self._mz_full = build_multishell_stark_matrix(
            self.shells, self.Z, Fz=1.0, Fx=0.0)
        self._mx_full = build_multishell_stark_matrix(
            self.shells, self.Z, Fz=0.0, Fx=1.0)

    def compute(self, fields_v_m, angle_deg=0.0, target_state=None, mj=None):
        """Diagonalize the Hamiltonian over electric-field magnitudes.

        ``angle_deg`` is the angle between E and B. A fixed-``mj`` symmetry
        sector can be selected only for a parallel or antiparallel field.
        When omitted for a parallel field, ``mj`` is inferred from the target.
        """
        fields = _field_array(fields_v_m)
        angle_deg = float(angle_deg)
        if not np.isfinite(angle_deg):
            raise ValueError("angle_deg must be finite")
        angle = np.deg2rad(angle_deg)
        parallel = abs(np.sin(angle)) < 1e-12

        target_full = None
        normalized_target = None
        if target_state is not None:
            target_full, normalized_target = _coupled_target_vector(
                self._basis_full, target_state)
            if mj is None and parallel:
                mj = normalized_target[3]
        if mj is not None:
            mj = float(mj)
            if not np.isfinite(mj):
                raise ValueError("mj must be finite")
            if not parallel:
                raise ValueError("a fixed-mj sector requires E parallel to B")
            if (normalized_target is not None
                    and not math.isclose(mj, normalized_target[3], abs_tol=1e-12)):
                raise ValueError("mj must match the target-state projection")

        selected = np.array([
            (self.max_l is None or state.l <= self.max_l)
            and (mj is None or math.isclose(
                state.ml + state.ms, mj, abs_tol=1e-12))
            for state in self._basis_full
        ], dtype=bool)
        indices = np.flatnonzero(selected)
        if indices.size == 0:
            raise ValueError("the selected max_l and mj sector is empty")
        basis = tuple(self._basis_full[index] for index in indices)
        h0 = self._h0_full[np.ix_(indices, indices)]
        field_template = (
            np.cos(angle) * self._mz_full[np.ix_(indices, indices)]
            + np.sin(angle) * self._mx_full[np.ix_(indices, indices)]
        )

        target = None
        if target_full is not None:
            target = target_full[indices]
            target_norm = np.linalg.norm(target)
            if target_norm < 1e-14:
                raise ValueError("target_state is absent from the selected sector")
            target = target / target_norm

        # A common scalar shift improves resolution of small Stark splittings
        # without changing any eigenvector or relative energy.
        center_ev = float(np.trace(h0).real / len(h0))
        centered_h0 = h0 - center_ev * np.eye(len(h0))
        energies = np.empty((len(fields), len(h0)), dtype=float)
        highlights = np.zeros_like(energies)
        zero_values, zero_vectors = np.linalg.eigh(centered_h0)
        zero_field_energies = zero_values + center_ev
        for row, field in enumerate(fields):
            if field == 0.0:
                values, vectors = zero_values, zero_vectors
            else:
                values, vectors = np.linalg.eigh(
                    centered_h0 + field * field_template)
            energies[row] = values + center_ev
            if target is not None:
                highlights[row] = np.abs(vectors.conj().T @ target)**2

        if target is None:
            reference_energy_ev = float(np.min(zero_field_energies))
        else:
            zero_weights = np.abs(zero_vectors.conj().T @ target)**2
            reference_energy_ev = float(
                zero_values[int(np.argmax(zero_weights))] + center_ev)

        return StarkMapResult(
            fields_v_m=fields.copy(),
            energies_ev=energies,
            zero_field_energies_ev=zero_field_energies,
            highlights=highlights,
            reference_energy_ev=reference_energy_ev,
            shells=self.shells,
            basis=basis,
            angle_deg=angle_deg,
            B=self.B,
            Z=self.Z,
            A=self.A,
            target_state=normalized_target,
            mj=mj,
        )
