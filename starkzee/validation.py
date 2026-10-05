"""Numerical convergence metrics for StarkZee profile calculations.

These helpers compare sampled profiles; they do not supply universal physical
acceptance tolerances. Callers must choose tolerances for their transition,
plasma regime, spectral window, and observable.
"""

import numpy as np
from scipy.integrate import trapezoid

from starkzee.utils import (
    energy_ev_to_frequency_thz,
    energy_ev_to_wavelength_nm,
    energy_ev_to_wavenumber_cm,
)


def _canonical_profile(energies_ev, profiles):
    energies = np.asarray(energies_ev, dtype=float)
    values = np.asarray(profiles, dtype=float)
    if energies.ndim != 1 or energies.size < 2:
        raise ValueError("energies_ev must be a one-dimensional grid with at least two points.")
    if values.shape != (3, energies.size):
        raise ValueError("profiles must contain pi, sigma-plus, and sigma-minus arrays.")
    if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(values)):
        raise ValueError("energies and profiles must be finite.")
    delta = np.diff(energies)
    if np.all(delta < 0):
        return energies[::-1], values[:, ::-1]
    if not np.all(delta > 0):
        raise ValueError("energies_ev must be strictly monotonic.")
    return energies, values


def profile_metrics(energies_ev, profiles):
    """Return area, centroid, peak, and polarization areas for one profile set."""
    energies, values = _canonical_profile(energies_ev, profiles)
    component_areas = trapezoid(values, energies, axis=1)
    total = np.sum(values, axis=0)
    area = float(trapezoid(total, energies))
    absolute_area = float(trapezoid(np.abs(total), energies))
    if absolute_area == 0.0:
        centroid = None
        peak = None
    else:
        centroid = float(trapezoid(energies * np.abs(total), energies) / absolute_area)
        peak = float(energies[np.argmax(np.abs(total))])
    return {
        "area_total": area,
        "absolute_area_total": absolute_area,
        "centroid_ev": centroid,
        "peak_ev": peak,
        "component_areas": {
            "pi": float(component_areas[0]),
            "sigma_plus": float(component_areas[1]),
            "sigma_minus": float(component_areas[2]),
        },
    }


def coordinate_jacobian_metrics(energies_ev, profiles):
    """Check integrated strength after energy-density coordinate transforms.

    The input components are densities per eV. They are transformed to vacuum
    wavelength (per nm), frequency (per THz), and wavenumber (per cm^-1) using
    the absolute Jacobian ``dE/dx``. Reported integrals use increasing
    coordinates even though wavelength decreases as photon energy increases.
    Differences are finite-grid quadrature errors, not physical-model errors.
    """
    energies, values = _canonical_profile(energies_ev, profiles)
    if np.any(energies <= 0.0):
        raise ValueError("coordinate Jacobians require positive photon energies.")

    energy_component_areas = trapezoid(values, energies, axis=1)
    energy_total_area = float(np.sum(energy_component_areas))
    result = {
        "energy_ev": {
            "coordinate_order_from_energy": "increasing",
            "area_total": energy_total_area,
            "area_relative_error": 0.0,
            "component_areas": [float(area) for area in energy_component_areas],
            "component_relative_errors": [0.0, 0.0, 0.0],
        }
    }
    conversions = {
        "wavelength_nm": energy_ev_to_wavelength_nm,
        "frequency_thz": energy_ev_to_frequency_thz,
        "wavenumber_cm": energy_ev_to_wavenumber_cm,
    }
    for name, convert in conversions.items():
        coordinate = np.asarray(convert(energies), dtype=float)
        density = values * energies[np.newaxis, :] / coordinate[np.newaxis, :]
        increasing = np.all(np.diff(coordinate) > 0.0)
        if not increasing:
            coordinate = coordinate[::-1]
            density = density[:, ::-1]
        component_areas = trapezoid(density, coordinate, axis=1)
        total_area = float(np.sum(component_areas))
        component_relative_errors = [
            (float((candidate - reference) / abs(reference))
             if reference != 0.0 else None)
            for candidate, reference in zip(
                component_areas, energy_component_areas, strict=True)
        ]
        result[name] = {
            "coordinate_order_from_energy": (
                "increasing" if increasing else "decreasing"
            ),
            "area_total": total_area,
            "area_relative_error": (
                float((total_area - energy_total_area) / abs(energy_total_area))
                if energy_total_area != 0.0 else None
            ),
            "component_areas": [float(area) for area in component_areas],
            "component_relative_errors": component_relative_errors,
        }
    return result


def compare_profiles(reference_energies_ev, reference_profiles,
                     candidate_energies_ev, candidate_profiles,
                     sample_energies_ev=None):
    """Compare two profile sets, interpolating the candidate on their overlap.

    Full-window area and centroid changes are reported separately from shape
    errors on the common energy interval. ``reference_overlap_fraction`` is the
    fraction of the reference absolute area contained in that interval, making
    a narrow-window comparison explicit instead of silently treating it as
    equivalent coverage. When ``sample_energies_ev`` is supplied, interpolate
    both profiles at those caller-selected core/wing locations and report total
    and component intensities and their differences. Samples must lie in the
    common interval; no extrapolation is performed.
    """
    ref_e, ref_p = _canonical_profile(reference_energies_ev, reference_profiles)
    cand_e, cand_p = _canonical_profile(candidate_energies_ev, candidate_profiles)
    ref_metrics = profile_metrics(ref_e, ref_p)
    cand_metrics = profile_metrics(cand_e, cand_p)

    overlap = (ref_e >= cand_e[0]) & (ref_e <= cand_e[-1])
    if np.count_nonzero(overlap) < 2:
        raise ValueError("reference and candidate energy grids do not overlap sufficiently.")
    overlap_e = ref_e[overlap]
    overlap_ref = ref_p[:, overlap]
    overlap_cand = np.vstack([
        np.interp(overlap_e, cand_e, component) for component in cand_p
    ])
    ref_total = np.sum(overlap_ref, axis=0)
    cand_total = np.sum(overlap_cand, axis=0)
    difference = cand_total - ref_total
    norm_l1 = float(trapezoid(np.abs(ref_total), overlap_e))
    peak_scale = float(np.max(np.abs(ref_total)))
    l1_relative = (float(trapezoid(np.abs(difference), overlap_e)) / norm_l1
                   if norm_l1 > 0 else None)
    linf_relative = (float(np.max(np.abs(difference))) / peak_scale
                     if peak_scale > 0 else None)
    ref_abs_area = ref_metrics["absolute_area_total"]
    overlap_abs_area = float(trapezoid(np.abs(ref_total), overlap_e))

    ref_area = ref_metrics["area_total"]
    area_relative = ((cand_metrics["area_total"] - ref_area) / abs(ref_area)
                     if ref_area != 0 else None)
    ref_centroid = ref_metrics["centroid_ev"]
    cand_centroid = cand_metrics["centroid_ev"]
    centroid_delta = (cand_centroid - ref_centroid
                      if ref_centroid is not None and cand_centroid is not None
                      else None)
    result = {
        "reference": ref_metrics,
        "candidate": cand_metrics,
        "area_relative_error": area_relative,
        "centroid_delta_ev": centroid_delta,
        "shape_l1_relative_on_overlap": l1_relative,
        "shape_linf_relative_on_overlap": linf_relative,
        "reference_overlap_fraction": (overlap_abs_area / ref_abs_area
                                       if ref_abs_area > 0 else None),
        "overlap_ev": [float(overlap_e[0]), float(overlap_e[-1])],
    }
    if sample_energies_ev is not None:
        samples = np.asarray(sample_energies_ev, dtype=float)
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError("sample_energies_ev must be a nonempty one-dimensional array.")
        if not np.all(np.isfinite(samples)):
            raise ValueError("sample_energies_ev must be finite.")
        overlap_min = max(ref_e[0], cand_e[0])
        overlap_max = min(ref_e[-1], cand_e[-1])
        if np.any(samples < overlap_min) or np.any(samples > overlap_max):
            raise ValueError("sample energies must lie in the profile overlap.")

        sampled = []
        for energy in samples:
            reference_components = np.array([
                np.interp(energy, ref_e, component) for component in ref_p
            ])
            candidate_components = np.array([
                np.interp(energy, cand_e, component) for component in cand_p
            ])
            reference_total = float(np.sum(reference_components))
            candidate_total = float(np.sum(candidate_components))
            sampled.append({
                "energy_ev": float(energy),
                "reference_total": reference_total,
                "candidate_total": candidate_total,
                "absolute_difference": candidate_total - reference_total,
                "relative_error": (
                    (candidate_total - reference_total) / abs(reference_total)
                    if reference_total != 0.0 else None
                ),
                "reference_components": [
                    float(value) for value in reference_components
                ],
                "candidate_components": [
                    float(value) for value in candidate_components
                ],
            })
        result["sampled_intensities"] = sampled
    return result

