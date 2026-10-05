"""Generate a JSON convergence report for one representative Balmer-alpha case.

This is a diagnostic, not a universal pass/fail test. Adjust the case and
sampling ranges for the intended scientific application.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starkzee.ffm import calculate_ffm_profile
from starkzee.static_profile import calculate_static_profile, line_reference_energy
from starkzee.validation import (
    compare_profiles,
    coordinate_jacobian_metrics,
    profile_metrics,
)


_SAMPLE_POINTS = (
    ("blue_wing", -0.008),
    ("blue_core", -0.002),
    ("line_center", 0.0),
    ("red_core", 0.002),
    ("red_wing", 0.008),
)


def _sample_point_metadata():
    return [
        {"label": label, "offset_ev": offset}
        for label, offset in _SAMPLE_POINTS
    ]


def _run_case(solver, energies, num_f, num_mu, max_beta, sdt_bin_tol=None,
              apply_doppler=False):
    common = dict(
        n_u=3, n_l=2, Z=1, B=1.0, Ne_m3=1e22, Te_ev=5.0,
        energies_ev=energies, num_f=num_f, num_mu=num_mu,
        max_beta=max_beta, microfield_model="holtsmark",
        quadratic_zeeman=True, fine_structure=True,
        apply_doppler=apply_doppler,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if solver == "static":
            profiles = calculate_static_profile(
                **common, frequency_dependent_width=False, A=1, Ti_ev=1.0)
        else:
            profiles = calculate_ffm_profile(
                **common, Ti_ev=1.0, A_ion=1.0,
                sdt_frequency_dependent_width=False, sdt_bin_tol=sdt_bin_tol)
    return np.asarray(profiles), sorted({str(item.message) for item in caught})


def _solver_report(solver, quick):
    e0 = line_reference_energy(3, 2, 1, 1, use_empirical_data=True)
    fine_points = 301 if quick else 801
    coarse_points = 151 if quick else 401
    reference_num_f = 8 if quick else 20
    reference_num_mu = 4 if quick else 8
    reference_max_beta = 10.0
    reference_grid = e0 + np.linspace(-0.02, 0.02, fine_points)
    sample_offsets = np.array([offset for _, offset in _SAMPLE_POINTS])
    sample_energies = e0 + sample_offsets
    reference, reference_warnings = _run_case(
        solver, reference_grid, reference_num_f, reference_num_mu,
        reference_max_beta, None)
    doppler_profile, doppler_warnings = _run_case(
        solver, reference_grid, reference_num_f, reference_num_mu,
        reference_max_beta, None, apply_doppler=True)

    variants = [
        ("field_quadrature", reference_grid,
         max(4, reference_num_f // 2), reference_num_mu, reference_max_beta, None),
        ("angle_quadrature", reference_grid,
         reference_num_f, max(2, reference_num_mu // 2), reference_max_beta, None),
        ("microfield_cutoff", reference_grid,
         reference_num_f, reference_num_mu, 6.0, None),
        ("energy_grid", e0 + np.linspace(-0.02, 0.02, coarse_points),
         reference_num_f, reference_num_mu, reference_max_beta, None),
        ("energy_window", e0 + np.linspace(-0.01, 0.01, coarse_points),
         reference_num_f, reference_num_mu, reference_max_beta, None),
    ]
    if solver == "ffm":
        variants.extend([
            ("sdt_bin_1e-4", reference_grid, reference_num_f,
             reference_num_mu, reference_max_beta, 1e-4),
            ("sdt_bin_1e-5", reference_grid, reference_num_f,
             reference_num_mu, reference_max_beta, 1e-5),
        ])

    comparisons = {}
    for name, grid, num_f, num_mu, max_beta, bin_tol in variants:
        candidate, caught = _run_case(
            solver, grid, num_f, num_mu, max_beta, bin_tol)
        comparisons[name] = {
            "settings": {
                "points": len(grid), "half_window_ev": float((grid[-1] - grid[0]) / 2),
                "num_f": num_f, "num_mu": num_mu, "max_beta": max_beta,
                "sdt_bin_tol": bin_tol, "apply_doppler": False,
            },
            "warnings": caught,
            "comparison": compare_profiles(
                reference_grid, reference, grid, candidate,
                sample_energies_ev=sample_energies),
        }
    return {
        "reference_settings": {
            "points": fine_points, "half_window_ev": 0.02,
            "num_f": reference_num_f, "num_mu": reference_num_mu,
            "max_beta": reference_max_beta, "sdt_bin_tol": None,
            "apply_doppler": False,
        },
        "reference_warnings": reference_warnings,
        "reference_metrics": profile_metrics(reference_grid, reference),
        "sample_points": _sample_point_metadata(),
        "reference_coordinate_jacobians": coordinate_jacobian_metrics(
            reference_grid, reference),
        "solver_doppler_effect": {
            "scope": (
                "supported solver Doppler path versus the intrinsic profile; "
                "this is a physical sensitivity comparison, not a numerical "
                "convergence error or an alternative inner-operator placement"
            ),
            "settings": {
                "points": fine_points, "half_window_ev": 0.02,
                "num_f": reference_num_f, "num_mu": reference_num_mu,
                "max_beta": reference_max_beta, "sdt_bin_tol": None,
                "apply_doppler": True,
            },
            "warnings": doppler_warnings,
            "coordinate_jacobians": coordinate_jacobian_metrics(
                reference_grid, doppler_profile),
            "comparison": compare_profiles(
                reference_grid, reference, reference_grid, doppler_profile,
                sample_energies_ev=sample_energies),
        },
        "comparisons": comparisons,
    }


def _matched_solver_report(quick, apply_doppler):
    """Compare static and FFM using the same exposed numerical controls."""
    e0 = line_reference_energy(3, 2, 1, 1, use_empirical_data=True)
    points = 301 if quick else 801
    num_f = 8 if quick else 20
    num_mu = 4 if quick else 8
    max_beta = 10.0
    grid = e0 + np.linspace(-0.02, 0.02, points)
    sample_offsets = np.array([offset for _, offset in _SAMPLE_POINTS])
    static, static_warnings = _run_case(
        "static", grid, num_f, num_mu, max_beta,
        apply_doppler=apply_doppler)
    ffm, ffm_warnings = _run_case(
        "ffm", grid, num_f, num_mu, max_beta,
        apply_doppler=apply_doppler)
    return {
        "scope": (
            "matched exposed controls; differences include the physical FFM "
            "ion-dynamics transformation and are not convergence error"
        ),
        "settings": {
            "points": points, "half_window_ev": 0.02,
            "num_f": num_f, "num_mu": num_mu, "max_beta": max_beta,
            "sdt_bin_tol": None, "apply_doppler": apply_doppler,
            "sample_points": _sample_point_metadata(),
        },
        "static_warnings": static_warnings,
        "ffm_warnings": ffm_warnings,
        "comparison": compare_profiles(
            grid, static, grid, ffm,
            sample_energies_ev=e0 + sample_offsets),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("static", "ffm", "both"), default="both")
    parser.add_argument("--full", action="store_true",
                        help="Use a denser, slower reference matrix.")
    parser.add_argument("--output", type=Path,
                        help="Write JSON to this path instead of stdout.")
    args = parser.parse_args()
    solvers = ("static", "ffm") if args.solver == "both" else (args.solver,)
    report = {
        "purpose": (
            "numerical-convergence and explicitly labeled physical-sensitivity "
            "deltas; no universal pass threshold"
        ),
        "case": {"transition": "H-alpha 3->2", "B_T": 1.0,
                 "Ne_m-3": 1e22, "Te_eV": 5.0, "Ti_eV": 1.0,
                 "microfield_model": "holtsmark", "use_empirical_data": True,
                 "atom": "H"},
        "doppler_placement_scope": (
            "Static uses its supported in-solver Voigt/FFT path; FFM applies "
            "Doppler after the nonlinear FFM transform. No ZEST-style "
            "inner-J placement is implemented or claimed here."
        ),
        "solvers": {solver: _solver_report(solver, not args.full) for solver in solvers},
    }
    if solvers == ("static", "ffm"):
        report["matched_static_ffm"] = {
            "intrinsic": _matched_solver_report(not args.full, False),
            "solver_doppler_enabled": _matched_solver_report(
                not args.full, True),
        }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()

