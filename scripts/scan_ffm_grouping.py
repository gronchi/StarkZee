"""Scan experimental complex-SDT grouping tolerances for one FFM case.

Every tolerance is independently validated against the ungrouped static
resolvent on the requested energy grid.  Failed cases are recorded rather than
silently replaced by the modulus-probability closure.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starkzee.ffm import calculate_ffm_profile
from starkzee.static_profile import line_reference_energy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upper-n", type=int, default=3)
    parser.add_argument("--magnetic-field", type=float, default=3.0)
    parser.add_argument("--density", type=float, default=1e20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num-f", type=int, default=2)
    parser.add_argument("--num-mu", type=int, default=2)
    parser.add_argument("--max-beta", type=float, default=10.0)
    parser.add_argument("--half-window-ev", type=float, default=0.02)
    parser.add_argument("--energy-points", type=int, default=201)
    parser.add_argument("--profile-rtol", type=float, default=1e-3)
    parser.add_argument(
        "--tolerance", nargs="+", type=float,
        default=[1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6,
                 1e-5, 3e-5, 1e-4, 3e-4])
    parser.add_argument("--width-tolerance", type=float)
    parser.add_argument("--output", type=Path,
                        help="write JSON to this path instead of stdout")
    args = parser.parse_args()

    if args.energy_points < 2:
        parser.error("--energy-points must be at least 2")
    center_ev = line_reference_energy(
        args.upper_n, 2, 1, 2.0, use_empirical_data=True, atom="D")
    energies_ev = center_ev + np.linspace(
        -args.half_window_ev, args.half_window_ev, args.energy_points)

    common = dict(
        n_u=args.upper_n, n_l=2, Z=1, B=args.magnetic_field,
        Ne_m3=args.density, Te_ev=args.temperature,
        Ti_ev=args.temperature, A_ion=2.0, energies_ev=energies_ev,
        num_f=args.num_f, num_mu=args.num_mu, max_beta=args.max_beta,
        use_empirical_data=True, atom="D", microfield_model="potekhin",
        electron_model="pppb-intra", electron_interference=True,
        sdt_frequency_dependent_width=False, sdt_bin_tol=None,
        numerical_inversion=False, apply_doppler=False,
        interference_group_width_tolerance_ev=args.width_tolerance,
        interference_group_profile_rtol=args.profile_rtol,
    )
    results = []
    for tolerance in args.tolerance:
        diagnostics = {}
        started = time.perf_counter()
        try:
            calculate_ffm_profile(
                **common, interference_diagnostics=diagnostics,
                interference_group_tolerance_ev=tolerance)
        except ValueError as exc:
            results.append({
                "tolerance_eV": tolerance,
                "accepted": False,
                "reason": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
            })
        else:
            results.append({
                "tolerance_eV": tolerance,
                "accepted": True,
                "polarizations": {
                    str(q): diagnostics[q]["grouping"] for q in (0, 1, -1)
                },
                "elapsed_seconds": time.perf_counter() - started,
            })

    report = {
        "purpose": "validated experimental complex-SDT grouping scan",
        "case": {
            "transition": f"D {args.upper_n}->2",
            "B_T": args.magnetic_field,
            "Ne_m-3": args.density,
            "Te_eV": args.temperature,
            "Ti_eV": args.temperature,
            "num_f": args.num_f,
            "num_mu": args.num_mu,
            "max_beta": args.max_beta,
            "half_window_eV": args.half_window_ev,
            "energy_points": args.energy_points,
            "profile_rtol": args.profile_rtol,
            "width_tolerance_eV": args.width_tolerance,
        },
        "results": results,
    }
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
