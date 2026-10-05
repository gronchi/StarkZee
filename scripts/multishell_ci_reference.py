"""Generate a case-specific multi-shell CI convergence report.

The report compares every requested lower/upper cutoff pair with the largest
basis at each field angle. Tolerances are mandatory because no universal safe
CI basis or target-line window is claimed. This remains a numerical diagnostic,
not a synthetic spectrum or external physical validation.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starkzee.microfield import calculate_normal_field  # noqa: E402
from starkzee.multishell import multishell_convergence_report  # noqa: E402


def _angles(value):
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "angles must be a comma-separated numeric list") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one angle is required")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-n", type=int, default=4)
    parser.add_argument("--lower-n", type=int, default=2)
    parser.add_argument("--max-lower-padding", type=int, default=1)
    parser.add_argument("--max-upper-padding", type=int, default=2)
    parser.add_argument("--angles", type=_angles, default=(0.0, 45.0, 90.0))
    parser.add_argument("--Z", type=int, default=1)
    parser.add_argument("--B", type=float, default=500.0,
                        help="magnetic field [T]")
    parser.add_argument("--F", type=float,
                        help="electric-field magnitude [V/m]; default is F0 at --Ne")
    parser.add_argument("--Ne", type=float, default=1e23,
                        help="density used only to obtain F0 when --F is omitted [m^-3]")
    parser.add_argument("--window-ev", type=float, default=0.2,
                        help="target-line half-window [eV]")
    parser.add_argument("--energy-tol-ev", type=float, required=True,
                        help="tracked-branch absolute convergence tolerance [eV]")
    parser.add_argument("--gf-rel-tol", type=float, required=True,
                        help="target and total gf relative convergence tolerance")
    parser.add_argument("--json-output", type=Path,
                        help="optional path for the complete JSON report")
    args = parser.parse_args()
    field = args.F if args.F is not None else calculate_normal_field(args.Ne)[0]

    try:
        report = multishell_convergence_report(
            target_n=args.target_n, lower_n=args.lower_n, Z=args.Z,
            B=args.B, field=field, half_width_ev=args.window_ev,
            max_lower_padding=args.max_lower_padding,
            max_upper_padding=args.max_upper_padding,
            angles_deg=args.angles,
            energy_tolerance_ev=args.energy_tol_ev,
            gf_relative_tolerance=args.gf_rel_tol,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(report["scope"])
    print("lower upper angle dimension dE_eV gf_target_rel gf_total_rel pass")
    for row in report["rows"]:
        convergence = row["convergence"]
        print(
            f"{row['lower_padding']:5d} {row['upper_padding']:5d} "
            f"{row['angle_deg']:5.1f} {row['dimension']:9d} "
            f"{convergence['tracked_energy_abs_error_ev']:.6e} "
            f"{convergence['gf_target_relative_error']:.6e} "
            f"{convergence['gf_total_relative_error']:.6e} "
            f"{str(convergence['passed']).lower()}"
        )
    print(f"all_rows_passed={str(report['all_rows_passed']).lower()}")
    if args.json_output is not None:
        args.json_output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"json_report={args.json_output}")


if __name__ == "__main__":
    main()
