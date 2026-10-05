"""Map signed PPP residues used by the modulus-probability FFM closure.

This is a diagnostic survey, not a physical acceptance test.  Its default
matrix is intentionally small; pass application-specific transitions and
plasma ranges, then repeat with increasing ``--quadrature`` values.
"""

import argparse
import json
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starkzee.ffm import calculate_ffm_profile
from starkzee.static_profile import line_reference_energy


def _quadrature(value):
    try:
        num_f, num_mu = (int(item) for item in value.lower().split("x"))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "quadrature must have the form NUM_FxNUM_MU") from exc
    if num_f < 2 or num_mu < 1:
        raise argparse.ArgumentTypeError(
            "quadrature requires NUM_F >= 2 and NUM_MU >= 1")
    return num_f, num_mu


def _run_case(n_u, magnetic_field_t, density_m3, temperature_ev,
              num_f, num_mu, max_beta, microfield_model):
    center_ev = line_reference_energy(
        n_u, 2, 1, 2.0, use_empirical_data=True, atom="D")
    energies_ev = center_ev + np.linspace(-0.002, 0.002, 5)
    diagnostics = {}
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calculate_ffm_profile(
            n_u=n_u, n_l=2, Z=1, B=magnetic_field_t,
            Ne_m3=density_m3, Te_ev=temperature_ev,
            Ti_ev=temperature_ev, A_ion=2.0, energies_ev=energies_ev,
            num_f=num_f, num_mu=num_mu, use_empirical_data=True, atom="D",
            max_beta=max_beta, microfield_model=microfield_model,
            electron_model="pppb-intra", electron_interference=True,
            sdt_frequency_dependent_width=False, sdt_bin_tol=None,
            numerical_inversion=False, apply_doppler=False,
            interference_diagnostics=diagnostics,
        )
    absolute_sum = sum(item["absolute_strength_sum"]
                       for item in diagnostics.values())
    negative_absolute_sum = sum(item["negative_absolute_strength"]
                                for item in diagnostics.values())
    minima = [item["minimum_a_over_signed_sum"]
              for item in diagnostics.values()
              if item["minimum_a_over_signed_sum"] is not None]
    return {
        "settings": {
            "transition": f"D {n_u}->2",
            "n_u": n_u,
            "B_T": magnetic_field_t,
            "Ne_m-3": density_m3,
            "Te_eV": temperature_ev,
            "Ti_eV": temperature_ev,
            "electron_model": "pppb-intra",
            "microfield_model": microfield_model,
            "max_beta": max_beta,
            "num_f": num_f,
            "num_mu": num_mu,
        },
        "aggregate": {
            "mode_count": sum(item["mode_count"]
                              for item in diagnostics.values()),
            "negative_mode_count": sum(item["negative_mode_count"]
                                       for item in diagnostics.values()),
            "negative_absolute_fraction": (
                negative_absolute_sum / absolute_sum
                if absolute_sum > 0 else 0.0),
            "minimum_a_over_signed_sum": min(minima) if minima else None,
        },
        "polarizations": {str(q): diagnostics[q] for q in (0, 1, -1)},
        "warnings": sorted({str(item.message) for item in caught}),
        "elapsed_seconds": time.perf_counter() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upper-n", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--magnetic-field", nargs="+", type=float, default=[3.0])
    parser.add_argument("--density", nargs="+", type=float, default=[1e20])
    parser.add_argument("--temperature", nargs="+", type=float, default=[1.0])
    parser.add_argument("--max-beta", nargs="+", type=float, default=[10.0])
    parser.add_argument(
        "--microfield-model", choices=("holtsmark", "potekhin"),
        default="potekhin")
    parser.add_argument(
        "--quadrature", nargs="+", type=_quadrature,
        default=[(2, 2), (4, 3)], metavar="NUM_FxNUM_MU")
    parser.add_argument("--output", type=Path,
                        help="write JSON to this path instead of stdout")
    args = parser.parse_args()

    cases = [
        _run_case(
            n_u, field, density, temperature, num_f, num_mu, max_beta,
            args.microfield_model)
        for (n_u, field, density, temperature, (num_f, num_mu), max_beta)
        in product(
            args.upper_n, args.magnetic_field, args.density,
            args.temperature, args.quadrature, args.max_beta)
    ]
    report = {
        "purpose": (
            "diagnostic map of signed non-Hermitian residues entering the "
            "StarkZee modulus-probability FFM closure; not an external "
            "validation or a universal acceptance domain"
        ),
        "probability_rule": "p_k = abs(a_k) / sum_j(abs(a_j))",
        "radiative_rule": "signed a_k + i*c_k retained",
        "cases": cases,
    }
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
