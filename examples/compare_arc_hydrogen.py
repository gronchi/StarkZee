#!/usr/bin/env python3
"""Compare field-free hydrogen quantities calculated by StarkZee and ARC.

The comparison covers resolved Ly-alpha and H-alpha components and prints
photon energies, radial dipoles, J-reduced dipoles, and Einstein A rates.
Absolute dipole values are compared because the sign of an individual matrix
element depends on the phase convention chosen for the atomic states.

ARC is an optional dependency. From the StarkZee repository, one convenient
way to run this example against a neighbouring ARC checkout is::

    uv run --project ../ARC-Alkali-Rydberg-Calculator \
        python examples/compare_arc_hydrogen.py --no-show

Alternatively, install ARC in the active environment. Set ``ARC_REPO`` or use
``--arc-repo`` when its checkout is not next to the StarkZee repository.

This is deliberately a field-free comparison. ARC's standard magnetic-field
treatment is a fixed-j, weak-field approximation and is not equivalent to
StarkZee's coupled Stark-Zeeman Hamiltonian at high magnetic field.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from starkzee.atomic import (  # noqa: E402
    einstein_a_from_strength,
    radial_dipole,
    reduced_hydrogenic_dipole_j,
)
from starkzee.utils import reduced_mass_rydberg_ev  # noqa: E402


# label, n_u, l_u, j_u, n_l, l_l, j_l
TRANSITIONS = [
    ("Ly-alpha 2p1/2 -> 1s1/2", 2, 1, 0.5, 1, 0, 0.5),
    ("Ly-alpha 2p3/2 -> 1s1/2", 2, 1, 1.5, 1, 0, 0.5),
    ("H-alpha 3s1/2 -> 2p1/2", 3, 0, 0.5, 2, 1, 0.5),
    ("H-alpha 3s1/2 -> 2p3/2", 3, 0, 0.5, 2, 1, 1.5),
    ("H-alpha 3p1/2 -> 2s1/2", 3, 1, 0.5, 2, 0, 0.5),
    ("H-alpha 3p3/2 -> 2s1/2", 3, 1, 1.5, 2, 0, 0.5),
    ("H-alpha 3d3/2 -> 2p1/2", 3, 2, 1.5, 2, 1, 0.5),
    ("H-alpha 3d3/2 -> 2p3/2", 3, 2, 1.5, 2, 1, 1.5),
    ("H-alpha 3d5/2 -> 2p3/2", 3, 2, 2.5, 2, 1, 1.5),
]


def _default_arc_repo() -> Path:
    configured = os.environ.get("ARC_REPO")
    if configured:
        return Path(configured).expanduser()
    return REPOSITORY_ROOT.parent / "ARC-Alkali-Rydberg-Calculator"


def _load_arc_hydrogen(arc_repo: Path):
    """Import ARC's Hydrogen class, optionally from a source checkout."""
    try:
        arc = importlib.import_module("arc")
    except ModuleNotFoundError as exc:
        if exc.name == "arc" and arc_repo.is_dir():
            sys.path.insert(0, str(arc_repo))
            try:
                arc = importlib.import_module("arc")
            except ModuleNotFoundError as dependency_error:
                raise SystemExit(
                    "ARC was found, but its dependency "
                    f"{dependency_error.name!r} is missing. Install ARC with "
                    f"`python -m pip install -e {arc_repo}` or run this script "
                    "with ARC's uv environment as shown in the module docstring."
                ) from dependency_error
        else:
            raise SystemExit(
                "ARC is not importable. Install ARC or pass its checkout with "
                "--arc-repo (or the ARC_REPO environment variable)."
            ) from exc
    return arc, arc.Hydrogen


def _starkzee_gross_energy_ev(n_u: int, n_l: int) -> float:
    """Reduced-mass hydrogen gross-structure photon energy."""
    return reduced_mass_rydberg_ev(Z=1, A=1) * (
        1.0 / n_l**2 - 1.0 / n_u**2
    )


def calculate_comparison(hydrogen) -> list[dict[str, float | str]]:
    """Calculate matched field-free quantities with both packages."""
    rows = []
    for label, n_u, l_u, j_u, n_l, l_l, j_l in TRANSITIONS:
        arc_energy_ev = abs(
            hydrogen.getEnergy(n_u, l_u, j_u)
            - hydrogen.getEnergy(n_l, l_l, j_l)
        )
        arc_radial_a0 = abs(
            hydrogen.getRadialMatrixElement(
                n_u, l_u, j_u, n_l, l_l, j_l,
                useLiterature=False,
            )
        )
        arc_reduced_j_a0 = abs(
            hydrogen.getReducedMatrixElementJ(
                n_u, l_u, j_u, n_l, l_l, j_l
            )
        )
        arc_rate_s = hydrogen.getTransitionRate(
            n_u, l_u, j_u, n_l, l_l, j_l, temperature=0.0
        )

        starkzee_energy_ev = _starkzee_gross_energy_ev(n_u, n_l)
        starkzee_radial_a0 = abs(radial_dipole(n_u, l_u, n_l, l_l))
        starkzee_reduced_j_a0 = abs(
            reduced_hydrogenic_dipole_j(
                n_u, l_u, j_u, n_l, l_l, j_l
            )
        )
        # For a J-resolved upper level, divide the squared reduced matrix
        # element by its 2*j_u+1 magnetic-sublevel degeneracy.
        starkzee_rate_s = einstein_a_from_strength(
            starkzee_energy_ev,
            starkzee_reduced_j_a0**2 / (2.0 * j_u + 1.0),
        )

        rows.append(
            {
                "label": label,
                "arc_energy_ev": arc_energy_ev,
                "starkzee_energy_ev": starkzee_energy_ev,
                "arc_radial_a0": arc_radial_a0,
                "starkzee_radial_a0": starkzee_radial_a0,
                "arc_reduced_j_a0": arc_reduced_j_a0,
                "starkzee_reduced_j_a0": starkzee_reduced_j_a0,
                "arc_rate_s": arc_rate_s,
                "starkzee_rate_s": starkzee_rate_s,
            }
        )
    return rows


def _relative_percent(starkzee: float, arc: float) -> float:
    return 100.0 * (starkzee - arc) / arc


def print_comparison(rows, arc_version: str) -> None:
    print(f"ARC version: {arc_version}")
    print("\nDirect values")
    print(
        f"{'Transition':30s} {'E_SZ (eV)':>11s} {'E_ARC (eV)':>11s} "
        f"{'R_SZ':>9s} {'R_ARC':>9s} {'dJ_SZ':>9s} {'dJ_ARC':>9s} "
        f"{'A_SZ (s^-1)':>13s} {'A_ARC (s^-1)':>13s}"
    )
    print("-" * 127)
    for row in rows:
        print(
            f"{row['label']:30s} "
            f"{row['starkzee_energy_ev']:11.7f} {row['arc_energy_ev']:11.7f} "
            f"{row['starkzee_radial_a0']:9.5f} {row['arc_radial_a0']:9.5f} "
            f"{row['starkzee_reduced_j_a0']:9.5f} "
            f"{row['arc_reduced_j_a0']:9.5f} "
            f"{row['starkzee_rate_s']:13.5e} {row['arc_rate_s']:13.5e}"
        )

    print()
    print("StarkZee - ARC relative differences")
    print(
        f"{'Transition':30s} {'energy (ppm)':>13s} "
        f"{'|R| (%)':>10s} {'|<J||d||J>| (%)':>17s} {'A (%)':>10s}"
    )
    print("-" * 86)
    for row in rows:
        energy_ppm = 1.0e6 * (
            row["starkzee_energy_ev"] - row["arc_energy_ev"]
        ) / row["arc_energy_ev"]
        radial_percent = _relative_percent(
            row["starkzee_radial_a0"], row["arc_radial_a0"]
        )
        reduced_percent = _relative_percent(
            row["starkzee_reduced_j_a0"], row["arc_reduced_j_a0"]
        )
        rate_percent = _relative_percent(
            row["starkzee_rate_s"], row["arc_rate_s"]
        )
        print(
            f"{row['label']:30s} {energy_ppm:13.3f} "
            f"{radial_percent:10.4f} {reduced_percent:17.4f} "
            f"{rate_percent:10.4f}"
        )


def plot_comparison(rows, output: Path, show: bool) -> None:
    labels = [str(row["label"]) for row in rows]
    y = np.arange(len(rows))
    energy_ppm = np.array(
        [
            1.0e6
            * (row["starkzee_energy_ev"] - row["arc_energy_ev"])
            / row["arc_energy_ev"]
            for row in rows
        ]
    )
    radial_percent = np.array(
        [
            _relative_percent(
                row["starkzee_radial_a0"], row["arc_radial_a0"]
            )
            for row in rows
        ]
    )
    rate_percent = np.array(
        [
            _relative_percent(row["starkzee_rate_s"], row["arc_rate_s"])
            for row in rows
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    panels = [
        (energy_ppm, "C0", "Photon energy difference (ppm)"),
        (radial_percent, "C1", "Radial dipole difference (%)"),
        (rate_percent, "C2", "Einstein A difference (%)"),
    ]
    for ax, (values, color, xlabel) in zip(axes, panels):
        ax.barh(y, values, color=color, alpha=0.85)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.25)

    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Field-free hydrogen: relative difference (StarkZee - ARC) / ARC"
    )
    fig.text(
        0.5,
        0.01,
        "ARC: low-n level data and numerical radial model; "
        "StarkZee: reduced-mass gross energy and analytic Coulomb radial model",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    print(f"\nSaved {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arc-repo",
        type=Path,
        default=_default_arc_repo(),
        help="path to an ARC source checkout if ARC is not installed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_suffix(".png"),
        help="output figure path",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save the figure without opening an interactive window",
    )
    args = parser.parse_args()

    arc, Hydrogen = _load_arc_hydrogen(args.arc_repo)
    hydrogen = Hydrogen()
    rows = calculate_comparison(hydrogen)
    print_comparison(rows, getattr(arc, "__version__", "unknown"))
    plot_comparison(rows, args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
