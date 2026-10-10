#!/usr/bin/env python3
"""Compare StarkZee and ARC DC Stark maps for atomic hydrogen.

The matched calculation uses the H n=3 target state
``|3d_5/2, mj=1/2>``, shells n=2...5, all allowed orbital states, B=0, and
an electric field parallel to the quantization axis. Energies in each map are
referenced to that package's zero-field target energy, removing the different
absolute-energy conventions while retaining differences in Stark shifts.

ARC is optional. With a neighbouring ARC checkout and its uv environment::

    uv run --project ../ARC-Alkali-Rydberg-Calculator -- \
        python examples/compare_arc_stark_map_hydrogen.py --no-show

Use ``--arc-repo`` or ``ARC_REPO`` if the ARC checkout is elsewhere.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
from scipy.constants import e as E_CHARGE, h as PLANCK


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from starkzee.stark_map import StarkMap  # noqa: E402


TARGET = (8, 2, 2.5, 0.5)  # n, l, j, mj
SHELLS = tuple(range(2, 10))
MAX_L = 4
B_T = 0.0
FIELD_MAX_V_M = 2.0e7
FIELD_POINTS = 241
MAP_LIMIT_GHZ = 6500.0
EV_TO_GHZ = E_CHARGE / PLANCK / 1.0e9


def _default_arc_repo() -> Path:
    configured = os.environ.get("ARC_REPO")
    if configured:
        return Path(configured).expanduser()
    return REPOSITORY_ROOT.parent / "ARC-Alkali-Rydberg-Calculator"


def _load_arc(arc_repo: Path):
    try:
        return importlib.import_module("arc")
    except ModuleNotFoundError as exc:
        if exc.name != "arc" or not arc_repo.is_dir():
            raise SystemExit(
                "ARC is not importable. Install it or pass its checkout with "
                "--arc-repo (or ARC_REPO)."
            ) from exc
        sys.path.insert(0, str(arc_repo))
        try:
            return importlib.import_module("arc")
        except ModuleNotFoundError as dependency_error:
            raise SystemExit(
                "ARC was found, but its dependency "
                f"{dependency_error.name!r} is missing. Run this example in "
                "ARC's environment as shown in the module docstring."
            ) from dependency_error


def calculate_starkzee(fields_v_m):
    calculation = StarkMap(
        shells=SHELLS,
        Z=1,
        A=1,
        B=B_T,
        max_l=MAX_L,
        quadratic_zeeman=True,
        fine_structure=True,
    )
    return calculation.compute(
        fields_v_m,
        angle_deg=0.0,
        target_state=TARGET,
        mj=TARGET[3],
    )


def calculate_arc(arc, fields_v_m):
    hydrogen = arc.Hydrogen()
    calculation = arc.StarkMap(hydrogen)
    calculation.defineBasis(
        *TARGET,
        min(SHELLS),
        max(SHELLS),
        MAX_L,
        Bz=B_T,
        progressOutput=False,
    )
    calculation.diagonalise(fields_v_m, progressOutput=False)
    energies_ghz = np.asarray(calculation.y, dtype=float)
    highlights = np.asarray(calculation.highlight, dtype=float)
    reference_ghz = energies_ghz[0, int(np.argmax(highlights[0]))]
    return energies_ghz - reference_ghz, highlights, calculation


def _dominant_branch(energies, highlights):
    indices = np.argmax(highlights, axis=1)
    return energies[np.arange(len(indices)), indices]


def _scatter_map(ax, fields_v_cm, energies_ghz, highlights, title, cmap):
    x = np.repeat(fields_v_cm, energies_ghz.shape[1])
    y = energies_ghz.ravel()
    weights = highlights.ravel()
    order = np.argsort(weights, kind="stable")
    scatter = ax.scatter(
        x[order], y[order], c=weights[order], s=5.0,
        cmap=cmap, norm=Normalize(0.0, 1.0),
        linewidths=0.0, rasterized=True,
    )
    ax.set_title(title)
    ax.set_xlabel("Electric field (V/cm)")
    ax.set_ylabel("Energy from zero-field target (GHz)")
    ax.set_ylim(-MAP_LIMIT_GHZ, MAP_LIMIT_GHZ)
    ax.grid(alpha=0.2)
    return scatter


def make_figure(fields_v_m, starkzee, arc_energies_ghz, arc_highlights,
                arc_version, output: Path, show: bool):
    fields_v_cm = fields_v_m / 100.0
    starkzee_energies_ghz = (
        starkzee.energies_ev - starkzee.reference_energy_ev
    ) * EV_TO_GHZ
    starkzee_branch = _dominant_branch(
        starkzee_energies_ghz, starkzee.highlights)
    arc_branch = _dominant_branch(arc_energies_ghz, arc_highlights)
    difference = starkzee_branch - arc_branch

    cmap = LinearSegmentedColormap.from_list(
        "target_overlap", ["0.82", "#d73027", "black"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    scatter = _scatter_map(
        axes[0, 0], fields_v_cm, starkzee_energies_ghz,
        starkzee.highlights, "StarkZee multi-shell map", cmap)
    _scatter_map(
        axes[0, 1], fields_v_cm, arc_energies_ghz,
        arc_highlights, f"ARC {arc_version} map", cmap)
    fig.colorbar(
        scatter, ax=axes[0, :], label=r"Target overlap $|\langle3d_{5/2},m_j=1/2|\psi\rangle|^2$",
        pad=0.015, shrink=0.92,
    )

    axes[1, 0].plot(
        fields_v_cm, starkzee_branch,
        label="StarkZee", color="C0", linewidth=2.0)
    axes[1, 0].plot(
        fields_v_cm, arc_branch,
        label="ARC", color="C1", linewidth=1.5, linestyle="--")
    axes[1, 0].set_xlabel("Electric field (V/cm)")
    axes[1, 0].set_ylabel("Maximum-overlap branch shift (GHz)")
    axes[1, 0].set_title("Target-character branch")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(
        fields_v_cm, difference,
        color="C3", linewidth=1.6)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_xlabel("Electric field (V/cm)")
    axes[1, 1].set_ylabel("StarkZee - ARC (GHz)")
    axes[1, 1].set_title("Target-branch difference")
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle(
        r"Hydrogen DC Stark map: $|3d_{5/2},m_j=1/2\rangle$, "
        r"$n=2\ldots5$, $B=0$",
        fontsize=14,
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")

    print(f"StarkZee basis dimension: {starkzee.energies_ev.shape[1]}")
    print(f"ARC basis dimension:      {arc_energies_ghz.shape[1]}")
    print(
        "Maximum absolute target-branch difference: "
        f"{np.max(np.abs(difference)):.6f} GHz")
    print(
        "RMS target-branch difference:              "
        f"{np.sqrt(np.mean(difference**2)):.6f} GHz")
    print(
        "Endpoint target-branch difference:         "
        f"{difference[-1]:.6f} GHz")
    print(f"Saved {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arc-repo", type=Path, default=_default_arc_repo(),
        help="path to an ARC source checkout if ARC is not installed")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).with_suffix(".png"),
        help="output figure path")
    parser.add_argument(
        "--no-show", action="store_true",
        help="save without opening an interactive window")
    args = parser.parse_args()

    arc = _load_arc(args.arc_repo)
    fields_v_m = np.linspace(0.0, FIELD_MAX_V_M, FIELD_POINTS)
    starkzee = calculate_starkzee(fields_v_m)
    arc_energies_ghz, arc_highlights, arc_calculation = calculate_arc(
        arc, fields_v_m)
    if starkzee.energies_ev.shape != arc_energies_ghz.shape:
        raise RuntimeError(
            "The matched StarkZee and ARC bases have different dimensions: "
            f"{starkzee.energies_ev.shape[1]} and {arc_energies_ghz.shape[1]}")
    if len(starkzee.basis) != len(arc_calculation.basisStates):
        raise RuntimeError("basis-state counts differ after matched setup")

    make_figure(
        fields_v_m, starkzee, arc_energies_ghz, arc_highlights,
        getattr(arc, "__version__", "unknown"), args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
