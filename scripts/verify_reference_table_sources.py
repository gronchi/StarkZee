"""Verify bundled reference data and, optionally, its pinned pystark sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "starkzee" / "data" / "reference_tables.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def verify_local(registry: dict) -> dict[str, str]:
    results = {}
    for table in registry["tables"]:
        path = ROOT / table["path"]
        if path.stat().st_size != table["size_bytes"]:
            raise AssertionError(f"size mismatch: {table['path']}")
        actual = _sha256(path)
        if actual != table["sha256"]:
            raise AssertionError(f"SHA-256 mismatch: {table['path']}")
        results[table["id"]] = actual
    return results


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _canonical_tree_hash(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def verify_pystark(registry: dict, checkout: Path) -> dict[str, object]:
    by_id = {table["id"]: table for table in registry["tables"]}
    expected_commit = by_id["rosato_tables"]["source"]["pinned_commit"]
    actual_commit = _git(checkout, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise AssertionError(f"pystark HEAD is {actual_commit}, expected {expected_commit}")

    stehle = by_id["stehle_tables"]
    if _git(checkout, "rev-parse", "HEAD:" + stehle["source"]["raw_path"]) != stehle["source"]["raw_git_tree"]:
        raise AssertionError("Stehle raw Git-tree mismatch")
    upstream_stehle = checkout / stehle["source"]["netcdf_path"]
    if upstream_stehle.read_bytes() != (ROOT / stehle["path"]).read_bytes():
        raise AssertionError("Stehle NetCDF is not byte-identical to pinned upstream")
    conversion = checkout / stehle["source"]["conversion_script"]
    if _sha256(conversion) != stehle["source"]["conversion_script_sha256"]:
        raise AssertionError("Stehle conversion-script SHA-256 mismatch")

    rosato = by_id["rosato_tables"]
    if _git(checkout, "rev-parse", "HEAD:" + rosato["source"]["raw_path"]) != rosato["source"]["raw_git_tree"]:
        raise AssertionError("Rosato raw Git-tree mismatch")
    reader = checkout / rosato["source"]["reader_path"]
    if _sha256(reader) != rosato["source"]["reader_sha256"]:
        raise AssertionError("Rosato reader SHA-256 mismatch")
    raw_root = checkout / rosato["source"]["raw_path"]
    raw_files = list(raw_root.rglob("*.txt"))
    if len(raw_files) != rosato["source"]["raw_file_count"]:
        raise AssertionError("Rosato raw file-count mismatch")
    if sum(path.stat().st_size for path in raw_files) != rosato["source"]["raw_total_bytes"]:
        raise AssertionError("Rosato raw byte-count mismatch")
    if _canonical_tree_hash(raw_root, raw_files) != rosato["source"]["raw_canonical_tree_sha256"]:
        raise AssertionError("Rosato canonical raw-tree SHA-256 mismatch")

    transitions = rosato["axes"]["transitions"]
    compared_values = 0
    with netcdf_file(ROOT / rosato["path"], "r", mmap=False) as dataset:
        detunings = dataset.variables["detunings"].data
        intensities = dataset.variables["intensities"].data
        for transition_index, transition in enumerate(transitions):
            for density_index in range(10):
                for temperature_index in range(5):
                    for field_index in range(6):
                        for angle_index in range(2):
                            filename = (
                                f"ls{density_index + 1:02d}{temperature_index + 1}"
                                f"{field_index + 1}{angle_index}.txt"
                            )
                            values = np.loadtxt(raw_root / transition / filename)
                            index = (
                                transition_index,
                                density_index,
                                temperature_index,
                                field_index,
                                angle_index,
                            )
                            if not np.array_equal(values[:, 0], detunings[index]):
                                raise AssertionError(f"Rosato detuning mismatch: {transition}/{filename}")
                            if not np.array_equal(values[:, 1], intensities[index]):
                                raise AssertionError(f"Rosato intensity mismatch: {transition}/{filename}")
                            compared_values += values.size

    return {
        "pystark_commit": actual_commit,
        "stehle_byte_identical": True,
        "rosato_raw_files": len(raw_files),
        "rosato_values_compared": compared_values,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pystark", type=Path,
        help="Optional pystark checkout at the pinned commit for upstream verification.",
    )
    args = parser.parse_args()
    registry = load_registry()
    result: dict[str, object] = {"local_sha256": verify_local(registry)}
    if args.pystark is not None:
        result["upstream"] = verify_pystark(registry, args.pystark.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
