"""Verify the manual-era Git commit, artifacts, and dataset continuity."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = ROOT / "starkzee" / "data" / "historical_provenance.json"
TABLES_PATH = ROOT / "starkzee" / "data" / "reference_tables.json"


def _git_bytes(*args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], check=True, capture_output=True
    ).stdout


def _git_text(*args: str) -> str:
    return _git_bytes(*args).decode("utf-8").strip()


def verify() -> dict[str, object]:
    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    tables = json.loads(TABLES_PATH.read_text(encoding="utf-8"))
    commit = history["historical_commit"]
    sha = commit["sha"]

    _git_text("cat-file", "-e", f"{sha}^{{commit}}")
    for name, expected in (
        ("starkzee", commit["starkzee_tree"]),
        ("starkzee/data", commit["data_tree"]),
        ("starkzee/models", commit["models_tree"]),
    ):
        actual = _git_text("rev-parse", f"{sha}:{name}")
        if actual != expected:
            raise AssertionError(f"historical tree mismatch: {name}")

    verified = []
    for artifact in history["artifacts"]:
        revision = f"{sha}:{artifact['path']}"
        blob = _git_text("rev-parse", revision)
        if blob != artifact["git_blob"]:
            raise AssertionError(f"historical blob mismatch: {artifact['path']}")
        content = _git_bytes("cat-file", "blob", blob)
        if len(content) != artifact["size_bytes"]:
            raise AssertionError(f"historical size mismatch: {artifact['path']}")
        digest = hashlib.sha256(content).hexdigest()
        if digest != artifact["sha256"]:
            raise AssertionError(f"historical SHA-256 mismatch: {artifact['path']}")
        verified.append(artifact["path"])

    manual = next(
        item for item in history["artifacts"]
        if item["role"] == "audited_manual_source"
    )
    if manual["sha256"] != history["independently_recorded_manual_sha256"]:
        raise AssertionError("historical manual does not match the recorded audit hash")

    current_tables = {table["path"]: table for table in tables["tables"]}
    for artifact in history["artifacts"]:
        identical = artifact.get("current_bytes_identical")
        if identical is None:
            continue
        current = current_tables[artifact["path"]]["sha256"]
        if (current == artifact["sha256"]) is not identical:
            raise AssertionError(
                f"current-byte relationship mismatch: {artifact['path']}"
            )

    return {
        "commit": sha,
        "verified_artifacts": verified,
        "manual_matches_original_audit": True,
        "dataset_relationships_verified": True,
    }


def main() -> int:
    print(json.dumps(verify(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
