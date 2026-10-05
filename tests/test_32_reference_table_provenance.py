import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "starkzee" / "data"
REGISTRY = json.loads((DATA / "reference_tables.json").read_text(encoding="utf-8"))


def test_registry_covers_every_runtime_reference_table():
    registered = {table["path"] for table in REGISTRY["tables"]}
    actual = {
        path.relative_to(ROOT).as_posix()
        for pattern in ("*.nc", "atomic_levels.json")
        for path in DATA.glob(pattern)
    }
    assert registered == actual


def test_registered_sizes_and_hashes_match_bundled_files():
    for table in REGISTRY["tables"]:
        path = ROOT / table["path"]
        assert path.stat().st_size == table["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == table["sha256"]


def test_unknown_provenance_is_explicit_not_inferred():
    tables = {table["id"]: table for table in REGISTRY["tables"]}
    rosato = tables["rosato_tables"]["source"]
    assert rosato["original_database_acquisition_url"] == "unknown"
    assert rosato["original_database_acquisition_date"] == "unknown"
    assert rosato["license"] == "unknown"
    assert "No per-node" in rosato["uncertainty"]
    assert tables["stehle_tables"]["source"]["license"] == "unknown"


def test_rosato_registry_shape_accounts_for_every_raw_profile():
    rosato = next(table for table in REGISTRY["tables"] if table["id"] == "rosato_tables")
    axes = rosato["axes"]
    profiles = (
        len(axes["transitions"])
        * axes["electron_density_nodes"]
        * axes["temperature_nodes"]
        * axes["magnetic_field_nodes"]
        * axes["viewing_angle_nodes"]
    )
    assert profiles == rosato["source"]["raw_file_count"] == 3000


def test_atomic_embedded_metadata_limit_is_recorded():
    atomic = json.loads((DATA / "atomic_levels.json").read_text(encoding="utf-8"))
    source = next(
        table["source"] for table in REGISTRY["tables"]
        if table["id"] == "atomic_levels"
    )
    assert source["release_for_archived_D_T_queries"].startswith("5.12")
    assert source["release_for_H"] == "unknown"
    assert "metadata" not in atomic["H"]
    for isotope in ("D", "T"):
        assert atomic[isotope]["metadata"]["source"] == "NIST ASD levels query"
        assert atomic[isotope]["metadata"]["units"] == "cm^-1"
