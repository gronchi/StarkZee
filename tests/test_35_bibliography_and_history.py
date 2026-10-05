import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "starkzee" / "data"
BIBLIOGRAPHY = json.loads((DATA / "bibliography.json").read_text(encoding="utf-8"))
HISTORY = json.loads((DATA / "historical_provenance.json").read_text(encoding="utf-8"))


def test_bibliography_registry_covers_documented_reference_keys_and_dois():
    rst = (ROOT / "docs" / "source" / "manual_references.rst").read_text(
        encoding="utf-8"
    )
    tex = (ROOT / "docs" / "manual.tex").read_text(encoding="utf-8")
    rst_keys = set(re.findall(r"^\.\. \[([^]]+)\]", rst, flags=re.MULTILINE))
    tex_keys = set(re.findall(r"\\bibitem\{([^}]+)\}", tex))
    registered_keys = {work["citation_key"] for work in BIBLIOGRAPHY["works"]}
    assert rst_keys == tex_keys == registered_keys

    registered_dois = {work["doi"].lower() for work in BIBLIOGRAPHY["works"]}
    rst_dois = {doi.lower() for doi in re.findall(r"10\.\d{4,9}/[^\s>`]+", rst)}
    tex_dois = {
        doi.lower().rstrip(".,;")
        for doi in re.findall(r"10\.\d{4,9}/[^\s]+", tex)
    }
    assert rst_dois == tex_dois == registered_dois


def test_every_bibliography_record_has_a_stable_identifier_and_audit_source():
    works = BIBLIOGRAPHY["works"]
    assert len(works) == 22
    assert len({work["id"] for work in works}) == len(works)
    assert len({work["doi"].lower() for work in works}) == len(works)
    for work in works:
        assert work["doi"].startswith("10.")
        assert work["verification_url"].startswith("https://")
        assert work["authors"]
        assert work["title"]
        assert work["year"] >= 1900


def test_nist_release_and_unresolved_hydrogen_acquisition_are_explicit():
    nist = next(work for work in BIBLIOGRAPHY["works"] if work["citation_key"] == "nist")
    assert nist["version"] == "5.12"
    assert nist["year"] == 2024
    assert "H I" in nist["dataset_note"]
    assert "unknown" in nist["dataset_note"]


def test_historical_registry_pins_audited_manual_and_companion_artifacts():
    commit = HISTORY["historical_commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", commit["sha"])
    assert commit["sha"] in commit["url"]
    artifacts = {item["role"]: item for item in HISTORY["artifacts"]}
    assert set(artifacts) == {
        "audited_manual_source", "compiled_manual", "manual_figure",
        "figure_generator", "atomic_levels_dataset", "rosato_dataset",
        "stehle_dataset",
    }
    assert artifacts["audited_manual_source"]["sha256"] == (
        "dbfcb4766218bbb1d458982865bc22d6b5158b99e1359135908d4bf8f0ef89b6"
    )
    assert HISTORY["independently_recorded_manual_sha256"] == (
        artifacts["audited_manual_source"]["sha256"]
    )
    for artifact in artifacts.values():
        assert re.fullmatch(r"[0-9a-f]{40}", artifact["git_blob"])
        assert re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])
        assert commit["sha"] in artifact["url"]


def test_historical_registry_does_not_overclaim_external_data_provenance():
    assert "not scientific accuracy" in HISTORY["scope"]
    assert "does not resolve" in HISTORY["external_dataset_lineage"]
    artifacts = {item["role"]: item for item in HISTORY["artifacts"]}
    assert artifacts["atomic_levels_dataset"]["current_bytes_identical"] is False
    assert artifacts["rosato_dataset"]["current_bytes_identical"] is True
    assert artifacts["stehle_dataset"]["current_bytes_identical"] is True
