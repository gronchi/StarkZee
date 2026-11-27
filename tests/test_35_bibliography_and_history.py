import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "starkzee" / "data"
BIBLIOGRAPHY = json.loads((DATA / "bibliography.json").read_text(encoding="utf-8"))
HISTORY = json.loads((DATA / "historical_provenance.json").read_text(encoding="utf-8"))


def _cited_keys(pattern, texts):
    return {
        key.strip()
        for text in texts
        for group in re.findall(pattern, text)
        for key in group.split(",")
    }


def test_generated_bibtex_matches_registry():
    """docs/references.bib must be regenerated whenever the registry changes."""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import generate_bibtex
    finally:
        sys.path.pop(0)
    bib = (ROOT / "docs" / "references.bib").read_text(encoding="utf-8")
    assert bib == generate_bibtex.render(BIBLIOGRAPHY)
    bib_keys = set(re.findall(r"^@\w+\{([^,]+),", bib, flags=re.MULTILINE))
    bib_dois = {doi.lower() for doi in re.findall(r"doi = \{([^}]+)\}", bib)}
    assert bib_keys == {work["citation_key"] for work in BIBLIOGRAPHY["works"]}
    assert bib_dois == {work["doi"].lower() for work in BIBLIOGRAPHY["works"]}


def test_both_manuals_cite_every_registered_work_from_the_generated_bib():
    registered_keys = {work["citation_key"] for work in BIBLIOGRAPHY["works"]}
    rst_texts = [
        path.read_text(encoding="utf-8")
        for path in (ROOT / "docs" / "source").rglob("*.rst")
    ]
    tex = (ROOT / "docs" / "manual.tex").read_text(encoding="utf-8")
    rst_keys = _cited_keys(r":cite:[a-z]+:`([^`]+)`", rst_texts)
    tex_keys = _cited_keys(r"\\cite[a-z]*\{([^}]+)\}", [tex])
    assert rst_keys == tex_keys == registered_keys

    # Neither manual may keep a hand-written reference list.
    references = (ROOT / "docs" / "source" / "manual_references.rst").read_text(
        encoding="utf-8"
    )
    assert ".. bibliography::" in references
    assert not any(re.search(r"^\.\. \[[^]]+\]", t, re.MULTILINE) for t in rst_texts)
    assert "\\addbibresource{references.bib}" in tex
    assert "\\printbibliography" in tex
    assert "thebibliography" not in tex
    conf = (ROOT / "docs" / "source" / "conf.py").read_text(encoding="utf-8")
    assert 'bibtex_bibfiles = ["../references.bib"]' in conf


def test_every_bibliography_record_has_a_stable_identifier_and_audit_source():
    works = BIBLIOGRAPHY["works"]
    assert len(works) == 22
    assert len({work["id"] for work in works}) == len(works)
    assert len({work["citation_key"] for work in works}) == len(works)
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
