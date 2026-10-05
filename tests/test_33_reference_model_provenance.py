import inspect
import json
from pathlib import Path

from starkzee.models import analytical


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "starkzee" / "models" / "reference_provenance.json"
REGISTRY = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_reference_model_registry_covers_distinct_models():
    assert set(REGISTRY["models"]) == {
        "stehle", "rosato", "stehle_param", "lomanowski", "voigt"
    }
    for record in REGISTRY["models"].values():
        assert record["directly_sourced"]
        assert record["starkzee_specific"]
        assert record["primary_source"]["doi"].startswith("10.")


def test_lomanowski_coefficients_match_paper_table_1():
    expected = {
        "32": (0.7665, 0.064, 3.710e-18),
        "42": (0.7803, 0.050, 8.425e-18),
        "52": (0.6796, 0.030, 1.310e-15),
        "62": (0.7149, 0.028, 3.954e-16),
        "72": (0.7120, 0.029, 6.258e-16),
        "82": (0.7159, 0.032, 7.378e-16),
        "92": (0.7177, 0.033, 8.947e-16),
        "43": (0.7449, 0.045, 1.330e-16),
        "53": (0.7356, 0.044, 6.640e-16),
        "63": (0.7118, 0.016, 2.481e-15),
        "73": (0.7137, 0.029, 3.270e-15),
        "83": (0.7133, 0.032, 4.343e-15),
        "93": (0.7165, 0.033, 5.588e-15),
    }
    assert analytical._LOMAN_COEFFS == expected


def test_published_parameterization_locations_are_explicit():
    source = REGISTRY["models"]["stehle_param"]["primary_source"]
    assert source["source_locations"] == ["Equation (1)", "Equation (2)", "Table 1"]
    assert source["doi"] == "10.1088/0029-5515/55/12/123028"


def test_local_lomanowski_mixture_is_not_misattributed_to_pystark():
    source = inspect.getsource(analytical.lomanowski)
    assert "StarkZee-specific" in source
    assert "pystark.make_lomanowski" not in source
    assert "mixing polynomials is unknown" in REGISTRY["models"]["lomanowski"]["starkzee_specific"]


def test_rosato_bibliography_matches_table_paper():
    rst = (ROOT / "docs" / "source" / "manual_references.rst").read_text(encoding="utf-8")
    tex = (ROOT / "docs" / "manual.tex").read_text(encoding="utf-8")
    for text in (rst, tex):
        assert "A new table of Balmer line shapes" in text
        assert "10.1016/j.jqsrt.2016.10.005" in text
        assert "J. Rosato, Y. Marandet, R. Stamm" in text
    assert "**190**, 1–5" not in rst
    assert "10.1103/PhysRevE.79.046408" not in tex


def test_example_does_not_claim_lomanowski_lacks_field_handling():
    page = (ROOT / "docs" / "source" / "examples" / "model_comparison.rst").read_text(
        encoding="utf-8"
    )
    assert "no magnetic-field treatment" not in page
    assert "Both wrappers can add a separable" in page
