import json

import pytest

from scripts.update_nist_levels import (
    NistParseError,
    build_nist_levels_url,
    find_incomplete_fine_structure_shells,
    parse_nist_levels,
    update_database,
)


NIST_SAMPLE = """Configuration\tTerm\tJ\tPrefix\tLevel (cm-1)\tSuffix
"1s"\t"2S"\t"1/2"\t""\t"0.0000000000"\t""
"2p"\t"2P*"\t"1/2"\t""\t"82258.9191133"\t""
"2p"\t"2P*"\t"3/2"\t""\t"82259.2850014"\t""
"2s"\t"2S"\t"1/2"\t""\t"82258.9543992821"\t""
"2"\t""\t""\t""\t"82259.158"\t""
"3p"\t"2P*"\t"1/2"\t""\t"97492.211200"\t""
"3p"\t"2P*"\t"3/2"\t""\t"97492.319611"\t""
"3s"\t"2S"\t"1/2"\t""\t"97492.221701"\t""
"3"\t""\t""\t""\t"97492.304"\t""
"3d"\t"2D"\t"3/2"\t""\t"97492.319433"\t""
"3d"\t"2D"\t"5/2"\t""\t"97492.355566"\t""
"4p"\t"2P*"\t"1/2"\t""\t"102823.8485825"\t""
"4"\t""\t""\t""\t"102823.904"\t""
""\t"Limit"\t"---"\t"("\t"109678.77174307"\t")"
"""


def test_build_nist_levels_url_uses_cm_minus_one_tab_delimited_output():
    url = build_nist_levels_url("H I")

    assert "energy1.pl" in url
    assert "spectrum=H+I" in url
    assert "units=0" in url
    assert "format=3" in url
    assert "conf_out=on" in url
    assert "j_out=on" in url
    assert "lande_out" not in url


def test_parse_nist_levels_splits_fine_structure_and_shell_averages():
    parsed = parse_nist_levels(NIST_SAMPLE, max_n=3)

    assert parsed.fine_structure_false == [
        {"n": 2, "energy": 82259.158},
        {"n": 3, "energy": 97492.304},
    ]
    assert {"n": 2, "l": 1, "j": 1.5, "energy": 82259.2850014} in parsed.fine_structure_true
    assert {"n": 3, "l": 2, "j": 2.5, "energy": 97492.355566} in parsed.fine_structure_true
    assert all(level["n"] <= 3 for level in parsed.fine_structure_true)


def test_parse_nist_levels_rejects_html_error_output():
    with pytest.raises(NistParseError, match="HTML/error"):
        parse_nist_levels("<html><h2>Error Message:</h2></html>")


def test_update_database_preserves_other_atoms_and_adds_metadata():
    parsed = parse_nist_levels(NIST_SAMPLE, max_n=2)
    database = {"He": {"fine_structure_false": []}}

    updated = update_database(database, atom="H", spectrum="H I", levels=parsed)

    assert "He" in updated
    assert updated["H"]["fine_structure_false"] == [{"n": 2, "energy": 82259.158}]
    assert updated["H"]["metadata"]["source"] == "NIST ASD levels query"
    assert updated["H"]["metadata"]["spectrum"] == "H I"
    assert json.dumps(updated)  # remains JSON serializable

def test_find_incomplete_fine_structure_shells_reports_missing_l_j_states():
    incomplete_sample = NIST_SAMPLE.replace('"3d"\t"2D"\t"5/2"\t""\t"97492.355566"\t""\n', '')

    parsed = parse_nist_levels(incomplete_sample, max_n=3)
    missing = find_incomplete_fine_structure_shells(parsed)

    assert missing == {3: [(2, 2.5)]}


