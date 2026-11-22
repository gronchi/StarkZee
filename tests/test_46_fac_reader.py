"""FAC ASCII reader on synthetic tables written in FAC's verbose layout."""
import numpy as np
import pytest

from starkzee.fac_reader import read_fac, read_fac_levels, read_fac_transitions
from starkzee.level_data import reduced_from_gf
from starkzee.utils import HARTREE_EV

HEAD = "FAC 1.1.5\nEndian\t= 0\nTSess\t= 1\nType\t= {t}\nVerbose\t= {v}\nHe Z\t=   2.0\nNBlocks\t= 1\n"
LEV = (HEAD.format(t=1, v=1) + "E0\t= 0, -7.90000000E+01\n\nNELE\t= 2\nNLEV\t= 3\n"
       "  ILEV  IBASE    ENERGY       P   VNL         2J\n"
       "     0     -1  0.00000000E+00 0   100          0 1*2                 1s2              1s+2(0)0\n"
       "     1     -1  2.00000000E+01 1   201          2 1*1.2*1             1s1.2p1          1s+1(1)1.2p-1(1)2\n"
       "     2     -1  2.05000000E+01 0   202          2 1*1.2*1             1s1.2d1          1s+1(1)1.2d-1(3)2\n")
M01, M12 = -0.8, 3.0                    # FAC reduced elements (lower bra), gf = M^2 dE[Ha]/3


def _tr_row(up, j_up, lo, j_lo, de, m, uta=False):
    gf = m * m * (de / HARTREE_EV) / 3.0
    if uta:
        return f"{up:6d} {j_up:10d} {lo:6d} {j_lo:10d} {de:13.6E} {0.01:13.6E} {gf:13.6E} {1e8:13.6E} {m:13.6E} {1.0:10.3E}\n"
    return f"{up:6d} {j_up:10d} {lo:6d} {j_lo:10d} {de:13.6E} {gf:13.6E} {1e8:13.6E} {m:13.6E}\n"


def _tr(multip, rows, verbose=1):
    return (HEAD.format(t=2, v=verbose) + f"\nNELE\t= 2\nNTRANS\t= {len(rows)}\nMULTIP\t= {multip}\n"
            "GAUGE\t= 2\nMODE\t= 1\n" + "".join(rows))


@pytest.fixture
def files(tmp_path):
    lev = tmp_path / "he.lev"
    lev.write_text(LEV)
    return tmp_path, lev


def test_levels(files):
    _, lev = files
    head, levels = read_fac_levels(lev)
    assert head["Z"] == 2.0 and head["E0_ev"] == -79.0 and head["E0_index"] == 0
    assert [lv.two_j for lv in levels] == [0, 2, 2]
    assert [lv.parity for lv in levels] == [1, -1, 1]
    assert (levels[1].n_valence, levels[1].l_valence, levels[1].nele) == (2, 1, 2)
    assert levels[2].configuration == "1s1.2d1" and levels[2].label == "1s+1(1)1.2d-1(3)2"


def test_signed_e1_elements_and_orientation(files):
    tmp, lev = files
    tr = tmp / "he.tr"
    tr.write_text(_tr(-1, [_tr_row(1, 2, 0, 0, 20.0, M01), _tr_row(2, 2, 1, 2, 0.5, M12)]))
    data = read_fac(lev, tr)
    assert data.all_signed and len(data.e1) == 2
    assert data.element(0, 1) == pytest.approx(M01 / np.sqrt(2), rel=1e-6)
    assert data.element(1, 2) == pytest.approx(M12 / np.sqrt(2), rel=1e-6)
    # <upper||d||lower> = (-1)^(J_u - J_l) <lower||d||upper>
    assert data.element(1, 0) == pytest.approx(-data.element(0, 1))
    assert data.element(2, 1) == pytest.approx(data.element(1, 2))
    # magnitude consistent with gf: S = 3 gf / (2 dE[Ha])
    t = data.e1[1]
    assert abs(t.reduced) == pytest.approx(reduced_from_gf(t.gf, t.energy_ev), rel=1e-6)


def test_inconsistent_m_column_is_rejected(files):
    tmp, lev = files
    bad = _tr_row(1, 2, 0, 0, 20.0, M01).replace(f"{M01:13.6E}", f"{2 * M01:13.6E}")
    tr = tmp / "bad.tr"
    tr.write_text(_tr(-1, [bad]))
    with pytest.raises(ValueError, match="not a reduced E1 element"):
        read_fac(lev, tr)


def test_multipole_zero_tables_are_unsigned(files):
    tmp, lev = files
    gf = 0.3
    row = f"{1:6d} {2:10d} {0:6d} {0:10d} {20.0:13.6E} {gf:13.6E} {1e8:13.6E} {gf:13.6E}\n"   # M = gf
    tr = tmp / "m0.tr"
    tr.write_text(_tr(0, [row]))
    data = read_fac(lev, tr)
    assert not data.all_signed and "MULTIP = 0" in data.notes[0]
    assert data.e1[0].reduced == pytest.approx(reduced_from_gf(gf, 20.0))


def test_uta_columns(files):
    tmp, lev = files
    tr = tmp / "uta.tr"
    tr.write_text(_tr(-1, [_tr_row(1, 2, 0, 0, 20.0, M01, uta=True)]))
    _, blocks = read_fac_transitions(tr)
    assert blocks[0]["uta"] and blocks[0]["sdev_ev"][0] == pytest.approx(0.01)
    assert read_fac(lev, tr).element(0, 1) == pytest.approx(M01 / np.sqrt(2), rel=1e-6)


def test_non_verbose_tables_are_rejected(files):
    tmp, _ = files
    tr = tmp / "nv.tr"
    tr.write_text(_tr(-1, ["     1      0  1.000000E+00\n"], verbose=0))
    with pytest.raises(ValueError, match="verbose"):
        read_fac_transitions(tr)
