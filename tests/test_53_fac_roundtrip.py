"""FAC interface end to end: tables written in FAC's verbose layout -> read_fac -> JLevelTransition -> profile.

The He I 447.1 nm radiator's level data (LS terms) is written as FAC .lev/.tr
tables (signed MULTIP = -1 elements, M = sqrt(2) <lower||d||upper> with the
lower-energy level as bra), read back and must give the same profiles.
"""
import warnings

import numpy as np
import pytest

from starkzee.fac_reader import jj_lande_factor, read_fac
from starkzee.he_i import he_i_4471
from starkzee.jlevel_radiator import JLevelTransition
from starkzee.static_profile import calculate_static_profile
from starkzee.utils import HARTREE_EV

HEAD = "FAC 1.1.5\nEndian\t= 0\nTSess\t= 1\nType\t= {t}\nVerbose\t= 1\nHe Z\t=   2.0\nNBlocks\t= 1\n"


def write_fac(data, path):
    """Write LevelData as FAC verbose .lev and .tr tables (energies with 14 digits)."""
    lv = data.levels
    rows = []
    for i in sorted(lv):
        p = 0 if lv[i].parity == 1 else 1
        rows.append(f"{i:6d} {-1:6d} {lv[i].energy_ev:22.14E} {p:1d} {100:5d} {lv[i].two_j:10d} "
                    f"x x 1s+1(1)1.2s+1(1)2\n")
    (path / "x.lev").write_text(HEAD.format(t=1) + "E0\t= 0, -7.90000000E+01\n\n"
                                f"NELE\t= 2\nNLEV\t= {len(rows)}\n  ILEV  IBASE    ENERGY       P   VNL         2J\n"
                                + "".join(rows))
    assigned = set()
    tr = []
    for t in data.e1:
        lo, up = sorted((t.lower, t.upper), key=lambda k: lv[k].energy_ev)
        de = lv[up].energy_ev - lv[lo].energy_ev
        m = np.sqrt(2.0) * data.element(lo, up)                 # FAC: lower-energy level as bra
        gf = m * m * (de / HARTREE_EV) / 3.0
        rate = 0.0
        if up not in assigned and lv[up].decay_rate_s and lv[lo].energy_ev < 22.0:   # radiative channel
            rate, _ = lv[up].decay_rate_s, assigned.add(up)
        tr.append(f"{up:6d} {lv[up].two_j:10d} {lo:6d} {lv[lo].two_j:10d} {de:22.14E} {gf:22.14E} "
                  f"{rate:22.14E} {m:22.14E}\n")
    (path / "x.tr").write_text(HEAD.format(t=2) + f"\nNELE\t= 2\nNTRANS\t= {len(tr)}\nMULTIP\t= -1\n"
                               "GAUGE\t= 2\nMODE\t= 1\n" + "".join(tr))
    return path / "x.lev", path / "x.tr"


def _manifold_levels(r):
    up = sorted({i for i, _, _ in r._states_u})
    lo = sorted({i for i, _, _ in r._states_l})
    return up, lo


@pytest.mark.parametrize("B", [0.0, 1.0])
def test_fac_tables_reproduce_the_he_i_radiator(tmp_path, B):
    ref = he_i_4471(B)
    data = ref.data
    lev, tr = write_fac(data, tmp_path)
    fac = read_fac(lev, tr, gf_rtol=1e-10, g_factors={i: l.g_J for i, l in data.levels.items()})
    assert fac.all_signed and len(fac.e1) == len(data.e1)
    for i, l in data.levels.items():
        assert fac.levels[i].energy_ev == pytest.approx(l.energy_ev, rel=1e-13)
        assert fac.levels[i].decay_rate_s is None and fac.decay_rate(i) == pytest.approx(data.decay_rate(i))
    for t in data.e1:
        assert fac.element(t.lower, t.upper) == pytest.approx(t.reduced, rel=1e-12)
    fac.magnetic = list(data.magnetic)          # FAC has no off-diagonal magnetic elements
    up, lo = _manifold_levels(ref)
    got = JLevelTransition(fac, up, lo, B, emitter_charge=0.0, reference_energy_ev=ref.line_energy_ev,
                           collision_n=ref.collision_n, collision_Z=ref.collision_Z,
                           collision_n_lower=ref.collision_n_lower, mass_amu=ref.mass_amu)
    E0 = ref.line_energy_ev
    grid = np.linspace(E0 - 3e-3, E0 + 4e-3, 701)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kw in ({}, dict(electron_operator=True)):
            a = calculate_static_profile(4, 2, 1, B, 1e21, 5.0, grid, num_f=6, num_mu=2, radiator=ref, Ti_ev=2.0, **kw)
            b = calculate_static_profile(4, 2, 1, B, 1e21, 5.0, grid, num_f=6, num_mu=2, radiator=got, Ti_ev=2.0, **kw)
            for x, y in zip(a, b):
                np.testing.assert_allclose(y, x, rtol=0, atol=1e-9 * np.max(x))


def test_missing_magnetic_mixing_is_flagged(tmp_path):
    ref = he_i_4471(1.0)
    lev, tr = write_fac(ref.data, tmp_path)
    fac = read_fac(lev, tr, gf_rtol=1e-10, g_factors={i: l.g_J for i, l in ref.data.levels.items()})
    up, lo = _manifold_levels(ref)
    with pytest.warns(UserWarning, match="off-diagonal magnetic"):
        JLevelTransition(fac, up, lo, 1.0)                # He I 4d J levels: 0.02 cm^-1 << mu_B B
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        JLevelTransition(fac, up, lo, 0.0)                # no field, no warning


@pytest.mark.parametrize("name, g", [
    ("2p+3(3)3.3s+1(1)4", 1.5011596),       # (2p3/2)^-1 3s J=2 = LS 3P2: 3/2 (with g_s)
    ("1s+1(1)1.2p-1(1)2", (2.0023193 + (1 - 1.0023193 / 3)) / 2),   # (1/2, 1/2) J=1: mean of g_j
    ("2p+4(0)0", 0.0),
    ("3d+1(5)5", 1 + 1.0023193 / 5),         # single 3d5/2: g_j = 1 + (g_s - 1)/(2l+1)
])
def test_jj_lande_factors(name, g):
    assert jj_lande_factor(name) == pytest.approx(g, rel=1e-6)


def test_jj_lande_factors_on_the_fe_xvii_sample():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "scratch" / "fac_samples"
    if not (root / "ne.lev").exists():
        pytest.skip("local FAC sample not available")
    data = read_fac(root / "ne.lev", g_factors="jj")
    assert any("jj coupling" in n for n in data.notes)
    assert all(lv.g_J is not None for lv in data.levels.values())
    assert data.levels[0].g_J == 0.0 and data.levels[1].g_J == pytest.approx(1.5012, abs=1e-4)
