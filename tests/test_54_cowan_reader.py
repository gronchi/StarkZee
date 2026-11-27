"""Cowan RCG interface: spectra printed in RCG's formats -> read_rcg -> JLevelTransition -> profile.

The He I 447.1 nm radiator's level data is printed as RCG output (formats 55, 15,
58/88 of spectr.for and the energy.for / sprin.for eigenvalue and g-value lists):
E1 rows with the first parity group as bra and the signed cancellation factor,
an M1 block with the off-diagonal magnetic elements, and the g-values. Reading it
back must give the same elements (to the print precision) and profiles.
"""
import dataclasses
import warnings

import numpy as np
import pytest

from starkzee.cowan_reader import (
    EV_PER_CM, RCG_GF_CONSTANT, configuration_parity, ls_lande_factor, read_rcg, select_levels,
)
from starkzee.he_i import he_i_4471
from starkzee.jlevel_radiator import JLevelTransition
from starkzee.static_profile import calculate_static_profile


def _conf(lv):                                   # '4d 3D' -> RCG configuration and designation
    nl, term = lv.configuration.split()
    return f"1s1 {nl}1", f"(2S) {term}"


def _row(i, a, b, E, lab, conf, unit, s2, gf, last, branch):
    Ea, Eb = E[a], E[b]
    dE = abs(Ea - Eb)
    lam = 1e8 / (dE * unit) if dE else np.inf
    lam_s = f"{lam:12.4f}" if lam < 1e7 else "*" * 12
    loggf = np.log10(gf) if gf > 0 else -99.999
    gA = 0.66698 * gf * (dE * unit) ** 2
    tail = f"{last:9.2E}" if branch else f"{last:8.4f}"
    return (f" {i:4d}{Ea:11.4f}{lab[a][2]:5.1f}{conf[a]:3d} {lab[a][1]:8s}{Eb:13.4f}{lab[b][2]:5.1f}{conf[b]:3d} "
            f"{lab[b][1]:8s}{dE:13.4f}{lam_s}{s2:8.3f}{gf:9.5f}{loggf:8.3f}{gA:10.3E}{tail}\n")


def write_rcg(data, path, unit_cm=1.0, offset=50.0, pmup=True, m1=True, g_values=True, header=True,
              branch=False, reprint=False):
    """Print LevelData as RCG spectrum output (E1 bra: even parity group)."""
    lv = data.levels
    groups = {+1: [i for i in lv if lv[i].parity == 1], -1: [i for i in lv if lv[i].parity == -1]}
    names = {p: sorted({_conf(lv[i])[0] for i in groups[p]}) for p in groups}
    conf = {i: names[lv[i].parity].index(_conf(lv[i])[0]) + 1 for i in lv}
    lab = {i: (_conf(lv[i])[0], _conf(lv[i])[1], lv[i].J) for i in lv}
    e0 = min(l.energy_ev for l in lv.values())
    E = {i: (lv[i].energy_ev - e0) / EV_PER_CM / unit_cm + offset for i in lv}
    out = []

    def head(kind, left, right):
        if not header:
            out.append(f" {kind:>9s}\n")
            return
        out.append(f"1         {kind:8s} spectrum     (Energies in units of{unit_cm:8.1f} cm-1 ="
                   f"{unit_cm / 8065.479:8.2f} eV)\n\n")
        for jn in range(1, max(len(left), len(right)) + 1):
            a = left[jn - 1] if jn <= len(left) else ""
            b = right[jn - 1] if jn <= len(right) else ""
            out.append(f"{jn:5d}      {a:18s}   ---      {b:18s}\n")
        out.append("\n0            E      J   Conf              Ep     Jp  Confp           delta E   lambda(a) "
                   "s/pmax**2   gf    log gf ga(sec-1) cf,brnch\n")

    pairs = []
    for t in data.e1:
        a, b = (t.lower, t.upper) if lv[t.lower].parity == 1 else (t.upper, t.lower)
        pairs.append((a, b, data.element(a, b)))
    pmax = max(abs(x) for _, _, x in pairs)
    if header and pmup:
        out.append("1                                                           elec dip spectrum\n\n\n\n")
        for i, a in enumerate(names[1], 1):
            for j, b in enumerate(names[-1], 1):
                p = pmax if (i, j) == (1, 1) else 0.37 * pmax
                out.append(f"{i:5d}     He I {a:13s}{j:10d}     He I {b:13s}          "
                           f"{'1s':6s}{'4d':8s}={p:15.7f}      frac= 1.0000       \n")
    for rep in range(2 if reprint else 1):
        head("elec dip", names[1], names[-1])
        for k, (a, b, x) in enumerate(sorted(pairs, key=lambda p: abs(E[p[0]] - E[p[1]])), 1):
            if reprint and k == len(pairs) // 2 and rep == 0:
                head("elec dip", names[1], names[-1])           # page header inside the table
            sig = abs(E[a] - E[b]) * unit_cm
            last = 0.12 if branch else np.sign(x) * 0.6
            out.append(_row(k, a, b, E, lab, conf, unit_cm, x * x / pmax ** 2, RCG_GF_CONSTANT * x * x * sig,
                            last, branch))
    if m1 and data.magnetic:
        for p in (1, -1):
            mags = [m for m in data.magnetic if lv[m.a].parity == p]
            if not mags:
                continue
            head(" mag dip", names[p], names[p])
            for k, m in enumerate(mags, 1):
                out.append(_row(k, m.a, m.b, E, lab, conf, unit_cm, m.reduced ** 2, 0.0, np.sign(m.reduced) * 0.9,
                                False))
    if g_values:
        for p in (1, -1):
            for J in sorted({lv[i].J for i in groups[p]}):
                ids = sorted((i for i in groups[p] if lv[i].J == J), key=lambda i: E[i])
                out.append(f"1  Eigenvalues      (J={J:4.1f})\n")
                for c in range(0, len(ids), 11):
                    out.append("\n" + " " * 28 + "".join(f"{E[i]:9.3f}" for i in ids[c:c + 11]) + "\n")
                out.append("   Config. no. \n" + " " * 25 + "".join(f"{conf[i]:9d}" for i in ids) + "\n")
                out.append("\n\n   g-values\n")
                for c in range(0, len(ids), 11):
                    out.append(" " * 28 + "".join(f"{lv[i].g_J:9.3f}" for i in ids[c:c + 11]) + "\n")
                out.append("\n\n   Eigenvectors   (LS coupling)\n\n")
    path.write_text("".join(out))
    return path


@pytest.fixture(scope="module")
def ref():
    return he_i_4471(1.0, singlets=False)          # the synthetic RCG output holds the triplets


def _key(lv):
    nl = lv.configuration.split()[-1][:2] if lv.configuration.startswith("1s1") else lv.configuration.split()[0]
    return nl, lv.two_j


def _map(ref_data, got):
    by_key = {_key(l): i for i, l in got.levels.items()}
    return {i: by_key[_key(l)] for i, l in ref_data.levels.items()}


def _check_elements(d, got, m, fitted=False):
    """Signs exact; S within the print precision of the better of the gf and S/Pmax^2 columns."""
    pmax2 = max(t.reduced ** 2 for t in d.e1)
    for t in d.e1:
        x, S = got.element(m[t.lower], m[t.upper]), t.reduced ** 2
        sig = abs(d.levels[t.upper].energy_ev - d.levels[t.lower].energy_ev) / EV_PER_CM
        dS = min(5e-6 / (RCG_GF_CONSTANT * sig), 5e-4 * pmax2) + (2e-3 if fitted else 2e-4) * S
        assert np.sign(x) == np.sign(t.reduced)
        assert abs(x * x - S) <= dS


def _read(path, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read_rcg(path, **kw)


def test_elements_signs_and_g_values(tmp_path, ref):
    d = ref.data
    got = _read(write_rcg(d, tmp_path / "he.out"), g_factors="rcg")
    m = _map(d, got)
    assert len(got.levels) == len(d.levels) and len(got.e1) == len(d.e1) and got.all_signed
    for i, l in d.levels.items():
        g = got.levels[m[i]]
        assert g.parity == l.parity and g.two_j == l.two_j
        assert g.energy_ev - got.levels[m[10]].energy_ev == pytest.approx(l.energy_ev - d.levels[10].energy_ev,
                                                                           abs=2e-8)
        assert g.g_J == pytest.approx(l.g_J, abs=5e-4)
    assert got.levels[m[4]].n_valence == 4 and got.levels[m[4]].l_valence == 2
    _check_elements(d, got, m)
    for x in d.magnetic:
        assert got.magnetic_element(m[x.a], m[x.b]) == pytest.approx(x.reduced, rel=1e-3)
    # rates from the printed gf (RCG constants): the 4d -> 2p sum is the NIST A of 447.1 nm
    for i in (4, 5, 6):
        assert got.decay_rate(m[i]) == pytest.approx(d.decay_rate(i), rel=2e-3)
    assert any("worst rel." in n for n in got.notes)


@pytest.mark.parametrize("B", [0.0, 1.0])
def test_rcg_output_reproduces_the_he_i_radiator(tmp_path, ref, B):
    d = ref.data
    got = _read(write_rcg(d, tmp_path / "he.out"), g_factors="rcg")
    m = _map(d, got)
    # decay rates of the reference (RCG's sums miss the channels outside the run)
    levels = {m[i]: dataclasses.replace(got.levels[m[i]], decay_rate_s=d.decay_rate(i)) for i in d.levels}
    got = dataclasses.replace(got, levels=levels)
    up = select_levels(got, "1s1 4s1", "1s1 4p1", "1s1 4d1", "1s1 4f1")
    lo = select_levels(got, "1s1 2s1", "1s1 2p1")
    assert sorted(up) == sorted(m[i] for i in range(10)) and sorted(lo) == sorted(m[i] for i in (10, 11, 12, 13))
    r0 = he_i_4471(B, singlets=False)
    r = JLevelTransition(got, up, lo, B, emitter_charge=0.0, reference_energy_ev=r0.line_energy_ev,
                         collision_n=r0.collision_n, collision_Z=r0.collision_Z,
                         collision_n_lower=r0.collision_n_lower, mass_amu=r0.mass_amu)
    E0 = r0.line_energy_ev
    grid = np.linspace(E0 - 3e-3, E0 + 4e-3, 701)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kw in ({}, dict(electron_operator=True)):
            a = calculate_static_profile(4, 2, 1, B, 1e21, 5.0, grid, num_f=6, num_mu=2, radiator=r0, Ti_ev=2.0, **kw)
            b = calculate_static_profile(4, 2, 1, B, 1e21, 5.0, grid, num_f=6, num_mu=2, radiator=r, Ti_ev=2.0, **kw)
            for x, y in zip(a, b):
                np.testing.assert_allclose(y, x, rtol=0, atol=5e-4 * np.max(x))


def test_pmax_is_fitted_without_the_radial_integrals_and_pages_are_merged(tmp_path, ref):
    d = ref.data
    got = _read(write_rcg(d, tmp_path / "he.out", pmup=False, reprint=True, m1=False, g_values=False))
    m = _map(d, got)
    assert len(got.e1) == len(d.e1) and not got.magnetic
    assert any("fitted" in n for n in got.notes)
    _check_elements(d, got, m, fitted=True)


def test_energy_units_header_and_unit_52_file(tmp_path, ref):
    d = ref.data
    coarse = _read(write_rcg(d, tmp_path / "kk.out", unit_cm=1000.0, m1=False, g_values=False))
    m = _map(d, coarse)
    line = coarse.levels[m[5]].energy_ev - coarse.levels[m[12]].energy_ev
    assert line == pytest.approx(d.levels[5].energy_ev - d.levels[12].energy_ev, abs=0.06 * EV_PER_CM)
    bare = write_rcg(d, tmp_path / "unit52", header=False, g_values=False)
    with pytest.raises(ValueError, match="energy_unit_cm"):
        read_rcg(bare)
    got = _read(bare, energy_unit_cm=1.0)
    assert len(got.e1) == len(d.e1) and len(got.magnetic) == len(d.magnetic)
    assert any("parities assigned" in n for n in got.notes)
    mm = _map_by_energy(d, got)
    for x in d.magnetic:
        assert got.magnetic_element(mm[x.a], mm[x.b]) == pytest.approx(x.reduced, rel=1e-3)


def _map_by_energy(ref_data, got):
    e0r = min(l.energy_ev for l in ref_data.levels.values())
    out = {}
    for i, l in ref_data.levels.items():
        out[i] = min(got.levels, key=lambda j: abs(got.levels[j].energy_ev - (l.energy_ev - e0r))
                     + 10 * abs(got.levels[j].two_j - l.two_j))
    return out


def test_branching_ratio_column_gives_unsigned_elements(tmp_path, ref):
    path = write_rcg(ref.data, tmp_path / "br.out", branch=True, m1=False, g_values=False)
    with pytest.warns(UserWarning, match="without sign"):
        got = read_rcg(path, g_factors="ls")
    assert not got.all_signed
    up = select_levels(got, "1s1 4s1", "1s1 4p1", "1s1 4d1", "1s1 4f1")
    lo = select_levels(got, "1s1 2s1", "1s1 2p1")
    with pytest.raises(ValueError):
        JLevelTransition(got, up, lo, 0.0)


def test_lande_factors_and_parities(tmp_path, ref):
    d = ref.data
    g_L = 1 - 5.48579909e-4 / ref.mass_amu                  # the He I data uses the finite-mass g_L
    assert ls_lande_factor("(2S) 3D", 3.0, g_L=g_L) == pytest.approx(d.levels[6].g_J, rel=1e-6)
    assert ls_lande_factor("2P", 0.5) == pytest.approx(1 - (d.levels[0].g_J - 1) / 3, rel=1e-12)
    assert ls_lande_factor("3P", 0) == 0.0
    ls = _read(write_rcg(d, tmp_path / "he.out", m1=False), g_factors="ls")
    rcg = _read(tmp_path / "he.out", g_factors="rcg")
    for i, l in ls.levels.items():
        assert l.g_J == pytest.approx(rcg.levels[i].g_J, abs=7e-4)     # pure LS vs printed (g_L < 1)
    assert any("pure LS" in n for n in ls.notes)
    assert configuration_parity("1s2 2p5 3d1") == -1
    assert configuration_parity("1s1 4f1") == -1 and configuration_parity("2p6") == 1
    assert configuration_parity("He I") is None
