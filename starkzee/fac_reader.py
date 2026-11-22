"""Reader for ASCII tables of the Flexible Atomic Code (FAC, M.F. Gu).

Written from the FAC table layout (header fields and the ``PrintENTable`` /
``PrintTRTable`` formats of ``faclib/dbase.c``), not from FAC's own reader code.
Use verbose tables, ``PrintTable('x.lev.b', 'x.lev', 1)`` and
``PrintTable('x.tr.b', 'x.tr', 1)``.

Levels (``.lev`` / ``.en``): header ``E0 = index, energy`` (absolute energy of the
reference level in eV), then per block ``NELE``, ``NLEV`` and rows
``ILEV IBASE ENERGY P VNL 2J NCOMPLEX SNAME NAME`` with ENERGY in eV relative to E0,
P = 0 even / 1 odd, VNL = 100 n + l of the valence electron.

Transitions (``.tr``): per block ``NELE``, ``NTRANS``, ``MULTIP``, ``GAUGE``,
``MODE`` and rows ``UPPER 2J LOWER 2J dE gf A M`` (UTA tables add a sdev column
after dE and a CI factor at the end). What M is depends on MULTIP:

* ``MULTIP = -1`` (E1): M is FAC's reduced matrix element computed with the
  lower level as bra, signed, with gf = M^2 dE[Ha] / 3. In the Edmonds
  convention <lower||d||upper> = M / sqrt(2) (gf = (2/3) dE S). The reader
  checks this relation and keeps the sign.
* ``MULTIP = 0``: all multipoles summed; M = gf, no sign. Elements are unsigned
  (|<J||d||J'>| from gf) and include non-E1 contributions.

Signs come from FAC's eigenvector phases: they are consistent within one
structure calculation (one level set), not between separate runs.
"""
import re
from pathlib import Path

import numpy as np

from starkzee.level_data import E1Element, JLevel, LevelData, reduced_from_gf
from starkzee.utils import HARTREE_EV

_KEY = re.compile(r"^\s*([A-Za-z0-9 ]+?)\s*=\s*(.*?)\s*$")


def _header(lines):
    """FAC version line and the key = value header up to NBlocks; returns (dict, next index)."""
    if not lines or not lines[0].startswith("FAC"):
        raise ValueError("not a FAC ASCII table (first line must start with 'FAC')")
    head = {"FAC": lines[0].strip()[3:].strip()}
    i = 1
    while i < len(lines):
        m = _KEY.match(lines[i])
        if not m:
            break
        key, value = m.groups()
        head[key] = value
        i += 1
        if key == "NBlocks":
            break
    for key in ("Type", "Verbose", "NBlocks"):
        if key not in head:
            raise ValueError(f"FAC header lacks '{key}'")
    z_key = next((k for k in head if k.endswith(" Z")), None)
    head["Z"] = float(head[z_key]) if z_key else None
    return head, i


def _blocks(lines, i):
    """Yield (block header dict, list of data rows) for the blocks after the file header."""
    n = len(lines)
    while i < n:
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            return
        block = {}
        while i < n and _KEY.match(lines[i]) and not lines[i].lstrip()[0].isdigit():
            key, value = _KEY.match(lines[i]).groups()
            block[key] = value
            i += 1
        rows = []
        while i < n and lines[i].strip():
            if lines[i].strip().startswith("ILEV"):      # column title line of level tables
                i += 1
                continue
            rows.append(lines[i])
            i += 1
        yield block, rows


def read_fac_levels(path):
    """Read a verbose FAC level table. Returns (header, list of JLevel)."""
    lines = Path(path).read_text().splitlines()
    head, i = _header(lines)
    if int(head["Verbose"]) != 1:
        raise ValueError("FAC level table must be verbose (PrintTable(..., 1))")
    if i < len(lines) and lines[i].startswith("E0"):
        index, energy = lines[i].split("=", 1)[1].split(",")
        head["E0_index"], head["E0_ev"] = int(index), float(energy)
        i += 1
    levels = []
    for block, rows in _blocks(lines, i):
        nele = int(block["NELE"]) if "NELE" in block else None
        for row in rows:
            a = row.split()
            if len(a) < 6:
                raise ValueError(f"short FAC level row: {row!r}")
            ilev, _, energy, p, vnl, twoj = int(a[0]), a[1], float(a[2]), int(a[3]), int(a[4]), int(a[5])
            rest = a[6:]
            levels.append(JLevel(index=ilev, energy_ev=energy, two_j=twoj, parity=-1 if p else 1,
                                 configuration=rest[1] if len(rest) > 1 else "",
                                 label=rest[2] if len(rest) > 2 else (rest[-1] if rest else ""),
                                 n_valence=vnl // 100, l_valence=vnl % 100, nele=nele))
        if "NLEV" in block and int(block["NLEV"]) != len(rows):
            raise ValueError(f"block NELE={nele}: NLEV = {block['NLEV']} but {len(rows)} rows")
    return head, levels


def read_fac_transitions(path):
    """Read a verbose FAC transition table. Returns (header, list of block dicts with arrays)."""
    lines = Path(path).read_text().splitlines()
    head, i = _header(lines)
    if int(head["Verbose"]) != 1:
        raise ValueError("FAC transition table must be verbose (PrintTable(..., 1))")
    blocks = []
    for block, rows in _blocks(lines, i):
        data = np.array([[float(x) for x in r.split()] for r in rows]) if rows else np.zeros((0, 8))
        ncol = data.shape[1]
        if ncol not in (8, 10):
            raise ValueError(f"unexpected number of columns ({ncol}) in FAC transition table")
        uta = ncol == 10
        cols = dict(upper=0, upper_2J=1, lower=2, lower_2J=3, energy_ev=4)
        cols.update(dict(sdev_ev=5, gf=6, rate_s=7, M=8, ci=9) if uta else dict(gf=5, rate_s=6, M=7))
        out = {k: int(v) if k in ("NELE", "NTRANS", "MULTIP", "GAUGE", "MODE") else v
               for k, v in block.items()}
        out["uta"] = uta
        for name, c in cols.items():
            out[name] = data[:, c].astype(int) if name in ("upper", "upper_2J", "lower", "lower_2J") else data[:, c]
        if "NTRANS" in out and out["NTRANS"] != len(rows):
            raise ValueError(f"NTRANS = {out['NTRANS']} but {len(rows)} rows")
        blocks.append(out)
    return head, blocks


def read_fac(lev_path, tr_path=None, gf_rtol=1e-4):
    """Levels and E1 elements of a FAC calculation as :class:`~starkzee.level_data.LevelData`.

    E1 blocks (MULTIP = -1) give signed elements <lower||d||upper> = M / sqrt(2),
    checked against gf to ``gf_rtol`` (M is stored in single precision). MULTIP = 0
    blocks give unsigned magnitudes from gf (all multipoles summed); other
    multipoles are skipped.
    """
    head, levels = read_fac_levels(lev_path)
    data = LevelData(levels={lv.index: lv for lv in levels}, source=f"FAC {head['FAC']}: {Path(lev_path).name}",
                     Z=head["Z"], reference_energy_ev=head.get("E0_ev"))
    if tr_path is None:
        return data
    _, blocks = read_fac_transitions(tr_path)
    for b in blocks:
        m = b.get("MULTIP")
        if m not in (-1, 0):
            data.notes.append(f"skipped MULTIP = {m} block ({len(b['gf'])} transitions)")
            continue
        if m == 0:
            data.notes.append("MULTIP = 0: unsigned magnitudes from gf (all multipoles summed)")
        for k in range(len(b["gf"])):
            lo, up = int(b["lower"][k]), int(b["upper"][k])
            de, gf, M = float(b["energy_ev"][k]), float(b["gf"][k]), float(b["M"][k])
            if m == -1:
                implied = M * M * (de / HARTREE_EV) / 3.0
                if not np.isclose(implied, gf, rtol=gf_rtol, atol=0.0):
                    raise ValueError(f"FAC E1 {lo}-{up}: gf = {gf} but M^2 dE/3 = {implied}; "
                                     "the M column is not a reduced E1 element")
                reduced, signed = M / np.sqrt(2.0), True
            else:
                reduced, signed = reduced_from_gf(gf, de), False
            data.e1.append(E1Element(lo, up, reduced, signed, gf=gf, rate_s=float(b["rate_s"][k]),
                                     energy_ev=de))
    return data.validate() if all(t.signed for t in data.e1) else data
