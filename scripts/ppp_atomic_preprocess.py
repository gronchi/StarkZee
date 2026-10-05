"""Pre-process atomic-structure output into a PPP atomic-database file.

The PPP manual (Calisti, Ferri, Mosse, Talin, March 2024, Sec. 3.3.1) says only
that the ``base`` database comes from MCDF, Cowan, FAC, HULLAC or SUPERSTRUCTURE
"written with a format that can be read by the PPP code"; it does not define that
format. The level and transition packages that ``pim`` writes into ``miscel.``
(Sec. 3.3.3) are documented, so this script emits that layout:

    levg lev ltrs                           (ground levels, total levels, transitions)
    lab  name  energy  spont  popu  2j+1  n     one line per level
    i  j  redu  dlambda  xstrs              one line per transition

Only ``lab, energy, popu, 2j+1, n`` and ``i, j, redu, xstrs`` are used by PPP;
``name, spont, dlambda`` and the fourth transition column are convenience values.
ASSUMPTION: the raw database read by ``pim`` has this same level/transition
structure. Verify against a real ``*.dat`` (e.g. Ar17.dat) before production use.

Input formats
-------------
fac   FAC ASCII tables (e.g. ne.lev / ne.tr from demo/structure/ref), E1.
csv   Generic tables, for converting Cowan / GRASP / HULLAC / SUPERSTRUCTURE
      output with a small per-code script:
        levels.csv       index,energy_ev,twoj,n[,name]
        transitions.csv  lower,upper,redu[,gf,a_s]   (index = levels.csv index)

Conventions
-----------
energy    eV, measured from the lowest level.
redu      reduced dipole matrix element <J||d||J'> in atomic units, recovered
          as |redu|^2 = 3 gf / (2 dE[Ha]). The SIGN IS LOST, yet relative signs
          of the Stark-coupling elements matter for PPP; signed values must come
          from the CSV path or --fac-multipole once that column is verified.
popu      ETL population relative to the lowest level,
          (g/g0) exp(-(E-E0)/kT_e); this reproduces the Ar17.dat example.
xstrs     2 = radiative (ground <-> excited), 1 = Stark coupling (same manifold).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HARTREE_EV = 27.211386245988
KB_EV_PER_K = 8.617333262e-5
HC_EV_ANGSTROM = 12398.419843320026
AU_TIME_S = 2.4188843265857e-17
MAX_BASE_NAME = 12  # PPP manual: "maximum character number = 12"
ORBITALS = "spdfghik"


@dataclass
class Level:
    index: int
    energy_ev: float
    twoj: int
    n: int
    name: str = ""
    ground: bool = False


@dataclass
class Transition:
    lower: int
    upper: int
    redu: float | None = None
    gf: float | None = None
    a_s: float | None = None


# --------------------------------------------------------------------------- readers
_FAC_LEV = re.compile(
    r"^\s*(\d+)\s+(-?\d+)\s+([-+]?\d\.\d+[Ee][-+]?\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(.*\S)\s*$"
)
_FAC_SHELL = re.compile(r"(\d+)([%s])" % ORBITALS)
_FLOAT = r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?"


def read_fac(lev_path: Path, tr_path: Path | None,
             multipole_col: bool = False) -> tuple[list[Level], list[Transition]]:
    levels = []
    for line in lev_path.read_text().splitlines():
        m = _FAC_LEV.match(line)
        if not m:
            continue
        ilev, _ibase, energy, _parity, _vnl, twoj, conf = m.groups()
        shells = _FAC_SHELL.findall(conf)
        if not shells:
            raise ValueError(f"cannot find an nl shell in FAC level line: {line!r}")
        n = max(int(s[0]) for s in shells)
        lorb = shells[-1][1]
        levels.append(Level(int(ilev), float(energy), int(twoj), n,
                            f"{lorb}{lorb}{shells[-1][0]}{lorb}{int(ilev) % 100:02d}"))
    if not levels:
        raise ValueError(f"no FAC levels recognized in {lev_path}")

    transitions = []
    if tr_path is not None:
        # FAC layout (checked on demo/structure/ref/ne.tr, FAC 1.1.5):
        #   UP 2J LOW 2J DE(eV) GF A(1/s) <8th column>
        # The 8th column equals GF in that file, so it is not used as a reduced
        # matrix element unless multipole_col is set.
        row = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)((?:\s+%s){3,4})\s*$" % _FLOAT)
        for line in tr_path.read_text().splitlines():
            m = row.match(line)
            if not m:
                continue
            up, _, low, _ = (int(m.group(i)) for i in range(1, 5))
            vals = [float(v) for v in m.group(5).split()]
            transitions.append(Transition(low, up, redu=vals[3] if multipole_col and len(vals) > 3 else None,
                                          gf=vals[1], a_s=vals[2]))
    return levels, transitions


def read_csv(levels_path: Path, tr_path: Path) -> tuple[list[Level], list[Transition]]:
    with levels_path.open(newline="") as f:
        levels = [Level(int(r["index"]), float(r["energy_ev"]), int(r["twoj"]), int(r["n"]),
                        r.get("name") or "") for r in csv.DictReader(f)]
    with tr_path.open(newline="") as f:
        def opt(r, key):
            return float(r[key]) if r.get(key) not in (None, "") else None
        transitions = [Transition(int(r["lower"]), int(r["upper"]), opt(r, "redu"),
                                  opt(r, "gf"), opt(r, "a_s")) for r in csv.DictReader(f)]
    return levels, transitions


# ------------------------------------------------------------------------ processing
def mark_ground(levels: list[Level], ground_n: int | None, n_ground: int | None) -> None:
    ordered = sorted(levels, key=lambda lv: lv.energy_ev)
    if n_ground is not None:
        for lv in ordered[:n_ground]:
            lv.ground = True
    else:
        n0 = ground_n if ground_n is not None else ordered[0].n
        for lv in levels:
            lv.ground = lv.n == n0
    if not any(lv.ground for lv in levels) or all(lv.ground for lv in levels):
        raise ValueError("ground/excited split is empty; adjust --ground-n / --n-ground")


def build(levels, transitions, te_k, min_redu=0.0):
    """Order levels ground-first, renumber 1..N, and build the PPP transition list."""
    e0 = min(lv.energy_ev for lv in levels)
    ordered = sorted(levels, key=lambda lv: (not lv.ground, lv.energy_ev, lv.index))
    new = {lv.index: k + 1 for k, lv in enumerate(ordered)}
    by_old = {lv.index: lv for lv in levels}
    g0 = by_old[min(levels, key=lambda lv: lv.energy_ev).index].twoj + 1
    kt = KB_EV_PER_K * te_k

    decay = {lv.index: 0.0 for lv in levels}
    rows = []
    for tr in transitions:
        lo, up = by_old.get(tr.lower), by_old.get(tr.upper)
        if lo is None or up is None:
            raise ValueError(f"transition {tr.lower}-{tr.upper} refers to an unknown level")
        if tr.lower == tr.upper:
            continue
        if lo.energy_ev > up.energy_ev:
            lo, up = up, lo
        de = up.energy_ev - lo.energy_ev
        redu = tr.redu
        if redu is None:
            if tr.gf is None or de <= 0:
                raise ValueError(f"transition {tr.lower}-{tr.upper}: need redu or gf")
            redu = math.sqrt(3.0 * tr.gf / (2.0 * de / HARTREE_EV))
        if abs(redu) < min_redu:
            continue
        if tr.a_s is not None and not lo.ground == up.ground:
            decay[up.index] += tr.a_s
        f_abs = (2.0 / 3.0) * (de / HARTREE_EV) * redu**2 / (lo.twoj + 1)
        dlam = HC_EV_ANGSTROM / de if de > 0 else 0.0
        xstrs = 2 if lo.ground != up.ground else 1
        rows.append((new[lo.index], new[up.index], redu, dlam, f_abs, xstrs))
    rows.sort(key=lambda r: (r[0], r[1]))

    level_rows = []
    for lv in ordered:
        g = lv.twoj + 1
        popu = (g / g0) * math.exp(-(lv.energy_ev - e0) / kt) if kt > 0 else float(lv.energy_ev == e0)
        level_rows.append((new[lv.index], lv.name or f"lev{lv.index:03d}", lv.energy_ev - e0,
                           decay[lv.index], popu, g, lv.n))
    return level_rows, rows, sum(lv.ground for lv in levels)


def format_database(level_rows, tr_rows, levg) -> str:
    out = [f"{levg} {len(level_rows)} {len(tr_rows)} (levg,lev,ltrs) ground level,total level & trans. number",
           "lab. /name /energy /spont. /popu. /2j+1 / /n"]
    for lab, name, e, sp, pop, g, n in level_rows:
        out.append(f"{lab:6d} {name:>9s} {e:12.6f} {sp:.5E} {pop:.5E} {g} {n}")
    out.append("i-lab. /j-lab. /redu. / /dlambda /xstrs")
    for i, j, redu, dlam, f, x in tr_rows:
        out.append(f"{i} {j} {redu:.5E} {dlam:.5E} {f:.5E} {x}")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--format", choices=["fac", "csv"], required=True)
    p.add_argument("--levels", type=Path, required=True, help="FAC *.lev.asc or levels.csv")
    p.add_argument("--transitions", type=Path, required=True, help="FAC *.tr.asc or transitions.csv")
    p.add_argument("-o", "--output", type=Path, required=True,
                   help="database file name (PPP limit: 12 characters)")
    p.add_argument("--te", type=float, default=1.0e7, help="electron temperature in K for ETL populations")
    p.add_argument("--ground-n", type=int, help="principal quantum number of the lower manifold "
                   "(default: n of the lowest level)")
    p.add_argument("--n-ground", type=int, help="instead, treat the N lowest levels as ground levels")
    p.add_argument("--fac-multipole", action="store_true",
                   help="take redu from the 8th column of the FAC transition file (signed)")
    p.add_argument("--min-redu", type=float, default=0.0, help="drop transitions with |redu| below this")
    a = p.parse_args(argv)

    if len(a.output.name) > MAX_BASE_NAME:
        p.error(f"PPP limits the database filename to {MAX_BASE_NAME} characters: {a.output.name}")
    if a.format == "fac":
        levels, transitions = read_fac(a.levels, a.transitions, a.fac_multipole)
    else:
        levels, transitions = read_csv(a.levels, a.transitions)
    mark_ground(levels, a.ground_n, a.n_ground)
    level_rows, tr_rows, levg = build(levels, transitions, a.te, a.min_redu)
    a.output.write_text(format_database(level_rows, tr_rows, levg))
    print(f"wrote {a.output}: {levg} ground / {len(level_rows)} levels / {len(tr_rows)} transitions",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
