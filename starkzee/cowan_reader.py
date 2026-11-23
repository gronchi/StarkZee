"""Reader for the spectrum output of Cowan's RCG (atomic structure code, R.D. Cowan).

Written from the RCG print formats and the code that fills them (``spectr.for``,
``sprin.for``, ``energy.for`` of RCG mod 11 as distributed with ADAS801), not
from sample output files. Give RCG's main print output (``out11`` / ``.out``,
read with ``read_rcg``) or ADAS801's spectrum file (unit 52: the same rows
without the page headers, so ``energy_unit_cm`` must then be given).

**Spectrum blocks.** Each multipole block starts with
``<elec dip|mag dip|elec qud> spectrum (Energies in units of U cm-1 = ... eV)``,
the configurations of the two parity groups (``jn  conf --- conf'``) and the
column titles; every line is one row ::

    i  E J conf lab   E' J' conf' lab'   dE  lambda  S/Pmax^2  gf  log gf  gA  cf

E, E' and dE are in units of U cm^-1. The first level (E) belongs to the first
parity group: RCG computes the amplitude <E||P||E'> = sum v P v' with that
level as bra, squares it for S/Pmax^2 and prints the *signed* cancellation factor
cf = sign(amp) amp^2 / (sum |terms|)^2. So sign(cf) is the sign of the reduced
element <E||P||E'> in RCG's phases, and

* E1: S = Pmax^2 (S/Pmax^2) = gf / (3.0375e-6 dE[cm^-1]) [(e a0)^2]
  (RCG's constant: (2/3)/219474.6). Pmax is the largest radial integral of the
  block (printed with the integrals when RCG prints them, ``= value frac=``);
  without it the reader fits Pmax^2 from the lines where both columns are
  precise. For each line the more precise of the two columns is used (gf has
  5 decimals, S/Pmax^2 3); the worst relative uncertainty is recorded in
  ``notes``. All elements of a block share the sign of Pmax, a common phase.
* M1 (``mag dip``): Pmax = 1 and S/Pmax^2 = |<a||L + g_s S||b>|^2 [mu_B^2]
  (RCG's operator is J + 1.00232 S) between levels of one parity group; with
  sign(cf) these are the off-diagonal magnetic elements (Paschen-Back mixing)
  that FAC tables lack. RCG lists each pair once and no diagonal elements.
* E2 blocks are skipped.

With the branching-ratio print option the last column is the branching ratio,
not cf; those elements are read unsigned (with a warning). A cf printed as
0.0000 leaves the sign undetermined (unsigned).

**Levels** are collected from the rows (energy, J, configuration number and
the LS designation of the leading eigenvector component) and numbered by energy
from 0; energy_ev is relative to the lowest level read. Parities come from the
configuration names (sum of l times occupation of tokens like ``4d1``); if
those are missing the two groups get +1 / -1 (only relative parity matters).
Level decay rates are the sums of the listed E1 rates, so they miss the decay
channels outside the run or outside RCG's wavelength and strength limits; RCG's
energies are ab initio (scaled Hartree-Fock), so for line positions replace them
by measured ones (``dataclasses.replace``).

**Lande factors** (``g_factors``): ``"rcg"`` pairs the printed ``g-values``
(LS-coupling eigenvector print, 3 decimals) with the preceding
``Eigenvalues (J= ...)`` list and the levels by J and energy; ``"ls"`` uses
the LS designation of each level (pure LS coupling); or a dict
{level index: g_J}.

RCG must list every pair the radiator needs, including the Stark pairs inside
the upper manifold (small dE, long wavelengths): set its wavelength limits and
strength cut-offs accordingly, and for B > 0 run the ``mag dip`` spectrum too.
"""
import dataclasses
import re
import warnings
from pathlib import Path

import numpy as np
from scipy.constants import c as _C, e as _E, h as _H

from starkzee.level_data import E1Element, JLevel, LevelData, MagneticElement
from starkzee.utils import G_S

EV_PER_CM = _H * _C * 100.0 / _E
RCG_GF_CONSTANT = 3.0375e-6                 # gf = RCG_GF_CONSTANT * S * dE[cm^-1] (E1)
RCG_GA_CONSTANT = 0.66698                   # gA = RCG_GA_CONSTANT * gf * dE[cm^-1]^2
_GF_HALF_DIGIT, _S2_HALF_DIGIT = 5e-6, 5e-4
_L_LETTERS = "spdfghiklmnoqrtuv"
_L_TERM = "SPDFGHIKLMNOQRTUV"

_HEADER = re.compile(r"(elec dip|mag dip|elec qud)\s+spectrum\s*(?:\(Energies in units of\s*([\d.]+)\s*cm-1)?")
_BARE_HEADER = re.compile(r"^\s*(elec dip|mag dip|elec qud)\s*$")
_CONFIG = re.compile(r"^\s*(\d+)\s{6}(.{1,18}?)\s*---\s*(.*?)\s*$")
_PMUP = re.compile(r"=\s*(-?\d+\.\d+)\s+frac=")
_EIGEN = re.compile(r"Eigenvalues\s+\(J=\s*([\d.]+)\)")
_GVAL = re.compile(r"^\s*g-values\s*$")
_ORBITAL = re.compile(r"^(\d+)([spdfghiklmnoqrtuv])(\d*)$")
_TERM = re.compile(r"(\d+)([SPDFGHIKLMNOQRTUV])")
# fixed columns of RCG formats 53/58/88 after the carriage-control character
_ROW = [("i", 4), ("E", 11), ("J", 5), ("c", 3), (None, 1), ("lab", 8), ("Ep", 13), ("Jp", 5), ("cp", 3),
        (None, 1), ("labp", 8), ("dE", 13), ("lam", 12), ("s2", 8), ("gf", 9), ("x", 18)]


def _split_row(line):
    """Fields of a spectrum row, or None. Tries with and without the carriage-control column."""
    for start in (1, 0):
        pos, out = start, {}
        for name, width in _ROW:
            if name:
                out[name] = line[pos:pos + width]
            pos += width
        out["last"] = line[pos:].strip()
        out["Estr"], out["Epstr"] = out["E"].strip(), out["Ep"].strip()
        try:
            int(out["i"])
            for k in ("E", "J", "Ep", "Jp", "dE", "s2", "gf"):
                out[k] = float(out[k])
            out["c"], out["cp"] = int(out["c"]), int(out["cp"])
            float(out["last"])
        except ValueError:
            continue
        if 2 * out["J"] != round(2 * out["J"]) or 2 * out["Jp"] != round(2 * out["Jp"]):
            continue
        out["key"] = (out["Estr"], out["J"], out["c"], out["lab"], out["Epstr"], out["Jp"], out["cp"], out["labp"])
        return out
    return None


def configuration_parity(name):
    """(-1)^(sum l w) of a configuration name like '1s2 2p5 3d1' (None if no orbital tokens)."""
    total, found = 0, False
    for token in name.split():
        m = _ORBITAL.match(token.lower())
        if m:
            found = True
            total += _L_LETTERS.index(m.group(2)) * (int(m.group(3)) if m.group(3) else 1)
    return None if not found else (1 if total % 2 == 0 else -1)


def _valence(name):
    for token in reversed(name.split()):
        m = _ORBITAL.match(token.lower())
        if m:
            return int(m.group(1)), _L_LETTERS.index(m.group(2))
    return None, None


def ls_lande_factor(designation, J, g_s=G_S, g_L=1.0):
    """Lande g_J in pure LS coupling of the last term ``2S+1 L`` in an RCG designation (0 for J = 0)."""
    terms = _TERM.findall(designation)
    if not terms:
        raise ValueError(f"no LS term in designation {designation!r}")
    mult, letter = terms[-1]
    S, L = (int(mult) - 1) / 2, _L_TERM.index(letter)
    if J == 0:
        return 0.0
    x = J * (J + 1)
    return (g_L * (x + L * (L + 1) - S * (S + 1)) + g_s * (x + S * (S + 1) - L * (L + 1))) / (2 * x)


def _same_block(a, b):
    """Whether piece b continues block a (page header, title page or the sorted reprint)."""
    if a["kind"] != b["kind"]:
        return False
    if None not in (a["unit_cm"], b["unit_cm"]) and a["unit_cm"] != b["unit_cm"]:
        return False
    return not any(a["configs"]) or not any(b["configs"]) or a["configs"] == b["configs"]


def parse_rcg_spectra(path, energy_unit_cm=None):
    """Spectrum blocks of an RCG print file (or ADAS801 unit-52 file) as a list of dicts.

    Each block: kind ('elec dip', 'mag dip', 'elec qud'), unit_cm, configs
    (two dicts {number: name}, empty without headers), pmup (radial integrals
    printed in the block) and rows (dicts of the row fields).
    """
    pieces, cur = [], None
    for line in Path(path).read_text(errors="replace").splitlines():
        m = _HEADER.search(line)
        bare = None if m else _BARE_HEADER.match(line)
        if m or bare:
            kind = (m or bare).group(1)
            unit = float(m.group(2)) if m and m.group(2) else None
            cur = {"kind": kind, "unit_cm": unit, "configs": ({}, {}), "pmup": [], "rows": []}
            pieces.append(cur)
            continue
        if cur is None:
            continue
        p = _PMUP.search(line)
        if p:
            cur["pmup"].append(float(p.group(1)))
            continue
        row = _split_row(line)
        if row is not None:
            cur["rows"].append(row)
            continue
        c = _CONFIG.match(line)
        if c:
            cur["configs"][0][int(c.group(1))] = c.group(2).strip()
            if c.group(3):
                cur["configs"][1][int(c.group(1))] = c.group(3).strip()
    # RCG repeats the header on every page and may print a table twice (sorted pass):
    # merge consecutive pieces of the same kind with compatible units and configurations
    blocks = []
    for p in pieces:
        b = blocks[-1] if blocks else None
        if b is not None and _same_block(b, p):
            b["unit_cm"] = b["unit_cm"] if b["unit_cm"] is not None else p["unit_cm"]
            if not any(b["configs"]):
                b["configs"] = p["configs"]
            b["pmup"].extend(p["pmup"])
            seen = {r["key"] for r in b["rows"]}
            b["rows"].extend(r for r in p["rows"] if r["key"] not in seen)
        else:
            blocks.append(p)
    for b in blocks:
        if b["unit_cm"] is None:
            if energy_unit_cm is None:
                raise ValueError(f"'{b['kind']}' block has no energy-unit header: give energy_unit_cm")
            b["unit_cm"] = float(energy_unit_cm)
    return blocks


def parse_rcg_g_values(path):
    """[(J, eigenvalues, g-values)] from the 'Eigenvalues (J= ...)' and 'g-values' prints."""
    lines = Path(path).read_text(errors="replace").splitlines()
    out, last, i = [], None, 0

    def numbers(k):
        """Rows (28x, 11f9.x), separated by blank lines, up to the next text line."""
        vals = []
        while k < len(lines):
            s = lines[k]
            if not s.strip():
                k += 1
                continue
            if s[:28].strip():
                break
            body = s[28:].rstrip()
            try:
                vals.extend(float(body[j:j + 9]) for j in range(0, len(body), 9))
            except ValueError:
                break
            k += 1
        return vals, k

    while i < len(lines):
        m = _EIGEN.search(lines[i])
        if m:
            vals, i = numbers(i + 1)
            last = (float(m.group(1)), vals)
            continue
        if _GVAL.match(lines[i]):
            vals, i = numbers(i + 1)
            if last is not None and len(vals) == len(last[1]):
                out.append((last[0], last[1], vals))
            continue
        i += 1
    return out


def read_rcg(paths, energy_unit_cm=None, g_factors=None, energy_tolerance=None):
    """Read RCG spectra (one file or a list, e.g. separate E1 and M1 runs) into LevelData.

    g_factors : None, ``"rcg"``, ``"ls"`` or dict {level index: g_J}.
    energy_tolerance : match tolerance for g-value eigenvalues [RCG units]
        (default 0.006, for the f9.2 print).
    """
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    blocks = [b for p in paths for b in parse_rcg_spectra(p, energy_unit_cm)]
    if not blocks:
        raise ValueError("no RCG spectrum block found")
    notes = []
    raw = {}             # level key -> dict(E, two_j, conf name, label, group)
    pending_e1, pending_m1 = [], []
    unsigned = poor = 0
    group_names = {}     # group key -> ordinal (for arbitrary parities)

    def group_key(configs, side, block_id):
        names = tuple(sorted(configs[side].items()))
        return names if names else ("block", block_id, side)

    def level(row, primed, gkey, configs, side, unit):
        E, J, c, lab, Es = ((row["Ep"], row["Jp"], row["cp"], row["labp"], row["Epstr"]) if primed else
                            (row["E"], row["J"], row["c"], row["lab"], row["Estr"]))
        key = (gkey, c, lab.strip(), int(round(2 * J)), Es)
        if key not in raw:
            raw[key] = dict(E_cm=E * unit, E_units=E, unit=unit, two_j=int(round(2 * J)), conf=c,
                            name=configs[side].get(c, ""), label=lab.strip(), group=gkey)
            group_names.setdefault(gkey, len(group_names))
        return key

    def sign_of(text):
        try:
            v = float(text)
        except ValueError:
            return None
        if "E" in text.upper():
            return None                   # branching ratio column
        if text.strip().startswith("-"):
            return -1.0
        return 1.0 if v > 0 else None

    for bid, b in enumerate(blocks):
        if b["kind"] == "elec qud":
            notes.append("RCG E2 block skipped")
            continue
        unit, rows, configs = b["unit_cm"], b["rows"], b["configs"]
        m1 = b["kind"] == "mag dip"
        g1 = group_key(configs, 0, bid)
        g2 = g1 if m1 else group_key(configs, 1, bid)
        if m1 and g1[0] == "block":
            g1 = g2 = None               # resolved against the E1 levels below
        if not m1:
            pmax2 = max(abs(x) for x in b["pmup"]) ** 2 if b["pmup"] else None
            if pmax2 is None:
                ratios, weights = [], []
                for r in rows:
                    sig = abs(r["dE"]) * unit
                    if r["gf"] > 0 and r["s2"] > 0 and sig > 0:
                        e2 = (_GF_HALF_DIGIT / r["gf"]) ** 2 + (_S2_HALF_DIGIT / r["s2"]) ** 2
                        ratios.append(r["gf"] / (RCG_GF_CONSTANT * sig * r["s2"]))
                        weights.append(1.0 / e2)
                if ratios:
                    pmax2 = float(np.average(ratios, weights=weights))
                    rel = float(np.sqrt(1.0 / np.sum(weights)))
                    notes.append(f"Pmax^2 = {pmax2:.6g} fitted from gf / (S/Pmax^2) (rel. unc. {rel:.1e})")
            else:
                chk = [r["gf"] / (RCG_GF_CONSTANT * abs(r["dE"]) * unit * r["s2"]) / pmax2
                       for r in rows if r["gf"] > 1e-2 and r["s2"] > 0.1 and r["dE"] != 0]
                if chk and abs(np.median(chk) - 1.0) > 0.01:
                    warnings.warn(f"RCG gf column is {np.median(chk):.3f} x Pmax^2 S/Pmax^2: using S/Pmax^2 only "
                                  "(gf not the E1 oscillator strength in this run?)", UserWarning, stacklevel=2)
                    for r in rows:
                        r["gf"] = 0.0
        for r in rows:
            if m1 and g1 is None:
                k1 = k2 = None
            else:
                k1 = level(r, False, g1, configs, 0, unit)
                k2 = level(r, True, g2, configs, 0 if m1 else 1, unit)
            s = sign_of(r["last"])
            if m1:
                pending_m1.append((r, k1, k2, s))
                continue
            sig = abs(r["dE"]) * unit
            cands = []
            if r["gf"] > 0 and sig > 0:
                cands.append((_GF_HALF_DIGIT / r["gf"], r["gf"] / (RCG_GF_CONSTANT * sig)))
            if r["s2"] > 0 and pmax2:
                cands.append((_S2_HALF_DIGIT / r["s2"], pmax2 * r["s2"]))
            if not cands:
                continue                 # below the print precision
            rel, S = min(cands)
            pending_e1.append((k1, k2, S, rel, s, sig))
            poor += rel > 0.02
    if not raw:
        raise ValueError("no E1 levels read")

    # parities
    parity = {}
    names_ok = True
    for gkey in group_names:
        ps = {configuration_parity(v["name"]) for v in raw.values() if v["group"] == gkey}
        if len(ps) == 1 and None not in ps:
            parity[gkey] = ps.pop()
        else:
            names_ok = False
    if not names_ok:
        for gkey, k in group_names.items():
            parity[gkey] = 1 if k % 2 == 0 else -1
        notes.append("parities assigned per RCG parity group (configuration names missing)")

    # numbering by energy
    keys = sorted(raw, key=lambda k: (raw[k]["E_cm"], raw[k]["two_j"]))
    e_min = raw[keys[0]]["E_cm"]
    index = {k: i for i, k in enumerate(keys)}
    levels = {}
    for k, i in index.items():
        v = raw[k]
        n, l = _valence(v["name"])
        levels[i] = JLevel(index=i, energy_ev=(v["E_cm"] - e_min) * EV_PER_CM, two_j=v["two_j"],
                           parity=parity[v["group"]], configuration=v["name"], label=v["label"],
                           n_valence=n, l_valence=l)

    def flip(a, b):
        return -1.0 if ((levels[a].two_j - levels[b].two_j) // 2) % 2 else 1.0

    e1, worst = {}, 0.0
    for k1, k2, S, rel, s, sig in pending_e1:
        a, b = index[k1], index[k2]                        # <a||d||b>, a in the first parity group
        amp = np.sqrt(S) * (s if s is not None else 1.0)
        lo, up = (a, b) if levels[a].energy_ev <= levels[b].energy_ev else (b, a)
        red = amp if lo == a else flip(b, a) * amp         # <lo||d||up> = (-1)^(J_b - J_a) <b..> reversed
        gf = RCG_GF_CONSTANT * S * sig
        rate = RCG_GA_CONSTANT * gf * sig ** 2 / (levels[up].two_j + 1)
        worst = max(worst, rel)
        unsigned += s is None
        e1[(lo, up)] = E1Element(lower=lo, upper=up, reduced=float(red), signed=s is not None, gf=gf,
                                 rate_s=rate, energy_ev=levels[up].energy_ev - levels[lo].energy_ev)
    if unsigned:
        warnings.warn(f"{unsigned} RCG E1 elements without sign (branching-ratio column or cf = 0)",
                      UserWarning, stacklevel=2)
    notes.append(f"E1 strengths from the gf / S/Pmax^2 columns, worst rel. print uncertainty {worst:.1e}")
    if poor:
        warnings.warn(f"{poor} RCG E1 line strengths have a print uncertainty above 2 % "
                      "(weak lines with small dE: gf has 5 decimals, S/Pmax^2 3)", UserWarning, stacklevel=2)

    magnetic, worst_m = {}, 0.0
    by_row = {}
    for k, i in index.items():
        v = raw[k]
        by_row.setdefault((v["conf"], v["label"], v["two_j"]), []).append((v["E_units"], i))
    for r, k1, k2, s in pending_m1:
        if k1 is None:
            pair = []
            for E, c, lab, J in ((r["E"], r["c"], r["lab"], r["J"]), (r["Ep"], r["cp"], r["labp"], r["Jp"])):
                cand = [i for Ev, i in by_row.get((c, lab.strip(), int(round(2 * J))), []) if Ev == E]
                if len(cand) != 1:
                    raise ValueError(f"M1 row {r['i']}: level not uniquely found among the E1 levels")
                pair.append(cand[0])
            a, b = pair
        else:
            a, b = index[k1], index[k2]
        if r["s2"] <= 0 or a == b:
            continue
        if s is None:
            warnings.warn(f"M1 element {a}-{b} without sign: skipped", UserWarning, stacklevel=2)
            continue
        worst_m = max(worst_m, _S2_HALF_DIGIT / r["s2"])
        magnetic[frozenset((a, b))] = MagneticElement(a=a, b=b, reduced=s * float(np.sqrt(r["s2"])))
    if pending_m1:
        notes.append(f"magnetic elements from the RCG M1 spectrum, worst rel. print uncertainty {worst_m:.1e}")

    data = LevelData(levels=levels, e1=list(e1.values()), magnetic=list(magnetic.values()),
                     source="RCG: " + ", ".join(str(Path(p).name) for p in paths), notes=notes)
    if g_factors is not None:
        data = _with_g(data, raw, index, g_factors, paths, energy_tolerance)
    return data.validate(energy_rtol=1e-3)


def _with_g(data, raw, index, g_factors, paths, tol):
    levels = dict(data.levels)
    if g_factors == "ls":
        for i, lv in levels.items():
            levels[i] = dataclasses.replace(lv, g_J=ls_lande_factor(lv.label, lv.J))
        data.notes.append("g_J in pure LS coupling from the RCG designations (mixing ignored)")
    elif g_factors == "rcg":
        tol = 0.006 if tol is None else tol
        blocks = [g for p in paths for g in parse_rcg_g_values(p)]
        if not blocks:
            raise ValueError("no 'g-values' print found (RCG LS eigenvector print)")
        missing, ambiguous = [], []
        inverse = {i: k for k, i in index.items()}
        for i, lv in levels.items():
            E = raw[inverse[i]]["E_units"]
            hits = [g for J, eig, gs in blocks if int(round(2 * J)) == lv.two_j
                    for e, g in zip(eig, gs) if abs(e - E) <= tol]
            if len(hits) == 1:
                levels[i] = dataclasses.replace(lv, g_J=float(hits[0]))
            elif hits and np.ptp(hits) < 1.5e-3:
                levels[i] = dataclasses.replace(lv, g_J=float(np.mean(hits)))
            elif hits:
                ambiguous.append(i)
            else:
                missing.append(i)
        if missing or ambiguous:
            warnings.warn(f"RCG g-values: no match for levels {missing}, ambiguous for {ambiguous} (g_J None)",
                          UserWarning, stacklevel=3)
        data.notes.append("g_J from the RCG g-values print (3 decimals)")
    else:
        for i, g in g_factors.items():
            levels[i] = dataclasses.replace(levels[i], g_J=float(g))
    return dataclasses.replace(data, levels=levels)


def select_levels(data, *configurations):
    """Indices of the levels whose configuration name is one of ``configurations`` (sorted by energy)."""
    want = {" ".join(c.split()) for c in configurations}
    return [i for i, lv in sorted(data.levels.items(), key=lambda t: t[1].energy_ev)
            if " ".join(lv.configuration.split()) in want]
