"""Download and normalize NIST ASD level data for StarkZee.

This script is intentionally a data-refresh tool, not part of the runtime
profile calculation path.  StarkZee should continue to read the committed
``starkzee/data/atomic_levels.json`` file for reproducible offline runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

NIST_LEVELS_URL = "https://physics.nist.gov/cgi-bin/ASD/energy1.pl"
L_SYMBOLS = {
    "s": 0,
    "p": 1,
    "d": 2,
    "f": 3,
    "g": 4,
    "h": 5,
    "i": 6,
    "k": 7,
    "l": 8,
    "m": 9,
}


@dataclass(frozen=True)
class ParsedNistLevels:
    """Normalized levels matching ``starkzee/data/atomic_levels.json``."""

    fine_structure_true: list[dict]
    fine_structure_false: list[dict]


class NistParseError(ValueError):
    """Raised when a NIST ASD response cannot be normalized."""


def default_data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "starkzee" / "data" / "atomic_levels.json"


def build_nist_levels_url(spectrum: str) -> str:
    """Return a NIST ASD levels query URL with cm^-1 tab-delimited output."""

    params = {
        "spectrum": spectrum,
        "units": "0",  # cm^-1
        "format": "3",  # tab-delimited text
        "output": "0",  # entire result
        "page_size": "15",
        "multiplet_ordered": "0",
        "conf_out": "on",
        "term_out": "on",
        "level_out": "on",
        "j_out": "on",
        "submit": "Retrieve Data",
    }
    return f"{NIST_LEVELS_URL}?{urlencode(params)}"


def fetch_nist_levels_text(spectrum: str, timeout: float = 30.0) -> str:
    """Fetch raw tab-delimited NIST ASD level data for ``spectrum``."""

    url = build_nist_levels_url(spectrum)
    request = Request(url, headers={"User-Agent": "StarkZee NIST level updater"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - explicit user-run data updater
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def parse_fraction(value: str) -> float | None:
    value = value.strip().strip('"')
    if not value or value == "---":
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        return float(num) / float(den)
    return float(value)


def parse_configuration(config: str) -> tuple[int, int | None] | None:
    """Parse NIST configurations like ``4f`` or term-average rows like ``4``."""

    config = config.strip().strip('"').lower()
    if not config:
        return None
    match = re.fullmatch(r"(\d+)([a-z]?)", config)
    if not match:
        return None
    n = int(match.group(1))
    letter = match.group(2)
    if not letter:
        return n, None
    if letter not in L_SYMBOLS:
        return None
    return n, L_SYMBOLS[letter]


def parse_energy(value: str) -> float | None:
    value = value.strip().strip('"').replace(" ", "")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _sorted_unique(rows: Iterable[dict], keys: tuple[str, ...]) -> list[dict]:
    seen = set()
    out = []
    for row in rows:
        ident = tuple(row.get(key) for key in keys)
        if ident in seen:
            continue
        seen.add(ident)
        out.append(row)
    return sorted(out, key=lambda r: tuple(r.get(key, -1) for key in keys))


def parse_nist_levels(text: str, max_n: int | None = None) -> ParsedNistLevels:
    """Parse NIST ASD tab-delimited levels into StarkZee JSON sections.

    Expected columns come from ``build_nist_levels_url`` and include
    ``Configuration``, ``J``, and ``Level (cm-1)``.  Rows with configurations
    like ``4f`` and a valid ``J`` become fine-structure levels.  Rows with
    configurations like ``4`` and no ``J`` become shell-averaged levels.
    """

    if "<html" in text.lower() or "Error Message" in text:
        raise NistParseError("NIST response appears to be HTML/error output, not tab-delimited level data.")

    reader = csv.DictReader(text.splitlines(), delimiter="\t")
    if not reader.fieldnames:
        raise NistParseError("NIST response did not contain a tab-delimited header.")

    fine_rows: list[dict] = []
    average_rows: list[dict] = []
    energy_column = next((name for name in reader.fieldnames if name.startswith("Level")), None)
    if energy_column is None:
        raise NistParseError(f"Could not find level-energy column in {reader.fieldnames!r}.")

    for row in reader:
        parsed_config = parse_configuration(row.get("Configuration", ""))
        energy = parse_energy(row.get(energy_column, ""))
        if parsed_config is None or energy is None:
            continue
        n, l_value = parsed_config
        if max_n is not None and n > max_n:
            continue
        j_value = parse_fraction(row.get("J", ""))
        if l_value is None:
            if j_value is None:
                average_rows.append({"n": n, "energy": energy})
            continue
        if j_value is None:
            continue
        fine_rows.append({"n": n, "l": l_value, "j": j_value, "energy": energy})

    if not fine_rows and not average_rows:
        raise NistParseError("No usable level rows were found in the NIST response.")

    return ParsedNistLevels(
        fine_structure_true=_sorted_unique(fine_rows, ("n", "l", "j")),
        fine_structure_false=_sorted_unique(average_rows, ("n",)),
    )


def find_incomplete_fine_structure_shells(levels: ParsedNistLevels) -> dict[int, list[tuple[int, float]]]:
    """Return missing ``(l, j)`` states for hydrogenic fine-structure shells."""

    present_by_n: dict[int, set[tuple[int, float]]] = {}
    for row in levels.fine_structure_true:
        present_by_n.setdefault(int(row["n"]), set()).add((int(row["l"]), float(row["j"])))

    missing: dict[int, list[tuple[int, float]]] = {}
    for n in sorted(present_by_n):
        expected: list[tuple[int, float]] = [(0, 0.5)]
        for l_value in range(1, n):
            expected.append((l_value, l_value - 0.5))
            expected.append((l_value, l_value + 0.5))
        absent = [state for state in expected if state not in present_by_n[n]]
        if absent:
            missing[n] = absent
    return missing


def format_incomplete_shells(missing: dict[int, list[tuple[int, float]]]) -> str:
    parts = []
    for n, states in missing.items():
        state_text = ", ".join(f"l={l_value},j={j_value:g}" for l_value, j_value in states)
        parts.append(f"n={n}: {state_text}")
    return "; ".join(parts)

def load_database(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_database(path: Path, database: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(database, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def update_database(database: dict, atom: str, spectrum: str, levels: ParsedNistLevels) -> dict:
    updated = dict(database)
    updated[atom] = {
        "fine_structure_false": levels.fine_structure_false,
        "fine_structure_true": levels.fine_structure_true,
        "metadata": {
            "source": "NIST ASD levels query",
            "spectrum": spectrum,
            "units": "cm^-1",
            "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "query_url": build_nist_levels_url(spectrum),
        },
    }
    return updated


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NIST ASD levels into StarkZee's atomic_levels.json schema.")
    parser.add_argument("atom", help="Atom key to update in atomic_levels.json, e.g. H")
    parser.add_argument("--spectrum", help="NIST ASD spectrum name, e.g. 'H I'. Defaults to '<atom> I'.")
    parser.add_argument("--max-n", type=int, default=6, help="Maximum principal quantum number to keep (default: 6, the range NIST returns as complete fine-structure shells for H I).")
    parser.add_argument("--output", type=Path, default=default_data_path(), help="JSON database path to update.")
    parser.add_argument("--input", type=Path, dest="input_path", help="Parse a saved NIST tab-delimited response instead of downloading.")
    parser.add_argument("--dry-run", action="store_true", help="Print normalized JSON for this atom without writing the database.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Allow writing fine-structure shells with missing (l, j) states.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spectrum = args.spectrum or f"{args.atom} I"
    if args.input_path:
        text = args.input_path.read_text(encoding="utf-8")
    else:
        text = fetch_nist_levels_text(spectrum)
    levels = parse_nist_levels(text, max_n=args.max_n)
    missing = find_incomplete_fine_structure_shells(levels)
    if missing and not args.allow_incomplete:
        raise NistParseError(
            "NIST response is missing fine-structure states required for complete hydrogenic shells: "
            + format_incomplete_shells(missing)
            + ". Use --max-n to restrict the range or --allow-incomplete to write anyway."
        )
    if missing:
        print("Warning: incomplete fine-structure shells: " + format_incomplete_shells(missing), file=sys.stderr)
    if args.dry_run:
        print(json.dumps(asdict(levels), indent=2, ensure_ascii=False))
        return 0
    database = load_database(args.output)
    updated = update_database(database, args.atom, spectrum, levels)
    write_database(args.output, updated)
    print(
        f"Updated {args.output} for {args.atom}: "
        f"{len(levels.fine_structure_true)} fine-structure levels, "
        f"{len(levels.fine_structure_false)} shell-average levels."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


