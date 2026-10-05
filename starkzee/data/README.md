# Bundled reference data

`reference_tables.json` is the machine-readable provenance registry for every
runtime reference-data file in this directory. It records byte size, SHA-256,
Git history, units, the closest archived source, and unresolved provenance
fields. An `unknown` value is deliberate; it must not be replaced by an
inference.

`bibliography.json` is the completed metadata audit for the 22 works represented
by the manual's 21 citation keys. `historical_provenance.json` identifies the
exact Git commit and blobs matching the separately audited standalone manual,
its PDF, figure and generator, and its runtime datasets. The latter can be
checked from a full clone with
`python scripts/verify_historical_provenance.py`. Repository identity does not
by itself establish scientific accuracy or external-data licensing.

The Rosato NetCDF has no embedded attributes. Its complete contents were
compared with the 3,000 raw text profiles at pystark commit
`9a8782ae041493f7fce6728f2636ca20f26b0a0c`: every parsed detuning and
intensity value matched exactly. This establishes the data mapping but not the
unarchived original transfer from the table authors, a data license, numerical
uncertainties, or the physical meaning of its signed fluctuations. The
intensity unit is interpreted as inverse eV from the area-normalized line-shape
use; the archived files do not explicitly state it.

The Stehle NetCDF is byte-identical to the file at the same pinned pystark
commit. That source repository contains the CDS/VizieR VI/98A raw distribution
and its text-to-NetCDF conversion script. The CDS ReadMe supplies the units and
profile layout. It does not supply a per-profile uncertainty field.

`atomic_levels.json` contains source URLs and acquisition timestamps for D and
T. Its H section predates that metadata convention. `scripts/update_nist_levels.py`
is the reproducible query/normalization path, but its current normalized schema
does not retain a separate uncertainty field.

`radiative_benchmarks.json` records independent H I E1 multiplet rates from
Table 6 of Wiese and Fuhr's 2009 NIST critical compilation (DOI
10.1063/1.3077727). It is validation data, not a runtime input. The comparison
tolerance accounts for StarkZee's analytic nonrelativistic gross-structure
energies versus the reduced-mass and relativistic treatment in the compilation.

`multishell_validation_cases.json` records explicitly scoped numerical cutoff
criteria for the experimental multi-shell CI. Five H-beta cases sample 100--
1000 T and fixed electric fields from zero to three Holtsmark normal fields at
three angles. They are internal convergence regressions, not external
benchmarks, infinite-basis proofs, or universal production cutoffs.

Run local checksum validation with:

```text
python scripts/verify_reference_table_sources.py
```

To repeat the upstream binary/raw comparison, clone pystark at the pinned
commit and pass the checkout path:

```text
python scripts/verify_reference_table_sources.py --pystark PATH
```

The canonical Rosato raw-tree hash is computed in sorted POSIX-relative-path
order by hashing, for each file, a four-byte big-endian path length, the path,
an eight-byte big-endian content length, and the content bytes. This prevents
path/content boundary ambiguity and is implemented by the verifier.
