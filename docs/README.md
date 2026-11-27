# StarkZee theory manual

[Read the PDF](manual.pdf).

See [review.md](review.md) for the consistency review, corrections and remaining
scientific validation limits.

This standalone edition reuses the scientific content of `../docs/manual.tex`
with the typography and layout of
`ColRadPy/doc/ColRadPy_theory_manual.tex`: Latin Modern text and math, A4 pages
with 2.5 cm margins, a separate title page, compact contents, running headers,
section-numbered equations, and blue implementation notes and orange caveats.
Implementation notes and caveats appear alongside the relevant theory,
including atomic units, empirical energies, microfields, damping, FFM and Doppler
broadening. A calculation-flow diagram precedes the derivation. Existing equations,
qualifications, comparison figures and bibliography are retained.

From the repository root, run:

```console
python docs/build_manual.py
```

Requires Python, pdfLaTeX (including the packages used in `manual.tex`) and Biber.
The build uses a temporary directory for auxiliary files and replaces
`manual.pdf` only after successful compilation and reference checks.

Edit `manual.tex` to develop this edition independently. The content began as
a copy of the working tree on 19 October 2026; subsequent consistency
corrections are recorded in `review.md`. It does not represent a committed
release. The two original comparison figures are copied from
`docs/figures/`; their captions retain the comparison limits.

The atomic-data and Stark-map chapter is shared with the original manual
through `../docs/atomic_api.tex`. Its code excerpts come directly from
`../examples/atomic_api_and_stark_maps.py`, and its three figures are shared
with the HTML documentation in `../docs/figures/examples/`. To regenerate
these figures before rebuilding the PDF, run:

```console
python examples/atomic_api_and_stark_maps.py
python docs/build_manual.py
```

The plotting script requires NumPy, SciPy, and Matplotlib and runs without
ARC or downloaded data. Keep the repository directory structure when
building either manual so the shared chapter, examples, and figures resolve.

`references.bib` is a copy of the generated `docs/references.bib`. To refresh
it after changing the canonical `starkzee/data/bibliography.json` registry,
run `python scripts/generate_bibtex.py`, then copy `docs/references.bib` here.
Do not edit bibliography metadata in this copy by hand.
