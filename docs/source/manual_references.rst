References
==========

Audit status and historical snapshot
------------------------------------

``starkzee/data/bibliography.json`` records the canonical metadata and
verification URL for all 22 works below, one citation key per work.
``docs/references.bib``, which both this site and the LaTeX manual read, is
generated from it by ``python scripts/generate_bibtex.py``.
The audit checks titles, author lists, publication containers, years,
volumes/issues, page ranges or article numbers, and DOI identifiers. The NIST
entry is release 5.12 (2024), which was current when the archived D I and T I
queries were downloaded on 13 July 2026. The H I table has no archived query
URL or acquisition timestamp, so its exact NIST release remains unknown.

The standalone manual audited on 7 September 2026 is now tied to immutable
StarkZee commit ``24118d1fdee16d3c5aef98992809c5a0db2faea0``. Its historical
``docs/manual.tex`` has SHA-256
``dbfcb4766218bbb1d458982865bc22d6b5158b99e1359135908d4bf8f0ef89b6``,
exactly the digest recorded in the original audit. That commit also contains
the compiled manual, referenced comparison figure, figure generator, model
implementation, and the three runtime datasets. Exact blob IDs, sizes, hashes,
and immutable URLs are in ``starkzee/data/historical_provenance.json`` and can
be checked from a full Git clone with
``python scripts/verify_historical_provenance.py``. This establishes which
repository bytes the manual accompanied; it does not validate their physics or
resolve the external license/acquisition unknowns retained in
``starkzee/data/reference_tables.json``.

Cited works
-----------

.. bibliography::
   :all:
