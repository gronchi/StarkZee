References
==========

Audit status and historical snapshot
------------------------------------

``starkzee/data/bibliography.json`` records the canonical metadata and
verification URL for all 22 works represented by the 21 citation keys below.
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

.. [baranger]
   M. Baranger,
   *Simplified Quantum-Mechanical Theory of Pressure Broadening*,
   Phys. Rev. **111**, 481–493 (1958).
   `DOI: 10.1103/PhysRev.111.481 <https://doi.org/10.1103/PhysRev.111.481>`__

   *Problem of Overlapping Lines in the Theory of Pressure Broadening*,
   Phys. Rev. **111**, 494–504 (1958).
   `DOI: 10.1103/PhysRev.111.494 <https://doi.org/10.1103/PhysRev.111.494>`__

.. [bethesalpeter]
   H.A. Bethe, E.E. Salpeter,
   *Quantum Mechanics of One- and Two-Electron Atoms*,
   Springer-Verlag, Berlin (1957).
   `DOI: 10.1007/978-3-662-12869-5 <https://doi.org/10.1007/978-3-662-12869-5>`__

.. [edmonds]
   A.R. Edmonds,
   *Angular Momentum in Quantum Mechanics*,
   Princeton University Press (1957).
   `DOI: 10.1515/9781400884186 <https://doi.org/10.1515/9781400884186>`__

.. [ferri]
   S. Ferri, O. Peyrusse, A. Calisti,
   *Stark-Zeeman line-shape model for multi-electron radiators in hot dense plasmas
   subjected to large magnetic fields*,
   Matter and Radiation at Extremes **7**, 015901 (2022).
   `DOI: 10.1063/5.0058552 <https://doi.org/10.1063/5.0058552>`__
   Strong-collision constants: prose immediately following Eq. (19); Table 1
   instead contains critical magnetic-field values.

.. [iglesias1985]
   C. A. Iglesias, H. E. DeWitt, J. L. Lebowitz, D. MacGowan, W. B. Hubbard,
   *Low-frequency electric microfield distributions in plasmas*,
   Physical Review A **31**, 1698–1702 (1985).
   `DOI: 10.1103/PhysRevA.31.1698 <https://doi.org/10.1103/PhysRevA.31.1698>`__

.. [iglesias2000]
   C. A. Iglesias, F. J. Rogers, R. Shepherd, A. Bar-Shalom, M. S. Murillo, D. P.
   Kilcrease, A. Calisti, R. W. Lee,
   *Fast electric microfield distribution calculations in extreme matter conditions*,
   Journal of Quantitative Spectroscopy and Radiative Transfer **65**, 303–315 (2000).
   `DOI: 10.1016/S0022-4073(99)00076-X <https://doi.org/10.1016/S0022-4073(99)00076-X>`__

.. [Gilleron2018]
   F. Gilleron, J.-C. Pain,
   *ZEST: A fast code for simulating Zeeman–Stark line-shape functions*,
   Atoms **6**, 11 (2018).
   `DOI: 10.3390/atoms6010011 <https://doi.org/10.3390/atoms6010011>`__
   Strong-collision constants: Section 2.2, which attributes the added term to
   Griem, Blaha & Kepple (1979).

.. [gordon]
   W. Gordon,
   *Zur Berechnung der Matrizen beim Wasserstoffatom*,
   Ann. Phys. **394**, 1031–1056 (1929).
   `DOI: 10.1002/andp.19293940807 <https://doi.org/10.1002/andp.19293940807>`__

.. [griem]
   H.R. Griem,
   *Principles of Plasma Spectroscopy*,
   Cambridge University Press (1997).
   `DOI: 10.1017/CBO9780511524578 <https://doi.org/10.1017/CBO9780511524578>`__

.. [griem1959]
   H.R. Griem, A.C. Kolb, K.Y. Shen,
   *Stark Broadening of Hydrogen Lines in a Plasma*,
   Phys. Rev. **116**, 4–16 (1959).
   `DOI: 10.1103/PhysRev.116.4 <https://doi.org/10.1103/PhysRev.116.4>`__

.. [griem1979]
   H. R. Griem, M. Blaha, P. C. Kepple,
   *Stark-profile calculations for Lyman-series lines of one-electron ions in
   dense plasmas*, Physical Review A **19**, 2421–2432 (1979).
   `DOI: 10.1103/PhysRevA.19.2421 <https://doi.org/10.1103/PhysRevA.19.2421>`__
   Strong-collision constants are established in the electron-broadening
   discussion surrounding Table I; the table itself compares scattering cross
   sections.

.. [griembaranger]
   H.R. Griem, M. Baranger, A.C. Kolb, G. Oertel,
   *Stark Broadening of Neutral Helium Lines in a Plasma*,
   Phys. Rev. **125**, 177–195 (1962).
   `DOI: 10.1103/PhysRev.125.177 <https://doi.org/10.1103/PhysRev.125.177>`__

.. [holtsmark]
   J. Holtsmark,
   *Über die Verbreiterung von Spektrallinien*,
   Ann. Phys. **363** (contemporary series volume **58**), 577–630 (1919).
   `DOI: 10.1002/andp.19193630702 <https://doi.org/10.1002/andp.19193630702>`__

.. [hooper]
   C.F. Hooper, Jr.,
   *Low-Frequency Component Electric Microfield Distributions in Plasmas*,
   Phys. Rev. **165**, 215–222 (1968).
   `DOI: 10.1103/PhysRev.165.215 <https://doi.org/10.1103/PhysRev.165.215>`__

.. [lomanowski]
   B.A. Lomanowski, A.G. Meigs, R.M. Sharples, M. Stamp, C. Guillemaut, and JET Contributors,
   *Inferring divertor plasma properties from hydrogen Balmer and Paschen series
   spectroscopy in JET-ILW*,
   Nucl. Fusion **55**, 123028 (2015).
   `DOI: 10.1088/0029-5515/55/12/123028 <https://doi.org/10.1088/0029-5515/55/12/123028>`__
   The parameterized MMM width and modified-Lorentzian profile used here are
   Eq. (1), Eq. (2), and Table 1.

.. [potekhin]
   A.Y. Potekhin, G. Chabrier, D. Gilles,
   *Electric microfield distributions in electron-ion plasmas*,
   Phys. Rev. E **65**, 036412 (2002).
   `DOI: 10.1103/PhysRevE.65.036412 <https://doi.org/10.1103/PhysRevE.65.036412>`__

.. [rosato]
   J. Rosato, Y. Marandet, R. Stamm,
   *A new table of Balmer line shapes for the diagnostic of magnetic fusion plasmas*,
   J. Quant. Spectrosc. Radiat. Transfer **187**, 333–337 (2017).
   `DOI: 10.1016/j.jqsrt.2016.10.005 <https://doi.org/10.1016/j.jqsrt.2016.10.005>`__

.. [stehle]
   C. Stehlé, R. Hutcheon,
   *Extensive tabulations of Stark broadened hydrogen line profiles*,
   Astron. Astrophys. Suppl. Ser. **140**, 93–97 (1999).
   `DOI: 10.1051/aas:1999118 <https://doi.org/10.1051/aas:1999118>`__
   Electronic tables: `CDS/VizieR VI/98A <https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=VI/98A>`__.

.. [nist]
   A. Kramida, Yu. Ralchenko, J. Reader, NIST ASD Team,
   *NIST Atomic Spectra Database (ver. 5.12)*,
   National Institute of Standards and Technology, Gaithersburg, MD (2024).
   `DOI: 10.18434/T4W30F <https://doi.org/10.18434/T4W30F>`__

.. [wiese2009]
   W. L. Wiese, J. R. Fuhr,
   *Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium*,
   J. Phys. Chem. Ref. Data **38**, 565--720 (2009).
   `DOI: 10.1063/1.3077727 <https://doi.org/10.1063/1.3077727>`__

.. [talin]
   B. Talin, A. Calisti, L. Godbert, R. Stamm, R.W. Lee, L. Klein,
   *Frequency-fluctuation model for line-shape calculations in plasma spectroscopy*,
   Phys. Rev. A **51**, 1918–1928 (1995).
   `DOI: 10.1103/PhysRevA.51.1918 <https://doi.org/10.1103/PhysRevA.51.1918>`__
