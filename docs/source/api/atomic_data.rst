atomic\_data
============

NIST atomic energy-level database and loader.  Field-free level energies
for each element are stored in ``starkzee/data/atomic_levels.json`` as
cm\ :sup:`-1` values above the ground state (NIST vacuum convention).
The loader is called by :func:`starkzee.radiator.build_hamiltonian`
when ``use_empirical_data=True``.
Use ``python scripts/update_nist_levels.py H --spectrum "H I"`` to refresh the bundled JSON from the NIST ASD levels query.  The downloader is intentionally a development tool; runtime profile calculations only read the committed JSON file.

.. automodule:: starkzee.atomic_data
   :members:
   :show-inheritance:



