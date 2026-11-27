C IV n = 5 → 4
==============

Li-like C IV n = 5 → 4 (strongest component 5g - 4f, 253.0 nm in air) at
B = 0, 100, 300 and 500 T, for :math:`n_e = 2\times10^{24}` m\ :sup:`-3` and
:math:`T_e = T_i = 10` eV. These are the gas-liner pinch conditions of Glenzer
*et al.*, *J. Phys. B* **27**, 5507 (1994), also used by Ferri *et al.* (2022).

The radiator (:func:`~starkzee.single_valence.c_iv_5_4`) Stark-mixes the n = 5
(5s - 5g) and n = 4 (4s - 4f) terms from NIST levels. Its reduced elements come
from the Coulomb approximation and are within 1.3-2.6 % of the NIST grade-A
line strengths. The diamagnetic term is included.

The script uses the per-state electron-impact operator
(``electron_operator=True``). The width depends strongly on that choice: at
B = 0 the FWHM is 19.5 Å with scalar widths, 10.8 Å with the per-state operator,
and 5.4 Å with the full interference operator. Add ``--plot`` to show the
profiles. See :doc:`../manual_multielectron`.

.. literalinclude:: ../../../examples/c_iv_n5_n4_stark_zeeman.py
   :language: python
   :linenos:
