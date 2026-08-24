Reproducing Figure 1 from Ferri et al. (2022)
================================================

Three related scripts computing Balmer-series Stark-Zeeman profiles for H,
in the style of Figure 1 of Ferri, Peyrusse & Calisti (2022):

- ``reproduce_fig1_Ha.py`` — Hα only, three B values, with/without quadratic
  Zeeman.
- ``reproduce_fig1_Hb.py`` — Hβ only, same layout.
- ``reproduce_fig1_Balmer.py`` — the full Balmer series (Hα–Hε) on individual
  local wavelength grids, comparing LTE-Boltzmann and Case B recombination
  relative-intensity weightings.

Hα (``reproduce_fig1_Ha.py``)
-------------------------------

.. literalinclude:: ../../../examples/reproduce_fig1_Ha.py
   :language: python
   :linenos:

.. figure:: ../../figures/examples/reproduce_fig1_Ha.png
   :width: 100%
   :alt: Hα Stark-Zeeman profiles at three B values, with and without quadratic Zeeman

   Hα at B = 1, 10, 50 T, solid = with quadratic Zeeman, dashed = without.

Hβ (``reproduce_fig1_Hb.py``)
-------------------------------

.. literalinclude:: ../../../examples/reproduce_fig1_Hb.py
   :language: python
   :linenos:

.. figure:: ../../figures/examples/reproduce_fig1_Hb.png
   :width: 100%
   :alt: Hβ Stark-Zeeman profiles at three B values, with and without quadratic Zeeman

   Hβ at B = 1, 10, 50 T, solid = with quadratic Zeeman, dashed = without.

Full Balmer series (``reproduce_fig1_Balmer.py``)
----------------------------------------------------

.. literalinclude:: ../../../examples/reproduce_fig1_Balmer.py
   :language: python
   :linenos:

.. figure:: ../../figures/examples/reproduce_fig1_Balmer.png
   :width: 100%
   :alt: Full Balmer series LTE-weighted Stark-Zeeman profiles, linear and log scale

   Hα–Hε weighted by LTE Boltzmann populations, linear (top) and log (bottom)
   scale.
