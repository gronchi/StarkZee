Balmer series — LineProfile class
====================================

**Script:** ``examples/example_balmer_lineprofile.py``

Computes Hα through Hε at DIII-D-like edge conditions (B = 12 T,
:math:`N_e = 10^{21}` m\ :sup:`-3`, :math:`T_e = 1` eV, :math:`T_i = 10` eV) using
the :class:`~starkzee.line_profile.LineProfile` class.  Each panel shows:

- Transverse (90°) and parallel (0°) broadened profiles
- Discrete stick spectrum at zero microfield (:math:`F = 0`)

``Ti_ev`` is supplied so the static solver folds in thermal Doppler
broadening — without it the Stark-Zeeman components at these conditions are
narrower than the grid spacing, and the profile comes out as an
undersampled, spiky Lorentzian sum rather than a smooth curve.

Each line's wavelength window is set individually rather than sharing one
fixed half-width: Stark broadening grows rapidly with :math:`n`, so a window
sized for Hα would truncate the wings of Hδ or Hε — at a fixed ±1 nm window
the profile is still at 6 % of its peak at the edge for Hε versus 0.4 % for
Hα — showing up as an apparently "clipped" profile that never reaches zero.
Each half-window below is instead sized so its own line has decayed to
<0.2 % of peak by the panel edge.

Code
----

.. literalinclude:: ../../../examples/example_balmer_lineprofile.py
   :language: python
   :linenos:

Result
------

.. figure:: ../../figures/examples/example_balmer_lineprofile.png
   :width: 100%
   :alt: Balmer series stick spectra and broadened profiles from the LineProfile class

   Transverse and parallel broadened profiles with the zero-field stick
   spectrum overlaid, for Hα through Hε.
