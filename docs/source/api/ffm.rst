ffm
===

``calculate_ffm_profile`` defaults to ``use_empirical_data=True`` with
``atom='H'``. Supply the matching isotope for D or T. Set this flag to
``False`` for analytical energies, ``Z > 1``, or shells outside the tables.

Frequency Fluctuation Model utilities for adding ion dynamics to a static
Stark-Zeeman transition set.  In the Sherman-Morrison form used by the fast
solver, the profile is

.. math::

   I(\omega) = \frac{r^2}{\pi}\,\operatorname{Re}
   \left[\frac{S(\omega)}{1-\nu_i S(\omega)}\right],

where

.. math::

   S(\omega) = \sum_k
   \frac{p_k}{\nu_i + \gamma_k + i(\omega - \omega_k)}.

Here :math:`p_k` are normalized Stark-dressed transition weights,
:math:`\gamma_k` are electron-impact widths, and :math:`\nu_i` is the ion
microfield fluctuation rate.

``A_ion`` denotes the emitter mass in the profile API. Optional
``A_perturber`` defaults to it, retaining historical same-species behavior.
The default ``fluctuation_rate_model='zest'`` uses
:math:`v=\sqrt{2kT/m_{\rm pert}}`; ``'ppp'`` uses
:math:`v=\sqrt{kT/\mu}` with emitter--perturber reduced mass. These expressions
are identical for equal masses and differ for a heavy radiator in a light-ion
background.

By default every SDT uses the shared resonance width :math:`\gamma(0)`, matching
the historical and ZEST fast-FFM approximation. Setting
``sdt_frequency_dependent_width=True`` instead evaluates the upper- plus
lower-shell width at each SDT's own detuning from the physical line reference.
This opt-in mode remains :math:`O(N)` in the Sherman--Morrison solver and is
distinct from the static solver's pointwise observation-frequency width.

Natural damping is independently state resolved by default: every SDT carries
:math:`\hbar(\Gamma_{u,i}+\Gamma_{l,j})/2`, with the rates rotated into the
local Stark--Zeeman eigenbasis. ``natural_width_mode='shell_average'`` restores
the historical common natural width. If SDT binning is requested, the merged
natural width is the intensity-weighted mean of its members.

With ``electron_interference=True``, the FFM numerator retains the PPP complex
weights :math:`a_k+i c_k`, while StarkZee's default experimental stationary
probabilities are :math:`p_k=|a_k|/\sum_j|a_j|`. This modulus rule is not
specified by the PPP manual; it preserves signed interference in the radiative
numerator and changes only the finite-ion-dynamics closure. This impact-limit mode deliberately rejects SDT
binning, per-SDT frequency-dependent widths, and the legacy numerical-inversion
switch until equivalent complex-weight formulations are independently tested.

``interference_group_tolerance_ev`` enables a separate experimental grouping
closure. Nearby modes are grouped in frequency and width, their complex
residues are summed, and the effective pole preserves the first complex pole
moment. StarkZee accepts the result only if all grouped real residues are
nonnegative, all widths are positive, and the grouped zero-fluctuation profile
matches the ungrouped profile within ``interference_group_profile_rtol``.
Otherwise it raises an explicit validity error; the ungrouped static profile
remains available.

.. automodule:: starkzee.ffm
   :members:
   :show-inheritance:
