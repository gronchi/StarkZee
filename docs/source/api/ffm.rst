ffm
===

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

.. automodule:: starkzee.ffm
   :members:
   :show-inheritance:
