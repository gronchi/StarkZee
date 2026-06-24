broadening
==========

Electron-impact broadening models.  The default GBK/Ferri model evaluates a
component half-width using a density-temperature prefactor, shell radius, and a
dynamical cutoff factor,

.. math::

   \gamma_e(\Delta\omega) = W_\mathrm{pref}\,\langle r^2\rangle_n
   \left[C_n + G(\Delta\omega)\right],

with

.. math::

   G(\Delta\omega) = \frac{1}{2}E_1(y),
   \qquad
   y = \left(\frac{n^2}{2Z}\right)^2
   \frac{\Delta\omega^2 + \omega_c^2}{\mathrm{Ry}\,T_e}.

The cutoff is chosen as

.. math::

   \omega_c = \max\left(\omega_p,\omega_e,\omega_L,\omega_{\alpha\alpha'}\right).

.. automodule:: starkzee.broadening
   :members:
   :show-inheritance:
