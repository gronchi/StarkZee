line\_profile
=============

High-level user interface for computing and storing Stark-Zeeman line profiles.
The observed profile combines the polarization-resolved components through the
standard viewing-angle relation

.. math::

   I(\omega,\alpha) = I_\pi(\omega)\sin^2\alpha
   + \frac{1}{2}\left[I_{\sigma^+}(\omega)+I_{\sigma^-}(\omega)\right]
     \left(1+\cos^2\alpha\right).

Optional post-processing convolutions are applied on a uniform wavelength grid,

.. math::

   I_\mathrm{obs}(\lambda) =
   \left[I_\mathrm{SZ}(\lambda) * G_D(\lambda)\right] * G_\mathrm{inst}(\lambda),

where :math:`G_D` is the thermal Doppler kernel and :math:`G_\mathrm{inst}` is
the instrumental response.

.. automodule:: starkzee.line_profile
   :members:
   :show-inheritance:
