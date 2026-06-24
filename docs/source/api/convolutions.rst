convolutions
============

Post-processing convolutions for Doppler and instrumental broadening.  Thermal
motion produces a Gaussian kernel on a wavelength grid,

.. math::

   G_D(\Delta\lambda) = \frac{1}{\sigma_\lambda\sqrt{2\pi}}
   \exp\left[-\frac{\Delta\lambda^2}{2\sigma_\lambda^2}\right].

The final observed profile is computed by FFT convolution,

.. math::

   I_\mathrm{conv}(\lambda) = \mathcal{F}^{-1}\left\{
   \mathcal{F}[I(\lambda)]\,\mathcal{F}[K(\lambda)]\right\}.

.. automodule:: starkzee.convolutions
   :members:
   :show-inheritance:
