lorentzian_fft
==============

FFT sums of many Lorentzian or Voigt components on a uniform grid: the opt-in
approximate ``method='fft'`` of
:func:`~starkzee.static_profile.calculate_static_profile`. The error is held below
``tol`` relative to the largest component peak; the default ``method='exact'``
does not use this module.

.. automodule:: starkzee.lorentzian_fft
   :members: lorentzian_sum, frequency_dependent_sum
