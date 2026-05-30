starkzee
============

A Python library implementing a coupled **Stark-Zeeman plasma line-shape model**
for hydrogen-like radiators, based on:

    S. Ferri, O. Peyrusse, A. Calisti, *et al.*,
    "Stark-Zeeman line-shape model for multi-electron ions in hot dense plasmas",
    *Matter and Radiation at Extremes* **7**, 015901 (2022).

The code computes emission line profiles of hydrogen-like ions in a magnetic
field B, accounting for:

- Quasi-static ion microfield broadening (Holtsmark / Hooper distributions)
- Electron impact broadening (GBK model with Larmor-frequency cutoff)
- Ion dynamics via the Frequency Fluctuation Model (FFM)
- Spin-orbit coupling and the quadratic (diamagnetic) Zeeman term
- Doppler and instrumental broadening as post-processing convolutions

.. code-block:: python

    import numpy as np
    from starkzee.line_profile import LineProfile

    lp = LineProfile(n_u=3, n_l=2, Z=1, B=5.0, Ne_m3=1e20, Te_ev=5.0)
    energies = lp.E0 + np.linspace(-0.005, 0.005, 1000)
    lp.compute_profile(energies, num_f=30, num_mu=8)

    import matplotlib.pyplot as plt
    plt.plot(lp.wavelengths_nm - lp.E0_wavelength_nm, lp.profile_transverse)
    plt.xlabel("Δλ (nm)")

----

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: Background

   theory
   examples

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api/index
