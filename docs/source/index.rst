starkzee
============

A Python library implementing a coupled **Stark-Zeeman plasma line-shape model**
for hydrogen-like radiators, based on:

    S. Ferri, O. Peyrusse, A. Calisti, *et al.*,
    "Stark-Zeeman line-shape model for multi-electron ions in hot dense plasmas",
    *Matter and Radiation at Extremes* **7**, 015901 (2022).

StarkZee implements the **Standard Lineshape Theory** for emission lines of
hydrogen-like ions in a magnetized plasma.  Ions are treated in the
**quasi-static approximation**: the ion microfield at the radiator site is
assumed stationary on the timescale of the emitted photon, and the spectral
profile is obtained by averaging Stark-Zeeman Hamiltonians over the ion
microfield distribution.  Electron broadening is represented by weak binary
collisions within the **Griem–Baranger–Kolb (GBK) binary-collision relaxation
model**, which accounts for the suppression of broadening at large frequency
detunings through a semi-classical exponential-integral factor and a
magnetic-field-dependent lower cutoff.  The static magnetic field enters the
radiator Hamiltonian directly — in the electric-dipole approximation —
producing coupled Stark-Zeeman energy levels and polarized π and σ± emission
components.  **Ion dynamics** are optionally included via the **Frequency
Fluctuation Model (FFM)**, which treats the microfield as a Markovian jump
process between quasi-static configurations.  The static ion microfield
distribution is evaluated using the analytical **Hooper screened distribution**,
parametrized by the electron–ion screening factor *a* = *r*\ :sub:`e` / λ\ :sub:`D`
(ratio of the mean inter-particle distance to the electron Debye length).

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
   :caption: Manual

   manual_introduction
   manual_workflow
   manual_formulation
   manual_approximations
   manual_reference_models
   manual_validity
   manual_operational_limits
   manual_references

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api/index



