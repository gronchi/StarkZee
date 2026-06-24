Calculation Workflow and Package Map
============================================

The public entry point is the ``LineProfile`` class. A typical calculation builds a wavelength, frequency, or energy grid, calls ``LineProfile.compute_profile(grid, grid_type)``, and then reads the total and polarization-resolved profiles from the resulting object. Internally, that high-level call delegates the work through a small number of focused modules:

#. **``line_profile.py``** manages spectral grids, unit conversions, profile normalization, optional Doppler/slit convolutions, and user-facing result attributes.

#. **``static_profile.py``** performs the quasi-static microfield average. It creates the field-strength grid, the Gauss-Legendre orientation grid, and accumulates :math:`P_\pi`, :math:`P_{\sigma^+}`, and :math:`P_{\sigma^-}`.

#. **``radiator.py``** constructs the field-free plus magnetic radiator Hamiltonian :math:`H_A` and the transition dipole matrices in the uncoupled :math:`|n,l,m_l,s,m_s\rangle` basis.

#. **``solve_starkzee()``** forms the complete Hamiltonian :math:`H = H_A + V_E` at each microfield point and diagonalizes it as one matrix. The Stark term is not added perturbatively.

#. **``broadening.py``** evaluates electron-impact widths, including the frequency-dependent Griem-Baranger-Kolb (GBK) model and the ZEST-inspired variant.

#. **``ffm.py``** optionally replaces the static profile by a dynamic-ion profile using the Frequency Fluctuation Model.

#. **``microfield.py``** supplies Hooper, Holtsmark, Potekhin, and custom tabulated microfield distributions.

#. **``models/``** contains comparison models: ``voigt``, ``lomanowski``, ``stehle_param``, ``stehle``, and ``rosato``.

The numerical calculation has one central loop: for each sampled microfield magnitude :math:`F` and orientation :math:`\mu=\cos\theta`, StarkZee builds :math:`V_E(F,\mu)`, diagonalizes upper and lower manifolds, rotates the transition dipoles into the dressed-state bases, evaluates electron-broadened component profiles, and adds the weighted contribution to the three polarization channels. This is the organizing idea behind the equations that follow.

.. _`sec:formulation`:

