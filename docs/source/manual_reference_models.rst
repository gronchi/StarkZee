Built-in Reference Models
=================================

The ``starkzee.models`` package provides five comparison lineshape models that share a common call signature. They are grouped into tabulated models (reading precomputed databases) and analytical models. Their curves are diagnostics, not five independent physical benchmarks. ``starkzee/models/reference_provenance.json`` records which parts are direct source transcriptions and which are StarkZee-specific wrapper operations or approximations.

Tabulated Models
------------------------

Stehlé (MMM) — ``stehle``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Stehlé model reads precomputed Stark lineshapes from the Model Microfield Method (MMM) database [stehle]_ for hydrogen Lyman, Balmer, and Paschen transitions over a grid of :math:`N_e` and :math:`T_e` values. The tables contain the *unmagnetized* (:math:`B = 0`) Stark profile **without fine structure** [stehle]_, as a function of a reduced detuning :math:`\Delta\omega/F_0` (Holtsmark normal field units). The pipeline is:

#. Linear interpolation over the required :math:`(N_e, T_e)` table corners. Exact coordinate nodes use only that source node. Positive detunings use log--log interpolation by default; ``interpolation='fintrp'`` selects the distributed Fortran-style three-point rule.

#. Thermal Doppler broadening via FFT convolution.

#. Normal Zeeman splitting applied *after* the table: the whole Stark+Doppler profile is rigidly copied to :math:`\omega \pm \mu_B B/\hbar` (the normal-Zeeman component shift -- half the electron Larmor/cyclotron frequency :math:`\omega_L = eB/m_e` used elsewhere for the plasma dynamical cutoffs, not :math:`\omega_L` itself) with angle-dependent :math:`\pi/\sigma` weights:

   .. math::

              I(\theta) = \sin^2\!\theta\,I_\pi + \tfrac{1+\cos^2\!\theta}{2}\,(I_{\sigma+} + I_{\sigma-})


The Stark and Zeeman effects are therefore treated as **separable**. This is valid only when the Zeeman splitting :math:`\mu_B B` is much smaller than the Stark width. The bundled file contains exactly 84 transitions: :math:`n_l=1,2,3`, with :math:`n_u=n_l+1,\ldots,30`. Requests outside that set are rejected. Its ten temperature nodes span 2500--1,259,600 K; density endpoints and cell availability depend on the transition and temperature. Exact endpoints are accepted when their source cell is available, while extrapolation beyond the stored coordinates is rejected. H, D, and T select the wrapper's line center and Doppler mass but share these hydrogen Stark tables; this is not an isotope-specific MMM calculation. The normal-Zeeman post-processing accepts nonnegative :math:`B`, but it does not make the field-free database a coupled Stark--Zeeman model.

The bundled NetCDF file has no embedded global or variable provenance/unit
attributes. It is nevertheless byte-identical to the NetCDF at pinned pystark
commit ``9a8782a``, whose tree contains the CDS/VizieR VI/98A raw distribution
and conversion script. Checksums, units, paths and unresolved license and
uncertainty fields are recorded in ``starkzee/data/reference_tables.json``.
This traceability and the implementation-level inventory checks do not replace
comparison with held-out physical data.

Rosato — ``rosato``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Rosato database [rosato]_ solves the Stark-Zeeman problem *jointly* and tabulates the result with :math:`B` as an explicit table axis (:math:`B \in \{0, 1, 2, 2.5, 3, 5\}` T). The pipeline is:

#. Interpolation over the :math:`(N_e, T_e, B)` grid. The :math:`B`-interpolation is *scaled*: before blending two bracketing profiles at :math:`B_0` and :math:`B_1`, each profile’s detuning axis is stretched by :math:`B_\mathrm{node}/B`. This exploits the fact that Zeeman splitting scales linearly with :math:`B`, aligning the :math:`\sigma` components before interpolating.

#. Angle dependence from *two real tables* (parallel and perpendicular to :math:`\vec{B}`), blended as :math:`I = I_\parallel\cos^2\theta + I_\perp\sin^2\theta`. Both tables carry the correct :math:`\pi/\sigma` lineshape and width — the angle enters as a physical interpolation, not just an amplitude weight.

#. Thermal Doppler broadening via FFT convolution.

Rosato supports D Balmer lines with upper n=3..7, and an approximate H mapping. Density covers 1e13..1e16 cm^-3, effective table temperature 0.316..31.6 eV and B=0..5 T, including endpoints. The effective temperature is Ti for D and 2*Ti for H; Te is not a table coordinate. The corrected code uses four-corner logarithmic density/temperature interpolation, retains Doppler tails and normalizes output per wavelength metre. No interpolation accuracy outside the tables is claimed.

Rosato's bundled data contain negative intensities at some nodes. By default,
``negative_table_policy='warn'`` warns and preserves signed values through
interpolation, Doppler smoothing and normalization. Explicit ``'raise'`` rejects
affected cells; ``'clip'`` projects values to zero and warns. Clipping can bias
noisy data and is not validation or replacement of the underlying data.
Reference wrappers return density per wavelength metre on air-centered axes;
main-solver stored profiles remain per eV. Apply coordinate/Jacobian conversions
and matched finite-window normalization before comparing amplitudes.

The Rosato NetCDF has no embedded attributes, but an exhaustive provenance
check establishes that its 3,000,000 detunings and 3,000,000 intensities equal
the 3,000 raw profiles at pinned pystark commit ``9a8782a`` after numeric
parsing. The archived reader declares density in cm\ :sup:`-3`, temperature and
detuning in eV, and magnetic field in tesla. Intensity is interpreted as
eV\ :sup:`-1` from the area-normalized line-shape convention, rather than from
an explicit source declaration. The original database-transfer URL/date,
license, numerical uncertainty, and intended treatment of signed fluctuations
remain unknown. The full machine-readable record and an optional upstream
verifier are in ``starkzee/data/reference_tables.json`` and
``scripts/verify_reference_table_sources.py``.

Analytical Models
-------------------------

Lomanowski-width sensitivity model — ``lomanowski``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This StarkZee-specific sensitivity model uses the power-law Stark FWHM from
Lomanowski *et al.* Eq. (1) and Table 1 [lomanowski]_:

.. math::

   \Delta\lambda_S = c_{n_u n_l}\,N_e^{a_{n_u n_l}}\,T_e^{-b_{n_u n_l}} \quad [\mathrm{nm}]


Coefficients :math:`(a, b, c)` are tabulated for H/D Balmer and Paschen lines up to :math:`n_u = 9`. The subsequent common-width modified-Lorentzian/Gaussian pseudo-mixture is local to StarkZee: it is not the profile prescribed by the cited paper and is not a ``pystark`` routine. Its mixing-polynomial provenance has not been recovered. The wrapper adds separable normal-Zeeman splitting, so the function does accept :math:`B`; it remains a sensitivity approximation rather than an independent benchmark.

Parameterized Stehlé — ``stehle_param``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The direct Lomanowski parameterization of the Stehlé/MMM tables [lomanowski]_: Eq. (1) and Table 1 give the Stark FWHM and coefficients, while Eq. (2) gives the modified-Lorentzian profile. StarkZee then performs frequency-domain Doppler convolution and applies a separable normal-Zeeman triplet. The paper reports its fit domain as :math:`10^{19}\leq N_e\leq10^{21}` m\ :sup:`-3` and :math:`1\leq T_e\leq10` eV, with under-10% FWHM error against the original tabulations within that domain; this is not a guarantee for the complete profile or for extrapolation.

Voigt — ``voigt``
^^^^^^^^^^^^^^^^^^^^^^^^^

A Faddeeva-evaluated Voigt profile using the Griem :math:`\alpha_{12}` Stark half-width [griem]_ combined with thermal Doppler broadening. Serves as a quick estimate when only order-of-magnitude Stark width accuracy is required.

Comparison with StarkZee
--------------------------------

.. list-table:: Comparison with StarkZee
   :header-rows: 1
   :widths: 22 30 25 23

   * - Feature
     - StarkZee
     - Stehlé
     - Rosato
   * - Magnetic field
     - Full simultaneous diagonalization of :math:`H = H_A + V_E`
     - :math:`B = 0` in tables; added as rigid triplet after
     - :math:`B` is a table axis
   * - Fine structure
     - Yes — spin-orbit, MV, Darwin
     - No (degenerate hydrogenic)
     - In tables
   * - Electron broadening
     - GBK semi-classical, frequency-dependent
     - Unified theory (more rigorous far wings)
     - In tables
   * - Ion dynamics
     - Quasi-static + optional FFM
     - Model Microfield Method (MMM) -- a distinct ion-dynamics treatment from StarkZee's FFM, not "static plus some dynamics"
     - In tables
   * - Microfield
     - Potekhin (default)
     - Holtsmark/Hooper variant
     - Holtsmark variant
   * - :math:`B`-treatment
     - Intrinsic to Hamiltonian
     - External convolution
     - :math:`B`-scaled interpolation

The most important physical distinction at :math:`B \ne 0`: StarkZee diagonalizes :math:`H = H_A + V_E` *simultaneously* for all field strengths. When :math:`\mu_B B \sim 3n\,e\,a_0\,F` (Zeeman and Stark splitting comparable) the eigenstates are genuine Stark-Zeeman hybrids — neither pure Zeeman nor pure Stark states. No post-processing convolution can reproduce this mixing. Separable Zeeman post-processing cannot reproduce this mixing. Rosato tables already contain it; interpolation accuracy must be checked separately.

.. figure:: ../figures/model_comparison.png
   :width: 100%
   :alt: StarkZee (static and FFM) compared against four built-in reference models for D-alpha

   Current D-alpha comparison generated from this working tree. The analytical
   and Stehle wrappers add separable Zeeman triplets; Rosato tables contain
   joint Stark-Zeeman physics. Rosato values below :math:`10^{-6}` of its peak
   are omitted from this visualization only. This is not an independent
   accuracy benchmark.

.. figure:: ../figures/model_comparison_dgamma.png
   :width: 100%
   :alt: StarkZee (static and FFM) compared against four built-in reference models for D-gamma

   The corresponding D-gamma (:math:`n=5\to2`) comparison at the same plasma
   conditions and with the same display-only Rosato cutoff.



