Calculation Workflow and Package Map
====================================

The public entry point is the :class:`~starkzee.line_profile.LineProfile`
class.  It converts the supplied wavelength, frequency, wavenumber, or energy
grid to photon energy, delegates to the explicitly selected solver, stores the
three polarization profiles, constructs the other spectral axes, and combines
the polarizations for the requested observation angle.  It does not normalize
profiles or apply an instrumental slit function.

``compute_profile`` is the historical name of the static path and is an alias
of ``compute_static_profile``.  ``compute_ffm_profile`` is the separate
dynamic-ion path and requires ``Ti_ev``.  The code does not switch between
these models automatically.  Both calls populate the same user-facing result
attributes.

Package responsibilities
------------------------

#. ``line_profile.py`` owns input-grid conversion, result storage, detuning
   axes, and the Stokes observation-angle combination.

#. ``static_profile.py`` performs the quasi-static microfield average and can
   include Doppler broadening when ``Ti_ev`` is supplied.

#. ``radiator.py`` constructs the field-free plus magnetic Hamiltonian
   :math:`H_A` and transition dipole matrices in the uncoupled
   :math:`|n,l,m_l,s,m_s\rangle` basis.

#. ``microfield.py`` supplies the field-magnitude quadrature and Hooper-like,
   Holtsmark, Potekhin, or custom tabulated distributions.

#. ``broadening.py`` evaluates the electron-impact widths, including the
   frequency-dependent GBK/PPPB and ZEST-inspired variants.

#. ``ffm.py`` collects the field-dressed transitions and replaces the frozen
   microfield average with Markov mixing at the ion fluctuation rate.  Its
   Doppler convolution is enabled by default.

#. ``convolutions.py`` provides standalone wavelength-space Doppler and
   instrumental broadening helpers.  Instrumental broadening is always an
   explicit post-processing operation.

#. ``models/`` contains the comparison models ``voigt``, ``lomanowski``,
   ``stehle_param``, ``stehle``, and ``rosato``.

Shared atomic calculation and solver split
------------------------------------------

The following diagram factors out the physical steps shared by the static and
FFM implementations.  The API chooses a solver before execution, but each
solver performs the same microfield-resolved diagonalization and dipole-basis
rotation before assembling the spectrum in a different way.

.. mermaid::
   :name: calculation-flowchart

   flowchart TD
       A["Inputs: transition, species, B,<br/>Ne, Te, Ti, and spectral grid"] --> B["Convert the spectral grid to energy"]
       B --> C["Generate microfield magnitudes F<br/>and probability weights W(F)"]
       B --> D["Generate orientation quadrature<br/>μ = cos θ"]
       B --> E["Build atomic Hamiltonians HA,<br/>Stark templates, and dipole matrices Dq"]

       C --> F["For every active (F, μ) point"]
       D --> F
       E --> F
       F --> G["Resolve Fz = Fμ and<br/>Fx = F sqrt(1-μ²)"]
       G --> H["Construct Hu = HA,u + VE,u<br/>and Hl = HA,l + VE,l"]
       H --> I["Diagonalize Hu and Hl"]
       I --> J["Stark-Zeeman dressed states:<br/>Eu, El, Vu, and Vl"]
       J --> K["Rotate dipoles:<br/>D'q = Vl† Dq Vu"]
       K --> L["Create Stark-Dressed Transitions:<br/>Epk = Eu - El and Sq = |D'q|²"]

       L --> M{"Selected spectrum assembly"}
       M -- Static --> N["Immediately add each SDT with<br/>microfield, electron-impact,<br/>natural, and optional Doppler broadening"]
       N --> O["Direct quasi-static average"]

       M -- FFM --> P["Collect weighted SDTs from<br/>all microfield configurations"]
       P --> Q["Calculate ion fluctuation energy νᵢ"]
       Q --> R["Optional SDT frequency binning"]
       R --> S["Sherman-Morrison or full<br/>Markov-system solve"]
       S --> T["Optional Doppler FFT<br/>enabled by default"]

       O --> U["π, σ+, and σ− profiles"]
       T --> U
       U --> V["Stokes observation-angle combination"]
       V --> W["Store all spectral axes and detunings"]
       W --> X["Optional external<br/>instrumental convolution"]

The dressed states are calculated at the central diagonalization step—not
when ``LineProfile`` is constructed and not after broadening.  For every
active microfield magnitude and orientation, StarkZee diagonalizes the full
upper and lower Hamiltonians separately.  Rotating :math:`D_q` with those
eigenvectors produces the transition energies and strengths that define the
Stark-Dressed Transitions (SDTs).

The static solver broadens and accumulates each SDT immediately.  The FFM
solver instead retains the weighted SDTs from all configurations, optionally
bins nearby frequencies, and mixes them through the ion fluctuation process.
Thus FFM does not avoid or postpone dressed-state construction; ion dynamics
enter only after the instantaneous dressed transitions are known.

``compute_discrete(Fz, Fx)`` is a third, shorter path: it performs the same
upper/lower diagonalization and dipole rotation once at the user-specified
field, then returns the unbroadened transition sticks without a plasma
microfield average.

.. _`sec:formulation`:
