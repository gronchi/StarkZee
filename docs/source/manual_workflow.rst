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
       A(["Inputs: transition, species,<br/>B, Ne, Te, Ti, spectral grid"]) --> B["Convert spectral grid to energy"]

       subgraph setup["Setup — computed once"]
           direction LR
           C["Microfield magnitudes F<br/>and weights W(F)"]
           D["Orientation quadrature<br/>μ = cos θ"]
           E["Atomic Hamiltonians HA,<br/>Stark templates, dipoles Dq"]
       end
       B --> C
       B --> D
       B --> E

       subgraph loop["For every active (F, μ) point — repeated"]
           direction TB
           G["Fz = Fμ,  Fx = F√(1-μ²)"]
           G --> H["Hu = HA,u + VE,u<br/>Hl = HA,l + VE,l"]
           H --> I["Diagonalize Hu, Hl"]
           I --> J["Dressed states:<br/>Eu, El, Vu, Vl"]
           J --> K["Rotate dipoles:<br/>D'q = Vl† Dq Vu"]
           K --> L["Stark-Dressed Transitions:<br/>Epk = Eu - El,  Sq = |D'q|²"]
       end
       C --> G
       D --> G
       E --> G

       L --> M{"Static or FFM?"}

       subgraph static_path["Static solver"]
           direction TB
           N["Broaden + accumulate each SDT<br/>(microfield, electron, natural,<br/>optional Doppler)"] --> O["Quasi-static average"]
       end
       M -- Static --> N

       subgraph ffm_path["FFM solver"]
           direction TB
           P["Collect weighted SDTs<br/>from all (F, μ)"] --> Q["Ion fluctuation energy νᵢ"]
           Q --> R["Optional SDT<br/>frequency binning"]
           R --> S["Sherman-Morrison or<br/>full Markov solve"]
           S --> T["Optional Doppler FFT<br/>(on by default)"]
       end
       M -- FFM --> P

       O --> U["π, σ+, σ− profiles"]
       T --> U
       U --> V["Stokes observation-angle<br/>combination"]
       V --> W["Store spectral axes<br/>and detunings"]
       W --> X(["Optional instrumental<br/>convolution"])

       classDef setupNode fill:#f1f5f9,stroke:#64748b,color:#1e293b
       classDef decision fill:#fff3cd,stroke:#c9971d,color:#4a3800
       classDef terminal fill:#dcfce7,stroke:#16a34a,color:#14532d
       classDef outNode fill:#f1f5f9,stroke:#64748b,color:#1e293b

       class C,D,E setupNode
       class M decision
       class A terminal
       class U,V,W,X outNode

       style loop fill:#fefce8,stroke:#ca8a04
       style static_path fill:#eff6ff,stroke:#2563eb
       style ffm_path fill:#f5f3ff,stroke:#7c3aed
       style setup fill:#f8fafc,stroke:#94a3b8

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
