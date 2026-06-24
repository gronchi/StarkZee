API reference
=============

.. toctree::
   :maxdepth: 1

   line_profile
   static_profile
   ffm
   radiator
   microfield
   broadening
   convolutions
   utils
   atomic_data

Module overview
---------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Module
     - Contents
   * - :mod:`starkzee.line_profile`
     - High-level :class:`LineProfile` class and :class:`DiscreteTransitions`
   * - :mod:`starkzee.static_profile`
     - Static Stark-Zeeman profile solver and discrete-transition enumerator
   * - :mod:`starkzee.ffm`
     - Frequency Fluctuation Model (dynamic ion broadening)
   * - :mod:`starkzee.radiator`
     - Basis states, Hamiltonian construction, dipole matrix elements,
       oscillator strengths, Einstein A coefficients
   * - :mod:`starkzee.microfield`
     - Holtsmark and Hooper microfield distributions, quadrature grid builder
   * - :mod:`starkzee.broadening`
     - GBK electron impact broadening with magnetic-field cutoff
   * - :mod:`starkzee.convolutions`
     - FFT-based Doppler and instrumental broadening (post-processing)
   * - :mod:`starkzee.utils`
     - Physical constants (via scipy.constants) and unit conversions
   * - :mod:`starkzee.atomic_data`
     - NIST atomic energy-level database loader; :class:`AtomicState` dataclass
