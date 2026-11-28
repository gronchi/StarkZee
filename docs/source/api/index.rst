API reference
=============

.. toctree::
   :maxdepth: 1

   line_profile
   atomic
   stark_map
   static_profile
   ffm
   radiator
   radiators
   ls_radiator
   jlevel_radiator
   level_data
   coulomb_approximation
   level_collision
   ion_lines
   angular
   multishell
   microfield
   broadening
   collision
   convolutions
   lorentzian_fft
   profile_table
   validation
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
   * - :mod:`starkzee.atomic`
     - Public field-free wavefunctions, dipoles, strengths, and radiative rates
   * - :mod:`starkzee.stark_map`
     - Multi-shell DC Stark maps and target-state overlap diagnostics
   * - :mod:`starkzee.static_profile`
     - Static Stark-Zeeman profile solver and discrete-transition enumerator
   * - :mod:`starkzee.ffm`
     - Frequency Fluctuation Model (dynamic ion broadening)
   * - :mod:`starkzee.radiator`
     - Basis states, Hamiltonian construction, dipole matrix elements,
       oscillator strengths, Einstein A coefficients
   * - :mod:`starkzee.radiators`
     - Radiator interface of the solvers, hydrogenic radiator, Zeeman
       treatments (exact, weak, strong)
   * - :mod:`starkzee.ls_radiator`
     - Multi-electron radiator from LS terms and signed reduced E1 elements
   * - :mod:`starkzee.jlevel_radiator`
     - Multi-electron radiator from J-level data (FAC, Cowan RCG)
   * - :mod:`starkzee.level_data`, :mod:`starkzee.fac_reader`,
       :mod:`starkzee.cowan_reader`
     - J-level data model and the FAC and RCG readers
   * - :mod:`starkzee.coulomb_approximation`
     - Coulomb-approximation radial integrals (:math:`r` and :math:`r^2`)
   * - :mod:`starkzee.level_collision`
     - Electron-impact operator for non-degenerate levels
   * - :mod:`starkzee.single_valence`, :mod:`starkzee.he_i`
     - Ready-made radiators: Li-like n = 5 → 4 lines, He I 447.1 nm
   * - :mod:`starkzee.angular`
     - Angular algebra: Wigner 3-j/6-j symbols (pywigxjpf when installed, Racah sums otherwise), dipole, sin²θ and L·S elements
   * - :mod:`starkzee.multishell`
     - Experimental multi-shell CI basis, signed radial integrals, and
       inter-shell Stark and diamagnetic matrices
   * - :mod:`starkzee.microfield`
     - Holtsmark and Hooper microfield distributions, quadrature grid builder
   * - :mod:`starkzee.broadening`
     - GBK electron impact broadening with magnetic-field cutoff
   * - :mod:`starkzee.convolutions`
     - Standalone wavelength-space FFT Doppler and instrumental broadening helpers
   * - :mod:`starkzee.lorentzian_fft`
     - FFT sums of many Lorentzian / Voigt components (``method='fft'`` of the static profile)
   * - :mod:`starkzee.profile_table`
     - :class:`StaticProfileTable`: dressed states cached on a field grid for density/temperature scans
   * - :mod:`starkzee.validation`
     - Profile area, centroid, overlap coverage, and convergence-error metrics
   * - :mod:`starkzee.utils`
     - Physical constants (via scipy.constants) and unit conversions
   * - :mod:`starkzee.atomic_data`
     - NIST atomic energy-level database loader; :class:`AtomicState` dataclass
