Physics background
==================

This page documents the physical model behind **StarkZee**.  The equations
follow the notation of Ferri, Peyrusse & Calisti (2022) [Ferri2022]_.

.. contents:: Contents
   :local:
   :depth: 2

Liouville space representation
-------------------------------

The spectral line-shape intensity :math:`I(\omega)` is the Fourier transform
of the dipole autocorrelation function [Baranger1958]_:

.. math::

    I(\omega) = \frac{1}{\pi}\,\mathrm{Re}
    \int_0^\infty C(t)\,e^{i\omega t}\,dt

In Liouville space notation:

.. math::

    C(t) = \langle\langle \vec{d}^{\,*} | U(t) | \vec{d}\,\rho_0 \rangle\rangle

where :math:`\vec{d}` is the electric transition dipole operator,
:math:`\rho_0` the equilibrium density matrix of the initial manifold, and
:math:`U(t)` the time-evolution propagator.  In the anti-symmetric subspace
the Liouvillian is

.. math::

    L = \frac{1}{\hbar}(H_u \otimes I^d - I \otimes H_l^d)

whose diagonal elements are the transition frequencies and whose off-diagonal
elements encode the Stark couplings.

Radiator Hamiltonian
--------------------

For a hydrogen-like radiator with nuclear charge :math:`Z` and principal
quantum number :math:`n` in an external field :math:`\vec{B} = B\hat{z}`,
the atomic Hamiltonian in the uncoupled
:math:`|n,l,m_l,s,m_s\rangle` basis is

.. math::

    H_A = H_0 + V_\text{SO} + H_Z^{(1)} + H_Z^{(2)}

**Unperturbed energy** (degenerate across the shell):

.. math::

    H_0 = -\frac{Z^2\,\text{Ry}}{n^2}

**Spin-orbit coupling** :math:`V_\text{SO} = \xi\,\vec{L}\cdot\vec{S}`:

.. math::

    \xi = \frac{Z^4\,\alpha^2\,\text{Ry}}{n^3\,l\,(l+\tfrac{1}{2})\,(l+1)}

where :math:`\alpha` is the fine-structure constant.  Together with the
mass-velocity and Darwin corrections (also included when
``fine_structure=True``), this reproduces the Dirac fine-structure
splitting and restores the :math:`2s_{1/2} = 2p_{1/2}` degeneracy
[BetheSalpeter1957]_.

**Linear Zeeman** (diagonal):

.. math::

    H_Z^{(1)} = \mu_B B\,(m_l + g_s\,m_s), \qquad g_s \approx 2.0023192

**Quadratic (diamagnetic) Zeeman**:

.. math::

    H_Z^{(2)} = \frac{e^2 B^2}{8 m_e}\,r^2\sin^2\theta

:math:`H_Z^{(2)}` is non-diagonal in :math:`l` (connects :math:`\Delta l = 0`
and :math:`\Delta l = \pm 2` states with the same :math:`m_l`, :math:`m_s`).
The off-diagonal radial integrals

.. math::

    \langle n,l_1 | r^2 | n,l_2 \rangle, \qquad |l_1 - l_2| = 2

do **not** reduce to a simple product of diagonal elements.  Using the
geometric-mean approximation
:math:`\sqrt{\langle r^2\rangle_{l_1}\langle r^2\rangle_{l_2}}`
overestimates the true value by 12 % (:math:`n=3`), 21 % (:math:`n=4`), and
41 % (:math:`n=5`).  StarkZee computes these integrals by direct numerical
quadrature (``scipy.integrate.quad``) over the analytical hydrogenic radial
wavefunctions [BetheSalpeter1957]_

.. math::

    R_{nl}(r) = \sqrt{\left(\frac{2Z}{n}\right)^{\!3}
    \frac{(n-l-1)!}{2n\,(n+l)!}}
    \;e^{-Zr/n}\!\left(\frac{2Zr}{n}\right)^{\!l}
    L_{n-l-1}^{2l+1}\!\!\left(\frac{2Zr}{n}\right)

with results cached after the first call.

Empirical field-free energies
-----------------------------

The analytic diagonal :math:`H_0 = -Z^2\,\text{Ry}_\text{red}/n^2`, together with
the spin-orbit and fine-structure terms, reproduces the field-free level
positions only to the accuracy of the hydrogenic Dirac formula.  For
quantitative line centers, StarkZee can instead inject *measured* field-free
energies through ``use_empirical_data=True`` (with the element symbol ``atom``).
The values are read from a tabulated database (``atomic_data.load_levels``)
holding NIST level energies in wavenumbers [cm\ :sup:`-1`], resolved by
:math:`(n, l, j)`.  The empirical Hamiltonian is kept in cm\ :sup:`-1`
throughout, so callers can operate entirely in wavenumber units; all Zeeman
contributions are converted from eV to cm\ :sup:`-1` before being added,
keeping every term on the same scale.

Because the field-free Hamiltonian is degenerate (the Dirac formula makes
:math:`2s_{1/2}` and :math:`2p_{1/2}` coincide), the empirical energies cannot
simply be placed on the diagonal of the uncoupled basis — ``numpy.linalg.eigh``
would mix the degenerate :math:`l` states arbitrarily.  StarkZee instead:

1. diagonalizes a degeneracy-broken field-free Hamiltonian (unperturbed energy
   + spin-orbit only, *omitting* the mass-velocity/Darwin term so that
   :math:`l` stays a good label), giving eigenvectors :math:`V`;
2. labels each coupled eigenstate :math:`k` by its dominant orbital component
   :math:`l` and its total angular momentum :math:`j = l \pm \tfrac{1}{2}`, the
   branch set by the sign of
   :math:`\langle \vec{L}\cdot\vec{S}\rangle = (E_k - E_n)/\xi_{nl}`;
3. assigns the tabulated energy :math:`D_k = E^\text{emp}(l, j)` [cm\ :sup:`-1`]
   to each eigenstate and reconstructs
   :math:`H_0^\text{emp} = V\,\mathrm{diag}(D)\,V^\dagger`.

The Zeeman terms (:math:`\mu_B B(m_l + g_s m_s)` and the diamagnetic
correction) are converted from eV to cm\ :sup:`-1` and added on top of
:math:`H_0^\text{emp}`.  Since the tabulated values are absolute level energies
(ground state at 0), transition wavenumbers follow directly as differences
:math:`E_u - E_l` and reproduce the observed NIST Lyman/Balmer line centers.

Full electron-radiator Hamiltonian
----------------------------------

Under the quasi-static ion approximation the radiator sits in a constant
microfield :math:`\vec{F}` at angle :math:`\theta` to :math:`\vec{B}`:

.. math::

    F_z = F\cos\theta, \qquad F_x = F\sin\theta

The Stark interaction is

.. math::

    V_E = -e\,(z\,F_z + x\,F_x) = -\hat{\boldsymbol{d}}\cdot\vec{F}

where :math:`\hat{\boldsymbol{d}} = e\vec{r}` is the electric dipole operator.
Within the :math:`n`-shell the radial element is analytic [BetheSalpeter1957]_:

.. math::

    \langle n,l | r | n,l-1 \rangle = \frac{3n}{2Z}\sqrt{n^2 - l^2}
    \quad [a_0]

The full electron-radiator Hamiltonian

.. math::

    H = H_A + V_E = H_0 + V_\text{SO} + H_Z^{(1)} + H_Z^{(2)} + V_E

is assembled and diagonalized **as a whole** by ``numpy.linalg.eigh`` at every
microfield quadrature point (``solve_starkzee``).  :math:`V_E` is **not**
treated perturbatively: there is no expansion in powers of the field.  The
Stark, Zeeman, and fine-structure terms all enter the same matrix on equal
footing, and the exact eigenvalues yield the Stark-dressed transition
frequencies :math:`\omega_k` and dipole weights :math:`|d_k|^2`.

Plasma microfield distributions
---------------------------------

The ion microfield is averaged over a probability distribution :math:`W(F)`.

**Holtsmark** (unscreened) [Holtsmark1919]_:

.. math::

    W_H(\beta) = \frac{2\beta}{\pi}
    \int_0^\infty y\,\sin(\beta y)\,e^{-y^{3/2}}\,dy,
    \qquad \beta = F/F_0

where :math:`F_0 = e/(4\pi\varepsilon_0 r_e^2)` is the Holtsmark normal field
and :math:`r_e = (3/4\pi N_e)^{1/3}` is the mean inter-particle distance.

**Hooper** (Debye-screened, default) [Hooper1968]_:

.. math::

    W(\beta, a) = \frac{2\beta}{\pi}
    \int_0^\infty y\,\sin(\beta y)\,
    e^{-y^{3/2}\,S(y,a)}\,dy

where the screening function is

.. math::

    S(y, a) = \left(1 + \frac{1.5\,a^2}{y^2}\right)^{-3/4},
    \qquad a = r_e / \lambda_D

and :math:`\lambda_D = \sqrt{\varepsilon_0 T_e / N_e e^2}` is the electron
Debye length.  Setting :math:`a = 0` recovers the Holtsmark distribution.

**Potekhin** (Zest-compatible, natively implemented) [Potekhin2002]_:

Fits for electric microfield distributions :math:`P(\beta)` based on Potekhin, Chabrier, and Gilles (2002) [Potekhin2002]_. It supports neutral/charged radiators and screened/unscreened cases.

- Unscreened Coulomb Potential (:math:`s = 0`):

  Neutral point:

  .. math::

      Q(\beta) = \frac{q_0 \beta^3 - 1.33 \beta^{9/2} + \beta^6}{q_1 + q_2 \beta^2 + q_3 \beta^3 - \frac{1}{3}\beta^{9/2} + \beta^6}

  with parameters :math:`q_n` as functions of the ion-ion coupling parameter :math:`\Gamma` (Eq. 17).

  Charged point:

  .. math::

      Q(\beta) = \frac{Q_0(\beta) + 0.873\sqrt{\Gamma} Q_M(\beta, \Gamma_\text{eff})}{1 + 0.873\sqrt{\Gamma}}

  where :math:`Q_M` is the Mayer distribution (Eq. 18).

- Screened Potential (Yukawa, :math:`s > 0`):

  Neutral point:

  .. math::

      Q(\beta) = \frac{a_0 \beta^3 - 2 \beta^{9/2} + \beta^6}{a_1 + a_2 \beta + a_3 \beta^2 + a_4 \beta^3 - \beta^{9/2} + \beta^6}

  with parameters :math:`a_n` as functions of screening parameter :math:`s` and :math:`\Gamma` (Eq. 30).

  Charged point:

  .. math::

      P(\beta) \approx \beta^2 S_N \left[ A e^{-a\beta^\alpha} + B e^{-b\beta^\gamma} + \frac{e^{-\Gamma\beta^{1/2}}}{1 + c\beta^{9/2}} \right]

  where :math:`S_N` is the normalization constant, and parameters :math:`A, a, \alpha, B, b, \gamma, c` are fitting functions of :math:`s` and :math:`\Gamma` (Eq. 36).

The profile integrator uses a uniform grid of ``num_f`` points in
:math:`[0, 10\,F_0]` and ``num_mu`` Gauss-Legendre points for the angle
:math:`\mu = \cos\theta \in [0, 1]`.

Electron impact broadening (GBK)
----------------------------------

Fast electrons are treated in the impact (completed-collision) approximation
[Griem1997]_ using the semi-classical Griem-Baranger-Kolb (GBK) model
[GriemBaranger1962]_.
Their contribution is a homogeneous Lorentzian broadening of half-width
(HWHM):

.. math::

    W_e(\Delta\omega) = W_\text{pref}\,\langle r^2\rangle_n
    \bigl[C_n + G(\Delta\omega)\bigr]

where the prefactor is

.. math::

    W_\text{pref} = \frac{4\pi}{3}\,N_e
    \sqrt{\frac{2 m_e}{\pi k_B T_e}}\,
    \left(\frac{\hbar}{m_e}\right)^{\!2}

and the shell-averaged mean-square radius is

.. math::

    \langle r^2\rangle_n = \frac{1}{n^2}
    \sum_{l=0}^{n-1}(2l+1)\,\frac{n^2}{2Z^2}
    \bigl[5n^2+1-3l(l+1)\bigr]\,a_0^2
    \;\propto\; \frac{n^4}{Z^2}

The GBK dynamical factor

.. math::

    G(\Delta\omega) = \tfrac{1}{2}\,E_1(y), \qquad
    y = \left(\frac{n^2}{2Z}\right)^{\!2}
    \frac{\Delta\omega^2 + \omega_c^2}{2\,\text{Ry}\cdot T_e}

uses the exponential integral :math:`E_1` and the cutoff frequency

.. math::

    \omega_c = \max(\omega_p,\;\omega_L,\;\omega_e)

where :math:`\omega_p = \sqrt{N_e e^2/\varepsilon_0 m_e}` is the plasma
frequency, :math:`\omega_L = eB/m_e` the electron Larmor frequency (dominant
at high :math:`B`), and :math:`\omega_e = 2\pi v_\text{th}/r_e` the
configuration-change rate.

Each Stark-dressed transition component :math:`(i \to j, q)` is broadened by
a Lorentzian of half-width :math:`\gamma_e`:

.. math::

    L_{ij}(\omega) = \frac{\gamma_e/\pi}{(\omega - \omega_{ij})^2 + \gamma_e^2}

and its contribution is weighted by :math:`|d_q(i,j)|^2` and the microfield
quadrature weight.

By default (``frequency_dependent_width=True``) :math:`\gamma_e` is evaluated
at the actual detuning of each component.  Setting
``frequency_dependent_width=False`` fixes it at the line-center value
:math:`\gamma_e(0)`, which is faster but less accurate in the far wings.

The strong-collision constants :math:`C_n` from Ferri *et al.* (2022):

===========  =====
n            C_n
===========  =====
≤ 2          1.50
3, 4         0.75
≥ 5          0.40
===========  =====

Thermal Doppler broadening
--------------------------

Thermal motion of the radiating ions Doppler-shifts each photon frequency by
:math:`\delta\omega = \omega_0\,v_z/c`.  Averaging over a Maxwell–Boltzmann
velocity distribution gives a Gaussian with :math:`1/e` half-width

.. math::

    \Delta E_D = E_0\sqrt{\frac{2\,T_i}{m_\text{ion}\,c^2/e}}

For H Balmer-:math:`\alpha` at :math:`T_i = 5` eV this gives
:math:`\Delta E_D \approx 0.062` meV — comparable to the Zeeman splitting at
1 T and negligible relative to the Stark width at
:math:`N_e \sim 10^{23}` m\ :sup:`-3`.

**Doppler broadening is not applied automatically.**  The static profile
solver returns the pure Stark-Zeeman result; convolutions are applied
afterwards via :mod:`starkzee.convolutions`:

.. math::

    I_\text{obs}(\lambda) =
    \bigl[I_\text{SZ}(\lambda) * G_D(\lambda)\bigr] * G_\text{inst}(\lambda)

where :math:`G_D` is the Doppler Gaussian and :math:`G_\text{inst}` the
instrumental slit function.  Both operate on a **uniform wavelength grid**;
convert energy grids to wavelength before calling.

Frequency Fluctuation Model
---------------------------

When ion dynamics are important [Talin1995]_ (high :math:`N_e`, low :math:`B`, or
high-:math:`n` lines) the static-ion approximation overestimates the central
peak height.  The FFM treats the microfield as a Markov jump process switching
between field configurations at rate :math:`\nu_i`.

The line profile is (Sherman–Morrison form):

.. math::

    I(\omega) = \frac{r^2}{\pi}\,\mathrm{Re}
    \frac{S(\omega)}{1 - \nu_i\,S(\omega)},
    \qquad
    S(\omega) = \sum_k
    \frac{p_k}{\nu_i + \gamma_k + i(\omega - \omega_k)}

where :math:`p_k = |d_k|^2 / r^2` are the normalized SDT weights,
:math:`\gamma_k` the electron-impact half-width, and
:math:`r^2 = \sum_k |d_k|^2`.  The ion fluctuation rate is

.. math::

    \nu_i = \frac{v_\text{th}}{r_i}\,\frac{\hbar}{e},
    \qquad v_\text{th} = \sqrt{\frac{2 T_i}{m_i}}

with :math:`r_i = (3/4\pi N_i)^{1/3}` the mean ion-sphere radius.

The limit :math:`\nu_i \to 0` recovers the static profile; the limit
:math:`\nu_i \to \infty` gives a single Lorentzian (motional narrowing).

Observation geometry
--------------------

The intensity observed at angle :math:`\theta` to :math:`\vec{B}` is

.. math::

    I(\theta) = I_\pi\sin^2\theta
    + \tfrac{1}{2}(I_{\sigma+} + I_{\sigma-})(1+\cos^2\theta)

Useful special cases:

- Transverse (:math:`\theta = 90°`): :math:`I_\pi + \tfrac{1}{2}(I_{\sigma+} + I_{\sigma-})`
- Along B (:math:`\theta = 0°`): :math:`I_{\sigma+} + I_{\sigma-}`
- Angle-averaged: :math:`\tfrac{2}{3}I_\pi + \tfrac{1}{3}(I_{\sigma+} + I_{\sigma-})`

At :math:`B = 0` the quantization axis is undefined; all three polarization
components are equal by spherical symmetry.

Oscillator strengths and line strengths
---------------------------------------

The line strength summed over all polarizations and substates is

.. math::

    S_{ul} = \sum_{q,i,j} |\langle l_j | r_q | u_i \rangle|^2 \quad [a_0^2]

The inter-shell hydrogenic radial dipole integrals
:math:`\langle n_l, l_l | r | n_u, l_u \rangle` are evaluated exactly with
Gordon's analytical formula [Gordon1929]_, and the angular factors follow the
spherical-tensor (Wigner-Eckart) algebra [Edmonds1957]_.

The weighted absorption oscillator strength is

.. math::

    gf = \frac{2}{3}\,\frac{\Delta E}{E_\text{H}}\,S_{ul}

and the Einstein A coefficient

.. math::

    A_{ul} = \frac{4\,\alpha^3}{3}\,
    \left(\frac{\Delta E}{E_\text{H}}\right)^{\!3}
    \frac{S_{ul}}{g_u\,\tau_\text{au}}

where :math:`E_\text{H} = 2\,\text{Ry}` is the Hartree energy,
:math:`g_u = 2n_u^2` the upper-shell statistical weight, and
:math:`\tau_\text{au} = \hbar / E_\text{H} \approx 2.419 \times 10^{-17}` s
the atomic unit of time.  Both :math:`gf` and :math:`A_{ul}` are validated
against NIST ASD tabulated values [NIST_ASD]_ to 0.5 % and 1 % respectively.

Physical features and validation
---------------------------------

Quadratic Zeeman polarization wings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At :math:`B \geq 500` T the diagonal QZ shifts within :math:`n=3` differ by
up to 5 meV, separating the :math:`3p\,(m_l=\pm 1)\to 2s` transitions (which
carry ≈ 21 % of the H\ :math:`\alpha` oscillator strength and the largest
:math:`n=3` QZ shift, :math:`+8.87` meV at :math:`B=1000` T) from the
dominant :math:`3d\to 2p` cluster by ≈ 7 meV.  The resulting wings appear
only in ``quadratic_zeeman=True`` profiles and are validated in
``test_12_halpha_qz_wings.py``.

±2μ\ :sub:`B`\ B Stark-Zeeman satellite features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A transverse microfield :math:`F_x` couples adjacent-:math:`m_l` states
within the same principal shell via
:math:`\Delta l = \pm 1,\;\Delta m_l = \pm 1` matrix elements.  The upper
eigenstate near :math:`E_{0,n} + 2\mu_B B` acquires a small admixture of the
neighbouring :math:`np` state, producing a :math:`\sigma^+` transition to the
zero-Zeeman lower state at photon energy :math:`E_0 + 2\mu_B B`.

**H**\ :math:`\beta` **(n=4→2).**
The satellite is a *distinct peak* at +117 meV (≈ 2 % of the main
:math:`\sigma^+` intensity), because :math:`n=4` has both
:math:`|4d,\,m_l{=}2\rangle` and :math:`|4f,\,m_l{=}2\rangle` *degenerate*
at :math:`+2\mu_B B`, greatly amplifying Stark mixing.

**H**\ :math:`\alpha` **(n=3→2).**
Only :math:`|3d,\,m_l{=}2\rangle` sits at :math:`+2\mu_B B` (no :math:`l=3`
substates exist for :math:`n=3`).  The satellite amplitude (≈ 0.07 % of main
peak) is buried in the Lorentzian tail of the :math:`\sigma^+` main peak, so
no distinct local maximum appears.

The mixing coefficient :math:`\beta = F_x\langle np|r|nd\rangle/(\mu_B B)`
scales as :math:`B^{-1}`, so :math:`\beta` is *larger* at lower :math:`B`.
However, the satellite position :math:`2\mu_B B` also shrinks with :math:`B`.
At :math:`B = 100` T, :math:`2\mu_B B = 11.6` meV falls inside the
Stark-broadened :math:`\sigma^+` cluster; the satellite merges into the tail
and is not resolved.  **High** :math:`B` **is required for the satellite to
appear as a separated peak.**  Validated in
``test_13_stark_zeeman_satellites.py``.

Numerical simplifications
--------------------------

- **Within-shell Stark matrix** — the quadratic Stark effect (coupling to
  :math:`n \pm 1` shells) is neglected.  This is valid when the Stark shift
  :math:`\ll Z^2\,\text{Ry}(1/n^2 - 1/(n+1)^2)`.

- **Resonance-center impact width (optional)** — setting
  ``frequency_dependent_width=False`` fixes :math:`\gamma_e` at its
  line-center value :math:`\gamma_e(0)`, avoiding repeated :math:`E_1`
  evaluations.  Faster but less accurate in the wings.

- **Gauss-Legendre angle quadrature** — the integration over microfield
  direction uses ``num_mu`` Gauss-Legendre points on
  :math:`\mu = \cos\theta \in [0, 1]` (default 6).

- **Cached radial integrals** — :math:`\langle r\rangle` and
  :math:`\langle r^2\rangle` matrix elements are computed once and cached
  with ``functools.lru_cache``.

Built-in reference models
-------------------------

The ``starkzee.models`` package provides five independent lineshape models that
share a common call signature and serve as benchmarks against StarkZee's fully
coupled solver.

**Tabulated models** (read precomputed NetCDF databases):

``stehle``
    The Stehlé MMM database [Stehle1999]_ contains the *unmagnetized* (:math:`B=0`)
    Stark + fine-structure profile for hydrogen Balmer/Paschen transitions, stored as
    a function of reduced detuning :math:`\Delta\omega/F_0` on a density/temperature
    grid.  Doppler broadening is applied first by FFT convolution, then Zeeman
    splitting is added *after* as a rigid normal-triplet shift of the
    Stark+Doppler profile:

    .. math::

        I(\theta) = \sin^2\!\theta\,I_\pi
        + \tfrac{1+\cos^2\!\theta}{2}\,(I_{\sigma+} + I_{\sigma-})

    Stark and Zeeman are treated as **separable** — valid only when
    :math:`\mu_B B \ll \Delta\omega_S`.  Covers arbitrary :math:`(n_u, n_l)`.

``rosato``
    The Rosato database [Rosato2009]_ solves Stark and Zeeman *jointly*; :math:`B`
    is an explicit table axis (:math:`B \in \{0,1,2,2.5,3,5\}` T).
    Interpolation in :math:`B` is *scaled*: each bracketing profile's detuning axis
    is stretched by :math:`B_\text{node}/B` before blending, exploiting the linear
    scaling of Zeeman splitting with :math:`B`.  Angle dependence comes from two
    real tables (parallel/perpendicular), blended as
    :math:`I = I_\parallel\cos^2\theta + I_\perp\sin^2\theta`.
    Restricted to deuterium Balmer lines,
    :math:`N_e \in [10^{13}, 10^{16}]` cm\ :sup:`-3`,
    :math:`T_e \in [0.316, 31.6]` eV, :math:`B \le 5` T.

Tabulated model comparison:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - Stehlé
     - Rosato
   * - Table contents
     - Field-free Stark only
     - Stark + Zeeman jointly
   * - :math:`B` enters as
     - Post-processing rigid triplet
     - Real table axis
   * - Angle dependence
     - Analytic :math:`\pi/\sigma` weights
     - Two precomputed angle tables
   * - Stark–Zeeman coupling
     - None (separable approx.)
     - Fully captured
   * - Coverage
     - Arbitrary :math:`(n_u,n_l)`, all :math:`B`
     - D Balmer only, :math:`B \le 5` T

**Analytical models** (no database required):

``lomanowski``
    Polynomial fits for the Stark FWHM from [Lomanowski2015]_:
    :math:`\Delta\lambda_S = c\,N_e^a\,T_e^{-b}` [nm],
    with tabulated :math:`(a,b,c)` for H/D Balmer/Paschen lines up to :math:`n_u=9`.

``stehle_param``
    A closed-form parametric fit to the Stehlé tables — fast approximation to the
    field-free Stark profile without reading the NetCDF database.

``voigt``
    Pseudo-Voigt using the Griem :math:`\alpha_{12}` Stark half-width [Griem1997]_
    plus Doppler broadening.  Suitable for order-of-magnitude estimates only.

**StarkZee vs.\ the tabulated models:**

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * - Feature
     - StarkZee
     - Stehlé
     - Rosato
   * - Magnetic field
     - Full simultaneous diagonalization of :math:`H = H_A + V_E`
     - :math:`B=0` tables, triplet convolved after
     - :math:`B` as table axis
   * - Fine structure
     - Spin-orbit, MV, Darwin
     - No (degenerate hydrogenic)
     - In tables
   * - Electron broadening
     - GBK semi-classical, frequency-dependent
     - Unified theory (rigorous far wings)
     - In tables
   * - Ion dynamics
     - Quasi-static + optional FFM
     - Static (some ion-dynamics in tables)
     - In tables
   * - B-treatment
     - Intrinsic to Hamiltonian
     - External convolution
     - B-scaled interpolation

The most important distinction at :math:`B \ne 0`: StarkZee diagonalizes
:math:`H = H_A + V_E` simultaneously for all fields.  When
:math:`\mu_B B \sim 3n\,e\,a_0\,F` (Zeeman and Stark splittings comparable), the
eigenstates are genuine Stark-Zeeman hybrids — neither pure Zeeman nor pure Stark
states.  No post-processing convolution reproduces this mixing.  For H\ :math:`\alpha`
at :math:`B = 10` T, :math:`\mu_B B \approx 0.58` meV and the Stark width at
:math:`N_e = 5\times10^{21}` m\ :sup:`-3` is :math:`\approx 0.8` meV — the same
order of magnitude, making the coupled treatment essential.

References
----------

.. [Ferri2022]
   S. Ferri, O. Peyrusse, A. Calisti,
   *Matter and Radiation at Extremes* **7**, 015901 (2022).

.. [Baranger1958]
   M. Baranger, *Phys. Rev.* **111**, 481 (1958);
   *Phys. Rev.* **112**, 855 (1958).

.. [BetheSalpeter1957]
   H. A. Bethe, E. E. Salpeter, *Quantum Mechanics of One- and Two-Electron
   Atoms*, Springer-Verlag (1957).

.. [Gordon1929]
   W. Gordon, *Ann. Phys.* **394**, 1031 (1929).

.. [Edmonds1957]
   A. R. Edmonds, *Angular Momentum in Quantum Mechanics*,
   Princeton University Press (1957).

.. [Talin1995]
   B. Talin, A. Calisti, L. Godbert, R. Stamm, R. W. Lee,
   *Phys. Rev. A* **51**, 1918 (1995).

.. [Holtsmark1919]
   J. Holtsmark, *Ann. Phys.* **58**, 577 (1919).

.. [Hooper1968]
   C. F. Hooper, *Phys. Rev.* **165**, 215 (1968).

.. [Griem1997]
   H. R. Griem, *Principles of Plasma Spectroscopy*,
   Cambridge University Press (1997).

.. [GriemBaranger1962]
   H. R. Griem, M. Baranger, A. C. Kolb, G. Oertel,
   *Phys. Rev.* **125**, 177 (1962).

.. [Potekhin2002]
   A. Y. Potekhin, G. Chabrier, D. Gilles,
   *Phys. Rev. E* **65**, 036412 (2002).

.. [NIST_ASD]
   A. Kramida, Yu. Ralchenko, J. Reader, NIST ASD Team,
   *NIST Atomic Spectra Database* (ver. 5.11),
   National Institute of Standards and Technology (2023).

.. [Stehle1999]
   C. Stehlé, R. Hutcheon,
   *Extensive tabulations of Stark broadened hydrogen line profiles*,
   *Astron. Astrophys. Suppl. Ser.* **140**, 93 (1999).

.. [Rosato2009]
   J. Rosato, H. Capes, R. Stamm,
   *Influence of ion dynamics and multiple body effects on hydrogen lines
   in magnetized fusion plasmas*,
   *Phys. Rev. E* **79**, 046408 (2009).

.. [Lomanowski2015]
   B. A. Lomanowski *et al.*,
   *Inferring divertor plasma properties from hydrogen Balmer and Paschen
   series spectroscopy in JET-ILW*,
   *Nucl. Fusion* **55**, 123028 (2015).
