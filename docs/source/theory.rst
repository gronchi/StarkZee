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
of the dipole autocorrelation function:

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
splitting and restores the :math:`2s_{1/2} = 2p_{1/2}` degeneracy.

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
wavefunctions

.. math::

    R_{nl}(r) = \sqrt{\left(\frac{2Z}{n}\right)^{\!3}
    \frac{(n-l-1)!}{2n\,(n+l)!}}
    \;e^{-Zr/n}\!\left(\frac{2Zr}{n}\right)^{\!l}
    L_{n-l-1}^{2l+1}\!\!\left(\frac{2Zr}{n}\right)

with results cached after the first call.

Stark perturbation
------------------

Under the quasi-static ion approximation the radiator sits in a constant
microfield :math:`\vec{F}` at angle :math:`\theta` to :math:`\vec{B}`:

.. math::

    F_z = F\cos\theta, \qquad F_x = F\sin\theta

The Stark interaction is

.. math::

    V_E = -e\,(z\,F_z + x\,F_x)

Within the :math:`n`-shell the radial element is analytic:

.. math::

    \langle n,l | r | n,l-1 \rangle = \frac{3n}{2Z}\sqrt{n^2 - l^2}
    \quad [a_0]

The combined Hamiltonian :math:`H = H_A + V_E` is diagonalized by
``numpy.linalg.eigh`` at every quadrature point to obtain the Stark-dressed
transition frequencies :math:`\omega_k` and dipole weights :math:`|d_k|^2`.

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

The profile integrator uses a uniform grid of ``num_f`` points in
:math:`[0, 10\,F_0]` and ``num_mu`` Gauss-Legendre points for the angle
:math:`\mu = \cos\theta \in [0, 1]`.

Electron impact broadening (GBK)
----------------------------------

Fast electrons are treated in the impact (completed-collision) approximation
[Griem1997]_.
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
against NIST ASD tabulated values to 0.5 % and 1 % respectively.

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

References
----------

.. [Ferri2022]
   S. Ferri, O. Peyrusse, A. Calisti,
   *Matter and Radiation at Extremes* **7**, 015901 (2022).

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
