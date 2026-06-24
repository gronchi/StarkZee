Physical Approximations and Domains of Validity
=======================================================

.. list-table:: Key physical approximations, their domains of validity, and failure modes.
   :header-rows: 1
   :widths: 24 26 25 25

   * - Approximation
     - Mathematical form
     - Domain of validity
     - Failure mode
   * - **Within-Shell Isolation** (:math:`\Delta n = 0`)
     - :math:`V_E = \mathrm{diag}_n(V_E)`, no :math:`n \to n\!\pm\!1` coupling
     - Stark shift :math:`\ll` shell spacing :math:`2Z^2\mathrm{Ry}/n^3`
     - Quadratic Stark; Inglis-Teller merging at :math:`N_e \gtrsim 10^{24}` m\ :sup:`-3`.
   * - **Quasi-Static Ion Microfields**
     - :math:`\vec{F}_\mathrm{ion} = \mathrm{const}`
     - Fluctuation rate :math:`\nu_i \ll` Stark width :math:`\Delta E_S`
     - Ion dynamics / motional narrowing; use FFM.
   * - **Semi-Classical Electron Impact**
     - Lorentzian with GBK width :math:`\gamma_e`
     - :math:`\lambda_{dB} \ll` impact parameter :math:`\rho`
     - Quantum close-coupling needed for cold dense plasmas.
   * - **Dipole Approximation**
     - :math:`V_\mathrm{rad} \propto \vec{r}\cdot\vec{E}`
     - :math:`\lambda \gg \langle r\rangle_n \sim n^2 a_0/Z`
     - Quadrupole/octupole for high-:math:`n` Rydberg states.
