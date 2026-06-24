Approximations and Numerical Choices
============================================

StarkZee is designed to be self-contained and fast enough for exploratory spectroscopy studies. The main approximations are explicit in the solver rather than hidden behind external atomic-structure packages:

#. **Hydrogenic basis and analytic radial functions.** Matrix elements are built from analytic hydrogenic wavefunctions [bethesalpeter]_,

   .. math::

      \begin{equation}
          R_{nl}(r) = \sqrt{\left(\frac{2Z}{n}\right)^3 \frac{(n-l-1)!}{2n(n+l)!}} e^{-Zr/n} \left(\frac{2Zr}{n}\right)^l L_{n-l-1}^{2l+1}\left(\frac{2Zr}{n}\right),
          
      \end{equation}

   giving self-contained :math:`r` and :math:`r^2` matrix elements without requiring Cowan, FAC, or another external structure code. Empirical NIST energies can still be injected for field-free level positions when accurate line centers matter.

#. **Within-shell Stark mixing.** The Stark operator is retained exactly inside each principal shell, but couplings to neighboring :math:`n` manifolds are neglected. This is appropriate below the Inglis-Teller regime, where Stark shifts remain small compared with the shell spacing.

#. **Quasi-static ions with optional dynamics.** The baseline profile treats ionic microfields as static during the radiative event and averages over their distribution. When ion motion is important, the FFM layer modifies the static profile through a fluctuation rate rather than rebuilding the atomic calculation.

#. **Electron-impact broadening.** Fast electrons enter through a Lorentzian impact width. By default (``frequency_de­pendent_width=True``) the GBK width :math:`\gamma_e(\Delta E)` is evaluated at each component detuning. Setting ``frequency_de­pendent_width=False`` fixes the width at the resonance-center value :math:`\gamma_e(0)`, which is faster and usually adequate when far wings are not the observable of interest.

#. **Microfield orientation quadrature.** The angular integral is evaluated with an :math:`N_\mu`-point Gauss-Legendre rule over :math:`\mu = \cos\theta \in [0,1]`, using the symmetry of the Hamiltonian under :math:`\mu\to-\mu`; see Section 3.7 .

