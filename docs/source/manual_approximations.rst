Approximations and Numerical Choices
============================================

StarkZee is designed to be self-contained and fast enough for exploratory spectroscopy studies. The main approximations are explicit in the solver rather than hidden behind external atomic-structure packages:

#. **Hydrogenic basis and analytic radial functions.** Matrix elements are built from analytic hydrogenic wavefunctions [bethesalpeter]_,

   .. math::

          R_{nl}(r) = \sqrt{\left(\frac{2Z}{n}\right)^3 \frac{(n-l-1)!}{2n(n+l)!}} e^{-Zr/n} \left(\frac{2Zr}{n}\right)^l L_{n-l-1}^{2l+1}\left(\frac{2Zr}{n}\right),


   giving self-contained :math:`r` and :math:`r^2` matrix elements without requiring Cowan, FAC, or another external structure code. Empirical NIST energies can still be injected for field-free level positions when accurate line centers matter.

#. **Within-shell Stark mixing in production profiles.** The Stark operator is retained exactly inside each principal shell, but couplings to neighboring :math:`n` manifolds are neglected by the static and FFM solvers. This is appropriate below the Inglis-Teller regime, where Stark shifts remain small compared with the shell spacing. The experimental :mod:`starkzee.multishell` module now supplies signed inter-shell Stark blocks for standalone convergence studies, but has not yet been integrated into profile accumulation.

#. **Within-shell quadratic Zeeman in production profiles.** Like the Stark operator above, the production diamagnetic term :math:`H_{QZ} = (e^2B^2/8m_e)\,r^2\sin^2\theta` is diagonalized inside a single principal shell. Ferri, Peyrusse & Calisti (2022) show that inter-:math:`n` mixing matters at fields relevant to magnetized white-dwarf atmospheres. The experimental :mod:`starkzee.multishell` reference now includes signed :math:`\langle n,l_1|r^2|n',l_2\rangle` elements, full truncated-basis diamagnetic blocks, and coherent target/neighbor/interference oscillator-strength accounting. Solver integration, quantitative truncation criteria, and external profile validation remain open, so existing profile examples still cannot claim reproduction of Ferri Fig. 1(c).

   Five internal H-beta regressions now sample 100--1000 T and fixed electric fields from zero through three Holtsmark normal fields at three field angles. The minimum tested passing basis grows with field strength. Their finite-reference energy and ``gf`` tolerances apply only at those declared points; continuous microfield quadrature and other transitions remain open.

#. **Quasi-static ions with optional dynamics.** The baseline profile treats ionic microfields as static during the radiative event and averages over their distribution. When ion motion is important, the FFM layer modifies the static profile through a fluctuation rate rather than rebuilding the atomic calculation.

#. **Electron-impact broadening.** The default static model evaluates the width at the shared observation-energy detuning from the fixed zero-field reference, not separately at each transition center. Constant-width mode evaluates it at resonance. Both are approximate prescriptions; compare them for the requested observable and converge the sampled intrinsic kernels before Doppler convolution. The opt-in ``electron_interference=True`` mode instead constructs the PPP Appendix-B impact-limit operator, including its upper--lower interference term, and diagonalizes the resulting non-Hermitian optical-coherence generator. It currently requires a constant :math:`G(0)` kernel and a finite within-shell intermediate-state closure; neither its closure nor its physical accuracy has yet been accepted against PPP/PPPB output.

#. **Microfield orientation quadrature.** The angular integral uses Gauss-Legendre quadrature over the nonnegative cosine of the field angle. Reflection symmetry of the spectrum under reversal of that cosine justifies the half-domain integration; the Hamiltonian matrix itself is not even in that coordinate.

