Physics Explanation of Use Cases and Operational Limits
===============================================================

The High-Magnetic-Field Regime
--------------------------------------

In laboratory magnetic confinement fusion devices (tokamaks, stellarators), the magnetic field reaches :math:`B \approx 1`--:math:`10` T:

- The Zeeman splitting :math:`\Delta E_Z \approx 0.1` meV is comparable to the fine-structure splitting of lower shells, requiring the full coupled Hamiltonian diagonalization.

- Typical microfield strengths :math:`F \sim 10^5`--:math:`10^6` V/m represent a weak-Stark regime where Stark broadening acts as a symmetric perturbation on the Zeeman triplet structure.

For astrophysical compact objects (magnetized white dwarfs with :math:`B \sim 10^3`--:math:`10^5` T, neutron stars with :math:`B \sim 10^8` T):

- The quadratic Zeeman term :math:`H_Z^{(2)} \propto B^2` dominates, causing significant blue-shifting and highly asymmetric splitting. The ``quadratic_zeeman=True`` flag enables exact numerical treatment of the :math:`\Delta l = \pm 2` coupling (see the Radiator Hamiltonian section).

- StarkZee's exact numerical computation of :math:`\langle n, l_1 | r^2 | n, l_2\rangle` avoids the geometric-mean overestimation of up to 41% for :math:`n=5`.

Density Limits
----------------------

- **Low density** (:math:`N_e \lesssim 10^{18}` m\ :sup:`-3`): The profile is dominated by thermal Doppler broadening; microfield weights converge to a delta function at :math:`F=0`.

- **High density** (:math:`N_e \gtrsim 10^{24}` m\ :sup:`-3`): The inter-particle spacing :math:`r_e` approaches the atomic radius :math:`\langle r\rangle_n`. The static and :math:`\Delta n = 0` approximations break down as the Stark shift exceeds the Rydberg shell spacing (Inglis-Teller limit).




