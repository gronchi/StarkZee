Physics Explanation of Use Cases and Operational Limits
===============================================================

The High-Magnetic-Field Regime
--------------------------------------

In laboratory magnetic confinement fusion devices (tokamaks, stellarators), the magnetic field reaches :math:`B \approx 1`--:math:`10` T:

- The Zeeman splitting :math:`\Delta E_Z \approx 0.1` meV is comparable to the fine-structure splitting of lower shells, requiring the full coupled Hamiltonian diagonalization.

- Typical microfield strengths :math:`F \sim 10^5`--:math:`10^6` V/m represent a weak-Stark regime where Stark broadening acts as a symmetric perturbation on the Zeeman triplet structure.

For astrophysical compact objects (magnetized white dwarfs with :math:`B \sim 10^3`--:math:`10^5` T, neutron stars with :math:`B \sim 10^8` T):

- The quadratic Zeeman term :math:`H_Z^{(2)} \propto B^2` dominates, causing significant blue-shifting and highly asymmetric splitting. The ``quadratic_zeeman=True`` flag enables exact numerical treatment of the :math:`\Delta l = \pm 2` coupling *within a single principal shell* (see the Radiator Hamiltonian section).

- StarkZee's exact numerical computation of :math:`\langle n, l_1 | r^2 | n, l_2\rangle` avoids the geometric-mean overestimation of up to 41% for :math:`n=5`, but this is still an intra-shell (same-:math:`n`) matrix element — **inter-:math:`n` configuration-interaction mixing driven by the quadratic term is not included.** Ferri, Peyrusse & Calisti (2022) show this coupling is crucial precisely in this regime (:math:`B \sim 10^2`\ –\ :math:`10^3` T, white-dwarf-like conditions): it produces a red shift of the high-PQN Balmer lines that grows with the field, which StarkZee's intra-shell treatment cannot reproduce. See :doc:`manual_approximations` and TODO item 1 for details.

Density Limits
----------------------

- **Low density** (:math:`N_e \lesssim 10^{18}` m\ :sup:`-3`): The profile is dominated by thermal Doppler broadening; microfield weights converge to a delta function at :math:`F=0`.

- **High density** (:math:`N_e \gtrsim 10^{24}` m\ :sup:`-3`): The inter-particle spacing :math:`r_e` approaches the atomic radius :math:`\langle r\rangle_n`. The static and :math:`\Delta n = 0` approximations break down as the Stark shift exceeds the Rydberg shell spacing (Inglis-Teller limit).




