# Report: Thermal Motional Stark Effect (TMSE) Line Broadening Relevance

This report evaluates when the **Thermal Motional Stark Effect (TMSE)** is relevant for spectral line shape modeling in magnetized plasmas and whether it should be implemented in **StarkZee**. This analysis is based on the paper *"Stark broadening by Lorentz fields in magnetically confined plasmas"* (J. Rosato, Y. Marandet, R. Stamm, 2014) located in the workspace ([Lorentz.html](file:///c:/Users/ronchig/Documents/StarkZee/Lorentz.html)).

---

## 1. Physics Background

When a radiating neutral atom moves with velocity $\vec{v}$ through an external magnetic field $\vec{B}$, it experiences an effective electric field in its rest frame due to the Lorentz force:
$$\vec{F}_L = \vec{v} \times \vec{B}$$

For a thermalized population of atoms, the velocities follow a Maxwell-Boltzmann distribution. This yields a statistical distribution of Lorentz fields across the emitters, which manifests as an inhomogeneous Stark broadening (TMSE).

The characteristic frequency width of this broadening scales as:
$$\omega_{\text{TMSE}} \propto n^2 B \sqrt{T_{\text{at}}}$$
where:
* $n$ is the principal quantum number of the upper state of the transition.
* $B$ is the magnetic field strength.
* $T_{\text{at}}$ is the atomic (neutral) temperature.

---

## 2. Competition with Other Broadening Mechanisms

TMSE competes with three main broadening mechanisms. It becomes dominant under the following specific regimes:

### A. TMSE vs. Doppler Broadening
The Doppler FWHM is given by $\Delta \lambda_D = 2 \sqrt{\ln 2} \frac{\lambda_0 v_T}{c}$, where $v_T = \sqrt{2 T_{\text{at}} / m_{\text{at}}}$. The ratio of the TMSE width to the Doppler width scales as:
$$\frac{\omega_{\text{TMSE}}}{\omega_D} \propto \frac{n^2 B}{\omega_0}$$
- **Key Insight**: Because the transition frequency $\omega_0$ converges to a constant at high $n$ (near the series limit) and the Stark sensitivity grows as $n^2$, **the ratio is independent of temperature**.
- **Relevance**: At **high $n$** (moderately high $n$ of order 10) and/or **high $B$** (several teslas), TMSE broadening always exceeds Doppler broadening.

### B. TMSE vs. Plasma Microfield (Stark) Broadening
Standard Stark broadening is caused by the electrostatic microfield from plasma ions, characterized by the Holtsmark field $F_0 \propto N_e^{2/3}$. The ratio between the typical Lorentz field $F_L \approx v_T B$ and the Holtsmark field $F_0$ scales as:
$$\frac{F_L}{F_0} \propto \frac{B \sqrt{T_{\text{at}}}}{N_e^{2/3}}$$
- **Key Insight**: In **low-density, high-field** environments, the Lorentz field is much stronger than the plasma microfield.
- **Relevance**: For typical magnetic fusion divertor conditions ($T_{\text{at}} = 1\text{ eV}$, $B = 3\text{ T}$, and $N_e = 10^{13} \text{ cm}^{-3}$), the Lorentz field $F_L$ is **about 10 times larger** than the Holtsmark field $F_0$. In this regime, TMSE is the dominant Stark mechanism.

### C. Revisited Inglis–Teller Limit
The Inglis-Teller limit determines the last resolvable line in a spectral series before merging into the continuum. Under strong magnetic fields and low densities, this merging is dominated by TMSE rather than the plasma microfield. The modified limit satisfies:
$$n^{10} B^2 T_{\text{at}} \simeq 1.5 \times 10^{14} \frac{m_{\text{at}}}{m_p}$$
- **Key Insight**: Neglecting TMSE in high-field, low-density diagnostics leads to a significant **overestimation of the plasma density** when using line-merging methods.

---

## 3. Implementation Verdict for StarkZee

| Plasma Regime | Transition Scope | Priority | Recommendation |
| :--- | :--- | :--- | :--- |
| **High Density** ($N_e \ge 10^{15}\text{ cm}^{-3}$) | All lines | **Low** | Do not implement; Holtsmark microfield $F_0$ dominates. |
| **Low Density** ($N_e \lesssim 10^{14}\text{ cm}^{-3}$), **High Field** ($B \ge 3\text{ T}$) | Low-$n$ (e.g. H-$\alpha$, H-$\beta$) | **Low** | Do not implement; Doppler/Zeeman dominate. |
| **Low Density** ($N_e \lesssim 10^{14}\text{ cm}^{-3}$), **High Field** ($B \ge 3\text{ T}$) | High-$n$ (e.g. $n \ge 8 - 10$ up to $16$) | **High** | **Implement**; TMSE is the dominant Stark mechanism. |

### Implementation Strategy
In the `StarkZee` solver, the total electric field in the radiator frame would be computed as the vector sum of the electrostatic ion microfield ($\vec{F}_{\text{ion}}$) and the motional field ($\vec{F}_L = \vec{v} \times \vec{B}$):
$$\vec{F}_{\text{total}} = \vec{F}_{\text{ion}} + \vec{v} \times \vec{B}$$

Since $\vec{v}$ is Maxwellian, $\vec{F}_L$ represents a 2D Gaussian distribution of fields perpendicular to $\vec{B}$. This can be implemented by:
1. Convolving the microfield distribution with the 2D Gaussian distribution of the Lorentz field of width $\sigma_F = B \sqrt{k_B T_{\text{at}} / M}$ in the transverse plane, or
2. Adding $\vec{F}_L$ as a vector quadrature loop in `static_profile.py`.
