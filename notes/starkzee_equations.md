# StarkZee Code Equations — As Implemented

Reference model: Ferri, Peyrusse & Calisti, *"Stark–Zeeman line-shape model for
multi-charged ion emission in a magnetized plasma"*, Matter Radiat. Extremes **7**,
015901 (2022). FFM: Calisti et al., Phys. Rev. A **42**, 5433 (1990).

This document extracts every equation currently implemented in the Python code,
organized by module, with `file:line` references.

Conventions: StarkZee works in **eV** for all energies (zest works in rad/s). The
uncoupled basis is `|n, l, m_l, m_s⟩` ordered (l, then m_l, then m_s) — see
`radiator.build_basis`.

---

## 1. Physical Constants & Units (`utils.py`, `radiator.py`)

Core constants are taken from `scipy.constants` (CODATA):

| Constant | Symbol | Source | Unit |
|----------|--------|--------|------|
| Bohr radius | a₀ | `physical_constants['Bohr radius']` | m |
| Rydberg energy (∞ mass) | R∞ | `Rydberg constant times hc in eV` | eV |
| Bohr magneton | μ_B | `Bohr magneton in eV/T` | eV/T |
| Fine-structure constant | α | `fine_structure` | — |
| ħ, m_e, e, c, ε₀, k_B, m_p | — | `scipy.constants` | SI |

### 1.1. Reduced-mass Rydberg
$$R_\text{atom}(Z,A) = \frac{R_\infty}{1 + m_e / M_\text{nucleus}(Z,A)}$$
- **Code**: [`reduced_mass_rydberg_ev`](starkzee/utils.py#L103) — isotope nuclear
  masses (H/D/T/⁴He) from CODATA; sets the absolute line center.

### 1.2. Unit conversions
$$E = \frac{hc}{\lambda},\quad E = hf,\quad E = hc\,\tilde\nu,\quad hc \approx 1239.84\ \text{eV·nm}$$
- **Code**: [`energy_ev_to_wavelength_nm`](starkzee/utils.py#L164),
  `*_to_frequency_thz`, `*_to_wavenumber_cm` (hc derived from ħ, c — no hardcoding).

### 1.3. Vacuum → air wavelength (Edlén 1966)
$$(n-1)\times10^8 = 8342.13 + \frac{2406030}{130-\sigma^2} + \frac{15997}{38.9-\sigma^2},\quad \sigma = \frac{1000}{\lambda_\text{vac}[\text{nm}]}$$
- **Code**: [`vacuum_to_air_wavelength_nm`](starkzee/utils.py#L192).

---

## 2. Atomic / Magnetic Hamiltonian (`radiator.py`)

The field-free + magnetic Hamiltonian for one shell n, in the `|n,l,m_l,m_s⟩` basis:
$$H_A = H_0 + V_\text{SO} + H_\text{MV+D} + H_Z^{(1)} + H_Z^{(2)}$$
- **Code**: [`build_hamiltonian`](starkzee/radiator.py#L391).

### 2.1. Hydrogenic radial wavefunction
$$R_{nl}(r) = N_{nl}\, e^{-Zr/n}\,(2Zr/n)^l\, L_{n-l-1}^{2l+1}(2Zr/n),\quad
N_{nl} = \sqrt{\left(\tfrac{2Z}{n}\right)^3 \frac{(n-l-1)!}{2n\,(n+l)!}}$$
- **Code**: [`radial_wavefunction`](starkzee/radiator.py#L53) (r in a₀).

### 2.2. Unperturbed energy (diagonal, degenerate over the shell)
$$E_n = -\frac{Z^2 R_\text{atom}(Z,A)}{n^2}$$
- **Code**: [lines 470–476](starkzee/radiator.py#L470). Uses the reduced-mass
  Rydberg → absolute energies match NIST.

### 2.3. Spin-orbit coupling
$$V_\text{SO} = \xi_{nl}\,\vec L\cdot\vec S,\qquad
\xi_{nl} = \frac{Z^4 \alpha^2 R_\infty}{n^3\,l\,(l+1)(l+\tfrac12)}$$
with $\vec L\cdot\vec S = L_z S_z + \tfrac12(L_+S_- + L_-S_+)$.
- **Code**: [lines 477–493](starkzee/radiator.py#L477). Off-diagonal in
  (m_l, m_s); the ladder terms couple $|m_l{+}1, m_s{-}1\rangle \leftrightarrow |m_l, m_s\rangle$.

### 2.4. Mass-velocity + Darwin (completes Dirac fine structure)
$$\Delta E_{l=0} = -A_\text{fs}\,(n - \tfrac34),\qquad
\Delta E_{l>0} = -A_\text{fs}\!\left(\frac{n}{l+\tfrac12} - \tfrac34\right),\qquad
A_\text{fs} = \frac{Z^4\alpha^2 R_\infty}{n^4}$$
- **Code**: [lines 495–503](starkzee/radiator.py#L495). Together with
  V_SO restores the Dirac degeneracy 2s₁/₂ = 2p₁/₂. Toggle: `fine_structure`.

### 2.5. Linear (paramagnetic) Zeeman
$$H_Z^{(1)} = \mu_B B\,(m_l + g_s m_s),\qquad g_s = |g_e|_\text{CODATA} \approx 2.00231930436$$
- **Code**: [lines 564–567](starkzee/radiator.py#L564).

### 2.6. Quadratic (diamagnetic) Zeeman
$$H_Z^{(2)} = \frac{e^2 B^2}{8 m_e}\, r^2 \sin^2\theta$$
Diagonal (Δl = 0) part uses the closed-form radial element and angular ⟨cos²θ⟩:
$$\langle r^2\rangle_{nl} = \frac{n^2}{2Z^2}\big[5n^2 + 1 - 3l(l+1)\big],\qquad
\langle\cos^2\theta\rangle_{l,m_l} = \frac{2l^2+2l-1-2m_l^2}{(2l-1)(2l+3)}$$
Off-diagonal (Δl = ±2) part (from sin²θ = 1 − cos²θ, the "1" drops for Δl=2):
$$\langle l, m_l|\cos^2\theta|l{+}2, m_l\rangle =
\sqrt{\frac{[(l{+}1)^2-m_l^2][(l{+}2)^2-m_l^2]}{(2l{+}1)(2l{+}3)^2(2l{+}5)}}$$
with the off-diagonal radial element $\langle n,l|r^2|n,l{\pm}2\rangle$ computed by
numerical integration.
- **Code**: [lines 568–592](starkzee/radiator.py#L568); radial off-diagonal
  [`radial_r2_element`](starkzee/radiator.py#L129). Toggle: `quadratic_zeeman`.

### 2.7. Angular dipole matrix elements (spherical tensor T_q^(1) of r̂)
$$q=0:\ \langle l,m|\cos\theta|l{+}1,m\rangle = \sqrt{\frac{(l{+}1)^2-m^2}{(2l{+}1)(2l{+}3)}}$$
$$q=\pm1:\ \langle\,\cdot\,|T_{\pm1}|\,\cdot\,\rangle = \mp\sqrt{\frac{(l\mp m)(l\mp m-1)}{2(2l-1)(2l+1)}}\ \text{(branch-dependent)}$$
- **Code**: [`angular_dipole_element`](starkzee/radiator.py#L302)
  (Condon-Shortley phases; nonzero only for |Δl|=1, Δm=q).

### 2.8. Radial dipole element
$$\langle n_1 l_1|r|n_2 l_2\rangle = \int_0^\infty R_{n_1 l_1}(r)\,r\,R_{n_2 l_2}(r)\,r^2\,dr$$
- **Code**: [`radial_dipole`](starkzee/radiator.py#L230) (numerical, |Δl|=1).

### 2.9. Diagonalization & dipole rotation
$$H_A\,|\psi_k\rangle = E_k\,|\psi_k\rangle,\qquad
d_q[i,j] = \langle\psi_l^j|\,r_q\,|\psi_u^i\rangle = \sum_{k_l,k_u}
U_l^{*}[k_l,j]\,U_u[k_u,i]\,(-R\,\text{ang}_q)$$
- **Code**: [`diagonalize_hamiltonian`](starkzee/radiator.py#L595),
  [`dipole_matrix_elements`](starkzee/radiator.py#L640),
  [`_uncoupled_dipole_matrices`](starkzee/radiator.py#L720) (units a₀).

### 2.10. Line strength, oscillator strength, Einstein A
$$S_{ul} = \sum_{q,i,j}\big|\langle l_j|r_q|u_i\rangle\big|^2\ [a_0^2]$$
$$gf = \frac{2}{3}\frac{\Delta E}{E_h}\,S_{ul},\qquad
A_{ul} = \frac{4\alpha^3}{3}\left(\frac{\Delta E}{E_h}\right)^3 \frac{S_{ul}}{2n_u^2\,\tau_\text{au}},\quad E_h = 2R_\infty$$
- **Code**: [`line_strength`](starkzee/radiator.py#L776),
  [`oscillator_strength`](starkzee/radiator.py#L812),
  [`einstein_a`](starkzee/radiator.py#L838).

---

## 3. Stark Perturbation & Static Profile (`static_profile.py`)

### 3.1. Linear (intra-shell) Stark matrix
$$V_E = -e\,(z\,F_z + x\,F_x),\qquad
\langle n,l|r|n,l{-}1\rangle = \frac{3n}{2Z}\sqrt{n^2-l^2}\ [a_0]$$
with $x/r = (T_{-1}+T_{+1})/\sqrt2$. Field-linear so $V_E = F_z M_z + F_x M_x$.
- **Code**: [`build_stark_matrix`](starkzee/static_profile.py#L70),
  templates [`_stark_templates`](starkzee/static_profile.py#L46).
- Quadratic (inter-n) Stark is **neglected** (valid when Stark shift ≪ Z²Ry/n³).

### 3.2. Combined Stark-Zeeman solve (inner loop)
$$H = H_A(B) + V_E(F_z, F_x),\qquad H|\psi_k\rangle = E_k|\psi_k\rangle$$
- **Code**: [`solve_starkzee`](starkzee/static_profile.py#L142).

### 3.3. Microfield magnitude quadrature
Fields $F = \beta F_0$ with weights $W(\beta)\,\Delta\beta$ (∑ weights ≈ 1).
- **Code**: [`microfield_quadrature`](starkzee/microfield.py#L496) (see §4).

### 3.4. Field-angle quadrature (μ = cos θ)
Gauss-Legendre mapped to **μ ∈ [0, 1]** with weights summing to 1:
$$\langle\cdot\rangle_\text{angle} = \int_0^1 (\cdot)\,d\mu = \tfrac12\!\int_{-1}^{1}(\cdot)\,d\mu$$
(the half-range is exact: the integrand is symmetric under F_z → −F_z, so the ½
solid-angle factor and the 2× hemisphere factor cancel — verified numerically).
For each (β, μ): $F_z = \beta F_0\,\mu$, $F_x = \beta F_0\sqrt{1-\mu^2}$.
- **Code**: [lines 292 & 359](starkzee/static_profile.py#L292).

### 3.5. Per-quadrature transition intensities
$$\Delta E_{ji} = E_u^i - E_l^j,\qquad
I_q[j,i] = \big|\,(U_l^\dagger D_q U_u)[j,i]\,\big|^2,\quad q\in\{0,+1,-1\}$$
- **Code**: [lines 379–382](starkzee/static_profile.py#L379).

### 3.6. Profile accumulation (Lorentzian or Voigt)
Each component is the microfield-weighted sum over transitions:
$$I_q(E) = \sum_{\beta,\mu} w_{\beta\mu}\sum_k I_q^k\,\mathcal{L}\!\big(E - \Delta E_k;\,w\big)$$
- Bare Lorentzian (no Doppler): $\mathcal{L}(x;w) = \dfrac{w/\pi}{x^2 + w^2}$.
- With Doppler ($T_i$ set): a Gaussian is accumulated in-loop and the Lorentzian
  applied by FFT afterwards (adaptive: which factor is in-loop depends on σ_D vs Δx).
- **Code**: [lines 389–401](starkzee/static_profile.py#L389). The half-width is
  $w = W_e(\Delta\omega) + w_\text{natural}$ (§5), with
  $w_\text{natural} = \hbar(\Gamma_u + \Gamma_l)/2$ from summed Einstein A.

### 3.7. Pseudo-Voigt (removed)
The Thompson (1987) pseudo-Voigt kernel `_pseudo_voigt` was never wired into the
profile pipeline (the code uses exact Gaussian/Lorentzian kernels + FFT) and was
removed as dead code in the 2026-08-22 review.

### 3.8. Observable polarization combinations
$$I(\theta) = I_\pi\sin^2\theta + \tfrac12(I_{\sigma+}+I_{\sigma-})(1+\cos^2\theta)$$
Transverse (90°): $I_\pi + \tfrac12(I_{\sigma+}+I_{\sigma-})$;
Parallel (0°): $I_{\sigma+}+I_{\sigma-}$;
Angle-averaged: $\tfrac23\,(I_\pi + I_{\sigma+}+I_{\sigma-})$
(both $\sin^2\theta$ and $\tfrac12(1+\cos^2\theta)$ average to $\tfrac23$ over the
sphere; isotropic check: $I_\pi = I_{\sigma\pm} = I \Rightarrow I(\theta) = 2I$
at every angle).
- **Code**: `line_profile.profile_at_angle`.

### 3.9. Discrete stick spectrum
Enumerate every (i, j, q) with $|d_q(i\to j)|^2 > $ threshold at a single (F_z, F_x).
- **Code**: [`discrete_transitions`](starkzee/static_profile.py#L426).

---

## 4. Microfield Distribution (`microfield.py`)

### 4.1. Holtsmark normal field & mean spacing
$$r_e = \left(\frac{3}{4\pi N_e}\right)^{1/3},\qquad
F_0 = \frac{e}{4\pi\varepsilon_0\, r_e^2}$$
- **Code**: [`calculate_normal_field`](starkzee/microfield.py#L11).

### 4.2. Debye length (classical)
$$\lambda_D = \sqrt{\frac{\varepsilon_0 T_e}{N_e\, e}}\quad(T_e\ \text{in eV})$$
Multi-species: $\lambda_D = \sqrt{\varepsilon_0 T_e / [N_e e (1 + \sum_i X_i Z_i^2)]}$.
- **Code**: [`calculate_debye_length`](starkzee/microfield.py#L38),
  [`calculate_multispecies_debye_length`](starkzee/microfield.py#L62).

### 4.3. Holtsmark distribution (unscreened)
$$W(\beta) = \frac{2\beta}{\pi}\int_0^\infty y\,\sin(\beta y)\,e^{-y^{3/2}}\,dy$$
- **Code**: [`holtsmark_distribution`](starkzee/microfield.py#L127).

### 4.4. Hooper screened distribution
$$W(\beta,a) = \frac{2\beta}{\pi}\int_0^\infty y\,\sin(\beta y)\,e^{-y^{3/2} S(y,a)}\,dy,\quad
S(y,a) = \left(1 + \frac{f_\text{emit}\,a^2}{y^2}\right)^{-3/4},\quad a = \frac{r_e}{\lambda_D}$$
where $f_\text{emit} = 1.5$ for a charged radiator (ion emitter, default `charged=True`), and $f_\text{emit} = 1.0$ for a neutral radiator (atom emitter, `charged=False`).
$a \to 0$ recovers Holtsmark.
- **Code**: [`hooper_distribution`](starkzee/microfield.py#L196).
- **NOTE**: depends on the screening parameter `a` and emitter charging `charged`; there is **no ion-coupling
  (Γ) correlation correction** (unlike zest's Potekhin fits).

### 4.5. Potekhin distribution models (Zest-compatible, natively implemented)
Fits for electric microfield distributions $P(\beta)$ based on Potekhin, Chabrier, and Gilles (*Phys. Rev. E* 65, 036412, 2002).
- **Code**: [`potekhin_distribution`](starkzee/microfield.py#L456).

#### 4.5.1. Unscreened Coulomb Potential ($s = 0$)
- **Neutral Point**:
  $$Q(\beta) = \frac{q_0 \beta^3 - 1.33 \beta^{9/2} + \beta^6}{q_1 + q_2 \beta^2 + q_3 \beta^3 - \frac{1}{3}\beta^{9/2} + \beta^6},\quad q_n = \alpha_n (1 + \beta_n \Gamma)^{-\gamma_n}$$
  with parameters $(\alpha_n, \beta_n, \gamma_n)$ as defined in Eq. 17.
- **Charged Point**:
  $$Q(\beta) = \frac{Q_0(\beta) + 0.873\sqrt{\Gamma} Q_M(\beta, \Gamma_\text{eff})}{1 + 0.873\sqrt{\Gamma}}$$
  where $Q_M$ is the Mayer distribution (Eq. 18):
  $$Q_M(\beta, \Gamma) = \text{erf}\left(\beta \sqrt{\frac{\Gamma}{2}}\right) - \sqrt{\frac{2\Gamma}{\pi}} \beta e^{-\Gamma\beta^2/2}$$

#### 4.5.2. Screened Potential (Yukawa, $s > 0$)
- **Neutral Point**:
  $$Q(\beta) = \frac{a_0 \beta^3 - 2 \beta^{9/2} + \beta^6}{a_1 + a_2 \beta + a_3 \beta^2 + a_4 \beta^3 - \beta^{9/2} + \beta^6}$$
  where parameters $a_0$ to $a_4$ are functions of $s$ and $\Gamma$ (Eq. 30–35).
- **Charged Point**:
  $$P(\beta) \approx \beta^2 S_N \left[ A e^{-a\beta^\alpha} + B e^{-b\beta^\gamma} + \frac{e^{-\Gamma\beta^{1/2}}}{1 + c\beta^{9/2}} \right]$$
  where $S_N$ is the normalization constant, and parameters $A, a, \alpha, B, b, \gamma, c$ are fitting functions of $s$ and $\Gamma$ (Eq. 36–52).

### 4.6. Quadrature normalization
Uniform β grid; W normalized so $\sum_i W_i\,\Delta\beta = 1$; returns
$\text{fields} = \beta F_0$, $\text{weights} = W\,\Delta\beta$.
- **Code**: [`_microfield_quadrature_impl`](starkzee/microfield.py#L569).

---

## 5. Electron-Impact Broadening (`broadening.py`)

GBK (Griem-Baranger-Kolb) semi-classical model with a magnetic cutoff.

### 5.1. Characteristic frequencies
$$\omega_p = \sqrt{\frac{N_e e^2}{\varepsilon_0 m_e}},\qquad
\omega_L = \frac{eB}{m_e},\qquad
\omega_e = \frac{2\pi}{\tau_e},\ \ \tau_e = \frac{r_e}{\sqrt{T_e e/m_e}}$$
- **Code**: [`calculate_plasma_frequency`](starkzee/broadening.py#L9),
  [`calculate_larmor_frequency`](starkzee/broadening.py#L32),
  [`calculate_configuration_frequency`](starkzee/broadening.py#L57).

### 5.2. Width prefactor
$$W_0 = \frac{4\pi}{3} N_e \sqrt{\frac{2 m_e}{\pi k_B T_e}}\left(\frac{\hbar}{m_e}\right)^2\frac{\hbar}{e}\quad[\text{eV/}a_0^2]$$
- **Code**: [`calculate_electron_impact_prefactor`](starkzee/broadening.py#L90).

### 5.3. GBK dynamical factor
$$G(\Delta\omega) = \tfrac12 E_1(y),\qquad
y = \left(\frac{n^2}{2Z}\right)^2\frac{\Delta\omega^2 + \omega_c^2}{\mathrm{Ry}_\infty\,T_e},\quad
\mathrm{Ry}_\infty = R_\infty \approx 13.6057\ \text{eV}$$
Note: Ry_∞ is the Rydberg energy (= e²/2a₀, ionization energy of hydrogen), **not** the Hartree (which is 2 Ry_∞).
- **Code**: [`gbk_model`](starkzee/broadening.py#L129). Δω and ω_c in eV.

### 5.4. Total electron half-width — PPPB/Ferri model (default)
$$W_e(\Delta\omega) = W_0\,\langle r^2\rangle_n\,\big[C_n + G(\Delta\omega,\omega_c)\big]$$

**Full shell-averaged squared radius** (statistical weight (2l+1), normalized by n²):
$$\langle r^2\rangle_n = \frac{1}{n^2}\sum_{l=0}^{n-1}(2l+1)\,\frac{n^2}{2Z^2}\big[5n^2+1-3l(l+1)\big]\quad[a_0^2]$$
Numerical values for H (Z=1): n=2: 33 a₀², n=3: 153 a₀², n=4: 468 a₀².
$$\omega_c = \max(\omega_p,\,\omega_L,\,\omega_e),\qquad
C_n = \begin{cases}1.50 & n\le2\\ 1.00 & n=3\\ 0.75 & n=4\\ 0.50 & n=5\\ 0.40 & n>5\end{cases}$$
- **Code**: [`electron_impact_width`](starkzee/broadening.py#L182). C_n from Ferri
  et al. (2021) Table 1; the magnetic cutoff ω_L suppresses the width at high B.
- **Frequency-dependent** by default: G is evaluated at each transition's detuning
  $\Delta E_i - E_0$ (`static_profile.py:367`).

### 5.5. ZEST electron broadening model (`electron_model='zest'`)

Same prefactor W₀ and G_n / G-function structure as §5.4, but with the **intra-shell**
squared-radius average replacing the full shell average:
$$W_e^\text{ZEST}(\Delta\omega) = W_0\,\langle r^2_\text{intra}\rangle_n\,\big[G_n + G_\text{ZEST}(\Delta\omega)\big]$$

**Justification — Δn = 0 interaction channels (Layzer complex).** The ZEST paper
(Calisti et al. 2014, §2.1) explicitly adopts the approximation that only states
belonging to the same Layzer complex (same principal quantum number n) may be mixed
by Stark or Zeeman effects ("Δn = 0 interaction channels").  The no-quenching
approximation also forbids transitions between the upper and lower manifolds.  Both
restrict the dipole sum in the broadening operator to within-shell matrix elements.
The restriction is also automatic from the G-function: inter-shell detunings
Δω_inter ≫ ω_p drive G(Δω_inter) → 0 exponentially (e.g. for n=3→2, the argument
is ~29000, giving G ≈ 0).

**Intra-shell squared radius (per-l).** From the intra-shell radial element
⟨n,l|r|n,l±1⟩ = (3n/2Z)√(n²−(l±1)²) and angular weight factors
C(l,l+1) = (l+1)/(2l+1), C(l,l−1) = l/(2l+1):
$$r^2_{\text{intra},l} = \frac{9n^2}{4Z^2}\big(n^2 - l(l+1) - 1\big)\quad[a_0^2]$$
This formula holds for all l ∈ [0, n−1], including the boundary cases l=0 (no l−1
neighbor) and l=n−1 (no l+1 neighbor).

**Shell average** (exact closed form, averaging over n² spatial states with weight (2l+1)):
$$\langle r^2_\text{intra}\rangle_n = \frac{9n^2(n^2-1)}{8Z^2}\quad[a_0^2]$$
Numerical values for H (Z=1): n=2: 13.5 a₀², n=3: 81 a₀², n=4: 270 a₀².
Ratio to full §5.4 average: n=2: 0.41, n=3: 0.53, n=4: 0.58.

**Maximum wave-number cutoff κ_m.** All three ZEST G-functions share a temperature-
and density-dependent upper cutoff replacing the fixed ρ_min of GBK:
$$\kappa_m = \min\!\left(\frac{Z}{n^2 a_0},\,\frac{Z\sqrt{2 m_e k_B T_e}}{\hbar\, n^2}\right)\quad[\text{m}^{-1}]$$
The first branch (Bohr radius limit) dominates at low T_e; the second (de Broglie
limit) at high T_e. With $x = \kappa_m \lambda_D$:

**G-function variants:**

*GBK-ZEST* (`electron_model='zest'` or `'zest-gbk'`):
$$G^\text{ZEST}(\Delta\omega) = \tfrac{1}{2}E_1\!\left(\frac{\Delta\omega^2 + \omega_p^2}{2\,x^2\,\omega_p^2}\right)$$
Recovers the GBK E₁ form with κ_m-based cutoff (ω_p only; no Larmor or ω_e term).

*Lee* (`'zest-lee'`): interpolates between the static and impact limits
$$G^\text{Lee}(\Delta\omega) = \min\!\left(G_0,\, G_\infty\right)$$
$$G_0 = \tfrac{1}{2}\!\left[\ln(1+x^2) - \frac{x^2}{1+x^2}\right],\qquad
G_\infty = \tfrac{1}{2}E_1\!\left(\frac{\Delta\omega^2}{2\,x^2\,\omega_p^2}\right)$$
$G_0$ is the Δω → 0 (static) limit; $G_\infty$ is the far-wing (impact) limit.

*Dufty RPA* (`'zest-dufty'`): numerically integrates the RPA response
$$G^\text{RPA}(\Delta\omega) = \int_0^{\kappa_m} \frac{e^{-\kappa^2/2\kappa_m^2}}{\kappa\,|\varepsilon(\kappa,\Delta\omega)|^2}\,d\kappa$$
where $\varepsilon(\kappa,\omega)$ is the dielectric function evaluated with the Dawson
function, capturing plasma-wave corrections beyond the GBK binary-collision picture.

**G_n constants** are identical to the C_n values in §5.4.

- **Code**: [`electron_impact_width_zest`](starkzee/broadening.py#L527).
  Activated by `electron_model='zest'`/`'zest-gbk'`/`'zest-lee'`/`'zest-dufty'`.

---

## 6. Frequency Fluctuation Model — ion dynamics (`ffm.py`)

### 6.1. Ion fluctuation (jump) rate
$$N_i = \frac{N_e}{Z},\quad r_i = \left(\frac{3}{4\pi N_i}\right)^{1/3},\quad
v_\text{th} = \sqrt{\frac{2 T_i e}{A m_p}},\qquad
\nu_i = \frac{v_\text{th}}{r_i}\cdot\frac{\hbar}{e}\ [\text{eV}]$$
- **Code**: [`calculate_ion_fluctuation_rate`](starkzee/ffm.py#L10).

### 6.2. Stark-dressed transitions (SDTs)
Per polarization q, accumulate over the (β, μ) quadrature the dressed frequencies
$\Delta E_k$ and weights $d_k^2 = w_{\beta\mu}\,I_q^k$.
- **Code**: [lines 160–191](starkzee/ffm.py#L160) (uses `solve_starkzee`).

### 6.3. FFM line shape — Sherman-Morrison (default)
$$I_q(\omega) = \frac{R_q^2}{\pi}\,\mathrm{Re}\!\left[\frac{S(\omega)}{1 - \nu_i S(\omega)}\right],\qquad
S(\omega) = \sum_k \frac{p_k}{\nu_i + \gamma_k + i(\omega - \omega_k)}$$
with $p_k = d_k^2 / \sum_k d_k^2$, $R_q^2 = \sum_k d_k^2$, and the homogeneous
half-width
$$\gamma_k = W_e(0) + w_\text{natural},\qquad w_\text{natural} = \frac{\hbar(\Gamma_u + \Gamma_l)}{2}$$
$\Gamma_u = \sum_{k<n_u} A(n_u \to k)$, $\Gamma_l = \sum_{k<n_l} A(n_l \to k)$ (summed
Einstein A over all lower levels) — single on-resonance value for all SDTs.
- **Code**: [lines 240–247](starkzee/ffm.py#L240). Natural linewidth via
  [`einstein_a`](starkzee/radiator.py#L838).
- Limits: $\nu_i \to 0$ → quasi-static (static profile); $\nu_i \to \infty$ →
  motional-narrowing (single Lorentzian).

### 6.4. FFM — full Liouville matrix inversion (optional)
$$A_{mj} = \delta_{mj}\big[(\omega - \omega_j) - i(\gamma_k + \nu_i)\big] + i\nu_i\,p_j,\qquad
I(\omega) = \frac{1}{\pi}\,\mathrm{Re}\Big(i\,\textstyle\sum_k d_k\,[A^{-1}b]_k\Big)$$
- **Code**: [lines 216–236](starkzee/ffm.py#L216) (`numerical_inversion=True`;
  falls back to Sherman-Morrison if singular).

### 6.5. FFM — thermal Doppler broadening
After the Markov accumulation, a Gaussian Doppler kernel is applied by zero-padded
FFT (same strategy as `calculate_static_profile`):
$$\sigma_D = E_0\sqrt{\frac{T_i}{m_\text{ion}c^2}},\qquad
\tilde{I}(k) \leftarrow \tilde{I}(k)\cdot e^{-2\pi^2 \sigma_D^2 k^2}$$
where $k$ are the frequency-domain grid points of the zero-padded ($2N$) array.
Controlled by `apply_doppler=True` (default); set `False` to obtain the
purely Stark-Zeeman FFM output without Doppler.
- **Code**: [lines 297–313](starkzee/ffm.py#L297).

---

## 7. Post-Processing Convolutions (`convolutions.py`)

### 7.1. Doppler width
$$\Delta E_D = E_0\,\frac{v_\text{th}}{c},\quad v_\text{th} = \sqrt{\frac{2 T_i}{m c^2/e}},\qquad
\text{FWHM} = 2\sqrt{\ln 2}\,\Delta E_D$$
Gaussian kernel $\exp[-(E-E_0)^2/\Delta E_D^2]$ (1/e half-width convention).
- **Code**: [`calculate_doppler_width_ev`](starkzee/convolutions.py#L9),
  [`apply_doppler_broadening`](starkzee/convolutions.py#L106).

### 7.2. Instrumental (slit) broadening
$$\sigma = \frac{\text{FWHM}}{2\sqrt{2\ln 2}},\qquad K(\lambda) = \exp\!\left[-\frac{(\lambda-\bar\lambda)^2}{2\sigma^2}\right]$$
- **Code**: [`apply_instrument_broadening`](starkzee/convolutions.py#L152).

### 7.3. FFT convolution
$$(f*g)[n] = \mathrm{IFFT}\big(\mathrm{FFT}(f)\cdot\mathrm{FFT}(g)\big)$$
Area-normalized kernel; edge-padded profile, zero-padded kernel.
- **Code**: [`convolve_fft`](starkzee/convolutions.py#L53).

---

## 8. Scope & Approximations (summary)

- **Hydrogenic radiator** (one active electron, charge Z); no quantum defects.
- **Intra-shell linear Stark only** — quadratic / inter-n Stark neglected.
- Full magnetic Hamiltonian: spin-orbit + Dirac fine structure + linear and
  **quadratic (diamagnetic) Zeeman** simultaneously diagonalized with the Stark term.
- **Microfield**: Holtsmark / Hooper-screened (classical Debye, neutral-point), plus native implementations of the Zest-compatible Potekhin (2002) model (incorporating ion-coupling $\Gamma$ and screened/unscreened, neutral/charged points).
- **Electron broadening**: Two models — (1) default PPPB/Ferri: GBK semi-classical,
  frequency-dependent, magnetic (Larmor) cutoff, full shell-averaged ⟨r²⟩_n;
  (2) ZEST model (`electron_model='zest'`/`'zest-gbk'`/`'zest-lee'`/`'zest-dufty'`):
  intra-shell ⟨r²_intra⟩_n = (9n²/8Z²)(n²−1), κ_m-based G-function, ω_p cutoff
  only (Δn = 0 / Layzer complex restriction).
- **Ion dynamics**: quasi-static + optional FFM (Sherman-Morrison or full inversion).
- **Doppler**: inside `calculate_ffm_profile` by default (`apply_doppler=True`,
  zero-padded FFT); also available via `convolutions.py` for post-processing.
- **Instrumental**: post-processing FFT convolution via `convolutions.py`.
