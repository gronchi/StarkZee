# StarkZee Code Equations — As Implemented

Reference model: Ferri, Peyrusse & Calisti, *"Stark–Zeeman line-shape model for
multi-charged ion emission in a magnetized plasma"*, Matter Radiat. Extremes **7**,
015901 (2022). FFM: Calisti et al., Phys. Rev. A **42**, 5433 (1990).

This document extracts every equation currently implemented in the Python code,
organized by module, with `file:line` references.

Conventions: StarkZee works in **eV** for all energies (zest works in rad/s). The
uncoupled basis is `|n, l, m_l, m_s⟩` ordered (l, then m_l, then m_s) — see
`atomic_hamiltonian.build_basis`.

---

## 1. Physical Constants & Units (`utils.py`, `atomic_hamiltonian.py`)

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
- **Code**: [`reduced_mass_rydberg_ev`](starkzee/utils.py#L98) — isotope nuclear
  masses (H/D/T/⁴He) from CODATA; sets the absolute line center.

### 1.2. Unit conversions
$$E = \frac{hc}{\lambda},\quad E = hf,\quad E = hc\,\tilde\nu,\quad hc \approx 1239.84\ \text{eV·nm}$$
- **Code**: [`energy_ev_to_wavelength_nm`](starkzee/utils.py#L159),
  `*_to_frequency_thz`, `*_to_wavenumber_cm` (hc derived from ħ, c — no hardcoding).

### 1.3. Vacuum → air wavelength (Edlén 1966)
$$(n-1)\times10^8 = 8342.13 + \frac{2406030}{130-\sigma^2} + \frac{15997}{38.9-\sigma^2},\quad \sigma = \frac{1000}{\lambda_\text{vac}[\text{nm}]}$$
- **Code**: [`vacuum_to_air_wavelength_nm`](starkzee/utils.py#L187).

---

## 2. Atomic / Magnetic Hamiltonian (`atomic_hamiltonian.py`)

The field-free + magnetic Hamiltonian for one shell n, in the `|n,l,m_l,m_s⟩` basis:
$$H_A = H_0 + V_\text{SO} + H_\text{MV+D} + H_Z^{(1)} + H_Z^{(2)}$$
- **Code**: [`build_hamiltonian`](starkzee/atomic_hamiltonian.py#L264).

### 2.1. Hydrogenic radial wavefunction
$$R_{nl}(r) = N_{nl}\, e^{-Zr/n}\,(2Zr/n)^l\, L_{n-l-1}^{2l+1}(2Zr/n),\quad
N_{nl} = \sqrt{\left(\tfrac{2Z}{n}\right)^3 \frac{(n-l-1)!}{2n\,(n+l)!}}$$
- **Code**: [`radial_wavefunction`](starkzee/atomic_hamiltonian.py#L54) (r in a₀).

### 2.2. Unperturbed energy (diagonal, degenerate over the shell)
$$E_n = -\frac{Z^2 R_\text{atom}(Z,A)}{n^2}$$
- **Code**: [line 342](starkzee/atomic_hamiltonian.py#L342). Uses the reduced-mass
  Rydberg → absolute energies match NIST.

### 2.3. Spin-orbit coupling
$$V_\text{SO} = \xi_{nl}\,\vec L\cdot\vec S,\qquad
\xi_{nl} = \frac{Z^4 \alpha^2 R_\infty}{n^3\,l\,(l+1)(l+\tfrac12)}$$
with $\vec L\cdot\vec S = L_z S_z + \tfrac12(L_+S_- + L_-S_+)$.
- **Code**: [lines 348–366](starkzee/atomic_hamiltonian.py#L348). Off-diagonal in
  (m_l, m_s); the ladder terms couple $|m_l{+}1, m_s{-}1\rangle \leftrightarrow |m_l, m_s\rangle$.

### 2.4. Mass-velocity + Darwin (completes Dirac fine structure)
$$\Delta E_{l=0} = -A_\text{fs}\,(n - \tfrac34),\qquad
\Delta E_{l>0} = -A_\text{fs}\!\left(\frac{n}{l+\tfrac12} - \tfrac34\right),\qquad
A_\text{fs} = \frac{Z^4\alpha^2 R_\infty}{n^4}$$
- **Code**: [lines 376–383](starkzee/atomic_hamiltonian.py#L376). Together with
  V_SO restores the Dirac degeneracy 2s₁/₂ = 2p₁/₂. Toggle: `fine_structure`.

### 2.5. Linear (paramagnetic) Zeeman
$$H_Z^{(1)} = \mu_B B\,(m_l + g_s m_s),\qquad g_s = |g_e|_\text{CODATA} \approx 2.00231930436$$
- **Code**: [lines 386–388](starkzee/atomic_hamiltonian.py#L386).

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
- **Code**: [lines 391–428](starkzee/atomic_hamiltonian.py#L391); radial off-diagonal
  [`radial_r2_element`](starkzee/atomic_hamiltonian.py#L96). Toggle: `quadratic_zeeman`.

### 2.7. Angular dipole matrix elements (spherical tensor T_q^(1) of r̂)
$$q=0:\ \langle l,m|\cos\theta|l{+}1,m\rangle = \sqrt{\frac{(l{+}1)^2-m^2}{(2l{+}1)(2l{+}3)}}$$
$$q=\pm1:\ \langle\,\cdot\,|T_{\pm1}|\,\cdot\,\rangle = \mp\sqrt{\frac{(l\mp m)(l\mp m-1)}{2(2l-1)(2l+1)}}\ \text{(branch-dependent)}$$
- **Code**: [`angular_dipole_element`](starkzee/atomic_hamiltonian.py#L176)
  (Condon-Shortley phases; nonzero only for |Δl|=1, Δm=q).

### 2.8. Radial dipole element
$$\langle n_1 l_1|r|n_2 l_2\rangle = \int_0^\infty R_{n_1 l_1}(r)\,r\,R_{n_2 l_2}(r)\,r^2\,dr$$
- **Code**: [`radial_dipole`](starkzee/atomic_hamiltonian.py#L135) (numerical, |Δl|=1).

### 2.9. Diagonalization & dipole rotation
$$H_A\,|\psi_k\rangle = E_k\,|\psi_k\rangle,\qquad
d_q[i,j] = \langle\psi_l^j|\,r_q\,|\psi_u^i\rangle = \sum_{k_l,k_u}
U_l^{*}[k_l,j]\,U_u[k_u,i]\,(-R\,\text{ang}_q)$$
- **Code**: [`diagonalize_hamiltonian`](starkzee/atomic_hamiltonian.py#L432),
  [`dipole_matrix_elements`](starkzee/atomic_hamiltonian.py#L466),
  [`_uncoupled_dipole_matrices`](starkzee/atomic_hamiltonian.py#L542) (units a₀).

### 2.10. Line strength, oscillator strength, Einstein A
$$S_{ul} = \sum_{q,i,j}\big|\langle l_j|r_q|u_i\rangle\big|^2\ [a_0^2]$$
$$gf = \frac{2}{3}\frac{\Delta E}{E_h}\,S_{ul},\qquad
A_{ul} = \frac{4\alpha^3}{3}\left(\frac{\Delta E}{E_h}\right)^3 \frac{S_{ul}}{2n_u^2\,\tau_\text{au}},\quad E_h = 2R_\infty$$
- **Code**: [`line_strength`](starkzee/atomic_hamiltonian.py#L592),
  [`oscillator_strength`](starkzee/atomic_hamiltonian.py#L629),
  [`einstein_a`](starkzee/atomic_hamiltonian.py#L654).

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
- **Code**: [`microfield_quadrature`](starkzee/microfield.py#L200) (see §4).

### 3.4. Field-angle quadrature (μ = cos θ)
Gauss-Legendre mapped to **μ ∈ [0, 1]** with weights summing to 1:
$$\langle\cdot\rangle_\text{angle} = \int_0^1 (\cdot)\,d\mu = \tfrac12\!\int_{-1}^{1}(\cdot)\,d\mu$$
(the half-range is exact: the integrand is symmetric under F_z → −F_z, so the ½
solid-angle factor and the 2× hemisphere factor cancel — verified numerically).
For each (β, μ): $F_z = \beta F_0\,\mu$, $F_x = \beta F_0\sqrt{1-\mu^2}$.
- **Code**: [lines 281–283, 343–344](starkzee/static_profile.py#L281).

### 3.5. Per-quadrature transition intensities
$$\Delta E_{ji} = E_u^i - E_l^j,\qquad
I_q[j,i] = \big|\,(U_l^\dagger D_q U_u)[j,i]\,\big|^2,\quad q\in\{0,+1,-1\}$$
- **Code**: [lines 347–356](starkzee/static_profile.py#L347).

### 3.6. Profile accumulation (Lorentzian or Voigt)
Each component is the microfield-weighted sum over transitions:
$$I_q(E) = \sum_{\beta,\mu} w_{\beta\mu}\sum_k I_q^k\,\mathcal{L}\!\big(E - \Delta E_k;\,w\big)$$
- Bare Lorentzian (no Doppler): $\mathcal{L}(x;w) = \dfrac{w/\pi}{x^2 + w^2}$.
- With Doppler ($T_i$ set): a Gaussian is accumulated in-loop and the Lorentzian
  applied by FFT afterwards (adaptive: which factor is in-loop depends on σ_D vs Δx).
- **Code**: [lines 362–396](starkzee/static_profile.py#L362). The half-width is
  $w = W_e(\Delta\omega) + w_\text{natural}$ (§5), with
  $w_\text{natural} = \hbar(\Gamma_u + \Gamma_l)/2$ from summed Einstein A.

### 3.7. Pseudo-Voigt (fast kernel option, Thompson 1987)
$$V \approx \eta\,\mathcal{L}(x;\Gamma) + (1-\eta)\,G(x;\sigma),\quad
\eta = 1.36603\tfrac{f_L}{f} - 0.47719\big(\tfrac{f_L}{f}\big)^2 + 0.11116\big(\tfrac{f_L}{f}\big)^3$$
with the combined FWHM f from the Thompson 5-term polynomial.
- **Code**: [`_pseudo_voigt`](starkzee/static_profile.py#L13).

### 3.8. Observable polarization combinations
$$I(\theta) = I_\pi\sin^2\theta + \tfrac12(I_{\sigma+}+I_{\sigma-})(1+\cos^2\theta)$$
Transverse (90°): $I_\pi + \tfrac12(I_{\sigma+}+I_{\sigma-})$;
Parallel (0°): $I_{\sigma+}+I_{\sigma-}$;
Angle-averaged: $\tfrac23 I_\pi + \tfrac13(I_{\sigma+}+I_{\sigma-})$.
- **Code**: `line_profile.profile_at_angle`.

### 3.9. Discrete stick spectrum
Enumerate every (i, j, q) with $|d_q(i\to j)|^2 > $ threshold at a single (F_z, F_x).
- **Code**: [`discrete_transitions`](starkzee/static_profile.py#L401).

---

## 4. Microfield Distribution (`microfield.py`)

### 4.1. Holtsmark normal field & mean spacing
$$r_e = \left(\frac{3}{4\pi N_e}\right)^{1/3},\qquad
F_0 = \frac{e}{4\pi\varepsilon_0\, r_e^2}$$
- **Code**: [`calculate_normal_field`](starkzee/microfield.py#L9).

### 4.2. Debye length (classical)
$$\lambda_D = \sqrt{\frac{\varepsilon_0 T_e}{N_e\, e}}\quad(T_e\ \text{in eV})$$
Multi-species: $\lambda_D = \sqrt{\varepsilon_0 T_e / [N_e e (1 + \sum_i X_i Z_i^2)]}$.
- **Code**: [`calculate_debye_length`](starkzee/microfield.py#L36),
  [`calculate_multispecies_debye_length`](starkzee/microfield.py#L60).

### 4.3. Holtsmark distribution (unscreened)
$$W(\beta) = \frac{2\beta}{\pi}\int_0^\infty y\,\sin(\beta y)\,e^{-y^{3/2}}\,dy$$
- **Code**: [`holtsmark_distribution`](starkzee/microfield.py#L105).

### 4.4. Hooper screened distribution
$$W(\beta,a) = \frac{2\beta}{\pi}\int_0^\infty y\,\sin(\beta y)\,e^{-y^{3/2} S(y,a)}\,dy,\quad
S(y,a) = \left(1 + \frac{1.5\,a^2}{y^2}\right)^{-3/4},\quad a = \frac{r_e}{\lambda_D}$$
$a \to 0$ recovers Holtsmark.
- **Code**: [`hooper_distribution`](starkzee/microfield.py#L145).
- **NOTE**: depends only on the screening parameter `a`; there is **no ion-coupling
  (Γ) correlation correction** (unlike zest's Potekhin fits).

### 4.5. Quadrature normalization
Uniform β grid; W normalized so $\sum_i W_i\,\Delta\beta = 1$; returns
$\text{fields} = \beta F_0$, $\text{weights} = W\,\Delta\beta$.
- **Code**: [`_microfield_quadrature_impl`](starkzee/microfield.py#L268).

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
y = \left(\frac{n^2}{2Z}\right)^2\frac{\Delta\omega^2 + \omega_c^2}{E_H\,T_e},\quad
E_H = 2R_\infty$$
- **Code**: [`gbk_model`](starkzee/broadening.py#L129). Δω and ω_c in eV.

### 5.4. Total electron half-width (HWHM)
$$W_e(\Delta\omega) = W_0\,\langle r^2\rangle_n\,\big[C_n + G(\Delta\omega,\omega_c)\big]$$
$$\langle r^2\rangle_n = \frac{1}{n^2}\sum_{l=0}^{n-1}(2l+1)\,\frac{n^2}{2Z^2}\big[5n^2+1-3l(l+1)\big]$$
$$\omega_c = \max(\omega_p,\,\omega_L,\,\omega_e),\qquad
C_n = \begin{cases}1.50 & n\le2\\ 0.75 & n=3,4\\ 0.40 & n\ge5\end{cases}$$
- **Code**: [`electron_impact_width`](starkzee/broadening.py#L182). C_n from Ferri
  et al. (2021) Table 1; the magnetic cutoff ω_L suppresses the width at high B.
- **Frequency-dependent** by default: G is evaluated at each transition's detuning
  $\Delta E_i - E_0$ (`static_profile.py:367`).

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
half-width $\gamma_k = W_e(0) + 10^{-4}$ eV (single on-resonance value for all SDTs).
- **Code**: [lines 240–247](starkzee/ffm.py#L240).
- Limits: $\nu_i \to 0$ → quasi-static (static profile); $\nu_i \to \infty$ →
  motional-narrowing (single Lorentzian).

### 6.4. FFM — full Liouville matrix inversion (optional)
$$A_{mj} = \delta_{mj}\big[(\omega - \omega_j) - i(\gamma_k + \nu_i)\big] + i\nu_i\,p_j,\qquad
I(\omega) = \frac{1}{\pi}\,\mathrm{Re}\Big(i\,\textstyle\sum_k d_k\,[A^{-1}b]_k\Big)$$
- **Code**: [lines 216–236](starkzee/ffm.py#L216) (`numerical_inversion=True`;
  falls back to Sherman-Morrison if singular).

> Doppler is **not** folded into the FFM kernel (unlike zest); apply it afterward
> via `convolutions.py` if needed.

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
- **Microfield**: Holtsmark / Hooper-screened, classical Debye, neutral-point;
  no ion-coupling (Γ) correlation correction.
- **Electron broadening**: GBK semi-classical, frequency-dependent, magnetic (Larmor)
  cutoff, single shell-averaged ⟨r²⟩_n width for all transitions.
- **Ion dynamics**: quasi-static + optional FFM (Sherman-Morrison or full inversion).
- **Doppler / instrument**: post-processing FFT convolution (not inside the FFM).
