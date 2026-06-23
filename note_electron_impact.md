# GBK Electron Impact Broadening: StarkZee vs ZEST / PPPB

This note summarizes the key physical parameters, formulas, and differences in the electron-impact broadening implementations in StarkZee, ZEST, and PPPB.

## 1. Core Formalism & G-Function

All models evaluate the weak-collision term using the Griem-Baranger-Kolb (GBK) integral:
$$G(\Delta\omega) = \frac{1}{2} E_1(y)$$
where the Rydberg constant $\text{Ryd} = 13.6057\text{ eV}$ (written as $E_H$ in GBK/PPPB and $Ryd$ in ZEST) is the denominator scale for all models.

### Standard/PPPB Model (`gbk_model`)
Uses a fixed cutoff parameter:
$$y = \left(\frac{n^2}{2Z}\right)^2 \frac{\Delta\omega^2 + \omega_c^2}{\text{Ryd} \cdot T_e}$$
where the frequency cutoff is $\omega_c = \max(\omega_p, \omega_e, \omega_L, \omega_{\alpha\alpha'})$.

### ZEST Model (`gbk_zest_model`)
Uses a temperature-dependent physical cutoff parameter:
$$y = \frac{\Delta\omega^2 + \omega_p^2}{2 x^2 \omega_p^2} \quad \text{with } x = \kappa_m \lambda_D$$
where $\omega_p$ is the plasma frequency, and $\kappa_m$ is the Debye-screening microfield parameter (interpolating between thermal and geometric branches). At $T_e > 13.6\text{ eV}$, this converges exactly to the ZEST paper's Eq. 14.

---

## 2. Strong-Collision Constant $C_n$
For impact parameters below the weak-collision cutoff $\rho_m = 1/\kappa_m$, a strong-collision constant is added at the broadening operator level:
$$W_e \propto C_n + G(\Delta\omega)$$
The constants are identical across all papers:
* $C_2 = 1.5$
* $C_3 = 1.0$
* $C_4 = 0.75$
* $C_5 = 0.5$
* $C_{n > 5} = 0.4$

---

## 3. Mean-Square Radius $\langle r^2 \rangle$

The electron impact width is directly proportional to the radiator mean-square radius. StarkZee supports two treatments:

### Full Shell-Average (PPPB/Ferri Default)
Includes all inter-shell ($n \neq n'$) and intra-shell dipole coupling channels:
$$\langle r^2 \rangle_n = \frac{1}{n^2} \sum_{l=0}^{n-1} (2l+1) \frac{n^2}{2Z^2} \left[5n^2 + 1 - 3l(l+1)\right] a_0^2$$
*(H values: $n=2$: $33\text{ }a_0^2$, $n=3$: $153\text{ }a_0^2$)*

### Intra-Shell Average (ZEST Default, `electron_model='zest'`)
Restricts interaction channels to $\Delta n = 0$ (Layzer complex / no-quenching approximation), yielding:
$$\langle r^2_{\text{intra}} \rangle_n = \frac{9n^2}{8Z^2}(n^2 - 1) a_0^2$$
*(H values: $n=2$: $13.5\text{ }a_0^2$, $n=3$: $81\text{ }a_0^2$)*

| $n$ | $\langle r^2 \rangle_{\text{full}}$ | $\langle r^2_{\text{intra}} \rangle$ | Ratio ($\text{intra}/\text{full}$) |
|---|-----------|------------|-------|
| 2 | $33\text{ }a_0^2$ | $13.5\text{ }a_0^2$ | 0.41 |
| 3 | $153\text{ }a_0^2$ | $81\text{ }a_0^2$ | 0.53 |
| 4 | $468\text{ }a_0^2$ | $270\text{ }a_0^2$ | 0.58 |

---

## 4. Anomaly: The $2\pi$ Factor in PPPB
In PPPB (Ferri et al. 2022), the thermal cutoff frequency $\omega_e = 1/\tau_e$ is used inconsistently:
1. **$\max()$ comparison:** Uses $\omega_e \approx 46\text{ meV}$ (without $2\pi$) to determine when the Larmor frequency $\omega_L$ takes over.
2. **$y$ argument:** Uses $2\pi\omega_e \approx 290\text{ meV}$ inside the $E_1(y)$ function to fit their plotted G-function curves.

StarkZee avoids this code-fitting artifact by consistently using the physical rate $\omega_e = 1/\tau_e = 46\text{ meV}$ in both places.
