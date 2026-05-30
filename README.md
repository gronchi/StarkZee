# StarkZee

**Coupled Stark-Zeeman plasma line-shape model for hydrogen-like radiators.**

Computes emission line profiles of H-like ions in a magnetised plasma, coupling quasi-static ion microfield broadening (Holtsmark / Hooper distributions), electron impact broadening (GBK model), and optionally ion dynamics via the Frequency Fluctuation Model (FFM).

Implements the model of [Ferri, Peyrusse & Calisti, *Matter and Radiation at Extremes* **7**, 015901 (2022)](https://doi.org/10.1063/5.0073040).

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-≥3.9-blue)

---

## Features

- Full Stark-Zeeman Hamiltonian diagonalisation at each microfield quadrature point — spin-orbit, linear Zeeman, quadratic (diamagnetic) Zeeman, and Stark perturbation
- Hooper screened and Holtsmark unscreened microfield distributions
- GBK electron impact broadening with frequency-dependent width and Larmor-frequency cutoff
- Frequency Fluctuation Model (FFM) for dynamic ion broadening
- FFT-based post-processing: thermal Doppler and instrumental broadening
- Observation-angle decomposition: π, σ+, σ− polarization components
- Spectral grids in any unit system: wavelength [nm], energy [eV], frequency [THz], wavenumber [cm⁻¹]
- Discrete stick spectrum at arbitrary field configurations

---

## Installation

```bash
git clone https://github.com/g-ronchi/starkzee.git
cd starkzee
pip install -e .            # core — numpy, scipy only
pip install -e ".[plot]"    # add matplotlib
pip install -e ".[dev]"     # add pytest + matplotlib
```

---

## Quick start

```python
import numpy as np
import matplotlib.pyplot as plt
from starkzee.line_profile import LineProfile

# Hydrogen Balmer-α (n=3→2) at typical tokamak edge conditions
lp = LineProfile(n_u=3, n_l=2, Z=1, B=5.0, Ne_m3=1e20, Te_ev=5.0)

# Provide the spectral grid in any unit system
wl_grid = np.linspace(lp.E0_wavelength_nm - 1.0,
                      lp.E0_wavelength_nm + 1.0, 1000)
lp.compute_profile(wl_grid, grid_type='wavelength_nm')

# Polarization components and observation-angle combinations
lp.profile_transverse   # π + ½(σ⁺ + σ⁻)  — perpendicular to B
lp.profile_parallel     # σ⁺ + σ⁻          — along B
lp.profile_at_angle(45) # Stokes formula at arbitrary θ

# Detuning grids are always available in all unit systems
lp.detuning_nm          # λ − λ₀  [nm]
lp.detuning_ev          # E − E₀  [eV]
lp.detuning_thz         # f − f₀  [THz]
lp.detuning_cm          # ν̃ − ν̃₀ [cm⁻¹]

# Quick plot
plt.plot(lp.detuning_nm, lp.profile_transverse)
plt.xlabel(r"$\lambda - \lambda_0$  (nm)")
plt.show()
```

### Discrete stick spectrum

```python
lp.compute_discrete(Fz=0.0, Fx=0.0)
for dλ, q, s in zip(lp.discrete.detuning_nm, lp.discrete.q, lp.discrete.strength):
    print(f"  Δλ = {dλ:+.4f} nm   q = {q:+d}   |d|² = {s:.4f} a₀²")
```

### With Doppler and instrumental broadening

```python
from starkzee.convolutions import apply_doppler_broadening, apply_instrument_broadening

profile = apply_doppler_broadening(
    lp.wavelengths_nm, lp.profile_transverse,
    Ti_ev=5.0, A_emitter=1, E0_ev=lp.E0,   # A_emitter=2 for deuterium
)
profile = apply_instrument_broadening(lp.wavelengths_nm, profile, fwhm_nm=0.05)
```

### FFM (ion dynamics)

```python
from starkzee.ffm import calculate_ffm_profile

pi, sp, sm = calculate_ffm_profile(
    n_u=3, n_l=2, Z=1, B=5.0, Ne_m3=1e20, Te_ev=5.0, Ti_ev=5.0,
    energies_ev=lp.energies_ev,
)
```

### CLI

```bash
starkzee -Z 1 -B 5 --Ne 1e20 --Te 5 -o profile.txt -p profile.png
```

---

## Package overview

| Module | Role |
|--------|------|
| `line_profile.py` | `LineProfile` — main high-level API |
| `static_profile.py` | Static Stark-Zeeman solver; Gauss-Legendre quadrature over the microfield |
| `ffm.py` | Frequency Fluctuation Model for dynamic ion broadening |
| `atomic_hamiltonian.py` | Hamiltonian construction: spin-orbit, Zeeman, radial matrix elements |
| `microfield.py` | Hooper and Holtsmark microfield distributions |
| `broadening.py` | GBK electron impact width with Larmor-frequency cutoff |
| `convolutions.py` | FFT Doppler and instrumental broadening (post-processing) |
| `utils.py` | Physical constants (CODATA via scipy) and unit conversions |
| `cli.py` | Command-line interface |

---

## Physics summary

The radiator Hamiltonian in the uncoupled $|n,l,m_l,m_s\rangle$ basis:

$$H_A = H_0 + V_\text{SO} + H_Z^{(1)} + H_Z^{(2)}$$

| Term | Expression |
|------|-----------|
| Unperturbed | $H_0 = -Z^2\,\text{Ry}/n^2$ |
| Spin-orbit | $V_\text{SO} = \xi\,\vec{L}\cdot\vec{S}$ |
| Linear Zeeman | $H_Z^{(1)} = \mu_B B\,(m_l + g_s m_s)$ |
| Quadratic Zeeman | $H_Z^{(2)} = \dfrac{e^2B^2}{8m_e}r^2\sin^2\theta$ |

The Stark perturbation $V_E = -e(zF_z + xF_x)$ is added and the combined Hamiltonian diagonalised at each microfield quadrature point.  The static profile is the microfield-weighted sum of Lorentzian-broadened transition intensities.

The FFM treats the ion microfield as a Markovian jump process:

$$I(\omega) = \frac{r^2}{\pi}\,\mathrm{Re}\,\frac{S(\omega)}{1 - \nu_i S(\omega)}, \qquad S(\omega) = \sum_k \frac{p_k}{\nu_i + \gamma_k + i(\omega - \omega_k)}$$

Full derivations are in [`docs/source/theory.rst`](docs/source/theory.rst) and [`docs/manual.tex`](docs/manual.tex).

---

## Tests

```bash
pytest tests/ -v
```

332 tests covering constants (CODATA), Hamiltonian construction, Zeeman splitting, Stark matrix elements, microfield distributions, GBK broadening, profile shapes, fine structure, quadratic Zeeman wings, satellite features, oscillator strengths, and FFM limiting cases.

---

## Scope and limitations

- Radiators are treated as **hydrogen-like** (one outer electron, nuclear charge Z). Multi-electron ions such as C IV require quantum-defect corrections — see `TODO.md`.
- The static solver uses **exact analytical** hydrogenic radial matrix elements within the $n$-shell; coupling to adjacent shells (quadratic Stark) is neglected.
- **Doppler and instrumental broadening** are not automatic — apply them explicitly via `convolutions.py` after computing the Stark-Zeeman profile.

---

## License

MIT — see [LICENSE](LICENSE).
