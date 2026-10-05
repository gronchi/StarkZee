# StarkZee

**Coupled Stark-Zeeman plasma line-shape model for hydrogen-like radiators.**

**StarkZee** implements the Standard Lineshape Theory for emission lines of hydrogen-like ions in a magnetized plasma.
Ions are treated in the quasi-static approximation: the ion microfield at the radiator site is assumed stationary on the timescale of the emitted photon, and the spectral profile is obtained by averaging Stark-Zeeman Hamiltonians over the ion microfield distribution.
Electron broadening is represented by weak binary collisions within the Griem–Baranger–Kolb (GBK) binary-collision relaxation model, which accounts for the suppression of broadening at large frequency detunings through a semi-classical exponential-integral factor and a magnetic-field-dependent lower cutoff.

The static magnetic field enters the radiator Hamiltonian directly — within the electric-dipole approximation — producing coupled Stark-Zeeman energy levels and polarized π and σ± emission components.
Ion dynamics (the finite velocity of the perturbing ions) are optionally included via the Frequency Fluctuation Model (FFM), which treats the microfield as a Markovian jump process between quasi-static configurations.
The default screened microfield uses the Potekhin fits; unscreened mode uses Holtsmark. Supply ion temperature explicitly: if it is missing for the default screened model, the code warns and assumes Ti=Te. The historical Hooper-like ansatz is available only by explicit selection and is not a validated probability distribution. See the audit repair status and numerical contracts before quantitative use.

Static, FFM, and discrete-transition calculations default to
`use_empirical_data=True`, using bundled NIST levels for H, D, and T. Set
`use_empirical_data=False` for analytical energies, shells beyond the table
coverage (H: n ≤ 8, D: n ≤ 6, T: n ≤ 3 with fine structure), or hydrogen-like
ions with `Z > 1`. `LineProfile` selects the isotope automatically; direct
solver calls require the matching `atom`. Low-level Hamiltonian functions
retain their analytical default and eV output convention.

Generate a numerical convergence report before quantitative use:

```powershell
python scripts/profile_convergence.py --solver both --output convergence.json
```

The report varies quadrature, microfield cutoff, grid/window, and FFM binning
and supplies metrics without imposing a universal acceptance threshold.

Model based on [Ferri, Peyrusse & Calisti, *Matter and Radiation at Extremes* **7**, 015901 (2022)](https://doi.org/10.1063/5.0058552).

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-≥3.9-blue)

---

## Features

- Full Stark-Zeeman Hamiltonian diagonalization at each microfield quadrature point — spin-orbit, linear Zeeman, quadratic (diamagnetic) Zeeman, and Stark perturbation
- Potekhin screened (default), Holtsmark unscreened, and explicit legacy
  Hooper-like microfield distributions
- GBK electron impact broadening with frequency-dependent width and Larmor-frequency cutoff
- Opt-in PPP impact-limit collision matrix with upper/lower interference and
  complex generalized Stark-dressed-transition weights
- Frequency Fluctuation Model (FFM) for dynamic ion broadening
- Explicit emitter/background charge and mass separation, with selectable
  ZEST or PPP-manual ion-fluctuation rates
- Optional in-solver thermal Doppler broadening plus standalone FFT Doppler and instrumental helpers
- Observation-angle decomposition: π, σ+, σ− polarization components
- Spectral grids in any unit system: wavelength [nm], energy [eV], frequency [THz], wavenumber [cm⁻¹]
- Discrete stick spectrum at arbitrary field configurations

---

## Installation

```bash
git clone https://github.com/g-ronchi/starkzee.git
cd starkzee
pip install -e .

```

---

## Quick start

```python
import numpy as np
import matplotlib.pyplot as plt
from starkzee.line_profile import LineProfile

# Hydrogen Balmer-α (n=3→2) at typical tokamak edge conditions
lp = LineProfile(n_u=3, n_l=2, B=5.0, Ne_m3=1e20, Te_ev=5.0)

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

The ``lp`` above has no ``Ti_ev``, so its static profile is Doppler-free and
can be broadened explicitly. If ``Ti_ev`` was supplied to ``LineProfile``, the
static solver already included Doppler and this first convolution must be
skipped.

```python
from starkzee.convolutions import apply_doppler_broadening, apply_instrument_broadening

profile = apply_doppler_broadening(
    lp.wavelengths_nm, lp.profile_transverse,
    Ti_ev=5.0, species='H',
)
profile = apply_instrument_broadening(lp.wavelengths_nm, profile, fwhm_nm=0.05)
```

### FFM (ion dynamics)

```python
from starkzee.ffm import calculate_ffm_profile

pi, sp, sm = calculate_ffm_profile(
    n_u=3, n_l=2, Z=1, B=5.0, Ne_m3=1e20, Te_ev=5.0, Ti_ev=5.0,
    A_ion=1, energies_ev=lp.energies_ev,
)
```

Here `A_ion` is the emitter mass. Same-species plasma remains the default. For
a heavy hydrogen-like emitter in a proton background, pass
`A_perturber=1`, `emitter_charge=Z-1`, and optionally
`fluctuation_rate_model="ppp"` to use the PPP-manual reduced-mass rate instead
of the default ZEST rate. The current Potekhin fit distinguishes neutral from
charged emitters but does not depend on the magnitude of `emitter_charge`.

### Comparison with all models — D_γ

```python
import numpy as np
import matplotlib.pyplot as plt
from starkzee.line_profile import LineProfile
import starkzee.models as models

# D_γ (n=5→2) at low-density edge conditions
Ne_m3   = 1e19    # electron density   [m⁻³]
Te_ev   = 0.5     # electron temperature [eV]
Ti_ev   = 0.5     # ion temperature      [eV]
B       = 3.0     # magnetic field       [T]
n_u, n_l = 5, 2
half_width_nm = 1.5

# Ti_ev supplied → compute_profile applies Doppler broadening automatically
lp = LineProfile(n_u=n_u, n_l=n_l, B=B, Ne_m3=Ne_m3,
                 Te_ev=Te_ev, Ti_ev=Ti_ev, species='D', view_angle_deg=90.0)

wl_vac = np.linspace(lp.E0_wavelength_nm - half_width_nm,
                     lp.E0_wavelength_nm + half_width_nm, 1000)
lp.compute_profile(wl_vac, grid_type='wavelength_nm')
sz = lp.profile

# Comparison models share the same air-wavelength grid
wl = np.linspace(lp.E0_wavelength_air_nm - half_width_nm,
                 lp.E0_wavelength_air_nm + half_width_nm, 1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4),
                               sharex=False, sharey=False)
fig.suptitle(r'D$_\gamma$  —  '
             f'$N_e={Ne_m3:.0e}$ m$^{{-3}}$, '
             f'$T_i=T_e={Ti_ev}$ eV, $B={B}$ T')

for ax in (ax1, ax2):
    ax.plot(lp.wavelengths_air_nm, sz / sz.max(),
            'k--', lw=2, label='StarkZee', zorder=10)

comparison_models = [
    ('Voigt',          models.voigt),
    ('Stehle',         models.stehle),
    ('Stehle (param)', models.stehle_param),
    ('Lomanowski',     models.lomanowski),
    ('Rosato',         models.rosato),
]
for label, func in comparison_models:
    try:
        p = func(wl, n_u, n_l, B, Ne_m3, Te_ev, Ti_ev, species='D')
        ax1.plot(wl, p / p.max(), label=label, alpha=0.85)
        ax2.plot(wl, p / p.max(), label=label, alpha=0.85)
    except Exception as exc:
        print(f'{label}: {exc}')

ax1.set_xlabel('wavelength (nm)')
ax1.set_ylabel('normalized intensity')
ax1.legend(fontsize=9)
ax1.grid(ls=':', alpha=0.4)

ax2.semilogy()
ax2.set_xlabel('wavelength (nm)')
ax2.grid(ls=':', alpha=0.4)

plt.tight_layout()
plt.show()
```

![StarkZee vs. Voigt, Stehlé, Stehlé-parameterized, and Rosato for D-Balmer-α at Ne=1e20 m⁻³, Ti=Te=1 eV, B=3 T](docs/figures/model_comparison.png)

*Generated by [`examples/model_comparison.py`](examples/model_comparison.py), which also produces the [D-Balmer-γ comparison](docs/figures/model_comparison_dgamma.png). The figures contrast StarkZee's coupled treatment with the analytical and tabulated reference models; Lomanowski is disabled here. See [Built-in Reference Models](docs/manual.tex) for the full comparison table.*

For the opt-in non-Hermitian PPP impact-limit collision matrix, run
[`examples/model_comparison_non-hermit.py`](examples/model_comparison_non-hermit.py).
It uses `electron_interference=True` for D-α and D-γ. The common Appendix-B
coefficient is evaluated directly from `W0 [C_nu + G_nu(0)]`; it is not
calibrated from a scalar full-shell or intra-shell width. If the complex-mode
real residues are signed, the default experimental FFM closure uses
`p_k=abs(a_k)/sum(abs(a))` while retaining signed `a_k+i*c_k` in the radiative
numerator. The script reports the size of that correction. An optional
complex-SDT grouping experiment is available through `--group-tolerance-ev`;
it is accepted only when the grouped static profile meets the requested error
bound and every grouped real residue is nonnegative.

---

## Package overview

| Module | Role |
|--------|------|
| `line_profile.py` | `LineProfile` — main high-level API |
| `static_profile.py` | Static Stark-Zeeman solver; Gauss-Legendre quadrature over the microfield |
| `ffm.py` | Frequency Fluctuation Model for dynamic ion broadening |
| `radiator.py` | Quantum basis, wavefunctions, dipole transitions, and Hamiltonian construction |
| `microfield.py` | Hooper and Holtsmark microfield distributions |
| `broadening.py` | GBK electron impact width with Larmor-frequency cutoff |
| `collision.py` | Opt-in PPP non-Hermitian impact-limit collision operator and complex SDTs |
| `convolutions.py` | Standalone wavelength-space FFT Doppler and instrumental broadening helpers |
| `atomic_data.py` | NIST atomic energy-level database loader (`starkzee/data/atomic_levels.json`) |
| `utils.py` | Physical constants (CODATA via scipy) and unit conversions |

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

The Stark perturbation $V_E = -e(zF_z + xF_x)$ is added and the combined Hamiltonian diagonalized at each microfield quadrature point.  The static profile is the microfield-weighted sum of Lorentzian-broadened transition intensities.

The FFM treats the ion microfield as a Markovian jump process:

$$I(\omega) = \frac{r^2}{\pi}\,\mathrm{Re}\,\frac{S(\omega)}{1 - \nu_i S(\omega)}, \qquad S(\omega) = \sum_k \frac{p_k}{\nu_i + \gamma_k + i(\omega - \omega_k)}$$

Full derivations are in [`docs/manual.tex`](docs/manual.tex) and the [Sphinx documentation](docs/source/).

---

## Tests

```bash
pytest tests/ -v
```

599 tests covering constants (CODATA), Hamiltonian construction, Zeeman splitting,
Stark matrix elements, microfield distributions, GBK and full PPP collision
operators, profile shapes, fine structure, Ar XVII manual benchmarks,
quadratic-Zeeman wings, oscillator strengths, multi-shell prerequisites, and
FFM limiting cases and empirical defaults. The suite currently emits 90 model/grid warnings.

---

## Documentation

A comprehensive PDF manual detailing the physics and implementation is available in the `docs` folder. To compile the LaTeX manual `docs/manual.tex`:

- **Windows**: Run the batch file `docs/compile_manual.bat`. It will perform a double-pass compilation and automatically handle PDF reader file locks.
- **Linux / macOS**: run `pdflatex manual.tex` twice inside `docs/` (or use
  `latexmk -pdf manual.tex`). The `docs/Makefile` builds the Sphinx manual, not
  the standalone PDF.

---

## Scope and limitations

- Radiators are treated as **hydrogen-like** (one outer electron, nuclear charge Z). Multi-electron ions such as C IV require quantum-defect corrections — see `TODO.md`.
- The static solver uses **exact analytical** hydrogenic radial matrix elements within the $n$-shell; coupling to adjacent shells (quadratic Stark) is neglected.
- **Doppler broadening**: FFM applies it internally by default (`apply_doppler=True`); the static solver applies it internally when `Ti_ev` is supplied. Instrumental broadening is always explicit post-processing via `convolutions.py`.

---

## License

MIT — see [LICENSE](LICENSE).
