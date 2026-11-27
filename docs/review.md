# Manual consistency review — 19 October 2026

Reviewed `manual.tex` against the current implementation and the locally
available primary references. Corrections are confined to this manual and its
PDF; no runtime implementation, data tables, or original `docs/` files were
changed. The notes and caveats remain beside the relevant derivations.

| Issue found | Correction |
| --- | --- |
| Temperatures alternated between kelvin and eV without distinct symbols. | Defined numerical eV temperatures and SI thermal energies `Theta_e`, `Theta_i`; used the latter consistently in SI formulas. |
| Angular-frequency and energy versions of collision operators and widths were mixed. | Defined energy-scaled Liouvillian and damping operators, the frequency-to-energy density Jacobian, and explicit joule-to-eV factors for cutoff and natural widths. |
| The analytic `LineProfile.E0` and the physical reference used in empirical broadening shared an ambiguous symbol; two unrelated polar angles also shared one symbol. | Distinguished the API's analytic comparison value from the physical `E0` used in profile formulas, and used a separate electron polar angle in the diamagnetic term. |
| The quadratic-Zeeman matrix formula retained the SI charge factor while labeling the result as eV, and could count the Bohr radius twice. | Printed the eV coefficient and dimensionless radial integral separately. |
| The broadening prefactor was labeled as energy per squared radius, while its SI expression and implementation return energy. | Labeled `W0` in eV and divided physical squared radii by `a0^2` in width formulas. |
| The physical positive Stark interaction and negative numerical matrix were not reconciled; the signed radiative dipole formula omitted its electron-charge sign. | Distinguished the positive intra-shell radial magnitude from the negative signed spatial integral, included the length conversion in the Stark matrix, and matched the numerical dipole definition to `radiator.py`. |
| The PPP collision function shared the name of the dimensionless GBK factor even though it includes the dimensional prefactor. | Used a separate dimensional function and stated its impact-limit calibration. Also distinguished energy poles from angular frequencies. |
| The shell-width sum implicitly included an undefined `C_1`. | Restricted the summed electron contributions to shells with `n >= 2`, matching the profile solvers. |
| The cutoff description conflated two thermal-speed conventions and claimed that hydrogenic emitter splittings vanish. | Specified the configuration speed used by the code, its factor of `sqrt(2)` difference from the most-probable speed, and the implemented zero placeholder for the emitter-splitting cutoff. |
| The Dufty description claimed a vanishing zero-detuning lower integration limit. | Documented its finite numerical floor, empty-interval behavior, and the small omitted contribution at line center. |
| The FFM equations mixed rates in inverse seconds, eV widths, and photon energies. | Used a separate fluctuation energy `eta_i` and energy-space propagator throughout; kept the physical crossing rate explicit. |
| The default FFM width was described as common to every SDT. | Distinguished the common electron contribution from state-dependent natural damping in both width modes. Qualified the static limit to require matched damping. |
| Binning was compared with a dimensionally incompatible fluctuation rate and presented as having a universal necessary accuracy criterion. | Compared energy scales, described intensity-weighted natural-width binning, and required an unbinned profile comparison rather than a universal guarantee. |
| The printed Doppler kernel was unnormalized and centered on the physical wavelength, despite the helper using a zero-lag kernel. | Printed a normalized lag kernel, explained discrete normalization, and documented the physical-center argument and its grid-mean fallback. |
| The broadening pipeline treated energy-space and wavelength-space Gaussian convolutions as identical. | Included the spectral-density Jacobian and distinguished the solvers' energy convolution from the narrow-line wavelength approximation. |
| Rosato's magnetic scaling was described in the wrong coordinate direction and its temperature coordinate was called electron temperature. | Described the scaled query, the density Jacobian, special low-field branches, and effective table temperature. |
| Some interpretation claims were stronger than their evidence. | Qualified the separable Zeeman scale check and Doppler-versus-Stark comparison; distinguished tabulated levels from necessarily measured levels; described the empirical-basis transformation accurately. |
| Comparison figures were described as current working-tree outputs without a regeneration in this edition. | Identified them as bundled figure copies and retained the model-comparison qualification. |

Validation:

- `python docs/build_manual.py` rebuilt the 31-page PDF without warnings or
  unresolved references. Rendered pages and text bounds were checked; no
  significant right-margin overflows remain.
- **41 existing tests passed:** `test_08_broadening.py`,
  `test_29_ffm_frequency_dependent_width.py`, and
  `test_30_state_resolved_natural_width.py`.
- Independent numerical checks reproduced the signed Stark matrix, electron
  prefactor, configuration speed, both equal-mass fluctuation conventions,
  zero-detuning Dufty integral, and the quoted H-alpha Doppler width.
- At `Ne=1e23 m^-3`, `Te=Ti=5 eV`, H perturbers:
  `W0 = 3.14393144106e-6 eV`, fluctuation energy
  `eta_i = 0.00152423387006 eV`. The H-alpha Doppler 1/e half-width is
  `0.194981 meV`. The omitted low-wave-number contribution to the zero-detuning
  Dufty factor is `2.49966670416e-9`.
- All 44 labels are unique; all 23 distinct cross-reference targets exist;
  all 22 citation keys occur in the canonical registry.

The review does not close the scientific limitations already stated in the
manual: the PPPB full-versus-projected closure and cutoff-convention questions,
the experimental interference/FFM closure, inter-shell production physics,
and independent matched-profile validation. No reference spectra were newly
generated. Successful compilation and these checks establish consistency for
the inspected formulas and paths, rather than external accuracy across all
plasma regimes.

Primary references consulted include the local copy of
[the PPP manual](https://hal.science/hal-04501367v1),
[Gilleron and Pain's ZEST paper](https://doi.org/10.3390/atoms6010011), and
[Potekhin, Chabrier and Gilles](https://www.ioffe.ru/astro/Stars/Paper/pcg02.pdf).
The runtime comparisons use `starkzee/radiator.py`, `static_profile.py`,
`broadening.py`, `collision.py`, `ffm.py`, `microfield.py`, `convolutions.py`,
and `models/rosato_impl.py` as implementation evidence.

## Second pass

The second pass corrected additional reference-model and notation issues:

- Stehle's normal-Zeeman shift is now written in SI angular-frequency units;
  the previous expression mixed the numerical eV/T Bohr magneton with SI
  `hbar`. The printed triplet uses the wrapper's normalized weights of
  `sin(alpha)^2/2` and `(1+cos(alpha)^2)/4`.
- The conversion between physical-field and reduced-field probability
  densities is explicit, including the `F0` Jacobian.
- The Stark/Zeeman scale comparison includes the hydrogenic `1/Z` factor and
  uses SI energies consistently on both sides.

All 17 existing Stehle interpolation tests passed. An independent calculation
also reproduced the triplet formula at viewing angles of 0, 37 and 90 degrees.
The earlier scientific validation limitations remain open.

The current 36-page PDF rebuilds without LaTeX warnings or unresolved
references, and rendered pages have no significant right-margin overflows.
The strict Sphinx HTML build (`-n -W --keep-going`) also succeeds.

## Atomic API and Stark-map documentation

The manual now includes `../docs/atomic_api.tex`, shared with the original
manual. It covers the public `starkzee.atomic` functions, radial accessors
on `LineProfile`, resolved Einstein coefficients and partial upper-state
decay rates, and the separate multi-shell `StarkMap` diagnostic. The source
baseline on the cover is the locally integrated `c4c4efa` commit of
9 October 2026; the working documentation remains uncommitted.

Three figures are generated by
`python examples/atomic_api_and_stark_maps.py`: field-free radial densities
and Balmer data, fixed-field H-alpha strengths and rates, and hydrogen
Stark maps at 0 and 5 T. The script checks radial normalization, strength
conservation, rate sums, and normalized target-state overlaps. Captions
identify the fields, basis, isotope convention, and reference energies.
Code excerpts are included directly from the runnable script in both the
PDF chapter and the HTML guide.

Validation for this update:

- All nine tests in `test_40_public_atomic_api.py` and `test_41_stark_map.py`
  pass.
- The strict Sphinx HTML build passes, renders all nine public atomic
  functions and the Stark-map result API, and includes all three figures.
- Both PDFs rebuild with resolved references. The `docs` build is warning
  free; its rendered API pages and right margins were checked. The original
  35-page manual retains layout and PDF-bookmark warnings in its older
  theory sections, outside the new shared chapter.
- The PDF builder searches fresh temporary auxiliaries before source-folder
  files, preventing stale `.aux` and `.bbl` files from shadowing the new build.

The rate documentation distinguishes shell averages, partial rates, and
E1-only total rates. The map documentation distinguishes target overlaps
from strengths or populations, and instantaneous overlap selection from
adiabatic state tracking. These examples do not establish convergence of
the finite multi-shell basis or independent physical validation.
