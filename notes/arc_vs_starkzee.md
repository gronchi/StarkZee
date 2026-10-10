# ARC and StarkZee: atomic-model overlap and useful extensions

Report date: 7 October 2026.

Scope: local ARC-Alkali-Rydberg-Calculator version `3.10.2`, commit
`4b4573e`, compared with StarkZee release commit `00715d4`. The review covered
atomic models, Stark maps, angular algebra, radiative rates, population
evolution, and relevant limitations. Numerical work comprises the angular
checks in Section 1 and the magnetic-field scale estimates in Section 6; a
full radial-integral or spectral comparison was not performed.

ARC has useful overlap with StarkZee, especially in atomic structure, dipole
matrix elements, and electric-field mixing. The recommended initial role is
an independent atomic benchmark and, later, an optional data provider for
neutral alkalis. Its plasma line-shape capabilities do not replace ours.

ARC's [original paper](https://arxiv.org/abs/1612.05529) and
[3.0 paper](https://arxiv.org/abs/2007.12016) provide the broader theoretical
context. ARC source links below are pinned to the reviewed commit; StarkZee
links point to files in this repository and may evolve after this report.

## 1. Hydrogenic matrix elements

ARC generally uses the coupled basis $|n,l,j,m_j\rangle$; StarkZee primarily
uses $|n,l,m_l,m_s\rangle$. These are different representations of the same
electronic angular-momentum space.

ARC separates radial integrals from angular factors using the Wigner–Eckart
theorem and 3-j/6-j symbols. We already implement the corresponding
construction in
[`reduced_hydrogenic_dipole_j`](../starkzee/radiator.py#L269), matching ARC's
[`getReducedMatrixElementJ`](https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator/blob/4b4573e/arc/alkali_atom_functions.py#L1311).

Targeted numerical comparisons gave:

| Quantity | Cases | Maximum absolute difference |
|---|---:|---:|
| Dipole-related 3-j symbols | 1,326 | $1.1\times10^{-16}$ |
| 6-j symbols | 56 | $1.1\times10^{-16}$ |
| Reduced-dipole angular factors | 56 | $1.3\times10^{-15}$ |

The checks used ARC's local `wigner.py` and StarkZee's angular routines, with
orbital angular momentum of the first state from 0 through 7 and dipole
partners with $\Delta l=\pm1$. The reduced-dipole comparison isolated the
angular factors using a common radial integral. This verifies the tested
angular factors, not the radial integrals or complete spectra.

For pure hydrogenic states, our analytical radial integrals remain a good
choice. ARC's numerical radial solver would provide a useful independent
check, particularly for weak transitions and cancellations. Comparisons must
align radial phases, reduced-mass assumptions, energy models, and polarization
conventions.

## 2. Multi-shell Stark calculations

ARC's
[`StarkMap`](https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator/blob/4b4573e/arc/calculations_atom_single.py#L674)
includes a selectable range of principal quantum numbers and orbital angular
momenta. This captures coupling between different $n$ shells.

StarkZee already has an experimental
[multi-shell module](../starkzee/multishell.py#L177), but it is not integrated
into the production static/FFM profile solvers. ARC offers an independent
implementation against which to check it.

For an isolated, nondegenerate level, the relevant second-order contribution
is

$$
\Delta E_a^{(2)}
=\sum_{b\ne a}
\frac{|\langle b|H_F|a\rangle|^2}{E_a-E_b}.
$$

Nearly degenerate hydrogenic manifolds require diagonalization, as both codes
do. Coupling to omitted shells can still shift levels and redistribute
emission strength.

A useful diagnostic is

$$
\eta_{ab}=\frac{|\langle b|H_F|a\rangle|}{|E_a-E_b|}.
$$

We should examine this over the microfields that contribute to the profile,
then compare spectra as neighboring shells are added.

This matters within the UV–near-IR wavelength range: high-$n$ Balmer and
Paschen transitions remain optical/near-IR even though their upper states are
Rydberg-like. Wavelength alone does not establish whether a single-shell
approximation is adequate.

ARC's standard Stark map fixes $m_j$, assumes the electric field defines the
symmetry axis, and adds a diagonal weak-field Zeeman shift. It omits
diamagnetism and off-diagonal magnetic mixing between different $j$ states.
Use it initially at $B=0$, or under carefully matched weak, parallel-field
assumptions, rather than as a general crossed-field benchmark.

### Hydrogen crossings and avoided crossings

Hydrogen can exhibit crossings and avoided crossings; they are not restricted
to high-$Z$ elements. The controlling scales are principally the shell number
$n$, electric-field strength, and the symmetries retained by the Hamiltonian.
For the extreme linear-Stark branches of adjacent hydrogen shells, an
order-of-magnitude crossing field is

$$
F_{\rm cross}\simeq
\frac{R_H\left[n^{-2}-(n+1)^{-2}\right]}
     {3n^2 e a_0}.
$$

The denominator uses the combined slopes of the two extreme parabolic states,
$\frac{3}{2}n(n-1)ea_0$ and
$\frac{3}{2}(n+1)n ea_0$. The estimate therefore identifies where the two
fans can first overlap; it is not the position of every state-specific
crossing or avoided crossing.

| Adjacent manifolds | $F_{\rm cross}$ [V/m] | Classical $F_{\rm ion}$ of lower $n$ [V/m] | $F_{\rm cross}/F_{\rm ion}$ |
|---|---:|---:|---:|
| 2 and 3 | $2.98\times10^9$ | $2.01\times10^9$ | 1.48 |
| 3 and 4 | $4.63\times10^8$ | $3.97\times10^8$ | 1.17 |
| 4 and 5 | $1.21\times10^8$ | $1.26\times10^8$ | 0.96 |
| 5 and 6 | $4.19\times10^7$ | $5.14\times10^7$ | 0.81 |
| 8 and 9 | $4.39\times10^6$ | $7.85\times10^6$ | 0.56 |
| 10 and 11 | $1.49\times10^6$ | $3.21\times10^6$ | 0.46 |

Here the comparison scale

$$
F_{\rm ion}\simeq\frac{F_{\rm au}}{16n^4}
$$

is the classical barrier-suppression estimate, not a sharp experimental
ionization threshold. For low shells, especially $n=2$ and 3, adjacent-shell
overlap is expected only near or above this scale. A discrete bound-state
Stark map is then questionable unless continuum coupling and field-ionization
widths are included. For $n\gtrsim4$, and increasingly for Rydberg states,
adjacent fans can overlap below the classical ionization scale.

Whether an apparent intersection is exact or avoided depends on symmetry:

- With parallel electric and magnetic fields, states in different conserved
  $m_j$ sectors can cross exactly. A fixed-$m_j$ ARC map does not display the
  other sectors.
- States with the same conserved quantum numbers generally form an avoided
  crossing when the electric-dipole Hamiltonian has a nonzero matrix element
  between them.
- The ideal nonrelativistic Coulomb-plus-uniform-field problem has additional
  parabolic-coordinate integrability, so some hydrogen crossings can remain
  exact. Fine structure, Lamb shifts, magnetic fields, transverse electric
  fields, and other perturbations can break the relevant symmetry and open a
  small gap.
- At $B=0$, the $+m$ and $-m$ partners remain degenerate. The fan emerging
  from the zero-field shell degeneracy should not itself be confused with an
  avoided crossing between initially separated levels.

For a hydrogenic ion, the shell separation scales as $Z^2$ while the orbital
radius and linear-Stark slope scale as $1/Z$. Consequently,

$$
F_{\rm cross}(Z)\propto Z^3,
\qquad
F_{\rm ion}(Z)\propto Z^3.
$$

Increasing $Z$ therefore moves the corresponding hydrogenic crossing to a
*higher* physical field while leaving its scale relative to classical
ionization approximately unchanged. Dense avoided-crossing patterns in heavy
neutral or multi-electron atoms arise from their level structure and quantum
defects, not from high nuclear charge alone.

The current StarkZee--ARC hydrogen comparison uses the
$3d_{5/2},m_j=1/2$ target, shells $n=2\ldots5$, and
$F\le5\times10^7$ V/m. This is nearly an order of magnitude below the
$n=3/4$ overlap estimate, so no adjacent-manifold crossing should appear in
the displayed $n=3$ window. A more revealing hydrogen example would use a
target near $n=8$, retain approximately $n=6\ldots10$, and scan through a few
$10^6$ V/m.

StarkZee's multi-shell Hamiltonian remains a truncated Hermitian bound-state
model. Above the ionization scale it will continue to draw discrete
eigenvalues even when the physical states should be treated as resonances;
such curves must not be interpreted as stable bound levels.

## 3. Radiative rates and population-weighted emission

ARC's
[transition-rate and lifetime routines](https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator/blob/4b4573e/arc/alkali_atom_functions.py#L2047)
implement spontaneous and blackbody-induced transitions. At zero radiation
temperature, the fine-structure-resolved E1 rate has the familiar form

$$
A_{u\rightarrow l}
=\frac{\omega_{ul}^{3}}{3\pi\epsilon_0\hbar c^3}
\frac{|\langle l\|\mathbf d\|u\rangle|^2}{2J_u+1}.
$$

StarkZee already calculates Einstein coefficients and state-resolved natural
damping in [`radiator.py`](../starkzee/radiator.py#L915). The immediate
opportunity is to compare branching fractions and lifetimes. A possible
precision improvement is to use resolved transition energies in radiative
rates; ours currently use gross-structure gaps.

For absolute emission or relative intensities across different lines, the
additional ingredient is the upper-state population. In the optically thin,
isotropic limit,

$$
j_\nu=\frac{1}{4\pi}
\sum_{u,l}N_u A_{ul}h\nu_{ul}\,\phi_{ul}(\nu),
\qquad \int\phi_{ul}\,d\nu=1.
$$

Our current profile weights are dipole strengths; they do not determine the
populations $N_u$. ARC's
[population evolution module](https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator/blob/4b4573e/arc/advanced/population_lifetime.py#L20)
illustrates radiative redistribution and cascades. Its rates do not constitute
a plasma collisional-radiative model: electron excitation, ionization,
recombination, and radiation trapping would need separate treatment.

ARC's blackbody temperature is the radiation-field temperature. It should not
automatically be equated with our electron temperature.

## 4. Quantum defects and model potentials for neutral alkalis

ARC combines measured energies with quantum-defect models,

$$
E_{nlj}=-\frac{R_M}{[n-\delta_{lj}(n)]^2},
$$

and, for alkalis, numerical radial functions in an effective core potential.
Its
[model potential](https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator/blob/4b4573e/arc/alkali_atom_functions.py#L431)
includes screened nuclear attraction and core polarization;
[Numerov integration](https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator/blob/4b4573e/arc/alkali_atom_functions.py#L503)
supplies radial matrix elements.

This could support neutral Li, Na, K, Rb, and Cs lines in the visible/near-IR.
For low-lying states, measured energies and literature dipoles should take
precedence over extrapolated Rydberg fits.

Our [multi-electron module](../starkzee/multielectron.py#L155) is still a stub
with a placeholder magnetic factor. An ARC-backed atomic-data interface could
supply energies, quantum numbers, reduced dipoles, and provenance. However:

- Neutral alkali model potentials cannot be transferred directly to arbitrary
  ions such as C IV.
- Changing the atomic structure also requires revisiting our hydrogenic
  collision/broadening operators.
- ARC's divalent support is more approximate: numerical radial wavefunctions
  are unimplemented there, and its lifetime routine explicitly notes missing
  electron-correlation effects. See
  [`DivalentAtom`](https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator/blob/4b4573e/arc/divalent_atom_functions.py#L785).

## 5. Interpretation and other useful tools

For emission from mixed states, preserve the coherent transformation already
used in [our solver](../starkzee/static_profile.py#L654):

$$
S_{ul}^{(q)}
=\left|\left(V_l^\dagger D_q V_u\right)_{lu}\right|^2.
$$

ARC's Stark-map visualization
[combines squared contributions](https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator/blob/4b4573e/arc/calculations_atom_single.py#L1007)
for its driving-strength highlight. That quantity should not be imported as a
general emission strength because it omits amplitude interference.

Other features have narrower value:

| ARC feature | Relevance to StarkZee |
|---|---|
| Hyperfine matrices and Breit–Rabi calculations | Useful for sufficiently narrow, resolved spectra; requires nuclear-spin data and an expanded basis |
| Dynamic polarizability and Floquet methods | Useful if coherent laser/RF fields perturb the emitter; distinct from stochastic plasma microfields |
| Literature dipoles with uncertainties and references | A useful model for an atomic-data interface |
| Pair-state $C_3/C_6$ calculations | Specialized neutral-atom interactions; not a direct substitute for plasma collision operators |
| `drawSpectraConvoluted` | A common-width Lorentzian plotting helper, not an additional plasma line-shape theory |

## 6. Zeeman support and estimated H/D magnetic-field validity

### What ARC actually implements

| ARC calculation | Magnetic treatment | Important omission |
|---|---|---|
| `getZeemanEnergyShift()` and `StarkMap(..., Bz=...)` | Diagonal expectation of the linear electronic Zeeman operator in a fixed-$j$ basis | Magnetic coupling between different $j$ states, diamagnetic $B^2$ term, and general crossed-field geometry |
| `breitRabi(n, l, j, B)` | Diagonalization of hyperfine and Zeeman interactions within one selected $n,l,j$ manifold | Mixing with other fine-structure manifolds and diamagnetism |

The relevant implementations are
[`getZeemanEnergyShift`](https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator/blob/4b4573e/arc/alkali_atom_functions.py#L2647)
and
[`breitRabi`](https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator/blob/4b4573e/arc/alkali_atom_functions.py#L3688).
The latter can describe the hyperfine Paschen–Back regime, in which nuclear
spin $I$ decouples from electronic $J$. This is distinct from the electronic
Paschen–Back regime, in which $L$ and $S$ decouple and different $j$ levels
mix. ARC's standard magnetic treatments do not capture that second regime
in general. Increasing the electric-field basis size alone does not add the
missing magnetic matrix elements.

There is also an H/D data limitation: in this checkout, the bundled
[`Hydrogen` class](https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator/blob/4b4573e/arc/alkali_atom_data.py#L136)
does not override the base class defaults `I=0`, `gI=0`, or the empty
`hyperfineStructureData`. There is no bundled `Deuterium` class. Thus ARC has
general hyperfine machinery, but does not provide ready-to-use physical H/D
Breit–Rabi calculations. These require isotope-specific nuclear spin,
magnetic moment, hyperfine coefficients, masses, and electronic energies.
Changing only an atomic mass would not supply all of these.

### Electronic fine-structure crossover

At zero electric field, define the characteristic field

$$
B_{\rm fs}(n,l)=\frac{\Delta E_{n,l,j=l\pm1/2}}{\mu_B},
\qquad l\ge1.
$$

Using leading hydrogenic fine structure with an approximate reduced-mass
Rydberg $R_M$ in energy units gives

$$
\Delta E_{\rm fs}\simeq
\frac{\alpha^2 R_M}{n^3 l(l+1)},\qquad
B_{\rm fs}^{\rm H}\simeq
\frac{12.51}{n^3 l(l+1)}\ {\rm T}.
$$

Here $\mu_B/h=13.9962$ GHz/T. The estimates below use SciPy physical
constants and StarkZee's `reduced_mass_rydberg_ev(1, A)`. They neglect detailed
recoil and QED corrections. As an independent check, the
[NIST hydrogen levels](https://physics.nist.gov/PhysRefData/Handbook/Tables/hydrogentable5.htm)
give a 2p separation of approximately $0.3659\ {\rm cm}^{-1}$, or 10.97 GHz,
corresponding to $B_{\rm fs}\simeq0.784$ T, close to the leading estimate.

| Shell $n$ | H: $B_{\rm fs}(np)$ [T] | H: smallest same-$l$ doublet scale in shell, $l=n-1$ [T] | D: same smallest scale [T] | H/D illustrative weak-field screen: $0.1\,B_{\rm fs,min}$ [mT] |
|---|---:|---:|---:|---:|
| 2 | 0.782 | 0.782 | 0.782 | 78.2 |
| 3 | 0.232 | 0.0772 | 0.0772 | 7.72 |
| 4 | 0.0977 | 0.0163 | 0.0163 | 1.63 |
| 5 | 0.0500 | 0.00500 | 0.00501 | 0.500 |
| 6 | 0.0290 | 0.00193 | 0.00193 | 0.193 |
| 8 | 0.0122 | 0.000436 | 0.000436 | 0.0436 |

The H and D electronic field scales are practically identical at this
precision. In this leading reduced-mass estimate, D values are larger by
approximately 0.0272%; this is not a precision calculation of isotope-dependent
fine structure.

Interpretation:

- $B\ll B_{\rm fs}$ is the regime for a fixed-$j$, first-order Zeeman
  approximation. The last column is an explicit order-of-magnitude screening
  choice, not a certified error tolerance or a sharp boundary. Missing
  off-diagonal amplitudes scale with $\mu_B B/\Delta E_{\rm fs}$, with
  state-dependent angular factors; associated energy corrections start at
  second order away from degeneracy.
- At $B\sim B_{\rm fs}$, diagonalize the electronic spin–orbit plus Zeeman
  Hamiltonian. ARC's diagonal magnetic approximation is no longer generally
  suitable. Some stretched substates are exceptions, but that does not make
  a complete multiplet valid.
- The smallest-shell column is appropriate when all orbital states in the
  shell matter, for example after Stark mixing. For selected zero-field
  transitions, use the participating $l$ values instead. The ground/other
  $s$ states have no same-$l$ spin–orbit doublet; this particular bound does
  not apply to them.

For H-α/D-α, the relevant upper-shell doublet scales are about 0.232 T for
3p and 0.0772 T for 3d. A field below roughly 8 mT is an illustrative
weak-electronic-field screen for the entire $n=3$ manifold. For H-β/D-β,
4d has a scale of 0.0326 T; including Stark-accessible 4f reduces the
whole-shell scale to 0.0163 T. These are electronic criteria only: hyperfine
resolution and electric mixing must also be assessed.

At the 1–5 T fields used in representative StarkZee plasma calculations,
ARC's standard fixed-$j$ Zeeman treatment should not be used as a general
reference for resolved H/D Balmer patterns. A broadened unresolved envelope
may be less sensitive, but that requires an actual profile comparison.

An additional data check is needed for high $n$: ARC defaults to
`preferQuantumDefects=True`, and `Hydrogen.minQuantumDefectN` is 8. For
$l\le4$, the zero-defect energy branch then lacks the resolved hydrogenic
fine structure. Use verified resolved energies for these comparisons;
`preferQuantumDefects=False` only helps where the required tabulated states
actually exist.

### Hyperfine crossover: a different scale

For an $nS_{1/2}$ state, a useful estimate is

$$
B_{\rm hfs}\simeq\frac{h\Delta\nu_{\rm hfs}}{g_J\mu_B},
\qquad g_J\simeq2.0023.
$$

Neglecting the small nuclear Zeeman correction and using the leading
$n^{-3}$ scaling for excited $S$ states gives:

| State | H crossover [mT] | D crossover [mT] |
|---|---:|---:|
| 1s | 50.7 | 11.7 |
| 2s | 6.34 | 1.46 |
| 3s | 1.88 | 0.433 |

The ground-state input intervals are approximately 1420.40575 MHz for H and
327.38435 MHz for D. See
[NIST's atomic-structure overview](https://www.physics.nist.gov/Pubs/AtSpec/node03.html)
and the H/D hyperfine tables in
[Kramida's compilation](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=842564).
The excited-state entries above are scaling estimates, not tabulated
measurements; do not apply the $S$-state values to $P,D,\ldots$ states.

Below this scale a linear shift of hyperfine $F$ levels can be appropriate;
near it, a Breit–Rabi calculation is needed; above it, nuclear/electronic
decoupling occurs. High field does not erase hyperfine offsets. Omitting
hyperfine structure in StarkZee is justified by the desired resolution and
linewidth, not merely by exceeding $B_{\rm hfs}$.

### Diamagnetic effects and the remaining StarkZee limitations

The electronic diamagnetic term is

$$
H_{\rm dia}=\frac{e^2 B^2}{8m_e}(x^2+y^2).
$$

For a field-free hydrogenic $ns$ state, neglecting small isotope corrections
to its radius,

$$
\frac{\langle x^2+y^2\rangle}{a_0^2}
=\frac{n^2(5n^2+1)}{3},\qquad
\Delta E_{\rm dia}\simeq
6.1565\times10^{-11}\,B^2
\frac{\langle x^2+y^2\rangle}{a_0^2}\ {\rm eV},
$$

where $B$ is in tesla. This gives the following illustrative level-shift
scales for both H and D:

| Shell | $ns$ diamagnetic coefficient [eV/T²] | Field where this shift is 1% of $\mu_B B$ [T] |
|---|---:|---:|
| 2 | $1.72\times10^{-9}$ | 336 |
| 3 | $8.50\times10^{-9}$ | 68.1 |
| 4 | $2.66\times10^{-8}$ | 21.8 |
| 5 | $6.46\times10^{-8}$ | 8.95 |
| 6 | $1.34\times10^{-7}$ | 4.33 |
| 8 | $4.22\times10^{-7}$ | 1.37 |
| 10 | $1.03\times10^{-6}$ | 0.563 |

These are representative field-free $ns$ expectations, not bounds on every
mixed state or 1% errors in an emission profile. Transition shifts involve
upper-minus-lower shifts, and even a shift small relative to $\mu_B B$ can
matter relative to a narrow linewidth. Diamagnetic mixing within nearly
degenerate shells can also matter earlier than this screen suggests.

StarkZee includes electronic spin–orbit/Zeeman mixing and the within-shell
diamagnetic operator, so the ARC fixed-$j$ thresholds above are not upper
field limits for StarkZee. Its remaining basis limitation is coupling to
omitted shells. The relevant tests are matrix elements of $H_F$ and
$H_{\rm dia}$ between retained and omitted states, divided by their energy
gaps, followed by multi-shell convergence of the actual spectrum.

For orientation only, equating the representative $ns$ diamagnetic shift to
the adjacent zero-field shell gap
$R_H[n^{-2}-(n+1)^{-2}]$ gives about 8,820 T at $n=3$, 1,600 T at $n=5$,
325 T at $n=8$, and 152 T at $n=10$. Mixing/error can become relevant well
below equality; these are not validated maximum fields. Plasma electric
microfields may set a tighter basis requirement independently of $B$.

Thus low-$n$ H/D emission at a few tesla lies beyond ARC's general linear
fixed-$j$ regime but is naturally addressed by StarkZee's electronic
Hamiltonian. This statement does not certify the complete plasma profile:
hyperfine resolution, microfield averaging, collision approximations, and
multi-shell convergence remain separate requirements. Moving-atom motional
electric fields are another separate effect; see
[`report_TMSE.md`](report_TMSE.md).

## 7. Radial equations and coupled versus uncoupled bases

There are real differences in the radial models and retained Hamiltonian
terms, while the angular basis choices are mathematically equivalent when
they span the same states. For low-$n$ H/D plasma emission, StarkZee's choices
are appropriate; ARC's choices are more flexible for alkalis and weak-field
atomic-control experiments.

### Radial functions and matrix elements

StarkZee uses analytical, nonrelativistic Coulomb wavefunctions:

$$
R_{nl}(r)=N_{nl}e^{-Zr/n}
\left(\frac{2Zr}{n}\right)^l
L_{n-l-1}^{2l+1}\left(\frac{2Zr}{n}\right),
\qquad
\int_0^\infty |R_{nl}(r)|^2r^2\,dr=1,
$$

where $r$ is measured in Bohr radii. See
[`radial_wavefunction`](../starkzee/radiator.py#L54).

ARC's alkali implementation numerically integrates a radial Schrödinger
equation using Numerov. Its
[`radialWavefunction`](https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator/blob/4b4573e/arc/alkali_atom_functions.py#L503)
returns the reduced radial function $u(r)=rR(r)$, normalized in the
implementation so that $\int |u(r)|^2\,dr=1$. Consequently, the apparently
different dipole-integral expressions are equivalent:

$$
\underbrace{\int R_a(r)R_b(r)\,r^3\,dr}_{\text{StarkZee}}
=
\underbrace{\int u_a(r)u_b(r)\,r\,dr}_{\text{ARC}}.
$$

These expressions use real radial functions; complex functions require
conjugation of the bra. The different integration measures do not constitute
a physical disagreement.

| Aspect | StarkZee | ARC |
|---|---|---|
| Radial potential | Pure Coulomb | Coulomb plus the implemented spin–orbit term for hydrogen; fitted core/polarization potential plus spin–orbit for alkalis |
| Radial dependence | $n,l,Z$; independent of $j$ | Can depend on $j$ through the spin–orbit potential and chosen energy |
| Dipole evaluation | Gordon analytical formula by default between different shells; quadrature for same-shell radial calls | Literature value, cached value, or numerical radial integration |
| Radial finite-mass treatment | Uses the usual electron-mass Bohr radius | Numerov equation includes a reduced-mass factor |
| Broader applicability | Hydrogen-like atoms | Alkalis with nonhydrogenic valence electrons |

StarkZee's analytical Coulomb integrals avoid radial mesh and boundary errors
and are efficient for low-$n$ hydrogenic calculations. The default
[`radial_dipole`](../starkzee/radiator.py#L239) uses Gordon's closed form for
different shells, with an alternative quadrature backend; same-shell calls
use quadrature. Production intra-shell Stark templates instead use an
analytical same-shell expression directly.

The limitation is that these Coulomb functions do not include core
polarization or fine-structure-dependent changes to radial shapes. Also,
although StarkZee corrects gross-structure level energies for isotope mass,
its radial functions do not receive that same mass correction. This is a
small precision limitation for H/D, separate from the angular basis choice.

ARC's flexibility does not automatically make its hydrogen values more
accurate: the selected energies, numerical integration, and cached/literature
data all matter. Neither radial implementation described here is a complete
relativistic Dirac radial calculation. Analytical formulas also require
numerical-stability checks at very high $n$, where floating-point cancellation
or overflow can become important.

### Coupled and uncoupled angular bases

ARC generally uses $|nljm_j\rangle$, whereas StarkZee uses
$|nlm_lm_s\rangle$ with $s=1/2$. Their relationship is a Clebsch–Gordan
transformation:

$$
|nljm_j\rangle
=\sum_{m_l,m_s}
\langle lm_l,\tfrac12m_s|jm_j\rangle
|nlm_lm_s\rangle.
$$

With identical physics and complete matching subspaces,

$$
H_{\rm coupled}=U^\dagger H_{\rm uncoupled}U,
$$

and both representations produce the same energies and transition strengths
when the dipole operators are transformed consistently.

| Property | StarkZee: $\lvert nlm_lm_s\rangle$ | ARC: $\lvert nljm_j\rangle$ |
|---|---|---|
| Field-free fine structure | Spin–orbit coupling must be diagonalized | Naturally diagonal |
| Electronic Zeeman operator | Simple diagonal form: $\mu_BB(m_l+g_sm_s)$ | Generally includes off-diagonal coupling between different $j$ |
| Strong-field electronic Paschen–Back regime | Particularly convenient | Equally possible if the off-diagonal magnetic terms are retained |
| Tabulated fine-structure energies/dipoles | Requires transformation | Natural representation |
| Hyperfine extension | Possible by adding nuclear spin | Convenient starting point for coupling $J$ and $I$ |
| Nonparallel electric and magnetic fields | Convenient full-shell treatment | Requires all necessary $m_j$ sectors and couplings |

The coupled basis itself does not impose a weak-field approximation. ARC's
standard Stark-map implementation makes that approximation by retaining
diagonal magnetic shifts. The same coupled basis could be used with a more
complete magnetic Hamiltonian.

Our uncoupled basis is not inherently larger. A complete electronic shell
has $2n^2$ states in either representation. ARC often uses a smaller fixed-$m_j$
block because its geometry permits it. StarkZee could also exploit conserved
$m_j=m_l+m_s$ for parallel fields; a transverse microfield couples those
blocks. Computational savings from symmetry reduction should therefore be
distinguished from the choice of angular basis.

### Hamiltonian completeness and phase conventions

Our [Hamiltonian](../starkzee/radiator.py#L437) includes electronic spin–orbit
mixing, linear Zeeman coupling, and diamagnetism. Its analytical fine structure
includes mass–velocity and Darwin corrections in addition to spin–orbit;
the empirical path instead uses measured fine-structure energies.

ARC's standard Stark map uses supplied fine-structure energies and several
$n$ shells, but its magnetic treatment is less complete. Its multi-shell
coverage and our more complete magnetic treatment are independent advantages,
not consequences of coupled versus uncoupled coordinates.

Signs need care. Our signed hydrogenic radial integral gives

$$
\langle2s|r|2p\rangle=-3\sqrt3\,a_0,
$$

while our production intra-shell Stark templates use the opposite radial
convention. These are related by the basis rephasing
$|nlm_lm_s\rangle\mapsto(-1)^l|nlm_lm_s\rangle$. Physical results are
unchanged when every operator is transformed consistently; mixing conventions
carelessly can corrupt interference calculations. See the
[Stark-template convention](../starkzee/static_profile.py#L42) and the
[collision-operator rephasing helper](../starkzee/collision.py#L23).
Individual signed ARC and StarkZee matrix elements must likewise be compared
only after matching state phases and spherical-component conventions.

The recommendation is to retain our analytical hydrogenic radial backend and
uncoupled basis. Useful improvements are independent ARC radial checks,
consistent finite-mass radial scaling if the required precision warrants it,
and multi-shell convergence, rather than switching bases merely to match ARC.

## Recommended first step

Build a small optional ARC comparison suite for Ly-α, H-α, H-β, and Paschen-β:
resolved energies, radial/reduced dipoles, branching fractions, and Stark
mixing at zero magnetic field with multiple shells. Keep NIST benchmarks
alongside it, since ARC is another model with its own approximations.

After that, the most useful development would be a general atomic-data
interface if neutral alkali emission is a real target. These are proposed
follow-up tasks; the review did not change production code or add a dependency
on ARC.
