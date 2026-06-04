# Notes

## Theoretical differences: Stehle vs StarkZee

### Stark broadening (common ground)

Both compute the intra-shell Stark Hamiltonian by diagonalizing the same `2n² x 2n²`
matrix in the uncoupled `|n, l, ml, ms>` basis and integrate over a quasi-static
microfield distribution. That part is equivalent.

### Key theoretical differences

| Feature | StarkZee | Stehle (tabulated) |
|---|---|---|
| Magnetic field | Full simultaneous diagonalization of H_Stark + H_Zeeman (linear + quadratic) | None — B=0 assumption baked into the tables |
| Fine structure | Yes — spin-orbit, mass-velocity, Darwin | No — pure degenerate hydrogenic levels |
| Electron broadening | GBK: semi-classical Lorentzian, frequency-dependent | Unified theory: quantum S-matrix, more rigorous at high detuning |
| Ion dynamics | Quasi-static (pure) + optional FFM | More sophisticated — closer to the unified/FFM treatment in the original tables |
| Microfield distribution | Hooper screened | Similar (Holtsmark or Hooper variant) |
| How B is added (in comparison) | Intrinsic to the Hamiltonian | Added externally as a convolution/shift on top of the B=0 table |

### Most important physical distinction

At B != 0, StarkZee diagonalizes `H = H_atom + H_Stark + H_Zeeman` **simultaneously**.
When the Zeeman and Stark splittings are comparable — which happens whenever
`mu_B * B ~ 3n * e * a0 * F` — the levels mix and you get genuine Stark-Zeeman
eigenstates that are neither pure Zeeman nor pure Stark states. No post-processing
convolution can reproduce this.

For H-alpha at B = 10 T: `mu_B * B ~ 0.58 meV`. The Stark splitting at the Holtsmark
field (Ne = 5e21 m^-3) is ~0.18 nm ~ 0.8 meV. They are the same order of magnitude,
so the coupling is real and significant. This is exactly the regime where Stehle's
B=0 tables break down and StarkZee's coupled treatment matters.

### Fine structure shift

Stehle places the line center at the gross-structure Rydberg energy
`E0 = Z^2 * Ry * (1/nl^2 - 1/nu^2)`. StarkZee with `fine_structure=True` includes
the Dirac corrections, which shift the physical center ~0.009 nm toward shorter
wavelengths for D-alpha.

### GBK vs unified theory (electron broadening)

GBK is a semi-classical model that becomes inaccurate in the far wings (large
detunings) and at high density where the impact approximation breaks down. The Stehle
tables use a more rigorous quantum treatment that is valid over a wider range. At
moderate densities (Ne <= 1e21 m^-3) and for the line core, the two are close.
