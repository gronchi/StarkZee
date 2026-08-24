# Stark-Zeeman broadening and line profile calculations for starkzee

import warnings

import numpy as np
from functools import lru_cache
from starkzee.utils import A0, reduced_mass_rydberg_ev, wavenumber_cm_to_energy_ev, energy_ev_to_wavenumber_cm
from scipy.constants import hbar as _HBAR, e as _E_CHARGE, m_p as _M_P, c as _C_LIGHT

_SQRT_2PI = np.sqrt(2.0 * np.pi)

from starkzee.radiator import (
    build_hamiltonian, build_basis, angular_dipole_element, radial_dipole,
    _uncoupled_dipole_matrices, einstein_a,
)
from starkzee.microfield import microfield_quadrature
from starkzee.broadening import (
    electron_impact_width, electron_impact_width_model, electron_impact_r2_scaling,
)

@lru_cache(maxsize=None)
def _stark_templates(n, Z):
    """Return the field-independent Stark coupling matrices M_z and M_x [eV/(V/m)].

    V_E(Fz, Fx) = Fz * M_z + Fx * M_x, so these only need to be built once
    per (n, Z) pair regardless of how many microfield quadrature points are used.

    Sign convention: the intra-shell radial element is taken as
    +(3n/2Z)√(n²−l²), which is the *opposite* sign of the actual integral with
    the standard radial functions (⟨2s|r|2p⟩ = −3√3 a₀; see
    :func:`~starkzee.radiator.radial_dipole`).  The two conventions are related
    by the rephasing |n,l,m⟩ → (−1)^l |n,l,m⟩, which flips every Δl = ±1
    element uniformly and leaves eigenvalues and all |d|² intensities
    unchanged — but any *new* operator mixing this convention with the signed
    radial integrals (e.g. an off-diagonal broadening operator) must pick one
    convention consistently.
    """
    basis = build_basis(n)
    dim = len(basis)
    M_z = np.zeros((dim, dim), dtype=complex)
    M_x = np.zeros((dim, dim), dtype=complex)
    for i, state_i in enumerate(basis):
        for j, state_j in enumerate(basis):
            if state_i.ms == state_j.ms and abs(state_i.l - state_j.l) == 1:
                l = max(state_i.l, state_j.l)
                r_val = (3.0 * n / (2.0 * Z)) * np.sqrt(n**2 - l**2)
                z_ang = angular_dipole_element(state_i.l, state_i.ml, state_j.l, state_j.ml, 0)
                x_ang = (angular_dipole_element(state_i.l, state_i.ml, state_j.l, state_j.ml, -1) +
                         angular_dipole_element(state_i.l, state_i.ml, state_j.l, state_j.ml,  1)) / np.sqrt(2.0)
                M_z[i, j] += -z_ang * r_val * A0
                M_x[i, j] += -x_ang * r_val * A0
    return M_z, M_x


def build_stark_matrix(n, Z, Fz, Fx):
    """Build the (2n²) × (2n²) Stark electric-field perturbation matrix in eV.

    The linear Stark interaction for an electron in an external electric field
    F = Fz ẑ + Fx x̂ is:

        V_E = −e (z Fz + x Fx)

    In the hydrogenic basis the matrix elements reduce to products of a radial
    element and an angular element.  The within-shell (Δn = 0) radial element
    ⟨n, l | r | n, l±1⟩ is given analytically:

        ⟨n, l | r | n, l−1⟩ = (3n/2Z) √(n² − l²)   [a₀]

    The angular elements ⟨l, m_l | cos θ | l±1, m_l⟩ (for Fz) and the
    combinations for Fx are provided by :func:`~starkzee.radiator.angular_dipole_element`.

    The x-component is constructed as:

        x/r = (T_{−1} + T_{+1}) / √2

    where T_q are the spherical tensor components of r̂.

    Parameters
    ----------
    n : int
        Principal quantum number.
    Z : int
        Nuclear charge.
    Fz : float
        Electric field component along B (z-axis) [V m⁻¹].
    Fx : float
        Electric field component perpendicular to B (x-axis) [V m⁻¹].

    Returns
    -------
    ndarray, shape (2n², 2n²), dtype complex
        Hermitian Stark perturbation matrix in eV.

    Notes
    -----
    This matrix operates within a single n-shell; the quadratic Stark effect
    (coupling to n ± 1, ±2 shells) is neglected, which is valid when the
    Stark shift ≪ the shell spacing Z² Ry (1/n² − 1/(n+1)²).
    """
    basis = build_basis(n)
    dim = len(basis)
    V_E = np.zeros((dim, dim), dtype=complex)
    
    # Hydrogenic radial matrix element within same n:
    # <n, l | r | n, l-1> = (3n / 2Z) * sqrt(n^2 - l^2)
    for i, state_i in enumerate(basis):
        for j, state_j in enumerate(basis):
            if state_i.ms == state_j.ms and abs(state_i.l - state_j.l) == 1:
                l = max(state_i.l, state_j.l)
                r_val = (3.0 * n / (2.0 * Z)) * np.sqrt(n**2 - l**2)
                
                # z coupling (q=0)
                z_ang = angular_dipole_element(state_i.l, state_i.ml, state_j.l, state_j.ml, 0)
                
                # x coupling: x/r = (T_{-1} + T_{+1})/√2
                # where angular_dipole_element(q=+1) = ⟨(x+iy)/(r√2)⟩
                #   and angular_dipole_element(q=-1) = ⟨(x-iy)/(r√2)⟩
                # Sum: (q=-1 + q=+1)/√2 = (x-iy+x+iy)/(r√2·√2) = x/r  ✓
                # (Using minus gave an anti-symmetric, non-Hermitian matrix.)
                x_ang = (angular_dipole_element(state_i.l, state_i.ml, state_j.l, state_j.ml, -1) +
                         angular_dipole_element(state_i.l, state_i.ml, state_j.l, state_j.ml, 1)) / np.sqrt(2.0)
                
                V_E[i, j] += -(z_ang * Fz + x_ang * Fx) * r_val * A0
                
    return V_E

def solve_starkzee(n, Z, B, Fz, Fx, quadratic_zeeman=True,
                               fine_structure=True, A=1,
                               use_empirical_data=False, atom="H"):
    """Diagonalize the combined Stark + Zeeman Hamiltonian for shell n.

    Adds the Stark perturbation :func:`build_stark_matrix` to the
    atomic/magnetic Hamiltonian :func:`~starkzee.radiator.build_hamiltonian` and
    diagonalizes the sum with ``numpy.linalg.eigh``:

        H = H_atom(B) + V_E(Fz, Fx)

    This is the inner-loop solver called for every (microfield magnitude,
    microfield angle) quadrature point during profile integration.

    Parameters
    ----------
    n : int
        Principal quantum number.
    Z : int
        Nuclear charge.
    B : float
        Magnetic field [T].
    Fz : float
        Electric field component along B [V m⁻¹].
    Fx : float
        Electric field component perpendicular to B [V m⁻¹].
    quadratic_zeeman : bool, optional
        Include the diamagnetic quadratic Zeeman term (default True).
    fine_structure : bool, optional
        Include MV + Darwin corrections (default True).
    use_empirical_data : bool, optional
        Use NIST empirical level energies for the field-free Hamiltonian
        instead of the analytic Rydberg formula (default False).  The
        internal Hamiltonian is assembled in cm⁻¹; eigenvalues are
        converted back to eV before returning so callers see no unit change.
    atom : str, optional
        Atom identifier passed to :func:`~starkzee.atomic_data.load_levels`
        when ``use_empirical_data=True`` (default ``"H"``).

    Returns
    -------
    eigenvalues : ndarray, shape (2n²,)
        Energy eigenvalues in ascending order [eV].
    eigenvectors : ndarray, shape (2n², 2n²)
        Orthonormal eigenstates as columns, in the canonical ``|n, l, m_l, m_s⟩`` basis.
    """
    H_atom = build_hamiltonian(n, Z, B, quadratic_zeeman, fine_structure, A,
                               use_empirical_data=use_empirical_data, atom=atom)
    # V_E(Fz, Fx) = Fz*M_z + Fx*M_x with field-independent templates cached per
    # (n, Z), avoiding a Python double-loop rebuild of the Stark matrix on every
    # call (build_stark_matrix produces exactly Fz*M_z + Fx*M_x).
    M_z, M_x = _stark_templates(n, Z)

    if use_empirical_data:
        # H_atom is in cm⁻¹; scale the Stark matrices (eV/(V/m)) to cm⁻¹/(V/m).
        ev_to_cm = energy_ev_to_wavenumber_cm(1.0)
        H_total = H_atom + (Fz * M_z + Fx * M_x) * ev_to_cm
        # Center near 0 using the mean of the empirical levels (≈82259 cm⁻¹ for H n=2).
        E_shift = H_total.diagonal().real.mean()
        for i in range(H_total.shape[0]):
            H_total[i, i] -= E_shift
        eigenvalues_cm, eigenvectors = np.linalg.eigh(H_total)
        eigenvalues_cm += E_shift
        return wavenumber_cm_to_energy_ev(eigenvalues_cm), eigenvectors

    H_total = H_atom + Fz * M_z + Fx * M_x
    # Shift diagonal to center eigenvalues near 0, reducing the spectral norm.
    # This improves the eigh solver's numerical precision.
    En = -(Z**2) * reduced_mass_rydberg_ev(Z, A) / (n**2)
    for i in range(H_total.shape[0]):
        H_total[i, i] -= En
    eigenvalues, eigenvectors = np.linalg.eigh(H_total)
    eigenvalues += En
    return eigenvalues, eigenvectors

def calculate_static_profile(n_u, n_l, Z, B, Ne_m3, Te_ev, energies_ev,
                                     num_f=20, num_mu=6, max_beta=10.0,
                                     use_screening=True,
                                     quadratic_zeeman=True, fine_structure=True,
                                     frequency_dependent_width=True, A=1,
                                     Ti_ev=None, species='H', electron_model='pppb',
                                     electron_operator=False,
                                     use_empirical_data=False, atom="H"):
    """Compute the static-ion Stark-Zeeman line profile for n_u → n_l.

    Integrates the Stark-Zeeman Hamiltonian over the plasma microfield distribution
    using Gauss-Legendre quadrature for both the field magnitude and the angle μ = cos θ
    between the microfield and B.  For each quadrature point the combined
    Hamiltonian H = H_atom(B) + V_E(Fz, Fx) is diagonalized and the transition
    intensities accumulated into three polarization components.

    Parameters
    ----------
    n_u, n_l : int
        Upper and lower principal quantum numbers.
    Z : int
        Nuclear charge (1 for hydrogen).
    B : float
        Magnetic field [T].  ``B=0`` is fully supported; at zero field the
        π/σ decomposition is physically meaningless (no preferred axis) and all
        three returned components are equal by spherical symmetry.
    Ne_m3 : float
        Electron density [m⁻³].
    Te_ev : float
        Electron temperature [eV].
    energies_ev : array-like
        Photon energies at which to evaluate the profile [eV].
    num_f : int, optional
        Number of microfield quadrature points (default 20).  Ferri et al.
        (2022) recommend ~50; 20 is sufficient for hydrogen but should be
        increased for multi-electron atoms (future: set dynamically by atom type).
    num_mu : int, optional
        Number of Gauss-Legendre angle points (default 6).  Ferri et al.
        (2022) recommend ~30; 6 is sufficient for hydrogen but should be
        increased for multi-electron atoms (future: set dynamically by atom type).
    max_beta : float, optional
        Upper limit of the reduced-microfield grid β = F/F₀ (default 10).
        The Holtsmark tail beyond β = 10 carries ~3 % of the probability, which
        the quadrature truncates and renormalizes away; increase this (together
        with ``num_f``) when the quasi-static far wings matter.  Forwarded to
        :func:`~starkzee.microfield.microfield_quadrature`.
    use_screening : bool, optional
        Use Hooper screened microfield distribution (default True).
    quadratic_zeeman : bool, optional
        Include diamagnetic (quadratic) Zeeman term (default True).
    fine_structure : bool, optional
        Include mass-velocity and Darwin corrections (Dirac fine structure)
        so that 2s_{1/2} = 2p_{1/2} (default True).
    frequency_dependent_width : bool, optional
        When True (default), evaluate the GBK electron-impact width **pointwise
        at the observation detuning from the gross-structure line center**,
        ``w(E − E0)``, as in the PPPB operator Φ(Δω) of Ferri et al. (2022)
        Eq. (19), where Δω is "the frequency detuning from the line center".
        Every component then shares the same frequency-dependent width
        function: a far-shifted σ± component receives a reduced width at its
        own position, and the far wings of *all* components relax toward the
        strong-collision floor C_n as G(Δω) → 0 (breakdown of the impact
        approximation), making the components non-Lorentzian.
        When False, use the single on-resonance value ``w(0)`` everywhere
        (faster; valid when the profile extent ≪ ω_c).  Note the pointwise
        profile is not exactly area-normalized per component (the physical
        GBK non-Lorentzian shape isn't either).
    Ti_ev : float, optional
        Ion temperature [eV].  When supplied, Doppler broadening is folded into
        the Lorentzian accumulation as a Voigt profile, eliminating the need for
        a separate post-processing convolution.  Default is ``None`` (bare
        Lorentzian, no Doppler).
    species : str, optional
        Emitting species (``'H'``, ``'D'``, or ``'T'``); used to determine the
        ion mass for the Doppler width.  Only relevant when *Ti_ev* is set.
        Default is ``'H'``.
    electron_model : str, optional
        Electron-impact width prescription, forwarded to
        :func:`~starkzee.broadening.electron_impact_width_model`.  ``'pppb'``
        (default) is the PPPB model (B-dependent ω_c cutoff); ``'zest'``,
        ``'zest-lee'``, ``'zest-dufty'`` select the ZEST model with the GBK,
        Lee, or Dufty RPA G-function.
    electron_operator : bool, optional
        When ``True``, use the electron-impact **operator** diagonal: each
        Stark-Zeeman dressed state gets a width scaled by its own ⟨k|r²|k⟩
        (:func:`~starkzee.broadening.electron_impact_r2_scaling`) instead of the
        shell-averaged scalar (default ``False``).  This is the **ZEST operator**
        treatment with the off-diagonal / ``c_k`` set to zero.  The full PPPB
        operator (off-diagonal → complex intensity ``a_k + i c_k``) and the
        lower-manifold ``d†·d`` contribution are not yet implemented — see the
        REVIEW note in :func:`~starkzee.broadening.electron_impact_r2_scaling`.
        Currently applied in the in-loop Lorentzian path; with Doppler-dominant
        grids (the post-FFT Lorentzian) the scalar resonance width is used.
    use_empirical_data : bool, optional
        Use NIST empirical level energies for the field-free Hamiltonian
        instead of the analytic Dirac (spin-orbit + MV + Darwin) formula
        (default False); forwarded to :func:`~starkzee.radiator.build_hamiltonian`.
        The empirical levels include the Lamb shift, so the resulting profile
        centroid matches the measured NIST wavelength more closely than the
        pure-Dirac default, which is Lamb-shift-free.  ``atomic_levels.json``
        tabulates H (n ≤ 8), D (n ≤ 6), and T (n ≤ 3) levels — pass the
        matching ``atom`` for the emitting *species*: the default ``atom="H"``
        combined with ``species='D'``/``'T'`` reproduces the plain-hydrogen
        line center, not the isotope-shifted one (~0.1-0.2 nm for Balmer-α).
    atom : str, optional
        Atom identifier passed to :func:`~starkzee.atomic_data.load_levels`
        when ``use_empirical_data=True`` (default ``"H"``).

    Notes on approximations
    -----------------------
    **sigma_D**: the Doppler width is computed as
    ``σ_D = E_mean × sqrt(Ti / mc²)`` where ``E_mean = mean(energies_ev)``.
    Strictly, each transition at energy ``dE_i`` should use ``σ_D(dE_i)``, but
    the fractional error is ``ΔE/E ≈ Δ_Zeeman/E0`` — about 0.03 % for H-alpha
    at 10 T — and is negligible in practice.

    **Lorentzian FFT step**: when Doppler is active and dominant (the
    Gaussian-loop path), the post-loop Lorentzian convolution uses
    ``w_resonance`` (the on-resonance GBK width) for both
    ``frequency_dependent_width`` settings — an FFT convolution cannot carry a
    frequency-dependent width.  The variation of ``w`` across the line is a few
    % there and is negligible relative to ``σ_D``.

    **Grid resolution**: in the Lorentzian-accumulation path (no Doppler, or
    σ_D ≤ 2Δx), components narrower than the grid spacing are undersampled and
    part of the integrated intensity is silently lost; a ``UserWarning`` is
    emitted when ``w(0) < 2Δx``.  Supply ``Ti_ev`` or refine the grid.

    Returns
    -------
    profile_pi : ndarray
        π (Δm = m_u − m_l = 0) polarization component.
    profile_sig_plus : ndarray
        σ+ (Δm = +1, blue-shifted at B > 0; dipole channel q = m_l − m_u = −1)
        polarization component.
    profile_sig_minus : ndarray
        σ− (Δm = −1, red-shifted; channel q = +1) polarization component.

    Notes
    -----
    Observable intensity at angle θ to B:

        I(θ) = I_π sin²θ + ½(I_σ+ + I_σ−)(1 + cos²θ)

    Transverse (θ = 90°): I_π + ½(I_σ+ + I_σ−).
    Along B (θ = 0°):     I_σ+ + I_σ−.
    Angle-averaged:       ⅔ (I_π + I_σ+ + I_σ−)
    (both sin²θ and ½(1+cos²θ) average to ⅔ over the sphere; for the isotropic
    case I_π = I_σ± = I this gives I(θ) = 2I at every angle, as it must).
    """
    # 1. Get microfield grid and weights
    fields, f_weights = microfield_quadrature(Ne_m3, Te_ev, num_points=num_f,
                                              max_beta=max_beta, use_screening=use_screening)
    
    # 2. Get angular integration points (Gauss-Legendre on mu = cos(theta) from 0 to 1)
    mu_points, mu_weights = np.polynomial.legendre.leggauss(num_mu)
    mu_points = 0.5 * (mu_points + 1.0)
    mu_weights = 0.5 * mu_weights
    
    D_q_uncoupled = _uncoupled_dipole_matrices(n_u, n_l, Z)

    # Precompute field-independent matrices (only depend on n, Z, B — not on microfield F).
    # H_atom is rebuilt identically at every quadrature point otherwise.
    # M_z/M_x templates let V_E = Fz*M_z + Fx*M_x without a Python double-loop each time.
    H_atom_u = build_hamiltonian(n_u, Z, B, quadratic_zeeman, fine_structure, A,
                                 use_empirical_data=use_empirical_data, atom=atom)
    H_atom_l = build_hamiltonian(n_l, Z, B, quadratic_zeeman, fine_structure, A,
                                 use_empirical_data=use_empirical_data, atom=atom)
    H_atom_u = H_atom_u.copy()
    H_atom_l = H_atom_l.copy()
    M_z_u, M_x_u = _stark_templates(n_u, Z)
    M_z_l, M_x_l = _stark_templates(n_l, Z)

    if use_empirical_data:
        # H_atom_{u,l} are in cm⁻¹ (NIST convention, incl. Lamb shift); scale the
        # (always-eV) Stark templates to cm⁻¹/(V/m) so Fz*M_z + Fx*M_x stays
        # unit-consistent with H_atom in the loop — mirrors the identical
        # ev_to_cm scaling in solve_starkzee.
        ev_to_cm = energy_ev_to_wavenumber_cm(1.0)
        M_z_u, M_x_u = M_z_u * ev_to_cm, M_x_u * ev_to_cm
        M_z_l, M_x_l = M_z_l * ev_to_cm, M_x_l * ev_to_cm
        # Center near 0 using the mean of the empirical levels (well-conditions
        # the eigensolve; equivalent role to En_u/En_l in the analytic branch).
        En_u = H_atom_u.diagonal().real.mean()
        En_l = H_atom_l.diagonal().real.mean()
    else:
        En_u = - (Z**2) * reduced_mass_rydberg_ev(Z, A) / (n_u**2)
        En_l = - (Z**2) * reduced_mass_rydberg_ev(Z, A) / (n_l**2)
    # Subtract shell unperturbed energies from diagonal to keep H_atom_u and H_atom_l
    # well-conditioned (norm ~1e-3 eV instead of ~3 eV, or the cm⁻¹ equivalent)
    # before diagonalizing in the loop.
    for i in range(H_atom_u.shape[0]):
        H_atom_u[i, i] -= En_u
    for i in range(H_atom_l.shape[0]):
        H_atom_l[i, i] -= En_l

    # Output arrays
    profile_pi = np.zeros_like(energies_ev)
    profile_sig_plus = np.zeros_like(energies_ev)
    profile_sig_minus = np.zeros_like(energies_ev)
    
    sigma_D = None
    if Ti_ev is not None:
        from starkzee.utils import species_to_ZA
        _, A_species = species_to_ZA(species)
        mc2_ev = A_species * _M_P * _C_LIGHT**2 / _E_CHARGE
        sigma_D = np.mean(energies_ev) * np.sqrt(Ti_ev / mc2_ev)

    # Grid spacing — drives the adaptive Doppler strategy.
    _dx = abs(energies_ev[1] - energies_ev[0])
    # σ_D > 2 dx → Gaussian is well-resolved on the grid: accumulate Gaussians
    # in the loop and apply the Lorentzian via FFT after.  Correct even when
    # w ≪ dx (avoids undersampled-Lorentzian amplitude errors at low density).
    # σ_D ≤ 2 dx → Gaussian aliases → accumulate Lorentzians in the loop and
    # apply the Gaussian via FFT after (needed for coarse grids / wide windows).
    _gaussian_loop = sigma_D is not None and sigma_D > 2.0 * _dx
    if _gaussian_loop:
        _two_sigma2 = 2.0 * sigma_D**2
        _gauss_norm = 1.0 / (sigma_D * _SQRT_2PI)

    # Natural linewidth: ħ(Γ_u + Γ_l)/2, summing Einstein A over all decay channels.
    # This is the physically correct minimum Lorentzian half-width — it replaces
    # the arbitrary numerical floor and ensures correct behavior at low Ne or high B.
    gamma_upper = sum(einstein_a(n_u, k, Z) for k in range(1, n_u))
    gamma_lower = sum(einstein_a(n_l, k, Z) for k in range(1, n_l)) if n_l > 1 else 0.0
    w_natural_ev = _HBAR * (gamma_upper + gamma_lower) / 2.0 / _E_CHARGE

    # Gross-structure line center — reference for transition energies and for
    # the GBK detuning axis of the pointwise electron width.  Must equal
    # En_u − En_l (the diagonal shift subtracted from H_atom_u/H_atom_l above)
    # so that dE_shifted + E0_line recovers the true absolute transition energy
    # regardless of which reference (analytic Rydberg or empirical-level mean)
    # En_u/En_l were centered on.
    if use_empirical_data:
        E0_line = float(wavenumber_cm_to_energy_ev(En_u - En_l))
        # In-loop eigenvalues (and hence dE_shifted) are in cm⁻¹; convert to eV.
        _shift_to_ev = float(wavenumber_cm_to_energy_ev(1.0))
    else:
        E0_line = (Z**2) * reduced_mass_rydberg_ev(Z, A) * (1.0/n_l**2 - 1.0/n_u**2)
        _shift_to_ev = 1.0

    # On-resonance width — used for the FFT Lorentzian step (both paths when
    # Doppler is active) and as the scalar w for frequency_dependent_width=False.
    # Keep the electron part separate so the operator path can rescale it per SDT.
    w_resonance_e = electron_impact_width_model(0.0, Ne_m3, Te_ev, B, Z, n=n_u,
                                                electron_model=electron_model)
    w_resonance = w_resonance_e + w_natural_ev

    # Pointwise electron width w_e(E − E0) on the observation grid (PPPB Φ(Δω),
    # Ferri et al. 2022 Eq. 19: Δω is the detuning from line center).  Computed
    # once per profile call; shared by every quadrature point and transition.
    if frequency_dependent_width and not _gaussian_loop:
        _w_e_grid = electron_impact_width_model(
            energies_ev - E0_line, Ne_m3, Te_ev, B, Z, n=n_u,
            electron_model=electron_model)[:, np.newaxis]

    # Undersampled-Lorentzian guard: in the Lorentzian-accumulation path,
    # components narrower than the grid spacing lose integrated intensity
    # (trapezoid mass simply falls between grid points).
    if not _gaussian_loop and w_resonance < 2.0 * _dx:
        warnings.warn(
            f"Lorentzian half-width at line center ({w_resonance:.3e} eV) is below "
            f"twice the grid spacing (dx = {_dx:.3e} eV); part of the integrated "
            "line intensity will be lost to undersampling. Supply Ti_ev (Doppler) "
            "or refine the energy grid.",
            UserWarning, stacklevel=2)

    # Main integration loop
    for fi, f_weight in zip(fields, f_weights):
        if f_weight <= 1e-15:
            continue
            
        for mu, mu_weight in zip(mu_points, mu_weights):
            weight = f_weight * mu_weight
            if weight <= 1e-15:
                continue
                
            Fz = fi * mu
            Fx = fi * np.sqrt(1.0 - mu**2)

            # Diagonalize using precomputed H_atom and Stark templates
            sz_energies_u, sz_vectors_u = np.linalg.eigh(H_atom_u + Fz * M_z_u + Fx * M_x_u)
            sz_energies_l, sz_vectors_l = np.linalg.eigh(H_atom_l + Fz * M_z_l + Fx * M_x_l)

            # Electron-impact operator diagonal: per-upper-state ⟨k|r²|k⟩/⟨r²⟩_avg
            # (c_k = 0 → ZEST operator). Off-diagonal (→ PPPB c_k) not implemented.
            if electron_operator:
                r2_scale_u = electron_impact_r2_scaling(sz_vectors_u, n_u, Z)
            
            # Compute all three dipole intensity matrices; dE is shared across q.
            V_l_adj = sz_vectors_l.conj().T
            # Compute transition energies using high-precision shifted eigenvalues
            # and add the gross structure line center back. This prevents catastrophic
            # cancellation from subtracting large energy levels directly.
            dE_shifted = sz_energies_u[np.newaxis, :] - sz_energies_l[:, np.newaxis]
            # _shift_to_ev converts cm⁻¹ → eV in the empirical branch (1.0 otherwise).
            dE = dE_shifted * _shift_to_ev + E0_line

            I_pi = np.abs(V_l_adj @ D_q_uncoupled[ 0] @ sz_vectors_u)**2
            I_sp = np.abs(V_l_adj @ D_q_uncoupled[-1] @ sz_vectors_u)**2
            I_sm = np.abs(V_l_adj @ D_q_uncoupled[ 1] @ sz_vectors_u)**2

            # Union of active transitions — compute kernel once for all q.
            mask = (I_pi > 1e-12) | (I_sp > 1e-12) | (I_sm > 1e-12)
            if np.any(mask):
                act_dE    = dE[mask]
                detuning  = energies_ev[:, np.newaxis] - act_dE[np.newaxis, :]

                if _gaussian_loop:
                    kernel = np.exp(-detuning**2 / _two_sigma2) * _gauss_norm
                else:
                    if frequency_dependent_width:
                        # Pointwise w(E − E0): column vector over the observation
                        # grid, shared by all transitions (PPPB Φ(Δω) convention).
                        w_e = _w_e_grid
                    else:
                        w_e = w_resonance_e
                    if electron_operator:
                        # Rescale the electron width per SDT by its upper-state ⟨r²⟩.
                        # r2_scale_u is per upper eigenstate (column of dE); broadcast over
                        # the lower index (rows) then select the active transitions.
                        # Broadcasting: (n_E, 1) or scalar × (n_act,) → per-point, per-SDT.
                        scale = np.broadcast_to(r2_scale_u[np.newaxis, :], dE.shape)[mask]
                        w_e = w_e * scale
                    w = w_e + w_natural_ev
                    kernel = (w / np.pi) / (detuning**2 + w**2)

                profile_pi        += weight * (kernel @ I_pi[mask])
                profile_sig_plus  += weight * (kernel @ I_sp[mask])
                profile_sig_minus += weight * (kernel @ I_sm[mask])

    # Post-loop FFT to complete the Voigt when Doppler is active.
    # Zero-pad to 2N so the circular convolution approximates a linear one,
    # eliminating the periodic wrap-around that otherwise creates a DC floor in
    # the far wings (the long Lorentzian tail from one side of the grid
    # folds back onto the other in a naive N-point circular FFT).
    if sigma_D is not None:
        N = len(energies_ev)
        N_pad = 2 * N
        k = np.fft.rfftfreq(N_pad, d=_dx)
        if _gaussian_loop:
            fft_filter = np.exp(-2.0 * np.pi * k * w_resonance)
        else:
            fft_filter = np.exp(-2.0 * np.pi**2 * sigma_D**2 * k**2)
        _padded = np.zeros(N_pad)
        for prof in (profile_pi, profile_sig_plus, profile_sig_minus):
            _padded[:N] = prof
            _padded[N:] = 0.0
            conv = np.fft.irfft(np.fft.rfft(_padded) * fft_filter, n=N_pad)
            prof[:] = conv[:N]

    return profile_pi, profile_sig_plus, profile_sig_minus


def discrete_transitions(n_u, n_l, Z, B, Fz=0.0, Fx=0.0,
                         quadratic_zeeman=True, fine_structure=True,
                         min_strength=0.0, A=1, radial_method=None,
                         use_empirical_data=False, atom="H"):
    """Return all discrete Stark-Zeeman dipole transitions at a single field configuration.

    Diagonalizes the Stark-Zeeman Hamiltonian for both shells and enumerates every
    (upper eigenstate i, lower eigenstate j, polarization q) triplet with
    non-zero dipole matrix element squared.

    Parameters
    ----------
    n_u, n_l : int
        Upper and lower principal quantum numbers.
    Z : int
        Nuclear charge.
    B : float
        Magnetic field [T].
    Fz : float, optional
        Electric field component along B [V m⁻¹] (default 0).
    Fx : float, optional
        Electric field component perpendicular to B [V m⁻¹] (default 0).
    quadratic_zeeman : bool, optional
        Include diamagnetic Zeeman term (default True).
    fine_structure : bool, optional
        Include mass-velocity + Darwin corrections (default True).
    min_strength : float, optional
        Discard transitions with dipole strength abs(d_q)² < min_strength [a₀²] (default 0).
    A : int, optional
        Atomic mass number of the emitter (1 = H, 2 = D, 3 = T).  Sets the
        reduced-mass Rydberg used for the absolute level energies (default 1).
    radial_method : {"gordon", "quad", None}, optional
        Backend for the radial dipole elements.  ``None`` (default) uses the
        module-level ``RADIAL_DIPOLE_METHOD``
        (``"gordon"`` exact closed form by default; ``"quad"`` is the numerical
        fallback).  The two agree to ~1e-15.
    use_empirical_data : bool, optional
        Use NIST empirical level energies (default False); forwarded to
        :func:`solve_starkzee`.
    atom : str, optional
        Atom identifier for empirical data (default ``"H"``); forwarded to
        :func:`solve_starkzee`.

    Returns
    -------
    dict with five equal-length arrays sorted by transition energy:

    ``energy_ev``
        Transition energy E_upper_i − E_lower_j [eV].
    ``q``
        Polarization integer q = m_l − m_u: 0 = π; **q = −1 is σ+**
        (Δm = m_u − m_l = +1, blue-shifted at B > 0); **q = +1 is σ−**
        (red-shifted).
    ``strength``
        abs(d_q(i→j))² [a₀²].  Summed over all transitions equals
        :func:`~starkzee.radiator.line_strength` (unitary invariance).
    ``upper_idx``
        Upper eigenstate index (0 … 2n_u²−1).
    ``lower_idx``
        Lower eigenstate index (0 … 2n_l²−1).
    """
    evals_u, evecs_u = solve_starkzee(
        n_u, Z, B, Fz, Fx, quadratic_zeeman, fine_structure, A,
        use_empirical_data=use_empirical_data, atom=atom)
    evals_l, evecs_l = solve_starkzee(
        n_l, Z, B, Fz, Fx, quadratic_zeeman, fine_structure, A,
        use_empirical_data=use_empirical_data, atom=atom)

    D_q = _uncoupled_dipole_matrices(n_u, n_l, Z, method=radial_method)
    dim_l, dim_u = D_q[0].shape

    # Transition energy E_upper_i - E_lower_j is shared across polarizations.
    dE = evals_u[np.newaxis, :] - evals_l[:, np.newaxis]   # (dim_l, dim_u)

    energies, q_vals, strengths, up_idx, lo_idx = [], [], [], [], []
    for q in (0, -1, 1):
        mixed = evecs_l.conj().T @ D_q[q] @ evecs_u   # (dim_l, dim_u)
        strength = np.abs(mixed) ** 2
        j_idx, i_idx = np.nonzero(strength > min_strength)
        if j_idx.size == 0:
            continue
        energies.append(dE[j_idx, i_idx])
        strengths.append(strength[j_idx, i_idx])
        q_vals.append(np.full(i_idx.shape, q, dtype=int))
        up_idx.append(i_idx)
        lo_idx.append(j_idx)

    if energies:
        energies = np.concatenate(energies)
        strengths = np.concatenate(strengths)
        q_vals = np.concatenate(q_vals)
        up_idx = np.concatenate(up_idx)
        lo_idx = np.concatenate(lo_idx)
    else:
        energies = np.array([])
        strengths = np.array([])
        q_vals = np.array([], dtype=int)
        up_idx = np.array([], dtype=int)
        lo_idx = np.array([], dtype=int)

    order = np.argsort(energies)
    return {
        'energy_ev':  energies[order],
        'q':          q_vals[order].astype(int),
        'strength':   strengths[order],
        'upper_idx':  up_idx[order].astype(int),
        'lower_idx':  lo_idx[order].astype(int),
    }

