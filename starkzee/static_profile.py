# PPP Component: Optimized Stark-Zeeman broadening and line profile calculations for starkzee

import numpy as np
from functools import lru_cache
from starkzee.utils import A0, reduced_mass_rydberg_ev
from scipy.constants import hbar as _HBAR, e as _E_CHARGE, m_p as _M_P, c as _C_LIGHT

# Pseudo-Voigt constants
_SQRT_2LN2 = np.sqrt(2.0 * np.log(2.0))
_SQRT_2PI  = np.sqrt(2.0 * np.pi)


def _pseudo_voigt(x, sigma, gamma):
    """Normalized pseudo-Voigt (Thompson et al. 1987), accurate to < 2e-4 of peak.

    Uses only ``exp`` and division — no Faddeeva function — so it is as fast as
    a plain Gaussian while correctly reproducing the Lorentzian far wings.

    Parameters
    ----------
    x : array-like
        Detuning from line center.
    sigma : float
        Gaussian standard deviation (same convention as ``scipy.special.voigt_profile``).
    gamma : float or array-like
        Lorentzian HWHM.  May be scalar or array broadcastable with *x*.
    """
    fG = 2.0 * _SQRT_2LN2 * sigma    # Gaussian FWHM
    fL = 2.0 * gamma                  # Lorentzian FWHM
    f5 = (fG**5 + 2.69269*fG**4*fL + 2.42843*fG**3*fL**2
          + 4.47163*fG**2*fL**3 + 0.07842*fG*fL**4 + fL**5)
    f   = f5 ** 0.2
    eta = 1.36603*(fL/f) - 0.47719*(fL/f)**2 + 0.11116*(fL/f)**3
    hwhm = f * 0.5
    sig  = f / (2.0 * _SQRT_2LN2)
    L = hwhm / (np.pi * (x**2 + hwhm**2))
    G = np.exp(-0.5 * x**2 / sig**2) / (sig * _SQRT_2PI)
    return eta * L + (1.0 - eta) * G
from starkzee.atomic_hamiltonian import (
    build_hamiltonian, build_basis, angular_dipole_element, radial_dipole,
    _uncoupled_dipole_matrices, einstein_a,
)
from starkzee.microfield import microfield_quadrature
from starkzee.broadening import electron_impact_width

@lru_cache(maxsize=None)
def _stark_templates(n, Z):
    """Return the field-independent Stark coupling matrices M_z and M_x [eV/(V/m)].

    V_E(Fz, Fx) = Fz * M_z + Fx * M_x, so these only need to be built once
    per (n, Z) pair regardless of how many microfield quadrature points are used.
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
    combinations for Fx are provided by :func:`~starkzee.atomic_hamiltonian.angular_dipole_element`.

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
                               fine_structure=True, A=1):
    """Diagonalize the combined Stark + Zeeman Hamiltonian for shell n.

    Adds the Stark perturbation :func:`build_stark_matrix` to the
    atomic/magnetic Hamiltonian :func:`~starkzee.atomic_hamiltonian.build_hamiltonian` and
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

    Returns
    -------
    eigenvalues : ndarray, shape (2n²,)
        Energy eigenvalues in ascending order [eV].
    eigenvectors : ndarray, shape (2n², 2n²)
        Orthonormal eigenstates as columns, in the canonical ``|n, l, m_l, m_s⟩`` basis.
    """
    H_atom = build_hamiltonian(n, Z, B, quadratic_zeeman, fine_structure, A)
    V_E = build_stark_matrix(n, Z, Fz, Fx)
    H_total = H_atom + V_E

    eigenvalues, eigenvectors = np.linalg.eigh(H_total)
    return eigenvalues, eigenvectors

def calculate_static_profile(n_u, n_l, Z, B, Ne_m3, Te_ev, energies_ev,
                                     num_f=20, num_mu=6, use_screening=True,
                                     quadratic_zeeman=True, fine_structure=True,
                                     frequency_dependent_width=True, A=1,
                                     Ti_ev=None, species='H'):
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
        Number of microfield quadrature points (default 20).
    num_mu : int, optional
        Number of Gauss-Legendre angle points (default 6).
    use_screening : bool, optional
        Use Hooper screened microfield distribution (default True).
    quadratic_zeeman : bool, optional
        Include diamagnetic (quadratic) Zeeman term (default True).
    fine_structure : bool, optional
        Include mass-velocity and Darwin corrections (Dirac fine structure)
        so that 2s_{1/2} = 2p_{1/2} (default True).
    frequency_dependent_width : bool, optional
        When True (default), evaluate the GBK electron-impact width once per
        discrete transition at its detuning from the gross-structure line center
        ``dE_i − E0``.  This is the physically correct interpretation: each
        Stark-Zeeman component sitting far from line center (e.g. a σ± component
        shifted by strong Zeeman) receives a reduced width, consistent with the
        breakdown of the impact approximation in the far wings.
        When False, use the single on-resonance value ``w(0)`` for every
        transition (faster; valid when Zeeman splitting ≪ ω_c).
    Ti_ev : float, optional
        Ion temperature [eV].  When supplied, Doppler broadening is folded into
        the Lorentzian accumulation as a Voigt profile, eliminating the need for
        a separate post-processing convolution.  Default is ``None`` (bare
        Lorentzian, no Doppler).
    species : str, optional
        Emitting species (``'H'``, ``'D'``, or ``'T'``); used to determine the
        ion mass for the Doppler width.  Only relevant when *Ti_ev* is set.
        Default is ``'H'``.

    Notes on approximations
    -----------------------
    **sigma_D**: the Doppler width is computed as
    ``σ_D = E_mean × sqrt(Ti / mc²)`` where ``E_mean = mean(energies_ev)``.
    Strictly, each transition at energy ``dE_i`` should use ``σ_D(dE_i)``, but
    the fractional error is ``ΔE/E ≈ Δ_Zeeman/E0`` — about 0.03 % for H-alpha
    at 10 T — and is negligible in practice.

    **Lorentzian FFT step**: when Doppler is active, the post-loop Lorentzian
    convolution uses ``w_resonance`` (the on-resonance GBK width) for both
    ``frequency_dependent_width`` settings.  The per-transition variation of
    ``w`` is ~4 % across the line and is negligible relative to ``σ_D``.

    Returns
    -------
    profile_pi : ndarray
        π (Δm = 0) polarization component.
    profile_sig_plus : ndarray
        σ+ (Δm = +1) polarization component.
    profile_sig_minus : ndarray
        σ− (Δm = −1) polarization component.

    Notes
    -----
    Observable intensity at angle θ to B:

        I(θ) = I_π sin²θ + ½(I_σ+ + I_σ−)(1 + cos²θ)

    Transverse (θ = 90°): I_π + ½(I_σ+ + I_σ−).
    Along B (θ = 0°):     I_σ+ + I_σ−.
    Angle-averaged:        ⅔ I_π + ⅓(I_σ+ + I_σ−).
    """
    # 1. Get microfield grid and weights
    fields, f_weights = microfield_quadrature(Ne_m3, Te_ev, num_points=num_f, use_screening=use_screening)
    
    # 2. Get angular integration points (Gauss-Legendre on mu = cos(theta) from 0 to 1)
    mu_points, mu_weights = np.polynomial.legendre.leggauss(num_mu)
    mu_points = 0.5 * (mu_points + 1.0)
    mu_weights = 0.5 * mu_weights
    
    D_q_uncoupled = _uncoupled_dipole_matrices(n_u, n_l, Z)

    # Precompute field-independent matrices (only depend on n, Z, B — not on microfield F).
    # H_atom is rebuilt identically at every quadrature point otherwise.
    # M_z/M_x templates let V_E = Fz*M_z + Fx*M_x without a Python double-loop each time.
    H_atom_u = build_hamiltonian(n_u, Z, B, quadratic_zeeman, fine_structure, A)
    H_atom_l = build_hamiltonian(n_l, Z, B, quadratic_zeeman, fine_structure, A)
    M_z_u, M_x_u = _stark_templates(n_u, Z)
    M_z_l, M_x_l = _stark_templates(n_l, Z)

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

    # Gross-structure line center — used to compute per-transition GBK detunings.
    E0_line = (Z**2) * reduced_mass_rydberg_ev(Z, A) * (1.0/n_l**2 - 1.0/n_u**2)

    # On-resonance width — used for the FFT Lorentzian step (both paths when
    # Doppler is active) and as the scalar w for frequency_dependent_width=False.
    w_resonance = electron_impact_width(0.0, Ne_m3, Te_ev, B, Z, n=n_u) + w_natural_ev
        
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
            
            # Compute all three dipole intensity matrices; dE is shared across q.
            V_l_adj = sz_vectors_l.conj().T
            dE = sz_energies_u[np.newaxis, :] - sz_energies_l[:, np.newaxis]

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
                        w = (electron_impact_width(act_dE - E0_line, Ne_m3, Te_ev, B, Z, n=n_u)
                             + w_natural_ev)
                    else:
                        w = w_resonance
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
                         min_strength=0.0, A=1):
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

    Returns
    -------
    dict with five equal-length arrays sorted by transition energy:

    ``energy_ev``
        Transition energy E_upper_i − E_lower_j [eV].
    ``q``
        Polarization integer: 0 = π, −1 = σ−, +1 = σ+.
    ``strength``
        abs(d_q(i→j))² [a₀²].  Summed over all transitions equals
        :func:`~starkzee.atomic_hamiltonian.line_strength` (unitary invariance).
    ``upper_idx``
        Upper eigenstate index (0 … 2n_u²−1).
    ``lower_idx``
        Lower eigenstate index (0 … 2n_l²−1).
    """
    evals_u, evecs_u = solve_starkzee(
        n_u, Z, B, Fz, Fx, quadratic_zeeman, fine_structure, A)
    evals_l, evecs_l = solve_starkzee(
        n_l, Z, B, Fz, Fx, quadratic_zeeman, fine_structure, A)

    D_q = _uncoupled_dipole_matrices(n_u, n_l, Z)
    dim_l, dim_u = D_q[0].shape

    energies, q_vals, strengths, up_idx, lo_idx = [], [], [], [], []

    for q in [0, -1, 1]:
        mixed = evecs_l.conj().T @ D_q[q] @ evecs_u   # (dim_l, dim_u)
        for i in range(dim_u):
            for j in range(dim_l):
                s = abs(mixed[j, i]) ** 2
                if s > min_strength:
                    energies.append(float(evals_u[i] - evals_l[j]))
                    q_vals.append(q)
                    strengths.append(float(s))
                    up_idx.append(i)
                    lo_idx.append(j)

    order = np.argsort(energies)
    return {
        'energy_ev':  np.array(energies)[order],
        'q':          np.array(q_vals,   dtype=int)[order],
        'strength':   np.array(strengths)[order],
        'upper_idx':  np.array(up_idx,   dtype=int)[order],
        'lower_idx':  np.array(lo_idx,   dtype=int)[order],
    }

