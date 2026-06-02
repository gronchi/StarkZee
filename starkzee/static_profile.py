# PPP Component: Optimized Stark-Zeeman broadening and line profile calculations for starkzee

import numpy as np
from starkzee.utils import A0
from starkzee.atomic_hamiltonian import (
    build_hamiltonian, build_basis, angular_dipole_element, radial_dipole,
    _uncoupled_dipole_matrices,
)
from starkzee.microfield import microfield_quadrature
from starkzee.broadening import electron_impact_width

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
    combinations for Fx are provided by :func:`angular_dipole_element`.

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
    atomic/magnetic Hamiltonian :func:`build_hamiltonian` and
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
        Orthonormal eigenstates as columns, in the canonical |n, l, m_l, m_s⟩ basis.
    """
    H_atom = build_hamiltonian(n, Z, B, quadratic_zeeman, fine_structure, A)
    V_E = build_stark_matrix(n, Z, Fz, Fx)
    H_total = H_atom + V_E

    eigenvalues, eigenvectors = np.linalg.eigh(H_total)
    return eigenvalues, eigenvectors

def calculate_static_profile(n_u, n_l, Z, B, Ne_m3, Te_ev, energies_ev,
                                     num_f=20, num_mu=6, use_screening=True,
                                     quadratic_zeeman=True, fine_structure=True,
                                     frequency_dependent_width=True, A=1):
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
        Use the frequency-dependent GBK electron-impact width; if False use the
        on-resonance value at all detunings (default True).

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

    # Output arrays
    profile_pi = np.zeros_like(energies_ev)
    profile_sig_plus = np.zeros_like(energies_ev)
    profile_sig_minus = np.zeros_like(energies_ev)
    
    # Pre-calculate electron impact width (once per line calculation to save massive compute)
    # The 1e-10 eV floor prevents 0/0 in the Lorentzian at exactly zero density;
    # it must be negligible compared to any physical width (W_e ∝ Ne).
    if frequency_dependent_width:
        # Cover a grid that is guaranteed to span the maximum possible detunings
        max_energy_span = np.max(energies_ev) - np.min(energies_ev)
        grid_limit = max(10.0 * max_energy_span, 10.0)
        w_grid_x = np.linspace(-grid_limit, grid_limit, 2000)
        w_grid_y = electron_impact_width(w_grid_x, Ne_m3, Te_ev, B, Z, n=n_u) + 1e-10
    else:
        w_resonance = electron_impact_width(0.0, Ne_m3, Te_ev, B, Z, n=n_u) + 1e-10
        
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
            
            # Diagonalize upper and lower states under microfield and magnetic field
            sz_energies_u, sz_vectors_u = solve_starkzee(n_u, Z, B, Fz, Fx, quadratic_zeeman, fine_structure, A)
            sz_energies_l, sz_vectors_l = solve_starkzee(n_l, Z, B, Fz, Fx, quadratic_zeeman, fine_structure, A)
            
            # Vectorized mixed dipole matrix calculation
            V_l_adj = sz_vectors_l.conj().T
            
            # Trans transition energies: dE = E_upper - E_lower
            dE = sz_energies_u[np.newaxis, :] - sz_energies_l[:, np.newaxis]
            
            for q, profile in zip([0, -1, 1], [profile_pi, profile_sig_plus, profile_sig_minus]):
                mixed_D = V_l_adj @ D_q_uncoupled[q] @ sz_vectors_u
                intensities = np.abs(mixed_D)**2
                
                # Vectorized accumulation using broadcasting
                mask = intensities > 1e-12
                if np.any(mask):
                    act_intensities = intensities[mask]
                    act_dE = dE[mask]
                    
                    detuning = energies_ev[:, np.newaxis] - act_dE[np.newaxis, :]

                    if frequency_dependent_width:
                        w = np.interp(detuning, w_grid_x, w_grid_y)
                    else:
                        w = w_resonance

                    lorentzian = (w / np.pi) / (detuning**2 + w**2)
                    profile += weight * (lorentzian @ act_intensities)
                        
    return profile_pi, profile_sig_plus, profile_sig_minus


def discrete_transitions(n_u, n_l, Z, B, Fz=0.0, Fx=0.0,
                         quadratic_zeeman=True, fine_structure=True,
                         min_strength=0.0):
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
        Discard transitions with \|d_q\|² < min_strength [a₀²] (default 0).

    Returns
    -------
    dict with five equal-length arrays sorted by transition energy:

    ``energy_ev``
        Transition energy E_upper_i − E_lower_j [eV].
    ``q``
        Polarization integer: 0 = π, −1 = σ−, +1 = σ+.
    ``strength``
        \|d_q(i→j)\|² [a₀²].  Summed over all transitions equals
        :func:`~starkzee.atomic_hamiltonian.line_strength` (unitary invariance).
    ``upper_idx``
        Upper eigenstate index (0 … 2n_u²−1).
    ``lower_idx``
        Lower eigenstate index (0 … 2n_l²−1).
    """
    evals_u, evecs_u = solve_starkzee(
        n_u, Z, B, Fz, Fx, quadratic_zeeman, fine_structure)
    evals_l, evecs_l = solve_starkzee(
        n_l, Z, B, Fz, Fx, quadratic_zeeman, fine_structure)

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

