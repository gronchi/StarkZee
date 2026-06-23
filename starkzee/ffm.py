# FFM Component: Optimized Frequency Fluctuation Model (FFM) for any transition in starkzee

import numpy as np
from scipy.constants import hbar as HBAR, e as E_CHARGE, m_p as _M_P, c as _C_LIGHT
from starkzee.radiator import _uncoupled_dipole_matrices, einstein_a
from starkzee.microfield import microfield_quadrature
from starkzee.static_profile import solve_starkzee
from starkzee.broadening import electron_impact_width, electron_impact_width_model

def calculate_ion_fluctuation_rate(Ne_m3, Ti_ev, Z_ion, A_ion):
    """Return the ion fluctuation (jumping) rate ν_i [eV].

    The FFM treats the ion microfield as a stochastic process that switches
    between field configurations at rate ν_i.  This rate is estimated as the
    inverse of the mean time for an ion to cross the inter-ion distance r_i at
    its thermal velocity:

        N_i  = N_e / Z_ion
        r_i  = (3 / 4π N_i)^{1/3}     — mean inter-ion spacing [m]
        v_th = √(2 k_B T_i / m_i)     — most-probable ion speed [m s⁻¹]
        ν_i  = (v_th / r_i) × ħ / e   — converted to eV

    A larger ν_i (high density, high T_i, light ions) pushes the profile
    toward the dynamical (motional) narrowing limit; ν_i → 0 recovers the
    static Holtsmark profile.

    Parameters
    ----------
    Ne_m3 : float
        Electron number density [m⁻³].
    Ti_ev : float
        Ion temperature [eV].
    Z_ion : int
        Ion charge number (used to derive ion density N_i = N_e / Z_ion).
    A_ion : float
        Ion atomic mass number (e.g. 1 for H⁺, 2 for D⁺).

    Returns
    -------
    float
        Ion fluctuation rate ν_i [eV].

    Notes
    -----
    The most-probable speed v_th = √(2 k_B T_i / m_i) is used here.
    Ferri et al. (2022) do not specify the ion v_th prefactor explicitly;
    this should be verified against Calisti et al., Phys. Rev. A 42, 5433 (1990).
    """
    Ni = Ne_m3 / Z_ion
    ri = (3.0 / (4.0 * np.pi * Ni))**(1.0 / 3.0)
    
    m_i = A_ion * _M_P
    v_th = np.sqrt(2.0 * Ti_ev * E_CHARGE / m_i)
    
    nu_rad = v_th / ri
    return nu_rad * HBAR / E_CHARGE

def calculate_ffm_profile(n_u, n_l, Z, B, Ne_m3, Te_ev, Ti_ev, A_ion, energies_ev,
                                  num_f=30, num_mu=10, use_screening=True,
                                  quadratic_zeeman=True, fine_structure=True,
                                  numerical_inversion=False,
                                  use_empirical_data=False, atom="H",
                                  electron_model='pppb', parallel_stark=False,
                                  apply_doppler=True):
    """Compute the dynamical Stark-Zeeman line profile using the Frequency Fluctuation Model.

    The FFM treats the ion microfield as a Markov jump process between
    Stark-dressed field configurations.  At each quadrature point (field
    magnitude F, angle μ = cos θ) the Stark-Zeeman Hamiltonian is diagonalized
    to obtain the dressed-state transition frequencies ω_k and weights d_k.
    These form the Stark-Dressed Transitions (SDTs).  The FFM then solves the
    Markov master equation to obtain a profile that interpolates between the
    quasi-static limit (ν_i → 0, identical to the static profile) and the
    motional-narrowing limit (ν_i → ∞, single Lorentzian).

    **Sherman-Morrison solver** (default, ``numerical_inversion=False``):

        I(ω) = (R²/π) Re [ S(ω) / (1 − ν_i S(ω)) ]

        S(ω) = Σ_k p_k / (ν_i + γ_k + i(ω − ω_k))

    where p_k = d_k² / Σ d_k² are the normalized SDT weights and γ_k is the
    electron-impact half-width.  This analytical result is exact for a
    Markov jump process with uniform jumping rate ν_i and O(N) per frequency
    point.

    **Full matrix inversion** (``numerical_inversion=True``):
    Solves the Liouville-space Markov equation A x = b directly for each
    frequency point, which is O(N³) but handles non-uniform jumping rates or
    more complex correlation structures.

    Parameters
    ----------
    n_u, n_l : int
        Upper and lower principal quantum numbers.
    Z : int
        Nuclear charge.
    B : float
        Magnetic field [T]. B = 0 is fully supported.
    Ne_m3 : float
        Electron number density [m⁻³].
    Te_ev : float
        Electron temperature [eV].  Used for Debye screening and electron
        impact width.
    Ti_ev : float
        Ion (emitter) temperature [eV].  Used both to compute the ion
        fluctuation rate ν_i and, when ``apply_doppler=True``, as the emitter
        thermal velocity for Doppler broadening.
    A_ion : float
        Atomic mass number of the perturbing ion species (e.g. 1 for H⁺, 2 for
        D⁺).  Also used as the emitter mass for the Doppler width (same-species
        assumption).
    energies_ev : array-like
        Photon energies at which to evaluate the profile [eV].
    num_f : int, optional
        Number of microfield quadrature points (default 30).  Ferri et al.
        (2022) recommend ~50; 30 is sufficient for hydrogen but should be
        increased for multi-electron atoms (future: set dynamically by atom type).
    num_mu : int, optional
        Number of Gauss-Legendre points for the field-angle integration
        over μ = cos θ ∈ [0, 1] (default 10).  Ferri et al. (2022) recommend
        ~30; 10 is sufficient for hydrogen but should be increased for
        multi-electron atoms (future: set dynamically by atom type).
    use_screening : bool, optional
        Use the Hooper screened microfield distribution (default True).
    quadratic_zeeman : bool, optional
        Include the diamagnetic quadratic Zeeman term (default True).
    fine_structure : bool, optional
        Include mass-velocity and Darwin corrections (default True).
    numerical_inversion : bool, optional
        If True, solve the full Markov matrix by direct inversion instead of
        the Sherman-Morrison approximation (default False).  Falls back to
        Sherman-Morrison if the matrix is singular.
    use_empirical_data : bool, optional
        Use NIST empirical level energies (default False); forwarded to
        :func:`~starkzee.static_profile.solve_starkzee`.
    atom : str, optional
        Atom identifier for empirical data (default ``"H"``); forwarded to
        :func:`~starkzee.static_profile.solve_starkzee`.
    electron_model : str, optional
        Electron-impact width prescription, forwarded to
        :func:`~starkzee.broadening.electron_impact_width_model`.  ``'pppb'``
        (default) is the PPPB model; ``'zest'``,
        ``'zest-lee'``, ``'zest-dufty'`` select the ZEST model with the GBK, Lee,
        or Dufty RPA G-function.  Note the FFM applies a single resonance width
        to all SDTs.
    parallel_stark : bool, optional
        If True, use the parallel-Stark approximation: only the field component
        parallel to B (Fz = F·μ) enters the Stark Hamiltonian; the perpendicular
        component Fx is set to zero.  This matches the ``parallel_stark=True``
        convention of the ZEST code (Ferri et al. 2022).  Default is False.
    apply_doppler : bool, optional
        If True (default), convolve the accumulated profile with a Gaussian
        thermal Doppler kernel of 1/e half-width
        σ_D = E₀ √(T_i / m_ion c²), using ``Ti_ev`` and ``A_ion``.
        Set to False to obtain the purely Stark-Zeeman-broadened FFM profile
        without Doppler.

    Returns
    -------
    profile_pi : ndarray, shape like *energies_ev*
        π polarization component (Δm = 0).
    profile_sig_plus : ndarray
        σ+ polarization component (Δm = +1).
    profile_sig_minus : ndarray
        σ− polarization component (Δm = −1).

    Notes
    -----
    **Static vs FFM guidance**: use the static profile
    (:func:`~starkzee.static_profile.calculate_static_profile`)
    when B ≥ 50 T or N_e < 10²¹ m⁻³, where ion dynamics are negligible.
    For lower B or higher densities (especially Hβ, Hδ at N_e ≥ 10²³ m⁻³)
    FFM is needed to reproduce the correct central dip depth and peak structure.

    **B = 0 note**: the π/σ decomposition is physically meaningless at B = 0
    (no preferred axis); all three components are equal by symmetry and their
    sum gives the isotropic total profile.

    References
    ----------
    Calisti, A. et al., Phys. Rev. A 42, 5433 (1990). — FFM formulation.
    Ferri, S., Peyrusse, O. & Calisti, A., Matter Radiat. Extremes 7, 015901 (2022).
    """
    # 1. Calculate ion fluctuation rate
    nu_i = calculate_ion_fluctuation_rate(Ne_m3, Ti_ev, Z, A_ion)

    # 2. Get microfield grid and weights
    fields, f_weights = microfield_quadrature(Ne_m3, Te_ev, num_points=num_f, use_screening=use_screening)
    
    # 3. Get angular integration points (Gauss-Legendre)
    mu_points, mu_weights = np.polynomial.legendre.leggauss(num_mu)
    mu_points = 0.5 * (mu_points + 1.0)
    mu_weights = 0.5 * mu_weights
    
    D_q_uncoupled = _uncoupled_dipole_matrices(n_u, n_l, Z)

    # Accumulate Stark-Dressed Transitions (SDTs)
    sdt_list = {0: [], 1: [], -1: []}
    
    for fi, f_weight in zip(fields, f_weights):
        if f_weight <= 1e-15:
            continue
            
        for mu, mu_weight in zip(mu_points, mu_weights):
            weight = f_weight * mu_weight
            if weight <= 1e-15:
                continue
                
            Fz = fi * mu
            Fx = 0.0 if parallel_stark else fi * np.sqrt(1.0 - mu**2)
            
            # Solve combined Stark-Zeeman Hamiltonian for upper and lower states
            sz_energies_u, sz_vectors_u = solve_starkzee(n_u, Z, B, Fz, Fx, quadratic_zeeman, fine_structure,
                                                         use_empirical_data=use_empirical_data, atom=atom)
            sz_energies_l, sz_vectors_l = solve_starkzee(n_l, Z, B, Fz, Fx, quadratic_zeeman, fine_structure,
                                                         use_empirical_data=use_empirical_data, atom=atom)
            
            V_l_adj = sz_vectors_l.conj().T
            dE = sz_energies_u[np.newaxis, :] - sz_energies_l[:, np.newaxis]
            
            # For each polarization q
            for q in [0, 1, -1]:
                mixed_D = V_l_adj @ D_q_uncoupled[q] @ sz_vectors_u
                intensities = np.abs(mixed_D)**2
                
                # Collect non-zero transitions
                j_indices, i_indices = np.where(intensities > 1e-12)
                for j, i in zip(j_indices, i_indices):
                    sdt_list[q].append({
                        "intensity": weight * intensities[j, i],
                        "frequency": dE[j, i]
                    })
                        
    output_profiles = {}
    
    # Total Lorentzian half-width at resonance: electron impact + natural linewidth.
    gamma_k = electron_impact_width_model(0.0, Ne_m3, Te_ev, B, Z, n=n_u,
                                          electron_model=electron_model)
    gamma_upper = sum(einstein_a(n_u, k, Z) for k in range(1, n_u))
    gamma_lower = sum(einstein_a(n_l, k, Z) for k in range(1, n_l)) if n_l > 1 else 0.0
    gamma_k += HBAR * (gamma_upper + gamma_lower) / 2.0 / E_CHARGE
    
    for q in [0, 1, -1]:
        sdts = sdt_list[q]
        if not sdts:
            output_profiles[q] = np.zeros_like(energies_ev)
            continue
            
        intensities = np.array([item["intensity"] for item in sdts])
        frequencies = np.array([item["frequency"] for item in sdts])
        
        r_q_sq = np.sum(intensities)
        if r_q_sq <= 1e-15:
            output_profiles[q] = np.zeros_like(energies_ev)
            continue
            
        normalized_intensities = intensities / r_q_sq
        M = len(energies_ev)
        N = len(sdts)
        
        if numerical_inversion and N > 1:
            # Full Liouville-space Markov mixing matrix numerical inversion
            # M_mj = delta_mj * (1j * (omega - w_j) + nu_i + gamma_j) - nu_i * p_m
            # We solve M X = p, where p is the normalized intensities.
            # The profile is then (r_q_sq / pi) * Re( sum_j X_j ).
            diag_term = 1j * (energies_ev[:, np.newaxis] - frequencies[np.newaxis, :]) + (nu_i + gamma_k) # shape (M, N)
            
            # Build (M, N, N) stacked matrices
            A = (diag_term[:, :, np.newaxis] * np.eye(N)[np.newaxis, :, :] 
                 - nu_i * normalized_intensities[np.newaxis, :, np.newaxis] * np.ones((N, N))[np.newaxis, :, :])
            
            # Construct right-hand side vector: shape (M, N, 1)
            B_rhs = np.broadcast_to(normalized_intensities[np.newaxis, :, np.newaxis], (M, N, 1))
            
            # Solve stacked system in a single NumPy call
            try:
                X = np.linalg.solve(A, B_rhs) # shape (M, N, 1)
                profile = (r_q_sq / np.pi) * np.real(np.sum(X[:, :, 0], axis=1))
                output_profiles[q] = np.maximum(0.0, profile)
            except np.linalg.LinAlgError:
                # Fallback to analytical Sherman-Morrison solver if singular
                numerical_inversion = False
                
        if not numerical_inversion or N <= 1:
            # Analytical Sherman-Morrison solver
            omega_diff = energies_ev[:, np.newaxis] - frequencies[np.newaxis, :]
            S_omega = np.sum(normalized_intensities[np.newaxis, :] / (nu_i + gamma_k + 1j * omega_diff), axis=1)

            numerator = S_omega
            denominator = 1.0 - nu_i * S_omega

            profile = (r_q_sq / np.pi) * np.real(numerator / denominator)
            output_profiles[q] = np.maximum(0.0, profile)

    # Thermal Doppler broadening: FFT convolution with Gaussian of 1/e half-width
    # σ_D = E₀ √(T_i / m_ion c²).  Uses zero-padded rfft to avoid wrap-around
    # artefacts, matching the strategy in calculate_static_profile.
    if apply_doppler and Ti_ev > 0:
        mc2_ev = A_ion * _M_P * (_C_LIGHT ** 2) / E_CHARGE
        sigma_D = np.mean(energies_ev) * np.sqrt(Ti_ev / mc2_ev)
        if sigma_D > 0:
            n_pts = len(energies_ev)
            n_pad = 2 * n_pts
            dx = abs(energies_ev[1] - energies_ev[0])
            k = np.fft.rfftfreq(n_pad, d=dx)
            fft_filter = np.exp(-2.0 * np.pi**2 * sigma_D**2 * k**2)
            padded = np.zeros(n_pad)
            for q in [0, 1, -1]:
                padded[:n_pts] = output_profiles[q]
                padded[n_pts:] = 0.0
                conv = np.fft.irfft(np.fft.rfft(padded) * fft_filter, n=n_pad)
                output_profiles[q] = conv[:n_pts]

    return output_profiles[0], output_profiles[1], output_profiles[-1]
