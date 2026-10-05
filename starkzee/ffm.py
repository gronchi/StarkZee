# FFM Component: Optimized Frequency Fluctuation Model (FFM) for any transition in starkzee

from collections.abc import MutableMapping

import numpy as np
from scipy.constants import hbar as HBAR, e as E_CHARGE, m_p as _M_P, c as _C_LIGHT
from starkzee.radiator import (
    _uncoupled_dipole_matrices, einstein_a, natural_decay_rates,
)
from starkzee.utils import reduced_mass_rydberg_ev
from starkzee.microfield import microfield_quadrature
from starkzee.static_profile import solve_starkzee, line_reference_energy
from starkzee.convolutions import uniform_energy_grid
from starkzee.broadening import (
    electron_impact_collision_coefficient, electron_impact_width_model,
)
from starkzee.collision import ppp_complex_sdts

def calculate_ion_fluctuation_rate(Ne_m3, Ti_ev, Z_ion, A_ion,
                                   A_emitter=None, model='zest'):
    """Return the ion fluctuation (jumping) rate ν_i [eV].

    The FFM treats the ion microfield as a stochastic process that switches
    between field configurations at rate ν_i.  This rate is estimated as the
    inverse of the mean time for an ion to cross the inter-ion distance r_i at
    its thermal velocity:

        N_i  = N_e / Z_ion
        r_i  = (3 / 4π N_i)^{1/3}     — mean inter-ion spacing [m]
        v_th = √(2 k_B T_i / m_i)     — ZEST most-probable speed [m s⁻¹]
        ν_i  = (v_th / r_i) × ħ / e   — converted to eV

    With ``model='ppp'``, the PPP manual Appendix C, Eq. (C5), is used
    instead: ``v_th = sqrt(k_B T_i / m_mu)``, where ``m_mu`` is the
    emitter--perturber reduced mass.

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
        Perturbing-ion atomic mass number (e.g. 1 for H⁺, 2 for D⁺).
    A_emitter : float or None, optional
        Emitter atomic mass number. Required by ``model='ppp'``; ignored by
        ``model='zest'``.
    model : {'zest', 'ppp'}, optional
        ``'zest'`` (default) retains ZEST Eq. (20). ``'ppp'`` uses the
        relative-speed estimate in PPP manual Eq. (C5).

    Returns
    -------
    float
        Ion fluctuation rate ν_i [eV].

    Notes
    -----
    For equal emitter and perturber masses the two formulae coincide because
    ``m_mu=m_i/2``. They differ for a heavy radiator in a light-ion plasma.
    """
    if (not np.isfinite(Ne_m3) or Ne_m3 <= 0 or not np.isfinite(Ti_ev)
            or Ti_ev < 0 or not np.isfinite(Z_ion) or Z_ion <= 0
            or not np.isfinite(A_ion) or A_ion <= 0):
        raise ValueError("Density, charge and perturber mass must be positive; Ti_ev must be nonnegative.")
    Ni = Ne_m3 / Z_ion
    ri = (3.0 / (4.0 * np.pi * Ni))**(1.0 / 3.0)

    model_key = str(model).lower()
    m_perturber = A_ion * _M_P
    if model_key == 'zest':
        v_th = np.sqrt(2.0 * Ti_ev * E_CHARGE / m_perturber)
    elif model_key == 'ppp':
        if A_emitter is None or not np.isfinite(A_emitter) or A_emitter <= 0:
            raise ValueError("model='ppp' requires a positive finite A_emitter.")
        m_emitter = A_emitter * _M_P
        reduced_mass = m_emitter * m_perturber / (m_emitter + m_perturber)
        v_th = np.sqrt(Ti_ev * E_CHARGE / reduced_mass)
    else:
        raise ValueError("model must be 'zest' or 'ppp'.")
    
    nu_rad = v_th / ri
    return nu_rad * HBAR / E_CHARGE


def _complex_ffm_profile_analytical(
        energies_ev, frequencies, gamma_k, strengths, nu_i,
        max_chunk_elements=2_000_000):
    """Evaluate the discrete complex-strength FFM closed form.

    This is Calisti et al. (2010), Eq. (23), or the PPP manual's
    generalized form of the Sherman--Morrison reduction.  The real parts of
    The normalized moduli ``|Re(strengths)|`` define the stationary Markov
    probabilities, while the original signed complex strengths enter the
    radiative numerator.  This preserves interference in the spectrum while
    supplying a nonnegative stationary distribution to the Markov reduction.
    """
    energies = np.asarray(energies_ev, dtype=float)
    omega = np.asarray(frequencies, dtype=float)
    gamma = np.asarray(gamma_k, dtype=float)
    amplitudes = np.asarray(strengths, dtype=complex)
    if omega.ndim != 1 or gamma.shape != omega.shape or amplitudes.shape != omega.shape:
        raise ValueError("frequencies, gamma_k, and strengths must be matching vectors.")
    if max_chunk_elements < 1:
        raise ValueError("max_chunk_elements must be positive.")

    total_strength = float(np.sum(amplitudes.real))
    probability_norm = float(np.sum(np.abs(amplitudes.real)))
    if total_strength <= 0 or probability_norm <= 0:
        return np.zeros_like(energies)
    probability = np.abs(amplitudes.real) / probability_norm
    complex_weight = amplitudes / total_strength

    # Bound temporary storage for large quadrature ensembles. A D-gamma
    # 60x11 run can contain roughly 264,000 modes; materializing the complete
    # energy-by-mode denominator would otherwise require several gigabytes.
    rows_per_chunk = max(1, max_chunk_elements // max(1, len(omega)))
    profile = np.empty_like(energies)
    pole_real = nu_i + gamma
    for start in range(0, len(energies), rows_per_chunk):
        stop = min(start + rows_per_chunk, len(energies))
        energy_block = energies[start:stop]
        denominator_k = (
            pole_real[np.newaxis, :]
            + 1j * (energy_block[:, np.newaxis] - omega[np.newaxis, :])
        )
        numerator_sum = np.sum(
            complex_weight[np.newaxis, :] / denominator_k, axis=1)
        probability_sum = np.sum(
            probability[np.newaxis, :] / denominator_k, axis=1)
        profile[start:stop] = (
            (total_strength / np.pi)
            * np.real(numerator_sum / (1.0 - nu_i * probability_sum))
        )
    return profile


def summarize_complex_sdt_probabilities(
        frequencies, gamma_k, strengths, *, relative_tolerance=1e-12,
        max_reported_modes=20):
    """Summarize signed residues used by the modulus-probability FFM closure.

    The returned values are plain Python scalars and lists so callers can
    serialize them directly.  ``reported_negative_modes`` is limited to the
    most negative residues; its neighbor distance is measured to the closer
    adjacent mode after sorting by frequency and includes both frequency and
    homogeneous-width separation.
    """
    omega = np.asarray(frequencies, dtype=float)
    gamma = np.asarray(gamma_k, dtype=float)
    amplitudes = np.asarray(strengths, dtype=complex)
    if (omega.ndim != 1 or gamma.shape != omega.shape
            or amplitudes.shape != omega.shape):
        raise ValueError(
            "frequencies, gamma_k, and strengths must be matching vectors.")
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be nonnegative.")
    if max_reported_modes < 0:
        raise ValueError("max_reported_modes must be nonnegative.")

    real = amplitudes.real
    signed_sum = float(np.sum(real))
    absolute_sum = float(np.sum(np.abs(real)))
    tolerance = relative_tolerance * absolute_sum
    negative_indices = np.flatnonzero(real < -tolerance)
    negative_absolute_strength = float(np.sum(np.abs(real[negative_indices])))

    nearest_distances = np.full(len(real), np.nan)
    if len(real) > 1:
        order = np.argsort(omega, kind="stable")
        poles = omega[order] - 1j * gamma[order]
        adjacent = np.abs(np.diff(poles))
        sorted_nearest = np.full(len(real), np.inf)
        sorted_nearest[1:] = np.minimum(sorted_nearest[1:], adjacent)
        sorted_nearest[:-1] = np.minimum(sorted_nearest[:-1], adjacent)
        nearest_distances[order] = sorted_nearest

    ranked = negative_indices[np.argsort(real[negative_indices])]
    reported_modes = []
    for index in ranked[:max_reported_modes]:
        reported_modes.append({
            "index": int(index),
            "a_k": float(real[index]),
            "c_k": float(amplitudes.imag[index]),
            "frequency_ev": float(omega[index]),
            "gamma_ev": float(gamma[index]),
            "nearest_frequency_neighbor_pole_distance_ev": float(
                nearest_distances[index]),
        })

    return {
        "mode_count": int(len(real)),
        "signed_strength_sum": signed_sum,
        "absolute_strength_sum": absolute_sum,
        "negative_mode_count": int(len(negative_indices)),
        "negative_absolute_strength": negative_absolute_strength,
        "negative_absolute_fraction": (
            negative_absolute_strength / absolute_sum
            if absolute_sum > 0 else 0.0),
        "minimum_a_over_signed_sum": (
            float(np.min(real) / signed_sum)
            if len(real) and signed_sum != 0 else None),
        "reported_negative_modes": reported_modes,
    }


def group_complex_sdts(frequencies, gamma_k, strengths,
                       frequency_tolerance_ev, width_tolerance_ev=None):
    """Group nearby complex SDTs while preserving their first pole moment.

    Modes are sorted by frequency and greedily placed into clusters whose
    total frequency and width spans do not exceed the supplied tolerances.
    Each cluster has residue ``A = sum(A_k)`` and complex pole
    ``z = sum(A_k z_k) / A``, where ``z_k = omega_k - i gamma_k``.  A cluster
    with a numerically vanishing total residue is left ungrouped because its
    moment-defined pole would be singular.

    Returns
    -------
    grouped_frequencies, grouped_widths, grouped_strengths, group_sizes
        One-dimensional arrays ordered by grouped frequency.
    """
    omega = np.asarray(frequencies, dtype=float)
    gamma = np.asarray(gamma_k, dtype=float)
    amplitudes = np.asarray(strengths, dtype=complex)
    if (omega.ndim != 1 or gamma.shape != omega.shape
            or amplitudes.shape != omega.shape):
        raise ValueError(
            "frequencies, gamma_k, and strengths must be matching vectors.")
    if frequency_tolerance_ev <= 0:
        raise ValueError("frequency_tolerance_ev must be positive.")
    if width_tolerance_ev is None:
        width_tolerance_ev = frequency_tolerance_ev
    if width_tolerance_ev <= 0:
        raise ValueError("width_tolerance_ev must be positive.")
    if not len(omega):
        return omega.copy(), gamma.copy(), amplitudes.copy(), np.array([], int)

    order = np.argsort(omega, kind="stable")
    clusters = []
    current = [int(order[0])]
    frequency_min = frequency_max = omega[order[0]]
    width_min = width_max = gamma[order[0]]
    for raw_index in order[1:]:
        index = int(raw_index)
        new_frequency_min = min(frequency_min, omega[index])
        new_frequency_max = max(frequency_max, omega[index])
        new_width_min = min(width_min, gamma[index])
        new_width_max = max(width_max, gamma[index])
        if (new_frequency_max - new_frequency_min <= frequency_tolerance_ev
                and new_width_max - new_width_min <= width_tolerance_ev):
            current.append(index)
            frequency_min, frequency_max = new_frequency_min, new_frequency_max
            width_min, width_max = new_width_min, new_width_max
        else:
            clusters.append(current)
            current = [index]
            frequency_min = frequency_max = omega[index]
            width_min = width_max = gamma[index]
    clusters.append(current)

    grouped_frequencies = []
    grouped_widths = []
    grouped_strengths = []
    group_sizes = []
    residue_scale = max(float(np.sum(np.abs(amplitudes))), 1.0)
    for cluster in clusters:
        indices = np.asarray(cluster, dtype=int)
        total = np.sum(amplitudes[indices])
        if abs(total) <= 1e-14 * residue_scale and len(indices) > 1:
            for index in indices:
                grouped_frequencies.append(omega[index])
                grouped_widths.append(gamma[index])
                grouped_strengths.append(amplitudes[index])
                group_sizes.append(1)
            continue
        pole = np.sum(
            amplitudes[indices] * (omega[indices] - 1j * gamma[indices])) / total
        grouped_frequencies.append(float(pole.real))
        grouped_widths.append(float(-pole.imag))
        grouped_strengths.append(total)
        group_sizes.append(len(indices))

    grouped_order = np.argsort(grouped_frequencies, kind="stable")
    return (
        np.asarray(grouped_frequencies)[grouped_order],
        np.asarray(grouped_widths)[grouped_order],
        np.asarray(grouped_strengths)[grouped_order],
        np.asarray(group_sizes, dtype=int)[grouped_order],
    )

def calculate_ffm_profile(n_u, n_l, Z, B, Ne_m3, Te_ev, Ti_ev, A_ion, energies_ev,
                                  num_f=30, num_mu=10, max_beta=10.0, use_screening=True,
                                  quadratic_zeeman=True, fine_structure=True,
                                  numerical_inversion=False,
                                  use_empirical_data=True, atom="H",
                                  electron_model='pppb', parallel_stark=False,
                                  apply_doppler=True, sdt_bin_tol=None,
                                  sdt_frequency_dependent_width=False,
                                  microfield_model=None, custom_table_path=None,
                                  charged=None, emitter_charge=None, Z_bar=1.0,
                                  A_perturber=None, fluctuation_rate_model='zest',
                                  species_charges=None,
                                  species_concentrations=None,
                                  natural_width_mode='state_resolved',
                                  electron_interference=False,
                                  interference_diagnostics=None,
                                  interference_group_tolerance_ev=None,
                                  interference_group_width_tolerance_ev=None,
                                  interference_group_profile_rtol=1e-3):
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
    frequency point, which is O(N³). The current implementation constructs the
    same uniform rank-one jumping matrix as the analytical solver, so this mode
    is an algebraic cross-check rather than a more general dynamics model.

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
        Emitter atomic mass number. It sets the Doppler width and reduced-mass
        Rydberg. For backward compatibility, it is also the perturber mass when
        ``A_perturber`` is omitted.
    A_perturber : float or None, optional
        Background perturbing-ion mass number. Use this for, e.g., Ar XVII in
        a proton plasma. ``None`` retains the same-species assumption.
    fluctuation_rate_model : {'zest', 'ppp'}, optional
        ZEST most-probable perturber speed (default), or PPP Appendix-C
        emitter--perturber reduced-mass relative speed.
    emitter_charge : float or None, optional
        Net radiator charge in units of e. ``None`` infers ``Z-1`` for a
        hydrogen-like radiator. The bundled Potekhin distribution currently
        uses only neutral versus charged-point status, not its magnitude.
    energies_ev : array-like
        Photon energies at which to evaluate the profile [eV].
    num_f : int, optional
        Number of microfield quadrature points (default 30). Converge it jointly
        with ``max_beta`` for the requested transition, plasma conditions,
        spectral window, and observable; the default is not a universal
        accuracy guarantee.
    num_mu : int, optional
        Number of Gauss-Legendre points for the field-angle integration
        over μ = cos θ ∈ [0, 1] (default 10). Check convergence for the
        requested field and polarization observable; the default is not a
        universal accuracy guarantee.
    max_beta : float, optional
        Upper limit of the reduced-microfield grid β = F/F₀ (default 10).
        The Holtsmark tail beyond β = 10 carries ~3 % of the probability;
        increase this (with ``num_f``) for far-wing studies.  Forwarded to
        :func:`~starkzee.microfield.microfield_quadrature`.
    use_screening : bool, optional
        If True (default), use Potekhin when ``microfield_model`` is omitted;
        if False, use Holtsmark. Ignored when an explicit model is given.
    microfield_model : {None, 'holtsmark', 'hooper', 'potekhin', 'custom'}, optional
        Explicit microfield-distribution selector, forwarded to
        :func:`~starkzee.microfield.microfield_quadrature` (see there for the
        full description of each model). With ``None``, screened calls select
        Potekhin and unscreened calls select Holtsmark. The Hooper-like ansatz
        requires explicit ``'hooper'`` selection and is not validated.
    quadratic_zeeman : bool, optional
        Include the diamagnetic quadratic Zeeman term (default True).
    fine_structure : bool, optional
        Include mass-velocity and Darwin corrections (default True).
    numerical_inversion : bool, optional
        If True, solve the full Markov matrix by direct inversion instead of
        using the exact Sherman-Morrison reduction (default False). Both paths
        currently use the same uniform rank-one Markov matrix. Falls back to
        Sherman-Morrison if the matrix is singular.
    use_empirical_data : bool, optional
        Use NIST empirical level energies (default True); forwarded to
        :func:`~starkzee.static_profile.solve_starkzee`.
    atom : str, optional
        Atom identifier for empirical data (default ``"H"``); forwarded to
        :func:`~starkzee.static_profile.solve_starkzee`.
    electron_model : str, optional
        Electron-impact width prescription, forwarded to
        :func:`~starkzee.broadening.electron_impact_width_model`.  ``'pppb'``
        (default) is the PPPB model; ``'zest'``,
        ``'zest-lee'``, ``'zest-dufty'`` select the ZEST model with the GBK, Lee,
        or Dufty RPA G-function.
    sdt_frequency_dependent_width : bool, optional
        If False (default), apply one resonance electron-impact width evaluated
        at G(0) to every SDT, matching the historical FFM and ZEST fast-FFM
        approximation. If True, evaluate the upper- plus lower-shell width for
        each post-binning SDT at its own detuning from the physical line
        reference. This is an optional PPPB-style approximation; it is distinct
        from the static solver's pointwise observation-frequency width.
    electron_interference : bool, optional
        Opt-in PPP impact-limit collision operator (default ``False``). Builds
        Appendix B Eq. (B1), retains upper--lower interference, and passes its
        complex generalized intensities ``a_k + i c_k`` into the FFM. It
        requires ``sdt_frequency_dependent_width=False``, no SDT binning, and
        the analytical FFM solver; these constraints avoid silently replacing
        the manual's non-Hermitian construction by incompatible approximations.
        Its Appendix-B coefficient is evaluated directly, so the ``pppb`` and
        ``pppb-intra`` scalar radius selectors produce the same full operator.
        The Markov probabilities are normalized from ``|a_k|``; the signed
        ``a_k + i c_k`` values are retained in the radiative numerator.
    interference_diagnostics : mutable mapping or None, optional
        When supplied with ``electron_interference=True``, clear and populate
        this mapping with one signed-residue diagnostic record per polarization
        channel ``q``. This does not change the calculated profile.
    interference_group_tolerance_ev : float or None, optional
        Experimental alternative to modulus probabilities. When positive,
        group complex SDTs close in frequency and homogeneous width, require
        nonnegative grouped real residues, and use those residues as the
        Markov probabilities. The grouped static profile must reproduce the
        ungrouped profile within ``interference_group_profile_rtol``. No
        grouping is applied by default.
    interference_group_width_tolerance_ev : float or None, optional
        Maximum width span within a group. Defaults to
        ``interference_group_tolerance_ev`` when grouping is enabled.
    interference_group_profile_rtol : float, optional
        Maximum peak-normalized absolute difference allowed between grouped
        and ungrouped zero-fluctuation profiles (default ``1e-3``).
    natural_width_mode : {'state_resolved', 'shell_average'}, optional
        Natural-damping treatment. ``'state_resolved'`` (default) rotates the
        uncoupled E1 decay rates into every Stark–Zeeman eigenbasis and assigns
        each SDT ``ħ(Γ_u,i + Γ_l,j)/2``. ``'shell_average'`` restores the
        historical single natural width shared by all SDTs.
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
    sdt_bin_tol : float or None, optional
        SDT frequency-binning tolerance [eV] (default ``None`` = no binning).
        Before solving the Markov system, SDTs of each polarization channel
        whose frequencies fall in the same bin of width ``sdt_bin_tol`` are
        merged (intensities summed, frequency = intensity-weighted mean).
        This shrinks the O(N) Sherman-Morrison sum but is an approximation:
        frequencies and state-resolved natural widths are replaced by
        intensity-weighted means. Compare against ``None`` while decreasing
        the tolerance for every requested transition and plasma regime; no
        nonzero tolerance is a universal safe default.

    Returns
    -------
    profile_pi : ndarray, shape like *energies_ev*
        π polarization component (Δm = m_u − m_l = 0; dipole channel q = 0).
    profile_sig_plus : ndarray
        σ+ polarization component (Δm = +1, blue-shifted at B > 0; dipole
        channel q = m_l − m_u = −1, same convention as
        :func:`~starkzee.static_profile.calculate_static_profile`).
    profile_sig_minus : ndarray
        σ− polarization component (Δm = −1, red-shifted; channel q = +1).

    Notes
    -----
    **Static vs FFM guidance**: there is no universal threshold in ``B`` or
    ``N_e`` alone.  Use the static profile when the ion fluctuation energy
    ``ν_i`` is small compared with the characteristic Stark-dressed-transition
    spread.  FFM becomes more relevant for faster, lighter ions, higher-``n``
    transitions, and line cores whose components are not well separated (often
    at lower ``B``).  If the regimes are not clearly separated, calculate both
    profiles with matched quadrature, microfield, electron-width, and Doppler
    settings and compare them over the observable spectral window.

    **B = 0 note**: the π/σ decomposition is physically meaningless at B = 0
    (no preferred axis); all three components are equal by symmetry and their
    sum gives the isotropic total profile.

    References
    ----------
    Calisti, A. et al., Phys. Rev. A 42, 5433 (1990). — FFM formulation.
    Ferri, S., Peyrusse, O. & Calisti, A., Matter Radiat. Extremes 7, 015901 (2022).
    """
    energies_ev, restore_grid = uniform_energy_grid(energies_ev, resample=apply_doppler and Ti_ev > 0)
    if natural_width_mode not in ('state_resolved', 'shell_average'):
        raise ValueError(
            "natural_width_mode must be 'state_resolved' or 'shell_average'.")
    if interference_diagnostics is not None:
        if not electron_interference:
            raise ValueError(
                "interference_diagnostics requires electron_interference=True.")
        if not isinstance(interference_diagnostics, MutableMapping):
            raise TypeError("interference_diagnostics must be a mutable mapping.")
        interference_diagnostics.clear()
    if interference_group_tolerance_ev is not None:
        if not electron_interference:
            raise ValueError(
                "interference grouping requires electron_interference=True.")
        if interference_group_tolerance_ev <= 0:
            raise ValueError(
                "interference_group_tolerance_ev must be positive.")
        if (interference_group_width_tolerance_ev is not None
                and interference_group_width_tolerance_ev <= 0):
            raise ValueError(
                "interference_group_width_tolerance_ev must be positive.")
        if interference_group_profile_rtol < 0:
            raise ValueError(
                "interference_group_profile_rtol must be nonnegative.")
    elif interference_group_width_tolerance_ev is not None:
        raise ValueError(
            "interference_group_width_tolerance_ev requires "
            "interference_group_tolerance_ev.")
    if electron_interference:
        if sdt_frequency_dependent_width:
            raise ValueError(
                "electron_interference=True implements the PPP impact-limit "
                "operator and requires sdt_frequency_dependent_width=False."
            )
        if sdt_bin_tol is not None:
            raise ValueError(
                "electron_interference=True requires sdt_bin_tol=None because "
                "complex generalized SDTs cannot be intensity-binned safely."
            )
        if numerical_inversion:
            raise ValueError(
                "electron_interference=True currently requires the analytical "
                "PPPB FFM solver (numerical_inversion=False)."
            )
        if electron_model.lower() not in ('pppb', 'ferri', 'pppb-intra', 'ferri-intra'):
            raise ValueError(
                "electron_interference=True currently supports only the PPPB/Ferri "
                "electron models."
            )
    if emitter_charge is None:
        emitter_charge = Z - 1
    if not np.isfinite(emitter_charge) or emitter_charge < 0:
        raise ValueError("emitter_charge must be finite and nonnegative.")
    if charged is None:
        charged = emitter_charge != 0
    perturber_mass = A_ion if A_perturber is None else A_perturber
    # Background charge is not the radiator's nuclear charge.
    nu_i = calculate_ion_fluctuation_rate(
        Ne_m3, Ti_ev, Z_bar, perturber_mass,
        A_emitter=A_ion, model=fluctuation_rate_model)

    # 2. Get microfield grid and weights
    fields, f_weights = microfield_quadrature(Ne_m3, Te_ev, num_points=num_f,
                                              max_beta=max_beta, use_screening=use_screening,
                                              microfield_model=microfield_model, Ti_ev=Ti_ev,
                                              custom_table_path=custom_table_path,
                                              charged=charged, emitter_charge=emitter_charge,
                                              Z_bar=Z_bar,
                                              species_charges=species_charges,
                                              species_concentrations=species_concentrations)
    
    # 3. Get angular integration points (Gauss-Legendre)
    mu_points, mu_weights = np.polynomial.legendre.leggauss(num_mu)
    mu_points = 0.5 * (mu_points + 1.0)
    mu_weights = 0.5 * mu_weights
    
    D_q_uncoupled = _uncoupled_dipole_matrices(n_u, n_l, Z)
    gamma_resonance_u = electron_impact_width_model(
        0.0, Ne_m3, Te_ev, B, Z, n=n_u, electron_model=electron_model)
    gamma_resonance_l = (electron_impact_width_model(
        0.0, Ne_m3, Te_ev, B, Z, n=n_l, electron_model=electron_model)
        if n_l > 1 else 0.0)
    ppp_impact_coefficient = None
    if electron_interference:
        ppp_impact_coefficient = electron_impact_collision_coefficient(
            Ne_m3, Te_ev, B, Z, n=n_u)
    if natural_width_mode == 'state_resolved':
        natural_rates_u_basis = natural_decay_rates(n_u, Z)
        natural_rates_l_basis = natural_decay_rates(n_l, Z)
    else:
        gamma_upper = sum(einstein_a(n_u, k, Z) for k in range(1, n_u))
        gamma_lower = (sum(einstein_a(n_l, k, Z) for k in range(1, n_l))
                       if n_l > 1 else 0.0)
        shell_natural_ev = HBAR * (gamma_upper + gamma_lower) / 2.0 / E_CHARGE

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
            
            # Solve combined Stark-Zeeman Hamiltonian for upper and lower states.
            # A=A_ion (same-species assumption) so the reduced-mass Rydberg — and
            # hence the absolute SDT frequencies — match the emitting isotope.
            sz_energies_u, sz_vectors_u = solve_starkzee(n_u, Z, B, Fz, Fx, quadratic_zeeman, fine_structure,
                                                         A=A_ion,
                                                         use_empirical_data=use_empirical_data, atom=atom)
            sz_energies_l, sz_vectors_l = solve_starkzee(n_l, Z, B, Fz, Fx, quadratic_zeeman, fine_structure,
                                                         A=A_ion,
                                                         use_empirical_data=use_empirical_data, atom=atom)

            if natural_width_mode == 'state_resolved':
                natural_rates_u = (np.abs(sz_vectors_u)**2).T @ natural_rates_u_basis
                natural_rates_l = (np.abs(sz_vectors_l)**2).T @ natural_rates_l_basis
                natural_widths = (HBAR / (2.0 * E_CHARGE)
                                  * (natural_rates_l[:, np.newaxis]
                                     + natural_rates_u[np.newaxis, :]))
            
            V_l_adj = sz_vectors_l.conj().T
            dE = sz_energies_u[np.newaxis, :] - sz_energies_l[:, np.newaxis]

            if electron_interference:
                natural_for_operator = (natural_widths
                                        if natural_width_mode == 'state_resolved'
                                        else shell_natural_ev)
                frequencies, widths, complex_strengths = ppp_complex_sdts(
                    dE, sz_vectors_u, sz_vectors_l, D_q_uncoupled,
                    n_u, n_l, Z, ppp_impact_coefficient,
                    natural_for_operator, electron_interference=True,
                )
                for q in [0, 1, -1]:
                    strengths = complex_strengths[q]
                    keep = np.abs(strengths) > 1e-12
                    for frequency, width, strength in zip(
                            frequencies[keep], widths[keep], strengths[keep]):
                        sdt_list[q].append({
                            "intensity": weight * strength.real,
                            "dispersion": weight * strength.imag,
                            "frequency": frequency,
                            "collision_width": width,
                            "natural_width": 0.0,
                        })
                continue
            
            # For each polarization q
            for q in [0, 1, -1]:
                mixed_D = V_l_adj @ D_q_uncoupled[q] @ sz_vectors_u
                intensities = np.abs(mixed_D)**2
                
                # Collect non-zero transitions
                j_indices, i_indices = np.where(intensities > 1e-12)
                for j, i in zip(j_indices, i_indices):
                    sdt_list[q].append({
                        "intensity": weight * intensities[j, i],
                        "frequency": dE[j, i],
                        "natural_width": (natural_widths[j, i]
                                          if natural_width_mode == 'state_resolved'
                                          else shell_natural_ev),
                    })
                        
    output_profiles = {}
    
    # The default diagonal treatment keeps W_e(n_u) + W_e(n_l) (Ferri,
    # Peyrusse & Calisti 2022 Eq. 8; ZEST) and drops upper-lower interference.
    # The opt-in branch has already encoded the full PPP operator in its SDTs.
    # n_l = 1 (Lyman) has no intra-shell dipole channel, so its impact width is 0.
    gamma_resonance = None
    if not sdt_frequency_dependent_width:
        gamma_resonance = gamma_resonance_u + gamma_resonance_l
    reference_energy_ev = line_reference_energy(
        n_u, n_l, Z, A_ion, use_empirical_data, atom)
    
    for q in [0, 1, -1]:
        sdts = sdt_list[q]
        if not sdts:
            output_profiles[q] = np.zeros_like(energies_ev)
            continue
            
        intensities = np.array([item["intensity"] for item in sdts])
        frequencies = np.array([item["frequency"] for item in sdts])
        natural_widths = np.array([item["natural_width"] for item in sdts])

        if electron_interference:
            dispersions = np.array([item["dispersion"] for item in sdts])
            gamma_k = np.array([item["collision_width"] for item in sdts])
            r_q_sq = float(np.sum(intensities))
            if r_q_sq <= 1e-15:
                output_profiles[q] = np.zeros_like(energies_ev)
                continue
            strengths = intensities + 1j * dispersions
            if interference_diagnostics is not None:
                interference_diagnostics[q] = summarize_complex_sdt_probabilities(
                    frequencies, gamma_k, strengths)
            if interference_group_tolerance_ev is not None:
                ungrouped_frequencies = frequencies
                ungrouped_gamma = gamma_k
                ungrouped_strengths = strengths
                frequencies, gamma_k, strengths, group_sizes = group_complex_sdts(
                    frequencies, gamma_k, strengths,
                    interference_group_tolerance_ev,
                    interference_group_width_tolerance_ev)
                grouped_static = _complex_ffm_profile_analytical(
                    energies_ev, frequencies, gamma_k, strengths, 0.0)
                ungrouped_static = _complex_ffm_profile_analytical(
                    energies_ev, ungrouped_frequencies, ungrouped_gamma,
                    ungrouped_strengths, 0.0)
                profile_scale = max(float(np.max(np.abs(ungrouped_static))),
                                    np.finfo(float).tiny)
                profile_error = float(np.max(np.abs(
                    grouped_static - ungrouped_static)) / profile_scale)
                grouped_real = strengths.real
                grouped_scale = float(np.sum(np.abs(grouped_real)))
                negative_tolerance = 1e-12 * grouped_scale
                negative_count = int(np.count_nonzero(
                    grouped_real < -negative_tolerance))
                nonpositive_width_count = int(np.count_nonzero(gamma_k <= 0))
                if (profile_error > interference_group_profile_rtol
                        or negative_count or nonpositive_width_count):
                    raise ValueError(
                        "Experimental complex-SDT grouping did not produce a "
                        "valid FFM reduction for "
                        f"q={q}: profile error={profile_error:.3e} "
                        f"(limit {interference_group_profile_rtol:.3e}), "
                        f"negative grouped residues={negative_count}, "
                        f"nonpositive grouped widths={nonpositive_width_count}. "
                        "The ungrouped static full-operator profile remains "
                        "available; change tolerances only after checking "
                        "static-profile fidelity."
                    )
                if interference_diagnostics is not None:
                    grouped_dynamic = _complex_ffm_profile_analytical(
                        energies_ev, frequencies, gamma_k, strengths, nu_i)
                    modulus_dynamic = _complex_ffm_profile_analytical(
                        energies_ev, ungrouped_frequencies, ungrouped_gamma,
                        ungrouped_strengths, nu_i)
                    dynamic_scale = max(
                        float(np.max(np.abs(modulus_dynamic))),
                        np.finfo(float).tiny)
                    interference_diagnostics[q]["grouping"] = {
                        "original_mode_count": int(len(ungrouped_strengths)),
                        "grouped_mode_count": int(len(strengths)),
                        "largest_group_size": int(np.max(group_sizes)),
                        "static_profile_relative_max_error": profile_error,
                        "modulus_ffm_relative_max_difference": float(np.max(
                            np.abs(grouped_dynamic - modulus_dynamic))
                            / dynamic_scale),
                    }
            output_profiles[q] = _complex_ffm_profile_analytical(
                energies_ev, frequencies, gamma_k, strengths, nu_i)
            continue

        # Optional SDT binning: merge near-degenerate transitions to shrink N.
        # This is approximate because component frequencies and natural widths
        # are replaced by intensity-weighted means; converge against unbinned.
        if sdt_bin_tol is not None and sdt_bin_tol > 0 and len(frequencies) > 1:
            bin_idx = np.round(frequencies / sdt_bin_tol).astype(np.int64)
            _, inv = np.unique(bin_idx, return_inverse=True)
            binned_I  = np.bincount(inv, weights=intensities)
            binned_wf = np.bincount(inv, weights=intensities * frequencies)
            binned_wn = np.bincount(inv, weights=intensities * natural_widths)
            keep = binned_I > 0
            intensities = binned_I[keep]
            frequencies = (binned_wf[keep] / binned_I[keep])
            natural_widths = (binned_wn[keep] / binned_I[keep])

        if sdt_frequency_dependent_width:
            sdt_detuning = frequencies - reference_energy_ev
            gamma_k = electron_impact_width_model(
                sdt_detuning, Ne_m3, Te_ev, B, Z, n=n_u,
                electron_model=electron_model)
            if n_l > 1:
                gamma_k += electron_impact_width_model(
                    sdt_detuning, Ne_m3, Te_ev, B, Z, n=n_l,
                    electron_model=electron_model)
            gamma_k = np.asarray(gamma_k, dtype=float) + natural_widths
        else:
            gamma_k = gamma_resonance + natural_widths

        r_q_sq = np.sum(intensities)
        if r_q_sq <= 1e-15:
            output_profiles[q] = np.zeros_like(energies_ev)
            continue
            
        normalized_intensities = intensities / r_q_sq
        M = len(energies_ev)
        N = len(intensities)   # post-binning count (== len(sdts) when unbinned)

        # Per-q solver choice: a LinAlgError fallback must not silently switch
        # the solver for the remaining polarizations, so keep the flag local.
        use_inversion = numerical_inversion

        if use_inversion and N > 1:
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
                use_inversion = False

        if not use_inversion or N <= 1:
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
        # Use the physical (analytic) gross-structure line center, not mean(energies_ev):
        # the latter depends on the requested observation window, not on the emitter.
        sigma_D = reference_energy_ev * np.sqrt(Ti_ev / mc2_ev)
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

    # σ+ (blue at B > 0) is the q = −1 dipole channel, σ− is q = +1 — the same
    # mapping used by calculate_static_profile (q = m_l − m_u; σ+ has m_u = m_l + 1).
    return tuple(restore_grid(output_profiles[q]) for q in (0, -1, 1))
