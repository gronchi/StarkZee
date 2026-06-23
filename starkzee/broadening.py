"""
Electron impact broadening operator and collision G-functions (Lee, GBK, and Dufty RPA).
"""

import numpy as np
from scipy.special import exp1, dawsn
from scipy.integrate import quad
from scipy.constants import (
    hbar as HBAR, m_e as M_E, e as E_CHARGE, epsilon_0 as EPSILON_0,
    physical_constants,
)
from starkzee.utils import RYDBERG_EV, HARTREE_EV

A_BOHR = physical_constants['Bohr radius'][0]  # m


def calculate_plasma_frequency(Ne_m3):
    """Return the electron plasma angular frequency ω_p [rad s⁻¹].

    The plasma frequency sets the lower cutoff for electron-impact broadening:
    perturbations with frequency ω < ω_p are screened by collective plasma
    oscillations and do not contribute to individual collisions.

        ω_p = √(N_e e² / (ε₀ m_e))

    Parameters
    ----------
    Ne_m3 : float
        Electron number density [m⁻³].

    Returns
    -------
    float
        Plasma angular frequency [rad s⁻¹].
    """
    omega_p = np.sqrt(Ne_m3 * (E_CHARGE**2) / (EPSILON_0 * M_E))
    return omega_p


def calculate_larmor_frequency(B):
    """Return the electron Larmor (cyclotron) angular frequency ω_L [rad s⁻¹].

    In a magnetic field B the electron gyrates at

        ω_L = e B / m_e

    This frequency acts as a lower cutoff for the GBK electron-broadening model
    when it exceeds both ω_p and ω_e: cyclotron motion prevents an electron from
    approaching the radiator more closely than the cyclotron radius, reducing the
    effective cross-section at low detunings.

    Parameters
    ----------
    B : float
        Magnetic field strength [T]. ``B=0`` returns 0 safely.

    Returns
    -------
    float
        Larmor angular frequency [rad s⁻¹].
    """
    return E_CHARGE * B / M_E


def calculate_configuration_frequency(Ne_m3, Te_ev):
    """Return the configuration-change frequency ω_e = 1 / τ_e [rad s⁻¹].

    τ_e is the mean time for an electron to cross the Wigner-Seitz radius r_e
    at the thermal velocity v_th:

        r_e  = (3 / 4π N_e)^(1/3)     — Wigner-Seitz inter-electron radius
        v_th = √(k_B T_e / m_e)        — thermal speed
        τ_e  = r_e / v_th
        ω_e  = 1 / τ_e  =  v_th / r_e

    **Note on the 2π factor.**  Ferri, Peyrusse & Calisti (2022) write
    ω_e = 2π / τ_e, which is dimensionally correct: 1/τ_e is a rate in Hz and
    multiplying by 2π gives an angular frequency in rad/s comparable to ω_p and
    ω_L.  However, that formula yields ω_e ≈ 290 meV at Ne = 10¹⁷ cm⁻³,
    Te = 5 eV, which exceeds ω_L even at B = 1 kT (ω_L ≈ 116 meV) and renders
    the max() cutoff insensitive to B.  Their published figure shows distinct
    curves at B = 500 T and 1 kT, consistent only with ω_e < ω_L(500 T) ≈ 58 meV.
    Their code (PPPB) therefore likely used 1/τ_e directly as an angular frequency
    (a common plasma-physics convention conflating collision rate and angular
    frequency), giving ω_e ≈ 46 meV.  StarkZee adopts the same convention.

    Parameters
    ----------
    Ne_m3 : float
        Electron number density [m⁻³].
    Te_ev : float
        Electron temperature [eV].

    Returns
    -------
    float
        Configuration-change frequency ω_e = v_th / r_e [rad s⁻¹].
    """
    re = (3.0 / (4.0 * np.pi * Ne_m3))**(1.0 / 3.0)
    v_th = np.sqrt(Te_ev * E_CHARGE / M_E)
    return v_th / re


def calculate_debye_length(Ne_m3, Te_ev):
    """Return the classical electron Debye screening length λ_D [m].

        λ_D = √(ε₀ k_B T_e / (e² N_e))

    Parameters
    ----------
    Ne_m3 : float
        Electron number density [m⁻³].
    Te_ev : float
        Electron temperature [eV].

    Returns
    -------
    float
        Debye length [m].
    """
    return np.sqrt(EPSILON_0 * Te_ev / (E_CHARGE * Ne_m3))


def calculate_cutoff_kappa_m(Z, n, Te_ev):
    """Return the maximum wave-number cutoff κ_m [m⁻¹] (ZEST convention).

    κ_m = min(κ_geo, κ_thermal) where:
        κ_geo     = Z / (n² a₀)                    — geometric (Bohr orbit)
        κ_thermal = Z √(2 m_e k_B T_e) / (ħ n²)  — thermal de Broglie

    The geometric limit dominates at T_e > 13.6 eV; the thermal limit dominates
    below that threshold (1 Rydberg).

    Parameters
    ----------
    Z : int
        Nuclear charge.
    n : int
        Principal quantum number of the upper level.
    Te_ev : float
        Electron temperature [eV].

    Returns
    -------
    float
        Maximum wave-number cutoff κ_m [m⁻¹].
    """
    p_th = np.sqrt(2.0 * M_E * Te_ev * E_CHARGE)    # √(2 m_e k_B T_e)
    kappa_geo     = Z / (n**2 * A_BOHR)
    kappa_thermal = Z * p_th / (HBAR * n**2)
    return min(kappa_geo, kappa_thermal)


def calculate_electron_impact_prefactor(Ne_m3, Te_ev):
    """Return the electron-impact width prefactor W₀ [eV].

    The total electron-impact half-width is

        W_e = W₀ × ⟨r²⟩_n × [C_n + G(Δω)]

    where ⟨r²⟩_n is the statistically averaged squared radius of the upper
    level (in a₀²), C_n is a strong-collision constant, and G is the GBK
    dynamical factor.  W₀ combines the electron density, temperature, and
    fundamental constants into a single prefactor:

        W₀ = (4π/3) N_e √(2m_e / (π k_B T_e)) × (ħ/m_e)² × (ħ/e)

    The factor (ħ/m_e)² converts the squared velocity integral from SI to
    atomic-unit area, and (ħ/e) converts rad s⁻¹ to eV.

    Parameters
    ----------
    Ne_m3 : float
        Electron number density [m⁻³].
    Te_ev : float
        Electron temperature [eV].

    Returns
    -------
    float
        Prefactor W₀ [eV a₀⁻²].  Multiply by ⟨r²⟩_n [a₀²] to obtain [eV].
    """
    Te_j = Te_ev * E_CHARGE

    term1 = (4.0 * np.pi / 3.0) * Ne_m3
    term2 = np.sqrt(2.0 * M_E / (np.pi * Te_j))
    term3 = (HBAR / M_E)**2
    term4 = HBAR / E_CHARGE

    return term1 * term2 * term3 * term4


def gbk_model(delta_omega_ev, omega_c_ev, Te_ev, Z, n=2):
    """Evaluate the semi-classical GBK dynamical broadening function G(Δω).

    The Griem–Baranger–Kolb (GBK) model accounts for the frequency dependence
    of electron-impact broadening using an exponential-integral form:

        G(Δω) = ½ E₁(y)

    where E₁ is the exponential integral and the dimensionless argument is

        y = (n² / 2Z)² × (Δω² + ω_c²) / (E_H T_e)

    with E_H = e²/2a₀ = 13.6057 eV (Rydberg energy) and T_e in eV.  GBK, Ferri and
    ZEST all use this same definition.  At line center (Δω = 0)
    G reduces to ½ E₁((n²/2Z)² ω_c² / (E_H T_e)) ≈ a positive constant; in
    the far wings where Δω ≫ ω_c the argument y grows and G → 0, suppressing
    the broadening at large detunings (the impact approximation breaks down).

    The cutoff frequency ω_c = max(ω_p, ω_e, ω_L, ω_αα′) prevents the logarithm
    from diverging at small impact parameters and encodes the transition
    from the impact regime to the quasi-static regime.

    Parameters
    ----------
    delta_omega_ev : float or array-like
        Frequency detuning from line center Δω [eV].
    omega_c_ev : float
        Cutoff angular frequency ω_c [eV] (= ħ ω_c in SI units).
    Te_ev : float
        Electron temperature [eV].
    Z : int
        Nuclear charge of the radiating ion.
    n : int, optional
        Principal quantum number of the *upper* level (default 2).

    Returns
    -------
    float or ndarray
        Dimensionless GBK factor G(Δω) ≥ 0.

    References
    ----------
    Griem, Kolb & Shen, Phys. Rev. 116, 4 (1959) — original G-function for hydrogen.
    Griem, Baranger, Kolb & Oertel, Phys. Rev. 125, 177 (1962) — helium extension.
    Ferri, Peyrusse & Calisti, Matter Radiat. Extremes 7, 015901 (2022) — C_n constants and ω_c formulation.
    """
    num = (delta_omega_ev**2 + omega_c_ev**2)
    y = ((n**2) / (2.0 * Z))**2 * (num / (RYDBERG_EV * Te_ev))

    g_val = 0.5 * exp1(np.maximum(1e-15, y))
    return g_val


def gbk_zest_model(delta_omega_ev, Ne_m3, Te_ev, Z, n=2):
    """Evaluate the GBK G-function G(Δω) with ω_p as the only cutoff (ZEST convention).

    Uses the κ_m-based argument written in eV units:

        x   = κ_m λ_D                            (dimensionless cutoff)
        arg = (Δω² + ω_p²) / (2 x² ω_p²)
        G   = ½ E₁(arg)

    where Δω and ω_p are in eV, and
    κ_m = min(Z/(n²a₀), Z√(2mₑkBTe)/(ħn²)).

    This is equivalent to the analytic form (n²/2Z)²(Δω²+ωp²)/(Ryd×Te) only in the
    geometric κ_m regime (Te > 13.6 eV).  Below that threshold the thermal branch
    of κ_m is active and the analytic form gives a different (larger) G.  This
    implementation uses the physical κ_m so it is consistent with :func:`lee_model`
    and :func:`dufty_model`: all three converge at large Δω where G → ½ E₁(Δω²/(2x²ωp²)).

    The only cutoff frequency is ω_p (no ω_e, no ω_L).

    Parameters
    ----------
    delta_omega_ev : float or array-like
        Frequency detuning Δω [eV].
    Ne_m3 : float
        Electron number density [m⁻³].
    Te_ev : float
        Electron temperature [eV].
    Z : int
        Nuclear charge.
    n : int, optional
        Principal quantum number of the upper level (default 2).

    Returns
    -------
    float or ndarray
        Dimensionless GBK factor G(Δω) ≥ 0.

    See Also
    --------
    gbk_model : GBK/Ferri formula using max(ω_p, ω_e, ω_L) as cutoff.
    lee_model : Lee analytical blend of impact and wing limits.
    """
    scalar = np.isscalar(delta_omega_ev)
    dw_ev  = np.abs(np.atleast_1d(np.asarray(delta_omega_ev, dtype=float)))

    omega_p_ev = calculate_plasma_frequency(Ne_m3) * HBAR / E_CHARGE
    lambda_D   = calculate_debye_length(Ne_m3, Te_ev)
    kappa_m    = calculate_cutoff_kappa_m(Z, n, Te_ev)
    x = kappa_m * lambda_D

    arg = (dw_ev**2 + omega_p_ev**2) / (2.0 * x**2 * omega_p_ev**2)
    arg = np.maximum(arg, 1e-30)
    res = 0.5 * exp1(arg)
    return float(res[0]) if scalar else res


def lee_model(delta_omega_ev, Ne_m3, Te_ev, Z, n=2):
    """Evaluate Lee's analytical approximation for the G-function G(Δω).

    The Lee model blends two limiting expressions:
        G₀   = ½ [ln(1 + x²) − x²/(1 + x²)],   x = κ_m λ_D  (impact limit, Δω → 0)
        G∞   = ½ E₁(Δω² / (2 κ_m² v_th²))                    (wing limit, Δω → ∞)
        G    = min(G₀, G∞)

    Unlike :func:`gbk_zest_model`, the G∞ argument contains only Δω²
    (not Δω² + ω_p²): the plasma cutoff is captured implicitly by G₀.

    Parameters
    ----------
    delta_omega_ev : float or array-like
        Frequency detuning Δω [eV].
    Ne_m3 : float
        Electron number density [m⁻³].
    Te_ev : float
        Electron temperature [eV].
    Z : int
        Nuclear charge.
    n : int, optional
        Principal quantum number of the upper level (default 2).

    Returns
    -------
    float or ndarray
        Lee G-function value G(Δω) ≥ 0.
    """
    scalar = np.isscalar(delta_omega_ev)
    dw_rad = np.abs(np.atleast_1d(np.asarray(delta_omega_ev, dtype=float))) * E_CHARGE / HBAR

    omega_p  = calculate_plasma_frequency(Ne_m3)
    lambda_D = calculate_debye_length(Ne_m3, Te_ev)
    kappa_m  = calculate_cutoff_kappa_m(Z, n, Te_ev)

    x  = kappa_m * lambda_D
    g0 = 0.5 * (np.log(1.0 + x**2) - x**2 / (1.0 + x**2))

    arg   = dw_rad**2 / (2.0 * (kappa_m * lambda_D * omega_p)**2)
    g_inf = 0.5 * exp1(np.maximum(arg, 1e-30))

    res = np.minimum(g0, g_inf)
    return float(res[0]) if scalar else res


def _dufty_integrand(kappa, dw_rad, lambda_D, Te_ev):
    v_th  = np.sqrt(2.0 * Te_ev * E_CHARGE / M_E)   # ZEST convention: √(2kT/m)
    x     = abs(dw_rad) / (kappa * v_th)
    ratio = (1.0 / (kappa * lambda_D))**2
    re_eps = 1.0 + ratio * (1.0 - 2.0 * x * dawsn(x))
    if x < 10.0:
        exp_x2 = np.exp(-x**2)
        im_eps = ratio * np.sqrt(np.pi) * x * exp_x2
    else:
        exp_x2 = 0.0
        im_eps = 0.0
    return exp_x2 / (kappa * (re_eps**2 + im_eps**2))


def dufty_model(delta_omega_ev, Ne_m3, Te_ev, Z, n=2):
    """Evaluate the Dufty RPA G-function G(Δω) by numerical integration.

    Integrates the electron RPA dielectric function over wave numbers κ_min to
    κ_m, accounting for Landau damping via the Dawson function:

        G(Δω) = ∫_{κ_min}^{κ_m} e^{−x²} / (κ |ε(κ, Δω)|²) dκ

    where x = |Δω| / (κ v_th) and v_th = √(2 k_B T_e / m_e).  This is the most
    accurate of the three G models but also the slowest (one quadrature per point).

    Parameters
    ----------
    delta_omega_ev : float or array-like
        Frequency detuning Δω [eV].
    Ne_m3 : float
        Electron number density [m⁻³].
    Te_ev : float
        Electron temperature [eV].
    Z : int
        Nuclear charge.
    n : int, optional
        Principal quantum number of the upper level (default 2).

    Returns
    -------
    float or ndarray
        Dufty RPA G-function value G(Δω) ≥ 0.

    See Also
    --------
    gbk_zest_model : Faster closed-form GBK approximation.
    """
    scalar  = np.isscalar(delta_omega_ev)
    dw_rad  = np.atleast_1d(np.asarray(delta_omega_ev, dtype=float)) * E_CHARGE / HBAR

    lambda_D = calculate_debye_length(Ne_m3, Te_ev)
    kappa_m  = calculate_cutoff_kappa_m(Z, n, Te_ev)
    v_th     = np.sqrt(2.0 * Te_ev * E_CHARGE / M_E)

    result = np.empty(len(dw_rad))
    for i, dw in enumerate(dw_rad):
        dw_abs    = abs(dw)
        kappa_min = dw_abs / (10.0 * v_th) if dw_abs > 1e-15 else 1.0 / (100.0 * lambda_D)
        if kappa_min >= kappa_m:
            result[i] = 0.0
        else:
            val, _ = quad(_dufty_integrand, kappa_min, kappa_m,
                          args=(dw, lambda_D, Te_ev), epsrel=1e-5)
            result[i] = val
    return float(result[0]) if scalar else result


def electron_impact_width(delta_omega_ev, Ne_m3, Te_ev, B, Z, n=2):
    """Return the total electron-impact half-width W_e(Δω) [eV].

    Implements the frequency-dependent GBK model for electron Stark broadening,
    extended to include a magnetic-field-dependent cutoff frequency.  The total
    half-width (HWHM of the Lorentzian) is:

        W_e(Δω) = W₀ × ⟨r²⟩_n × [C_n + G(Δω, ω_c)]

    **Prefactor W₀** — see :func:`calculate_electron_impact_prefactor`.

    **Mean squared radius** ⟨r²⟩_n is the statistical (2l+1)-weighted average
    of ⟨r²⟩_{n,l} over all l subshells:

        ⟨r²⟩_{n,l} = (n²/2Z²) [5n² + 1 − 3l(l+1)]   [a₀²]
        ⟨r²⟩_n     = (1/n²) Σ_{l=0}^{n-1} (2l+1) ⟨r²⟩_{n,l}

    Scales approximately as n⁴/Z², so broadening grows rapidly with n.

    **Strong-collision constant C_n** (Ferri, Peyrusse & Calisti,
    Matter Radiat. Extremes 7, 015901 (2022), Table 1):

    =========  ======
    n          C_n
    =========  ======
    ≤ 2        1.50
    3          1.00
    4          0.75
    5          0.50
    > 5        0.40
    =========  ======

    **Cutoff frequency** ω_c = max(ω_p, ω_L, ω_e, ω_αα′) (Ferri, Peyrusse &
    Calisti 2022, below Eq. 20), where ω_p is the plasma frequency, ω_L the
    electron Larmor frequency, ω_e = 1/τ_e the configuration-change frequency
    (see :func:`calculate_configuration_frequency`), and ω_αα′ the
    state-to-state transition frequency.  ω_αα′ = 0 for hydrogen (degenerate
    l-subshells within a shell); it is non-zero for multi-electron atoms and is
    currently a zero placeholder pending implementation.

    Parameters
    ----------
    delta_omega_ev : float or array-like
        Frequency detuning from line center Δω [eV].  Pass ``0.0`` for the
        on-resonance (line-center) width.
    Ne_m3 : float
        Electron number density [m⁻³].
    Te_ev : float
        Electron temperature [eV].
    B : float
        Magnetic field [T]. ``B=0`` is valid; ω_L = 0 in that case.
    Z : int
        Nuclear charge of the radiating ion (1 for hydrogen).
    n : int, optional
        Principal quantum number of the *upper* level (default 2).
        The width refers to the upper-level broadening only; the lower-level
        contribution is neglected, consistent with the semi-classical model.

    Returns
    -------
    float or ndarray
        Electron-impact half-width W_e [eV] (HWHM of the Lorentzian component
        at detuning Δω).  Always positive.

    Notes
    -----
    A negligible floor of 1e-10 eV is added by the caller in
    :func:`~starkzee.static_profile.calculate_static_profile` solely to
    prevent 0/0 in the Lorentzian at exactly zero density; the raw value
    returned here is the physical width without that floor.
    """
    prefactor = calculate_electron_impact_prefactor(Ne_m3, Te_ev)

    r2_avg = sum(
        (2*l + 1) * (n**2 / (2.0 * Z**2)) * (5.0*n**2 + 1.0 - 3.0*l*(l + 1.0))
        for l in range(n)
    ) / n**2

    if n <= 2:
        Cn = 1.5
    elif n == 3:
        Cn = 1.0
    elif n == 4:
        Cn = 0.75
    elif n == 5:
        Cn = 0.5
    else:
        Cn = 0.40

    omega_p = calculate_plasma_frequency(Ne_m3)
    omega_L = calculate_larmor_frequency(B)
    omega_e = calculate_configuration_frequency(Ne_m3, Te_ev)

    omega_aa_prime = 0.0  # state-to-state transition cutoff; non-zero for multi-electron atoms
    omega_c_rad = max(omega_p, omega_L, omega_e, omega_aa_prime)
    omega_c_ev = omega_c_rad * HBAR / E_CHARGE

    # Griem (Phys. Rev. A 16, 1979) approach: combine cutoffs in quadrature with the detuning
    # omega_p = calculate_plasma_frequency(Ne_m3)
    # omega_p_ev = omega_p * HBAR / E_CHARGE
    # delta_omega_s = (13.0 * n**2 * HBAR * Ne_m3**(2.0/3.0)) / (Z * M_E)
    # delta_omega_s_ev = delta_omega_s * HBAR / E_CHARGE
    # omega_c = sqrt(omega_p^2 + delta_omega_s^2 + delta_omega^2)
    # omega_c_ev = np.sqrt(omega_p_ev**2 + delta_omega_s_ev**2 + delta_omega_ev**2)

    g_val = gbk_model(delta_omega_ev, omega_c_ev, Te_ev, Z, n=n)

    width_ev = prefactor * r2_avg * (Cn + g_val)
    return width_ev


def electron_impact_width_zest(delta_omega_ev, Ne_m3, Te_ev, Z, n=2, model='gbk'):
    """Return the electron-impact half-width W_e(Δω) [eV] using the ZEST broadening model.

    Implements the ZEST electron broadening operator:

        W_e(Δω) = W₀ × ⟨r²_intra⟩_n × [G_n + G(Δω)]

    **Prefactor** W₀ = (4π/3) N_e √(2m_e/πkT_e) (ħ/m_e)² (ħ/e).

    **Intra-shell squared radius** ⟨r²_intra⟩_n is the mean over the n²-degenerate
    spatial basis of the intra-shell dipole sum:

        r²_intra,i = Σ_{j same shell} |⟨i|r|j⟩|²

    which uses only within-shell matrix elements, unlike the full ⟨r²⟩_{n,l} used by
    :func:`electron_impact_width`.  For hydrogen-like ions the per-l value is

        r²_intra,l = (9n²/4Z²) (n² − l(l+1) − 1)     [a₀²]

    derived from the intra-shell radial element ⟨n,l|r|n,l±1⟩ = (3n/2Z)√(n²−(l±1)²)
    and the angular sum factors C(l, l+1) = (l+1)/(2l+1), C(l, l−1) = l/(2l+1).
    The shell average is exact in closed form:

        ⟨r²_intra⟩_n = (9n²/8Z²)(n²−1)     [a₀²]

    This matches the average computed in ZEST's ``_get_level_widths_sp``.

    The difference from :func:`electron_impact_width` is in both the G-function and r²:

    - **r²**: intra-shell sum ⟨r²_intra⟩ < full diagonal ⟨r²⟩ (ratio ~0.41 at n=2, ~0.53 at n=3)
    - **Minimum impact parameter**: ZEST uses ρ_min = 1/κ_m (temperature-dependent);
      StarkZee uses the fixed n²a₀/(2Z).
    - **Cutoff frequency**: ZEST uses ω_p only; StarkZee uses max(ω_p, ω_e, ω_L).
    - **G_n / C_n values**: identical in both.

    Parameters
    ----------
    delta_omega_ev : float or array-like
        Frequency detuning Δω [eV].
    Ne_m3 : float
        Electron number density [m⁻³].
    Te_ev : float
        Electron temperature [eV].
    Z : int
        Nuclear charge.
    n : int, optional
        Principal quantum number of the upper level (default 2).
    model : {'gbk', 'lee', 'dufty'}, optional
        G-function approximation: ``'gbk'`` → :func:`gbk_zest_model`,
        ``'lee'`` → :func:`lee_model`, ``'dufty'`` → :func:`dufty_model`.

    Returns
    -------
    float or ndarray
        Electron-impact half-width W_e [eV].

    See Also
    --------
    electron_impact_width : StarkZee/Ferri GBK with fixed ρ_min and ω_e, ω_L cutoffs.
    """
    v_th = np.sqrt(Te_ev * E_CHARGE / M_E)
    const_factor = (E_CHARGE**2 * A_BOHR / (4.0 * np.pi * EPSILON_0 * HBAR))**2
    prefactor = (
        (4.0 * np.pi / 3.0) * Ne_m3
        * np.sqrt(2.0 / (np.pi * v_th**2))
        * const_factor
        * (HBAR / E_CHARGE)
    )

    # Intra-shell squared radius: mean of sum_{j same shell} |<i|r|j>|^2 over all
    # n^2 spatial basis states.  Exact closed form derived from the intra-shell
    # radial element <n,l|r|n,l+-1> = (3n/2Z)*sqrt(n^2-(l+-1)^2) and angular sum
    # factors C(l, l+1) = (l+1)/(2l+1), C(l, l-1) = l/(2l+1), giving
    # r2_intra_l = (9n^2/4Z^2)*(n^2 - l(l+1) - 1) and shell average (9n^2/8Z^2)*(n^2-1).
    r2_intra_avg = (9.0 * n**2 * (n**2 - 1)) / (8.0 * Z**2)

    if n <= 2:
        Gn = 1.5
    elif n == 3:
        Gn = 1.0
    elif n == 4:
        Gn = 0.75
    elif n == 5:
        Gn = 0.5
    else:
        Gn = 0.4

    if model == 'lee':
        g_val = lee_model(delta_omega_ev, Ne_m3, Te_ev, Z, n)
    elif model == 'dufty':
        g_val = dufty_model(delta_omega_ev, Ne_m3, Te_ev, Z, n)
    else:
        g_val = gbk_zest_model(delta_omega_ev, Ne_m3, Te_ev, Z, n)

    return prefactor * r2_intra_avg * (Gn + g_val)


# Accepted ``electron_model`` selectors for :func:`electron_impact_width_model`.
ELECTRON_MODELS = ('pppb', 'ferri', 'zest', 'zest-gbk', 'zest-lee', 'zest-dufty')


def electron_impact_width_model(delta_omega_ev, Ne_m3, Te_ev, B, Z, n=2,
                                electron_model='pppb'):
    """Return the electron-impact half-width W_e(Δω) [eV] from the selected model.

    Thin dispatcher that lets the profile solvers switch between the two published
    electron-impact prescriptions without changing any call site:

    - ``'pppb'`` (default) → :func:`electron_impact_width`.
      The PPPB / Ferri, Peyrusse & Calisti (2022) form: fixed minimum impact
      parameter ρ_min = n²a₀/(2Z), GBK G-function, and a **B-dependent** cutoff
      ω_c = max(ω_p, ω_e, ω_L).
    - ``'zest'`` / ``'zest-gbk'`` → :func:`electron_impact_width_zest` (``model='gbk'``).
    - ``'zest-lee'`` → ZEST with Lee's analytic G-function (``model='lee'``).
    - ``'zest-dufty'`` → ZEST with the Dufty RPA G-function (``model='dufty'``).

    The ZEST variants use ω_p as the only cutoff (κ_m-based ρ_min, no Larmor
    term), consistent with the ZEST formulation, so ``B`` does not enter them;
    it is accepted here only to give every model a uniform signature.

    Parameters
    ----------
    delta_omega_ev : float or array-like
        Frequency detuning from line center Δω [eV].
    Ne_m3 : float
        Electron number density [m⁻³].
    Te_ev : float
        Electron temperature [eV].
    B : float
        Magnetic field [T].  Used only by the ``'pppb'`` model (Larmor cutoff).
    Z : int
        Nuclear charge.
    n : int, optional
        Principal quantum number of the upper level (default 2).
    electron_model : str, optional
        One of :data:`ELECTRON_MODELS` (default ``'pppb'``).

    Returns
    -------
    float or ndarray
        Electron-impact half-width W_e [eV].

    See Also
    --------
    electron_impact_width : PPPB model.
    electron_impact_width_zest : ZEST model (GBK / Lee / Dufty G-functions).
    """
    m = electron_model.lower()
    if m in ('pppb', 'ferri'):
        return electron_impact_width(delta_omega_ev, Ne_m3, Te_ev, B, Z, n=n)
    if m in ('zest', 'zest-gbk'):
        return electron_impact_width_zest(delta_omega_ev, Ne_m3, Te_ev, Z, n=n, model='gbk')
    if m == 'zest-lee':
        return electron_impact_width_zest(delta_omega_ev, Ne_m3, Te_ev, Z, n=n, model='lee')
    if m == 'zest-dufty':
        return electron_impact_width_zest(delta_omega_ev, Ne_m3, Te_ev, Z, n=n, model='dufty')
    raise ValueError(
        f"Unknown electron_model {electron_model!r}; choose one of {ELECTRON_MODELS}."
    )


def electron_impact_r2_scaling(eigenvectors, n, Z):
    r"""Per-eigenstate ⟨k|r²|k⟩ / ⟨r²⟩_avg — the electron-impact **operator** diagonal.

    The semi-classical electron-impact width is linear in the upper-state
    operator ``R⃗·R⃗ = r²`` (Ferri/PPPB Eq. 19; ZEST Eq. 8).  StarkZee's scalar
    width functions evaluate it with the *shell-averaged* ``⟨r²⟩_avg``; ZEST and
    PPPB instead use the operator's value resolved on each Stark-Zeeman dressed
    state.  Because the width is linear in r², the operator-diagonal width of
    dressed state ``k`` is simply

        W_e^{op}(k) = W_e^{scalar} × ⟨k|r²|k⟩ / ⟨r²⟩_avg ,

    i.e. the scalar width times the factor returned here.

    ``r²`` is purely radial, so in the ``|n, l, m_l, m_s⟩`` basis it is diagonal
    with value ``⟨r²⟩_{n,l} = (n²/2Z²)[5n²+1−3l(l+1)]``; the Stark-Zeeman
    eigenstates mix l, so ⟨k|r²|k⟩ varies from state to state.  The factor
    averages to 1 over the shell (trace preserved), redistributing width among
    components.

    **This is the diagonal of the broadening operator.**  Keeping the diagonal
    only (and discarding off-diagonal ⟨k|r²|k'⟩) is exactly the ``c_k = 0``
    approximation that recovers the **ZEST operator**.  The off-diagonal part is
    what generates the complex SDT intensity ``a_k + i c_k`` of the full **PPPB**
    operator (non-Hermitian Liouvillian) — *not yet implemented*; see the
    REVIEW note below.

    Parameters
    ----------
    eigenvectors : ndarray, shape (2n², 2n²)
        Stark-Zeeman eigenvectors (columns) in the ``|n,l,m_l,m_s⟩`` basis, as
        returned by the profile solvers' ``eigh`` call.
    n : int
        Principal quantum number of the shell these eigenvectors belong to.
    Z : int
        Nuclear charge.

    Returns
    -------
    ndarray, shape (2n²,)
        Per-eigenstate scaling factor ⟨k|r²|k⟩ / ⟨r²⟩_avg (mean ≈ 1).

    Notes
    -----
    REVIEW (future): only the operator **diagonal** is used (``c_k = 0`` →
    ZEST).  The full PPPB operator keeps the off-diagonal ⟨k|r²|k'⟩, builds the
    non-Hermitian Liouvillian, and yields the complex intensity ``a_k + i c_k``
    (estimated ~1–2 % asymmetry for Hβ at 1 kT).  Implementing it also calls for
    the lower-manifold ``d†·d`` piece (currently neglected, as in the scalar
    model).  See ``FFM_implementation_plan.md``.
    """
    from starkzee.radiator import build_basis
    basis = build_basis(n)
    r2_diag = np.array(
        [(n**2 / (2.0 * Z**2)) * (5.0 * n**2 + 1.0 - 3.0 * s.l * (s.l + 1.0)) for s in basis]
    )
    r2_avg = r2_diag.mean()
    return (np.abs(eigenvectors)**2).T @ r2_diag / r2_avg
