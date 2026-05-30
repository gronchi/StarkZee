# Broadening Component: Electron impact broadening for starkzee

import numpy as np
from scipy.special import exp1
from scipy.constants import hbar as HBAR, m_e as M_E, e as E_CHARGE, epsilon_0 as EPSILON_0, k as K_B
from starkzee.utils import RYDBERG_EV


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
    """Return the ion-configuration change frequency ω_e = 2π / τ_e [rad s⁻¹].

    τ_e is the mean time for an electron to cross the inter-particle distance
    r_e at the thermal velocity v_th:

        r_e  = (3 / 4π N_e)^(1/3)     — mean inter-electron spacing
        v_th = √(k_B T_e / m_e)        — thermal speed (non-relativistic)
        τ_e  = r_e / v_th
        ω_e  = 2π / τ_e

    ω_e represents the typical rate at which the local electric field seen by
    the radiator changes due to electron motion, and is used as a lower cutoff
    in the GBK broadening model.

    Parameters
    ----------
    Ne_m3 : float
        Electron number density [m⁻³].
    Te_ev : float
        Electron temperature [eV].

    Returns
    -------
    float
        Configuration-change angular frequency [rad s⁻¹].
    """
    re = (3.0 / (4.0 * np.pi * Ne_m3))**(1.0 / 3.0)
    v_th = np.sqrt(Te_ev * E_CHARGE / M_E)
    tau_e = re / v_th
    return 2.0 * np.pi / tau_e


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

    with E_H = 2 Ry the Hartree energy and T_e in eV.  At line centre (Δω = 0)
    G reduces to ½ E₁((n²/2Z)² ω_c² / (E_H T_e)) ≈ a positive constant; in
    the far wings where Δω ≫ ω_c the argument y grows and G → 0, suppressing
    the broadening at large detunings (the impact approximation breaks down).

    The cutoff frequency ω_c = max(ω_p, ω_e, ω_L) prevents the logarithm
    from diverging at small impact parameters and encodes the transition
    from the impact regime to the quasi-static regime.

    Parameters
    ----------
    delta_omega_ev : float or array-like
        Frequency detuning from line centre Δω [eV].
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
    Griem, Baranger, Kolb & Oertel, Phys. Rev. 116, 4 (1959).
    """
    E_H = 2.0 * RYDBERG_EV

    num = (delta_omega_ev**2 + omega_c_ev**2)
    den = E_H * Te_ev
    y = ((n**2) / (2.0 * Z))**2 * (num / den)

    g_val = 0.5 * exp1(np.maximum(1e-15, y))
    return g_val


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

    **Strong-collision constant C_n** (from Table 1 of Ferri et al. 2021):

    =========  ======
    n          C_n
    =========  ======
    ≤ 2        1.50
    3, 4       0.75
    ≥ 5        0.40
    =========  ======

    **Cutoff frequency** ω_c = max(ω_p, ω_L, ω_e) where ω_p is the plasma
    frequency, ω_L the electron Larmor frequency, and ω_e the configuration
    change frequency.  The magnetic field raises ω_c at high B, reducing the
    resonant broadening contribution.

    Parameters
    ----------
    delta_omega_ev : float or array-like
        Frequency detuning from line centre Δω [eV].  Pass ``0.0`` for the
        on-resonance (line-centre) width.
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
    A small floor of 1e-4 eV is added by the caller in
    :func:`~starkzee.static_profile.calculate_static_profile` to
    prevent numerical singularities at zero density; the raw value returned
    here is the physical width without that floor.
    """
    prefactor = calculate_electron_impact_prefactor(Ne_m3, Te_ev)

    r2_avg = sum(
        (2*l + 1) * (n**2 / (2.0 * Z**2)) * (5.0*n**2 + 1.0 - 3.0*l*(l + 1.0))
        for l in range(n)
    ) / n**2

    if n <= 2:
        Cn = 1.5
    elif n <= 4:
        Cn = 0.75
    else:
        Cn = 0.40

    omega_p = calculate_plasma_frequency(Ne_m3)
    omega_L = calculate_larmor_frequency(B)
    omega_e = calculate_configuration_frequency(Ne_m3, Te_ev)

    omega_c_rad = max(omega_p, omega_L, omega_e)
    omega_c_ev = omega_c_rad * HBAR / E_CHARGE

    g_val = gbk_model(delta_omega_ev, omega_c_ev, Te_ev, Z, n=n)

    width_ev = prefactor * r2_avg * (Cn + g_val)
    return width_ev
