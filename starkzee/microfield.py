# Plasma electric microfield distributions

import numpy as np
import math
from functools import lru_cache
from scipy.integrate import quad, quad_vec
from scipy.constants import epsilon_0 as EPSILON_0, e as E_CHARGE
from scipy.special import gamma as gamma_fn, erf



def calculate_normal_field(Ne_m3, Z_bar=1.0):
    """Return the Holtsmark normal electric field F₀ [V m⁻¹] and mean inter-particle distance r_e [m].

    F₀ is the characteristic field strength that scales the microfield distribution.
    It is defined as the Coulomb field of an ion with average charge Z_bar * e at the
    mean inter-particle distance (Wigner-Seitz radius) r_e:

        r_e = (3 * Z_bar / (4π N_e))^(1/3)
        F₀  = Z_bar * e / (4π ε₀ r_e²)

    Note that the screening parameter `a` used in screened microfield distributions (like
    Hooper or Potekhin) is defined as the ratio of the mean inter-particle distance `r_e`
    to the Debye length `λ_D`, i.e., a = r_e / λ_D. The mean inter-particle distance
    characterizes the average spacing between ions in the plasma, whereas the Debye length
    characterizes the scale over which electrostatic fields are screened by the plasma.

    Parameters
    ----------
    Ne_m3 : float
        Electron number density [m⁻³].
    Z_bar : float, optional
        Average background ion charge (default is 1.0).

    Returns
    -------
    F0 : float
        Normal (Holtsmark) electric field [V m⁻¹].
    re : float
        Mean inter-particle distance (Wigner-Seitz radius) [m].
    """
    re = (3.0 * Z_bar / (4.0 * np.pi * Ne_m3))**(1.0 / 3.0)
    F0 = (Z_bar * E_CHARGE) / (4.0 * np.pi * EPSILON_0 * (re**2))
    return F0, re


def calculate_debye_length(Te_ev, Ne_m3):
    """Return the electron Debye screening length λ_D [m].

    The Debye length characterizes the spatial scale over which the plasma
    screens individual Coulomb fields:

        λ_D = √(ε₀ T_e / (N_e e))     [T_e in eV, so e T_e gives k_B T_e in J]

    Parameters
    ----------
    Te_ev : float
        Electron temperature [eV].
    Ne_m3 : float
        Electron number density [m⁻³].

    Returns
    -------
    float
        Debye length [m].
    """
    lambda_D = np.sqrt(EPSILON_0 * Te_ev / (Ne_m3 * E_CHARGE))
    return lambda_D


def calculate_multispecies_debye_length(Te_ev, Ne_m3, species_charges=None, species_concentrations=None):
    """Return the multi-species Debye length λ_D [m].

    Extends the single-species Debye length to a plasma containing multiple ion
    species by summing their charge-weighted contributions to the screening:

        λ_D = √(ε₀ T_e / (N_e e × (1 + Σ_i X_i Z_i²)))

    where X_i = N_i / N_e is the fractional concentration and Z_i the charge of
    species i.  The ``1`` in the denominator accounts for the electron contribution.
    If no species information is given the function falls back to the single-species
    result :func:`calculate_debye_length`.

    Parameters
    ----------
    Te_ev : float
        Electron (and ion) temperature [eV].  The formula assumes T_e = T_i.
    Ne_m3 : float
        Electron number density [m⁻³].
    species_charges : list of float, optional
        Charge numbers Z_i for each ion species.
    species_concentrations : list of float, optional
        Relative concentrations X_i = N_i / N_e for each species.  Must have
        the same length as ``species_charges``.

    Returns
    -------
    float
        Multi-species Debye length [m].
    """
    if species_charges is None or species_concentrations is None:
        return calculate_debye_length(Te_ev, Ne_m3)

    sum_term = sum(x * z**2 for x, z in zip(species_concentrations, species_charges))
    lambda_D = np.sqrt(EPSILON_0 * Te_ev / (Ne_m3 * E_CHARGE * (1.0 + sum_term)))
    return lambda_D


def calculate_coupling_parameter(Z_bar, Ti_ev, R_ii):
    """Return the ion-ion coupling parameter Gamma_ii.

    Parameters
    ----------
    Z_bar : float
        Average ion charge.
    Ti_ev : float
        Ion temperature [eV].
    R_ii : float
        Wigner-Seitz radius [m].

    Returns
    -------
    float
        Ion-ion coupling parameter.
    """
    return (Z_bar**2 * E_CHARGE) / (4.0 * np.pi * EPSILON_0 * Ti_ev * R_ii)


def _holtsmark_integrand(y, beta):
    """Integrand y sin(βy) exp(−y^{3/2}) for the Holtsmark characteristic function."""
    if y == 0:
        return 0.0
    return y * math.sin(beta * y) * math.exp(-y**1.5)


def _holtsmark_integrand_vec(y, beta_arr):
    """Vectorized integrand y sin(βy) exp(−y^{3/2}) for the Holtsmark characteristic function."""
    if y == 0:
        return np.zeros_like(beta_arr)
    return y * np.sin(beta_arr * y) * np.exp(-y**1.5)


def holtsmark_distribution(beta, method='vectorized'):
    """Return the Holtsmark microfield probability density W(β) at reduced field β.

    Supports both scalar and array-like inputs.

    Parameters
    ----------
    beta : float or ndarray
        Reduced electric field β = F / F₀.
    method : {'vectorized', 'exact', 'potekhin'}, optional
        Computation method (default is 'vectorized'):
        - 'vectorized': fast Gauss-Legendre quadrature.
        - 'exact': exact adaptive quadrature.
        - 'potekhin': Zest-compatible Potekhin analytical fit at Gamma=0.
    """
    method_lower = method.lower()
    if method_lower == 'exact':
        if np.ndim(beta) == 0:
            return _holtsmark_distribution_cached(float(beta))
        beta_arr = np.asarray(beta, dtype=float)
        val, _ = quad_vec(_holtsmark_integrand_vec, 0, 15, args=(beta_arr,))
        res = (2.0 * beta_arr / np.pi) * val
        return np.maximum(res, 0.0)
    elif method_lower == 'vectorized':
        # Vectorized Gauss-Legendre quadrature (96 nodes)
        nodes, weights = np.polynomial.legendre.leggauss(96)
        y = 7.5 * nodes + 7.5
        w = 7.5 * weights
        const_factor = y * np.exp(-y**1.5) * w
        beta_arr = np.asarray(beta, dtype=float)

        if np.ndim(beta_arr) == 0:
            if beta_arr > 20.0:
                return _holtsmark_distribution_cached(float(beta_arr))
            sin_term = np.sin(beta_arr * y)
            integral = np.sum(sin_term * const_factor)
            res = (2.0 * beta_arr / np.pi) * integral
            return float(max(0.0, res)) if beta_arr > 1e-5 else 0.0
        else:
            res = np.zeros_like(beta_arr)
            mask_small = (beta_arr > 1e-5) & (beta_arr <= 20.0)
            mask_large = beta_arr > 20.0

            if np.any(mask_small):
                b_small = beta_arr[mask_small]
                sin_term = np.sin(b_small[:, None] * y[None, :])
                integral = np.sum(sin_term * const_factor[None, :], axis=1)
                res[mask_small] = (2.0 * b_small / np.pi) * integral

            if np.any(mask_large):
                res[mask_large] = [
                    _holtsmark_distribution_cached(float(b)) for b in beta_arr[mask_large]
                ]

            return np.maximum(res, 0.0)
    elif method_lower == 'potekhin':
        return potekhin_distribution(beta, gamma=0.0, s=0.0, charged=False)
    else:
        raise ValueError(f"Unknown method '{method}'. Must be 'vectorized', 'exact', or 'potekhin'.")


@lru_cache(maxsize=None)
def _holtsmark_distribution_cached(beta):
    """Cached scalar backend for holtsmark_distribution."""
    if beta <= 1e-5:
        return 0.0
    val, _ = quad(_holtsmark_integrand, 0, 15, args=(beta,), limit=100)
    w_beta = (2.0 * beta / np.pi) * val
    return max(0.0, w_beta)


def _hooper_integrand_vec(y, beta_arr, a, charged):
    """Vectorized integrand for Hooper screened microfield distribution."""
    if y == 0:
        return np.zeros_like(beta_arr)
    fac = 1.5 if charged else 1.0
    screening = (1.0 + fac * (a**2) / (y**2 + 1e-8))**(-0.75)
    return y * np.sin(beta_arr * y) * np.exp(-y**1.5 * screening)


def hooper_distribution(beta, a, charged=True, method='vectorized'):
    """Return the Hooper screened microfield probability density W(β, a).

    Supports both scalar and array-like inputs.

    Parameters
    ----------
    beta : float or ndarray
        Reduced electric field β = F / F₀.
    a : float
        Screening parameter a = r_e / λ_D.
    charged : bool, optional
        True for a charged point (ion radiator), False for neutral (atom radiator).
        Default is True.
    method : {'vectorized', 'exact'}, optional
        Computation method (default is 'vectorized'):
        - 'vectorized': fast Gauss-Legendre quadrature.
        - 'exact': exact adaptive quadrature.
    """
    method_lower = method.lower()
    if method_lower == 'exact':
        if np.ndim(beta) == 0:
            return _hooper_distribution_cached(float(beta), float(a), charged)
        beta_arr = np.asarray(beta, dtype=float)
        val, _ = quad_vec(_hooper_integrand_vec, 0, 15, args=(beta_arr, float(a), charged))
        res = (2.0 * beta_arr / np.pi) * val
        return np.maximum(res, 0.0)
    elif method_lower == 'vectorized':
        # Vectorized Gauss-Legendre quadrature (96 nodes)
        nodes, weights = np.polynomial.legendre.leggauss(96)
        y = 7.5 * nodes + 7.5
        w = 7.5 * weights

        fac = 1.5 if charged else 1.0
        screening = (1.0 + fac * (float(a)**2) / (y**2 + 1e-8))**(-0.75)
        const_factor = y * np.exp(-y**1.5 * screening) * w
        beta_arr = np.asarray(beta, dtype=float)

        if np.ndim(beta_arr) == 0:
            if beta_arr > 20.0:
                return _hooper_distribution_cached(float(beta_arr), float(a), charged)
            sin_term = np.sin(beta_arr * y)
            integral = np.sum(sin_term * const_factor)
            res = (2.0 * beta_arr / np.pi) * integral
            return float(max(0.0, res)) if beta_arr > 1e-5 else 0.0
        else:
            res = np.zeros_like(beta_arr)
            mask_small = (beta_arr > 1e-5) & (beta_arr <= 20.0)
            mask_large = beta_arr > 20.0

            if np.any(mask_small):
                b_small = beta_arr[mask_small]
                sin_term = np.sin(b_small[:, None] * y[None, :])
                integral = np.sum(sin_term * const_factor[None, :], axis=1)
                res[mask_small] = (2.0 * b_small / np.pi) * integral

            if np.any(mask_large):
                res[mask_large] = [
                    _hooper_distribution_cached(float(b), float(a), charged) for b in beta_arr[mask_large]
                ]

            return np.maximum(res, 0.0)
    else:
        raise ValueError(f"Unknown method '{method}'. Must be 'vectorized' or 'exact'.")


@lru_cache(maxsize=None)
def _hooper_distribution_cached(beta, a, charged=True):
    """Cached scalar backend for hooper_distribution."""
    if beta <= 1e-5:
        return 0.0

    def hooper_integrand(y, beta, a, charged):
        if y == 0:
            return 0.0
        fac = 1.5 if charged else 1.0
        screening = (1.0 + fac * (a**2) / (y**2 + 1e-8))**(-0.75)
        return y * math.sin(beta * y) * math.exp(-y**1.5 * screening)

    val, _ = quad(hooper_integrand, 0, 15, args=(beta, a, charged), limit=100)
    w_beta = (2.0 * beta / np.pi) * val
    return max(0.0, w_beta)





def _Q_neutral_unscreened(beta, gamma):
    """Cumulative distribution Q(beta) for a Neutral Point at s = 0."""
    alpha = [14.600, 103.20, 11.127, 16.178]
    beta_n = [0.41, 1.54, 0.58, 0.60]
    gamma_n = [0.707, 1.64, 0.572, 0.915]

    q = [alpha[i] * (1.0 + beta_n[i] * gamma)**(-gamma_n[i]) for i in range(4)]

    beta = np.maximum(beta, 1e-30)  # Avoid division by zero
    b3 = beta**3
    b45 = beta**4.5
    b6 = beta**6

    num = q[0] * b3 - 1.33 * b45 + b6
    den = q[1] + q[2] * beta**2 + q[3] * b3 - (1.0 / 3.0) * b45 + b6
    return num / den


def _Q_Mayer(beta, gamma_eff):
    """Mayer distribution Q_M(beta, gamma_eff)."""
    x = 0.5 * gamma_eff * beta**2
    # Taylor expansion for small x to avoid numerical cancellation/underflow
    small_mask = x < 1e-4
    res = np.zeros_like(x)

    # Large x case
    res[~small_mask] = erf(np.sqrt(x[~small_mask])) - 2.0 * np.sqrt(x[~small_mask] / np.pi) * np.exp(-x[~small_mask])

    # Small x case: Q_M approx 4/(3 * sqrt(pi)) * x^1.5
    res[small_mask] = (4.0 / (3.0 * np.sqrt(np.pi))) * x[small_mask]**1.5
    return res


def _Q_charged_unscreened(beta, gamma):
    """Cumulative distribution Q(beta) for a Charged Point at s = 0."""
    gamma_eff = 0.774 + gamma**0.25 + gamma

    q = 9.19 + 2.178 * gamma**1.64
    gamma_prime = gamma / (1.0 + 0.19 * gamma**0.627)

    beta = np.maximum(beta, 1e-30)
    b2 = beta**2
    b3 = beta**3
    b45 = beta**4.5
    b6 = beta**6

    exp_factor = np.exp(-gamma_prime * np.sqrt(beta))

    num_Q0 = q * b3 * exp_factor + b6
    term_den = (2.25 * np.pi) * q * (1.0 + gamma**0.6)**(-2.75) + 15.3 * b2 + 1.238 * q * b3 + b45
    den_Q0 = term_den * exp_factor + b6
    Q0 = num_Q0 / den_Q0

    QM = _Q_Mayer(beta, gamma_eff)

    Q = (Q0 + 0.873 * np.sqrt(gamma) * QM) / (1.0 + 0.873 * np.sqrt(gamma))
    return Q


def _Q_neutral_screened(beta, gamma, s):
    """Cumulative distribution Q(beta) for a Neutral Point at s > 0."""
    g = np.sqrt(0.08 + gamma)
    a0 = (97.0 * s**2 + 1.29 * s**7) / (1.0 + 3.1e-3 * s**5) + (59.0 + 8.1 * s**2) * g

    alpha = (0.068 + 0.038 * s**7) / (1.0 + 0.030 * s**7)
    a1 = (1.16 / (1.0 + 0.188 * s**6)) * (1.0 + (103.0 * g**alpha) / (1.0 + 0.33 * s))
    a2 = (95.0 * s) / (1.0 + 6.0e-3 * s**7) + 1.2 * s**2 * g
    a3 = 27.0 * s**3 + 36.0 * g
    a4 = ((1.894 + s) / (2.0 + s)) * a0

    beta = np.maximum(beta, 1e-30)
    b3 = beta**3
    b45 = beta**4.5
    b6 = beta**6

    num = a0 * b3 - 2.0 * b45 + b6
    den = a1 + a2 * beta + a3 * beta**2 + a4 * b3 - b45 + b6
    return num / den


def _P_charged_screened(beta, gamma, s):
    """Probability density P(beta) for a Charged Point at s > 0."""
    # Parameter evaluations as functions of s
    A1 = 0.59 + 2540.0 * s**4 + 3.0 * s**14
    A2 = 0.55 + (10.0 * np.sqrt(s) + 2.0 * s**4.5) / (1.0 + 20.0 * np.sqrt(s))
    A3 = 2.17e-3 * s**5
    A4 = 14.8 / (1.0 + 117.0 * s**3.5)

    a0 = 1.15 + 2.0 * s**1.8

    alpha1 = 0.1 + 1.1 / (1.0 + 0.145 * s**3)
    alpha2 = 5.4 / (1.0 + 20.0 * s**2) + 1.1 / (1.0 + 14.0 * s**0.35)

    B1 = 0.386 + 300.0 * s**2 + 1.1 * s**9.5
    B2 = 0.038 + 0.79 * s**0.75
    B3 = 3.7e-3 * s**5.5 / (1.0 + 4.0e-3 * s**9)

    b0 = (1.0 + 0.54 * s**2.5) / (1.0 + 0.07 * s)

    gamma1 = 0.1 + 1.1 / (1.0 + 0.174 * s**2.5)
    gamma2 = 5.4 / (1.0 + 21.0 * s**1.5) + 1.1 / (1.0 + 19.0 * s**0.16)

    c = (0.097 / (1.0 + 210.0 * s**2.5)) * np.exp(-1.3 * s**1.5)

    # Parameters as functions of gamma
    A = (A1 / (1.0 + A4 * gamma)) * ((1.0 + A2 * gamma**2) / (1.0 + A3 * gamma**4))
    a = a0 + 0.5 * gamma
    alpha = (alpha1 + 2.0 * alpha2 * gamma) / (1.0 + alpha2 * gamma)

    B = B1 / (1.0 + B2 * gamma**2 + B3 * gamma**4)
    b = b0 + 0.25 * gamma
    g_param = (gamma1 + 1.5 * gamma2 * gamma) / (1.0 + gamma)

    # Third term integration factor
    def term3_integrand(b_val):
        # Guard exponential against overflow
        exp_arg = -gamma * np.sqrt(b_val)
        if exp_arg < -100.0:
            return 0.0
        return b_val**2 * np.exp(exp_arg) / (1.0 + c * b_val**4.5)

    term3, _ = quad(term3_integrand, 0.0, np.inf, limit=200)

    # Compute Normalization Constant SN
    term1 = A * gamma_fn(3.0 / alpha) / (alpha * a**(3.0 / alpha))
    term2 = B * gamma_fn(3.0 / g_param) / (g_param * b**(3.0 / g_param))

    SN = 1.0 / (term1 + term2 + term3)

    # Compute P(beta)
    b_arr = np.asarray(beta)

    # Avoid overflows in exponentials
    exp1_arg = -a * b_arr**alpha
    exp2_arg = -b * b_arr**g_param
    exp3_arg = -gamma * np.sqrt(b_arr)

    # Compute terms safely
    t1 = A * np.exp(np.clip(exp1_arg, -100, 100))
    t2 = B * np.exp(np.clip(exp2_arg, -100, 100))
    t3 = np.exp(np.clip(exp3_arg, -100, 100)) / (1.0 + c * b_arr**4.5)

    res = SN * b_arr**2 * (t1 + t2 + t3)
    return res


def _P_from_Q_grid(Q_func, beta_grid, gamma, s=0.0):
    """Compute P(beta) from a cumulative Q(beta) function using central differences."""
    n = len(beta_grid)
    if n == 1:
        # Fallback for single value to calculate derivative via local central difference
        b = beta_grid[0]
        h = 1e-5 if b > 1e-5 else 1e-7
        grid = np.array([b - h, b + h])
        q_vals = Q_func(grid, gamma) if s == 0.0 else Q_func(grid, gamma, s)
        val = (q_vals[1] - q_vals[0]) / (2.0 * h)
        return np.array([max(0.0, val)])
    elif n < 1:
        return np.zeros_like(beta_grid)

    Q_vals = Q_func(beta_grid, gamma) if s == 0.0 else Q_func(beta_grid, gamma, s)

    P_vals = np.zeros_like(beta_grid)
    # Central differences for internal points
    h = beta_grid[1:] - beta_grid[:-1]
    P_vals[1:-1] = (Q_vals[2:] - Q_vals[:-2]) / (beta_grid[2:] - beta_grid[:-2])

    # Forward and backward differences for edge points
    P_vals[0] = (Q_vals[1] - Q_vals[0]) / h[0]
    P_vals[-1] = (Q_vals[-1] - Q_vals[-2]) / h[-1]

    # Clip negative values due to numerical noise
    return np.maximum(P_vals, 0.0)


def potekhin_distribution(beta, gamma, s=0.0, charged=True):
    """Return the Potekhin screened/unscreened microfield probability density.

    Parameters
    ----------
    beta : float or ndarray
        Dimensionless field strength beta = F/F_0.
    gamma : float
        Ion-ion coupling parameter Gamma_ii.
    s : float, optional
        Screening parameter s = R_ii / lambda_e (default 0.0).
    charged : bool, optional
        True for a charged point (ion radiator), False for neutral (atom radiator).

    Returns
    -------
    float or ndarray
        Potekhin probability density values on the beta grid.
    """
    is_scalar = np.isscalar(beta)
    beta_arr = np.atleast_1d(np.asarray(beta, dtype=float))

    if s <= 1e-5:
        # Unscreened Coulomb limit
        if charged:
            res = _P_from_Q_grid(_Q_charged_unscreened, beta_arr, gamma)
        else:
            res = _P_from_Q_grid(_Q_neutral_unscreened, beta_arr, gamma)
    else:
        # Screened Yukawa potential
        if charged:
            res = _P_charged_screened(beta_arr, gamma, s)
        else:
            res = _P_from_Q_grid(_Q_neutral_screened, beta_arr, gamma, s)

    if is_scalar:
        return res[0]
    return res


def microfield_quadrature(Ne_m3, Te_ev, num_points=50, max_beta=10.0, use_screening=True,
                          species_charges=None, species_concentrations=None, custom_table_path=None,
                          charged=True, Z_bar=1.0):
    """Build a quadrature grid of plasma electric microfield magnitudes and weights.

    Discretises the microfield integral ∫ W(F) dF over a uniform grid of
    ``num_points`` values of the reduced field β = F / F₀ in [0, max_beta].
    The returned arrays satisfy:

        Σ_i weight_i ≈ 1

    so that integrals of the form ∫ f(F) W(F) dF can be approximated by a
    simple weighted sum Σ_i weight_i × f(fields_i).

    The distribution W(β) is one of:

    - **Holtsmark** (``use_screening=False``): unscreened, valid when r_e ≪ λ_D.
    - **Hooper** (``use_screening=True``, default): screened by the Debye length,
      appropriate for most laboratory plasmas.
    - **Custom table** (``custom_table_path`` provided): loads a user-supplied
      two-column file [β, W(β)] (e.g. from APEX or MD simulations) and
      interpolates it onto the β grid.

    Parameters
    ----------
    Ne_m3 : float
        Electron number density [m⁻³].
    Te_ev : float
        Electron temperature [eV].  Used to compute the Debye length when
        ``use_screening=True``.
    num_points : int, optional
        Number of quadrature points in β (default 50).  20 points is usually
        sufficient; 50 gives better accuracy in the far wings.
    max_beta : float, optional
        Upper limit of the β grid (default 10).  W(β) is negligible beyond
        β ≈ 5–8 for typical screening parameters.
    use_screening : bool, optional
        If True (default) use the Hooper screened distribution; if False use
        the unscreened Holtsmark distribution.
    species_charges : list of float, optional
        Ion charge numbers for multi-species Debye screening.
    species_concentrations : list of float, optional
        Relative ion concentrations N_i / N_e for each species.
    custom_table_path : str or None, optional
        Path to a plain-text file with columns [β, W(β)].  If loading fails the
        function falls back to the analytical Hooper distribution with a warning.
    charged : bool, optional
        True for a charged point (ion radiator), False for neutral (atom radiator).
        Default is True.
    Z_bar : float, optional
        Average background ion charge (default is 1.0).

    Returns
    -------
    fields : ndarray, shape (num_points,)
        Electric field magnitudes F = β × F₀ [V m⁻¹].
    weights : ndarray, shape (num_points,)
        Quadrature weights W(β) dβ (dimensionless, sum ≈ 1).

    Notes
    -----
    The grid starts at β = 0 (F = 0) where W(0) = 0.  Points with weight
    ≤ 1e-15 are skipped by the profile integrator for efficiency.

    The screening parameter `a` is computed as the ratio of the mean inter-particle
    distance `r_e` (which scales with `Z_bar`) to the Debye length `λ_D`:

        a = r_e / λ_D

    The Debye length represents the scale over which electrostatic fields are screened
    by the plasma, whereas the mean inter-particle distance (Wigner-Seitz radius) represents
    the typical spacing between particles.

    Results are cached internally; repeated calls with identical arguments
    return the pre-computed arrays without re-evaluating the distribution
    integrals.
    """
    sc  = tuple(species_charges)       if species_charges       is not None else None
    scc = tuple(species_concentrations) if species_concentrations is not None else None
    return _microfield_quadrature_impl(Ne_m3, Te_ev, num_points, max_beta,
                                       use_screening, sc, scc, custom_table_path, charged, Z_bar)


@lru_cache(maxsize=None)
def _microfield_quadrature_impl(Ne_m3, Te_ev, num_points, max_beta, use_screening,
                                species_charges, species_concentrations, custom_table_path, charged, Z_bar):
    """Cached backend for :func:`microfield_quadrature`."""
    F0, re = calculate_normal_field(Ne_m3, Z_bar)
    beta_grid = np.linspace(0.0, max_beta, num_points)

    w_grid = None
    if custom_table_path is not None:
        try:
            print(f"Loading custom microfield database from: {custom_table_path} ...")
            data = np.loadtxt(custom_table_path)
            custom_beta = data[:, 0]
            custom_W    = data[:, 1]
            w_grid = np.interp(beta_grid, custom_beta, custom_W, left=0.0, right=0.0)
        except Exception as e:
            print(f"Error loading custom microfield file, falling back to analytical Hooper: {e}")

    if w_grid is None:
        if use_screening:
            lambda_D = calculate_multispecies_debye_length(Te_ev, Ne_m3, species_charges, species_concentrations)
            a = re / lambda_D
        else:
            a = 0.0

        if a > 0:
            w_grid = hooper_distribution(beta_grid, a, charged=charged)
        else:
            w_grid = holtsmark_distribution(beta_grid)

    dbeta = beta_grid[1] - beta_grid[0]
    total_area = np.sum(w_grid) * dbeta
    if total_area > 0:
        w_grid = w_grid / total_area

    fields  = beta_grid * F0
    weights = w_grid * dbeta

    return fields, weights


