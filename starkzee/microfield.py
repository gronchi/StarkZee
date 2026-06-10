# Plasma electric microfield distributions

import numpy as np
from functools import lru_cache
from scipy.integrate import quad
from scipy.constants import epsilon_0 as EPSILON_0, e as E_CHARGE


def calculate_normal_field(Ne_m3):
    """Return the Holtsmark normal electric field F₀ [V m⁻¹] and mean inter-particle distance r_e [m].

    F₀ is the characteristic field strength that scales the Holtsmark distribution.
    It is defined as the Coulomb field of one electron at the mean inter-particle
    spacing r_e:

        r_e = (3 / 4π N_e)^(1/3)
        F₀  = e / (4π ε₀ r_e²)

    Parameters
    ----------
    Ne_m3 : float
        Electron number density [m⁻³].

    Returns
    -------
    F0 : float
        Normal (Holtsmark) electric field [V m⁻¹].
    re : float
        Mean inter-electron distance [m].
    """
    re = (3.0 / (4.0 * np.pi * Ne_m3))**(1.0 / 3.0)
    F0 = E_CHARGE / (4.0 * np.pi * EPSILON_0 * (re**2))
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


def _holtsmark_integrand(y, beta):
    """Integrand y sin(βy) exp(−y^{3/2}) for the Holtsmark characteristic function."""
    if y == 0:
        return 0.0
    return y * np.sin(beta * y) * np.exp(-y**1.5)


@lru_cache(maxsize=None)
def holtsmark_distribution(beta):
    """Return the Holtsmark microfield probability density W(β) at reduced field β.

    The Holtsmark distribution describes the statistical distribution of electric
    microfields in an ideal, unscreened, spatially uncorrelated plasma of point
    charges.  It is computed via the Fourier transform:

        W(β) = (2β/π) ∫₀^∞ y sin(βy) exp(−y^{3/2}) dy

    where β = F / F₀ is the field normalized to the Holtsmark normal field F₀.
    W(β) is normalized so that ∫₀^∞ W(β) dβ = 1.

    Parameters
    ----------
    beta : float
        Reduced electric field β = F / F₀ ≥ 0.

    Returns
    -------
    float
        Holtsmark probability density W(β) ≥ 0.

    Notes
    -----
    Returns 0 for β ≤ 1e-5 (the distribution vanishes at zero field).
    The upper integration limit is truncated to 15 rather than ∞ because
    exp(−y^{3/2}) decays faster than any exponential for large y.

    References
    ----------
    Holtsmark, J., Ann. Phys. 58, 577 (1919).
    """
    if beta <= 1e-5:
        return 0.0
    val, _ = quad(_holtsmark_integrand, 0, 15, args=(beta,), limit=100)
    w_beta = (2.0 * beta / np.pi) * val
    return max(0.0, w_beta)


@lru_cache(maxsize=None)
def hooper_distribution(beta, a):
    """Return the Hooper screened microfield probability density W(β, a).

    The Hooper distribution extends the Holtsmark distribution to screened
    plasmas by multiplying the characteristic function integrand by a screening
    factor that depends on the screening parameter a = r_e / λ_D:

        T(y, a) = exp(−y^{3/2} × S(y, a))

    where the screening function is the analytical approximation

        S(y, a) = (1 + 1.5 a² / y²)^{−3/4}

    which transitions smoothly from S → 1 (Holtsmark) for y ≫ a to
    S → 0 (fully screened) for y ≪ a.  The distribution is computed via

        W(β, a) = (2β/π) ∫₀^∞ y sin(βy) T(y, a) dy

    Parameters
    ----------
    beta : float
        Reduced electric field β = F / F₀ ≥ 0.
    a : float
        Screening parameter a = r_e / λ_D.  ``a = 0`` recovers Holtsmark.
        Typical tokamak edge values: a ≈ 0.3–1.0.

    Returns
    -------
    float
        Screened microfield probability density W(β, a) ≥ 0.

    Notes
    -----
    Returns 0 for β ≤ 1e-5.  A small regularisation (1e-8) in the screening
    denominator prevents division by zero at y → 0.

    References
    ----------
    Hooper, C.F., Phys. Rev. 149, 77 (1966).
    """
    if beta <= 1e-5:
        return 0.0

    def hooper_integrand(y, beta, a):
        if y == 0:
            return 0.0
        screening = (1.0 + 1.5 * (a**2) / (y**2 + 1e-8))**(-0.75)
        return y * np.sin(beta * y) * np.exp(-y**1.5 * screening)

    val, _ = quad(hooper_integrand, 0, 15, args=(beta, a), limit=100)
    w_beta = (2.0 * beta / np.pi) * val
    return max(0.0, w_beta)


def microfield_quadrature(Ne_m3, Te_ev, num_points=50, max_beta=10.0, use_screening=True,
                          species_charges=None, species_concentrations=None, custom_table_path=None):
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

    Results are cached internally; repeated calls with identical arguments
    return the pre-computed arrays without re-evaluating the distribution
    integrals.
    """
    sc  = tuple(species_charges)       if species_charges       is not None else None
    scc = tuple(species_concentrations) if species_concentrations is not None else None
    return _microfield_quadrature_impl(Ne_m3, Te_ev, num_points, max_beta,
                                       use_screening, sc, scc, custom_table_path)


@lru_cache(maxsize=None)
def _microfield_quadrature_impl(Ne_m3, Te_ev, num_points, max_beta, use_screening,
                                species_charges, species_concentrations, custom_table_path):
    """Cached backend for :func:`microfield_quadrature`."""
    F0, re = calculate_normal_field(Ne_m3)
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

        w_grid = np.zeros_like(beta_grid)
        for i, beta in enumerate(beta_grid):
            if a > 0:
                w_grid[i] = hooper_distribution(beta, a)
            else:
                w_grid[i] = holtsmark_distribution(beta)

    dbeta = beta_grid[1] - beta_grid[0]
    total_area = np.sum(w_grid) * dbeta
    if total_area > 0:
        w_grid = w_grid / total_area

    fields  = beta_grid * F0
    weights = w_grid * dbeta

    return fields, weights
