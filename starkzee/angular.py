"""Angular-momentum algebra: Wigner 3-j and 6-j symbols and closed-form matrix elements.

Every angular factor used by StarkZee lives here: the Wigner symbols, the rank-1
dipole elements (:func:`angular_dipole_element`), the sin²θ elements of the
diamagnetic term (:func:`sin2_theta_element`) and the spin-orbit L·S elements
(:func:`spin_orbit_element`).

The Wigner symbols have two back ends, selected with :func:`set_backend` (the active one is ``BACKEND``):

* ``"pywigxjpf"``: the WIGXJPF library (H. T. Johansson and C. Forssén, SIAM J. Sci.
  Comput. 38, A376 (2016)) through the optional package ``pywigxjpf``. Exact
  prime-factor arithmetic, rounded once to double precision, at any j up to
  ``MAX_TWO_J / 2``. Used by default when the package is installed.
* ``"racah"``: the dependency-free Racah sums in double precision below. Used when
  pywigxjpf is not installed, and for arguments beyond ``MAX_TWO_J``.

Both follow the Condon--Shortley convention and return 0 for arguments that violate
a selection rule or are not (half-)integers. Results are cached.
"""

import math
import threading
from functools import lru_cache

import numpy as np

MAX_TWO_J = 400          # largest doubled j handled by pywigxjpf (table size)

BACKEND = "racah"
_wig = None
_thread_state = threading.local()


def _init_pywigxjpf():
    """Import pywigxjpf and allocate its tables; return the module, or None if unusable."""
    try:
        import pywigxjpf as wig
        wig.wig_table_init(MAX_TWO_J, 6)       # factorial tables sized for 3j and 6j symbols
        wig.wig_temp_init(MAX_TWO_J)
        if abs(wig.wig3jj(1, 1, 2, 1, -1, 0) - 0.4082482904638631) > 1e-15:   # (1/2 1/2 1; 1/2 -1/2 0)
            return None
    except Exception:                          # not installed, or failed to build/load
        return None
    _thread_state.ready = True
    return wig


def set_backend(name="auto"):
    """Select the back end: ``"auto"`` (pywigxjpf if available), ``"pywigxjpf"`` or ``"racah"``."""
    global _wig, BACKEND
    if name not in ("auto", "pywigxjpf", "racah"):
        raise ValueError(f"Unknown Wigner backend {name!r}; choose 'auto', 'pywigxjpf' or 'racah'.")
    if name == "racah":
        _wig = None
    else:
        if _wig is None:
            _wig = _init_pywigxjpf()
        if _wig is None and name == "pywigxjpf":
            raise ImportError("pywigxjpf is not available")
    BACKEND = "pywigxjpf" if _wig is not None else "racah"
    _wigner_3j_doubled.cache_clear()
    _wigner_6j_doubled.cache_clear()


def _doubled(x):
    """2x as an int if x is a (half-)integer, else None."""
    two_x = 2 * x
    nearest = round(two_x)
    return int(nearest) if math.isclose(two_x, nearest, abs_tol=1e-12) else None


def _pywigxjpf_ready():
    if not getattr(_thread_state, "ready", False):       # temporary arrays are per thread
        _wig.wig_temp_init(MAX_TWO_J)
        _thread_state.ready = True
    return _wig


# ---------------------------------------------------------------- public functions
def wigner_3j(j1, j2, j3, m1, m2, m3):
    """Return the Wigner 3-j symbol.

    The 3-j symbol encodes the coupling of two angular momenta and is related
    to the Clebsch-Gordan coefficients by:

        ⟨j₁ m₁ j₂ m₂ | j₃ −m₃⟩ = (−1)^{j₁−j₂+m₃} √(2j₃+1) ×
                                    (j₁  j₂  j₃ )
                                    (m₁  m₂  m₃ )

    Parameters
    ----------
    j1, j2, j3 : float
        Angular momentum quantum numbers (integer or half-integer).
    m1, m2, m3 : float
        Magnetic quantum numbers (integer or half-integer).

    Returns
    -------
    float
        Wigner 3-j symbol value.  Returns 0 if any selection rule is violated
        or if the arguments are not valid (half-)integers.

    References
    ----------
    Racah, G., Phys. Rev. 62, 438 (1942).
    """
    doubled = tuple(_doubled(x) for x in (j1, j2, j3, m1, m2, m3))
    if None in doubled:
        return 0.0
    return _wigner_3j_doubled(*doubled)


def wigner_6j(j1, j2, j3, j4, j5, j6):
    """Return a Wigner 6-j symbol.

    Integer and half-integer arguments are supported.  Invalid angular-
    momentum triangles return zero, following the convention used by
    :func:`wigner_3j`.  Both back ends use the same Condon--Shortley
    convention as the J-coupled hydrogenic and multi-electron paths.
    """
    doubled = tuple(_doubled(x) for x in (j1, j2, j3, j4, j5, j6))
    if None in doubled or min(doubled) < 0:
        return 0.0
    return _wigner_6j_doubled(*doubled)


@lru_cache(maxsize=None)
def _wigner_3j_doubled(a, b, c, ma, mb, mc):
    if _wig is not None and 0 <= min(a, b, c) and max(a, b, c) <= MAX_TWO_J:
        return _pywigxjpf_ready().wig3jj(a, b, c, ma, mb, mc) + 0.0     # + 0.0 turns -0.0 into 0.0
    return _wigner_3j_racah(a / 2, b / 2, c / 2, ma / 2, mb / 2, mc / 2)


@lru_cache(maxsize=None)
def _wigner_6j_doubled(a, b, c, d, e, f):
    if _wig is not None and max(a, b, c, d, e, f) <= MAX_TWO_J:
        return _pywigxjpf_ready().wig6jj(a, b, c, d, e, f) + 0.0
    return _wigner_6j_racah(a / 2, b / 2, c / 2, d / 2, e / 2, f / 2)


# ---------------------------------------------------------------- closed forms in the |l m_l m_s> basis
@lru_cache(maxsize=None)
def angular_dipole_element(l1, m1, l2, m2, q):
    """Return the angular part of the dipole matrix element ⟨l₁, m₁ | T_q^(1) | l₂, m₂⟩.

    The spherical tensor components T_q^(1) of the position unit vector r̂ are:

    - q = 0  (π):  T₀ = cos θ
    - q = +1 (σ+): T₊₁ = sin θ e^{iφ} / √2
    - q = −1 (σ−): T₋₁ = sin θ e^{−iφ} / √2

    The matrix elements are expressed via Clebsch-Gordan coefficients and are
    non-zero only when abs(l₁ − l₂) = 1 and m₁ − m₂ = q.  Explicit formulas:

    **q = 0** (Δm = 0, both states share the same m):

        ⟨l, m | cos θ | l+1, m⟩ = √((l+1)² − m²) / √((2l+1)(2l+3))

    evaluated with l = max(l₁, l₂) − 1.

    **q = +1** (m₁ = m₂ + 1):

        ⟨l−1, m | T₊₁ | l, m⟩ = +√((l−m)(l−m−1) / (2(2l−1)(2l+1)))
        ⟨l, m+1 | T₊₁ | l−1, m⟩ = −√((l+m)(l+m+1) / (2(2l−1)(2l+1)))

    **q = −1** (m₁ = m₂ − 1):

        ⟨l−1, m | T₋₁ | l, m⟩ = −√((l+m)(l+m−1) / (2(2l−1)(2l+1)))
        ⟨l, m−1 | T₋₁ | l−1, m⟩ = +√((l−m)(l−m+1) / (2(2l−1)(2l+1)))

    Parameters
    ----------
    l1, m1 : int
        Orbital and magnetic quantum numbers of the bra state.
    l2, m2 : int
        Orbital and magnetic quantum numbers of the ket state.
    q : int
        Polarization index: 0 (π), +1 (σ+), or −1 (σ−).

    Returns
    -------
    float
        Angular matrix element (real).  Returns 0 if abs(l₁ − l₂) ≠ 1 or
        m₁ − m₂ ≠ q.

    Notes
    -----
    The full dipole matrix element between two basis states is
    ``−radial_dipole(n1,l1,n2,l2,Z) × angular_dipole_element(l1,m1,l2,m2,q)``
    (the minus sign comes from d = −e r).
    """
    if abs(l1 - l2) != 1:
        return 0.0

    if q == 0:
        if m1 != m2:
            return 0.0
        l = max(l1, l2)
        return np.sqrt((l**2 - m1**2) / ((2*l - 1) * (2*l + 1)))

    elif q == 1:
        if m1 != m2 + 1:
            return 0.0
        # l1 = l-1, l2 = l
        if l1 == l2 - 1:
            l = l2
            m = m2
            return np.sqrt((l - m) * (l - m - 1) / (2.0 * (2*l - 1) * (2*l + 1)))
        # l1 = l, l2 = l-1
        else:
            l = l1
            m = m2
            return -np.sqrt((l + m) * (l + m + 1) / (2.0 * (2*l - 1) * (2*l + 1)))

    elif q == -1:
        if m1 != m2 - 1:
            return 0.0
        # l1 = l-1, l2 = l
        if l1 == l2 - 1:
            l = l2
            m = m2
            return -np.sqrt((l + m) * (l + m - 1) / (2.0 * (2*l - 1) * (2*l + 1)))
        # l1 = l, l2 = l-1
        else:
            l = l1
            m = m2
            return np.sqrt((l - m) * (l - m + 1) / (2.0 * (2*l - 1) * (2*l + 1)))

    return 0.0


def sin2_theta_element(l1, ml1, l2, ml2):
    """Return ⟨l₁ m | sin²θ | l₂ m⟩ (diamagnetic / quadratic-Zeeman angular factor).

    Non-zero only for equal m and l₂ − l₁ ∈ {0, ±2}:

        ⟨l m | cos²θ | l m⟩ = (2l² + 2l − 1 − 2m²) / ((2l − 1)(2l + 3)),   sin² = 1 − cos²
        ⟨l m | sin²θ | l+2 m⟩ = −√(((l+1)² − m²)((l+2)² − m²) / ((2l+1)(2l+3)²(2l+5)))
    """
    if ml1 != ml2:
        return 0.0
    if l1 == l2:
        if l1 == 0:
            return 2.0 / 3.0
        cos2 = (2.0 * l1**2 + 2.0 * l1 - 1.0 - 2.0 * ml1**2) / (
            (2.0 * l1 - 1.0) * (2.0 * l1 + 3.0))
        return 1.0 - cos2
    if abs(l1 - l2) == 2:
        lower = min(l1, l2)
        numerator = (((lower + 1.0)**2 - ml1**2)
                     * ((lower + 2.0)**2 - ml1**2))
        denominator = ((2.0 * lower + 1.0) * (2.0 * lower + 3.0)**2
                       * (2.0 * lower + 5.0))
        return -np.sqrt(numerator / denominator)
    return 0.0


def spin_orbit_element(l, ml_bra, ms_bra, ml_ket, ms_ket):
    """Return ⟨l m_l m_s | L·S | l m_l' m_s'⟩ for s = 1/2 (in units of ħ²).

    L·S = L_z S_z + (L₊S₋ + L₋S₊)/2, so the element is m_l m_s on the diagonal and
    ½ √(l(l+1) − m_l'(m_l' ± 1)) when (m_l, m_s) = (m_l' ± 1, m_s' ∓ 1).
    """
    if ml_bra == ml_ket and ms_bra == ms_ket:
        return ml_bra * ms_bra
    if ml_bra == ml_ket + 1 and ms_bra == ms_ket - 1 and ms_ket == 0.5:
        return 0.5 * np.sqrt(l * (l + 1.0) - ml_ket * (ml_ket + 1))
    if ml_bra == ml_ket - 1 and ms_bra == ms_ket + 1 and ms_ket == -0.5:
        return 0.5 * np.sqrt(l * (l + 1.0) - ml_ket * (ml_ket - 1))
    return 0.0


# ---------------------------------------------------------------- Racah sums (fallback)
def _wigner_3j_racah(j1, j2, j3, m1, m2, m3):
    """3-j symbol from the Racah sum formula in double precision."""
    if m1 + m2 + m3 != 0:
        return 0.0
    if not (abs(j1 - j2) <= j3 <= j1 + j2):
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0

    def is_valid(x):
        return (2 * x) == int(2 * x)
    if not (is_valid(j1) and is_valid(j2) and is_valid(j3)
            and is_valid(m1) and is_valid(m2) and is_valid(m3)):
        return 0.0
    # j1 + j2 + j3 and each j + m must be integers (otherwise the symbol is zero)
    if (j1 + j2 + j3) % 1 or (j1 + m1) % 1 or (j2 + m2) % 1 or (j3 + m3) % 1:
        return 0.0

    def tri_coeff(a, b, c):
        num = (math.factorial(int(a + b - c))
               * math.factorial(int(a - b + c))
               * math.factorial(int(-a + b + c)))
        den = math.factorial(int(a + b + c + 1))
        return num / den

    try:
        t_min = max(0, int(j2 - j3 - m1), int(j1 - j3 + m2))
        t_max = min(int(j1 + j2 - j3), int(j1 - m1), int(j2 + m2))

        sum_val = 0.0
        for t in range(t_min, t_max + 1):
            term = ((-1)**t) / (
                math.factorial(t)
                * math.factorial(int(j1 + j2 - j3 - t))
                * math.factorial(int(j1 - m1 - t))
                * math.factorial(int(j2 + m2 - t))
                * math.factorial(int(j3 - j2 + m1 + t))
                * math.factorial(int(j3 - j1 - m2 + t))
            )
            sum_val += term

        prefactor = ((-1)**int(j1 - j2 - m3)) * math.sqrt(tri_coeff(j1, j2, j3)) * math.sqrt(
            math.factorial(int(j1 + m1)) * math.factorial(int(j1 - m1))
            * math.factorial(int(j2 + m2)) * math.factorial(int(j2 - m2))
            * math.factorial(int(j3 + m3)) * math.factorial(int(j3 - m3))
        )
        return prefactor * sum_val
    except ValueError:
        return 0.0


def _wigner_6j_racah(j1, j2, j3, j4, j5, j6):
    """6-j symbol from the Racah sum formula in double precision."""
    values = (j1, j2, j3, j4, j5, j6)
    if any(value < 0 or not math.isclose(2.0 * value, round(2.0 * value),
                                         abs_tol=1e-12)
           for value in values):
        return 0.0

    def triangle(a, b, c):
        return (a + b >= c and a + c >= b and b + c >= a
                and math.isclose(a + b + c, round(a + b + c),
                                 abs_tol=1e-12))

    triangles = ((j1, j2, j3), (j1, j5, j6),
                 (j4, j2, j6), (j4, j5, j3))
    if not all(triangle(*triple) for triple in triangles):
        return 0.0

    def factorial(value):
        nearest = round(value)
        if value < -1e-12 or not math.isclose(value, nearest, abs_tol=1e-12):
            raise ValueError("factorial argument is not a non-negative integer")
        return math.factorial(int(nearest))

    def delta(a, b, c):
        return math.sqrt(
            factorial(a + b - c)
            * factorial(a - b + c)
            * factorial(-a + b + c)
            / factorial(a + b + c + 1.0)
        )

    lower = max(j1 + j2 + j3, j1 + j5 + j6,
                j4 + j2 + j6, j4 + j5 + j3)
    upper = min(j1 + j2 + j4 + j5, j1 + j3 + j4 + j6,
                j2 + j3 + j5 + j6)
    z_min = int(round(lower))
    z_max = int(round(upper))
    if z_min > z_max:
        return 0.0

    total = 0.0
    for z in range(z_min, z_max + 1):
        denominator = (
            factorial(z - j1 - j2 - j3)
            * factorial(z - j1 - j5 - j6)
            * factorial(z - j4 - j2 - j6)
            * factorial(z - j4 - j5 - j3)
            * factorial(j1 + j2 + j4 + j5 - z)
            * factorial(j1 + j3 + j4 + j6 - z)
            * factorial(j2 + j3 + j5 + j6 - z)
        )
        total += (-1.0)**z * factorial(z + 1.0) / denominator

    return math.prod(delta(*triple) for triple in triangles) * total


set_backend("auto")
