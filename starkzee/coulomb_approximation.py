"""Coulomb approximation (Bates & Damgaard 1949) for radial dipole integrals.

An electron outside a core of net charge Z_core moves, beyond the core, in the
Coulomb field -Z_core/r. With the effective quantum number n* from the measured
binding energy, E = -Z_core^2 Ry_M / n*^2, its radial function there is the
Whittaker function, written as the asymptotic series

    P(r) = N exp(-Z r / n*) r^n* sum_k a_k r^-k,
    a_0 = 1,  a_k = a_(k-1) n* / (2 Z k) [l(l+1) - (n* - k)(n* - k + 1)].

Following Bates and Damgaard, the series is integrated term by term,
int_0^inf r^p exp(-alpha r) dr = Gamma(p+1) / alpha^(p+1), which gives the
normalization and <a| r |b> in closed form. The number of terms is the nodal
count of the nearest hydrogenic function, K = round(n* - l - 1) (at least 0):
for integer n* the series terminates there and the functions are exactly
hydrogenic, so the result equals Gordon's formula. The sign follows the
positive-near-origin convention of :func:`starkzee.radiator.radial_dipole`
(the large-r lobe has sign (-1)^K).

Accuracy is typically a few percent for low-l valence transitions and better for
Rydberg states; it degrades for strongly penetrating orbitals and where the
integral nearly cancels. Reference: D.R. Bates and A. Damgaard, Phil. Trans. R.
Soc. A 242, 101 (1949).
"""
from functools import lru_cache
from math import exp, lgamma, log, sqrt

from scipy.constants import atomic_mass as _U, m_e as _M_E, physical_constants as _pc

RYDBERG_INF_CM = _pc["Rydberg constant"][0] / 100.0       # cm^-1


def reduced_rydberg_cm(mass_amu=None):
    """Rydberg constant [cm^-1] for an atom of mass mass_amu [u] (None: infinite mass)."""
    if mass_amu is None:
        return RYDBERG_INF_CM
    return RYDBERG_INF_CM / (1.0 + _M_E / (mass_amu * _U))


def effective_n(energy_cm, limit_cm, Z_core=1.0, mass_amu=None):
    """n* = Z_core sqrt(Ry_M / (limit - E)) for a level at energy_cm below the series limit."""
    binding = limit_cm - energy_cm
    if not binding > 0:
        raise ValueError("the level must lie below its series limit")
    return Z_core * sqrt(reduced_rydberg_cm(mass_amu) / binding)


def _coefficients(n_star, l, Z):
    K = max(0, int(round(float(n_star) - l - 1)))
    a = [1.0]
    for k in range(1, K + 1):
        a.append(a[-1] * n_star / (2.0 * Z * k) * (l * (l + 1) - (n_star - k) * (n_star - k + 1)))
    return a, K


def _overlap(a, na, b, nb, alpha, extra, m):
    """sum_kk' a_k b_k' Gamma(na + nb + extra + 1 - k - k') / alpha^(...), in arithmetic m."""
    total = m.zero
    log_alpha = m.log(alpha)
    for k, ak in enumerate(a):
        for kp, bk in enumerate(b):
            p = na + nb + extra - k - kp
            if p <= -1:
                raise ValueError("Coulomb-approximation series term diverges (n* too small for l)")
            total += ak * bk * m.exp(m.lgamma(p + 1) - (p + 1) * log_alpha)
    return total


class _Double:
    """Double-precision arithmetic with the interface used by _overlap."""
    zero = 0.0
    exp, log, lgamma, sqrt, convert = staticmethod(exp), staticmethod(log), staticmethod(lgamma), \
        staticmethod(sqrt), staticmethod(float)


_ARITHMETIC = None


def _arithmetic():
    """40-digit mpmath arithmetic if available (imported on first use), else double precision.

    The series alternates and cancels at large n*, hence the extended precision.
    """
    global _ARITHMETIC
    if _ARITHMETIC is None:
        try:
            import mpmath as _mpmath

            class _Extended:
                ctx = _mpmath.mp.clone()
                ctx.dps = 40
                zero = ctx.mpf(0)
                exp, log, sqrt, convert = ctx.exp, ctx.log, ctx.sqrt, ctx.mpf
                lgamma = staticmethod(lambda x, _c=ctx: _c.loggamma(x))
            _ARITHMETIC = _Extended
        except ImportError:            # pragma: no cover - mpmath ships with SciPy/SymPy installs
            _ARITHMETIC = _Double
    return _ARITHMETIC


def _moment(n_star_a, l_a, n_star_b, l_b, Z_core, power):
    """Signed <n*_a l_a| r^power |n*_b l_b> [a0^power] (positive-near-origin functions)."""
    m = _arithmetic()
    na, nb, Z = m.convert(n_star_a), m.convert(n_star_b), m.convert(Z_core)
    a, Ka = _coefficients(na, l_a, Z)
    b, Kb = _coefficients(nb, l_b, Z)
    norm_a = _overlap(a, na, a, na, 2 * Z / na, 0, m)
    norm_b = _overlap(b, nb, b, nb, 2 * Z / nb, 0, m)
    integral = _overlap(a, na, b, nb, Z * (1 / na + 1 / nb), power, m)
    sign = -1.0 if (Ka + Kb) % 2 else 1.0                     # positive near the origin
    return sign * float(integral / m.sqrt(norm_a * norm_b))


@lru_cache(maxsize=None)
def coulomb_radial(n_star_a, l_a, n_star_b, l_b, Z_core=1.0):
    """Signed <n*_a l_a| r |n*_b l_b> [a0] in the Coulomb approximation.

    Z_core is the net charge of the core seen by the active electron (1 for a
    neutral atom, 2 for a singly charged ion, ...). The series is summed in
    40-digit arithmetic when mpmath is installed (cancellation grows with n*),
    otherwise in double precision (relative error ~1e-7 at n* = 10).
    """
    if abs(l_a - l_b) != 1:
        raise ValueError("|l_a - l_b| must be 1")
    return _moment(n_star_a, l_a, n_star_b, l_b, Z_core, 1)


@lru_cache(maxsize=None)
def coulomb_r2(n_star_a, l_a, n_star_b, l_b, Z_core=1.0):
    """Signed <n*_a l_a| r^2 |n*_b l_b> [a0^2] in the Coulomb approximation (|l_a - l_b| = 0 or 2).

    For the diamagnetic (quadratic Zeeman) term inside a shell; equal to the
    hydrogenic values for integer n*. Like all Coulomb-approximation moments it
    ignores the core region, which matters little for r^2 (weighted to large r).
    """
    if abs(l_a - l_b) not in (0, 2):
        raise ValueError("|l_a - l_b| must be 0 or 2")
    return _moment(n_star_a, l_a, n_star_b, l_b, Z_core, 2)
