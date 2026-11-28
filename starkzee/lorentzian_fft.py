r"""FFT evaluation of large sums of Lorentzian (or Voigt) components on a uniform grid.

Approximate counterpart of the direct kernel in ``calculate_static_profile``
(``method='fft'``): the cost no longer scales as grid x components, only as the
number of components plus a few FFTs of the grid size per width group.

**Lorentzian components.** Each component is :math:`\frac{1}{\pi}\,\mathrm{Im}\,
\frac{1}{x - c - i w}`. Bin it to the nearest grid node :math:`x_m` and write
:math:`c + i w = x_m + i w_{ref} + \Delta`, with :math:`\Delta = (c - x_m) + i (w - w_{ref})`:

.. math::

   \frac{1}{x_n - c - i w} = \sum_p \frac{\Delta^p}{z^{p+1}},
   \qquad z = x_n - x_m - i w_{ref}.

For node distances :math:`\lvert n - m\rvert > R` (far field)
:math:`\lvert\Delta\rvert \le` ``RATIO`` :math:`\lvert z\rvert`, so the series
converges geometrically; each power p is one FFT convolution of the binned moments
:math:`\sum a\,\Delta^p` with the fixed kernel :math:`1/z^{p+1}`. The pairs with
:math:`\lvert n - m\rvert \le R` (near field) are added exactly. R and the width
groups are chosen so that R stays small whether the widths are below or above the
grid step.

**Voigt components** (constant widths with a common Gaussian σ).
:math:`V = \mathrm{Re}\, w(\zeta) / (\sigma\sqrt{2\pi})`, with
:math:`\zeta = (x - c + i w)/(\sigma\sqrt2)` and the Faddeeva function w, which is
entire: the expansion about the binned node,
:math:`w(z_0 - \eta) = \sum_p (-\eta)^p\, w^{(p)}(z_0)/p!`, converges for every pair
(no near field). The derivatives come from the recurrence
:math:`w' = -2 z w + 2i/\sqrt\pi`, :math:`w^{(p+1)} = -2 z w^{(p)} - 2 p\, w^{(p-1)}`
near the core and from the asymptotic series for
:math:`\lvert z_0\rvert \ge` ``ASYMPTOTIC_Z``, where the recurrence loses accuracy.

**Frequency-dependent widths.** :math:`w_k(E) = u(E)\,a_k + l(E)\,b_k + n_k` breaks
the convolution structure. The sum is evaluated at fixed widths on Chebyshev nodes in
u + l (:math:`a_k = b_k = 1`) or in u and q = l/u, and interpolated to each grid
point; the number of nodes follows from the width range (the sum is analytic in the
width away from w = 0).

Components farther than ``MARGIN_FRACTION`` of the span outside the grid are added
directly (exactly). The tolerance is relative to the largest component peak.
"""
import numpy as np
from scipy import sparse
from scipy.fft import fft, ifft, next_fast_len
from scipy.special import voigt_profile, wofz

RATIO = 0.2                     # far-field bound |Delta| / |z| of the Lorentzian series
NEAR_TARGET = 2                 # width groups sized for a near-field radius of about this
VOIGT_ETA = 0.125               # bound of |eta| (binning offset + width spread, units sigma sqrt 2)
ASYMPTOTIC_Z = 10.0             # |z| from which the Faddeeva derivatives use the asymptotic series
ASYMPTOTIC_TERMS = 30
MARGIN_FRACTION = 0.5           # components within this fraction of the span outside use the FFT
MAX_NODES = 48                  # cap of the interpolation nodes per width parameter
MAX_TERMS = 40
_SQRT_PI = np.sqrt(np.pi)


def _terms(ratio, tol, scale=1.0):
    """Series terms so that scale * ratio^P / (1 - ratio) <= tol."""
    p = np.log(tol * (1 - ratio) / scale) / np.log(ratio)
    return int(min(MAX_TERMS, max(1, np.ceil(p))))


def _bin(centres, origin, h, size):
    idx = np.clip(np.rint((centres - origin) / h).astype(np.int64), 0, size - 1)
    return idx, centres - (origin + idx * h)


def _scatter(idx, size, values):
    """Sum the rows of values (K, M) into the nodes idx: (size, M)."""
    K = len(idx)
    S = sparse.csr_matrix((np.ones(K), (idx, np.arange(K))), shape=(size, K))
    return np.asarray(S @ values)


# ---------------------------------------------------------------- Lorentzian

def _lorentz_groups(widths, dx):
    """Width groups whose spread keeps the far field within RATIO at radius ~NEAR_TARGET."""
    order = np.argsort(widths, kind="stable")
    w = widths[order]
    groups, start = [], 0
    while start < len(w):
        lo = w[start]
        # eps = (w_hi - lo)/2 allowed by |Delta| <= RATIO * hypot((NEAR_TARGET+1) dx, w_ref)
        hi = lo
        for _ in range(60):     # fixed point for w_hi (w_ref depends on it)
            w_ref = 0.5 * (lo + hi)
            eps2 = (RATIO * np.hypot((NEAR_TARGET + 1) * dx, w_ref)) ** 2 - (0.5 * dx) ** 2
            new = lo + 2.0 * np.sqrt(max(eps2, 0.0))
            if abs(new - hi) <= 1e-9 * new:
                break
            hi = new
        stop = max(start + 1, int(np.searchsorted(w, hi, side="right")))
        groups.append(order[start:stop])
        start = stop
    return groups


class _LorentzGroup:
    """One width group binned on the nodes x0 + j dx.

    The far-field moments sum a Delta^p, Delta = (c - x_m) + i eps, depend only on the
    width offsets eps = w - w_ref, so their FFTs serve every reference width
    w_ref >= w_ref_min: the frequency-dependent nodes shift all widths of a group together.
    """

    def __init__(self, x0, dx, nf, c, a, eps, w_ref_min, tol):
        d_max = np.hypot(0.5 * dx, np.max(np.abs(eps)))
        # near-field radius: smallest R >= -1 with d_max <= RATIO * hypot((R+1) dx, w_ref);
        # valid for every larger w_ref
        need = d_max / RATIO
        R = -1 if w_ref_min >= need else int(np.ceil(np.sqrt(need ** 2 - w_ref_min ** 2) / dx)) - 1
        self.R = R = max(R, -1)
        self.scale = np.hypot((R + 1) * dx, w_ref_min)
        self.terms = terms = _terms(RATIO, tol)
        self.dx, self.nf, self.x0, self.c, self.a = dx, nf, x0, c, a
        self.size = size = next_fast_len(3 * nf - 2)
        idx, delta = _bin(c, x0, dx, nf)
        K = len(c)
        # binning matrix (nf x K), one entry per component (CSC: no sorting)
        S = sparse.csc_matrix((np.ones(K), idx, np.arange(K + 1)), shape=(nf, K))
        A = (delta + 1j * eps) / self.scale
        powers = np.cumprod(np.concatenate([np.ones((K, 1), complex),
                                            np.repeat(A[:, None], terms - 1, axis=1)], axis=1),
                            axis=1)                                  # (K, terms)
        self.moment_fft = [fft(np.asarray(S @ (powers * a[:, col:col + 1])).T, size, axis=-1,
                               workers=-1) for col in range(a.shape[1])]   # (terms, size) per column
        if R >= 0:              # near field: 2R+1 entries per component, values set per evaluation
            nodes = idx[:, None] + np.arange(-R, R + 1)[None, :]
            ok = (nodes >= 0) & (nodes < nf)
            d2 = np.where(ok, ((x0 + nodes * dx) - c[:, None]) ** 2, np.inf)   # inf: no contribution
            self.near_rows = np.clip(nodes, 0, nf - 1).ravel()
            self.near_ptr = np.arange(0, K * (2 * R + 1) + 1, 2 * R + 1)
            self.near_d2 = d2

    def evaluate(self, w_ref, widths):
        """(C, nf) for the reference width w_ref and the component widths (K,) or scalar."""
        nf, R = self.nf, self.R
        j = np.arange(-(nf - 1), nf)
        z = j * self.dx - 1j * w_ref
        kernel = np.where(np.abs(j) > R, 1.0 / z, 0.0)
        base = self.scale / z
        kernel_fft = np.empty((self.terms, self.size), dtype=complex)
        for p in range(self.terms):
            kernel_fft[p] = fft(kernel, self.size, workers=-1)
            kernel = kernel * base
        acc = np.stack([np.sum(m * kernel_fft, axis=0) for m in self.moment_fft])
        out = ifft(acc, axis=-1, workers=-1)[:, nf - 1:2 * nf - 1].imag / np.pi
        if R >= 0:                                                   # near field, exact
            w = np.broadcast_to(widths, self.c.shape)[:, None]
            data = ((w / np.pi) / (self.near_d2 + w * w)).ravel()
            near = sparse.csc_matrix((data, self.near_rows, self.near_ptr), shape=(nf, len(self.c)))
            out += np.asarray(near @ self.a).T
        return out


# ---------------------------------------------------------------- Voigt

def _faddeeva_taylor(z, terms):
    """w^(p)(z)/p! for p < terms, (terms, len(z)); recurrence or asymptotic series."""
    q = np.empty((terms, len(z)), dtype=complex)
    big = np.abs(z) >= ASYMPTOTIC_Z
    zs = z[~big]
    if zs.size:
        q[0, ~big] = wofz(zs)
        if terms > 1:
            q[1, ~big] = -2 * zs * q[0, ~big] + 2j / _SQRT_PI
        for p in range(1, terms - 1):
            q[p + 1, ~big] = (-2 * zs * q[p, ~big] - 2 * q[p - 1, ~big]) / (p + 1)
    if np.any(big):
        zb = z[big]
        # w(z) ~ (i/sqrt(pi)) sum_k (2k-1)!!/2^k z^-(2k+1); derivative p of z^-m / p!
        # = (-1)^p C(m+p-1, p) z^-(m+p)
        inv = 1.0 / zb
        inv2 = inv * inv
        for p in range(terms):
            total = np.zeros_like(zb)
            coef = 1.0                                    # (2k-1)!!/2^k
            zk = inv ** (p + 1)                           # z^-(2k+1+p) at k = 0
            binom = 1.0                                   # C(2k+p, p)
            for k in range(ASYMPTOTIC_TERMS):
                total += coef * binom * zk
                coef *= (2 * k + 1) / 2.0
                zk = zk * inv2
                # C(2k+2+p, p) / C(2k+p, p) = (2k+p+1)(2k+p+2) / ((2k+1)(2k+2))
                binom *= (2 * k + p + 1) * (2 * k + p + 2) / ((2 * k + 1) * (2 * k + 2))
            q[p, big] = (1j / _SQRT_PI) * (-1) ** p * total
    return q


def _voigt_groups(widths, sigma):
    """Width groups of spread VOIGT_ETA * max(sigma sqrt 2, w): the Faddeeva function is
    smooth on that scale in the upper half plane."""
    order = np.argsort(widths, kind="stable")
    w = widths[order]
    s2 = sigma * np.sqrt(2.0)
    groups, start = [], 0
    while start < len(w):
        spread = VOIGT_ETA * max(s2, w[start])
        stop = int(np.searchsorted(w, w[start] + spread, side="right"))
        groups.append(order[start:stop])
        start = stop
    return groups


def _voigt_group(x0, dx, nf, c, a, w, sigma, tol):
    """Faddeeva-series sum (C, nf) of one width group on the nodes x0 + j dx."""
    s2 = sigma * np.sqrt(2.0)
    r0 = max(s2, w.min())                                # smoothness scale [eV]
    m = max(1, int(np.ceil(dx / (VOIGT_ETA * r0))))      # |binning offset| <= VOIGT_ETA r0 / 2
    h = dx / m
    nh = (nf - 1) * m + 1
    w_ref = 0.5 * (w.min() + w.max())
    idx, delta = _bin(c, x0, h, nh)
    eta = (delta - 1j * (w - w_ref)) / s2
    # Cauchy bound on a circle of radius r0/s2 (z units): |w| <= 6.5 there, against the
    # peak ~1/(sqrt(pi) Im z0) of the kernel
    ratio = max(np.max(np.abs(eta)) * s2 / r0, 1e-300)
    terms = _terms(min(ratio, 0.5), tol, scale=6.5 * max(1.0, _SQRT_PI * w_ref / s2))
    powers = np.cumprod(np.concatenate([np.ones((len(c), 1), complex),
                                        np.repeat(-eta[:, None], terms - 1, axis=1)], axis=1), axis=1)
    C = a.shape[1]
    moments = _scatter(idx, nh, (powers[:, :, None] * a[:, None, :]).reshape(len(c), -1))
    moments = moments.reshape(nh, terms, C)
    j = np.arange(-(nh - 1), nh)
    q = _faddeeva_taylor((j * h + 1j * w_ref) / s2, terms)
    size = next_fast_len(3 * nh - 2)
    acc = np.zeros((C, size), dtype=complex)
    for p in range(terms):
        acc += fft(moments[:, p, :].T, size, axis=-1, workers=-1) * fft(q[p], size, workers=-1)
    full = ifft(acc, axis=-1, workers=-1)[:, nh - 1:2 * nh - 1].real / (sigma * np.sqrt(2 * np.pi))
    return full[:, ::m]


# ---------------------------------------------------------------- far components

class _FarLorentz:
    """Components with |c - x_mid| >= 2 H (H = half span): multipole expansion
    1/(x - c - i w) = -(1/H) sum_p t^p / zeta^(p+1), t = (x - x_mid)/H, zeta = (c + i w - x_mid)/H,
    |t| <= 1 <= |zeta|/2 on the grid."""

    def __init__(self, grid, c, a, tol):
        self.mid = 0.5 * (grid[0] + grid[-1])
        self.H = 0.5 * (grid[-1] - grid[0])
        self.t = (grid - self.mid) / self.H
        self.c, self.a = c, a
        self.terms = _terms(0.5, tol)

    def evaluate(self, widths):
        zeta = (self.c + 1j * np.broadcast_to(widths, self.c.shape) - self.mid) / self.H
        inv = 1.0 / zeta
        powers = np.cumprod(np.repeat(inv[:, None], self.terms, axis=1), axis=1)   # zeta^-(p+1)
        moments = powers.T @ self.a                                  # (terms, C)
        total = np.zeros((len(self.t), self.a.shape[1]), dtype=complex)
        for p in range(self.terms - 1, -1, -1):                      # Horner in t
            total = total * self.t[:, None] + moments[p]
        return (-total / self.H).imag / np.pi


# ---------------------------------------------------------------- fixed widths

def _direct(grid, centres, amplitudes, widths, sigma):
    """Exact sum (n, C) for a few components; widths (K,) or (n, K)."""
    d = grid[:, np.newaxis] - centres[np.newaxis, :]
    if sigma is None:
        kernel = (widths / np.pi) / (d ** 2 + widths ** 2)
    else:
        kernel = voigt_profile(d, sigma, widths)
    return kernel @ amplitudes


def _layout(grid):
    """Step, lead nodes, node count and origin of the binning grid.

    Coordinates are relative to grid[0] (centres are shifted by the caller), and the
    step is the span / (n - 1), as numpy.linspace builds the grid: node n is then
    grid[n] to rounding, without the drift of n * (grid[1] - grid[0]).
    """
    dx = (grid[-1] - grid[0]) / (len(grid) - 1)
    lead = int(np.ceil(MARGIN_FRACTION * (grid[-1] - grid[0]) / dx))
    return dx, lead, len(grid) + 2 * lead, -lead * dx


def _near(grid, centres):
    """Components handled on the binned grid; the others satisfy |c - x_mid| >= 2 H."""
    reach = MARGIN_FRACTION * (grid[-1] - grid[0])
    return (centres > grid[0] - reach) & (centres < grid[-1] + reach)


def _fixed_sum_near(grid, centres, amplitudes, widths, sigma, tol):
    dx, lead, nf, x0 = _layout(grid)
    out = np.zeros((amplitudes.shape[1], nf))
    if sigma is None:
        for g in _lorentz_groups(widths, dx):
            w = widths[g]
            w_ref = 0.5 * (w.min() + w.max())
            group = _LorentzGroup(x0, dx, nf, centres[g] - grid[0], amplitudes[g], w - w_ref, w_ref, tol)
            out += group.evaluate(w_ref, w)
    else:
        for g in _voigt_groups(widths, sigma):
            out += _voigt_group(x0, dx, nf, centres[g] - grid[0], amplitudes[g], widths[g], sigma, tol)
    return out[:, lead:lead + len(grid)].T


def lorentzian_sum(grid, centres, amplitudes, widths, sigma=None, tol=1e-8):
    """sum_k a_k L(E; c_k, w_k) (Voigt with Gaussian std sigma) on a uniform increasing grid.

    centres, widths (K,); amplitudes (K, C). Returns (n, C).
    """
    grid = np.asarray(grid, float)
    centres = np.asarray(centres, float)
    amplitudes = np.asarray(amplitudes, float)
    amplitudes = amplitudes[:, None] if amplitudes.ndim == 1 else amplitudes
    widths = np.broadcast_to(np.asarray(widths, float), np.shape(centres))
    out = np.zeros((len(grid), amplitudes.shape[1]))
    if not len(centres):
        return out
    near = _near(grid, centres)
    if np.any(~near):
        if sigma is None:
            out += _FarLorentz(grid, centres[~near], amplitudes[~near], tol).evaluate(widths[~near])
        else:
            out += _direct(grid, centres[~near], amplitudes[~near], widths[~near], sigma)
    if np.any(near):
        out += _fixed_sum_near(grid, centres[near], amplitudes[near], widths[near], sigma, tol)
    return out


# ---------------------------------------------------------------- frequency-dependent widths

def _nodes(lo, hi, w_mid_min, tol):
    """Chebyshev (2nd kind) nodes and barycentric weights on [lo, hi] for the width parameter."""
    span = hi - lo
    count = 1
    if span > 0 and w_mid_min > 0:
        x0 = max(2.0 * w_mid_min / span, 1.0 + 1e-12)
        if 1.0 / x0 > tol:
            rho = x0 + np.sqrt(x0 * x0 - 1.0)
            count = int(min(MAX_NODES, max(2, np.ceil(np.log(4.0 / tol) / np.log(rho)))))
    if count == 1:
        return np.array([0.5 * (lo + hi)]), np.array([1.0])
    j = np.arange(count)
    nodes = 0.5 * (lo + hi) + 0.5 * span * np.cos(np.pi * j / (count - 1))
    weights = (-1.0) ** j
    weights[[0, -1]] *= 0.5
    return nodes, weights


def _basis(nodes, weights, x):
    """Barycentric Lagrange basis (len(x), J) at the points x."""
    if len(nodes) == 1:
        return np.ones((len(x), 1))
    diff = x[:, np.newaxis] - nodes[np.newaxis, :]
    exact = diff == 0
    diff[exact] = 1.0
    terms = weights / diff
    basis = terms / terms.sum(axis=1, keepdims=True)
    hit = exact.any(axis=1)
    basis[hit] = exact[hit].astype(float)
    return basis


def frequency_dependent_sum(grid, centres, amplitudes, natural, w_upper, w_lower,
                            scale_u=None, scale_l=None, floor=None, tol=1e-8):
    """sum_k a_k L(E; c_k, w_k(E)), w_k(E) = w_upper(E) s_u,k + w_lower(E) s_l,k + n_k.

    w_upper, w_lower: grid arrays (n,) or scalars; natural, scales: (K,), scalars or None
    (scale 1); widths floored at `floor`. Returns (n, C).
    """
    grid = np.asarray(grid, float)
    centres = np.asarray(centres, float)
    n, K = len(grid), len(centres)
    amplitudes = np.asarray(amplitudes, float)
    amplitudes = amplitudes[:, None] if amplitudes.ndim == 1 else amplitudes
    out = np.zeros((n, amplitudes.shape[1]))
    if not K:
        return out
    u = np.broadcast_to(np.asarray(w_upper, float), (n,))
    l = np.broadcast_to(np.asarray(w_lower, float), (n,))
    nat = np.broadcast_to(np.asarray(natural, float), (K,))
    su = None if scale_u is None else np.broadcast_to(np.asarray(scale_u, float), (K,))
    sl = None if scale_l is None else np.broadcast_to(np.asarray(scale_l, float), (K,))

    near = _near(grid, centres)
    c, a, nat_n = centres, amplitudes, nat

    def fixed(wk):
        if floor is not None:
            wk = np.maximum(wk, floor)
        return lorentzian_sum(grid, c, a, wk, None, tol)

    if su is None and sl is None and floor is None:      # one parameter: w = s + n, s = u + l
        s = u + l
        nodes, bw = _nodes(s.min(), s.max(), 0.5 * (s.min() + s.max()) + nat_n.min(), tol)
        basis = _basis(nodes, bw, s)
        dx, lead, nf, x0 = _layout(grid)
        acc = np.zeros((len(nodes), a.shape[1], nf))
        # width groups and moments are shared by the nodes: offsets eps = n - mid(n)
        cn, an, natn = c[near] - grid[0], a[near], nat_n[near]
        for g in _lorentz_groups(nodes.min() + natn, dx):
            ng = natn[g]
            mid = 0.5 * (ng.min() + ng.max())
            group = _LorentzGroup(x0, dx, nf, cn[g], an[g], ng - mid, nodes.min() + mid, tol)
            for j, sj in enumerate(nodes):
                acc[j] += group.evaluate(sj + mid, sj + ng)
        far = _FarLorentz(grid, c[~near], a[~near], tol) if np.any(~near) else None
        for j, sj in enumerate(nodes):
            part = acc[j][:, lead:lead + n].T
            if far is not None:
                part = part + far.evaluate(sj + nat_n[~near])
            out += basis[:, j:j + 1] * part
        return out
    if su is None and sl is None:                        # floored widths: generic nodes
        s = u + l
        nodes, bw = _nodes(s.min(), s.max(), 0.5 * (s.min() + s.max()) + nat_n.min(), tol)
        basis = _basis(nodes, bw, s)
        for j, sj in enumerate(nodes):
            out += basis[:, j:j + 1] * fixed(sj + nat_n)
        return out

    su_n = 1.0 if su is None else su
    sl_n = 1.0 if sl is None else sl
    if u.min() <= 0:
        raise ValueError("frequency-dependent widths with scales need w_upper > 0 on the grid")
    q = l / u                                            # nearly constant: few nodes in q
    q_mid = 0.5 * (q.min() + q.max())
    coef = su_n + q_mid * sl_n                           # w = u * coef + n at q = q_mid
    nu, bu = _nodes(u.min(), u.max(), 0.5 * (u.min() + u.max()) + np.min(nat_n / coef), tol)
    # w = u (su + q sl) + n vanishes at q = -(su + n/u) / sl
    nq, bq = _nodes(q.min(), q.max(),
                    q_mid + np.min((su_n + nat_n / u.max()) / np.maximum(sl_n, 1e-300)), tol)
    basis_u, basis_q = _basis(nu, bu, u), _basis(nq, bq, q)
    for i, ui in enumerate(nu):
        for j, qj in enumerate(nq):
            out += (basis_u[:, i] * basis_q[:, j])[:, None] * fixed(ui * (su_n + qj * sl_n) + nat_n)
    return out
