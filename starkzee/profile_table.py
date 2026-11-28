"""Static Stark-Zeeman profiles at many densities and temperatures from one set of eigensystems.

The radiator Hamiltonian H(F, mu) = H_atom(B) + F (mu M_z + sqrt(1 - mu^2) M_x)
depends on the plasma only through the microfield F. The density sets the scale
of the field distribution, F_H ~ Ne^(2/3), and so which fields occur and with which
probability, but the dressed states at a given F are the same for every density.

:class:`StaticProfileTable` therefore integrates over a fixed grid of absolute field
magnitudes covering beta = F / F_H(Ne) in [beta_min, max_beta] for every density of
the range. The grid is geometric, F_k = F_0 r^k: beta_k is then geometric with the
same ratio at every density, so each profile uses about ``num_f`` nodes, with the
weights W(beta) beta d(ln beta) of the integral in ln beta.

The dressed-state data of each (F_k, mu) node (transition energies, intensities,
natural widths, electron-operator scales) are computed once and cached; a profile
at (Ne, Te, Ti) only recomputes the microfield weights, the electron and Doppler
widths and the line-shape sum (``method='exact'`` or ``'fft'``).

The log rule is a different discretization of the same field integral than the
uniform beta grid of :func:`~starkzee.static_profile.calculate_static_profile`; both
converge to the same profile as ``num_f`` grows (compare at your resolution).
The distribution mass below beta_min (W ~ beta^2 there) is dropped.
"""
import inspect

import numpy as np

from starkzee.microfield import calculate_holtsmark_field, microfield_weights
from starkzee.radiators import HydrogenicTransition
from starkzee.static_profile import _isotope_from_species, calculate_static_profile

# options that define the cached dressed-state data: fixed per table
FIXED_OPTIONS = ("quadratic_zeeman", "fine_structure", "A", "species", "use_empirical_data", "atom",
                 "electron_model", "electron_operator", "natural_width_mode")
# options of the microfield weights (per call)
MICROFIELD_OPTIONS = ("use_screening", "species_charges", "species_concentrations", "custom_table_path",
                      "charged", "emitter_charge", "microfield_model")
_DEFAULTS = {k: v.default for k, v in inspect.signature(calculate_static_profile).parameters.items()
             if v.default is not inspect.Parameter.empty}


class StaticProfileTable:
    """Reusable dressed states of one line at one magnetic field, for many (Ne, Te, Ti).

    Parameters
    ----------
    n_u, n_l, Z, B :
        Transition and magnetic field [T], as in ``calculate_static_profile``.
    Ne_range : (float, float)
        Lowest and highest electron density [m^-3] the table must serve.
    num_f : int
        Field nodes per profile: the grid ratio is (max_beta / beta_min)^(1/(num_f - 1)).
    beta_min : float
        Lowest reduced field of the quadrature (default 0.03).
    num_mu, max_beta, Z_bar :
        Angle quadrature, field cut-off in units of F_H and mean ion charge, as in
        ``calculate_static_profile``.
    radiator : object, optional
        Radiator to use (built from the atomic-structure options when ``None``).
    **options :
        The options of FIXED_OPTIONS (atomic structure, electron model, electron
        operator, natural-width mode), fixed for the table because the cached
        data depend on them.

    Examples
    --------
    >>> table = StaticProfileTable(3, 2, 1, B=2.0, Ne_range=(1e19, 1e21), Ti_ev=None)  # doctest: +SKIP
    >>> pi, sp, sm = table.profile(energies_ev, Ne_m3=3e20, Te_ev=5.0, Ti_ev=5.0)        # doctest: +SKIP
    """

    def __init__(self, n_u, n_l, Z, B, Ne_range, num_f=20, num_mu=6, max_beta=10.0, Z_bar=1.0,
                 radiator=None, beta_min=0.03, **options):
        unknown = set(options) - set(FIXED_OPTIONS)
        if unknown:
            raise TypeError(f"StaticProfileTable fixes only {FIXED_OPTIONS}; pass {sorted(unknown)} "
                            "to profile().")
        ne_lo, ne_hi = (float(x) for x in Ne_range)
        if not (0 < ne_lo <= ne_hi and np.isfinite(ne_hi)):
            raise ValueError("Ne_range must be (low, high) with 0 < low <= high.")
        if num_f < 2:
            raise ValueError("num_f must be >= 2.")
        self.n_u, self.n_l, self.Z, self.B = n_u, n_l, Z, B
        self.Ne_range = (ne_lo, ne_hi)
        if not 0 < beta_min < max_beta:
            raise ValueError("need 0 < beta_min < max_beta.")
        self.num_mu, self.max_beta, self.Z_bar, self.beta_min = num_mu, max_beta, Z_bar, beta_min
        self.options = {k: options.get(k, _DEFAULTS[k]) for k in FIXED_OPTIONS}
        if radiator is None:
            o = self.options
            A = _isotope_from_species(Z, o["A"], o["species"])
            radiator = HydrogenicTransition(n_u, n_l, Z, B, o["quadratic_zeeman"], o["fine_structure"], A,
                                            use_empirical_data=o["use_empirical_data"], atom=o["atom"])
        self.radiator = radiator
        log_step = np.log(max_beta / beta_min) / (num_f - 1)
        f_lo = beta_min * calculate_holtsmark_field(ne_lo, Z_bar)
        f_hi = max_beta * calculate_holtsmark_field(ne_hi, Z_bar)
        count = int(np.ceil(np.log(f_hi / f_lo) / log_step * (1 - 1e-12))) + 1
        self.fields = f_lo * np.exp(log_step * np.arange(count))
        self._cache = {}

    @property
    def cached_points(self):
        """Number of (field, angle) nodes whose dressed states are cached."""
        return len(self._cache)

    def weights(self, Ne_m3, Te_ev, Ti_ev=None, **microfield):
        """Microfield weights on the table's field grid at (Ne, Te, Ti)."""
        lo, hi = self.Ne_range
        if not lo * (1 - 1e-12) <= Ne_m3 <= hi * (1 + 1e-12):
            raise ValueError(f"Ne_m3 = {Ne_m3:.3e} outside the table range {lo:.3e} - {hi:.3e}.")
        return microfield_weights(self.fields, Ne_m3, Te_ev, max_beta=self.max_beta, Z_bar=self.Z_bar,
                                  Ti_ev=Ti_ev, rule="log", beta_min=self.beta_min, **microfield)

    def profile(self, energies_ev, Ne_m3, Te_ev, Ti_ev=None, **kwargs):
        """(pi, sigma+, sigma-) on energies_ev at (Ne, Te, Ti), as ``calculate_static_profile``.

        kwargs are its per-call options (frequency_dependent_width, apply_doppler,
        method, fft_tol, the microfield options, ...); the options of FIXED_OPTIONS,
        the quadrature and the electron interference operator belong to the table.
        """
        fixed = set(kwargs) & (set(FIXED_OPTIONS) | {"num_f", "num_mu", "max_beta", "Z_bar",
                                                       "radiator", "electron_interference"})
        if fixed:
            raise TypeError(f"{sorted(fixed)} are fixed by the table (or unsupported).")
        microfield = {k: kwargs[k] for k in MICROFIELD_OPTIONS if k in kwargs}
        if "emitter_charge" not in microfield:
            microfield["emitter_charge"] = self.radiator.default_emitter_charge
        weights = self.weights(Ne_m3, Te_ev, Ti_ev, **microfield)
        return calculate_static_profile(
            self.n_u, self.n_l, self.Z, self.B, Ne_m3, Te_ev, energies_ev,
            num_mu=self.num_mu, max_beta=self.max_beta, Z_bar=self.Z_bar, Ti_ev=Ti_ev,
            radiator=self.radiator, **self.options, **kwargs,
            _quadrature=(self.fields, weights), _point_cache=self._cache)
