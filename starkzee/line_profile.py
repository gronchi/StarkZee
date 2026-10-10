r"""
line_profile.py — High-level LineProfile class for Stark-Zeeman calculations.

Typical usage::

    import numpy as np
    from starkzee.line_profile import LineProfile

    lp = LineProfile(n_u=3, n_l=2, B=100.0, Ne_m3=1e23, Te_ev=5.0, species='H')

    # Supply a grid in any of: energy [eV], wavelength [nm],
    # frequency [THz], or wavenumber [cm⁻¹].
    lp.compute_profile(np.linspace(1.80, 2.00, 2000), grid_type='energy_ev')
    lp.compute_profile(np.linspace(620, 660, 2000),   grid_type='wavelength_nm')
    lp.compute_profile(np.linspace(450, 490, 2000),   grid_type='frequency_thz')
    lp.compute_profile(np.linspace(15000, 16000, 2000), grid_type='wavenumber_cm')

    # Static solver under its explicit name, and the dynamical-ion FFM solver
    # (requires Ti_ev; stores results in the same attributes):
    lp.compute_static_profile(np.linspace(620, 660, 2000), grid_type='wavelength_nm')
    lp.compute_ffm_profile(np.linspace(620, 660, 2000), grid_type='wavelength_nm',
                           sdt_bin_tol=1e-5)

    # Results are always available in all unit systems
    lp.energies_ev          # energy grid [eV]
    lp.detuning_ev          # E - E0       [eV]
    lp.wavelengths_nm       # wavelength grid [nm]
    lp.frequencies_thz      # frequency grid [THz]
    lp.wavenumbers_cm       # wavenumber grid [cm⁻¹]
    lp.detuning_nm          # λ - λ0       [nm]
    lp.detuning_thz         # f - f0       [THz]
    lp.detuning_cm          # ν̃ - ν̃0     [cm⁻¹]
    lp.profile_pi           # π component
    lp.profile_sig_plus     # σ+ component
    lp.profile_sig_minus    # σ- component
    lp.profile              # intensity at view_angle_deg (default 90°)
    lp.profile_transverse   # π + 0.5*(σ+ + σ-)  (90° observation)
    lp.profile_parallel     # σ+ + σ-             (0° observation)

    # use_empirical_data=True also populates the NIST-based counterpart of E0
    # (self.E0 itself stays on the analytic reference for backward
    # compatibility -- see D08/C05 in StarkZee_audit_response.md):
    lp.compute_profile(np.linspace(620, 660, 2000), grid_type='wavelength_nm',
                       use_empirical_data=True, atom='H')
    lp.E0_empirical              # empirical gross-structure line center [eV]
    lp.E0_empirical_wavelength_nm
    lp.E0_empirical_wavelength_air_nm
    lp.E0_empirical_frequency_thz
    lp.E0_empirical_wavenumber_cm

    # Discrete transitions at a single field configuration
    lp.compute_discrete(Fz=0.0, Fx=1e8)
    lp.discrete.energy_ev    # transition energies [eV]
    lp.discrete.detuning_ev  # detuning from E0   [eV]
    lp.discrete.wavelength_nm
    lp.discrete.frequency_thz
    lp.discrete.wavenumber_cm
    lp.discrete.q            # polarization channel (0 = π, -1 = σ+, +1 = σ-)
    lp.discrete.strength     # |d_q|²  [a₀²]
"""

import numpy as np

from starkzee.utils import (
    reduced_mass_rydberg_ev,
    energy_ev_to_wavelength_nm,
    vacuum_to_air_wavelength_nm,
    wavelength_nm_to_energy_ev,
    energy_ev_to_frequency_thz,
    frequency_thz_to_energy_ev,
    energy_ev_to_wavenumber_cm,
    wavenumber_cm_to_energy_ev,
)
from starkzee.static_profile import (
    calculate_static_profile,
    discrete_transitions,
)
from starkzee.ffm import calculate_ffm_profile


def _empirical_gross_structure_energy_ev(n_u, n_l, atom):
    """Return the empirical (NIST, shell-averaged) gross-structure transition
    energy [eV] for n_u -> n_l.

    Unlike the B-dependent conditioning reference used internally by
    :func:`~starkzee.static_profile.calculate_static_profile` (which folds in
    the diagonal quadratic-Zeeman shift at the requested B for numerical
    conditioning, and is not itself meant to be a physically meaningful line
    center), this is the plain zero-field shell-averaged NIST energy
    difference -- independent of B and quadratic_zeeman, directly comparable
    to the analytic :attr:`LineProfile.E0`.
    """
    from starkzee.atomic_data import empirical_shell_energy_cm
    wavenumber_cm = (empirical_shell_energy_cm(atom, n_u)
                     - empirical_shell_energy_cm(atom, n_l))
    return wavenumber_cm_to_energy_ev(wavenumber_cm)


class DiscreteTransitions:
    r"""Discrete Stark-Zeeman dipole transitions at a single (B, Fz, Fx) configuration.

    Attributes
    ----------
    energy_ev : ndarray
        Transition energies E_upper − E_lower [eV], sorted ascending.
    detuning_ev : ndarray
        Detuning from the field-free line center [eV].
    wavelength_nm : ndarray
        Photon wavelength for each transition [nm].
    frequency_thz : ndarray
        Photon frequency [THz].
    wavenumber_cm : ndarray
        Photon wavenumber [cm⁻¹].
    q : ndarray of int
        Polarization index q = m_l − m_u: 0 = π; **−1 = σ+** (Δm = m_u − m_l
        = +1, blue-shifted at B > 0); **+1 = σ−** (red-shifted).
    strength : ndarray
        Dipole matrix element squared :math:`|d_q(i \to j)|^2` [:math:`a_0^2`].
    upper_idx : ndarray of int
        Upper eigenstate index (0 … 2n_u²−1).
    lower_idx : ndarray of int
        Lower eigenstate index (0 … 2n_l²−1).
    einstein_a_s : ndarray
        Lazily evaluated Einstein A coefficient of every dressed transition
        [s⁻¹].
    upper_partial_decay_rate_s : ndarray
        Lazily evaluated E1 decay rate from every upper eigenstate into this
        object's lower principal shell [s⁻¹]. Other lower shells and non-E1
        decay channels are excluded.
    """

    def __init__(self, energy_ev, q, strength, upper_idx, lower_idx, E0,
                 upper_dimension=None):
        self.energy_ev     = energy_ev
        self.detuning_ev   = energy_ev - E0
        self.wavelength_nm = energy_ev_to_wavelength_nm(energy_ev)
        self.detuning_nm   = self.wavelength_nm - energy_ev_to_wavelength_nm(np.asarray(E0))
        self.frequency_thz = energy_ev_to_frequency_thz(energy_ev)
        self.detuning_thz  = self.frequency_thz - energy_ev_to_frequency_thz(np.asarray(E0))
        self.wavenumber_cm = energy_ev_to_wavenumber_cm(energy_ev)
        self.detuning_cm   = self.wavenumber_cm - energy_ev_to_wavenumber_cm(np.asarray(E0))
        self.q             = q
        self.strength      = strength
        self.upper_idx     = upper_idx
        self.lower_idx     = lower_idx
        self.upper_dimension = upper_dimension
        self._einstein_a_s = None
        self._upper_partial_decay_rate_s = None

    @property
    def einstein_a_s(self):
        """Einstein A coefficient of every dressed transition [s^-1].

        Values use each field-shifted transition energy and dressed dipole
        strength. They are evaluated lazily and do not affect profile-solver
        runtime.
        """
        if self._einstein_a_s is None:
            from starkzee.atomic import einstein_a_from_strength
            self._einstein_a_s = einstein_a_from_strength(
                self.energy_ev, self.strength)
        return self._einstein_a_s

    @property
    def upper_partial_decay_rate_s(self):
        """Rate from each upper eigenstate into this lower shell [s^-1].

        This is a partial E1 decay rate. Other lower principal shells and
        non-E1 channels are not included.
        """
        if self._upper_partial_decay_rate_s is None:
            dimension = self.upper_dimension
            if dimension is None:
                dimension = int(self.upper_idx.max()) + 1 if self.upper_idx.size else 0
            self._upper_partial_decay_rate_s = np.bincount(
                self.upper_idx, weights=self.einstein_a_s,
                minlength=dimension).astype(float, copy=False)
        return self._upper_partial_decay_rate_s

    def __repr__(self):
        qs = np.unique(self.q).tolist()
        return (f"DiscreteTransitions({len(self.energy_ev)} transitions, "
                f"polarizations={qs})")


class LineProfile:
    """Stark-Zeeman line profile for a single hydrogenic n_u → n_l transition.

    Stores all input parameters and computed results as attributes.  Call
    :meth:`compute_profile` to run the static profile solver and
    :meth:`compute_discrete` to enumerate the discrete eigenstate transitions
    at a specific field configuration.

    Parameters
    ----------
    n_u, n_l : int
        Upper and lower principal quantum numbers.
    B : float
        Magnetic field [T].
    Ne_m3 : float
        Electron density [m⁻³].
    Te_ev : float
        Electron temperature [eV].
    species : str, optional
        Emitting species: ``'H'`` / ``'hydrogen'``, ``'D'`` / ``'deuterium'``,
        or ``'T'`` / ``'tritium'``.  Default is ``'H'``.
    Ti_ev : float, optional
        Ion temperature [eV].  When supplied, :meth:`compute_profile` applies
        thermal Doppler broadening automatically after the static Stark-Zeeman
        calculation.  Default is ``None`` (no Doppler broadening).
    view_angle_deg : float, optional
        Default observation angle relative to **B** [degrees].  Used by
        :meth:`compute_profile` to set ``lp.profile`` via the Stokes formula.
        Can be overridden per-call in :meth:`compute_profile`.  Default is
        ``90`` (perpendicular to **B**).
    """

    #: Accepted values for the ``grid_type`` parameter of :meth:`compute_profile`.
    GRID_TYPES = ('energy_ev', 'wavelength_nm', 'frequency_thz', 'wavenumber_cm')

    def __init__(self, n_u, n_l, B, Ne_m3, Te_ev, species='H', Ti_ev=None, view_angle_deg=90):
        from starkzee.utils import species_to_ZA
        Z, A = species_to_ZA(species)
        self.n_u     = n_u
        self.n_l     = n_l
        self.species = species
        self.Z       = Z
        self.A       = A
        self.B       = B
        self.Ne_m3   = Ne_m3
        self.Te_ev          = Te_ev
        self.Ti_ev          = Ti_ev
        self.view_angle_deg = view_angle_deg

        # Field-free line center — use reduced-mass-corrected Rydberg so that
        # E0 matches the observed transition energy for each species.
        self.E0                  = Z**2 * reduced_mass_rydberg_ev(Z, A) * (1.0/n_l**2 - 1.0/n_u**2)
        self.E0_wavelength_nm     = energy_ev_to_wavelength_nm(self.E0)
        self.E0_wavelength_air_nm = vacuum_to_air_wavelength_nm(self.E0_wavelength_nm)
        self.E0_frequency_thz     = energy_ev_to_frequency_thz(self.E0)
        self.E0_wavenumber_cm     = energy_ev_to_wavenumber_cm(self.E0)
        self.reference_energy_ev = self.E0
        self.result_metadata = None

        # Empirical (NIST) counterpart of E0 -- populated by compute_profile()/
        # compute_static_profile()/compute_ffm_profile() only when called with
        # use_empirical_data=True (the default); None after analytical calls.
        # self.E0 above is NOT updated
        # in that case (kept as the analytic reference for backward
        # compatibility) -- use E0_empirical explicitly when you need the
        # measured value. See D08/C05 in StarkZee_audit_response.md.
        self.E0_empirical                  = None
        self.E0_empirical_wavelength_nm     = None
        self.E0_empirical_wavelength_air_nm = None
        self.E0_empirical_frequency_thz     = None
        self.E0_empirical_wavenumber_cm     = None

        # Profile results — set by compute_profile()
        self.energies_ev       = None
        self.detuning_ev       = None
        self.wavelengths_nm    = None
        self.wavelengths_air_nm = None
        self.detuning_nm       = None
        self.frequencies_thz   = None
        self.detuning_thz      = None
        self.wavenumbers_cm    = None
        self.detuning_cm       = None
        self.profile_pi        = None
        self.profile_sig_plus  = None
        self.profile_sig_minus = None
        self.profile           = None  # set by compute_profile() via Stokes formula at view_angle_deg

        # Discrete transitions — set by compute_discrete()
        self.discrete = None

    # ── Field-free atomic conveniences ──────────────────────────────────────

    def radial_wavefunction(self, r_a0, l, shell='upper'):
        r"""Return a field-free radial basis function for this line.

        Parameters
        ----------
        r_a0 : float or array-like
            Radius in Bohr radii.
        l : int
            Orbital angular momentum.
        shell : {'upper', 'lower'}, optional
            Select ``n_u`` or ``n_l`` (default ``'upper'``).
        """
        if shell == 'upper':
            n = self.n_u
        elif shell == 'lower':
            n = self.n_l
        else:
            raise ValueError("shell must be 'upper' or 'lower'")
        from starkzee.atomic import radial_wavefunction
        return radial_wavefunction(r_a0, n, l, self.Z)

    def radial_dipole(self, l_u, l_l, method=None):
        r"""Return this line's signed field-free radial dipole integral [a0]."""
        from starkzee.atomic import radial_dipole
        return radial_dipole(
            self.n_u, l_u, self.n_l, l_l, self.Z, method=method)

    @property
    def field_free_einstein_a_s(self):
        """Isotope-aware field-free shell-averaged Einstein A [s^-1]."""
        from starkzee.atomic import einstein_a
        return einstein_a(self.n_u, self.n_l, self.Z, A=self.A)

    # ── Profile computation ──────────────────────────────────────────────────

    @staticmethod
    def _to_energy_ev(grid, grid_type):
        """Convert *grid* from *grid_type* units to energy [eV]."""
        grid = np.asarray(grid, dtype=float)
        if grid_type == 'energy_ev':
            return grid
        if grid_type == 'wavelength_nm':
            return wavelength_nm_to_energy_ev(grid)
        if grid_type == 'frequency_thz':
            return frequency_thz_to_energy_ev(grid)
        if grid_type == 'wavenumber_cm':
            return wavenumber_cm_to_energy_ev(grid)
        raise ValueError(
            f"Unknown grid_type '{grid_type}'. "
            f"Choose from {LineProfile.GRID_TYPES}."
        )

    def compute_profile(self, grid, grid_type='energy_ev', view_angle_deg=None, **kwargs):
        """Compute the static Stark-Zeeman profile on the supplied grid.

        Parameters
        ----------
        grid : array-like
            Spectral axis values in the units specified by *grid_type*.
        grid_type : str, optional
            Unit system of *grid*.  One of:

            ``'energy_ev'``
                Photon energy [eV] (default).
            ``'wavelength_nm'``
                Vacuum wavelength [nm].
            ``'frequency_thz'``
                Photon frequency [THz].
            ``'wavenumber_cm'``
                Wavenumber [cm⁻¹].

        view_angle_deg : float, optional
            Observation angle relative to **B** [degrees] for this call.  When
            given, overrides the class-level default for ``lp.profile``; the
            class attribute is not modified.  When omitted, falls back to
            ``self.view_angle_deg`` (default ``90``).

        **kwargs
            Forwarded to
            :func:`~starkzee.static_profile.calculate_static_profile`
            (``num_f``, ``num_mu``, ``use_screening``, ``quadratic_zeeman``,
            ``fine_structure``, ``frequency_dependent_width``).
            ``natural_width_mode`` selects state-resolved (default) or legacy
            shell-average radiative damping. ``electron_interference=True``
            selects the opt-in full PPP impact-limit collision operator.

        Returns
        -------
        self
            Allows method chaining.

        Notes
        -----
        After this call all spectral-axis attributes are populated regardless
        of which *grid_type* was supplied:
        ``energies_ev``, ``wavelengths_nm``, ``frequencies_thz``,
        ``wavenumbers_cm``, and the corresponding ``detuning_*`` arrays.
        """
        energies_ev = self._to_energy_ev(grid, grid_type)
        kwargs.setdefault('atom', {1: 'H', 2: 'D', 3: 'T'}.get(self.A, self.species))

        pi, sp, sm = calculate_static_profile(
            n_u=self.n_u, n_l=self.n_l, Z=self.Z,
            B=self.B, Ne_m3=self.Ne_m3, Te_ev=self.Te_ev,
            energies_ev=energies_ev,
            A=self.A,
            Ti_ev=self.Ti_ev,
            species=self.species,
            **kwargs,
        )
        self._maybe_set_empirical_reference(kwargs)
        self._store_profile(energies_ev, pi, sp, sm, view_angle_deg)
        return self

    # ``compute_static_profile`` is the explicit name used since the FFM became
    # available on this class; ``compute_profile`` is kept as the historical
    # alias so existing scripts keep working.
    def compute_static_profile(self, grid, grid_type='energy_ev', view_angle_deg=None, **kwargs):
        """Alias of :meth:`compute_profile` (static-ion Stark-Zeeman solver)."""
        return self.compute_profile(grid, grid_type=grid_type,
                                    view_angle_deg=view_angle_deg, **kwargs)

    def compute_ffm_profile(self, grid, grid_type='energy_ev', view_angle_deg=None, **kwargs):
        """Compute the dynamical (FFM) Stark-Zeeman profile on the supplied grid.

        Runs :func:`~starkzee.ffm.calculate_ffm_profile` with this profile's
        stored plasma parameters (``Ti_ev`` is required — set it in the
        constructor; ``A`` is the emitter mass and remains the perturber mass
        only when ``A_perturber`` is omitted) and stores the results in the same attributes
        as :meth:`compute_profile` (``profile_pi``, ``profile_sig_plus``,
        ``profile_sig_minus``, ``profile``, and every spectral-axis /
        ``detuning_*`` array).

        Parameters
        ----------
        grid, grid_type, view_angle_deg
            Same meaning as in :meth:`compute_profile`.
        **kwargs
            Forwarded to :func:`~starkzee.ffm.calculate_ffm_profile`
            (``num_f``, ``num_mu``, ``max_beta``, ``use_screening``,
            ``quadratic_zeeman``, ``fine_structure``, ``numerical_inversion``,
            ``use_empirical_data``, ``atom``, ``electron_model``,
            ``parallel_stark``, ``apply_doppler``, ``sdt_bin_tol``,
            ``sdt_frequency_dependent_width``, ``natural_width_mode``,
            ``interference_diagnostics``,
            ``interference_group_tolerance_ev``,
            ``interference_group_width_tolerance_ev``, and
            ``interference_group_profile_rtol``, ``emitter_charge``,
            ``A_perturber``, and ``fluctuation_rate_model``).
            ``electron_interference=True`` passes the complex PPP SDTs into
            the analytical FFM.

        Returns
        -------
        self
            Allows method chaining.
        """
        if self.Ti_ev is None:
            raise ValueError(
                "compute_ffm_profile requires Ti_ev (ion temperature); "
                "pass it to the LineProfile constructor.")
        energies_ev = self._to_energy_ev(grid, grid_type)
        kwargs.setdefault('atom', {1: 'H', 2: 'D', 3: 'T'}.get(self.A, self.species))

        pi, sp, sm = calculate_ffm_profile(
            n_u=self.n_u, n_l=self.n_l, Z=self.Z, B=self.B,
            Ne_m3=self.Ne_m3, Te_ev=self.Te_ev, Ti_ev=self.Ti_ev,
            A_ion=self.A, energies_ev=energies_ev,
            **kwargs,
        )
        self._maybe_set_empirical_reference(kwargs)
        self._store_profile(energies_ev, pi, sp, sm, view_angle_deg)
        return self

    def _maybe_set_empirical_reference(self, kwargs):
        """Populate E0_empirical (and its derived units) when the caller
        uses empirical energies (the default); clear stale values otherwise.

        Reads use_empirical_data/atom from the same kwargs dict that gets
        forwarded to the solver, so it always reflects what that specific
        call actually asked for.
        """
        from starkzee.static_profile import line_reference_energy
        self.reference_energy_ev = line_reference_energy(
            self.n_u, self.n_l, self.Z, self.A,
            kwargs.get('use_empirical_data', True), kwargs.get('atom', self.species))
        self.result_metadata = dict(
            reference_energy_ev=self.reference_energy_ev,
            use_empirical_data=kwargs.get('use_empirical_data', True),
            atom=kwargs.get('atom', self.species), density_unit='per eV',
            electron_interference=kwargs.get('electron_interference', False),
            microfield_model=('custom' if kwargs.get('custom_table_path') else kwargs.get('microfield_model')) or
                ('potekhin' if kwargs.get('use_screening', True) else 'holtsmark'),
            charged=kwargs.get('charged', kwargs.get('emitter_charge', self.Z - 1) != 0),
            emitter_charge=kwargs.get('emitter_charge', self.Z - 1),
            Z_bar=kwargs.get('Z_bar', 1.0),
            A_perturber=kwargs.get('A_perturber', self.A),
            fluctuation_rate_model=kwargs.get('fluctuation_rate_model', 'zest'))
        if not kwargs.get('use_empirical_data', True):
            for suffix in ('', '_wavelength_nm', '_wavelength_air_nm', '_frequency_thz', '_wavenumber_cm'):
                setattr(self, 'E0_empirical' + suffix, None)
            return
        atom = kwargs.get('atom', 'H')
        self.E0_empirical = _empirical_gross_structure_energy_ev(self.n_u, self.n_l, atom)
        self.E0_empirical_wavelength_nm = energy_ev_to_wavelength_nm(self.E0_empirical)
        self.E0_empirical_wavelength_air_nm = vacuum_to_air_wavelength_nm(self.E0_empirical_wavelength_nm)
        self.E0_empirical_frequency_thz = energy_ev_to_frequency_thz(self.E0_empirical)
        self.E0_empirical_wavenumber_cm = energy_ev_to_wavenumber_cm(self.E0_empirical)

    def _store_profile(self, energies_ev, pi, sp, sm, view_angle_deg=None):
        """Store solver output and populate every spectral-axis attribute."""
        self.energies_ev      = energies_ev
        self.detuning_ev      = energies_ev - self.reference_energy_ev
        self.wavelengths_nm   = energy_ev_to_wavelength_nm(energies_ev)
        self.wavelengths_air_nm = vacuum_to_air_wavelength_nm(self.wavelengths_nm)
        self.detuning_nm      = self.wavelengths_nm - energy_ev_to_wavelength_nm(self.reference_energy_ev)
        self.frequencies_thz = energy_ev_to_frequency_thz(energies_ev)
        self.detuning_thz    = self.frequencies_thz - energy_ev_to_frequency_thz(self.reference_energy_ev)
        self.wavenumbers_cm  = energy_ev_to_wavenumber_cm(energies_ev)
        self.detuning_cm     = self.wavenumbers_cm - energy_ev_to_wavenumber_cm(self.reference_energy_ev)

        self.profile_pi        = pi
        self.profile_sig_plus  = sp
        self.profile_sig_minus = sm

        angle = view_angle_deg if view_angle_deg is not None else self.view_angle_deg
        self.profile = self.profile_at_angle(angle)

    # ── Discrete transitions ─────────────────────────────────────────────────

    def compute_discrete(self, Fz=0.0, Fx=0.0, **kwargs):
        """Enumerate discrete eigenstate transitions at field (Fz, Fx) [V/m].

        All keyword arguments are forwarded to
        :func:`~starkzee.static_profile.discrete_transitions`
        (``quadratic_zeeman``, ``fine_structure``, ``min_strength``).

        Returns *self* to allow method chaining.
        """
        kwargs.setdefault('A', self.A)
        kwargs.setdefault('atom', {1: 'H', 2: 'D', 3: 'T'}.get(self.A, self.species))
        raw = discrete_transitions(
            n_u=self.n_u, n_l=self.n_l, Z=self.Z,
            B=self.B, Fz=Fz, Fx=Fx,
            **kwargs,
        )
        from starkzee.static_profile import line_reference_energy
        reference = line_reference_energy(self.n_u, self.n_l, self.Z, kwargs['A'],
                                         kwargs.get('use_empirical_data', True), kwargs['atom'])
        self.discrete = DiscreteTransitions(
            energy_ev  = raw['energy_ev'],
            q          = raw['q'],
            strength   = raw['strength'],
            upper_idx  = raw['upper_idx'],
            lower_idx  = raw['lower_idx'],
            E0         = reference,
            upper_dimension = 2 * self.n_u**2,
        )
        return self

    # ── Derived profile observables ──────────────────────────────────────────

    def spectral_density(self, grid_type='energy_ev', theta_deg=None):
        """Return (coordinate, intensity per coordinate unit), with Jacobian.

        Stored profile arrays remain per eV for compatibility. Wavelengths
        here are vacuum nm; reference wrappers use air wavelengths per metre
        and must be converted before comparing absolute densities.
        """
        if self.profile is None:
            raise ValueError('Compute a profile first.')
        values = self.profile if theta_deg is None else self.profile_at_angle(theta_deg)
        if grid_type == 'energy_ev':
            return self.energies_ev.copy(), values.copy()
        if grid_type == 'wavelength_nm':
            return self.wavelengths_nm.copy(), values * self.energies_ev / self.wavelengths_nm
        if grid_type == 'frequency_thz':
            return self.frequencies_thz.copy(), values * self.energies_ev / self.frequencies_thz
        if grid_type == 'wavenumber_cm':
            return self.wavenumbers_cm.copy(), values * self.energies_ev / self.wavenumbers_cm
        raise ValueError('Choose energy_ev, wavelength_nm, frequency_thz or wavenumber_cm.')

    @property
    def profile_transverse(self):
        """π + 0.5*(σ+ + σ−) — total intensity at 90° to B."""
        if self.profile_pi is None:
            return None
        return self.profile_pi + 0.5 * (self.profile_sig_plus + self.profile_sig_minus)

    @property
    def profile_parallel(self):
        """σ+ + σ− — total intensity along B (0°)."""
        if self.profile_sig_plus is None:
            return None
        return self.profile_sig_plus + self.profile_sig_minus

    def profile_at_angle(self, theta_deg):
        """Total intensity at observation angle *theta_deg* relative to B.

        Uses the standard Stokes formula:
            I(θ) = I_π sin²θ + ½(I_σ+ + I_σ−)(1 + cos²θ)
        """
        if self.profile_pi is None:
            return None
        theta = np.deg2rad(theta_deg)
        s2, c2 = np.sin(theta)**2, np.cos(theta)**2
        return (self.profile_pi * s2
                + 0.5 * (self.profile_sig_plus + self.profile_sig_minus) * (1.0 + c2))

    # ── Dunder ───────────────────────────────────────────────────────────────

    def __repr__(self):
        parts = []
        if self.profile_pi is not None:
            parts.append(f"profile({len(self.energies_ev)} pts)")
        if self.discrete is not None:
            parts.append(f"discrete({len(self.discrete.energy_ev)} transitions)")
        state = ", ".join(parts) if parts else "no results"
        return (f"LineProfile(n={self.n_u}→{self.n_l}, Z={self.Z}, "
                f"B={self.B} T, Ne={self.Ne_m3:.1e} m⁻³, [{state}])")
