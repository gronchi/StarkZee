# Stark-Zeeman broadening and line profile calculations for starkzee

import warnings

import numpy as np
from scipy.special import voigt_profile
from functools import lru_cache
from starkzee.utils import A0, reduced_mass_rydberg_ev, wavenumber_cm_to_energy_ev, energy_ev_to_wavenumber_cm
from scipy.constants import hbar as _HBAR, e as _E_CHARGE, m_p as _M_P, c as _C_LIGHT
from scipy.constants import atomic_mass as _ATOMIC_MASS


from starkzee.angular import angular_dipole_element
from starkzee.radiator import (
    build_hamiltonian, build_basis, basis_index, intrashell_radial_dipole,
    _uncoupled_dipole_matrices,
)
from starkzee.microfield import microfield_quadrature
from starkzee.radiators import HydrogenicTransition
from starkzee.convolutions import uniform_energy_grid
from starkzee.collision import generalized_lorentzian_components

try:                                    # optional: fused, multithreaded kernel evaluation
    import numexpr as _numexpr
except ImportError:                     # pragma: no cover - numpy fallback
    _numexpr = None

# numexpr pays off only above this many grid x transition elements
NUMEXPR_MIN_ELEMENTS = 20_000
# memory budget [bytes] for the eigenvector batch of one manifold
EIGH_BATCH_BYTES = 64 * 2**20

def _isotope_from_species(Z, A, species):
    """Mass number from A and/or a hydrogen-isotope species name; they must agree with Z and each other."""
    if species is None:
        return A
    from starkzee.utils import species_to_ZA
    Z_s, A_s = species_to_ZA(species)
    if Z_s != Z:
        raise ValueError(f"species={species!r} has Z={Z_s}, but Z={Z} was given.")
    if A is not None and A != A_s:
        raise ValueError(f"species={species!r} is A={A_s}, but A={A} was given.")
    return A_s


@lru_cache(maxsize=None)
def line_reference_energy(n_u, n_l, Z=1, A=None, use_empirical_data=False, atom=None):
    """Fixed zero-field shell-trace reference [eV], or analytic gross energy.

    Empirical means include fine structure and Lamb shifts; this is a defined
    shell reference, not an intensity-weighted observed centroid.
    """
    if use_empirical_data:
        upper = build_hamiltonian(n_u, Z, 0, True, True, A, True, atom)
        lower = build_hamiltonian(n_l, Z, 0, True, True, A, True, atom)
        return float(wavenumber_cm_to_energy_ev(np.trace(upper).real / len(upper)
                                                - np.trace(lower).real / len(lower)))
    return Z**2 * reduced_mass_rydberg_ev(Z, A) * (1 / n_l**2 - 1 / n_u**2)


def _stark_templates(n, Z, A=None):
    """M_z, M_x [eV/(V/m)] of shell n with lengths in the reduced Bohr radius (see _stark_templates_a0)."""
    from starkzee.utils import mass_number, reduced_mass_factor
    A = mass_number(Z, A)
    return _stark_templates_cached(n, Z, A)


@lru_cache(maxsize=None)
def _stark_templates_cached(n, Z, A):
    M_z, M_x = _stark_templates_a0(n, Z)
    from starkzee.utils import reduced_mass_factor
    scale = 1.0 / reduced_mass_factor(Z, A)          # a_mu / a0
    return M_z * scale, M_x * scale


@lru_cache(maxsize=None)
def _stark_templates_a0(n, Z):
    """Return the field-independent Stark coupling matrices M_z and M_x [eV/(V/m)].

    V_E(Fz, Fx) = Fz * M_z + Fx * M_x, so these only need to be built once
    per (n, Z) pair regardless of how many microfield quadrature points are used.

    Sign convention: the intra-shell radial element is taken as
    +(3n/2Z)√(n²−l²), which is the *opposite* sign of the actual integral with
    the standard radial functions (⟨2s|r|2p⟩ = −3√3 a₀; see
    :func:`~starkzee.radiator.radial_dipole`).  The two conventions are related
    by the rephasing |n,l,m⟩ → (−1)^l |n,l,m⟩, which flips every Δl = ±1
    element uniformly and leaves eigenvalues and all |d|² intensities
    unchanged — but any *new* operator mixing this convention with the signed
    radial integrals (e.g. an off-diagonal broadening operator) must pick one
    convention consistently.
    """
    basis = build_basis(n)
    dim = len(basis)
    M_z = np.zeros((dim, dim), dtype=complex)
    M_x = np.zeros((dim, dim), dtype=complex)
    index = basis_index(n)
    for i, state_i in enumerate(basis):
        # z and x keep m_s, change l by 1 and m_l by at most 1
        for l2 in (state_i.l - 1, state_i.l + 1):
            for ml2 in (state_i.ml - 1, state_i.ml, state_i.ml + 1):
                j = index.get((l2, ml2, state_i.ms))
                if j is None:
                    continue
                state_j = basis[j]
                r_val = intrashell_radial_dipole(n, state_i.l, state_j.l, Z)
                z_ang = angular_dipole_element(state_i.l, state_i.ml, state_j.l, state_j.ml, 0)
                x_ang = (angular_dipole_element(state_i.l, state_i.ml, state_j.l, state_j.ml, -1) -
                         angular_dipole_element(state_i.l, state_i.ml, state_j.l, state_j.ml,  1)) / np.sqrt(2.0)
                M_z[i, j] += -z_ang * r_val * A0
                M_x[i, j] += -x_ang * r_val * A0
    return M_z, M_x


def build_stark_matrix(n, Z, Fz, Fx, A=None):
    """Build the (2n²) × (2n²) Stark electric-field perturbation matrix in eV.

    The linear Stark interaction for an electron in an external electric field
    F = Fz ẑ + Fx x̂ is:

        V_E = −e (z Fz + x Fx)

    In the hydrogenic basis the matrix elements reduce to products of a radial
    element and an angular element.  The within-shell (Δn = 0) radial element
    ⟨n, l | r | n, l±1⟩ is given analytically:

        ⟨n, l | r | n, l−1⟩ = (3n/2Z) √(n² − l²)   [a₀]

    The angular elements ⟨l, m_l | cos θ | l±1, m_l⟩ (for Fz) and the
    combinations for Fx are provided by :func:`~starkzee.angular.angular_dipole_element`.

    The x-component is constructed as:

        x/r = (T_{−1} + T_{+1}) / √2

    where T_q are the spherical tensor components of r̂.

    Parameters
    ----------
    n : int
        Principal quantum number.
    Z : int
        Nuclear charge.
    Fz : float
        Electric field component along B (z-axis) [V m⁻¹].
    Fx : float
        Electric field component perpendicular to B (x-axis) [V m⁻¹].

    Returns
    -------
    ndarray, shape (2n², 2n²), dtype complex
        Hermitian Stark perturbation matrix in eV.

    Notes
    -----
    This matrix operates within a single n-shell; the quadratic Stark effect
    (coupling to n ± 1, ±2 shells) is neglected, which is valid when the
    Stark shift ≪ the shell spacing Z² Ry (1/n² − 1/(n+1)²).
    """
    basis = build_basis(n)
    dim = len(basis)
    V_E = np.zeros((dim, dim), dtype=complex)
    
    # Hydrogenic radial matrix element within same n (intrashell_radial_dipole):
    # <n, l | r | n, l-1> = (3n / 2Z) * sqrt(n^2 - l^2)
    for i, state_i in enumerate(basis):
        for j, state_j in enumerate(basis):
            if state_i.ms == state_j.ms and abs(state_i.l - state_j.l) == 1:
                r_val = intrashell_radial_dipole(n, state_i.l, state_j.l, Z)
                
                # z coupling (q=0)
                z_ang = angular_dipole_element(state_i.l, state_i.ml, state_j.l, state_j.ml, 0)
                
                # x coupling: x/r = (T_{-1} - T_{+1})/√2 with the standard
                # components T_{+1} = -(x+iy)/(r√2), T_{-1} = (x-iy)/(r√2).
                x_ang = (angular_dipole_element(state_i.l, state_i.ml, state_j.l, state_j.ml, -1) -
                         angular_dipole_element(state_i.l, state_i.ml, state_j.l, state_j.ml, 1)) / np.sqrt(2.0)
                
                V_E[i, j] += -(z_ang * Fz + x_ang * Fx) * r_val * A0

    from starkzee.utils import reduced_mass_factor
    return V_E / reduced_mass_factor(Z, A)          # lengths in the reduced Bohr radius

def solve_starkzee(n, Z, B, Fz, Fx, quadratic_zeeman=True,
                               fine_structure=True, A=None,
                               use_empirical_data=False, atom=None):
    """Diagonalize the combined Stark + Zeeman Hamiltonian for shell n.

    Adds the Stark perturbation :func:`build_stark_matrix` to the
    atomic/magnetic Hamiltonian :func:`~starkzee.radiator.build_hamiltonian` and
    diagonalizes the sum with ``numpy.linalg.eigh``:

        H = H_atom(B) + V_E(Fz, Fx)

    This is the inner-loop solver called for every (microfield magnitude,
    microfield angle) quadrature point during profile integration.

    Parameters
    ----------
    n : int
        Principal quantum number.
    Z : int
        Nuclear charge.
    B : float
        Magnetic field [T].
    Fz : float
        Electric field component along B [V m⁻¹].
    Fx : float
        Electric field component perpendicular to B [V m⁻¹].
    quadratic_zeeman : bool, optional
        Include the diamagnetic quadratic Zeeman term (default True).
    fine_structure : bool, optional
        Include MV + Darwin corrections (default True).
    use_empirical_data : bool, optional
        Use NIST empirical level energies for the field-free Hamiltonian
        instead of the analytic Rydberg formula (default False).  The
        internal Hamiltonian is assembled in cm⁻¹; eigenvalues are
        converted back to eV before returning so callers see no unit change.
    atom : str, optional
        Atom identifier passed to :func:`~starkzee.atomic_data.load_levels`
        when ``use_empirical_data=True`` (default: the isotope given by ``A``).

    Returns
    -------
    eigenvalues : ndarray, shape (2n²,)
        Energy eigenvalues in ascending order [eV].
    eigenvectors : ndarray, shape (2n², 2n²)
        Orthonormal eigenstates as columns, in the canonical ``|n, l, m_l, m_s⟩`` basis.
    """
    H_atom = build_hamiltonian(n, Z, B, quadratic_zeeman, fine_structure, A,
                               use_empirical_data=use_empirical_data, atom=atom)
    # V_E(Fz, Fx) = Fz*M_z + Fx*M_x with field-independent templates cached per
    # (n, Z), avoiding a Python double-loop rebuild of the Stark matrix on every
    # call (build_stark_matrix produces exactly Fz*M_z + Fx*M_x).
    M_z, M_x = _stark_templates(n, Z, A)

    if use_empirical_data:
        # H_atom is in cm⁻¹; scale the Stark matrices (eV/(V/m)) to cm⁻¹/(V/m).
        ev_to_cm = energy_ev_to_wavenumber_cm(1.0)
        H_total = H_atom + (Fz * M_z + Fx * M_x) * ev_to_cm
        # Center near 0 using the mean of the empirical levels (≈82259 cm⁻¹ for H n=2).
        E_shift = H_total.diagonal().real.mean()
        for i in range(H_total.shape[0]):
            H_total[i, i] -= E_shift
        eigenvalues_cm, eigenvectors = np.linalg.eigh(H_total)
        eigenvalues_cm += E_shift
        return wavenumber_cm_to_energy_ev(eigenvalues_cm), eigenvectors

    H_total = H_atom + Fz * M_z + Fx * M_x
    # Shift diagonal to center eigenvalues near 0, reducing the spectral norm.
    # This improves the eigh solver's numerical precision.
    En = -(Z**2) * reduced_mass_rydberg_ev(Z, A) / (n**2)
    for i in range(H_total.shape[0]):
        H_total[i, i] -= En
    eigenvalues, eigenvectors = np.linalg.eigh(H_total)
    eigenvalues += En
    return eigenvalues, eigenvectors

def _eigh_by_m(H, m_values):
    """eigh of H block by block in M (H must not couple different M); eigenvectors of pure M.

    Returns (energies ascending, vectors, M of each eigenvector).
    """
    m = np.asarray(m_values, dtype=float)
    energies = np.empty(len(m))
    vectors = np.zeros(H.shape, dtype=complex)
    labels = np.empty(len(m))
    start = 0
    for M in np.unique(m):
        idx = np.flatnonzero(m == M)
        e, v = np.linalg.eigh(H[np.ix_(idx, idx)])
        cols = slice(start, start + len(idx))
        energies[cols], labels[cols] = e, M
        vectors[np.ix_(idx, np.arange(start, start + len(idx)))] = v
        start += len(idx)
    order = np.argsort(energies, kind="stable")
    return energies[order], vectors[:, order], labels[order]


def _transition_widths(w_u, w_l, w_natural, scale_u=None, scale_l=None, floor=None):
    """Lorentzian half widths w = w_u*scale_u + w_l*scale_l + w_natural [eV], floored at `floor`.

    Every term is a scalar or broadcasts to (grid, transitions): the electron widths
    are columns over the grid (frequency-dependent) or scalars, the operator scales
    and state-resolved natural widths are rows over the transitions.
    """
    u = w_u if scale_u is None else w_u * scale_u
    l = w_l if scale_l is None else w_l * scale_l
    w = u + l + w_natural
    return w if floor is None else np.maximum(w, floor)


def _lorentzian_kernel(energies, centres, w_u, w_l, w_natural,
                       scale_u=None, scale_l=None, floor=None):
    """Area-normalized Lorentzians (w/pi) / ((E - E_k)^2 + w^2) on the grid x transitions.

    energies (N,), centres (P,), widths as in _transition_widths. The same arithmetic
    as the plain numpy expressions, evaluated in one fused, multithreaded pass when
    numexpr is available (widths included), otherwise in place.
    """
    E = energies[:, np.newaxis]
    c = centres[np.newaxis, :]
    terms = [t for t in (w_u, w_l, w_natural, scale_u, scale_l) if t is not None]
    full = np.prod(np.broadcast_shapes(*(np.shape(t) for t in terms))) >= E.size * c.size
    if _numexpr is not None and E.size * c.size >= NUMEXPR_MIN_ELEMENTS and not full:
        # widths vary along one axis only: build them first (cheap), fuse the rest
        w = _transition_widths(w_u, w_l, w_natural, scale_u, scale_l, floor)
        return _numexpr.evaluate("(w / pi) / ((E - c)**2 + w**2)",
                                 local_dict={"E": E, "c": c, "w": w, "pi": np.pi})
    if _numexpr is not None and E.size * c.size >= NUMEXPR_MIN_ELEMENTS:
        # widths span grid x transitions: build them inside the fused expression
        names = {"E": E, "c": c, "wu": w_u, "wl": w_l, "wn": w_natural, "pi": np.pi}
        u = "wu" if scale_u is None else "(wu * su)"
        l = "wl" if scale_l is None else "(wl * sl)"
        w = f"(({u} + {l}) + wn)"
        if floor is not None:
            w = f"where({w} < wmin, wmin, {w})"
            names["wmin"] = floor
        if scale_u is not None:
            names["su"] = scale_u
        if scale_l is not None:
            names["sl"] = scale_l
        return _numexpr.evaluate(f"({w} / pi) / ((E - c)**2 + {w}**2)", local_dict=names)
    w = _transition_widths(w_u, w_l, w_natural, scale_u, scale_l, floor)
    out = E - c
    out *= out
    out += np.square(w)
    return np.divide(np.divide(w, np.pi), out, out=out)


def _batched_eigh(H0, M_z, M_x, Fz, Fx):
    """Yield eigh(H0 + Fz*M_z + Fx*M_x) for every (Fz, Fx), in stacked eigh calls.

    The points are diagonalized in chunks of at most EIGH_BATCH_BYTES of
    eigenvectors, so memory stays bounded. numpy's stacked eigh runs the same
    routine per matrix: the result is identical to a loop of single calls.
    """
    n = H0.shape[0]
    chunk = max(1, int(EIGH_BATCH_BYTES // (16 * n * n)))
    for i in range(0, len(Fz), chunk):
        fz = Fz[i:i + chunk, np.newaxis, np.newaxis]
        fx = Fx[i:i + chunk, np.newaxis, np.newaxis]
        yield from zip(*np.linalg.eigh(H0 + fz * M_z + fx * M_x))


def calculate_static_profile(n_u, n_l, Z, B, Ne_m3, Te_ev, energies_ev,
                                     num_f=20, num_mu=6, max_beta=10.0,
                                     use_screening=True,
                                     quadratic_zeeman=True, fine_structure=True,
                                     frequency_dependent_width=True, A=None,
                                     Ti_ev=None, species=None, electron_model='pppb',
                                     electron_operator=False,
                                     electron_interference=False,
                                     use_empirical_data=True, atom=None,
                                     microfield_model=None,
                                     force_frequency_dependent_width=False,
                                     custom_table_path=None, charged=None,
                                     emitter_charge=None, Z_bar=1.0,
                                     apply_doppler=True, species_charges=None,
                                     species_concentrations=None,
                                     natural_width_mode='state_resolved', radiator=None,
                                     method='exact', fft_tol=1e-8,
                                     _quadrature=None, _point_cache=None):
    """Compute the static-ion Stark-Zeeman line profile for n_u → n_l.

    Integrates the Stark-Zeeman Hamiltonian over the plasma microfield distribution
    using Gauss-Legendre quadrature for both the field magnitude and the angle μ = cos θ
    between the microfield and B.  For each quadrature point the combined
    Hamiltonian H = H_atom(B) + V_E(Fz, Fx) is diagonalized and the transition
    intensities accumulated into three polarization components.

    Parameters
    ----------
    n_u, n_l : int
        Upper and lower principal quantum numbers.
    Z : int
        Nuclear charge (1 for hydrogen).
    B : float
        Magnetic field [T].  ``B=0`` is fully supported; at zero field the
        π/σ decomposition is physically meaningless (no preferred axis) and all
        three returned components are equal by spherical symmetry.
    Ne_m3 : float
        Electron density [m⁻³].
    Te_ev : float
        Electron temperature [eV].
    energies_ev : array-like
        Photon energies at which to evaluate the profile [eV].
    num_f : int, optional
        Number of microfield quadrature points (default 20). Converge it jointly
        with ``max_beta`` for the requested transition, plasma conditions,
        spectral window, and observable; the default is not a universal
        accuracy guarantee.
    num_mu : int, optional
        Number of Gauss-Legendre angle points (default 6). Check convergence
        for the requested field and polarization observable; the default is not
        a universal accuracy guarantee. Ignored at B = 0, where the problem is
        rotationally invariant: one field direction is solved and each
        polarization of the orientation average is 1/3 of its total.
    max_beta : float, optional
        Upper limit of the reduced-microfield grid β = F/F₀ (default 10).
        The Holtsmark tail beyond β = 10 carries ~3 % of the probability, which
        the quadrature truncates and renormalizes away; increase this (together
        with ``num_f``) when the quasi-static far wings matter.  Forwarded to
        :func:`~starkzee.microfield.microfield_quadrature`.
    use_screening : bool, optional
        If True (default), use Potekhin when ``microfield_model`` is omitted;
        a missing ``Ti_ev`` warns and assumes ``Ti_ev=Te_ev``. If False, use
        Holtsmark. Ignored when an explicit model is given.
    microfield_model : {None, 'holtsmark', 'hooper', 'potekhin', 'custom'}, optional
        Explicit microfield-distribution selector, forwarded to
        :func:`~starkzee.microfield.microfield_quadrature` (see there for the
        full description of each model). With ``None``, screened calls select
        Potekhin and unscreened calls select Holtsmark. Explicit ``'potekhin'``
        requires ``Ti_ev``; the Hooper-like ansatz requires explicit
        ``'hooper'`` selection and is not a validated distribution.
    quadratic_zeeman : bool, optional
        Include diamagnetic (quadratic) Zeeman term (default True).
    fine_structure : bool, optional
        Include mass-velocity and Darwin corrections (Dirac fine structure)
        so that 2s_{1/2} = 2p_{1/2} (default True).
    frequency_dependent_width : bool, optional
        When True (default), evaluate the GBK electron-impact width **pointwise
        at the observation detuning from the gross-structure line center**,
        ``w(E − E0)``, as in the PPPB operator Φ(Δω) of Ferri et al. (2022)
        Eq. (19), where Δω is "the frequency detuning from the line center".
        Every component then shares the same frequency-dependent width
        function: a far-shifted σ± component receives a reduced width at its
        own position, and the far wings of *all* components relax toward the
        strong-collision floor C_n as G(Δω) → 0 (breakdown of the impact
        approximation), making the components non-Lorentzian.
        When False, use the single on-resonance value ``w(0)`` everywhere
        (faster; valid when the profile extent ≪ ω_c).  Note the pointwise
        profile is not exactly area-normalized per component (the physical
        GBK non-Lorentzian shape isn't either).

        The requested width is always honored. A frequency-dependent kernel
        is not generally a unit-area Lorentzian and needs grid-convergence checks.
    force_frequency_dependent_width : bool, optional
        Compatibility no-op: frequency dependence is now honored unconditionally.
    custom_table_path : str, optional
        Two-column beta/density table, forwarded without silent fallback.
    charged : bool or None, optional
        Potekhin charged-point selector; ``None`` derives it from
        ``emitter_charge``.
    emitter_charge : float or None, optional
        Net radiator charge in units of e. ``None`` infers ``Z-1`` for a
        hydrogen-like radiator. Current analytic microfield fits distinguish
        neutral from charged points but do not use the charge magnitude.
    Z_bar : float, optional
        Background ion charge, distinct from radiator nuclear charge (default 1).
    apply_doppler : bool, optional
        Set False to supply Ti_ev for microfield coupling without Doppler.
    Ti_ev : float, optional
        Ion temperature [eV].  When supplied, Doppler broadening is folded into
        the Lorentzian accumulation as a Voigt profile, eliminating the need for
        a separate post-processing convolution.  Default is ``None`` (bare
        Lorentzian, no Doppler).
    species : str, optional
        Hydrogen isotope (``'H'``, ``'D'``, or ``'T'``); an alternative way to
        give Z = 1 and A. It must agree with ``Z`` and with ``A`` when both are given.
    A : int or float, optional
        Mass number of the emitter (default: from ``species``, else the most
        abundant isotope of Z). Sets the reduced-mass terms, the measured-level
        table and the Doppler mass (nucleus + electron).
    electron_model : str, optional
        Electron-impact width prescription, forwarded to
        :func:`~starkzee.broadening.electron_impact_width_model`.  ``'pppb'``
        (default) is the PPPB model (B-dependent ω_c cutoff); ``'zest'``,
        ``'zest-lee'``, ``'zest-dufty'`` select the ZEST model with the GBK,
        Lee, or Dufty RPA G-function. ``'none'`` switches electron broadening off
        (ion quasi-static + natural + Doppler only; any radiator), e.g. to
        separate the ion and electron contributions; Lorentzian widths are then
        floored at the grid spacing so that unbroadened components keep their area.
    electron_operator : bool, optional
        When ``True``, use the electron-impact **operator** diagonal: each
        Stark-Zeeman dressed state gets a width scaled by its own ⟨k|r²|k⟩
        (:func:`~starkzee.broadening.electron_impact_r2_scaling`) instead of the
        shell-averaged scalar (default ``False``).  This is the **ZEST operator**
        treatment with the off-diagonal / ``c_k`` set to zero.  The lower-manifold
        ``d†·d`` contribution **is** included: each transition width is
        ``W_e(n_u) + W_e(n_l)`` (Ferri et al. 2022 Eq. 8; ZEST
        ``w_init_eigen[i] + w_final_eigen[j]``), operator-diagonal per shell.
        Applied in every accumulation path, including constant-width Voigt
        kernels. The separate ``electron_interference`` option retains the
        full off-diagonal PPP impact-limit operator instead.
    electron_interference : bool, optional
        Opt-in full PPP impact-limit collision operator (default ``False``).
        When true, construct Appendix B Eq. (B1) of the 2024 PPP manual in the
        optical-coherence basis, retain its upper--lower interference term,
        diagonalize ``L_f - i Phi``, and evaluate the complex generalized SDT
        intensities ``a_k + i c_k``. This implies the operator treatment and
        requires ``frequency_dependent_width=False``: the manual's default
        implementation evaluates ``G(0)`` before the non-Hermitian solve. The
        Appendix-B coefficient is evaluated directly, so the ``pppb`` and
        ``pppb-intra`` scalar radius selectors produce the same full operator.
    natural_width_mode : {'state_resolved', 'shell_average'}, optional
        Natural-damping treatment. ``'state_resolved'`` (default) rotates the
        uncoupled E1 spontaneous-decay rates into each Stark–Zeeman eigenbasis
        and applies ``ħ(Γ_u,i + Γ_l,j)/2`` to each transition. The compatibility
        mode ``'shell_average'`` applies the historical common width obtained
        by averaging each shell over all ``2n²`` substates.
    use_empirical_data : bool, optional
        Use NIST empirical level energies for the field-free Hamiltonian
        instead of the analytic Dirac (spin-orbit + MV + Darwin) formula
        (default True); forwarded to :func:`~starkzee.radiator.build_hamiltonian`.
        The empirical levels include the Lamb shift, so the resulting profile
        centroid matches the measured NIST wavelength more closely than the
        analytical model, which is Lamb-shift-free. Set this to False for
        analytical energies or hydrogen-like ions with Z > 1. ``atomic_levels.json``
        tabulates H (n ≤ 8), D (n ≤ 6), and T (n ≤ 3) levels; higher levels use the
        Dirac energy below the ionization limit (with a warning). The table follows
        the isotope (``A`` / ``species``).
    atom : str, optional
        Atom identifier passed to :func:`~starkzee.atomic_data.load_levels`
        when ``use_empirical_data=True`` (default: the isotope given by ``A``).
    radiator : object, optional
        Radiator supplying the manifolds, dipoles, line energies and electron
        broadening (see :mod:`starkzee.radiators`). ``None`` (default) builds
        :class:`~starkzee.radiators.HydrogenicTransition` from ``n_u``, ``n_l``,
        ``Z`` and the atomic-structure options above, which are then ignored.
    method : {'exact', 'fft'}, optional
        ``'exact'`` (default) evaluates every Lorentzian (or Voigt) component on
        the grid, a cost of grid x components per quadrature point. ``'fft'``
        gathers the components of all quadrature points and sums them with FFT
        convolutions (:mod:`starkzee.lorentzian_fft`): its cost is nearly
        independent of the number of components, so it pays off for lines with
        many components (high n, multi-electron radiators) or fine quadrature.
        It is approximate, with the error controlled by ``fft_tol``; frequency-
        dependent widths are interpolated between Chebyshev nodes in the width.
        Not available with ``electron_interference``. Nonuniform grids are
        resampled to a uniform one.
    fft_tol : float, optional
        Target error of ``method='fft'``, relative to the largest single-
        component peak (default 1e-8; measured errors are 10-100x smaller).
    _quadrature, _point_cache : internal
        Used by :class:`~starkzee.profile_table.StaticProfileTable`: the
        microfield nodes and weights ``(fields, weights)`` replacing the
        quadrature, and a dict caching the Ne-independent data of each
        (field, angle) point across calls.

    Notes on approximations
    -----------------------
    All outputs are spectral densities per eV. Nonuniform monotonic input
    grids are supported: when FFT Doppler convolution is needed, evaluate on
    an increasing uniform energy grid at least as fine as the smallest input
    spacing, then interpolate back. Vary spacing and window independently to
    check interpolation, truncation and narrow-component errors.

    Constant electron widths with Doppler use direct Voigt kernels (including
    state-dependent operator widths). Frequency-dependent kernels are sampled
    first and Gaussian-convolved afterward; unresolved Lorentzians still need
    refinement. Finite output windows need not conserve the total line area.
    State-resolved natural widths include all E1 decays to lower principal
    shells. Fine-structure corrections to those radiative rates are neglected.

    Returns
    -------
    profile_pi : ndarray
        π (Δm = m_u − m_l = 0) polarization component.
    profile_sig_plus : ndarray
        σ+ (Δm = +1, blue-shifted at B > 0; dipole channel q = m_l − m_u = −1)
        polarization component.
    profile_sig_minus : ndarray
        σ− (Δm = −1, red-shifted; channel q = +1) polarization component.

    Notes
    -----
    Observable intensity at angle θ to B:

        I(θ) = I_π sin²θ + ½(I_σ+ + I_σ−)(1 + cos²θ)

    Transverse (θ = 90°): I_π + ½(I_σ+ + I_σ−).
    Along B (θ = 0°):     I_σ+ + I_σ−.
    Angle-averaged:       ⅔ (I_π + I_σ+ + I_σ−)
    (both sin²θ and ½(1+cos²θ) average to ⅔ over the sphere; for the isotropic
    case I_π = I_σ± = I this gives I(θ) = 2I at every angle, as it must).
    """
    energies_ev, restore_grid = uniform_energy_grid(
        energies_ev, resample=(apply_doppler and Ti_ev is not None and Ti_ev > 0
                               and frequency_dependent_width) or method == 'fft')
    if method not in ('exact', 'fft'):
        raise ValueError("method must be 'exact' or 'fft'.")
    if method == 'fft':
        # the FFT sums place node n at grid[0] + n * span/(n-1): evaluate on that exact
        # linspace and interpolate back (a no-op for numpy.linspace grids)
        _lin = np.linspace(energies_ev[0], energies_ev[-1], len(energies_ev))
        if not np.array_equal(_lin, energies_ev):
            _work, _restore = energies_ev, restore_grid
            restore_grid = lambda values: _restore(np.interp(_work, _lin, values))
            energies_ev = _lin
    if method == 'fft' and electron_interference:
        raise ValueError("method='fft' does not support electron_interference=True.")
    no_electrons = electron_model.lower() == 'none'
    if no_electrons:
        if electron_interference:
            raise ValueError("electron_interference=True needs an electron model (got 'none').")
        electron_operator = False
    if electron_interference:
        if frequency_dependent_width:
            raise ValueError(
                "electron_interference=True implements the PPP impact-limit "
                "operator and requires frequency_dependent_width=False."
            )
        if electron_model.lower() not in ('pppb', 'ferri', 'pppb-intra', 'ferri-intra'):
            raise ValueError(
                "electron_interference=True currently supports only the PPPB/Ferri "
                "electron models."
            )
    if radiator is None:
        A = _isotope_from_species(Z, A, species)
        radiator = HydrogenicTransition(n_u, n_l, Z, B, quadratic_zeeman, fine_structure, A,
                                        use_empirical_data=use_empirical_data, atom=atom)
    if emitter_charge is None:
        emitter_charge = radiator.default_emitter_charge
    if not np.isfinite(emitter_charge) or emitter_charge < 0:
        raise ValueError("emitter_charge must be finite and nonnegative.")
    if charged is None:
        charged = emitter_charge != 0
    if _point_cache is not None and electron_interference:
        raise ValueError("cached quadrature points do not support electron_interference=True.")
    # 1. Get microfield grid and weights
    if _quadrature is not None:
        fields, f_weights = (np.asarray(x, dtype=float) for x in _quadrature)
    else:
        fields, f_weights = microfield_quadrature(Ne_m3, Te_ev, num_points=num_f,
                                                  max_beta=max_beta, use_screening=use_screening,
                                                  microfield_model=microfield_model, Ti_ev=Ti_ev,
                                                  custom_table_path=custom_table_path,
                                                  charged=charged, emitter_charge=emitter_charge,
                                                  Z_bar=Z_bar,
                                                  species_charges=species_charges,
                                                  species_concentrations=species_concentrations)
    
    # 2. Get angular integration points (Gauss-Legendre on mu = cos(theta) from 0 to 1)
    mu_points, mu_weights = np.polynomial.legendre.leggauss(num_mu)
    mu_points = 0.5 * (mu_points + 1.0)
    mu_weights = 0.5 * mu_weights
    # At B = 0 only the field direction breaks the isotropy: solve F along z and
    # average the polarizations over orientations afterwards (exact; the angle
    # quadrature gives the same result to rounding at num_mu >= 2).
    isotropic = B == 0
    if isotropic:
        mu_points, mu_weights = np.array([1.0]), np.array([1.0])
    # With F along z at B = 0 the dressed states can be chosen with pure M, and the
    # full PPP generator splits into Delta M blocks (only Delta M = 0, +-1 radiate).
    m_u_basis = getattr(radiator.upper, "m_values", None)
    m_l_basis = getattr(radiator.lower, "m_values", None)
    m_blocks = (isotropic and electron_interference
                and m_u_basis is not None and m_l_basis is not None)
    if m_blocks:
        for mats, m in (((radiator.upper.H0, radiator.upper.M_z), m_u_basis),
                        ((radiator.lower.H0, radiator.lower.M_z), m_l_basis)):
            off = np.not_equal.outer(np.asarray(m, float), np.asarray(m, float))
            # rounding-level off-M elements (e.g. from basis transforms) are dropped
            if any(np.any(np.abs(np.asarray(a)[off]) > 1e-9 * max(np.max(np.abs(a)), 1e-300)) for a in mats):
                m_blocks = False            # radiator couples different M: use the full generator
    
    # Field-independent matrices of the radiator: H0 (shell energy removed from the
    # diagonal), Stark templates (V_E = Fz*M_z + Fx*M_x), dipoles between manifolds.
    D_q_uncoupled = radiator.dipoles
    H_atom_u, H_atom_l = radiator.upper.H0, radiator.lower.H0
    M_z_u, M_x_u = radiator.upper.M_z, radiator.upper.M_x
    M_z_l, M_x_l = radiator.lower.M_z, radiator.lower.M_x

    # Output arrays
    profile_pi = np.zeros_like(energies_ev)
    profile_sig_plus = np.zeros_like(energies_ev)
    profile_sig_minus = np.zeros_like(energies_ev)
    
    sigma_D = None
    if apply_doppler and Ti_ev is not None and Ti_ev > 0:
        # every radiator carries the emitter mass [u] (nucleus + bound electrons)
        if getattr(radiator, "mass_amu", None) is None:
            raise ValueError("Doppler broadening needs the emitter mass: give the radiator mass_amu "
                             "(or pass apply_doppler=False).")
        mc2_ev = radiator.mass_amu * _ATOMIC_MASS * _C_LIGHT**2 / _E_CHARGE
        # Use the physical (analytic) gross-structure line center, not mean(energies_ev):
        # the latter depends on the requested observation window (its offset and width),
        # not on the emitter, so an asymmetric or shifted window would silently change the
        # modeled Doppler width without changing the physics.
        _E0_center_ev = radiator.E0_physical
        sigma_D = _E0_center_ev * np.sqrt(Ti_ev / mc2_ev)

    # Direct Voigt evaluation preserves constant per-transition widths without
    # sampling narrow Lorentzians or substituting a different physical model.
    # force_frequency_dependent_width is retained as a compatibility no-op.
    _dx = abs(energies_ev[1] - energies_ev[0])
    _voigt_loop = (sigma_D is not None and not frequency_dependent_width
                   and not electron_interference)

    if natural_width_mode not in ('state_resolved', 'shell_average'):
        raise ValueError(
            "natural_width_mode must be 'state_resolved' or 'shell_average'.")
    if natural_width_mode == 'state_resolved':
        natural_rates_u_basis = radiator.upper.decay_rates
        natural_rates_l_basis = radiator.lower.decay_rates
        # Conservative scalar used only by the undersampling diagnostic below.
        min_natural_ev = (_HBAR / (2.0 * _E_CHARGE)
                          * (natural_rates_u_basis.min()
                             + natural_rates_l_basis.min()))
    else:
        shell_natural_ev = radiator.shell_natural_width_ev()
        min_natural_ev = shell_natural_ev

    # Numerical reconstruction offset for transition energies: E_ref,u - E_ref,l of
    # the radiator (the diagonal shift removed from H0), so dE_shifted*unit_to_ev +
    # E0_line is the absolute transition energy. E0_physical is the fixed line
    # reference used for damping, independent of the conditioning shift.
    E0_line = radiator.E0_line
    _shift_to_ev = radiator.unit_to_ev
    E0_physical = radiator.E0_physical

    # On-resonance width — used for the FFT Lorentzian step (both paths when
    # Doppler is active) and as the scalar w for frequency_dependent_width=False.
    # Keep the upper- and lower-shell electron parts separate so the operator
    # path can rescale each per Stark-Zeeman dressed state by its own ⟨r²⟩.
    #
    # Upper + lower level contributions: the electron-impact width of a line is
    # φ(n_u) + φ(n_l) — Ferri, Peyrusse & Calisti (2022) Eq. (8) / ZEST, which
    # keep both the upper- and lower-state d·d† terms and drop only the
    # upper–lower *interference* term. The opt-in full PPP path below uses the
    # common impact coefficient W0[C_nu + G_nu(0)] for all three Appendix-B
    # terms instead of rescaling either scalar shell width.
    # n_l = 1 (Lyman) has no intra-shell dipole channel, so its impact width is
    # taken as 0 in the default diagonal approximation.
    def _electron_widths(detuning):
        if no_electrons:
            zero = np.zeros(np.shape(detuning)) if np.ndim(detuning) else 0.0
            return zero, zero
        return radiator.electron_widths(detuning, Ne_m3, Te_ev, B, electron_model)

    w_resonance_e_u, w_resonance_e_l = _electron_widths(0.0)
    w_resonance_e = w_resonance_e_u + w_resonance_e_l
    w_resonance = w_resonance_e + min_natural_ev
    ppp_impact_coefficient = None
    if electron_interference:
        ppp_impact_coefficient = radiator.ppp_impact_coefficient(Ne_m3, Te_ev, B)
        # Use the selected-shell mean self-term scale for grid-resolution
        # diagnostics. The scalar electron_model selector does not calibrate
        # the full operator and therefore must not alter this warning.
        w_resonance = (ppp_impact_coefficient * radiator.ppp_self_term_r2()
                       + min_natural_ev)

    # Pointwise electron width w_e(E − E0) on the observation grid (PPPB Φ(Δω),
    # Ferri et al. 2022 Eq. 19: Δω is the detuning from line center).  Computed
    # once per profile call; shared by every quadrature point and transition.
    # Upper and lower shells kept separate (see the on-resonance block above).
    if frequency_dependent_width:
        _w_e_grid_u, _w_e_grid_l = _electron_widths(energies_ev - E0_physical)
        _w_e_grid_u = _w_e_grid_u[:, np.newaxis]
        if np.ndim(_w_e_grid_l):
            _w_e_grid_l = _w_e_grid_l[:, np.newaxis]

    # Undersampled-Lorentzian guard: in the Lorentzian-accumulation path,
    # components narrower than the grid spacing lose integrated intensity
    # (trapezoid mass simply falls between grid points).
    if not _voigt_loop and w_resonance < 2.0 * _dx:
        warnings.warn(
            f"Lorentzian half-width at line center ({w_resonance:.3e} eV) is below "
            f"twice the grid spacing (dx = {_dx:.3e} eV); part of the integrated "
            "line intensity can be lost to undersampling. Refine the energy grid; "
            "post-convolution Doppler cannot recover unresolved intrinsic components.",
            UserWarning, stacklevel=2)

    # Quadrature points (field magnitude x angle) with non-negligible weight
    points = [(f_weight * mu_weight, fi * mu, fi * np.sqrt(1.0 - mu**2))
              for fi, f_weight in zip(fields, f_weights) if f_weight > 1e-15
              for mu, mu_weight in zip(mu_points, mu_weights)
              if f_weight * mu_weight > 1e-15]
    # Ne-independent data of each quadrature point (active transitions), optionally
    # cached across calls by StaticProfileTable under the key (Fz, Fx)
    cache = _point_cache
    todo = [(Fz, Fx) for _, Fz, Fx in points if cache is None or (Fz, Fx) not in cache]
    if not m_blocks and todo:
        # Diagonalize the quadrature points in stacked eigh calls, chunk by chunk
        Fz_all, Fx_all = (np.array(col) for col in zip(*todo))
        eig_u = _batched_eigh(H_atom_u, M_z_u, M_x_u, Fz_all, Fx_all)
        eig_l = _batched_eigh(H_atom_l, M_z_l, M_x_l, Fz_all, Fx_all)

    def _point_record(sz_energies_u, sz_vectors_u, sz_energies_l, sz_vectors_l):
        """Active transitions of one point: (centres, intensities (P, 3), natural widths
        or None, upper scales or None, lower scales or None); () when none radiate."""
        natural_widths = None
        if natural_width_mode == 'state_resolved':
            natural_rates_u = (np.abs(sz_vectors_u)**2).T @ natural_rates_u_basis
            natural_rates_l = (np.abs(sz_vectors_l)**2).T @ natural_rates_l_basis
            natural_widths = (_HBAR / (2.0 * _E_CHARGE)
                              * (natural_rates_l[:, np.newaxis]
                                 + natural_rates_u[np.newaxis, :]))
        V_l_adj = sz_vectors_l.conj().T
        # Transition energies from the shifted eigenvalues plus the gross-structure line
        # centre (no cancellation between large level energies); _shift_to_ev converts
        # cm-1 -> eV in the empirical branch (1.0 otherwise).
        dE = (sz_energies_u[np.newaxis, :] - sz_energies_l[:, np.newaxis]) * _shift_to_ev + E0_line
        I_pi = np.abs(V_l_adj @ D_q_uncoupled[ 0] @ sz_vectors_u)**2
        I_sp = np.abs(V_l_adj @ D_q_uncoupled[-1] @ sz_vectors_u)**2
        I_sm = np.abs(V_l_adj @ D_q_uncoupled[ 1] @ sz_vectors_u)**2
        # Union of active transitions: one kernel for all q.
        mask = (I_pi > 1e-12) | (I_sp > 1e-12) | (I_sm > 1e-12)
        if not np.any(mask):
            return ()
        scale_u = scale_l = None
        if electron_operator:
            # Electron-impact operator diagonal: per-dressed-state <k|r^2|k>/<r^2>_avg
            # (c_k = 0 -> ZEST operator) on the upper AND lower shell (ZEST:
            # w_init_eigen[i] + w_final_eigen[j]). Historical PPPB/Ferri uses full
            # closure, PPPB-intra and every ZEST selector the projected intra-shell sum.
            r2_form = ('full' if electron_model.lower() in ('pppb', 'ferri')
                       else 'intra')
            r2_scale_u, r2_scale_l = radiator.electron_r2_scaling(
                sz_vectors_u, sz_vectors_l, r2_form)
            # r2_scale_u per upper eigenstate (columns of dE), r2_scale_l per lower
            # (rows); broadcast to dE.shape, then select the active transitions.
            scale_u = np.broadcast_to(r2_scale_u[np.newaxis, :], dE.shape)[mask]
            if r2_scale_l is not None:
                scale_l = np.broadcast_to(r2_scale_l[:, np.newaxis], dE.shape)[mask]
        return (dE[mask], np.column_stack((I_pi[mask], I_sp[mask], I_sm[mask])),
                None if natural_widths is None else natural_widths[mask], scale_u, scale_l)

    # method='fft': components gathered over all quadrature points, summed after the loop
    sticks = {"centre": [], "amplitude": [], "width": [], "natural": [],
              "scale_u": [], "scale_l": []}
    if frequency_dependent_width:
        # Pointwise w(E - E0): column vector over the observation grid, shared by all
        # transitions (PPPB Phi(Delta omega) convention).
        w_e_u, w_e_l = _w_e_grid_u, _w_e_grid_l
    else:
        w_e_u, w_e_l = w_resonance_e_u, w_resonance_e_l
    # no electrons: keep (nearly) unbroadened components on the grid, floor at one grid step
    floor = _dx if no_electrons else None

    # Main integration loop
    for weight, Fz, Fx in points:
        record = cache.get((Fz, Fx)) if cache is not None else None
        if record is None:
            # Diagonalize using precomputed H_atom and Stark templates
            if m_blocks:
                sz_energies_u, sz_vectors_u, m_dressed_u = _eigh_by_m(H_atom_u + Fz * M_z_u, m_u_basis)
                sz_energies_l, sz_vectors_l, m_dressed_l = _eigh_by_m(H_atom_l + Fz * M_z_l, m_l_basis)
            else:
                sz_energies_u, sz_vectors_u = next(eig_u)
                sz_energies_l, sz_vectors_l = next(eig_l)

            if electron_interference:
                # Opt-in full PPP operator: complex generalized SDTs (supersedes the
                # diagonal operator approximation).
                if natural_width_mode == 'state_resolved':
                    natural_rates_u = (np.abs(sz_vectors_u)**2).T @ natural_rates_u_basis
                    natural_rates_l = (np.abs(sz_vectors_l)**2).T @ natural_rates_l_basis
                    natural_for_operator = (_HBAR / (2.0 * _E_CHARGE)
                                            * (natural_rates_l[:, np.newaxis]
                                               + natural_rates_u[np.newaxis, :]))
                else:
                    natural_for_operator = shell_natural_ev
                dE = ((sz_energies_u[np.newaxis, :] - sz_energies_l[:, np.newaxis])
                      * _shift_to_ev + E0_line)
                frequencies, widths, complex_strengths = radiator.ppp_sdts(
                    dE, sz_vectors_u, sz_vectors_l, ppp_impact_coefficient,
                    natural_for_operator,
                    **({"m_values": (m_dressed_u, m_dressed_l)} if m_blocks else {}))
                components = generalized_lorentzian_components(
                    energies_ev, frequencies, widths, complex_strengths)
                profile_pi += weight * components[0]
                profile_sig_plus += weight * components[-1]
                profile_sig_minus += weight * components[1]
                continue

            record = _point_record(sz_energies_u, sz_vectors_u, sz_energies_l, sz_vectors_l)
            if cache is not None:
                cache[(Fz, Fx)] = record
        if not record:
            continue
        act_dE, intensities, natural_act, scale_u, scale_l = record
        natural_width = natural_act if natural_act is not None else shell_natural_ev

        if method == 'fft':
            n_act = len(act_dE)
            sticks["centre"].append(act_dE)
            sticks["amplitude"].append(weight * intensities)
            if frequency_dependent_width:
                sticks["natural"].append(np.broadcast_to(natural_width, (n_act,)))
                if scale_u is not None:
                    sticks["scale_u"].append(scale_u)
                if scale_l is not None:
                    sticks["scale_l"].append(scale_l)
            else:
                sticks["width"].append(np.broadcast_to(_transition_widths(
                    w_e_u, w_e_l, natural_width, scale_u, scale_l, floor), (n_act,)))
            continue
        if _voigt_loop:
            w = _transition_widths(w_e_u, w_e_l, natural_width, scale_u, scale_l, floor)
            kernel = voigt_profile(energies_ev[:, np.newaxis] - act_dE[np.newaxis, :],
                                   sigma_D, w)
        else:
            kernel = _lorentzian_kernel(energies_ev, act_dE, w_e_u, w_e_l, natural_width,
                                        scale_u, scale_l, floor)

        # one matrix product for the three polarizations
        acc = kernel @ intensities
        acc *= weight
        profile_pi        += acc[:, 0]
        profile_sig_plus  += acc[:, 1]
        profile_sig_minus += acc[:, 2]

    if method == 'fft' and sticks["centre"]:
        from starkzee.lorentzian_fft import frequency_dependent_sum, lorentzian_sum
        cat = {k: (np.concatenate(v) if v else None) for k, v in sticks.items()}
        if frequency_dependent_width:
            sums = frequency_dependent_sum(
                energies_ev, cat["centre"], cat["amplitude"], cat["natural"],
                _w_e_grid_u[:, 0], (_w_e_grid_l[:, 0] if np.ndim(_w_e_grid_l) else _w_e_grid_l),
                cat["scale_u"], cat["scale_l"], _dx if no_electrons else None, fft_tol)
        else:
            sums = lorentzian_sum(energies_ev, cat["centre"], cat["amplitude"], cat["width"],
                                  sigma_D if _voigt_loop else None, fft_tol)
        profile_pi += sums[:, 0]
        profile_sig_plus += sums[:, 1]
        profile_sig_minus += sums[:, 2]

    if isotropic:
        # orientation average of dipole emission: every lab polarization gets 1/3
        # of the field-frame total
        average = (profile_pi + profile_sig_plus + profile_sig_minus) / 3.0
        for prof in (profile_pi, profile_sig_plus, profile_sig_minus):
            prof[:] = average

    # Post-loop FFT to complete the Voigt when Doppler is active.
    # Zero-pad to 2N so the circular convolution approximates a linear one,
    # eliminating the periodic wrap-around that otherwise creates a DC floor in
    # the far wings (the long Lorentzian tail from one side of the grid
    # folds back onto the other in a naive N-point circular FFT).
    if sigma_D is not None and not _voigt_loop:
        N = len(energies_ev)
        N_pad = 2 * N
        k = np.fft.rfftfreq(N_pad, d=_dx)
        fft_filter = np.exp(-2.0 * np.pi**2 * sigma_D**2 * k**2)
        _padded = np.zeros(N_pad)
        for prof in (profile_pi, profile_sig_plus, profile_sig_minus):
            _padded[:N] = prof
            _padded[N:] = 0.0
            conv = np.fft.irfft(np.fft.rfft(_padded) * fft_filter, n=N_pad)
            prof[:] = conv[:N]

    return tuple(restore_grid(p) for p in (profile_pi, profile_sig_plus, profile_sig_minus))


def discrete_transitions(n_u, n_l, Z, B, Fz=0.0, Fx=0.0,
                         quadratic_zeeman=True, fine_structure=True,
                         min_strength=0.0, A=None, radial_method=None,
                         use_empirical_data=True, atom=None):
    """Return all discrete Stark-Zeeman dipole transitions at a single field configuration.

    Diagonalizes the Stark-Zeeman Hamiltonian for both shells and enumerates every
    (upper eigenstate i, lower eigenstate j, polarization q) triplet with
    non-zero dipole matrix element squared.

    Parameters
    ----------
    n_u, n_l : int
        Upper and lower principal quantum numbers.
    Z : int
        Nuclear charge.
    B : float
        Magnetic field [T].
    Fz : float, optional
        Electric field component along B [V m⁻¹] (default 0).
    Fx : float, optional
        Electric field component perpendicular to B [V m⁻¹] (default 0).
    quadratic_zeeman : bool, optional
        Include diamagnetic Zeeman term (default True).
    fine_structure : bool, optional
        Include mass-velocity + Darwin corrections (default True).
    min_strength : float, optional
        Discard transitions with dipole strength abs(d_q)² < min_strength [a₀²] (default 0).
    A : int, optional
        Atomic mass number of the emitter (1 = H, 2 = D, 3 = T).  Sets the
        reduced-mass Rydberg used for the absolute level energies (default 1).
    radial_method : {"gordon", "quad", None}, optional
        Backend for the radial dipole elements.  ``None`` (default) uses the
        module-level ``RADIAL_DIPOLE_METHOD``
        (``"gordon"`` exact closed form by default; ``"quad"`` is the numerical
        fallback).  The two agree to ~1e-15.
    use_empirical_data : bool, optional
        Use NIST empirical level energies (default True); forwarded to
        :func:`solve_starkzee`.
    atom : str, optional
        Atom identifier for empirical data (default ``"H"``); forwarded to
        :func:`solve_starkzee`.

    Returns
    -------
    dict with five equal-length arrays sorted by transition energy:

    ``energy_ev``
        Transition energy E_upper_i − E_lower_j [eV].
    ``q``
        Polarization integer q = m_l − m_u: 0 = π; **q = −1 is σ+**
        (Δm = m_u − m_l = +1, blue-shifted at B > 0); **q = +1 is σ−**
        (red-shifted).
    ``strength``
        abs(d_q(i→j))² [a₀²].  Summed over all transitions equals
        :func:`~starkzee.radiator.line_strength` (unitary invariance).
    ``upper_idx``
        Upper eigenstate index (0 … 2n_u²−1).
    ``lower_idx``
        Lower eigenstate index (0 … 2n_l²−1).
    """
    evals_u, evecs_u = solve_starkzee(
        n_u, Z, B, Fz, Fx, quadratic_zeeman, fine_structure, A,
        use_empirical_data=use_empirical_data, atom=atom)
    evals_l, evecs_l = solve_starkzee(
        n_l, Z, B, Fz, Fx, quadratic_zeeman, fine_structure, A,
        use_empirical_data=use_empirical_data, atom=atom)

    from starkzee.utils import reduced_mass_factor
    a_mu = 1.0 / reduced_mass_factor(Z, A)                     # dipoles in the reduced Bohr radius
    D_q = {q: D * a_mu for q, D in _uncoupled_dipole_matrices(n_u, n_l, Z, method=radial_method).items()}
    dim_l, dim_u = D_q[0].shape

    # Transition energy E_upper_i - E_lower_j is shared across polarizations.
    dE = evals_u[np.newaxis, :] - evals_l[:, np.newaxis]   # (dim_l, dim_u)

    energies, q_vals, strengths, up_idx, lo_idx = [], [], [], [], []
    for q in (0, -1, 1):
        mixed = evecs_l.conj().T @ D_q[q] @ evecs_u   # (dim_l, dim_u)
        strength = np.abs(mixed) ** 2
        j_idx, i_idx = np.nonzero(strength > min_strength)
        if j_idx.size == 0:
            continue
        energies.append(dE[j_idx, i_idx])
        strengths.append(strength[j_idx, i_idx])
        q_vals.append(np.full(i_idx.shape, q, dtype=int))
        up_idx.append(i_idx)
        lo_idx.append(j_idx)

    if energies:
        energies = np.concatenate(energies)
        strengths = np.concatenate(strengths)
        q_vals = np.concatenate(q_vals)
        up_idx = np.concatenate(up_idx)
        lo_idx = np.concatenate(lo_idx)
    else:
        energies = np.array([])
        strengths = np.array([])
        q_vals = np.array([], dtype=int)
        up_idx = np.array([], dtype=int)
        lo_idx = np.array([], dtype=int)

    order = np.argsort(energies)
    return {
        'energy_ev':  energies[order],
        'q':          q_vals[order].astype(int),
        'strength':   strengths[order],
        'upper_idx':  up_idx[order].astype(int),
        'lower_idx':  lo_idx[order].astype(int),
    }

