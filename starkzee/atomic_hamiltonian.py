# atomic_hamiltonian.py — Hydrogenic basis, Hamiltonian construction, and diagonalization in magnetic fields

import numpy as np
import math
from dataclasses import dataclass
from functools import lru_cache
from scipy.special import assoc_laguerre
from scipy.integrate import quad
from scipy.constants import hbar as HBAR, m_e as M_E, e as E_CHARGE, fine_structure as FINE_STRUCTURE
from starkzee.utils import A0, RYDBERG_EV, BOHR_MAGNETON_EV_T


@dataclass(frozen=True)
class AtomicState:
    """Hydrogenic basis state |n, l, m_l, s=½, m_s⟩."""
    index: int
    n:     int
    l:     int
    ml:    int
    s:     float = 0.5
    ms:    float = 0.5

    def __repr__(self):
        return f"|n={self.n}, l={self.l}, ml={self.ml}, ms={self.ms:.1f}>"

def build_basis(n):
    """Return the ordered list of 2n² hydrogenic basis states for shell n.

    Each state is an :class:`AtomicState` labelled by (n, l, m_l, s=½, m_s).
    The ordering is: outer loop over l ∈ {0, …, n−1}, then m_l ∈ {−l, …, +l},
    then m_s ∈ {+½, −½}.  This ordering is fixed throughout the package; all
    Hamiltonian and dipole matrices use the same index convention.

    Parameters
    ----------
    n : int
        Principal quantum number.

    Returns
    -------
    list of AtomicState
        2n² basis states in the canonical ordering described above.
    """
    basis = []
    idx = 0
    for l in range(n):
        for ml in range(-l, l + 1):
            for ms in [0.5, -0.5]:
                basis.append(AtomicState(idx, n, l, ml, 0.5, ms))
                idx += 1
    return basis

def radial_wavefunction(r, n, l, Z):
    """Return the hydrogenic radial wavefunction R_{nl}(r) [a₀^{−3/2}].

    Evaluates the normalised hydrogenic radial wavefunction using the
    associated Laguerre polynomial representation:

        R_{nl}(r) = N_{nl} × exp(−Z r / n) × (2Z r/n)^l × L_{n-l-1}^{2l+1}(2Z r/n)

    where r is in units of a₀ and N_{nl} is chosen so that
    ∫₀^∞ |R_{nl}|² r² dr = 1.

    Parameters
    ----------
    r : float or array-like
        Radial coordinate in Bohr radii (a₀).
    n : int
        Principal quantum number.
    l : int
        Orbital angular momentum quantum number (0 ≤ l < n).
    Z : int
        Nuclear charge.

    Returns
    -------
    float or ndarray
        Radial wavefunction value [a₀^{−3/2}].  Returns 0 if l ≥ n.

    Notes
    -----
    Uses ``scipy.special.assoc_laguerre`` for the polynomial, which handles
    the high-n case accurately.
    """
    if l >= n:
        return 0.0
    num = math.factorial(n - l - 1)
    den = 2 * n * math.factorial(n + l)
    prefactor = np.sqrt((2.0 * Z / n)**3 * num / den)
    
    rho = 2.0 * Z * r / n
    lag = assoc_laguerre(rho, n - l - 1, 2 * l + 1)
    return prefactor * np.exp(-Z * r / n) * (rho**l) * lag

@lru_cache(maxsize=None)
def radial_r2_element(n, l1, l2, Z):
    """Return the radial matrix element ⟨n, l₁ | r² | n, l₂⟩ [a₀²].

    Used for off-diagonal elements (|l₁ − l₂| = 2) arising from the angular
    decomposition of r² sin²θ in the quadratic Zeeman term.  The diagonal
    (l₁ = l₂) elements are computed analytically inside
    :func:`build_hamiltonian` using the closed-form formula:

        ⟨r²⟩_{n,l} = (n²/2Z²) [5n² + 1 − 3l(l+1)]

    Off-diagonal elements are computed here by numerical integration:

        ⟨n, l₁ | r² | n, l₂⟩ = ∫₀^∞ R_{nl₁}(r) r² R_{nl₂}(r) r² dr

    Results are cached with :func:`functools.lru_cache` to avoid redundant
    integration during the microfield loop.

    Parameters
    ----------
    n : int
        Principal quantum number (same for both bra and ket).
    l1, l2 : int
        Orbital quantum numbers; the selection rule |l₁ − l₂| = 2 must hold.
    Z : int
        Nuclear charge.

    Returns
    -------
    float
        Off-diagonal radial matrix element [a₀²].
    """
    val, _ = quad(
        lambda r: radial_wavefunction(r, n, l1, Z) * radial_wavefunction(r, n, l2, Z) * r**4,
        0, 300, limit=300
    )
    return val

@lru_cache(maxsize=None)
def radial_dipole(n1, l1, n2, l2, Z):
    """Return the radial transition matrix element ⟨n₁, l₁ | r | n₂, l₂⟩ [a₀].

    Computes the radial part of the electric-dipole matrix element by numerical
    integration:

        ⟨n₁, l₁ | r | n₂, l₂⟩ = ∫₀^∞ R_{n₁l₁}(r) r R_{n₂l₂}(r) r² dr

    The selection rule |l₁ − l₂| = 1 is enforced; 0 is returned immediately
    for forbidden pairs.  Results are cached with :func:`functools.lru_cache`.

    Parameters
    ----------
    n1 : int
        Principal quantum number of the first (bra) state.
    l1 : int
        Orbital quantum number of the bra state.
    n2 : int
        Principal quantum number of the second (ket) state.
    l2 : int
        Orbital quantum number of the ket state.
    Z : int
        Nuclear charge (same for both states; the code is for hydrogen-like ions).

    Returns
    -------
    float
        Radial dipole matrix element [a₀].  Always ≥ 0 (the sign convention
        is handled by the angular part :func:`angular_dipole_element`).

    Notes
    -----
    The upper integration limit is set to 150 a₀, which comfortably contains
    the hydrogenic wavefunction for n ≤ 7 (the largest n used in practice).
    """
    if abs(l1 - l2) != 1:
        return 0.0
    val, _ = quad(lambda r: radial_wavefunction(r, n1, l1, Z) * radial_wavefunction(r, n2, l2, Z) * r**3, 0, 150)
    return val

def angular_dipole_element(l1, m1, l2, m2, q):
    """Return the angular part of the dipole matrix element ⟨l₁, m₁ | T_q^(1) | l₂, m₂⟩.

    The spherical tensor components T_q^(1) of the position unit vector r̂ are:

    - q = 0  (π):  T₀ = cos θ
    - q = +1 (σ+): T₊₁ = sin θ e^{iφ} / √2
    - q = −1 (σ−): T₋₁ = sin θ e^{−iφ} / √2

    The matrix elements are expressed via Clebsch-Gordan coefficients and are
    non-zero only when |l₁ − l₂| = 1 and m₁ − m₂ = q.  Explicit formulas:

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
        Angular matrix element (real).  Returns 0 if |l₁ − l₂| ≠ 1 or
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

def build_hamiltonian(n, Z, B, include_quadratic=True, include_fine_structure=True):
    """Build the (2n²) × (2n²) atomic Hamiltonian matrix in eV.

    Constructs the full magnetic Hamiltonian in the uncoupled |n, l, m_l, m_s⟩
    basis returned by :func:`build_basis`.  Four physical contributions are
    included:

    **1. Unperturbed energy** (diagonal, degenerate across the shell):

        E_n = −Z² Ry / n²

    **2. Spin-orbit coupling** L·S (off-diagonal in m_l, m_s):

        H_SO = ξ_{nl} L·S,   ξ_{nl} = Z⁴ α² Ry / (n³ l (l+½)(l+1))

    Written with ladder operators: L·S = L_z S_z + ½(L₊ S₋ + L₋ S₊).

    **3. Mass-velocity and Darwin corrections** (diagonal, only when
    ``include_fine_structure=True``):

        ΔE_{l=0} = −A (n − ¾)
        ΔE_{l>0} = −A (n/(l+½) − ¾),   A = Z⁴ α² Ry / n⁴

    Together with spin-orbit, these reproduce the Dirac fine-structure formula
    and restore the 2s_{1/2} = 2p_{1/2} degeneracy.

    **4. Linear Zeeman** (diagonal):

        H_LZ = μ_B B (m_l + g_s m_s),   g_s = 2.0023192

    **5. Quadratic (diamagnetic) Zeeman** (diagonal and off-diagonal in l,
    only when ``include_quadratic=True`` and B > 0):

        H_QZ = (e²B²/8m_e) r² sin²θ

    Matrix elements in the |n, l, m_l⟩ basis involve ⟨l, m_l | sin²θ | l', m_l⟩
    for |l − l'| ∈ {0, 2} and the radial elements ⟨n, l | r² | n, l'⟩.

    Parameters
    ----------
    n : int
        Principal quantum number.
    Z : int
        Nuclear charge.
    B : float
        Magnetic field [T].  B = 0 is fully supported (skips the QZ term).
    include_quadratic : bool, optional
        Include the diamagnetic quadratic Zeeman term H_QZ (default True).
        At B ≤ 10 T the QZ shift is typically < 0.1 meV for n ≤ 3 and can be
        omitted for speed.
    include_fine_structure : bool, optional
        Include mass-velocity and Darwin corrections alongside spin-orbit
        (default True).  When False, only H_SO is added, breaking the
        2s_{1/2} = 2p_{1/2} degeneracy — useful for isolated unit tests.

    Returns
    -------
    ndarray, shape (2n², 2n²), dtype complex
        Hermitian Hamiltonian matrix in eV.

    Notes
    -----
    The Stark electric-field perturbation is **not** included here; it is
    added separately in :func:`~starkzee.static_profile.build_stark_matrix`
    and combined in :func:`~starkzee.static_profile.solve_starkzee`.

    At B = 0 the pi/σ decomposition of the resulting profile is physically
    meaningless (no preferred axis); only the total profile is physically
    observable.
    """
    basis = build_basis(n)
    dim = len(basis)
    H = np.zeros((dim, dim), dtype=complex)

    # 1. Unperturbed energy (En = -Z^2 * Ry / n^2)
    En = - (Z**2) * RYDBERG_EV / (n**2)
    for i in range(dim):
        H[i, i] += En

    # 2. Spin-Orbit Coupling: xi * (L . S)
    # xi = Z^4 * alpha^2 * Ry / (n^3 * l * (l+1) * (l+1/2))
    for i, state_i in enumerate(basis):
        for j, state_j in enumerate(basis):
            if state_i.l == state_j.l and state_i.l > 0:
                l = state_i.l
                xi = (Z**4) * (FINE_STRUCTURE**2) * RYDBERG_EV / ((n**3) * l * (l + 1.0) * (l + 0.5))
                
                # L_z * S_z term
                if i == j:
                    H[i, j] += xi * state_i.ml * state_i.ms
                # L_+ * S_- term
                elif state_i.ml == state_j.ml + 1 and state_i.ms == state_j.ms - 1:
                    term_l = np.sqrt(l * (l + 1.0) - state_j.ml * (state_j.ml + 1))
                    term_s = 1.0 if state_j.ms == 0.5 else 0.0
                    H[i, j] += 0.5 * xi * term_l * term_s
                # L_- * S_+ term
                elif state_i.ml == state_j.ml - 1 and state_i.ms == state_j.ms + 1:
                    term_l = np.sqrt(l * (l + 1.0) - state_j.ml * (state_j.ml - 1))
                    term_s = 1.0 if state_j.ms == -0.5 else 0.0
                    H[i, j] += 0.5 * xi * term_l * term_s

    # 2b. Mass-velocity and Darwin corrections (completes Dirac fine structure).
    # These are diagonal in |n,l,ml,ms> and do not depend on ml or ms.
    # Together with SO they give exact Dirac eigenvalues, restoring the
    # 2s_{1/2} = 2p_{1/2} degeneracy that SO alone breaks.
    #
    # For l=0: ΔE = -A*(n - 3/4)        [MV + Darwin, Darwin is non-zero only at l=0]
    # For l>0: ΔE = -A*(n/(l+1/2) - 3/4) [MV only; Darwin vanishes for l>0]
    # where A = Z^4 * alpha^2 * Ry / n^4.
    if include_fine_structure:
        A = (Z**4) * (FINE_STRUCTURE**2) * RYDBERG_EV / (n**4)
        for i, state in enumerate(basis):
            l = state.l
            if l == 0:
                H[i, i] += -A * (n - 0.75)
            else:
                H[i, i] += -A * (n / (l + 0.5) - 0.75)

    # 3. Linear Zeeman Effect: mu_B * B * (L_z + g_s * S_z)
    g_s = 2.0023192
    for i, state in enumerate(basis):
        H[i, i] += BOHR_MAGNETON_EV_T * B * (state.ml + g_s * state.ms)

    # 4. Quadratic Zeeman Effect: (e^2 * B^2 / 8me) * r^2 * sin^2(theta)
    if include_quadratic and B > 0:
        quad_coeff_ev = (E_CHARGE * (B**2) * (A0**2)) / (8.0 * M_E)
        
        # We need the matrix elements of r^2 * sin^2(theta)
        # sin^2(theta) = 1 - cos^2(theta)
        # Radial matrix element <n, l1 | r^2 | n, l2> is only diagonal:
        # <r^2>_nl = (n^2 / 2Z^2) * [5n^2 + 1 - 3l(l+1)]
        # For off-diagonal in l, <r^2>_nl1_nl2 is 0. So r^2 is diagonal in l.
        for i, state_i in enumerate(basis):
            for j, state_j in enumerate(basis):
                if state_i.ms == state_j.ms and state_i.ml == state_j.ml:
                    # check l selection rule: delta_l = 0 or delta_l = +-2
                    # Since r^2 is diagonal in l, we only have delta_l = 0 or delta_l = +-2 for sin^2(theta)
                    l1, l2 = state_i.l, state_j.l
                    ml = state_i.ml
                    
                    if l1 == l2:
                        r2_val = (n**2 / (2.0 * Z**2)) * (5.0 * n**2 + 1.0 - 3.0 * l1 * (l1 + 1.0))
                        # <l, ml | cos^2(theta) | l, ml>
                        if l1 == 0:
                            cos2_val = 1.0 / 3.0
                        else:
                            cos2_val = (2.0 * l1**2 + 2.0 * l1 - 1.0 - 2.0 * ml**2) / ((2.0 * l1 - 1.0) * (2.0 * l1 + 3.0))
                        sin2_val = 1.0 - cos2_val
                        H[i, j] += quad_coeff_ev * r2_val * sin2_val
                        
                    elif abs(l1 - l2) == 2:
                        # Off-diagonal element due to cos^2(theta)
                        # <l_lower, ml | cos^2(theta) | l_lower+2, ml>
                        l_low = min(l1, l2)
                        num_term = ((l_low + 1.0)**2 - ml**2) * ((l_low + 2.0)**2 - ml**2)
                        den_term = (2.0 * l_low + 1.0) * (2.0 * l_low + 3.0)**2 * (2.0 * l_low + 5.0)
                        cos2_val = np.sqrt(num_term / den_term)
                        # sin2_val = - cos2_val (since delta_l = 2, the 1 in 1-cos^2 is orthogonal and yields 0)
                        sin2_val = - cos2_val
                        
                        r2_val = radial_r2_element(n, l1, l2, Z)
                        H[i, j] += quad_coeff_ev * r2_val * sin2_val

    return H

def diagonalize_hamiltonian(n, Z, B, include_quadratic=True, include_fine_structure=True):
    """Diagonalise the field-free (F = 0) magnetic Hamiltonian for shell n.

    Builds the (2n²) × (2n²) Hamiltonian via :func:`build_hamiltonian`
    with no applied electric field and diagonalises it using ``numpy.linalg.eigh``
    (real symmetric solver, numerically stable).

    Parameters
    ----------
    n : int
        Principal quantum number.
    Z : int
        Nuclear charge.
    B : float
        Magnetic field [T].  B = 0 is valid (returns hydrogenic/fine-structure
        eigenvalues).
    include_quadratic : bool, optional
        Include the diamagnetic (quadratic Zeeman) term (default True).
    include_fine_structure : bool, optional
        Include mass-velocity and Darwin corrections alongside spin-orbit
        coupling (default True).

    Returns
    -------
    eigenvalues : ndarray, shape (2n²,)
        Energy eigenvalues in ascending order [eV].
    eigenvectors : ndarray, shape (2n², 2n²)
        Columns are the corresponding orthonormal eigenstates expressed in the
        |n, l, m_l, m_s⟩ basis of :func:`build_basis`.
    """
    H = build_hamiltonian(n, Z, B, include_quadratic, include_fine_structure)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return eigenvalues, eigenvectors

def dipole_matrix_elements(n_u, n_l, Z, B, include_quadratic=True,
                                       include_fine_structure=True):
    """Return transition dipole matrices between the eigenstate bases of n_u and n_l.

    Diagonalises the Hamiltonian for both shells and rotates the uncoupled
    dipole matrices D_q into the eigenstate basis:

        d_q[i, j] = ⟨ψ_l^j | T_q^(1) | ψ_u^i⟩

    where ψ_u^i and ψ_l^j are the i-th upper and j-th lower eigenstates.

    Parameters
    ----------
    n_u, n_l : int
        Upper and lower principal quantum numbers.
    Z : int
        Nuclear charge.
    B : float
        Magnetic field [T].
    include_quadratic : bool, optional
        Include diamagnetic Zeeman term (default True).
    include_fine_structure : bool, optional
        Include MV + Darwin corrections (default True).

    Returns
    -------
    eigenvalues_u : ndarray, shape (2n_u²,)
        Upper-shell energy eigenvalues [eV].
    eigenvalues_l : ndarray, shape (2n_l²,)
        Lower-shell energy eigenvalues [eV].
    d_elements : dict
        Keys are q ∈ {0, 1, −1}.  Values are complex arrays of shape
        (2n_u², 2n_l²) where ``d_elements[q][i, j]`` is the dipole matrix
        element between upper eigenstate i and lower eigenstate j.
    """
    # Diagonalize atomic Hamiltonian for upper and lower shells
    eigenvalues_u, eigenvectors_u = diagonalize_hamiltonian(n_u, Z, B, include_quadratic, include_fine_structure)
    eigenvalues_l, eigenvectors_l = diagonalize_hamiltonian(n_l, Z, B, include_quadratic, include_fine_structure)
    
    basis_u = build_basis(n_u)
    basis_l = build_basis(n_l)
    dim_u = len(basis_u)
    dim_l = len(basis_l)
    
    # Precompute radial dipole transition elements between all (l_u, l_l)
    radial_dipoles = {}
    for l_u in range(n_u):
        for l_l in range(n_l):
            if abs(l_u - l_l) == 1:
                radial_dipoles[(l_u, l_l)] = radial_dipole(n_u, l_u, n_l, l_l, Z)
                
    # Build dipole matrices in upper and lower mixed bases
    # <mixed_l | d_q | mixed_u> = sum_ku,kl U_l[kl, mixed_l]* * U_u[ku, mixed_u] * <basis_l | d_q | basis_u>
    d_elements = {}
    
    for q in [0, 1, -1]:
        d_q = np.zeros((dim_u, dim_l), dtype=complex)
        for i in range(dim_u): # upper mixed state
            for j in range(dim_l): # lower mixed state
                val = 0.0
                for ku, state_ku in enumerate(basis_u):
                    for kl, state_kl in enumerate(basis_l):
                        # Selection rules
                        if state_ku.ms == state_kl.ms and abs(state_ku.l - state_kl.l) == 1:
                            r_dip = radial_dipoles.get((state_ku.l, state_kl.l), 0.0)
                            ang_dip = angular_dipole_element(state_kl.l, state_kl.ml, state_ku.l, state_ku.ml, q)
                            
                            # Note: dipole d = -e * r, we express in units of e * a0
                            # d_element = - r_dip * ang_dip
                            val += np.conj(eigenvectors_l[kl, j]) * eigenvectors_u[ku, i] * (-r_dip * ang_dip)
                d_q[i, j] = val
        d_elements[q] = d_q
        
    return eigenvalues_u, eigenvalues_l, d_elements


@lru_cache(maxsize=None)
def _uncoupled_dipole_matrices(n_u, n_l, Z):
    """Return the electric-dipole operator matrices in the uncoupled |n,l,ml,ms⟩ basis.

    Builds the three polarization matrices D_q (q = 0, +1, −1) without
    diagonalising the Hamiltonian.  The (j, i) element is

        D_q[j, i] = ⟨basis_l[j] | r̂_q | basis_u[i]⟩ = −R_{n_u l_u, n_l l_l} × Y_q(l_u,ml_u→l_l,ml_l)

    This is the shared kernel for :func:`~starkzee.static_profile.calculate_static_profile`,
    :func:`~starkzee.static_profile.discrete_transitions`, and
    :func:`~starkzee.ffm.calculate_ffm_profile`.

    Parameters
    ----------
    n_u, n_l : int
        Upper and lower principal quantum numbers.
    Z : int
        Nuclear charge.

    Returns
    -------
    D_q : dict {0: ndarray, 1: ndarray, −1: ndarray}
        Each value has shape (2n_l², 2n_u²).  Elements are in units of a₀.
    """
    basis_u = build_basis(n_u)
    basis_l = build_basis(n_l)
    dim_u   = len(basis_u)
    dim_l   = len(basis_l)

    r_cache = {}
    for l_u in range(n_u):
        for l_l in range(n_l):
            if abs(l_u - l_l) == 1:
                r_cache[(l_u, l_l)] = radial_dipole(n_u, l_u, n_l, l_l, Z)

    D_q = {}
    for q in [0, 1, -1]:
        D = np.zeros((dim_l, dim_u), dtype=complex)
        for kl, sl in enumerate(basis_l):
            for ku, su in enumerate(basis_u):
                if su.ms == sl.ms and abs(su.l - sl.l) == 1:
                    R   = r_cache.get((su.l, sl.l), 0.0)
                    ang = angular_dipole_element(sl.l, sl.ml, su.l, su.ml, q)
                    D[kl, ku] = -R * ang
        D_q[q] = D

    return D_q


def line_strength(n_u, n_l, Z):
    """Compute the line strength S_ul = Σ_{q,i,j} |<l_j|r_q|u_i>|² in a₀².

    Sums over all polarizations q ∈ {0,±1}, all uncoupled upper states i, and
    all lower states j, restricted to Δl = ±1 and Δms = 0 (dipole selection
    rules).  Fine structure mixing is neglected (field-free, B = 0 limit with
    the states labelled by |n, l, ml, ms>).

    The profile returned by calculate_static_profile integrates to
    approximately S_ul at B = 0 and Ne → 0.  The Gauss-Legendre weights over
    the microfield angle [0, 1] sum to 1 and the microfield distribution is
    normalized to 1, so the total quadrature weight is unity.

    Returns:
        S_ul (float): line strength in a₀²
    """
    basis_u = build_basis(n_u)
    basis_l = build_basis(n_l)
    S = 0.0
    r_cache = {}
    for state_u in basis_u:
        for state_l in basis_l:
            if state_u.ms != state_l.ms:
                continue
            if abs(state_u.l - state_l.l) != 1:
                continue
            key = (state_u.l, state_l.l)
            if key not in r_cache:
                r_cache[key] = radial_dipole(n_u, state_u.l, n_l, state_l.l, Z)
            R = r_cache[key]
            for q in [0, 1, -1]:
                ang = angular_dipole_element(state_l.l, state_l.ml, state_u.l, state_u.ml, q)
                S += (R * ang) ** 2
    return S


def oscillator_strength(n_u, n_l, Z):
    """Compute the weighted absorption oscillator strength gf = g_l × f_lu.

    Uses:  gf = (2/3) × (ΔE / E_hartree) × S_ul

    where E_hartree = 2 × RYDBERG_EV ≈ 27.211 eV, g_l = 2n_l² is the
    statistical weight of the lower shell, and S_ul is the total line
    strength (a₀²) summed over all allowed Δl = ±1 pairs.

    The result is independent of Z for hydrogen-like ions (the Z-dependence
    of ΔE ∝ Z² and S_ul ∝ Z⁻² cancel exactly).

    Convention matches NIST ASD for full shell-to-shell transitions (e.g. Hα,
    Hβ).  For Lyman series (n_l = 1) only l_u = 1 upper states contribute,
    so gf is the weighted oscillator strength for the specific 2p/3p/... level.

    Returns:
        gf (float): dimensionless weighted oscillator strength
    """
    E_hartree = 2.0 * RYDBERG_EV
    delta_E_ev = (Z**2) * RYDBERG_EV * (1.0/n_l**2 - 1.0/n_u**2)
    S = line_strength(n_u, n_l, Z)
    return (2.0/3.0) * (delta_E_ev / E_hartree) * S


def einstein_a(n_u, n_l, Z):
    """Compute the Einstein A coefficient for spontaneous emission, in s⁻¹.

    Uses the atomic-unit formula:

        A_ul = (4α³/3) × (ΔE/E_hartree)³ × S_ul / (2n_u²) / τ_au

    where α is the fine-structure constant, τ_au = ħ/E_hartree ≈ 2.419 × 10⁻¹⁷ s
    is the atomic unit of time, and g_u = 2n_u² is the full upper-shell
    statistical weight (NIST convention: all 2n_u² substates, including any
    non-radiating ones).

    This gives the shell-averaged spontaneous emission rate: the total Ly-α
    photon emission rate from a uniformly populated n=2 level equals
    n_{n=2} × einstein_a(2, 1, Z).  For NIST's level-specific A coefficient
    (e.g. A(2p→1s) using g_k = 6), multiply by 2n_u² / g_participating, where
    g_participating = Σ_{l_u that have an allowed lower state} 2(2l_u+1).

    Returns:
        A_ul (float): spontaneous emission rate in s⁻¹
    """
    E_hartree = 2.0 * RYDBERG_EV
    hbar_ev_s = HBAR / E_CHARGE
    tau_au = hbar_ev_s / E_hartree
    delta_E_hartree = (Z**2) * RYDBERG_EV * (1.0/n_l**2 - 1.0/n_u**2) / E_hartree
    S = line_strength(n_u, n_l, Z)
    g_u = 2 * n_u**2
    return (4.0/3.0) * FINE_STRUCTURE**3 * delta_E_hartree**3 * S / g_u / tau_au



