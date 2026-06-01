"""
test_18_discrete_transitions.py — discrete_transitions() output correctness.

Tests:
  1. Returns a dict with the five required keys
  2. Sum of strengths equals line_strength(n_u, n_l, Z)
  3. At B=0, F=0: all transition energies are equal (degeneracy, no FS)
  4. At large B: sigma+/sigma- peaks clearly split from pi
  5. Energies are sorted ascending
  6. min_strength filter removes weak transitions
  7. With Stark field: more transitions appear (Stark mixes ml states)
  8. All transition energies are positive (upper > lower shell)
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.static_profile import discrete_transitions
from starkzee.atomic_hamiltonian import line_strength
from starkzee.utils import reduced_mass_rydberg_ev, BOHR_MAGNETON_EV_T


def relerr(got, ref):
    return abs(got - ref) / abs(ref)


# ── 1. Correct return structure ───────────────────────────────────────────────

def test_return_structure():
    """Must return a dict with the five required arrays of equal length."""
    tr = discrete_transitions(n_u=2, n_l=1, Z=1, B=1.0)
    for key in ('energy_ev', 'q', 'strength', 'upper_idx', 'lower_idx'):
        assert key in tr, f"Missing key '{key}'"
    n = len(tr['energy_ev'])
    for key in tr:
        assert len(tr[key]) == n, f"Array '{key}' has wrong length"
    assert n > 0, "No transitions returned"


# ── 2. Sum of strengths == line_strength ────────────────────────────────────

@pytest.mark.parametrize("n_u,n_l,Z,B", [
    (2, 1, 1, 0.0),   # H Ly-α, B=0
    (2, 1, 1, 5.0),   # H Ly-α, B=5T
    (3, 2, 1, 0.0),   # H Hα
    (3, 2, 6, 10.0),  # C VI Hα
])
def test_strength_sum_equals_line_strength(n_u, n_l, Z, B):
    """Σ|d_q|² over all transitions must equal line_strength() (unitary invariance)."""
    tr = discrete_transitions(n_u=n_u, n_l=n_l, Z=Z, B=B)
    S_got = tr['strength'].sum()
    S_ref = line_strength(n_u, n_l, Z)
    assert relerr(S_got, S_ref) < 1e-10, (
        f"n={n_u}→{n_l},Z={Z},B={B}T: Σstrength={S_got:.8e}, S_ul={S_ref:.8e}"
    )


# ── 3. At B=0, F=0: intensity-weighted centroid equals the Bohr energy ───────
# Spin-orbit coupling (always present) splits 2p_{1/2} from 2p_{3/2} but
# preserves the intensity-weighted mean energy.  That centroid must equal E0
# because the dipole strengths per J-level are proportional to (2J+1), so
# the SO redistributes intensity symmetrically around E0.

@pytest.mark.parametrize("n_u,n_l,Z", [(2,1,1),(3,2,1),(2,1,6)])
def test_centroid_at_zero_field(n_u, n_l, Z):
    """Intensity-weighted centroid must equal the Bohr energy at B=0, F=0."""
    tr = discrete_transitions(n_u=n_u, n_l=n_l, Z=Z, B=0.0, Fz=0.0, Fx=0.0,
                               include_fine_structure=False)
    E0 = (Z**2) * reduced_mass_rydberg_ev(Z, 1) * (1.0/n_l**2 - 1.0/n_u**2)
    centroid = np.sum(tr['energy_ev'] * tr['strength']) / np.sum(tr['strength'])
    assert relerr(centroid, E0) < 1e-8, (
        f"n={n_u}→{n_l},Z={Z}: centroid={centroid:.8f} eV, E0={E0:.8f} eV"
    )


# ── 4. Large-B Zeeman splitting: sigma split from pi ────────────────────────

def test_zeeman_sigma_split_at_large_B():
    """At B=50T, sigma+/sigma- centroids should be shifted ≈ ±μ_B*B from pi."""
    n_u, n_l, Z, B = 2, 1, 1, 50.0
    tr = discrete_transitions(n_u=n_u, n_l=n_l, Z=Z, B=B,
                               include_fine_structure=False)

    def centroid(mask):
        w = tr['strength'][mask]
        return np.sum(tr['energy_ev'][mask] * w) / np.sum(w)

    E_pi = centroid(tr['q'] == 0)
    E_sp = centroid(tr['q'] == -1)   # code's q=-1 is σ+
    E_sm = centroid(tr['q'] ==  1)   # code's q=+1 is σ−

    expected_shift = BOHR_MAGNETON_EV_T * B  # ≈ 2.89 meV at 50T
    assert abs(E_sp - E_pi) > 0.5 * expected_shift, (
        f"σ+ centroid not clearly split from π: ΔE={E_sp-E_pi:.4e} eV"
    )
    assert abs(E_sm - E_pi) > 0.5 * expected_shift, (
        f"σ− centroid not clearly split from π: ΔE={E_sm-E_pi:.4e} eV"
    )
    # Sigma components should be on opposite sides of pi
    assert (E_sp - E_pi) * (E_sm - E_pi) < 0, "σ+ and σ− not on opposite sides of π"


# ── 5. Energies sorted ascending ────────────────────────────────────────────

@pytest.mark.parametrize("B,Fz", [(0.0, 0.0),(10.0, 0.0),(5.0, 1e8)])
def test_energies_sorted(B, Fz):
    """Returned energies must be in non-decreasing order."""
    tr = discrete_transitions(n_u=3, n_l=2, Z=1, B=B, Fz=Fz)
    diffs = np.diff(tr['energy_ev'])
    assert np.all(diffs >= -1e-15), (
        f"B={B}T, Fz={Fz}: energies not sorted, min diff={diffs.min():.3e}"
    )


# ── 6. min_strength filter reduces number of transitions ────────────────────

def test_min_strength_filter():
    """Applying a min_strength threshold must return fewer transitions."""
    tr_all = discrete_transitions(n_u=3, n_l=2, Z=1, B=5.0)
    s_max = tr_all['strength'].max()
    tr_filtered = discrete_transitions(n_u=3, n_l=2, Z=1, B=5.0,
                                        min_strength=0.01 * s_max)
    assert len(tr_filtered['energy_ev']) < len(tr_all['energy_ev']), (
        "min_strength filter had no effect"
    )
    assert np.all(tr_filtered['strength'] >= 0.01 * s_max)


# ── 7. Stark field mixes states and splits degenerate transitions ────────────

def test_stark_breaks_degeneracy():
    """With a Stark field, the zero-field degenerate manifold splits."""
    tr_B0 = discrete_transitions(n_u=2, n_l=1, Z=1, B=0.0,
                                   include_fine_structure=False)
    tr_F  = discrete_transitions(n_u=2, n_l=1, Z=1, B=0.0,
                                   Fz=1e8, include_fine_structure=False)

    spread_B0 = tr_B0['energy_ev'].max() - tr_B0['energy_ev'].min()
    spread_F  = tr_F['energy_ev'].max()  - tr_F['energy_ev'].min()
    assert spread_F > spread_B0 + 1e-6, (
        f"Stark field did not split transitions: "
        f"spread(F=0)={spread_B0:.3e}, spread(F≠0)={spread_F:.3e}"
    )


# ── 8. All energies positive ─────────────────────────────────────────────────

@pytest.mark.parametrize("n_u,n_l", [(2,1),(3,2),(4,3)])
def test_energies_positive(n_u, n_l):
    """All transition energies must be positive (upper shell higher than lower)."""
    tr = discrete_transitions(n_u=n_u, n_l=n_l, Z=1, B=1.0)
    assert np.all(tr['energy_ev'] > 0), (
        f"n={n_u}→{n_l}: found non-positive transition energy "
        f"min={tr['energy_ev'].min():.6e} eV"
    )
