"""
test_20_empirical_static_and_ffm_api.py — Regression tests for functionality
merged in from a parallel checkout (2026-08-23 handoff):

  1. calculate_static_profile(use_empirical_data=True, atom=...) — the cm⁻¹/eV
     unit handling and E0_line reconciliation (previously only exercised at the
     bare-Hamiltonian level by test_empirical_hamiltonian.py).
  2. LineProfile.compute_static_profile / compute_ffm_profile — the FFM path
     newly exposed on the high-level API, with D/T isotope levels.
  3. calculate_ffm_profile(sdt_bin_tol=...) — SDT frequency binning must not
     change the physical result beyond its own tolerance.
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.static_profile import calculate_static_profile
from starkzee.ffm import calculate_ffm_profile
from starkzee.line_profile import LineProfile
from starkzee.utils import vacuum_to_air_wavelength_nm, wavelength_nm_to_energy_ev

# NIST air-wavelength references (nm), same table used in models/analytical.py.
_NIST_AIR_NM = {'H': 656.279, 'D': 656.1012}


@pytest.mark.parametrize("species", ["H", "D"])
def test_calculate_static_profile_empirical_matches_nist(species):
    """Empirical-level static profile centroid lands on the NIST air wavelength."""
    lp0 = LineProfile(n_u=3, n_l=2, B=1e-3, Ne_m3=1e20, Te_ev=1.0, species=species)
    wl_nm = np.linspace(lp0.E0_wavelength_nm - 1.0, lp0.E0_wavelength_nm + 1.0, 2000)
    energies_ev = wavelength_nm_to_energy_ev(wl_nm)

    pi, sp, sm = calculate_static_profile(
        n_u=3, n_l=2, Z=1, B=1e-3, Ne_m3=1e20, Te_ev=1.0, energies_ev=energies_ev,
        A=lp0.A, num_f=8, num_mu=4, use_empirical_data=True, atom=species,
    )
    total = pi + sp + sm
    assert np.all(np.isfinite(total)) and total.sum() > 0

    wl_air_nm = vacuum_to_air_wavelength_nm(wl_nm)
    centroid_air_nm = float(np.sum(wl_air_nm * total) / np.sum(total))
    nist = _NIST_AIR_NM[species]
    assert abs(centroid_air_nm - nist) < 0.01, (
        f"{species} empirical centroid {centroid_air_nm:.4f} nm vs NIST {nist} nm "
        f"(diff {1e3*(centroid_air_nm - nist):+.2f} pm) — E0_line/unit reconciliation broken"
    )


def test_calculate_static_profile_empirical_vs_analytic_close():
    """Empirical and analytic (Dirac) H centroids should agree to the Lamb-shift scale."""
    lp0 = LineProfile(n_u=3, n_l=2, B=1e-3, Ne_m3=1e20, Te_ev=1.0, species='H')
    wl_nm = np.linspace(lp0.E0_wavelength_nm - 1.0, lp0.E0_wavelength_nm + 1.0, 2000)
    energies_ev = wavelength_nm_to_energy_ev(wl_nm)
    kwargs = dict(n_u=3, n_l=2, Z=1, B=1e-3, Ne_m3=1e20, Te_ev=1.0,
                 energies_ev=energies_ev, num_f=8, num_mu=4)

    pi_e, sp_e, sm_e = calculate_static_profile(**kwargs, use_empirical_data=True, atom='H')
    pi_a, sp_a, sm_a = calculate_static_profile(**kwargs)

    def centroid(pi, sp, sm):
        tot = pi + sp + sm
        return float(np.sum(energies_ev * tot) / np.sum(tot))

    diff_ev = abs(centroid(pi_e, sp_e, sm_e) - centroid(pi_a, sp_a, sm_a))
    assert diff_ev < 1e-3, f"empirical vs analytic centroid differ by {diff_ev:.2e} eV (expect Lamb-shift scale, ~1e-5 eV)"


def test_lineprofile_compute_static_profile_alias():
    """compute_static_profile is a working alias of compute_profile."""
    lp_a = LineProfile(n_u=3, n_l=2, B=10.0, Ne_m3=1e20, Te_ev=1.0)
    lp_b = LineProfile(n_u=3, n_l=2, B=10.0, Ne_m3=1e20, Te_ev=1.0)
    wl_nm = np.linspace(lp_a.E0_wavelength_nm - 0.5, lp_a.E0_wavelength_nm + 0.5, 300)

    lp_a.compute_profile(wl_nm, grid_type='wavelength_nm', num_f=6, num_mu=3)
    lp_b.compute_static_profile(wl_nm, grid_type='wavelength_nm', num_f=6, num_mu=3)

    np.testing.assert_array_equal(lp_a.profile_pi, lp_b.profile_pi)
    np.testing.assert_array_equal(lp_a.profile_sig_plus, lp_b.profile_sig_plus)
    np.testing.assert_array_equal(lp_a.profile_sig_minus, lp_b.profile_sig_minus)
    assert lp_b.profile is not None
    assert np.array_equal(lp_b.energies_ev, wavelength_nm_to_energy_ev(wl_nm))


def test_lineprofile_compute_ffm_profile_requires_ti():
    """compute_ffm_profile without Ti_ev raises rather than silently misbehaving."""
    lp = LineProfile(n_u=3, n_l=2, B=10.0, Ne_m3=1e20, Te_ev=1.0)  # no Ti_ev
    wl_nm = np.linspace(lp.E0_wavelength_nm - 0.5, lp.E0_wavelength_nm + 0.5, 100)
    with pytest.raises(ValueError, match="Ti_ev"):
        lp.compute_ffm_profile(wl_nm, grid_type='wavelength_nm', num_f=4, num_mu=2)


def test_lineprofile_compute_ffm_profile_matches_calculate_ffm_profile():
    """The LineProfile.compute_ffm_profile wrapper matches the underlying function call."""
    lp = LineProfile(n_u=3, n_l=2, B=10.0, Ne_m3=1e20, Te_ev=1.0, Ti_ev=1.0, species='H')
    wl_nm = np.linspace(lp.E0_wavelength_nm - 0.5, lp.E0_wavelength_nm + 0.5, 200)
    energies_ev = wavelength_nm_to_energy_ev(wl_nm)

    lp.compute_ffm_profile(wl_nm, grid_type='wavelength_nm', num_f=6, num_mu=3)

    pi, sp, sm = calculate_ffm_profile(
        n_u=3, n_l=2, Z=1, B=10.0, Ne_m3=1e20, Te_ev=1.0, Ti_ev=1.0, A_ion=lp.A,
        energies_ev=energies_ev, num_f=6, num_mu=3,
    )
    np.testing.assert_array_equal(lp.profile_pi, pi)
    np.testing.assert_array_equal(lp.profile_sig_plus, sp)
    np.testing.assert_array_equal(lp.profile_sig_minus, sm)


def test_lineprofile_compute_ffm_profile_empirical_isotope():
    """compute_ffm_profile forwards use_empirical_data/atom; D centroid lands on NIST."""
    lp = LineProfile(n_u=3, n_l=2, B=1e-3, Ne_m3=1e20, Te_ev=1.0, Ti_ev=1.0, species='D')
    wl_nm = np.linspace(lp.E0_wavelength_nm - 1.0, lp.E0_wavelength_nm + 1.0, 1000)
    lp.compute_ffm_profile(wl_nm, grid_type='wavelength_nm', num_f=6, num_mu=3,
                           use_empirical_data=True, atom='D', apply_doppler=False)
    assert np.all(np.isfinite(lp.profile)) and lp.profile.sum() > 0
    centroid_air_nm = float(np.sum(lp.wavelengths_air_nm * lp.profile) / np.sum(lp.profile))
    assert abs(centroid_air_nm - _NIST_AIR_NM['D']) < 0.01


# ── SDT binning (sdt_bin_tol) ────────────────────────────────────────────────

def test_ffm_sdt_bin_tol_matches_unbinned():
    """Binning SDTs before the Markov solve must not change the profile beyond tol."""
    kwargs = dict(n_u=3, n_l=2, Z=1, B=10.0, Ne_m3=1e20, Te_ev=1.0, Ti_ev=1.0, A_ion=1,
                  num_f=8, num_mu=4, apply_doppler=False)
    en = np.linspace(1.85, 1.93, 500)

    unbinned = calculate_ffm_profile(**kwargs, energies_ev=en)
    binned = calculate_ffm_profile(**kwargs, energies_ev=en, sdt_bin_tol=1e-5)

    for a, b in zip(unbinned, binned):
        assert np.all(np.isfinite(b))
        peak = max(a.max(), 1e-300)
        assert np.max(np.abs(a - b)) / peak < 5e-2


def test_ffm_sdt_bin_tol_none_is_default():
    """sdt_bin_tol=None (default) reproduces the pre-binning code path exactly."""
    kwargs = dict(n_u=3, n_l=2, Z=1, B=10.0, Ne_m3=1e20, Te_ev=1.0, Ti_ev=1.0, A_ion=1,
                  num_f=6, num_mu=3, apply_doppler=False)
    en = np.linspace(1.85, 1.93, 200)
    default = calculate_ffm_profile(**kwargs, energies_ev=en)
    explicit_none = calculate_ffm_profile(**kwargs, energies_ev=en, sdt_bin_tol=None)
    for a, b in zip(default, explicit_none):
        np.testing.assert_array_equal(a, b)
