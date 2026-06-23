"""
test_19_electron_model.py — Selectable electron-impact model (Ferri/PPPB vs ZEST).

Verifies the ``electron_model`` dispatcher and its propagation through the
static-profile, FFM, and LineProfile paths:
  1. The dispatcher matches the underlying width functions for every selector.
  2. The default ('ferri') is byte-identical to the historical behavior.
  3. Selecting a ZEST model actually changes the profile.
  4. The parameter flows through LineProfile.compute_profile(**kwargs).
  5. An unknown model raises ValueError.
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.broadening import (
    electron_impact_width,
    electron_impact_width_zest,
    electron_impact_width_model,
    electron_impact_r2_scaling,
    ELECTRON_MODELS,
)
from starkzee.radiator import build_hamiltonian
from starkzee.static_profile import calculate_static_profile
from starkzee.ffm import calculate_ffm_profile
from starkzee.line_profile import LineProfile

NE, TE, B, Z, N = 1e23, 5.0, 1000.0, 1, 4


def test_dispatcher_matches_underlying():
    f = lambda m: electron_impact_width_model(0.0, NE, TE, B, Z, N, electron_model=m)
    assert f('ferri') == electron_impact_width(0.0, NE, TE, B, Z, n=N)
    assert f('pppb') == electron_impact_width(0.0, NE, TE, B, Z, n=N)
    assert f('zest') == electron_impact_width_zest(0.0, NE, TE, Z, n=N, model='gbk')
    assert f('zest-gbk') == electron_impact_width_zest(0.0, NE, TE, Z, n=N, model='gbk')
    assert f('zest-lee') == electron_impact_width_zest(0.0, NE, TE, Z, n=N, model='lee')
    assert f('zest-dufty') == electron_impact_width_zest(0.0, NE, TE, Z, n=N, model='dufty')


def test_all_models_positive_finite():
    for m in ELECTRON_MODELS:
        w = electron_impact_width_model(0.0, NE, TE, B, Z, N, electron_model=m)
        assert np.isfinite(w) and w > 0.0


def test_case_insensitive():
    assert (electron_impact_width_model(0.0, NE, TE, B, Z, N, electron_model='ZEST')
            == electron_impact_width_model(0.0, NE, TE, B, Z, N, electron_model='zest'))


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        electron_impact_width_model(0.0, NE, TE, B, Z, N, electron_model='bogus')


def test_vectorized_detuning():
    dw = np.array([0.0, 0.05, 0.1])
    w = electron_impact_width_model(dw, NE, TE, B, Z, N, electron_model='zest-lee')
    assert w.shape == dw.shape and np.all(np.isfinite(w)) and np.all(w > 0)


def test_static_default_is_ferri():
    en = np.linspace(2.5, 2.6, 300)
    base = dict(n_u=4, n_l=2, Z=1, B=B, Ne_m3=NE, Te_ev=TE,
                energies_ev=en, num_f=10, num_mu=4)
    d_default = calculate_static_profile(**base)
    d_ferri = calculate_static_profile(**base, electron_model='ferri')
    for a, b in zip(d_default, d_ferri):
        assert np.array_equal(a, b)


def test_static_zest_differs():
    en = np.linspace(2.5, 2.6, 300)
    base = dict(n_u=4, n_l=2, Z=1, B=B, Ne_m3=NE, Te_ev=TE,
                energies_ev=en, num_f=10, num_mu=4)
    d_ferri = calculate_static_profile(**base, electron_model='ferri')
    d_zest = calculate_static_profile(**base, electron_model='zest')
    assert not np.allclose(d_ferri[0], d_zest[0])
    assert np.all(np.isfinite(d_zest[0]))


def test_lineprofile_forwards_electron_model():
    en = np.linspace(2.5, 2.6, 300)
    lp = LineProfile(n_u=4, n_l=2, B=B, Ne_m3=NE, Te_ev=TE)
    lp.compute_profile(en, grid_type='energy_ev', num_f=10, num_mu=4,
                       electron_model='zest-lee')
    assert np.all(np.isfinite(lp.profile))


def test_ffm_default_is_ferri_and_zest_differs():
    en = np.linspace(2.5, 2.6, 200)
    args = (4, 2, 1, B, NE, TE, 5.0, 1.0, en)
    kw = dict(num_f=8, num_mu=4)
    f_default = calculate_ffm_profile(*args, **kw)
    f_ferri = calculate_ffm_profile(*args, **kw, electron_model='ferri')
    f_zest = calculate_ffm_profile(*args, **kw, electron_model='zest')
    for a, b in zip(f_default, f_ferri):
        assert np.array_equal(a, b)
    assert not np.allclose(f_default[0], f_zest[0])


# ── electron-impact operator (ZEST operator diagonal, c_k = 0) ────────────────

def test_r2_scaling_trace_preserved_and_varies():
    # The operator diagonal must average to 1 (trace preserved) and vary per state.
    _, V = np.linalg.eigh(build_hamiltonian(4, 1, B))
    s = electron_impact_r2_scaling(V, n=4, Z=1)
    assert s.shape == (2 * 4**2,)
    assert abs(s.mean() - 1.0) < 1e-9
    assert s.max() - s.min() > 0.3   # l=0..3 spread is real


def test_r2_scaling_no_field_is_unit_per_l():
    # With no Stark/Zeeman mixing beyond ml, eigenstates keep definite l, so the
    # scaling equals ⟨r²⟩_{n,l}/⟨r²⟩_avg for whichever l each eigenstate carries.
    _, V = np.linalg.eigh(build_hamiltonian(2, 1, 0.0, quadratic_zeeman=False,
                                            fine_structure=False))
    s = electron_impact_r2_scaling(V, n=2, Z=1)
    assert abs(s.mean() - 1.0) < 1e-9
    assert np.all(s > 0)


def test_operator_default_off():
    en = np.linspace(2.45, 2.65, 400)
    base = dict(n_u=4, n_l=2, Z=1, B=B, Ne_m3=NE, Te_ev=TE,
                energies_ev=en, num_f=10, num_mu=4)
    d_default = calculate_static_profile(**base)
    d_off = calculate_static_profile(**base, electron_operator=False)
    for a, b in zip(d_default, d_off):
        assert np.array_equal(a, b)


def test_operator_changes_profile_but_conserves_norm():
    en = np.linspace(2.35, 2.75, 1500)   # wide window so wings are captured
    base = dict(n_u=4, n_l=2, Z=1, B=B, Ne_m3=NE, Te_ev=TE,
                energies_ev=en, num_f=14, num_mu=6)
    d_sc = calculate_static_profile(**base, electron_operator=False)
    d_op = calculate_static_profile(**base, electron_operator=True)
    assert not np.allclose(d_sc[0], d_op[0])
    tot_sc = np.trapezoid(d_sc[0] + 0.5 * (d_sc[1] + d_sc[2]), en)
    tot_op = np.trapezoid(d_op[0] + 0.5 * (d_op[1] + d_op[2]), en)
    # Operator redistributes width, it does not add or remove total intensity.
    assert abs(tot_op - tot_sc) / tot_sc < 5e-3


def test_operator_composes_with_zest_model():
    en = np.linspace(2.45, 2.65, 400)
    base = dict(n_u=4, n_l=2, Z=1, B=B, Ne_m3=NE, Te_ev=TE,
                energies_ev=en, num_f=10, num_mu=4)
    d_zest_scalar = calculate_static_profile(**base, electron_model='zest')
    d_zest_op = calculate_static_profile(**base, electron_model='zest',
                                         electron_operator=True)
    assert np.all(np.isfinite(d_zest_op[0]))
    assert not np.allclose(d_zest_scalar[0], d_zest_op[0])


def test_ffm_numerical_inversion_matches_analytical():
    en = np.linspace(1.88, 1.90, 50)
    # Using a small number of quadrature points to avoid memory issues (2x2 grid)
    p_ana = calculate_ffm_profile(n_u=3, n_l=2, Z=1, B=1.0, Ne_m3=1e23, Te_ev=1.0, Ti_ev=1.0, A_ion=1.0,
                                  energies_ev=en, num_f=2, num_mu=2, numerical_inversion=False)
    p_num = calculate_ffm_profile(n_u=3, n_l=2, Z=1, B=1.0, Ne_m3=1e23, Te_ev=1.0, Ti_ev=1.0, A_ion=1.0,
                                  energies_ev=en, num_f=2, num_mu=2, numerical_inversion=True)
    # Ensure they match for all three polarizations
    for a, b in zip(p_ana, p_num):
        assert np.allclose(a, b, rtol=1e-5, atol=1e-7)
