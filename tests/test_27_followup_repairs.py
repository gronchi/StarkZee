"""Observable regressions from the independent follow-up audit."""
import numpy as np
import pytest
from starkzee.convolutions import apply_doppler_broadening, apply_instrument_broadening, convolve_fft
from starkzee.static_profile import calculate_static_profile, line_reference_energy
from starkzee.ffm import calculate_ffm_profile
from starkzee.microfield import microfield_quadrature, _P_charged_screened
from starkzee.models import rosato_impl as r
from starkzee.radiator import radial_dipole
from starkzee.line_profile import LineProfile


@pytest.mark.parametrize('n', [1000, 1001])
def test_gaussian_centroid(n):
    x = np.arange(n, dtype=float)
    p = np.zeros(n); p[n//2] = 1
    out = apply_instrument_broadening(x, p, 10)
    assert np.sum(x*out)/out.sum() == pytest.approx(n//2, abs=1e-10)


def test_off_center_doppler():
    x = np.linspace(655, 658, 3001)
    p = np.zeros(len(x)); p[1000] = 1
    out = apply_doppler_broadening(x, p, 1, lambda0_nm=656)
    assert x[out.argmax()] == 656
    np.testing.assert_array_equal(apply_doppler_broadening(x, p, 0), p)


def test_fft_rejects_nonuniform():
    with pytest.raises(ValueError, match='uniform'):
        convolve_fft([0, 1, 3], [0, 1, 0], [0, 1, 0])


@pytest.mark.parametrize('solver', ['static', 'ffm'])
def test_solver_coordinate_reversal(solver):
    grid = 1239.841984 / np.linspace(650, 660, 121)
    kw = dict(n_u=3, n_l=2, Z=1, B=1, Ne_m3=1e22, Te_ev=5,
              Ti_ev=5, num_f=4, num_mu=2, microfield_model='holtsmark')
    fn = calculate_static_profile if solver == 'static' else calculate_ffm_profile
    if solver == 'ffm': kw['A_ion'] = 1
    p = fn(energies_ev=grid, **kw)
    q = fn(energies_ev=grid[::-1], **kw)
    np.testing.assert_allclose(p, np.array(q)[:, ::-1], rtol=1e-12, atol=1e-12)


def test_radial_high_n():
    assert radial_dipole(10, 0, 9, 1, 1, method='quad') == pytest.approx(
        radial_dipole(10, 0, 9, 1, 1, method='gordon'), rel=1e-9)


def test_potekhin_production_value():
    assert _P_charged_screened(np.array([1.]), 4., 1.)[0] == pytest.approx(.3725552442350799, rel=1e-10)


def test_rosato_upper_endpoints_and_names():
    assert r._set_bounds(1e16, 31.6, 5) == (10, 5, 6)
    names = r._set_name_file(3, 10, 2, 3, 0)
    assert names[:4] == 'ls09' and names[44:48] == 'ls10'
    assert len(names) == 88


def test_rosato_bilinear_positivity_and_exact_nodes():
    w = np.broadcast_to(np.linspace(-1, 1, 1000), (2, 2, 2, 1000)).copy()
    p = np.zeros_like(w)
    for d in range(2):
        for t in range(2): p[d, t] = 1 + d + 3*t
    u, v = .8, .8
    density = r.density_val[0] * (r.density_val[1]/r.density_val[0])**u
    temp = r.temperature_val[0] * (r.temperature_val[1]/r.temperature_val[0])**v
    out = r._ls_interpol(density, temp, 0, .5, 31, w, p, 2, 2, 2)
    np.testing.assert_allclose(out, 1 + u + 3*v)
    out = r._ls_interpol(r.density_val[1], r.temperature_val[1], 0, .5, 31, w, p, 2, 2, 2)
    np.testing.assert_allclose(out, 5)


def test_rosato_real_top_cell_not_constant():
    if r._NC_DB is None: pytest.skip('No bundled Rosato data')
    names = r._set_name_file(3, 10, 2, 3, 0)
    w, p = r._read_file('D_alpha', names)
    a = r._ls_interpol(5e15, .5, 1, .01, 1001, w, p, 10, 2, 3, 'clip')
    b = r._ls_interpol(9e15, .5, 1, .01, 1001, w, p, 10, 2, 3, 'clip')
    assert np.max(abs(a-b)) > .001*np.max(a)
    assert np.min(a) >= 0 and np.min(b) >= 0


def test_negative_rosato_source_policies():
    w = np.broadcast_to(np.linspace(-1, 1, 1000), (2, 2, 2, 1000)).copy()
    p = np.ones_like(w); p[0, 0, 0, 500] = -.01
    with pytest.raises(ValueError, match='source table'):
        r._ls_interpol(1e13, .316, 0, .5, 31, w, p, 2, 2, 2, 'raise')
    with pytest.warns(UserWarning, match='preserving signed'):
        preserved = r._ls_interpol(1e13, .316, 0, 1, 1000, w, p, 2, 2, 2)
    np.testing.assert_allclose(preserved, p[0, 0, 0], atol=1e-12)
    assert preserved[500] == pytest.approx(-.01)
    with pytest.warns(UserWarning, match='projecting'):
        out = r._ls_interpol(1e13, .316, 0, .5, 31, w, p, 2, 2, 2, 'clip')
    assert np.min(out) >= 0


def test_custom_failure_is_not_silent():
    with pytest.raises(ValueError, match='custom microfield'):
        microfield_quadrature(1e20, 5, custom_table_path='nonexistent-audit-table.txt')


def test_rosato_preserves_negative_values_after_doppler(monkeypatch):
    # A broad signed trough survives Gaussian smoothing. Check the public
    # wrapper so a hidden post-convolution positivity projection cannot recur.
    def signed_profile(density, temperature, field, wmax, npts, *args):
        x = np.linspace(-1, 1, npts)
        return 1 - 2*np.exp(-(x/.3)**2)
    monkeypatch.setattr(r, '_ls_interpol', signed_profile)
    grid = np.linspace(654, 658, 2001)
    out = r.rosato(grid, 3, 2, 1, 1e20, 1, 1, species='D')
    assert np.all(np.isfinite(out))
    assert out.min() < 0
    assert np.trapezoid(out, grid*1e-9) == pytest.approx(1)


@pytest.mark.parametrize('solver', ['static', 'ffm'])
def test_custom_and_charge_forwarding(solver, tmp_path, monkeypatch):
    table = tmp_path / 'microfield.txt'
    table.write_text('0 0\n1 1\n2 0.5\n10 0\n')
    import importlib
    module = importlib.import_module('starkzee.static_profile' if solver == 'static' else 'starkzee.ffm')
    seen = {}
    original = module.microfield_quadrature
    def capture(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'microfield_quadrature', capture)
    kw = dict(n_u=3, n_l=2, Z=2, B=1, Ne_m3=1e22, Te_ev=5, Ti_ev=5,
              energies_ev=np.linspace(7.5, 7.6, 51), num_f=4, num_mu=2,
              microfield_model='custom', custom_table_path=str(table), Z_bar=2,
              use_empirical_data=False)
    if solver == 'static': fn = module.calculate_static_profile
    else: fn = module.calculate_ffm_profile; kw['A_ion'] = 4
    out = fn(**kw)
    assert seen['charged'] is True and seen['Z_bar'] == 2
    assert seen['custom_table_path'] == str(table)
    assert np.all(np.isfinite(out))


def test_constant_width_operator_not_discarded():
    kw = dict(n_u=3, n_l=2, Z=1, B=1, Ne_m3=1e22, Te_ev=5, Ti_ev=5,
              energies_ev=np.linspace(1.88, 1.90, 201), num_f=4, num_mu=2,
              microfield_model='holtsmark', frequency_dependent_width=False)
    a = np.array(calculate_static_profile(**kw, electron_operator=True))
    b = np.array(calculate_static_profile(**kw, electron_operator=False))
    assert np.max(abs(a-b)) > 1e-3*np.max(b)


def test_empirical_reference_lifecycle():
    lp = LineProfile(3, 2, B=1, Ne_m3=1e22, Te_ev=5, species='D')
    grid = np.linspace(1.88, 1.90, 100)
    kw = dict(num_f=3, num_mu=2, microfield_model='holtsmark')
    lp.compute_profile(grid, use_empirical_data=True, **kw)
    ref = line_reference_energy(3, 2, 1, 2, True, 'D')
    assert lp.reference_energy_ev == ref
    np.testing.assert_allclose(lp.detuning_ev, grid-ref)
    before = lp.profile.copy()
    with pytest.raises(ValueError):
        lp.compute_profile(grid, use_empirical_data=True, atom='invalid', **kw)
    np.testing.assert_array_equal(lp.profile, before)
    assert lp.reference_energy_ev == ref
    lp.compute_discrete(use_empirical_data=True)
    np.testing.assert_allclose(lp.discrete.detuning_ev, lp.discrete.energy_ev-ref)
    lp.compute_profile(grid, use_empirical_data=False, **kw)
    assert lp.E0_empirical is None
    assert lp.reference_energy_ev == lp.E0


def test_density_jacobian():
    lp = LineProfile(3, 2, B=1, Ne_m3=1e22, Te_ev=5)
    grid = np.linspace(1.8, 2, 501)
    lp.compute_profile(grid, num_f=3, num_mu=2, microfield_model='holtsmark')
    energy_area = np.trapezoid(lp.profile, grid)
    for unit in ('wavelength_nm', 'frequency_thz', 'wavenumber_cm'):
        x, density = lp.spectral_density(unit)
        assert abs(np.trapezoid(density, x)) == pytest.approx(energy_area, rel=2e-6)
