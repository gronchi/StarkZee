"""Dependency-light audit checks: run with the StarkZee conda interpreter.

No pytest required. Does not modify tables or working-tree files.
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from starkzee.convolutions import apply_doppler_broadening, apply_instrument_broadening
from starkzee.radiator import (
    einstein_a, einstein_a_substate_rates, natural_decay_rates, radial_dipole,
)
from starkzee.models import rosato_impl as r
from starkzee.static_profile import calculate_static_profile
from starkzee.ffm import calculate_ffm_profile


def main():
    results = {'python': sys.version, 'numpy': np.__version__}
    grid = np.linspace(655, 658, 3001)
    p = np.zeros(len(grid)); p[1000] = 1
    out = apply_doppler_broadening(grid, p, 1, lambda0_nm=656)
    results['doppler_peak_nm'] = float(grid[out.argmax()])
    assert results['doppler_peak_nm'] == 656
    x = np.arange(1000.)
    p = np.zeros(1000); p[500] = 1
    out = apply_instrument_broadening(x, p, 10)
    results['even_centroid_error_samples'] = float(np.dot(x, out)/out.sum()-500)
    assert abs(results['even_centroid_error_samples']) < 1e-10
    results['radial_10s_9p'] = [radial_dipole(10, 0, 9, 1, 1, method=m)
                               for m in ('gordon', 'quad')]
    np.testing.assert_allclose(*results['radial_10s_9p'], rtol=1e-10)
    partial_21 = einstein_a_substate_rates(2, 1, 1)
    total_2 = natural_decay_rates(2, 1)
    np.testing.assert_allclose(partial_21, total_2, rtol=0, atol=0)
    np.testing.assert_allclose(np.mean(partial_21), einstein_a(2, 1, 1),
                               rtol=2e-15)
    assert np.count_nonzero(total_2 == 0.0) == 2
    results['natural_2s_rate_s-1'] = float(total_2[0])
    results['natural_2p_rate_s-1'] = float(total_2[2])
    common = dict(n_u=3, n_l=2, Z=1, B=1, Ne_m3=1e22, Te_ev=5,
                  Ti_ev=5, num_f=4, num_mu=2, microfield_model='holtsmark')
    grid = 1239.841984/np.linspace(650, 660, 121)
    for name, fn, extra in [('static', calculate_static_profile, {}),
                            ('ffm', calculate_ffm_profile, {'A_ion': 1})]:
        a = np.asarray(fn(energies_ev=grid, **common, **extra))
        b = np.asarray(fn(energies_ev=grid[::-1], **common, **extra))[:, ::-1]
        results[name+'_reversal_max_error'] = float(np.max(abs(a-b)))
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    assert r._set_bounds(1e16, 31.6, 5) == (10, 5, 6)
    if r._NC_DB is None:
        raise RuntimeError('Bundled Rosato database is required for this verification.')
    raw = np.asarray(r._NC_DB.variables['intensities'][:])
    results['rosato_raw_negative_entries'] = int(np.count_nonzero(raw < 0))
    results['rosato_raw_minimum'] = float(raw.min())
    cases = 0
    rejected = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for n in range(3, 8):
            for di in range(2, 11):
                for ti in range(2, 6):
                    density = r.density_val[di-2]*(r.density_val[di-1]/r.density_val[di-2])**.8
                    temp = r.temperature_val[ti-2]*(r.temperature_val[ti-1]/r.temperature_val[ti-2])**.8
                    for ai in range(2):
                        w, p = r._read_file(r._LINE_NAMES[n-3], r._set_name_file(n, di, ti, 3, ai))
                        preserved = r._ls_interpol(density, temp, 1, .001, 301, w, p, di, ti, 3)
                        det = np.linspace(-.001, .001, 301)
                        expected = sum(weight*np.interp(det, w[d, t, 0], p[d, t, 0], left=0, right=0)
                                       for d, t, weight in ((0, 0, .04), (1, 0, .16),
                                                             (0, 1, .16), (1, 1, .64)))
                        np.testing.assert_allclose(preserved, expected, rtol=1e-10, atol=1e-9)
                        try:
                            r._ls_interpol(density, temp, 1, .001, 301, w, p, di, ti, 3, 'raise')
                        except ValueError as exc:
                            assert 'source table' in str(exc)
                            rejected += 1
                        out = r._ls_interpol(density, temp, 1, .001, 301, w, p, di, ti, 3, 'clip')
                        assert np.all(np.isfinite(out)) and np.min(out) >= 0
                        cases += 1
    results['rosato_checked_cells'] = cases
    results['rosato_cells_with_negative_source_values'] = rejected
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
