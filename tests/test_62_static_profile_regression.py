"""Regression: calculate_static_profile against frozen reference profiles.

The reference file was written by the implementation before the fused kernel and
batched eigh (one Lorentzian expression per polarization, eigh per quadrature
point). The cases cover every branch of the quadrature loop: frequency-dependent
and fixed widths, with and without Doppler (FFT and direct-Voigt paths), state-
resolved and shell-average natural widths, the electron operator, no electrons,
B = 0, the PPP interference operator (full and M-block) and multi-electron
radiators. Speed-ups must reproduce it to rounding.

Regenerate (only for an intended change of the physics):
    python tests/test_62_static_profile_regression.py
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from starkzee.he_i import he_i_4471                                      # noqa: E402
from starkzee.single_valence import c_iv_5_4                             # noqa: E402
from starkzee.static_profile import calculate_static_profile, line_reference_energy  # noqa: E402

REFERENCE = os.path.join(os.path.dirname(__file__), "data", "static_profile_reference.npz")
TOL = 1e-12                    # of the peak; the measured differences are ~1e-15
QUAD = dict(num_f=8, num_mu=3)


def _grid(n_u, n_l, Z, half, n=300):
    E0 = line_reference_energy(n_u, n_l, Z)
    return np.linspace(E0 - half, E0 + half, n)


def _hydrogen(n_u=3, n_l=2, B=5.0, Ne=1e20, half=3e-3, **kw):
    return calculate_static_profile(n_u, n_l, 1, B, Ne, 5.0, _grid(n_u, n_l, 1, half), **QUAD, **kw)


def _he_i():
    r = he_i_4471(2.0)
    grid = np.linspace(r.line_energy_ev - 3e-3, r.line_energy_ev + 4e-3, 300)
    return calculate_static_profile(4, 2, 1, 2.0, 1e21, 5.0, grid, radiator=r, Ti_ev=2.0, **QUAD)


def _c_iv():
    r = c_iv_5_4(300.0)
    grid = np.linspace(r.line_energy_ev - 0.05, r.line_energy_ev + 0.05, 300)
    return calculate_static_profile(5, 4, 4, 300.0, 2e24, 10.0, grid, radiator=r, Ti_ev=10.0,
                                    electron_operator=True, **QUAD)


CASES = {
    "doppler_frequency_dependent": lambda: _hydrogen(Ti_ev=5.0),
    "no_doppler": lambda: _hydrogen(),
    "voigt_loop_fixed_width": lambda: _hydrogen(Ti_ev=5.0, frequency_dependent_width=False),
    "fixed_width_no_doppler": lambda: _hydrogen(frequency_dependent_width=False),
    "shell_average_natural": lambda: _hydrogen(Ti_ev=5.0, natural_width_mode="shell_average"),
    "electron_operator": lambda: _hydrogen(Ti_ev=5.0, electron_operator=True),
    "electron_operator_fixed": lambda: _hydrogen(electron_operator=True, frequency_dependent_width=False),
    "no_electrons": lambda: _hydrogen(Ti_ev=5.0, electron_model="none"),
    "zero_field": lambda: _hydrogen(B=0.0, Ti_ev=5.0),
    "interference": lambda: _hydrogen(B=2.0, frequency_dependent_width=False, electron_interference=True),
    "interference_m_blocks": lambda: _hydrogen(B=0.0, frequency_dependent_width=False,
                                               electron_interference=True),
    "balmer_delta": lambda: _hydrogen(6, 2, B=2.0, half=1e-2, Ti_ev=5.0),
    "lyman_alpha_zest": lambda: _hydrogen(2, 1, B=3.0, Ne=1e21, half=5e-3, Ti_ev=5.0, electron_model="zest"),
    "c_vi_8_7": lambda: calculate_static_profile(8, 7, 6, 3.0, 1e20, 50.0, _grid(8, 7, 6, 2e-2), Ti_ev=50.0,
                                                 use_empirical_data=False, **QUAD),
    "he_i_4471": _he_i,
    "c_iv_5_4_operator": _c_iv,
}


def _evaluate(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.array(CASES[name]())


@pytest.fixture(scope="module")
def reference():
    return np.load(REFERENCE)


@pytest.mark.parametrize("name", CASES)
def test_static_profile_matches_reference(reference, name):
    ref = reference[name]
    out = _evaluate(name)
    assert out.shape == ref.shape
    assert ref.max() > 0
    assert np.max(np.abs(out - ref)) < TOL * ref.max()


@pytest.mark.parametrize("name", ["doppler_frequency_dependent", "electron_operator", "no_electrons",
                                  "c_vi_8_7", "he_i_4471"])
def test_numpy_fallback_matches_reference(reference, monkeypatch, name):
    """Without numexpr the in-place numpy kernel gives the same profiles."""
    from starkzee import static_profile
    monkeypatch.setattr(static_profile, "_numexpr", None)
    ref = reference[name]
    assert np.max(np.abs(_evaluate(name) - ref)) < TOL * ref.max()


if __name__ == "__main__":
    os.makedirs(os.path.dirname(REFERENCE), exist_ok=True)
    np.savez_compressed(REFERENCE, **{name: _evaluate(name) for name in CASES})
    print("wrote", REFERENCE)
