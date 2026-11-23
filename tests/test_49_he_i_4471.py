"""He I 447.1 nm from NIST data: data consistency and the field-induced 4f component."""
import warnings

import numpy as np
import pytest

from starkzee.angular import wigner_6j
from starkzee.he_i import (
    LOWER_2, UPPER_4, forbidden_component_energy_ev, he_i_4471, load_he_i_triplets, nist_multiplet_reduced,
)
from starkzee.ls_radiator import reduced_element
from starkzee.static_profile import calculate_static_profile


@pytest.fixture(scope="module")
def data():
    return load_he_i_triplets()


def test_nist_line_strengths_follow_ls_coupling(data):
    red = nist_multiplet_reduced(data)
    for line in data["transitions"]["2p 3P - 4d 3D"]["lines"]:
        J, Jp = line["J_upper"], line["J_lower"]
        ls = (2 * J + 1) * (2 * Jp + 1) * wigner_6j(2, J, 1, Jp, 1, 1) ** 2 * red**2
        assert line["S_au"] == pytest.approx(ls, rel=1e-3)


def test_coulomb_approximation_against_nist(data):
    T, E, M = data["terms"], data["electrons"], data["mass_amu"]
    ca = reduced_element(T["4d 3D"], E["4d 3D"], T["2p 3P"], E["2p 3P"], 1.0, M)
    assert abs(ca) == pytest.approx(nist_multiplet_reduced(data), rel=0.02)


def test_radiator_structure(data):
    r = he_i_4471(1.0, data=data)
    dims = [sum(int(2 * J + 1) for J in data["terms"][k].energies_cm) for k in (UPPER_4, LOWER_2)[0]]
    assert len(r.states[0]) == sum(dims) == 48 and len(r.states[1]) == 12
    assert abs(r.couplings[("4d 3D", "2p 3P")]) == pytest.approx(nist_multiplet_reduced(data))
    assert r.line_energy_ev == pytest.approx(2.771986, abs=2e-6)          # 447.1 nm
    assert forbidden_component_energy_ev(data) - r.line_energy_ev == pytest.approx(7.40 / 8065.544, rel=2e-3)


def test_doppler_width_uses_the_radiator_mass():
    from scipy.constants import atomic_mass, c, e
    r = he_i_4471(0.0)
    E0 = r.line_energy_ev
    sigma = E0 * np.sqrt(20.0 / (r.mass_amu * atomic_mass * c**2 / e))     # Ti = 20 eV
    grid = np.linspace(E0 - 8 * sigma, E0 + 8 * sigma, 2001)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pi, sp, sm = calculate_static_profile(4, 2, 1, 0.0, 1e14, 5.0, grid, num_f=4, num_mu=2,
                                              radiator=r, Ti_ev=20.0, species="H")   # species ignored
    tot = pi + 0.5 * (sp + sm)
    # Doppler-dominated: second moment of the profile ~ sigma^2 (fine structure is << sigma)
    mean = np.trapezoid(grid * tot, grid) / np.trapezoid(tot, grid)
    rms = np.sqrt(np.trapezoid((grid - mean) ** 2 * tot, grid) / np.trapezoid(tot, grid))
    assert rms == pytest.approx(sigma, rel=0.05)


def _forbidden_ratio_and_peak(ne, B=0.0):
    r = he_i_4471(B)
    E_a, E_f = r.line_energy_ev, forbidden_component_energy_ev()
    grid = np.linspace(E_a - 3e-3, E_a + 4e-3, 1401)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pi, sp, sm = calculate_static_profile(4, 2, 1, B, ne, 5.0, grid, num_f=12, num_mu=4, radiator=r)
    tot = pi + 0.5 * (sp + sm)
    near = lambda E0, w: np.abs(grid - E0) < w                       # noqa: E731
    ratio = np.trapezoid(tot[near(E_f, 4e-4)], grid[near(E_f, 4e-4)]) / \
        np.trapezoid(tot[near(E_a, 4e-4)], grid[near(E_a, 4e-4)])
    window = near(E_f, 6e-4)
    return ratio, grid[window][np.argmax(tot[window])] - E_f


def test_forbidden_component_grows_and_moves_away_with_density():
    r19, _ = _forbidden_ratio_and_peak(1e19)
    r20, p20 = _forbidden_ratio_and_peak(1e20)
    r21, p21 = _forbidden_ratio_and_peak(1e21)
    r22, p22 = _forbidden_ratio_and_peak(1e22)
    assert r19 < r20 < r21 < r22 < 1
    assert r20 / r19 > 5 and r21 / r20 > 5                     # roughly ~ F^2 growth below saturation
    assert p22 > p21 > 0                                      # blue shift away from the allowed line
