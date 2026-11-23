"""One-valence-electron radiators: C IV (Li-like) n = 5 -> 4 from NIST levels."""
import warnings

import numpy as np
import pytest

from starkzee.single_valence import (
    air_wavelength_A, c_iv_5_4, load_single_valence, partial_rate, shell, term_reduced, total_rate,
)
from starkzee.static_profile import calculate_static_profile


@pytest.fixture(scope="module")
def data():
    return load_single_valence("c_iv")


def test_ritz_wavelengths_reproduce_the_observed_lines(data):
    T = data["terms"]
    for line in data["observed_lines"]:
        up, lo = (s.strip() for s in line["identification"].split(" (")[0].split(" - "))
        lam = air_wavelength_A(T[up].mean_energy_cm - T[lo].mean_energy_cm)
        assert lam == pytest.approx(line["air_A"], abs=line["uncertainty_A"])


@pytest.mark.parametrize("name, lower, upper, tol", [
    ("4p 2P - 5d 2D", "4p 2P", "5d 2D", 0.02),
    ("4d 2D - 5p 2P", "4d 2D", "5p 2P", 0.03),
])
def test_coulomb_approximation_against_nist(data, name, lower, upper, tol):
    lines = data["transitions"][name]["lines"]
    S_nist = sum(l["S_au"] for l in lines)
    assert 2 * term_reduced(data, upper, lower) ** 2 == pytest.approx(S_nist, rel=tol)      # (2S+1)|<L||d||L'>|^2
    # hydrogenic integrals are far off for the penetrating low-l terms
    assert abs(2 * term_reduced(data, upper, lower, "hydrogenic") ** 2 / S_nist - 1) > 0.2
    A_upper = {}
    for l in lines:
        A_upper[l["J_upper"]] = A_upper.get(l["J_upper"], 0.0) + l["A_s"]
    for A in A_upper.values():                           # LS: the same rate from every J of the upper term
        assert partial_rate(data, upper, lower) == pytest.approx(A, rel=tol + 0.01)


def test_high_l_terms_are_hydrogenic(data):
    # 5g - 4f: n* within 1e-4 of integer, CA -> hydrogenic, rate -> Z^4 A_H(5g-4f)
    assert term_reduced(data, "5g 2G", "4f 2F") == pytest.approx(term_reduced(data, "5g 2G", "4f 2F", "hydrogenic"),
                                                                rel=5e-4)
    assert total_rate(data, "5g 2G") == pytest.approx(4**4 * 4.2534e6, rel=2e-3)


def test_radiator_structure(data):
    r = c_iv_5_4(1.0)
    assert shell(data, 5) == ["5s 2S", "5p 2P", "5d 2D", "5f 2F", "5g 2G"]
    assert len(r.states[0]) == 2 * sum(2 * l + 1 for l in range(5)) == 50
    assert len(r.states[1]) == 2 * sum(2 * l + 1 for l in range(4)) == 32
    assert air_wavelength_A(r.line_energy_ev * 8065.543937) == pytest.approx(2529.98, abs=0.02)
    assert r.default_emitter_charge == 3 and r.collision_Z == 4
    assert r.collision_n == pytest.approx(5, abs=1e-3) and r.collision_n_lower == pytest.approx(4, abs=1e-3)
    # every Delta l = 1 pair inside each shell and between the shells
    assert len(r.couplings) == 4 + 3 + 7


def test_profile_is_normalized_and_broadened():
    r = c_iv_5_4(0.0)
    E0 = r.line_energy_ev
    grid = np.linspace(E0 - 0.06, E0 + 0.06, 1201)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pi, sp, sm = calculate_static_profile(5, 4, 4, 0.0, 2e24, 10.0, grid, num_f=12, num_mu=4,
                                              radiator=r, Ti_ev=10.0, electron_operator=True)
    tot = pi + 0.5 * (sp + sm)
    above = grid[tot >= tot.max() / 2]
    fwhm = above.max() - above.min()
    doppler_fwhm = E0 * np.sqrt(8 * np.log(2) * 10.0 / (12.0 * 931.494e6))
    assert fwhm > 3 * doppler_fwhm                       # Stark-dominated at 2e18 cm^-3
    assert abs(grid[np.argmax(tot)] - E0) < 0.1 * fwhm   # peak on the 5g - 4f component


# ---------------------------------------------------------------- Li-like isoelectronic sequence
LI_LIKE = (("c_iv", 4, 2529.98, "air"), ("n_v", 5, 1619.69, "vac"), ("o_vi", 6, 1124.83, "vac"))


@pytest.mark.parametrize("ion, Zc, lam, medium", LI_LIKE)
def test_li_like_5_4_line_and_structure(ion, Zc, lam, medium):
    from starkzee.single_valence import single_valence_transition
    d = load_single_valence(ion)
    assert d["core_charge"] == Zc and d["ion_charge"] == Zc - 1
    r = single_valence_transition(d, 5, 4, 0.0)
    sigma = r.line_energy_ev * 8065.543937
    got = air_wavelength_A(sigma) if medium == "air" else 1e8 / sigma
    assert got == pytest.approx(lam, abs=0.02)
    assert len(r.states[0]) == 50 and len(r.states[1]) == 32
    assert term_reduced(d, "5g 2G", "4f 2F") == pytest.approx(term_reduced(d, "5g 2G", "4f 2F", "hydrogenic"),
                                                             rel=5e-4)


def test_li_like_quantum_defects_and_widths_decrease_with_core_charge():
    from starkzee.coulomb_approximation import effective_n
    from starkzee.single_valence import n_v_5_4, o_vi_5_4
    defects = []
    for ion, *_ in LI_LIKE:
        d = load_single_valence(ion)
        defects.append(5 - effective_n(d["terms"]["5s 2S"].mean_energy_cm, d["limit_cm"], d["core_charge"],
                                       d["mass_amu"]))
    assert defects[0] > defects[1] > defects[2] > 0.1
    widths = []
    for make, Z in ((c_iv_5_4, 4), (n_v_5_4, 5), (o_vi_5_4, 6)):
        r = make(0.0)
        E0 = r.line_energy_ev
        grid = np.linspace(E0 - 0.06, E0 + 0.06, 1201)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tot = sum(calculate_static_profile(5, 4, Z, 0.0, 2e24, 10.0, grid, num_f=10, radiator=r,
                                               apply_doppler=False, electron_operator=True))
        widths.append(np.ptp(grid[tot >= tot.max() / 2]) / E0**2)       # FWHM in wavelength units
    assert widths[0] > widths[1] > widths[2]
