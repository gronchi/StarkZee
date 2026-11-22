"""Coulomb approximation: hydrogenic limit, symmetry, alkali resonance f-values, LS helper."""
import pytest

from starkzee.coulomb_approximation import coulomb_radial, effective_n, reduced_rydberg_cm
from starkzee.ls_radiator import (
    ActiveElectron, LSTerm, hydrogenic_radial, one_electron_reduced, reduced_element,
)

HARTREE_CM = 219474.6313632


@pytest.mark.parametrize("Z", [1, 2])
def test_integer_n_star_gives_hydrogenic_integrals_with_sign(Z):
    for n1 in range(1, 9):
        for n2 in range(1, 9):
            for l1 in range(n1):
                for l2 in (l1 - 1, l1 + 1):
                    if 0 <= l2 < n2:
                        assert coulomb_radial(float(n1), l1, float(n2), l2, float(Z)) == pytest.approx(
                            hydrogenic_radial(n1, l1, n2, l2, Z), rel=1e-12)


def test_symmetric():
    assert coulomb_radial(3.68, 1, 4.997, 2, 1.0) == coulomb_radial(4.997, 2, 3.68, 1, 1.0)


@pytest.mark.parametrize("name,E_p,limit,mass,f_ref", [
    # term-averaged np energies; f of the ns - np resonance multiplet
    ("Li 2s-2p", (2 * 14903.66 + 4 * 14904.00) / 6, 43487.15, 7.016, 0.7474),
    ("Na 3s-3p", (2 * 16956.17 + 4 * 16973.37) / 6, 41449.45, 22.99, 0.9604),
])
def test_alkali_resonance_f_values(name, E_p, limit, mass, f_ref):
    R = coulomb_radial(effective_n(0.0, limit, 1, mass), 0, effective_n(E_p, limit, 1, mass), 1, 1.0)
    f = (2 / 3) * E_p / HARTREE_CM * R**2           # s -> p, l_max/(2l+1) = 1
    assert f == pytest.approx(f_ref, rel=0.015)


def test_effective_n():
    R = reduced_rydberg_cm(1.007825)
    assert effective_n(R * (1 - 1 / 9), R, 1, 1.007825) == pytest.approx(3.0, rel=1e-14)
    with pytest.raises(ValueError):
        effective_n(10.0, 5.0)


def test_reduced_element_options():
    # hydrogen-like levels: the Coulomb approximation equals the hydrogenic option
    R = reduced_rydberg_cm(1.007825)
    d = LSTerm("3d", 2, 0.5, 1, {1.5: R * (1 - 1 / 9), 2.5: R * (1 - 1 / 9)})
    p = LSTerm("2p", 1, 0.5, -1, {0.5: R * (1 - 1 / 4), 1.5: R * (1 - 1 / 4)})
    e3d, e2p = ActiveElectron(3, 2, 0, R), ActiveElectron(2, 1, 0, R)
    hyd = reduced_element(d, e3d, p, e2p, mass_amu=1.007825, radial="hydrogenic")
    ca = reduced_element(d, e3d, p, e2p, mass_amu=1.007825)
    assert hyd == pytest.approx(one_electron_reduced(2, 2, 1, 1, 0, hydrogenic_radial(3, 2, 2, 1, 1)), rel=1e-15)
    assert ca == pytest.approx(hyd, rel=1e-12)
    with pytest.raises(ValueError):
        reduced_element(d, ActiveElectron(3, 2), p, ActiveElectron(2, 1))     # no series limit
