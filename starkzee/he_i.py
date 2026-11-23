"""He I triplet lines from NIST data: the 447.1 nm line (2p 3P - 4d 3D) with its Stark-mixed n = 4 manifold.

The upper manifold holds the n = 4 triplet terms 4s 3S, 4p 3P, 4d 3D and 4f 3F,
which the plasma microfield mixes (4d-4f is only 7.4 cm^-1 apart, so the
"forbidden" 2p-4f component near 447.0 nm appears with density); the lower
manifold holds 2s 3S and 2p 3P. Level energies are NIST values
(``data/he_i_triplets_nist.json``); He I is very close to pure LS coupling (the
NIST J-resolved 2p-4d line strengths follow LS ratios to 1e-4).

Reduced E1 elements: Coulomb approximation (``radial="coulomb"``, default) or
hydrogenic (``radial="hydrogenic"``), with n* from the NIST energies and the
ionization limit. With ``scale_to_nist=True`` the magnitude of the 2p-4d
element is set from the NIST multiplet line strength (sign kept from the
approximation); the Coulomb approximation is within 1.3% of it.

Natural widths use the NIST 4d -> 2p rates only (decays to 3p and the rates of
4s, 4p, 4f are not in the data); they are negligible next to the Stark widths.
Electron broadening uses the generalized impact operator with the effective
quantum numbers of 4d and 2p and core charge 1.
"""
import json
import os

import numpy as np

from starkzee.coulomb_approximation import effective_n
from starkzee.ls_radiator import ActiveElectron, LSTerm, LSTransition, reduced_element
from starkzee.utils import wavenumber_cm_to_energy_ev

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "he_i_triplets_nist.json")

UPPER_4 = ("4s 3S", "4p 3P", "4d 3D", "4f 3F")
LOWER_2 = ("2s 3S", "2p 3P")


def load_he_i_triplets(path=_DATA):
    """NIST He I triplet data: dict with ``terms`` (LSTerm), ``electrons`` (ActiveElectron),
    ``limit_cm``, ``mass_amu`` and the raw ``transitions``."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    limit = raw["ionization_limit_cm"]["value"]
    terms, electrons = {}, {}
    rates = {}
    for line in raw["transitions"]["2p 3P - 4d 3D"]["lines"]:
        rates[float(line["J_upper"])] = rates.get(float(line["J_upper"]), 0.0) + line["A_s"]
    for label, t in raw["terms"].items():
        energies = {float(J): E for J, E in t["levels"].items()}
        terms[label] = LSTerm(label, t["L"], float(t["S"]), t["parity"], energies,
                              rates if label == "4d 3D" else 0.0)
        electrons[label] = ActiveElectron(t["n"], t["l"], 0, limit)
    return dict(terms=terms, electrons=electrons, limit_cm=limit, mass_amu=raw["mass_amu"],
                transitions=raw["transitions"], source=raw["source"])


def nist_multiplet_reduced(data, name="2p 3P - 4d 3D"):
    """|<L S||d||L' S>| [e a0] from the NIST multiplet line strength: S_mult = (2S+1) |<L||d||L'>|^2."""
    S_mult = sum(line["S_au"] for line in data["transitions"][name]["lines"])
    return float(np.sqrt(S_mult / 3.0))


def he_i_4471(B, radial="coulomb", scale_to_nist=True, data=None, **radiator_options):
    """LSTransition for He I 447.1 nm (2p 3P - 4d 3D) with the n = 4 triplet manifold.

    B is the magnetic field [T]; further keyword arguments go to LSTransition
    (e.g. ``electron_widths_ev``, ``upper_perturbers``). Returns the radiator; its
    ``line_energy_ev`` is the 4d - 2p term-centroid energy (also the Doppler/electron
    reference).
    """
    d = load_he_i_triplets() if data is None else data
    T, E, M = d["terms"], d["electrons"], d["mass_amu"]
    upper = [T[k] for k in UPPER_4]
    lower = [T[k] for k in LOWER_2]
    pairs = [("4s 3S", "4p 3P"), ("4p 3P", "4d 3D"), ("4d 3D", "4f 3F"),     # Stark mixing, n = 4
             ("2s 3S", "2p 3P"),                                              # lower manifold
             ("4s 3S", "2p 3P"), ("4d 3D", "2p 3P"), ("4p 3P", "2s 3S")]      # radiative
    couplings = {(a, b): reduced_element(T[a], E[a], T[b], E[b], Z_core=1.0, mass_amu=M, radial=radial)
                 for a, b in pairs}
    if scale_to_nist:
        value = couplings[("4d 3D", "2p 3P")]
        couplings[("4d 3D", "2p 3P")] = np.copysign(nist_multiplet_reduced(d), value)
    line_ev = float(wavenumber_cm_to_energy_ev(T["4d 3D"].mean_energy_cm - T["2p 3P"].mean_energy_cm))
    options = dict(mass_amu=M, emitter_charge=0.0, reference_energy_ev=line_ev,
                   collision_n=effective_n(T["4d 3D"].mean_energy_cm, d["limit_cm"], 1.0, M),
                   collision_n_lower=effective_n(T["2p 3P"].mean_energy_cm, d["limit_cm"], 1.0, M),
                   collision_Z=1.0)
    options.update(radiator_options)
    radiator = LSTransition(upper, lower, couplings, B, **options)
    radiator.line_energy_ev = line_ev
    radiator.couplings = couplings
    return radiator


def forbidden_component_energy_ev(data=None):
    """2p 3P - 4f 3F term-centroid energy [eV] (the field-induced component near 447.0 nm)."""
    d = load_he_i_triplets() if data is None else data
    T = d["terms"]
    return float(wavenumber_cm_to_energy_ev(T["4f 3F"].mean_energy_cm - T["2p 3P"].mean_energy_cm))
