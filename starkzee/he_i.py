"""He I lines from NIST data: the 447.1 nm line (2p 3P - 4d 3D) with its Stark-mixed n = 4 manifold.

The upper manifold holds the n = 4 triplet terms 4s 3S, 4p 3P, 4d 3D and 4f 3F,
which the plasma microfield mixes (4d-4f is only 7.4 cm^-1 apart, so the
"forbidden" 2p-4f component near 447.0 nm appears with density), and with
``singlets=True`` the n = 4 singlet terms 4s 1S, 4p 1P, 4d 1D and 4f 1F; the lower
manifold holds 2s 3S and 2p 3P. Level energies are NIST values
(``data/he_i_triplets_nist.json``).

Singlet-triplet mixing. The NIST 4f 1F3 level lies 0.024 cm^-1 from 4f 3F3 and
4d 1D2 1.97 cm^-1 from 4d 3D2; spin-orbit couples each J = L pair, which makes
the two 4f J = 3 levels nearly equal singlet-triplet mixtures (and 4d J = 2 a
0.5 % one). The Zeeman term conserves S, so the triplet M_S = 0 sublevels stay
degenerate with the singlet ones and the mixing persists at any field. The
coupling of each pair is fitted to the NIST levels by
:func:`singlet_triplet_coupling`; singlets enter the 2p 3P line only through it.

With equally populated upper sublevels (the assumption of every StarkZee
profile) the mixing only shares the 3F strength between two levels 0.024 cm^-1
apart, so the 447.1 nm profile, its forbidden component included, changes by
less than 0.3 % (Ne = 1e21-1e22 m^-3, B = 0-4 T). The default is therefore the
triplet manifold (48 states instead of 64); ``singlets=True`` adds the singlets
for level-resolved work or non-statistical populations.

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
SINGLETS_4 = ("4s 1S", "4p 1P", "4d 1D", "4f 1F")
LOWER_2 = ("2s 3S", "2p 3P")


def singlet_triplet_coupling(triplet, singlet):
    """Unmixed energies and coupling of the J = L pair of a 1snl singlet and triplet term.

    triplet, singlet: LSTerm with the observed (NIST) levels; L >= 1 and the
    triplet has its three J levels. The triplet levels J = L -+ 1 do not mix;
    the unmixed J = L triplet energy follows from them with the Lande interval
    rule E(J) = E_T + (zeta/4)[J(J+1) - L(L+1) - 2]. The two observed J = L
    levels then fix the unmixed singlet energy and the coupling:

        E_SS = E_1 + E_2 - E_TT,   V^2 = -(E_1 - E_TT)(E_2 - E_TT).

    Returns (E_TT, E_SS, V) [cm^-1], with V >= 0 (the sign is a phase convention
    of the pair; the 2p 3P line depends on it only through interference between
    the singlet and triplet Stark paths). The 2x2 eigenvalues reproduce both
    observed levels exactly; the interval rule is an approximation (He I also
    has spin-spin and spin-other-orbit terms).
    """
    L = triplet.L
    if L < 1 or singlet.L != L or set(triplet.energies_cm) != {L - 1.0, float(L), L + 1.0}:
        raise ValueError("needs a triplet with J = L-1, L, L+1 and a singlet of the same L >= 1")
    E = triplet.energies_cm
    zeta = 4.0 * (E[L + 1.0] - E[L - 1.0]) / (4 * L + 2)
    E_T = E[L + 1.0] - zeta * 2 * L / 4.0
    E_TT = E_T - zeta / 2.0
    E1, E2 = E[float(L)], singlet.energies_cm[float(L)]
    V2 = -(E1 - E_TT) * (E2 - E_TT)
    if V2 < 0:
        raise ValueError("the interval-rule triplet energy is not between the observed J = L levels")
    return E_TT, E1 + E2 - E_TT, float(np.sqrt(V2))


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
    """``|<L S||d||L' S>|`` [e a0] from the NIST multiplet line strength: ``S_mult = (2S+1) |<L||d||L'>|^2``."""
    S_mult = sum(line["S_au"] for line in data["transitions"][name]["lines"])
    return float(np.sqrt(S_mult / 3.0))


def he_i_4471(B, radial="coulomb", scale_to_nist=True, data=None, singlets=False, **radiator_options):
    """LSTransition for He I 447.1 nm (2p 3P - 4d 3D) with the n = 4 manifold.

    B is the magnetic field [T]; ``singlets`` adds the n = 4 singlet terms with
    the singlet-triplet coupling of the 4d and 4f J = L pairs (see the module
    notes; it needs the exact Zeeman treatment). Further keyword arguments go to
    LSTransition (e.g. ``electron_widths_ev``, ``upper_perturbers``). Returns the
    radiator; its ``line_energy_ev`` is the 4d - 2p term-centroid energy (also
    the Doppler/electron reference).
    """
    d = load_he_i_triplets() if data is None else data
    T, E, M = d["terms"], d["electrons"], d["mass_amu"]
    upper = [T[k] for k in UPPER_4]
    lower = [T[k] for k in LOWER_2]
    pairs = [("4s 3S", "4p 3P"), ("4p 3P", "4d 3D"), ("4d 3D", "4f 3F"),     # Stark mixing, n = 4
             ("2s 3S", "2p 3P"),                                              # lower manifold
             ("4s 3S", "2p 3P"), ("4d 3D", "2p 3P"), ("4p 3P", "2s 3S")]      # radiative
    term_couplings = {}
    if singlets:
        upper = upper + [T[k] for k in SINGLETS_4]
        pairs += [("4s 1S", "4p 1P"), ("4p 1P", "4d 1D"), ("4d 1D", "4f 1F")]   # singlet Stark mixing
        mixed = {}
        for t3, t1 in (("4d 3D", "4d 1D"), ("4f 3F", "4f 1F")):
            E_TT, E_SS, V = singlet_triplet_coupling(T[t3], T[t1])
            L = float(T[t3].L)
            mixed[t3] = LSTerm(t3, T[t3].L, T[t3].S, T[t3].parity, {**T[t3].energies_cm, L: E_TT},
                               T[t3].decay_rates)
            mixed[t1] = LSTerm(t1, T[t1].L, T[t1].S, T[t1].parity, {L: E_SS}, T[t1].decay_rates)
            term_couplings[(t3, t1)] = {L: V}
        upper = [mixed.get(t.label, t) for t in upper]
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
    if term_couplings:
        options["term_couplings"] = term_couplings
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
