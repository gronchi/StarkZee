"""Radiators for one valence electron outside closed shells (Li-like, Na-like ions, alkali atoms).

The terms are 2S+1 = 2 terms ``nl 2L`` (L = l, parent 1S) read from a level
table (``data/<ion>_nist.json``, NIST values). A line n_u -> n_l is built as an
:class:`~starkzee.ls_radiator.LSTransition` with

* the upper manifold = all nl terms of shell n_u, the lower = all of shell n_l,
  Stark-mixed through the Delta l = 1 couplings inside each shell;
* radiative dipoles between every Delta l = 1 pair of the two shells;
* reduced elements from the Coulomb approximation (``radial="coulomb"``) with
  n* from the NIST energies and the series limit, or hydrogenic integrals with
  the core charge (``radial="hydrogenic"``);
* natural widths from the E1 rates (same radial model) of every term to all
  lower terms in the table;
* the generalized electron impact operator with the effective n of the most
  hydrogenic (largest l) term of each shell and the core charge;
* the diamagnetic (quadratic Zeeman) term inside each shell, with <r^2> from the
  same radial model (``diamagnetic=True``). The field-free radial functions are
  kept: the mixing of other shells by the field is ~0.2 gamma^2 n^7 / Z^4
  (gamma = B / 2.35e5 T), 3e-4 for C IV n = 5 at 500 T.

Optional perturbing shells add their Delta n != 0 couplings to the collision
self terms only.
"""
import json
import os
from fractions import Fraction

import numpy as np
from scipy.constants import c as _C, epsilon_0 as _EPS0, hbar as _HBAR, e as _E
from scipy.constants import physical_constants as _pc

from starkzee.coulomb_approximation import coulomb_r2, effective_n
from starkzee.ls_radiator import ActiveElectron, LSTerm, LSTransition, one_electron_r2_reduced, reduced_element
from starkzee.radiator import radial_r2_element
from starkzee.utils import core_mass_factors, vacuum_to_air_wavelength_nm, wavenumber_cm_to_energy_ev

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_A0 = _pc["Bohr radius"][0]
_L_LETTER = "SPDFGHIKLMNOQ"
# A [s^-1] = _RATE * sigma[cm^-1]^3 * |<L||r||L'>|^2 [a0^2] / (2L+1)
_RATE = (2 * np.pi * 100 * _C) ** 3 * _E**2 * _A0**2 / (3 * np.pi * _EPS0 * _HBAR * _C**3)


def load_single_valence(name_or_path):
    """Level table of a one-valence-electron ion: a bundled name (e.g. ``"c_iv"``) or a JSON path.

    Returns a dict with ``terms`` and ``electrons`` (by label), ``limit_cm``,
    ``mass_amu``, ``core_charge``, ``ion_charge`` and the raw ``transitions``.
    """
    path = name_or_path if name_or_path.endswith(".json") else os.path.join(_DATA_DIR, f"{name_or_path}_nist.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    limit = raw["ionization_limit_cm"]["value"]
    terms, electrons = {}, {}
    for label, t in raw["terms"].items():
        energies = {float(Fraction(J)): E for J, E in t["levels"].items()}
        parity = 1 if t["l"] % 2 == 0 else -1
        terms[label] = LSTerm(label, t["l"], 0.5, parity, energies)
        electrons[label] = ActiveElectron(t["n"], t["l"], 0, limit)
    return dict(terms=terms, electrons=electrons, limit_cm=limit, mass_amu=raw["mass_amu"],
                core_charge=float(raw["core_charge"]), ion_charge=float(raw["ion_charge"]),
                transitions=raw.get("transitions", {}), observed_lines=raw.get("observed_lines", []),
                ion=raw.get("ion", ""), source=raw["source"])


def shell(data, n):
    """Labels of the terms of shell n, by increasing l."""
    return sorted((k for k, e in data["electrons"].items() if e.n == n), key=lambda k: data["electrons"][k].l)


def term_reduced(data, a, b, radial="coulomb"):
    """Signed <a||d||b> [e a0] between two terms of the table (Delta l = 1)."""
    T, E = data["terms"], data["electrons"]
    return reduced_element(T[a], E[a], T[b], E[b], Z_core=data["core_charge"],
                           mass_amu=data["mass_amu"], radial=radial)


def term_r2(data, a, b, radial="coulomb"):
    """Radial ``<a| r^2 |b>`` [a0^2] between two terms of one shell (``|Delta l|`` = 0 or 2), in the reduced Bohr radius."""
    T, E = data["terms"], data["electrons"]
    ea, eb = E[a], E[b]
    scale = core_mass_factors(data["mass_amu"])[0] ** -2      # a_mu^2 / a0^2
    if radial == "coulomb":
        Z, M, lim = data["core_charge"], data["mass_amu"], data["limit_cm"]
        return scale * coulomb_r2(effective_n(T[a].mean_energy_cm, lim, Z, M), ea.l,
                                  effective_n(T[b].mean_energy_cm, lim, Z, M), eb.l, Z)
    if radial == "hydrogenic":
        if ea.n != eb.n:
            raise ValueError("hydrogenic r^2 elements are implemented inside one shell")
        return scale * radial_r2_element(ea.n, ea.l, eb.l, data["core_charge"])
    raise ValueError("radial must be 'coulomb' or 'hydrogenic'")


def shell_diamagnetic(data, labels, radial="coulomb"):
    """Term elements {(a, b): {k: <a||r^2 C^k||b>}} inside a shell (input of LSTransition)."""
    E = data["electrons"]
    out = {}
    for i, a in enumerate(labels):
        for b in labels[i:]:
            la, lb = E[a].l, E[b].l
            if abs(la - lb) not in (0, 2):
                continue
            r2 = term_r2(data, a, b, radial)
            ranks = (0, 2) if la == lb else (2,)
            out[(a, b)] = {k: float(one_electron_r2_reduced(la, la, lb, lb, 0, k, r2)) for k in ranks}
    return out


def partial_rate(data, upper, lower, radial="coulomb"):
    """E1 rate [s^-1] of term ``upper`` (each J) to all J of term ``lower`` (LS coupling)."""
    T = data["terms"]
    sigma = T[upper].mean_energy_cm - T[lower].mean_energy_cm
    if sigma <= 0:
        return 0.0
    red = term_reduced(data, upper, lower, radial)
    return float(_RATE * sigma**3 * red**2 / (2 * T[upper].L + 1))


def total_rate(data, label, radial="coulomb"):
    """Total E1 decay rate [s^-1] of a term to all lower terms of the table."""
    E = data["electrons"]
    return sum(partial_rate(data, label, k, radial) for k in data["terms"]
               if abs(E[k].l - E[label].l) == 1 and data["terms"][k].mean_energy_cm < data["terms"][label].mean_energy_cm)


def _dl1_pairs(a_labels, b_labels, data):
    E = data["electrons"]
    return [(a, b) for a in a_labels for b in b_labels if abs(E[a].l - E[b].l) == 1]


def single_valence_transition(data, n_upper, n_lower, B, radial="coulomb", natural_widths=True,
                              upper_perturber_shells=(), lower_perturber_shells=(), diamagnetic=True,
                              **radiator_options):
    """LSTransition for the n_upper -> n_lower line of a one-valence-electron ion.

    ``data`` is a table from :func:`load_single_valence` (or a bundled name).
    ``diamagnetic`` adds the quadratic Zeeman term inside both shells (default).
    Further keyword arguments go to :class:`~starkzee.ls_radiator.LSTransition`.
    The radiator gets ``line_energy_ev`` (centroid of the strongest, largest-l
    component), ``couplings`` and ``ion_data`` (the table).
    """
    d = load_single_valence(data) if isinstance(data, str) else data
    up, lo = shell(d, n_upper), shell(d, n_lower)
    up_p = [k for n in upper_perturber_shells for k in shell(d, n)]
    lo_p = [k for n in lower_perturber_shells for k in shell(d, n)]
    pairs = []
    for labels in (up, lo):                                           # Stark mixing inside each shell
        pairs += [(a, b) for a, b in zip(labels[:-1], labels[1:])]
    pairs += _dl1_pairs(up, lo, d)                                     # radiative
    pairs += _dl1_pairs(up, up_p, d) + _dl1_pairs(lo, lo_p, d)         # collision perturbers
    couplings = {(a, b): term_reduced(d, a, b, radial) for a, b in pairs}

    def term(label):
        t = d["terms"][label]
        if not natural_widths:
            return t
        return LSTerm(t.label, t.L, t.S, t.parity, t.energies_cm, total_rate(d, label, radial))

    T = d["terms"]
    top_u, top_l = up[-1], lo[-1]
    # strongest component: largest l of the lower shell, fed from l + 1 of the upper shell
    feed = next(k for k in reversed(up) if d["electrons"][k].l == d["electrons"][top_l].l + 1) \
        if any(d["electrons"][k].l == d["electrons"][top_l].l + 1 for k in up) else top_u
    line_ev = float(wavenumber_cm_to_energy_ev(T[feed].mean_energy_cm - T[top_l].mean_energy_cm))
    options = dict(mass_amu=d["mass_amu"], emitter_charge=d["ion_charge"], reference_energy_ev=line_ev,
                   collision_n=effective_n(T[top_u].mean_energy_cm, d["limit_cm"], d["core_charge"], d["mass_amu"]),
                   collision_n_lower=effective_n(T[top_l].mean_energy_cm, d["limit_cm"], d["core_charge"],
                                                 d["mass_amu"]),
                   collision_Z=d["core_charge"])
    if diamagnetic:
        options["diamagnetic"] = {**shell_diamagnetic(d, up, radial), **shell_diamagnetic(d, lo, radial)}
    options.update(radiator_options)
    radiator = LSTransition([term(k) for k in up], [term(k) for k in lo], couplings, B,
                            upper_perturbers=[term(k) for k in up_p], lower_perturbers=[term(k) for k in lo_p],
                            **options)
    radiator.line_energy_ev = line_ev
    radiator.couplings = couplings
    radiator.ion_data = d
    return radiator


def c_iv_5_4(B, **options):
    """C IV n = 5 -> 4 line (strongest component 5g - 4f, 253.0 nm air) from NIST levels."""
    return single_valence_transition(load_single_valence("c_iv"), 5, 4, B, **options)


def n_v_5_4(B, **options):
    """N V n = 5 -> 4 line (5g - 4f, 162.0 nm vacuum) from NIST levels."""
    return single_valence_transition(load_single_valence("n_v"), 5, 4, B, **options)


def o_vi_5_4(B, **options):
    """O VI n = 5 -> 4 line (5g - 4f, 112.5 nm vacuum) from NIST levels (n >= 4 energies to 1-10 cm^-1)."""
    return single_valence_transition(load_single_valence("o_vi"), 5, 4, B, **options)


def air_wavelength_A(sigma_cm):
    """Air wavelength [Angstrom] for a vacuum wavenumber [cm^-1] (:func:`~starkzee.utils.vacuum_to_air_wavelength_nm`)."""
    return 10.0 * vacuum_to_air_wavelength_nm(1e7 / np.asarray(sigma_cm, dtype=float))
