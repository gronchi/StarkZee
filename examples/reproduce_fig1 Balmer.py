#!/usr/bin/env python3
"""
reproduce_fig1_Balmer.py
========================
Balmer-series Stark-Zeeman profiles for H, plotted individually.

Each line (H-alpha ... H-epsilon) is computed on its own local wavelength
grid of 1000 points spanning +/- 4 nm around the line centre.  This avoids
undersampling at low densities where the intrinsic width is much narrower
than a global grid spacing.

Two weighting modes are compared:
  - LTE Boltzmann populations  (orange)
  - Case B recombination       (blue)
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.utils import (wavelength_nm_to_energy_ev, energy_ev_to_wavelength_nm,
                             RYDBERG_EV, reduced_mass_rydberg_ev)
from starkzee.static_profile import calculate_static_profile
from starkzee.atomic_hamiltonian import line_strength


# ---------------------------------------------------------------------------
# Balmer line definitions
# ---------------------------------------------------------------------------
BALMER_LINES = [
    (3, "H-alpha",   r"H$\alpha$"),
    (4, "H-beta",    r"H$\beta$"),
    (5, "H-gamma",   r"H$\gamma$"),
    (6, "H-delta",   r"H$\delta$"),
    (7, "H-epsilon", r"H$\epsilon$"),
]

# Standard Case B recombination relative intensities (relative to H-beta = 1)
# (Osterbrock & Ferland 2006, T ~ 10,000 K).
_CASEB = {3: 2.86, 4: 1.00, 5: 0.47, 6: 0.26, 7: 0.16}


def compute_single_line(n_u, Z, B, Ne, Te, half_span_nm=0.5, n_pts=1000,
                        quadratic_zeeman=False):
    """Compute the Stark-Zeeman profile for a single n_u -> 2 Balmer line.

    Returns
    -------
    wavelengths_nm : ndarray, shape (n_pts,)
        Local wavelength grid centred on the gross-structure line centre.
    profile : ndarray, shape (n_pts,)
        Transverse-observation profile (pi + 0.5*(sigma+ + sigma-)).
    E0 : float
        Gross-structure transition energy [eV].
    """
    Ry = reduced_mass_rydberg_ev(Z, A=1)
    E0 = Z**2 * Ry * (1.0 / 2.0**2 - 1.0 / n_u**2)
    lam0_nm = energy_ev_to_wavelength_nm(E0)

    local_wl_nm = np.linspace(lam0_nm - half_span_nm, lam0_nm + half_span_nm, n_pts)
    local_energies = wavelength_nm_to_energy_ev(local_wl_nm)

    pi, sp, sm = calculate_static_profile(
        n_u=n_u, n_l=2, Z=Z, B=B, Ne_m3=Ne, Te_ev=Te,
        energies_ev=local_energies,
        num_f=20, num_mu=6,
        use_screening=True,
        quadratic_zeeman=quadratic_zeeman,
        frequency_dependent_width=False,
    )
    profile = pi + 0.5 * (sp + sm)
    return local_wl_nm, profile, E0


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    Ne, Te = 1e18, 5
    B = 3.0
    Z = 1
    Ry = reduced_mass_rydberg_ev(Z, A=1)

    print("Balmer-series Stark-Zeeman profiles (individual lines)")
    print(f"  Ne = {Ne:.1e} m-3, Te = {Te} eV, Z = {Z}, B = {B} T")
    print()

    # Compute each line
    results = {}
    for n_u, name_ascii, name_tex in BALMER_LINES:
        print(f"  {name_ascii} (n={n_u}->2) ...", end=" ", flush=True)
        wl_nm, profile, E0 = compute_single_line(n_u, Z, B, Ne, Te)

        S_n = line_strength(n_u, n_l=2, Z=Z)
        E_exc = Ry * (1.0 - 1.0 / n_u**2)

        # LTE weight: exp(-E_exc / Te) * E0^3
        # (g_n cancels between Boltzmann pop and Einstein A, see docstring)
        w_lte = np.exp(-E_exc / Te) * E0**3

        # Case B weight: CASEB[n] / S_n  (so integrated profile -> CASEB[n])
        w_caseb = _CASEB[n_u] / S_n

        results[n_u] = {
            'wl_nm': wl_nm,
            'profile': profile,
            'E0': E0,
            'S_n': S_n,
            'w_lte': w_lte,
            'w_caseb': w_caseb,
            'name_ascii': name_ascii,
            'name_tex': name_tex,
        }
        print("done.")

    print("\nAll done. Plotting ...")

    # ── Figure: 2 panels — top linear, bottom log ────────────────────────────
    fig, (ax_lin, ax_log) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"Balmer-series LTE profiles  --  "
        f"H, Ne = {Ne:.0e} m$^{{-3}}$, Te = {Te} eV, B = {B} T",
        fontsize=13, y=0.98)

    line_colors = ["#e15759", "#f28e2b", "#59a14f", "#4e79a7", "#b07aa1"]

    # Global normalisation: peak of the strongest weighted line
    max_val = max(r['w_lte'] * r['profile'].max() for r in results.values())

    for i, (n_u, name_ascii, name_tex) in enumerate(BALMER_LINES):
        r = results[n_u]
        wl_ang = r['wl_nm']
        y = r['w_lte'] * r['profile'] / max_val

        ax_lin.plot(wl_ang, y, "-", color=line_colors[i], lw=1.5, label=name_tex)
        ax_log.plot(wl_ang, y, "-", color=line_colors[i], lw=1.5, label=name_tex)

    # Linear panel
    ax_lin.set_ylabel("Normalized intensity", fontsize=11)
    ax_lin.set_ylim(bottom=0)
    ax_lin.legend(fontsize=10, loc="upper right", ncol=5)
    ax_lin.grid(True, alpha=0.3)
    ax_lin.set_title("Linear scale", fontsize=11)

    # Log panel
    ax_log.set_ylabel("Normalized intensity", fontsize=11)
    ax_log.set_yscale("log")
    ax_log.set_ylim(1e-5, 2.0)
    ax_log.legend(fontsize=10, loc="upper right", ncol=5)
    ax_log.grid(True, which="both", alpha=0.3)
    ax_log.set_title("Log scale", fontsize=11)
    ax_log.set_xlabel(r"Wavelength ($\AA$)", fontsize=12)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "reproduce_fig1_Balmer.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {out}")
    plt.show()
