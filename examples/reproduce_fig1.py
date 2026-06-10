#!/usr/bin/env python3
"""
reproduce_fig1.py
=================
Balmer-series Stark-Zeeman profiles for H at B = 100, 500, 1000 T.

Conditions (Ferri, Peyrusse & Calisti 2021, Fig. 1):
  Z = 1, Ne = 1e23 m^-3, Te = 5 eV, transverse observation.

Produces a single figure with two panels sharing the x-axis:
  top    — log intensity scale (reveals weak satellites and line wings)
  bottom — linear intensity scale (shows relative line strengths)
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.utils import wavelength_nm_to_energy_ev, energy_ev_to_wavelength_nm, RYDBERG_EV
from starkzee.static_profile import calculate_static_profile
from starkzee.atomic_hamiltonian import line_strength

# Standard Case B recombination line intensities relative to Hβ = 1
# (Osterbrock & Ferland 2006, T ≈ 10 000 K, ne → 0, optically thick Lyman series).
# Used to weight each profile so that ∫ weight × profile dE ∝ Case B emissivity,
# i.e. the integrated line flux of each Balmer transition carries the correct
# relative power.  Because the Stark/Zeeman widths grow with n, the peak-height
# ratios will be somewhat closer to one than the ideal Case B values, which is
# physically expected at Ne = 1e23 m^-3.
_CASEB = {3: 2.86, 4: 1.00, 5: 0.47, 6: 0.26, 7: 0.16}


def balmer_spectrum(B, Ne, Te, wavelengths_nm):
    """Transverse intensity summed over Hα–Hε on a common wavelength grid.

    Each line n→2 is weighted by _CASEB[n] / S_ul(n) so that the integrated
    line fluxes reproduce the Case B Balmer decrement.
    """
    energies = wavelength_nm_to_energy_ev(wavelengths_nm)
    total = np.zeros_like(energies)
    for n_u in [3, 4, 5, 6, 7]:
        print(f"    n={n_u}->2 ... ", end="", flush=True)
        S_n = line_strength(n_u, n_l=2, Z=1)
        weight = _CASEB[n_u] / S_n          # ∫ weight × profile dE ≈ _CASEB[n_u]
        pi, sp, sm = calculate_static_profile(
            n_u=n_u, n_l=2, Z=1, B=B, Ne_m3=Ne, Te_ev=Te,
            energies_ev=energies,
            num_f=20, num_mu=6,
            use_screening=True,
            quadratic_zeeman=True,
            frequency_dependent_width=False,
        )
        total += weight * (pi + 0.5 * (sp + sm))
        print("done.")
    return total


if __name__ == "__main__":
    Ne, Te = 1e23, 5.0
    B_list  = [100.0, 500.0, 1000.0]
    colors  = ["#4e79a7", "#f28e2b", "#e15759"]

    wavelengths_nm  = np.linspace(350.0, 720.0, 1500)
    wavelengths_ang = wavelengths_nm * 10.0

    print(f"Ne = {Ne:.1e} m^-3, Te = {Te} eV")

    spectra = {}
    for B in B_list:
        print(f"\nB = {int(B)} T:")
        spectra[B] = balmer_spectrum(B, Ne, Te, wavelengths_nm)

    # Normalize each spectrum to its own peak so relative shapes are visible
    normed = {B: s / s.max() for B, s in spectra.items()}

    # Approximate Balmer line centers (vacuum, Å)
    line_centers = {
        r"H$\alpha$":   6563,
        r"H$\beta$":    4861,
        r"H$\gamma$":   4340,
        r"H$\delta$":   4102,
        r"H$\epsilon$": 3970,
    }

    fig, axes = plt.subplots(
        3, 1, figsize=(11, 10), sharex=True,
        gridspec_kw={"hspace": 0.15},
    )

    for ax, B, color in zip(axes, B_list, colors):
        label = f"B = {int(B)} T"
        ax.plot(wavelengths_ang, normed[B], color=color, lw=1.6, label=label)

        ax.set_yscale("log")
        ax.set_ylim(1e-4, 3.0)
        ax.set_ylabel("Normalized intensity (log)", fontsize=11)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=10, loc="upper right")

        for name, wl in line_centers.items():
            ax.axvline(wl, color="gray", lw=0.6, ls="--", alpha=0.5)
            ax.text(wl, 2.0, name, ha="center", va="bottom", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.7, pad=1, edgecolor="none"))

    axes[-1].set_xlabel(r"Wavelength ($\AA$)", fontsize=12)
    axes[-1].set_xlim(3500, 7200)

    fig.suptitle(
        r"Balmer-series Stark-Zeeman profiles  —  H, $N_e = 10^{23}$ m$^{-3}$, $T_e = 5$ eV",
        fontsize=12, y=0.98,
    )

    out = os.path.join(os.path.dirname(__file__), "reproduce_fig1.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out}")
    plt.show()
