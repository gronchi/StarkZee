#!/usr/bin/env python3
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starkzee.utils import wavelength_nm_to_energy_ev, energy_ev_to_wavelength_nm, RYDBERG_EV
from starkzee.static_profile import calculate_static_profile


# ---------------------------------------------------------------------------
def generate_balmer_spectrum(B, Ne, Te, wavelengths_nm, quadratic_zeeman):
    """Combined Balmer series spectrum (Hα … Hε) on the given wavelength grid."""
    Z = 1
    # (n_upper, relative-scale)  — approximate oscillator-strength weighting
    balmer_lines = [
        (4, 0.35, "Hβ"),
    ]

    global_energies = wavelength_nm_to_energy_ev(wavelengths_nm)
    total = np.zeros_like(wavelengths_nm)

    for n_u, scale, name in balmer_lines:
        print(f"    {name} (n={n_u}→2) …", end=" ", flush=True)
        pi, sp, sm = calculate_static_profile(
            n_u=n_u, n_l=2, Z=Z, B=B, Ne_m3=Ne, Te_ev=Te,
            energies_ev=global_energies,
            num_f=20, num_mu=6,
            use_screening=True,
            quadratic_zeeman=quadratic_zeeman,
            frequency_dependent_width=False,
        )
        # Transverse observation: I = Iπ + ½(Iσ+ + Iσ−)
        total += scale * (pi + 0.5 * (sp + sm))
        print("done.")

    return total


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    Ne, Te = 1e21, 10.
    B_list = [1.0, 10.0, 50]

    # Global wavelength grid 3700 – 7200 Å (370 – 720 nm), 2000 points
    # (2000 pts gives ~1.75 Å ≈ 0.5 meV @ Hα — marginal for QZ wings)
    wavelengths_nm  = np.linspace(480, 490, 1000)
    wavelengths_ang = wavelengths_nm * 10.0

    print("reproduce_fig1: Balmer-series Stark-Zeeman profiles")
    print(f"  Ne = {Ne:.1e} m-3, Te = {Te} eV, Z = 1")
    print()

    spectra = {}
    for B in B_list:
        print(f"B = {int(B)} T:")
        print("  Without quadratic Zeeman:")
        nq = generate_balmer_spectrum(B, Ne, Te, wavelengths_nm, quadratic_zeeman=False)
        print("  With    quadratic Zeeman:")
        yq = generate_balmer_spectrum(B, Ne, Te, wavelengths_nm, quadratic_zeeman=True)
        spectra[B] = (nq, yq)
        print()

    print("All done. Plotting …")

    # ── Figure layout: 3 rows (one per B), each panel full Balmer range ──────
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(
        r"Stark-Zeeman Balmer-series lines (H, $N_e=10^{21}$ m$^{-3}$, $T_e=10$ eV)"
        "\n"
        r"Solid = with quadratic Zeeman, Dashed = without",
        fontsize=13)

    colors = ["tab:blue", "tab:orange", "tab:red"]
    B_labels = [f"B = {B_list[0]} T", f"B = {B_list[1]} T", f"B = {B_list[2]} T"]
    balmer_wl = {     # approximate vacuum wavelengths (Å)
        r"H$\alpha$": 6563,
        r"H$\beta$":  4861,
        r"H$\gamma$": 4340,
        r"H$\delta$": 4102,
        r"H$\epsilon$": 3970,
    }

    for ax, B, label, color in zip(axes, B_list, B_labels, colors):
        nq, yq = spectra[B]
        peak = max(yq.max(), nq.max(), 1.0)

        ax.plot(wavelengths_ang, nq / peak, "--", color=color,
                lw=1.3, alpha=0.75, label=f"{label}  (no QZ)")
        ax.plot(wavelengths_ang, yq / peak, "-",  color=color,
                lw=1.8, label=f"{label}  (with QZ)")

        ax.set_yscale("log")
        ax.set_ylim(1e-5, 3.0)
        ax.set_ylabel("Norm. intensity", fontsize=10)
        ax.legend(loc=4)
        ax.grid(True, which="both", alpha=0.3)

        # Annotate line names
        for name, wl in balmer_wl.items():
            ax.text(wl, 1.3, name, ha="center", va="bottom",
                    fontsize=9, color="black",
                    bbox=dict(facecolor="white", alpha=0.5, pad=1, edgecolor="none"))

    axes[-1].set_xlabel(r"Wavelength ($\AA$)", fontsize=12)
#    axes[-1].set_xlim(3800, 7200)

    plt.tight_layout()
    plt.show()
