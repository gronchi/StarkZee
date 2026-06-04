import numpy as np
import matplotlib.pyplot as plt
from starkzee.line_profile import LineProfile
import starkzee.models as models

# 1. Setup parameters
Ne = 1e19      # m^-3
T = 0.5        # eV (Ti = Te)
B = 3.0        # T
species = 'D'  # Deuterium

# Transitions to plot: (n_upper, n_lower, name, half_width_nm)
transitions = [
    (3, 2, r"$\mathrm{D}_\alpha$", 0.5),
    (5, 2, r"$\mathrm{D}_\gamma$", 1.5)
]

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

for ax, (n_u, n_l, name, hw) in zip(axes, transitions):
    # Initialize StarkZee — Ti_ev passed here so Doppler is folded into the Voigt accumulation
    lp = LineProfile(n_u=n_u, n_l=n_l, B=B, Ne_m3=Ne, Te_ev=T, Ti_ev=T, species=species)

    # Generate wavelength grid (air wavelengths matching StarkZee's centroid)
    wl = np.linspace(lp.E0_wavelength_air_nm - hw, lp.E0_wavelength_air_nm + hw, 1000)

    # StarkZee profile — Doppler already included via Voigt
    wl_vac = np.linspace(lp.E0_wavelength_nm - hw, lp.E0_wavelength_nm + hw, 10000)
    lp.compute_profile(wl_vac, grid_type='wavelength_nm')
    sz_prof = lp.profile

    # Plot StarkZee (using its computed air wavelengths)
    ax.plot(lp.wavelengths_air_nm, sz_prof / sz_prof.max(), 'k--', lw=2.0, label='StarkZee', zorder=10)
    
    # Plot each comparison model
    for label, model_func in [
        ('Voigt', models.voigt),
        ('Stehle', models.stehle),
        ('Stehle (param)', models.stehle_param),
        ('Lomanowski', models.lomanowski),
        ('Rosato', models.rosato)
    ]:
        try:
            prof = model_func(wl, n_u, n_l, B, Ne, T, T, species=species)
            ax.plot(wl, prof / prof.max(), label=label, alpha=0.85)
        except Exception as e:
            print(f"Model {label} failed: {e}")
            
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized Intensity")
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, ls=':', alpha=0.5)

plt.tight_layout()
output_img = 'examples/minimal_comparison.png'
plt.savefig(output_img, dpi=150)
print(f"Saved: {output_img}")
plt.show()
