# Command Line Interface runner for starkzee Stark-Zeeman Line-Shape Model

import argparse
import numpy as np
import matplotlib.pyplot as plt

from starkzee.utils import wavelength_nm_to_energy_ev, energy_ev_to_wavelength_nm, RYDBERG_EV
from starkzee.static_profile import calculate_static_profile
from starkzee.ffm import calculate_ffm_profile
from starkzee.convolutions import apply_instrument_broadening, convolve_fft

def run_cli(Z, B, Ne, Te, Ti, A_emitter, inst_fwhm, use_ffm, output_file, plot_file):
    # Setup unperturbed Lyman-alpha energy (n_u=2 to n_l=1)
    E_upper = - (Z**2) * RYDBERG_EV / 4.0
    E_lower = - (Z**2) * RYDBERG_EV
    E0 = E_upper - E_lower
    
    grid_width = 0.04 * (Z / 6.0)**2
    # Ensure uniform wavelength grid for accurate FFT convolutions
    wavelengths_nm = np.linspace(energy_ev_to_wavelength_nm(E0 + grid_width), 
                                 energy_ev_to_wavelength_nm(E0 - grid_width), 500)
    energies_ev = wavelength_nm_to_energy_ev(wavelengths_nm)
    
    print(f"Running simulation for Z={Z}, B={B}T, Ne={Ne:.1e} m^-3, Te={Te} eV")
    if use_ffm:
        print("Using Frequency Fluctuation Model (Dynamical Profile)")
        pi, sig_plus, sig_minus = calculate_ffm_profile(
            n_u=2, n_l=1, Z=Z, B=B, Ne_m3=Ne, Te_ev=Te, Ti_ev=Ti, A_ion=A_emitter, energies_ev=energies_ev
        )
    else:
        print("Using Static Microfield Model")
        pi, sig_plus, sig_minus = calculate_static_profile(
            n_u=2, n_l=1, Z=Z, B=B, Ne_m3=Ne, Te_ev=Te, energies_ev=energies_ev
        )
        
    # Apply post-processing convolutions if requested
    if Ti > 0:
        from scipy.constants import m_p as _MP, c as _C, e as _E
        print(f"Applying thermal Doppler broadening for Ti={Ti} eV, emitter mass A={A_emitter}...")
        mc2_ev = A_emitter * _MP * _C**2 / _E
        v_th_over_c = np.sqrt(2.0 * Ti / mc2_ev)
        lambda0_nm = np.mean(wavelengths_nm)
        w_nm = lambda0_nm * v_th_over_c
        x = wavelengths_nm - lambda0_nm
        kernel = np.exp(-x**2 / w_nm**2)
        pi        = convolve_fft(wavelengths_nm, pi,        kernel)
        sig_plus  = convolve_fft(wavelengths_nm, sig_plus,  kernel)
        sig_minus = convolve_fft(wavelengths_nm, sig_minus, kernel)
        
    if inst_fwhm > 0:
        print(f"Applying instrumental slit broadening (FWHM = {inst_fwhm} nm)...")
        pi = apply_instrument_broadening(wavelengths_nm, pi, inst_fwhm)
        sig_plus = apply_instrument_broadening(wavelengths_nm, sig_plus, inst_fwhm)
        sig_minus = apply_instrument_broadening(wavelengths_nm, sig_minus, inst_fwhm)
        
    total = 0.5 * (sig_plus + sig_minus) + pi
    
    # Save text output
    if output_file:
        data_to_save = np.column_stack((wavelengths_nm, energies_ev, pi, sig_plus, sig_minus, total))
        np.savetxt(
            output_file, data_to_save, 
            header="Wavelength(nm) Energy(eV) Pi Sig+ Sig- Total_Transverse",
            fmt="%.6e"
        )
        print(f"Calculated profile saved to {output_file}")
        
    # Plot Matplotlib chart
    if plot_file:
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 6))
        
        plt.plot(wavelengths_nm, total, color='#00f2fe', label='Total Transverse Profile', linewidth=2.5)
        plt.plot(wavelengths_nm, pi, color='#ff007f', label='Pi components (q=0)', linestyle='--', alpha=0.8)
        plt.plot(wavelengths_nm, sig_plus, color='#4facfe', label='Sigma+ components (q=1)', alpha=0.8)
        plt.plot(wavelengths_nm, sig_minus, color='#10b981', label='Sigma- components (q=-1)', alpha=0.8)
        
        plt.title(f"Stark-Zeeman Lyman-alpha Profile (Z={Z}, B={B} T, Ne={Ne:.1e} m^-3)", fontsize=14, pad=15)
        plt.xlabel("Wavelength (nm)", fontsize=12)
        plt.ylabel("Intensity (arbitrary units)", fontsize=12)
        plt.grid(color='white', alpha=0.05)
        plt.legend(frameon=True, facecolor=(0.08, 0.1, 0.18, 0.8), edgecolor=(1.0, 1.0, 1.0, 0.1))
        
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"Profile plot saved to {plot_file}")

def main():
    parser = argparse.ArgumentParser(description="starkzee Stark-Zeeman Line-Shape Calculator")
    parser.add_argument("-Z", type=int, default=6, help="Nuclear charge (nuclear charge of radiator)")
    parser.add_argument("-B", type=float, default=500.0, help="Magnetic field in Tesla")
    parser.add_argument("--Ne", type=float, default=1e25, help="Electron density in m^-3")
    parser.add_argument("--Te", type=float, default=100.0, help="Electron temperature in eV")
    parser.add_argument("--Ti", type=float, default=100.0, help="Ion temperature in eV (0 to disable Doppler)")
    parser.add_argument("--mass", type=float, default=12.0, help="Atomic mass of emitter (e.g. 12 for Carbon)")
    parser.add_argument("--fwhm", type=float, default=0.0, help="Instrumental slit FWHM in nm")
    parser.add_argument("--static", action="store_true", help="Use static microfield model instead of FFM")
    parser.add_argument("-o", "--output", type=str, default="starkzee_profile.txt", help="Output text file path")
    parser.add_argument("-p", "--plot", type=str, default="starkzee_profile.png", help="Output plot image path")

    args = parser.parse_args()
    run_cli(args.Z, args.B, args.Ne, args.Te, args.Ti, args.mass, args.fwhm, not args.static, args.output, args.plot)


if __name__ == "__main__":
    main()
