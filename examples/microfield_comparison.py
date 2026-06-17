"""
Simulate and reproduce Figure 3 of the ZEST paper.
Plots microfield distributions at a charged emitter for different Gamma and U (s).
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
import zest
import zest.microfield as mf

from starkzee.microfield import (
    hooper_distribution as sz_hooper,
    holtsmark_distribution as sz_holtsmark,
    potekhin_distribution as sz_potekhin,
)

if __name__ == '__main__':
    # Grid of beta = F/F_0 from 0.0 to 5.0
    beta = np.linspace(0.0, 5.0, 500)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Left Column: U = 1 (s = 1.0) ---
    # Curves for Gamma = 10, 5, 1, 0 (all with U = 1, charged = True)
    gammas_left = [10.0, 5.0, 1.0, 0.0]
    colors_left = ['C0', 'C1', 'C2', 'C3']
    
    for gamma, color in zip(gammas_left, colors_left):
        # Zest Potekhin
        p_zest = mf.get_microfield_distribution(beta, gamma=gamma, s=1.0, charged=True)
        ax1.plot(beta, p_zest, label=f"ZEST $\\Gamma = {int(gamma)}$", color=color, linewidth=2.0)
        
        # StarkZee Potekhin (natively implemented equivalent)
        p_sz = sz_potekhin(beta, gamma=gamma, s=1.0, charged=True)
        ax1.plot(beta, p_sz, color='black', linestyle='--', linewidth=1.2)
        
    # Hooper screened distribution (charged point, a=1.0)
    p_hooper_screened_zest = mf.hooper_distribution(beta, a=1.0, charged=True)
    ax1.plot(beta, p_hooper_screened_zest, label="ZEST Hooper ($a = 1.0$)", color='purple', linestyle=':', linewidth=2.0)
    
    # StarkZee Hooper screened distribution equivalent
    p_hooper_screened_sz = sz_hooper(beta, 1.0)
    ax1.plot(beta, p_hooper_screened_sz, color='black', linestyle='--', linewidth=1.2)
        
    ax1.set_title("Charged Emitter with Screening ($U = 1$)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("$\\beta = E/E_0$", fontsize=10)
    ax1.set_ylabel("$P(\\beta)$", fontsize=10)
    ax1.set_xlim(0.0, 5.0)
    ax1.set_ylim(0.0, 2.5)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=8.5, ncol=2)
    
    # --- Right Column: U = 0 (s = 0.0) ---
    # Mayer curve (Gamma = 10)
    p_mayer = mf.P_from_Q_grid(mf.Q_Mayer, beta, 10.0)
    ax2.plot(beta, p_mayer, label="Mayer ($\\Gamma = 10$)", color='k', linestyle=':', linewidth=2.0)
    
    # Curves for Gamma = 10, 5, 1, 0 (all with U = 0, charged = True)
    gammas_right = [10.0, 5.0, 1.0, 0.0]
    colors_right = ['C0', 'C1', 'C2', 'C3']
    
    for gamma, color in zip(gammas_right, colors_right):
        # Zest Potekhin
        p_zest = mf.get_microfield_distribution(beta, gamma=gamma, s=0.0, charged=True)
        ax2.plot(beta, p_zest, label=f"ZEST $\\Gamma = {int(gamma)}$", color=color, linewidth=2.0)
        
        # StarkZee Potekhin (natively implemented equivalent)
        p_sz = sz_potekhin(beta, gamma=gamma, s=0.0, charged=True)
        ax2.plot(beta, p_sz, color='black', linestyle='--', linewidth=1.2)
        
    # Holtzmark curve (Gamma = 0, U = 0, neutral / charged = False)
    p_holtz_zest = mf.get_microfield_distribution(beta, gamma=0.0, s=0.0, charged=False)
    ax2.plot(beta, p_holtz_zest, label="ZEST Holtzmark", color='gray', linestyle=':', linewidth=2.0)
    
    # StarkZee Holtsmark equivalent
    p_holtz_sz = sz_holtsmark(beta)
    ax2.plot(beta, p_holtz_sz, color='black', linestyle='--', linewidth=1.2)
    
    # Hooper unscreened distribution (charged point, a=0.0)
    p_hooper_unscreened_zest = mf.hooper_distribution(beta, a=0.0, charged=True)
    ax2.plot(beta, p_hooper_unscreened_zest, label="ZEST Hooper ($a = 0.0$)", color='purple', linestyle=':', linewidth=2.0)
    
    # StarkZee Hooper unscreened equivalent
    p_hooper_unscreened_sz = sz_hooper(beta, 0.0)
    ax2.plot(beta, p_hooper_unscreened_sz, color='black', linestyle='--', linewidth=1.2)
    
    
    ax2.set_title("Charged Emitter without Screening ($U = 0$)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("$\\beta = E/E_0$", fontsize=10)
    ax2.set_ylabel("$P(\\beta)$", fontsize=10)
    ax2.set_xlim(0.0, 5.0)
    ax2.set_ylim(0.0, 2.5)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=8.5, ncol=2)
    
    plt.suptitle("Figure 3: Charged Emitter Microfield Distributions (ZEST vs StarkZee)", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("examples/microfield_comparison_fig3.png", dpi=150)
    # plt.show()
    plt.close()
    
    # --- Figure 4: Differences (ZEST - StarkZee) ---
    fig2, (ad1, ad2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left Column Differences (U = 1)
    for gamma, color in zip(gammas_left, colors_left):
        p_zest = mf.get_microfield_distribution(beta, gamma=gamma, s=1.0, charged=True)
        p_sz = sz_potekhin(beta, gamma=gamma, s=1.0, charged=True)
        ad1.plot(beta, p_zest - p_sz, label=f"$\\Gamma = {int(gamma)}$", color=color, linewidth=2.0)
        
    p_hooper_screened_zest = mf.hooper_distribution(beta, a=1.0, charged=True)
    p_hooper_screened_sz = sz_hooper(beta, 1.0)
    ad1.plot(beta, p_hooper_screened_zest - p_hooper_screened_sz, label="Hooper ($a = 1.0$)", color='purple', linestyle='--', linewidth=2.0)
    
    ad1.set_title("Differences with Screening ($U = 1$)", fontsize=11, fontweight='bold')
    ad1.set_xlabel("$\\beta = E/E_0$", fontsize=10)
    ad1.set_ylabel("$\\Delta P(\\beta)$ (ZEST - StarkZee)", fontsize=10)
    ad1.set_xlim(0.0, 5.0)
    ad1.grid(True, linestyle='--', alpha=0.5)
    ad1.legend(fontsize=8.5)
    
    # Right Column Differences (U = 0)
    for gamma, color in zip(gammas_right, colors_right):
        p_zest = mf.get_microfield_distribution(beta, gamma=gamma, s=0.0, charged=True)
        p_sz = sz_potekhin(beta, gamma=gamma, s=0.0, charged=True)
        ad2.plot(beta, p_zest - p_sz, label=f"$\\Gamma = {int(gamma)}$", color=color, linewidth=2.0)
    
    p_holtz_zest = mf.get_microfield_distribution(beta, gamma=0.0, s=0.0, charged=False)
    p_holtz_sz = sz_holtsmark(beta)
    ad2.plot(beta, p_holtz_zest - p_holtz_sz, label="Holtzmark", color='gray', linestyle='--', linewidth=2.0)
    
    p_hooper_unscreened_zest = mf.hooper_distribution(beta, a=0.0, charged=True)
    p_hooper_unscreened_sz = sz_hooper(beta, 0.0)
    ad2.plot(beta, p_hooper_unscreened_zest - p_hooper_unscreened_sz, label="Hooper ($a = 0.0$)", color='purple', linestyle='--', linewidth=2.0)
    
    ad2.set_title("Differences without Screening ($U = 0$)", fontsize=11, fontweight='bold')
    ad2.set_xlabel("$\\beta = E/E_0$", fontsize=10)
    ad2.set_ylabel("$\\Delta P(\\beta)$ (ZEST - StarkZee)", fontsize=10)
    ad2.set_xlim(0.0, 5.0)
    ad2.grid(True, linestyle='--', alpha=0.5)
    ad2.legend(fontsize=8.5)
    
    plt.suptitle("Figure 3 Residuals: ZEST - StarkZee Differences", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("examples/microfield_residuals_fig3.png", dpi=150)
    # plt.show()
    plt.close()

    # --- Speed Test ---
    print("\n" + "="*65)
    print("MICROFIELD DISTRIBUTION SPEED TEST")
    print("="*65)
    import time
    
    # We will test on a grid of 500 beta values
    test_beta = np.linspace(0.01, 5.0, 500)
    n_runs = 20
    
    print(f"Timing over {n_runs} runs on a grid of {len(test_beta)} beta points:\n")
    
    tests = [
        ("StarkZee Holtsmark (exact)", lambda: sz_holtsmark(test_beta, method='exact')),
        ("StarkZee Holtsmark (vectorized)", lambda: sz_holtsmark(test_beta, method='vectorized')),
        ("StarkZee Hooper (vectorized, a=0.5)", lambda: sz_hooper(test_beta, a=0.5, method='vectorized')),
        ("StarkZee Hooper (exact, a=0.5)", lambda: sz_hooper(test_beta, a=0.5, method='exact')),
        ("StarkZee Potekhin (s=0.3, Gamma=1.0)", lambda: sz_potekhin(test_beta, gamma=1.0, s=0.3, charged=True)),
        ("StarkZee Potekhin (s=0.0, Gamma=1.0)", lambda: sz_potekhin(test_beta, gamma=1.0, s=0.0, charged=True)),
    ]
    
    # Check if zest is imported and available
    has_zest = False
    try:
        import zest
        import zest.microfield as mf
        has_zest = True
    except ImportError:
        pass

    if has_zest:
        tests.extend([
            ("ZEST Hooper (a=0.5)", lambda: mf.hooper_distribution(test_beta, a=0.5, charged=True)),
            ("ZEST Hooper (a=0.5)", lambda: mf.hooper_distribution(test_beta, a=0.5, charged=True)),
            ("ZEST Potekhin (s=0.3, Gamma=1.0)", lambda: mf.get_microfield_distribution(test_beta, gamma=1.0, s=0.3, charged=True)),
            ("ZEST Potekhin (s=0.0, Gamma=1.0)", lambda: mf.get_microfield_distribution(test_beta, gamma=1.0, s=0.0, charged=True)),
        ])
    
    print(f"{'Method/Function':<40} | {'Average Time (ms)':<20}")
    print("-" * 65)
    for name, fn in tests:
        # Warmup
        fn()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            fn()
        t1 = time.perf_counter()
        avg_time_ms = ((t1 - t0) / n_runs) * 1000.0
        print(f"{name:<40} | {avg_time_ms:>17.3f} ms")
    print("="*65)


