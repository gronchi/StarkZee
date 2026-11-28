"""Time calculate_static_profile on the regression cases and check them against the reference.

Run:  python scripts/benchmark_static_profile.py [--numpy] [--fft [tol]]

--numpy disables numexpr (in-place numpy kernel); --fft uses method='fft' (default
fft_tol 1e-8; the interference cases stay exact). Each case is evaluated once to
warm the caches, then timed; the difference from tests/data/static_profile_reference.npz
is reported relative to the peak.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from starkzee import static_profile                                       # noqa: E402
import test_62_static_profile_regression as cases                         # noqa: E402

if "--numpy" in sys.argv:
    static_profile._numexpr = None
if "--fft" in sys.argv:
    k = sys.argv.index("--fft")
    tol = float(sys.argv[k + 1]) if len(sys.argv) > k + 1 else 1e-8
    original = static_profile.calculate_static_profile

    def fft_profile(*args, **kw):
        if not kw.get("electron_interference"):
            kw.update(method="fft", fft_tol=tol)
        return original(*args, **kw)
    cases.calculate_static_profile = fft_profile
    print(f"method: fft (fft_tol {tol:g})")
print("kernel:", "numexpr" if static_profile._numexpr is not None else "numpy")
reference = np.load(cases.REFERENCE)
for name in cases.CASES:
    cases._evaluate(name)
    t = time.perf_counter()
    out = cases._evaluate(name)
    dt = time.perf_counter() - t
    ref = reference[name]
    print(f"{name:30s} {dt * 1e3:8.1f} ms   max |diff| / peak {np.max(np.abs(out - ref)) / ref.max():.1e}")
