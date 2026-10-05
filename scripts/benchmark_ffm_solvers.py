"""Benchmark dense and analytical FFM solvers on identical complex SDTs.

Run from the repository root, for example::

    python scripts/benchmark_ffm_solvers.py
    python scripts/benchmark_ffm_solvers.py --sizes 16 32 64 128 --energies 31

This is deliberately a standalone benchmark rather than a pytest timing test:
absolute timings and reliable speedup thresholds depend strongly on the BLAS
library, processor load, and available memory.
"""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from starkzee.ffm import _complex_ffm_profile_analytical


def _explicit_matrix_profile(
        energies, frequencies, gamma, strengths, fluctuation_rate):
    """Evaluate the complex FFM response by batched dense matrix inversion."""
    total_strength = float(np.sum(strengths.real))
    probability = strengths.real / total_strength
    complex_weight = strengths / total_strength
    mode_count = len(frequencies)
    diagonal = (
        fluctuation_rate + gamma[np.newaxis, :]
        + 1j * (energies[:, np.newaxis] - frequencies[np.newaxis, :])
    )
    matrix = (
        diagonal[:, :, np.newaxis] * np.eye(mode_count)[np.newaxis, :, :]
        - fluctuation_rate
        * probability[np.newaxis, :, np.newaxis]
        * np.ones((1, mode_count, mode_count))
    )
    rhs = np.broadcast_to(
        complex_weight[np.newaxis, :, np.newaxis],
        (len(energies), mode_count, 1),
    )
    response = np.linalg.solve(matrix, rhs)[..., 0]
    return total_strength / np.pi * np.real(np.sum(response, axis=1))


def _best_time(function, repeats):
    """Return the best wall time and the corresponding result."""
    best = np.inf
    result = None
    for _ in range(repeats):
        start = perf_counter()
        current = function()
        elapsed = perf_counter() - start
        if elapsed < best:
            best = elapsed
            result = current
    return best, result


def run(sizes, energy_count, repeats, seed):
    rng = np.random.default_rng(seed)
    energies = np.linspace(-0.055, 0.061, energy_count)
    fluctuation_rate = 4e-3

    print(f"energies={energy_count}, repeats={repeats}, seed={seed}")
    print(" modes   dense (s)   analytic (s)      speedup     max |difference|")
    print("------  ----------  -------------  -----------  ----------------")
    for mode_count in sizes:
        frequencies = np.linspace(-0.035, 0.041, mode_count)
        frequencies += rng.normal(scale=2e-4, size=mode_count)
        gamma = rng.uniform(3e-4, 2e-3, size=mode_count)
        real_strength = rng.lognormal(mean=-0.2, sigma=0.7, size=mode_count)
        dispersion = rng.normal(scale=0.15, size=mode_count) * real_strength
        dispersion -= (
            np.sum(dispersion) * real_strength / np.sum(real_strength)
        )
        strengths = real_strength + 1j * dispersion

        # Warm both code paths before timing to reduce one-time dispatch effects.
        dense_reference = _explicit_matrix_profile(
            energies, frequencies, gamma, strengths, fluctuation_rate)
        analytical_reference = _complex_ffm_profile_analytical(
            energies, frequencies, gamma, strengths, fluctuation_rate)
        np.testing.assert_allclose(
            analytical_reference, dense_reference, rtol=2e-11, atol=2e-11)

        dense_time, dense = _best_time(
            lambda: _explicit_matrix_profile(
                energies, frequencies, gamma, strengths, fluctuation_rate),
            repeats,
        )
        analytical_time, analytical = _best_time(
            lambda: _complex_ffm_profile_analytical(
                energies, frequencies, gamma, strengths, fluctuation_rate),
            repeats,
        )
        difference = float(np.max(np.abs(dense - analytical)))
        speedup = dense_time / analytical_time
        print(
            f"{mode_count:6d}  {dense_time:10.6f}  {analytical_time:13.6f}"
            f"  {speedup:11.1f}  {difference:16.3e}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[16, 32, 64, 96],
        help="numbers of SDT modes to benchmark (default: 16 32 64 96)",
    )
    parser.add_argument(
        "--energies", type=int, default=23,
        help="number of observation energies (default: 23)",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="timing repetitions; the best is reported (default: 3)",
    )
    parser.add_argument("--seed", type=int, default=511918)
    args = parser.parse_args()
    if any(size < 2 for size in args.sizes):
        parser.error("every size must be at least 2")
    if args.energies < 1 or args.repeats < 1:
        parser.error("--energies and --repeats must be positive")
    run(args.sizes, args.energies, args.repeats, args.seed)


if __name__ == "__main__":
    main()
