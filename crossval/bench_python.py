#!/usr/bin/env python
"""
Time the Python `coherent_search` Fourier interpolation, for speed
cross-validation against CoherentSearch.jl.  Prints a JSON timing report to
stdout (so the Julia driver can parse it).

Usage:
    python bench_python.py FILE.fft
"""
import sys
import json
import time
import numpy as np
import coherent_search.utils as utils
import coherent_search.fourierinterp as fi


def timeit(fn, repeats):
    fn()  # warm up
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    ft = utils.fftfile(sys.argv[1])
    m = 32
    numbetween = 16

    lobin, numbins = 10000, 1024
    coeffs = fi.get_finterp_FFT_coeffs(
        numbetween, m, utils.next_pow_of_2((numbins + m) * numbetween)
    )

    kernel_best = timeit(
        lambda: fi.finterp_FFT(lobin, numbins, numbetween, ft.amps, m, coeffs=coeffs),
        repeats=50,
    )

    report = {
        "impl": "python",
        "finterp_FFT": {
            "lobin": lobin,
            "numbins": numbins,
            "numbetween": numbetween,
            "m": m,
            "best_seconds": kernel_best,
        },
    }
    print(json.dumps(report))


if __name__ == "__main__":
    main()
