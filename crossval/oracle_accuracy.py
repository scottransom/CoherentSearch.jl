#!/usr/bin/env python
"""
Generate Python "oracle" reference data for cross-validating CoherentSearch.jl.

This imports the original Python `coherent_search` package and computes:
  1. A `finterp_FFT` kernel reference (the indexing-critical path).
  2. An end-to-end coherent-fold profile reference over a block of trial
     fundamental frequencies (the same algorithm CoherentSearch.search_block
     implements).
  3. The peak boxcar matched-filter detection metric on those profiles.

`coherent_search` is imported from the sibling repo's `src/` in preference to
whatever is installed, and the resolved path is printed: an oracle has to track
the checked-out source, and a stale non-editable install once pinned this
comparison to a superseded metric with no visible sign that it had.

Outputs are written to <outdir>/oracle.json plus raw little-endian binary
arrays, which crossval_accuracy.jl reads back and compares.

Usage:
    python oracle_accuracy.py FILE.fft OUTDIR
"""
import os
import sys
import json

# Prefer the sibling repo's working tree over an installed copy (see above).
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "coherent_search", "src")
if os.path.isdir(_SRC):
    sys.path.insert(0, os.path.abspath(_SRC))

import numpy as np
import coherent_search.utils as utils
import coherent_search.fourierinterp as fi
import coherent_search.coherent_search as cs_mod
from coherent_search.coherent_search import snr_metric


def main():
    fftfile = sys.argv[1]
    outdir = sys.argv[2]

    # Provenance, always: which copy of the oracle is actually being compared.
    print(f"oracle: coherent_search from {os.path.dirname(cs_mod.__file__)}")

    ft = utils.fftfile(fftfile)

    # ---- shared test configuration (Julia reads this back verbatim) ----
    m = 32
    numbetween = 16
    nharms = 32

    # 1) Kernel reference: finterp_FFT over a chunk of bins near the pulsar.
    k_lobin = 10000
    k_numbins = 40
    kernel_ref = fi.finterp_FFT(k_lobin, k_numbins, numbetween, ft.amps, m)
    kernel_ref.astype(np.complex128).tofile(f"{outdir}/kernel_ref.bin")

    # 2) End-to-end reference: coherent-fold metric over trial fundamentals.
    #    Mirrors CoherentSearch.search_block exactly.
    lodr = 0.5 / nharms
    r0 = 10010.0
    L = 256
    rfund = r0 + np.arange(L) * lodr

    ftprofs = np.zeros((L, nharms + 1), dtype=np.complex128)
    for h in range(1, nharms + 1):
        rs = rfund * h
        lobin = int(np.floor(rs.min()))
        hibin = int(np.ceil(rs.max())) + 1
        numbins = hibin - lobin
        if lobin >= m // 2 and lobin + numbins + m // 2 <= len(ft.amps) and hibin < ft.N // 2:
            amps = fi.finterp_FFT(lobin, numbins, numbetween, ft.amps, m)
            trs = np.arange(numbins * numbetween) / numbetween + lobin
            # Interpolate real/imag separately (matches the Julia complex interp).
            re = np.interp(rs, trs, amps.real)
            im = np.interp(rs, trs, amps.imag)
            ftprofs[:, h] = re + 1j * im
    profs = np.fft.irfft(ftprofs, axis=1)   # (L, nbins), normalised

    # Dump the raw profiles first.  The Julia side reads these back and runs its
    # own metric on identical inputs, isolating the metric port from any FFT /
    # indexing convention differences (which the profiles + kernel checks guard).
    profs.astype(np.float64).tofile(f"{outdir}/profs_ref.bin")

    # Peak boxcar matched-filter S/N, exactly as coherent_search.py computes it.
    # This is now the *only* metric on either side; the older on-pulse penalties
    # (non / sd2) were retired upstream and here.  Note the pooled block sigma is
    # a full MAD over every bin of every profile, which is what the Julia
    # `snr_metrics` reference reproduces -- the production search subsamples it.
    fsp, maxfrac = 1.5, 0.3
    snr_metric(profs.copy(), fsp, maxfrac).astype(np.float64).tofile(
        f"{outdir}/metric_boxcar_ref.bin")

    rfund.astype(np.float64).tofile(f"{outdir}/rfund.bin")

    meta = {
        "fftfile": fftfile,
        "N": int(ft.N),
        "T": float(ft.T),
        "m": m,
        "numbetween": numbetween,
        "nharms": nharms,
        "nbins": 2 * nharms,
        "boxcar_fsp": fsp,
        "boxcar_maxfrac": maxfrac,
        "kernel": {"lobin": k_lobin, "numbins": k_numbins, "len": int(kernel_ref.size)},
        "e2e": {"r0": r0, "lodr": lodr, "L": L},
    }
    with open(f"{outdir}/oracle.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote oracle data to {outdir}/")


if __name__ == "__main__":
    main()
