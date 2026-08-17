#!/usr/bin/env python3
"""
compare_riptide.py -- head-to-head of CoherentSearch.jl against riptide's FFA.

Runs riptide's ``rseek`` and this repository's ``coherent_search.jl`` over the
same observation with matched settings, times both, cross-matches the candidate
lists by frequency, and prints a side-by-side table.

The two searches are different algorithms, so "matched" needs stating precisely:

  * **Both codes obey the same bins-vs-period limit, and both use the same
    trick to live with it.**  riptide requires ``P >= tsamp * bins``, and
    downsamples to keep the fold between ``bmin`` and ``bmax`` bins.  Our
    equivalent: a ``k``-decimated fold of ``Hk = nharms/k`` harmonics at
    frequency ``f`` needs ``f * Hk < 1/(2*tsamp)``, i.e. ``P > tsamp * nbins``
    for ``nbins = 2*Hk`` -- the identical constraint, from the identical
    sampling limit.  Both therefore reach high frequency by folding into fewer
    bins: riptide by downsampling, us by harmonic decimation.

    So the honest comparison matches the **total frequency coverage** and lets
    each side degrade its own way:

        nharms   = bmax / 2                 (bins at the deep end)
        maxdecim = bmax / bmin              (bins span, bmax..bmin)
        fmax     = 1 / Pmin                 (top of the searched band)
        hifreq   = fmax / maxdecim          (our FUNDAMENTAL range tops out here;
                                             decimation carries it up to fmax)

    With ``--bmin 20 --bmax 120`` that is ``--nharms 60 --maxdecim 6``, both
    covering 0.1-200 Hz in 120…20 bins.  Setting our ``hifreq`` to ``1/Pmin``
    instead -- the obvious-looking choice -- would have us search 6x the band
    riptide does and lose the timing comparison for no reason.

    **Matched coverage is not matched work, and the gap is 2.8x.**  riptide's
    ``b`` sweep is how it TILES frequency: each period is covered exactly once,
    at whatever ``b`` the downsampling ladder lands on.  Our ``k`` sweep does
    double duty -- it extends the band AND it is a harmonic-depth ladder -- so
    every frequency below ``hifreq`` is folded at all six depths.  Same set of
    depths, very different multiplicity.  The WORK block prints the resulting
    ratios; ``--preset matched`` removes the difference by running one depth
    per side.

  * **Pushing to the sampling limit.** ``Pmin`` defaults to ``tsamp * bmin``,
    riptide's own floor, so both searches run the widest band the data support
    at the chosen bin range.  Lowering ``bmin`` (raising ``maxdecim``) buys both
    sides more high-frequency reach at shallower depth.

  * **Threads.** riptide's C extension is built without OpenMP (see its
    ``setup.py``) and its Python driver is serial, so the FFA itself runs on one
    core.  Measured, ``rseek`` uses 1.0-1.15 cores end to end, the excess being
    numpy/BLAS in de-reddening and peak-finding; the report prints the measured
    core count for both sides rather than taking this on trust.

    That BLAS threading is left ENABLED by default -- riptide should get any
    benefit going.  Measured, there is none: pinning every thread pool to 1
    (``--rseek-threads 1``) moved the median wall clock 4.68 -> 4.54 s over 6
    interleaved pairs, i.e. nothing against ~8% scatter, while cutting CPU from
    114% to 99%.  The extra core is overhead, not speed.

    The like-for-like comparison is our ``-t 1``; ``--threads`` runs an extra
    multi-threaded pass, reported separately rather than as the headline.

  * **Wall-clock.** Both numbers are whole-process wall time, which includes
    each language's interpreter start-up.  On a thermally-limited machine the
    MULTI-THREADED number is the least reliable one here: back-to-back heavy
    runs heat the CPU and it clocks down, so the threaded figure drifts between
    invocations far more than the single-threaded one.  Treat ``-t 1`` as the
    result and the threaded row as indicative.  ``--repeat`` runs each search
    several times and reports the minimum (least contaminated by other load)
    alongside the median.  A first ``rseek`` run in a fresh shell also pays
    Python import cost; warm-up is done once before timing.

**The two S/N values ARE the same statistic, up to one exactly known factor.**
Both are a boxcar matched filter maximised over phase on a folded profile.  They
differ only in how the template is normalised:

  * riptide (``cpp/snr.hpp``, ``snr1``) correlates against a **zero-mean,
    unit-L2** boxcar -- height ``+h`` on the ``w`` on-pulse bins, ``-b`` on the
    rest -- and divides by ``stdnoise``.  Unit L2 means the per-phase statistic
    has variance exactly 1.
  * ours (``_boxcar_scan``) sums the ``w`` median-subtracted on-pulse bins and
    divides by ``sigma*sqrt(w)``, i.e. it normalises as though the baseline were
    known rather than estimated from the same profile.

Working the two templates out, ours = ``sqrt(1 - w/nbins)`` x riptide's,
*pointwise* -- so under noise our per-phase statistic has variance ``1 - duty``,
not 1.  Measured on 200k pure-noise 120-bin profiles the ratio of the two codes'
per-width means matches ``sqrt(1 - duty)`` to four decimals at every width:
0.9962 at w=1 up to 0.8758 at w=28.

So they are directly comparable after multiplying ours by ``1/sqrt(1 - ducy)``,
and a raw difference of a few tenths at small duty really is not meaningful --
but at 30% duty ours reads ~16% low, which is a real and correctable bias
against broad pulses, not a difference of definition.  The duty cycles are
defined identically (best boxcar width / profile bins) and need no correction.

This is an OCCASIONAL benchmark, not a development-loop tool: the default
``bench`` preset takes ~5 minutes at ``--repeat 3``.  Run it when something has
changed that could plausibly move the ratio, not on every commit.

Usage:

    python3 compare/compare_riptide.py FILE.fft                      # bench preset
    python3 compare/compare_riptide.py --preset matched FILE.fft     # equal-work timing
    python3 compare/compare_riptide.py --preset quick FILE.fft       # ~1 min sanity check
    python3 compare/compare_riptide.py --repeat 3 --threads 4 FILE.fft
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import re
import resource
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def read_inf(path):
    """Return (N, dt, T, dm) from a PRESTO .inf file."""
    N = dt = dm = None
    with open(path) as fh:
        for line in fh:
            if line.startswith(" Number of bins"):
                N = int(line.split("=")[-1])
            elif line.startswith(" Width of each time series bin"):
                dt = float(line.split("=")[-1])
            elif line.startswith(" Dispersion measure"):
                dm = float(line.split("=")[-1])
    if N is None or dt is None:
        raise SystemExit(f"{path}: missing 'Number of bins' or bin width")
    return N, dt, N * dt, dm


class Timing:
    """Wall/CPU timings for one search.  `cores` = CPU seconds / wall seconds."""

    def __init__(self, walls, cores, stdout, stderr=""):
        self.min = min(walls)
        self.median = statistics.median(walls)
        self.cores = max(cores)
        self.stdout = stdout
        self.stderr = stderr


def _child_cpu():
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def run_timed(cmd, repeat, env=None, cwd=None):
    """Run `cmd` `repeat` times and return a `Timing`.

    One untimed warm-up run precedes the timed ones so neither side is charged
    for cold page cache or first-import costs that a real batch would not pay
    on every file.  CPU time is measured too: how many cores each search
    actually uses is the crux of whether a wall-clock comparison is fair, so it
    is measured rather than assumed from the build flags.
    """
    subprocess.run(cmd, capture_output=True, env=env, cwd=cwd)   # warm-up
    walls, cores, out, err = [], [], "", ""
    for _ in range(repeat):
        c0, t0 = _child_cpu(), time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)
        wall = time.perf_counter() - t0
        cpu = _child_cpu() - c0
        walls.append(wall)
        cores.append(cpu / wall if wall > 0 else 0.0)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr[-4000:])
            raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
        out, err = proc.stdout, proc.stderr
    return Timing(walls, cores, out, err)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------
class Cand:
    """One candidate from either search.  `ducy` is a fraction, or None."""

    def __init__(self, freq, snr, ducy=None, extra=""):
        self.freq = freq
        self.snr = snr
        self.ducy = ducy
        self.extra = extra

    @property
    def period_ms(self):
        return 1000.0 / self.freq


def parse_rseek(text):
    """Parse rseek's pandas table: period freq width ducy dm snr."""
    cands = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 6 or parts[0].startswith("period"):
            continue
        try:
            freq = float(parts[1])
            width = int(parts[2])
            ducy = float(parts[3].rstrip("%")) / 100.0
            snr = float(parts[5])
        except ValueError:
            continue
        cands.append(Cand(freq, snr, ducy, extra=f"w={width}"))
    return cands


def parse_cohout(path):
    """Parse a .cohout / -o candidate file: rank S/N freq period #harm [ducy%]."""
    cands = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                snr = float(parts[1])
                freq = float(parts[2])
                nharm = int(parts[4])
            except ValueError:
                continue
            ducy = None
            if len(parts) >= 6:
                try:
                    ducy = float(parts[5]) / 100.0
                except ValueError:
                    ducy = None
            cands.append(Cand(freq, snr, ducy, extra=f"H={nharm}"))
    return cands


def rseek_stages(stderr):
    """Sum rseek's own per-stage DEBUG timings -> (dict of stage->s, total_s).

    riptide logs "'<stage>' runtime: N ms" for each phase, which gives its
    compute time excluding Python start-up -- the number to compare against our
    wall clock minus our fixed cost.
    """
    stages = {}
    for m in re.finditer(r"'(\w+)' runtime: ([\d.]+) ms", stderr):
        stages[m.group(1)] = float(m.group(2)) / 1000.0
    return stages, sum(stages.values())


# ---------------------------------------------------------------------------
# Cross-matching
# ---------------------------------------------------------------------------
def harmonic_of(freq, bases, T, tol_bins, numharm=8):
    """Is `freq` a simple ratio n/m of one of `bases`?  -> ("n/m", base) or None.

    riptide does no harmonic filtering, so its list contains the f/2, 2f, 3f/2 …
    family of every real signal as separate entries; we collapse that family
    into one candidate.  Without this check those entries read as detections we
    missed, when they are the same signal.
    """
    for base in bases:
        for n in range(1, numharm + 1):
            for m in range(1, numharm + 1):
                pred = base * n / m
                if abs(freq - pred) * T <= tol_bins:
                    if n == 1 and m == 1:
                        continue          # that is a plain match, handled already
                    return (f"{n}/{m}" if m > 1 else f"{n}", base)
    return None


def crossmatch(a, b, T, tol_bins):
    """Greedy best-S/N-first match of two candidate lists by Fourier bin distance.

    Tolerance is in Fourier bins (|df| * T), which is the natural unit: it is
    the resolution of the observation and so does not depend on the frequency.

    Returns a list of (cand_a | None, cand_b | None) pairs, strongest first.
    """
    unused = list(b)
    pairs = []
    for ca in sorted(a, key=lambda c: -c.snr):
        best, bestd = None, None
        for cb in unused:
            d = abs(ca.freq - cb.freq) * T
            if d <= tol_bins and (bestd is None or d < bestd):
                best, bestd = cb, d
        if best is not None:
            unused.remove(best)
        pairs.append((ca, best))
    pairs.extend((None, cb) for cb in sorted(unused, key=lambda c: -c.snr))
    return pairs


# ---------------------------------------------------------------------------
# Work accounting
#
# The timing ratio is meaningless without knowing how much searching each side
# did, and the two codes do NOT do the same amount.  What IS matched, exactly,
# is the *frequency resolution rule*:
#
#   riptide's FFA at base period `b` samples emits `m = n/b` shifts spanning
#   the periods `b` to `b+1` samples, so its trial spacing in Fourier bins is
#       dr = n / ((m-1) * b^2) = 1/b   (verified numerically: 1/b to 2e-4)
#   i.e. exactly one PHASE BIN of drift across the observation.  That is not a
#   tunable -- it is what the transform produces.
#
#   Our `--hidr` is the step of the HIGHEST harmonic, so the fundamental steps
#   by `hidr/nharms`; a decimation-`k` pass reports `k*rf`, hence
#       dr = k*hidr/nharms = hidr/(nharms/k) = 1/nbins   at hidr = 0.5.
#
# So `--hidr 0.5` with `nharms = bmax/2` IS the FFA's native resolution, at
# every k.  There is nothing to correct there.
#
# What differs is how many folds each code applies PER FREQUENCY:
#
#   * riptide folds each frequency ONCE, at the single `b` its downsampling
#     ladder happens to land on -- sawtoothing 20..120 within each cycle, so
#     the depth and the resolution at a given frequency are whatever the ladder
#     gives (b=36 at 3 Hz, b=109 at 1 Hz on the reference observation).
#   * we fold every frequency below `hifreq` SIX times (k=1..6, 120..20 bins),
#     because each decimation covers [k*lofreq, k*hifreq] and those overlap.
#     That is our harmonic-sum ladder, and it is not redundant: the 7.1185 Hz
#     pulsar scores 12.97 at k=6 (10 harmonics) but only 11.89 with k=1 alone.
#   * riptide's boxcar bank is built from `bins_min` and reused for every
#     profile, so at b=120 it only reaches 6/120 = 5% duty; ours is built from
#     each profile's own `nbins` and reaches `boxcar_maxfrac` = 30% everywhere.
#     On the reference observation rseek is width-LIMITED on the one real
#     pulsar (reports w=6, its maximum, ducy 6.5% for a ~10% pulse).
#
# Net on the reference observation: we evaluate ~2.8x the profiles and ~3.6x
# the boxcar (bins x widths) work.  Quote the timing ratio against these.
# ---------------------------------------------------------------------------
def width_bank(nbins, maxfrac, fsp=1.5):
    """Geometric boxcar bank -- identical rule in both codes.

    riptide calls this with `bins_min` for every profile
    (`generate_width_trials`); we call it with each profile's own `nbins`
    (`boxcar_widths`).  Same formula, different argument.
    """
    wmax = max(1, int(maxfrac * nbins))
    out, w = [], 1
    while w <= wmax:
        out.append(w)
        w = int(max(w + 1, fsp * w))
    return out


def riptide_work(N, tsamp, pmin, pmax, bmin, bmax, ducy_max=0.3, wtsp=1.5):
    """Port of riptide's `periodogram_length`, plus work counters.

    Faithful to `cpp/periodogram.hpp`: the same downsampling ladder, the same
    `ceilshift` row cut, the same `floor(n/f)` downsampled size.  Returns
    profiles, folded profile bins, and boxcar (bins x widths) operations, plus
    the (frequency -> foldbins) map needed to report `dr` versus frequency.
    """
    ds_ini = pmin / (tsamp * bmin)
    ds_geo = (bmax + 1.0) / bmin
    nds = math.ceil(math.log(pmax / pmin) / math.log(ds_geo))
    nw = len(width_bank(bmin, ducy_max, wtsp))    # ONE bank, from bins_min
    prof = 0
    binsum = 0.0
    bcops = 0.0
    spans = []                                    # (flo, fhi, b) per FFA block
    for ids in range(nds):
        f = ds_ini * ds_geo ** ids
        tau = f * tsamp
        pmax_samples = pmax / tau
        n = math.floor(N / f)
        for b in range(bmin, min(bmax, n, int(pmax_samples)) + 1):
            rows = n // b
            if rows < 2:
                continue
            pceil = min(pmax_samples, b + 1.0)
            rows_eval = min(rows, math.ceil(b * (rows - 1.0) * (1.0 - b / pceil)))
            if rows_eval <= 0:
                continue
            prof += rows_eval
            binsum += rows_eval * b
            bcops += rows_eval * b * nw
            fhi = 1.0 / (tau * b)
            flo = 1.0 / (tau * b * b / (b - (rows_eval - 1) / (rows - 1.0)))
            spans.append((flo, fhi, b))
    return dict(profiles=prof, bins=binsum, bcops=bcops, nwidths=nw, spans=spans)


def coherent_work(T, lofreq, hifreq, nharms, maxdecim, hidr, maxfrac=0.3, fsp=1.5):
    """Work counters for our search, from the same quantities the CLI is given."""
    lodr = hidr / nharms
    ntrials = math.floor((hifreq * T - lofreq * T) / lodr) + 1
    ks = [k for k in range(1, maxdecim + 1) if nharms // k >= 2]
    binsum = 0.0
    bcops = 0.0
    stages = []
    for k in ks:
        nbins = 2 * (nharms // k)
        nw = len(width_bank(nbins, maxfrac, fsp))
        binsum += ntrials * nbins
        bcops += ntrials * nbins * nw
        stages.append((k, nbins, nw, k * lofreq, k * hifreq))
    return dict(trials=ntrials, profiles=ntrials * len(ks), bins=binsum,
                bcops=bcops, lodr=lodr, stages=stages)


def print_work(rw, cw, fmax):
    """Side-by-side work accounting, printed next to the timing so the ratio
    is never quoted on its own."""
    print("-" * 78)
    print("WORK  (what each side actually searched -- read this beside the timing)")
    print("-" * 78)
    print(f"  {'':26} {'rseek':>16} {'coherent':>16} {'ours/rseek':>11}")
    for lab, key in (("profiles evaluated", "profiles"),
                     ("profile bins folded", "bins"),
                     ("boxcar bins x widths", "bcops")):
        a, b = rw[key], cw[key]
        print(f"  {lab:26} {a:16,.0f} {b:16,.0f} {b / a:10.2f}x")
    print(f"  boxcar widths: rseek {rw['nwidths']} for EVERY profile (bank built from"
          f" bins_min);")
    print("                 ours " + ", ".join(
        f"{nw}@{nbins}b" for _, nbins, nw, _, _ in cw["stages"]) + " (bank per profile)")
    print()
    print("  Trial spacing is the SAME RULE on both sides: dr = 1/nbins Fourier bins,")
    print("  i.e. one phase bin of drift across T.  For us that is --hidr/nharms per")
    print(f"  fundamental ({cw['lodr']:.6g} bins) times k; for rseek it is 1/b, with b")
    print("  set by its downsampling ladder rather than chosen.")
    print()
    print(f"  {'freq (Hz)':>12} {'rseek b':>9} {'rseek dr':>10} | "
          f"{'our nbins':>10} {'our dr':>9} {'our folds':>10}")
    spans = sorted(rw["spans"])
    for probe in (0.2, 1.0, 3.0, 7.1185, 10.0, 30.0, 60.0, 120.0, 0.98 * fmax):
        rb = next((b for flo, fhi, b in spans if flo <= probe <= fhi), None)
        ours = [nb for _, nb, _, klo, khi in cw["stages"] if klo <= probe <= khi]
        if rb is None or not ours:
            continue
        deep = max(ours)                      # deepest fold covering this frequency
        print(f"  {probe:12.4g} {rb:9d} {1.0 / rb:10.5f} | "
              f"{deep:10d} {1.0 / deep:9.5f} {len(ours):10d}")
    print()
    if len(cw["stages"]) > 1:
        print("  'our folds' is how many decimations cover that frequency: every one of")
        print("  them is a separate profile + boxcar scan, where rseek does exactly one.")
        print("  That is where the work ratio above comes from -- NOT from a finer")
        print("  frequency grid, which is matched exactly.  It buys sensitivity: the")
        print("  shallow folds are our harmonic-sum ladder (see CLAUDE.md).")
        print("  For a pure algorithm-vs-algorithm timing, use --preset matched, which")
        print("  runs ONE depth on each side and equalises the work to a few percent.")
    else:
        print("  One fold per frequency on BOTH sides, at the same depth and the same")
        print("  grid, so the wall clock below is a direct read on the two")
        print("  implementations rather than on how much each chose to search.")
    print()


# ---------------------------------------------------------------------------
# Presets
#
# A comparison is only as good as the fraction of it that is actual searching.
# `quick` spends ~1 s per side on interpreter start-up out of ~5 s, so it
# flatters whichever side starts faster; use it to check the harness runs.
# `bench` is ~4x the work -- start-up falls to <3% of our runtime and ~6% of
# rseek's -- and is the configuration to quote.  Measured 2026-08-11 on
# PM0063_034C1_DM445.0_red.fft (T=2097 s, 4-core i7-10510U): the two presets
# agree on the ratio (2.11x vs 2.27x), which is the point of having both.
#
# `bench` and `matched` answer DIFFERENT questions, and both are worth having:
#
#   bench   -- "which code searches this observation, end to end, faster?"
#              Coverage is matched (0.1-200 Hz both sides) but the work is not:
#              we fold everything below hifreq six times and rseek folds it
#              once, so we do ~2.8x the profiles.  Quote the work table with it.
#
#   matched -- "which code is faster at doing the SAME work?"  One fold depth
#              on each side, so profiles / folded bins / boxcar ops agree to
#              within a few percent and the wall clock is a clean read on the
#              two implementations.
#
# `bench` also puts riptide OUTSIDE its own documented operating range:
# `ffa_search`'s docstring says bins_max should be "approx. 10% larger" than
# bins_min, and riptide's example pipeline config uses 240/260 and 480/520.
# bmax/bmin = 6 makes its fold depth `b` sawtooth over the whole 20..120 range
# within each downsampling cycle, which (a) drops its mean trial density to
# ~43 per Fourier bin instead of ~b, and (b) leaves its boxcar bank -- built
# once from bins_min -- undersized for every profile deeper than bmin.  That is
# not sabotage, it is forced: one rseek invocation cannot reach 200 Hz on this
# data with bins_min > 20.  It is simply not the regime riptide is written for,
# which is exactly why `matched` exists.
#
# This is an occasional benchmark, not a development-loop tool: `bench` takes
# ~5 minutes at --repeat 3.
# ---------------------------------------------------------------------------
PRESETS = {
    # ~5 s/side.  Sanity check only; Pmin is raised well above the sampling
    # limit to keep it quick, which also makes it a narrow-band test.
    "quick": dict(Pmin=0.05, Pmax=10.0, bmin=30, bmax=120),
    # The one to quote for END-TO-END search speed: the widest band the data
    # support at 120..20 bins (Pmin defaults to tsamp*bmin).  ~20 s (rseek) /
    # ~32 s (us, -t 1) on the reference observation, workstation -- compute
    # dominated, start-up under 4% either side.
    "bench": dict(Pmin=None, Pmax=10.0, bmin=20, bmax=120),
    # The one to quote for ALGORITHM speed.  bmax/bmin < 2 makes the derived
    # maxdecim 1, so we run a single 130-bin fold against rseek's 120-130 --
    # same band (Pmin defaults to tsamp*120 = the deepest fold this data
    # supports), the same dr = 1/nbins grid, and the same 9-width boxcar bank
    # up to 30% duty, because at bins_min=120 riptide's bank finally matches
    # ours.  Measured 2026-08-16 on PM0063, workstation, -t 1, interleaved:
    # rseek 18.35/18.33 s, ours 17.86/16.45 s, against work counts that agree
    # to 4-8% (we do slightly less: rseek's b runs 120-130, ours is fixed).
    # So at equal work the two implementations are a wash, us ~1.05-1.11x
    # ahead.  rseek wins the pulsar though: S/N 12.6 (w=13, ducy 10.3%) vs our
    # 11.89 at the same depth -- our 12.97 in `bench` comes from the k=6 fold.
    "matched": dict(Pmin=None, Pmax=10.0, bmin=120, bmax=130),
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fftfile", help="PRESTO .fft file (its .inf and .dat must sit beside it)")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="bench",
                    help="search-range preset; 'bench' is compute-dominated and is "
                         "the one to quote, 'quick' is a fast sanity check whose "
                         "ratio is diluted by both interpreters' start-up")
    ap.add_argument("--Pmin", type=float, default=None, help="rseek min trial period (s)")
    ap.add_argument("--Pmax", type=float, default=None, help="rseek max trial period (s)")
    ap.add_argument("--bmin", type=int, default=None, help="rseek min phase bins")
    ap.add_argument("--bmax", type=int, default=None, help="rseek max phase bins")
    ap.add_argument("--smin", type=float, default=7.0, help="rseek S/N floor")
    ap.add_argument("--threshold", type=float, default=None,
                    help="coherent_search S/N floor (default: same as --smin)")
    ap.add_argument("--nharms", type=int, default=None,
                    help="override the derived nharms (default: bmax/2)")
    ap.add_argument("--maxdecim", type=int, default=None,
                    help="override the derived maxdecim (default: bmax/bmin)")
    ap.add_argument("--hidr", type=float, default=0.5,
                    help="coherent_search trial step at the highest harmonic; 0.5 "
                         "is exactly the FFA's own resolution (dr = 1/nbins) and is "
                         "what makes the frequency grids match")
    ap.add_argument("--threads", type=int, default=0,
                    help="also run coherent_search with this many threads "
                         "(rseek is single-threaded, so -t 1 is the fair comparison)")
    ap.add_argument("--repeat", type=int, default=3, help="timed runs per search")
    ap.add_argument("--tol-bins", type=float, default=3.0,
                    help="cross-match tolerance in Fourier bins")
    ap.add_argument("--rseek-threads", type=int, default=0,
                    help="pin rseek's BLAS/OMP thread pools to this many threads "
                         "(0 = leave its environment alone, the default -- measured "
                         "to make no difference to its wall clock)")
    ap.add_argument("--rseek", default=None, help="path to the rseek executable")
    ap.add_argument("--julia", default="julia", help="path to the julia executable")
    ap.add_argument("--sysimage", default=None, help="julia sysimage to use")
    ap.add_argument("--noharmremove", action="store_true",
                    help="disable our harmonic-family collapse, which rseek also "
                         "does not do, for a like-for-like candidate count")
    ap.add_argument("--keep", action="store_true", help="keep the .cohout output")
    args = ap.parse_args(argv)
    for key, val in PRESETS[args.preset].items():      # explicit flags win
        if getattr(args, key) is None:
            setattr(args, key, val)

    fft = os.path.abspath(args.fftfile)
    stem = fft[:-4] if fft.endswith(".fft") else fft
    inf = stem + ".inf"
    for p in (fft, inf):
        if not os.path.isfile(p):
            raise SystemExit(f"missing input: {p}")
    N, dt, T, dm = read_inf(inf)

    # Pmin defaults to riptide's own floor, tsamp*bmin: the shortest period the
    # data support at this bin count, for both codes (see the module docstring).
    if args.Pmin is None:
        args.Pmin = dt * args.bmin

    # riptide raises "Must have: period_min >= tsamp * bins_min" from deep inside
    # its C extension; catch it here, where we can say what to do about it.  (A
    # failing rseek run would otherwise look like a suspiciously fast one.)
    if args.Pmin < dt * args.bmin:
        raise SystemExit(
            f"riptide requires Pmin >= tsamp * bmin: with tsamp={dt:g} s and "
            f"bmin={args.bmin} that is Pmin >= {dt * args.bmin:g} s "
            f"(you asked for {args.Pmin:g}).  Raise --Pmin, or lower --bmin to "
            f"at most {int(args.Pmin / dt)}.")

    rseek = args.rseek or shutil.which("rseek") or (
        "/home/sransom/python_venvs/pixiPSR/.pixi/envs/default/bin/rseek")
    if not os.path.isfile(rseek):
        raise SystemExit("rseek not found; pass --rseek /path/to/rseek")

    # --- matched configuration -------------------------------------------
    nharms = args.nharms if args.nharms else args.bmax // 2
    maxdecim = args.maxdecim if args.maxdecim else max(1, args.bmax // args.bmin)
    threshold = args.threshold if args.threshold is not None else args.smin
    # Our FUNDAMENTAL range stops at fmax/maxdecim; decimation carries coverage
    # the rest of the way to fmax, so the two searches span the same band.
    fmax = 1.0 / args.Pmin
    lofreq, hifreq = 1.0 / args.Pmax, fmax / maxdecim

    print("=" * 78)
    print("CoherentSearch.jl  vs  riptide (rseek)")
    print("=" * 78)
    print(f"observation : {os.path.basename(fft)}")
    print(f"              N={N}  dt={dt:g} s  T={T:.2f} s  DM={dm}")
    print(f"rseek       : --Pmin {args.Pmin} --Pmax {args.Pmax} "
          f"--bmin {args.bmin} --bmax {args.bmax} --smin {args.smin}")
    print(f"              searches {lofreq:.4g}-{fmax:.4g} Hz, {args.bmin}-{args.bmax} phase bins")
    print(f"coherent    : --lofreq {lofreq:.6g} --hifreq {hifreq:.6g} "
          f"--nharms {nharms} --maxdecim {maxdecim} --hidr {args.hidr:g} "
          f"--threshold {threshold}")
    nbins_hi, nbins_lo = 2 * nharms, 2 * (nharms // maxdecim)
    print(f"              fundamentals {lofreq:.4g}-{hifreq:.4g} Hz; decimation to k="
          f"{maxdecim} covers {lofreq:.4g}-{maxdecim * hifreq:.4g} Hz, "
          f"{nbins_hi} down to {nbins_lo} profile bins")
    cov = maxdecim * hifreq
    if abs(cov - fmax) / fmax > 0.01:
        print(f"              NOTE: our coverage tops out at {cov:.4g} Hz vs rseek's "
              f"{fmax:.4g} Hz -- bands are NOT matched")
    else:
        print(f"              coverage matched: both search up to {fmax:.4g} Hz")
    if nbins_lo != args.bmin or nbins_hi != args.bmax:
        print(f"              NOTE: bin span {nbins_lo}-{nbins_hi} != rseek's "
              f"{args.bmin}-{args.bmax}; depth is not exactly matched")
    print(f"              (both codes are limited to nbins <= P/tsamp = "
          f"{args.Pmin / dt:.0f} bins at the top of the band)")
    print()

    # --- work accounting (cheap; printed before anything is timed) ----------
    rwork = riptide_work(N, dt, args.Pmin, args.Pmax, args.bmin, args.bmax)
    cwork = coherent_work(T, lofreq, hifreq, nharms, maxdecim, args.hidr)
    print_work(rwork, cwork, fmax)

    # --- rseek -------------------------------------------------------------
    rcmd = [rseek, "--Pmin", str(args.Pmin), "--Pmax", str(args.Pmax),
            "--bmin", str(args.bmin), "--bmax", str(args.bmax),
            "--smin", str(args.smin), "-f", "presto", inf]
    renv = None
    if args.rseek_threads > 0:
        renv = dict(os.environ)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            renv[var] = str(args.rseek_threads)
    print(f"running rseek ({args.repeat}x + warm-up) ...", flush=True)
    rt = run_timed(rcmd, args.repeat, env=renv)
    rcands = parse_rseek(rt.stdout)

    # --- coherent_search ----------------------------------------------------
    out = stem + ".cmp.cohout"
    def coherent_cmd(threads):
        c = [args.julia]
        if args.sysimage:
            c += [f"--sysimage={args.sysimage}"]
        c += [f"--project={REPO}", f"-t{threads}",
              os.path.join(REPO, "bin", "coherent_search.jl"),
              "--lofreq", repr(lofreq), "--hifreq", repr(hifreq),
              "--nharms", str(nharms), "--maxdecim", str(maxdecim),
              "--hidr", repr(args.hidr),
              "--threshold", str(threshold), "--noprogress",
              "-o", out]
        if args.noharmremove:
            c += ["--noharmremove"]
        c += [fft]
        return c

    print(f"running coherent_search -t 1 ({args.repeat}x + warm-up) ...", flush=True)
    ct = run_timed(coherent_cmd(1), args.repeat)
    ccands = parse_cohout(out)

    # Our fixed cost: the same command over a near-empty frequency band, so it
    # pays Julia's boot, the precompiled-code load, and FFTW planning for this
    # exact nharms/maxdecim, but does essentially no searching.  Subtracting it
    # answers "is the comparison just measuring Julia's start-up?".
    fixed_cmd = [c for c in coherent_cmd(1)]
    fixed_cmd[fixed_cmd.index("--hifreq") + 1] = repr(lofreq * 1.002)
    print("measuring coherent_search fixed cost (near-empty band) ...", flush=True)
    ft_fixed = run_timed(fixed_cmd, max(2, args.repeat // 2))

    mt = None
    if args.threads > 1:
        print(f"running coherent_search -t {args.threads} ({args.repeat}x) ...", flush=True)
        mt = run_timed(coherent_cmd(args.threads), args.repeat)

    # --- timing -------------------------------------------------------------
    print()
    print("-" * 78)
    print("TIMING  (whole-process wall clock, including interpreter start-up)")
    print("-" * 78)
    print(f"  {'search':<26} {'min (s)':>9} {'median (s)':>11} {'cores used':>11}")
    print(f"  {'rseek':<26} {rt.min:>9.2f} {rt.median:>11.2f} {rt.cores:>11.2f}")
    print(f"  {'coherent_search -t 1':<26} {ct.min:>9.2f} {ct.median:>11.2f} {ct.cores:>11.2f}")
    if mt:
        print(f"  {'coherent_search -t %d' % args.threads:<26} "
              f"{mt.min:>9.2f} {mt.median:>11.2f} {mt.cores:>11.2f}")
    faster = rt.min / ct.min
    verdict = f"{faster:.2f}x FASTER" if faster >= 1 else f"{1 / faster:.2f}x SLOWER"
    print(f"\n  like-for-like (-t 1): coherent_search is {verdict} than rseek")
    if mt:
        print(f"  with {args.threads} threads:  {rt.min / mt.min:.2f}x rseek wall clock -- "
              f"riptide's C extension has no OpenMP,")
        print("                        so this axis is ours alone, not a like-for-like win")
    # --- start-up vs compute --------------------------------------------
    stages, rcompute = rseek_stages(rt.stderr)
    if rcompute > 0:
        cfixed = ft_fixed.min
        ccompute = ct.min - cfixed
        print()
        print(f"  {'':<26} {'start-up':>9} {'searching':>11} {'ratio':>8}")
        print(f"  {'rseek':<26} {rt.min - rcompute:>9.2f} {rcompute:>11.2f}")
        print(f"  {'coherent_search -t 1':<26} {cfixed:>9.2f} {ccompute:>11.2f} "
              f"{ccompute / rcompute:>7.2f}x")
        print(f"  rseek stages: "
              + ", ".join(f"{k} {v:.2f}s" for k, v in sorted(stages.items(),
                                                             key=lambda kv: -kv[1])))
        print("  (rseek's stage timings are its own DEBUG log; our start-up is the same")
        print("   command over a near-empty band, so it pays boot + JIT + FFTW planning.)")
        if "find_peaks" in stages:
            print(f"  Do NOT compare our column against ffa_search alone: riptide's "
                  f"find_peaks\n  ({stages['find_peaks']:.1f}s, "
                  f"{100 * stages['find_peaks'] / rcompute:.0f}% of its compute) is a "
                  f"separate pass doing the candidate work\n  that we do inline in the "
                  f"search loop.  The totals are what line up.")

    if args.rseek_threads > 0:
        print(f"\n  rseek's BLAS/OMP pools were pinned to {args.rseek_threads} thread(s).")
    else:
        print(f"\n  rseek measured at {rt.cores:.2f} cores, and its BLAS threading was left")
        print("  enabled.  Pinning it to 1 was measured to change its wall clock by less")
        print("  than the run-to-run scatter, so the extra core is overhead, not speed.")

    # --- candidates ---------------------------------------------------------
    print()
    print("-" * 78)
    print(f"CANDIDATES  (cross-matched within {args.tol_bins:g} Fourier bins "
          f"= {args.tol_bins / T:.2e} Hz)")
    print("-" * 78)
    pairs = crossmatch(ccands, rcands, T, args.tol_bins)
    matched_freqs = [ca.freq for ca, cb in pairs if ca and cb]
    print(f"  {'freq (Hz)':>15} {'P (ms)':>12} | {'coh S/N':>8} {'coh ducy':>9} {'':>6}"
          f" | {'rseek S/N':>9} {'rs ducy':>8} {'':>6}  note")
    both = conly = ronly = ronly_harm = 0
    snr = lambda c, w: ("%*.2f" % (w, c.snr)) if c else "%*s" % (w, "-")
    duc = lambda c, w: ("%*.2f%%" % (w - 1, 100 * c.ducy)) if c and c.ducy is not None \
                       else "%*s" % (w, "-")
    for ca, cb in pairs:
        ref = ca or cb
        note = ""
        if ca and cb:
            both += 1
        elif ca:
            conly += 1
        else:
            ronly += 1
            # riptide performs no harmonic filtering (its own --help says so),
            # while we collapse the f/2, 2f, 3f/2 ... family into one candidate.
            # An "rseek only" row that is a simple ratio of something we DID find
            # is therefore the same signal, not a miss.
            h = harmonic_of(cb.freq, matched_freqs, T, args.tol_bins)
            if h is not None:
                ronly_harm += 1
                ratio, base = h
                note = f"= {ratio} x {base:.6f} Hz (we collapse harmonics)"
        print(f"  {ref.freq:>15.9f} {ref.period_ms:>12.6f} |"
              f" {snr(ca, 8)} {duc(ca, 9)} {(ca.extra if ca else ''):>6} |"
              f" {snr(cb, 9)} {duc(cb, 8)} {(cb.extra if cb else ''):>6}  {note}")
    print()
    print(f"  matched by both: {both}   coherent_search only: {conly}   rseek only: {ronly}"
          + (f" (of which {ronly_harm} are harmonics of a matched signal)" if ronly_harm else ""))
    print()
    print("  S/N is NOT the same statistic on both sides (time-domain matched filter")
    print("  vs coherent Fourier boxcar) -- compare which signals are found, and the")
    print("  duty cycles, which ARE defined identically.")
    print("  rseek does no harmonic filtering; we collapse harmonic families by")
    print("  default.  Pass --noharmremove for a like-for-like candidate count.")

    if not args.keep and os.path.isfile(out):
        os.remove(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
