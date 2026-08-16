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

Neither S/N is the other's S/N: riptide's is a matched-filter S/N of a
time-domain fold, ours is the boxcar matched-filter S/N of a coherently
summed Fourier fold.  They are on similar scales and worth comparing, but a
difference of a few tenths is not meaningful.  The duty cycles, however, are
defined identically (best boxcar width / profile bins) and are comparable.

This is an OCCASIONAL benchmark, not a development-loop tool: the default
``bench`` preset takes ~5 minutes at ``--repeat 3``.  Run it when something has
changed that could plausibly move the ratio, not on every commit.

Usage:

    python3 compare/compare_riptide.py FILE.fft                      # bench preset
    python3 compare/compare_riptide.py --preset quick FILE.fft       # ~1 min sanity check
    python3 compare/compare_riptide.py --repeat 3 --threads 4 FILE.fft
"""

from __future__ import annotations

import argparse
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
# This is an occasional benchmark, not a development-loop tool: `bench` takes
# ~5 minutes at --repeat 3.
# ---------------------------------------------------------------------------
PRESETS = {
    # ~5 s/side.  Sanity check only; Pmin is raised well above the sampling
    # limit to keep it quick, which also makes it a narrow-band test.
    "quick": dict(Pmin=0.05, Pmax=10.0, bmin=30, bmax=120),
    # The one to quote: the widest band the data support at 120..20 bins
    # (Pmin defaults to tsamp*bmin).  ~24 s (rseek) / ~32 s (us, -t 1) on the
    # reference observation -- compute-dominated, start-up under 4% either side.
    "bench": dict(Pmin=None, Pmax=10.0, bmin=20, bmax=120),
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
    print(f"              searches {lofreq:.4g}-{hifreq:.4g} Hz, {args.bmin}-{args.bmax} phase bins")
    print(f"coherent    : --lofreq {lofreq:.6g} --hifreq {hifreq:.6g} "
          f"--nharms {nharms} --maxdecim {maxdecim} --threshold {threshold}")
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
