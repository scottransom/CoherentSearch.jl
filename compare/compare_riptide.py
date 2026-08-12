#!/usr/bin/env python3
"""
compare_riptide.py -- head-to-head of CoherentSearch.jl against riptide's FFA.

Runs riptide's ``rseek`` and this repository's ``coherent_search.jl`` over the
same observation with matched settings, times both, cross-matches the candidate
lists by frequency, and prints a side-by-side table.

The two searches are different algorithms, so "matched" needs stating precisely:

  * **Frequency band.** ``rseek`` searches periods ``[Pmin, Pmax]``, i.e.
    frequencies ``[1/Pmax, 1/Pmin]``.  We search *fundamentals* over that same
    range.  With ``--maxdecim k > 1`` we additionally detect signals at ``2f``
    … ``kf``, so our coverage EXTENDS above ``1/Pmin`` at reduced harmonic
    depth.  That is extra science, but it is also extra work, so the timing
    comparison is not in our favour -- see the coverage line in the report.

  * **Harmonic depth.** ``rseek``'s ``--bmin/--bmax`` bound the number of phase
    bins it folds into (it uses more bins at longer periods).  Our fold uses
    ``2*nharms`` bins at ``k=1``, falling to ``2*nharms/k`` under decimation.
    So ``nharms = bmax/2`` and ``maxdecim = bmax/bmin`` line the two up: with
    ``--bmin 30 --bmax 120`` that is ``--nharms 60 --maxdecim 4``, spanning
    120…30 bins exactly as riptide spans 120…30.  ``--maxdecim 6`` (as in
    ``timed_runs.sh``) searches deeper than riptide does, down to 20 bins.

  * **Threads.** riptide's C extension is built without OpenMP (see its
    ``setup.py``) and its Python driver is serial, so ``rseek`` is
    single-threaded.  The like-for-like comparison is therefore our ``-t 1``;
    ``--threads`` runs an additional multi-threaded pass, reported separately
    rather than as the headline number.

  * **Wall-clock.** Both numbers are whole-process wall time, which includes
    each language's interpreter start-up.  ``--repeat`` runs each search
    several times and reports the minimum (least contaminated by other load)
    alongside the median.  A first ``rseek`` run in a fresh shell also pays
    Python import cost; warm-up is done once before timing.

Neither S/N is the other's S/N: riptide's is a matched-filter S/N of a
time-domain fold, ours is the boxcar matched-filter S/N of a coherently
summed Fourier fold.  They are on similar scales and worth comparing, but a
difference of a few tenths is not meaningful.  The duty cycles, however, are
defined identically (best boxcar width / profile bins) and are comparable.

Usage:

    python3 compare/compare_riptide.py PM0063_034C1_DM445.0_red.fft
    python3 compare/compare_riptide.py --repeat 3 --threads 4 FILE.fft
"""

from __future__ import annotations

import argparse
import os
import shutil
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

    def __init__(self, walls, cores, stdout):
        self.min = min(walls)
        self.median = statistics.median(walls)
        self.cores = max(cores)
        self.stdout = stdout


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
    walls, cores, out = [], [], ""
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
        out = proc.stdout
    return Timing(walls, cores, out)


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
# Driver
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fftfile", help="PRESTO .fft file (its .inf and .dat must sit beside it)")
    ap.add_argument("--Pmin", type=float, default=0.05, help="rseek min trial period (s)")
    ap.add_argument("--Pmax", type=float, default=10.0, help="rseek max trial period (s)")
    ap.add_argument("--bmin", type=int, default=30, help="rseek min phase bins")
    ap.add_argument("--bmax", type=int, default=120, help="rseek max phase bins")
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

    fft = os.path.abspath(args.fftfile)
    stem = fft[:-4] if fft.endswith(".fft") else fft
    inf = stem + ".inf"
    for p in (fft, inf):
        if not os.path.isfile(p):
            raise SystemExit(f"missing input: {p}")
    N, dt, T, dm = read_inf(inf)

    rseek = args.rseek or shutil.which("rseek") or (
        "/home/sransom/python_venvs/pixiPSR/.pixi/envs/default/bin/rseek")
    if not os.path.isfile(rseek):
        raise SystemExit("rseek not found; pass --rseek /path/to/rseek")

    # --- matched configuration -------------------------------------------
    nharms = args.nharms if args.nharms else args.bmax // 2
    maxdecim = args.maxdecim if args.maxdecim else max(1, args.bmax // args.bmin)
    threshold = args.threshold if args.threshold is not None else args.smin
    lofreq, hifreq = 1.0 / args.Pmax, 1.0 / args.Pmin

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
    print(f"              fundamentals {lofreq:.4g}-{hifreq:.4g} Hz, "
          f"{nbins_lo}-{nbins_hi} profile bins")
    if maxdecim > 1:
        print(f"              NOTE: decimation also covers up to {maxdecim * hifreq:.4g} Hz "
              f"({maxdecim}x), beyond rseek's band -- extra coverage AND extra work")
    if nbins_lo != args.bmin or nbins_hi != args.bmax:
        print(f"              NOTE: bin span {nbins_lo}-{nbins_hi} != rseek's "
              f"{args.bmin}-{args.bmax}; depth is not exactly matched")
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
              "--threshold", str(threshold), "--noplot", "--noprogress",
              "-o", out]
        if args.noharmremove:
            c += ["--noharmremove"]
        c += [fft]
        return c

    print(f"running coherent_search -t 1 ({args.repeat}x + warm-up) ...", flush=True)
    ct = run_timed(coherent_cmd(1), args.repeat)
    ccands = parse_cohout(out)

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
        print(f"  with {args.threads} threads:  {rt.min / mt.min:.2f}x rseek wall clock "
              f"(riptide's C extension is built without OpenMP, so rseek cannot")
        print(f"  {'':<15}use them -- this axis is ours alone, not a like-for-like win)")
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
