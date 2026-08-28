#!/usr/bin/env python3
"""
mc_simulate.py -- detection-efficiency Monte Carlo for CoherentSearch.jl.

Injects von-Mises-cored pulsars drawn from the MeerKAT TPA width population into
white noise, hands the SAME noise realisation to four codes, and records what
each one recovers.  Design settled in `monte_carlo.md` and
`Summary_and_Future_Work.md` 3.2; the population and profile model live in
`mc_profiles.py`.

  prepfold          folds at the KNOWN period (`-nosearch`) -- the reference a
                    reader will want, scored BOTH by its chi-squared sigma (the
                    standard) and by an snr1 boxcar on its .bestprof profile (so
                    there is one column in the same statistic as the searches)
  accelsearch       -numharm 16 -zmax 0, the standard incoherent harmonic sum
  rseek             riptide's FFA; config A always, the deep 4-range tiling on a
                    subset (measured 4.05x the cost)
  coherent_search   this repository, at its BARE DEFAULTS

**Statistic values are recorded, not booleans.**  Every code runs at a permissive
threshold and the best match to each injection is stored with its statistic, so
detection fraction at any threshold is a post-processing choice.  A fraction of
realisations get NO injections at all, which is what makes the codes comparable:
ours and rseek's thresholds are single-trial, accelsearch's sigma is already
trials-corrected and prepfold's is a chi-squared, so a common nominal threshold
means nothing and only a matched EMPIRICAL false-alarm rate does.

**False alarms are counted on every realisation too**, not just the empty ones: a
candidate matching no injection (nor a simple harmonic of one) is a false alarm,
and the count is reported alongside the trials each code searched so the rates
are comparable per unit of searching.

Output is one JSON object per line per realisation.  Combining runs from several
machines is `cat`; re-running is safe because each line carries its realisation
index and the driver skips indices already present.  Seeds come from
`SeedSequence(master).spawn()`, so a realisation's noise depends only on its
index -- workers need no coordination and results are reproducible.

Usage:

    # one worker, 20 realisations, to ./mcout
    mc_simulate.py --outdir mcout --nreal 20

    # fill a machine: N independent workers partitioning the index space
    mc_simulate.py --outdir mcout --nreal 100000 --workers 48

    # a single worker of an externally-managed pool
    mc_simulate.py --outdir mcout --nreal 100000 --nworkers 48 --worker 7
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_profiles as MP

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Our shipped defaults, and the rseek configuration that MATCHES their coverage.
# nharms 60 x maxdecim 6 above hifreq 125 Hz reaches 750 Hz, so rseek's Pmin is
# 1/750 -- and at bmin 20 that is above its own tsamp*bmin floor for dt = 60 us.
# Getting this wrong (Pmin = 1/hifreq) makes us search 6x its band and is the
# mistake `compare_riptide.py`'s docstring records having made once already.
COH_LOFREQ, COH_HIFREQ, COH_NHARMS, COH_MAXDECIM = 0.1, 125.0, 60, 6
RSEEK_A = dict(Pmin=1.0 / (COH_HIFREQ * COH_MAXDECIM), Pmax=10.0, bmin=20, bmax=120)
# riptide-at-its-best: narrow bins ranges where periods are long, but a WIDE one
# at the short end -- a literal pipeline-style tiling there pins b near 22 and is
# worse than config A (measured 9.6 vs 11.8 on the same 271 Hz pulsar), because
# bmin 20/bmax 120 lets b climb to the sampling limit.
RSEEK_B = [dict(Pmin=1.0 / (COH_HIFREQ * COH_MAXDECIM), Pmax=0.010, bmin=20, bmax=120),
           dict(Pmin=0.010, Pmax=0.05, bmin=160, bmax=174),
           dict(Pmin=0.05, Pmax=0.25, bmin=480, bmax=520),
           dict(Pmin=0.25, Pmax=10.0, bmin=960, bmax=1040)]

INF_TEMPLATE = """ Data file name without suffix          =  {stem}
 Telescope used                         =  GBT
 Instrument used                        =  Fake
 Object being observed                  =  MCsim
 J2000 Right Ascension (hh:mm:ss.ssss)  =  00:00:00.0000
 J2000 Declination     (dd:mm:ss.ssss)  =  00:00:00.0000
 Data observed by                       =  MonteCarlo
 Epoch of observation (MJD)             =  60000.000000000000000
 Barycentered?           (1 yes, 0 no)  =  1
 Number of bins in the time series      =  {N}
 Width of each time series bin (sec)    =  {dt:.10g}
 Any breaks in the data? (1 yes, 0 no)  =  0
 Type of observation (EM band)          =  Radio
 Beam diameter (arcsec)                 =  981
 Dispersion measure (cm-3 pc)           =  0.0
 Central freq of low channel (MHz)      =  1000.0
 Total bandwidth (MHz)                  =  800
 Number of channels                     =  1024
 Channel bandwidth (MHz)                =  0.78125
 Data analyzed by                       =  MonteCarlo
 Any additional notes:
    Synthetic white-noise realisation for the sensitivity Monte Carlo.
"""


# ---------------------------------------------------------------------------
# Shelling out
# ---------------------------------------------------------------------------
def run(cmd, cwd=None, env=None):
    """Run `cmd`, returning (wall_seconds, stdout, stderr, returncode)."""
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)
    return time.perf_counter() - t0, p.stdout, p.stderr, p.returncode


# ---------------------------------------------------------------------------
# Candidate parsing.  One `Cand` shape for every code so the scorer is generic.
# ---------------------------------------------------------------------------
class Cand:
    __slots__ = ("freq", "stat", "ducy", "extra")

    def __init__(self, freq, stat, ducy=None, extra=None):
        self.freq = float(freq)
        self.stat = float(stat)
        self.ducy = ducy
        self.extra = extra or {}


def parse_rseek(text):
    out = []
    for line in text.splitlines():
        f = line.split()
        if len(f) != 6 or f[0].startswith("period"):
            continue
        try:
            out.append(Cand(float(f[1]), float(f[5]), float(f[3].rstrip("%")) / 100.0,
                            {"width": int(f[2])}))
        except ValueError:
            continue
    return out


def parse_cohout(path):
    out = []
    if not os.path.isfile(path):
        return out
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.split()
            if len(f) < 5:
                continue
            try:
                ducy = float(f[5]) / 100.0 if len(f) >= 6 and f[5] != "-" else None
                out.append(Cand(float(f[2]), float(f[1]), ducy, {"nharm": int(f[4])}))
            except ValueError:
                continue
    return out


def parse_accel(path):
    """ACCEL_0 -> candidates carrying PRESTO's SINGLE-TRIAL sigma.

    `presto.sifting` recomputes `candidate_sigma(opt_ipow, 1, 1)` from the
    optimised harmonic powers, which is the number to compare -- not the sigma
    printed in the ACCEL file, which carries accelsearch's own trials
    correction.  `prelim_reject=False`: the default rejection applies a sigma
    cut, and this study needs the sub-threshold values.
    """
    if not os.path.isfile(path):
        return []
    try:
        from presto import sifting
    except ImportError:
        return []
    try:
        cl = sifting.candlist_from_candfile(path)
    except Exception:
        return []
    return [Cand(c.f, c.sigma, None, {"numharm": c.numharm, "snr": c.snr})
            for c in cl.cands if getattr(c, "sigma", None) is not None]


# ---------------------------------------------------------------------------
# prepfold scoring
# ---------------------------------------------------------------------------
_SIGMA_RE = re.compile(r"\(~([\d.]+)\s*sigma\)")


def boxcar_widths(nbins, fsp=1.5, maxfrac=0.3):
    out, w = [], 1
    while w <= max(1, int(maxfrac * nbins)):
        out.append(w)
        w = int(max(w + 1, fsp * w))
    return out


def snr1(prof):
    """riptide's `snr1` on a folded profile: peak over the zero-mean unit-L2
    boxcar bank, with a robust baseline and scale.

    Plain `snr1` is CORRECT here, with no band-limited correction: prepfold folds
    in the time domain, so its phase bins really are independent -- which is the
    whole distinction that made our own Fourier fold need `_boxcar_shape!`.
    """
    p = np.asarray(prof, dtype=float)
    n = len(p)
    p = p - np.median(p)
    sd = 1.4826 * np.median(np.abs(p - np.median(p)))
    if sd <= 0:
        return float("nan"), None
    c = np.concatenate([p, p]).cumsum()
    c = np.concatenate([[0.0], c])
    best, bestw = -np.inf, 1
    for w in boxcar_widths(n):
        s = c[w:w + n] - c[0:n]
        d = w / n
        v = s.max() / (sd * math.sqrt(w * (1.0 - d)))
        if v > best:
            best, bestw = v, w
    return float(best), bestw / n


def read_bestprof(path):
    sigma, prof = float("nan"), []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                m = _SIGMA_RE.search(line)
                if m:
                    sigma = float(m.group(1))
            elif line.strip():
                try:
                    prof.append(float(line.split()[1]))
                except (IndexError, ValueError):
                    pass
    return sigma, prof


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def harmonic_label(fc, f0, T, tol_bins, nmax=8):
    """Is `fc` a simple ratio n/m of `f0`?  -> "n/m", or None.

    Tolerance is in Fourier bins (|df|*T), the natural unit -- it is the
    observation's own resolution and so is frequency independent.
    """
    for n in range(1, nmax + 1):
        for m in range(1, nmax + 1):
            if abs(fc - f0 * n / m) * T <= tol_bins:
                if n == 1 and m == 1:
                    return "1"
                if math.gcd(n, m) != 1:
                    continue
                return f"{n}/{m}"
    return None


def score(cands, injections, T, tol_bins):
    """Best match per injection, plus the false-alarm count.

    A candidate is a false alarm only if it matches NO injection at any simple
    harmonic ratio -- otherwise every real detection's own 2f and f/2 entries
    (which rseek and accelsearch do not collapse and we do) would be counted as
    false positives, which would be nonsense.
    """
    hits = []
    claimed = set()
    for inj in injections:
        best, best_lab = None, None
        for i, c in enumerate(cands):
            lab = harmonic_label(c.freq, inj["f0"], T, tol_bins)
            if lab is None:
                continue
            claimed.add(i)
            # Prefer the fundamental over a harmonic, then the strongest.
            key = (lab != "1", -c.stat)
            if best is None or key < (best_lab != "1", -best.stat):
                best, best_lab = c, lab
        hits.append(None if best is None else
                    dict(stat=best.stat, freq=best.freq, ducy=best.ducy,
                         harmonic=best_lab, **best.extra))
    fa = [c for i, c in enumerate(cands) if i not in claimed]
    return hits, fa


def fa_summary(fa):
    """False alarms, as a count and as the sorted statistics of the top few.

    Keeping the top values (not just a count at one threshold) is what lets the
    analysis build a false-alarm RATE curve per code without re-running anything.
    """
    s = sorted((c.stat for c in fa), reverse=True)
    # 200, not 20: rseek at `--smin 6` reports ~150 candidates on a 2^24 file,
    # so a short tail saturates exactly where the false-alarm RATE is
    # interesting.  A truncated list reads as a rate ceiling, not as missing
    # data, which is the kind of quiet wrong answer that is hard to spot later.
    return dict(n=len(s), top=[round(v, 3) for v in s[:200]], truncated=len(s) > 200)


# ---------------------------------------------------------------------------
# One realisation
# ---------------------------------------------------------------------------
def realisation(idx, args, pop, tools, seedseq):
    rng = np.random.default_rng(seedseq)
    N, dt = args.nsamp, args.dt
    T = N * dt
    empty = (idx % args.noise_every) == 0 if args.noise_every > 0 else False
    ninj = 0 if empty else args.injections

    rec = dict(index=idx, host=os.uname().nodename, t_start=time.time(),
               N=N, dt=dt, T=T, empty=empty, injections=[], timing={}, results={},
               config=dict(coh=dict(lofreq=COH_LOFREQ, hifreq=COH_HIFREQ,
                                    nharms=COH_NHARMS, maxdecim=COH_MAXDECIM,
                                    threshold=args.threshold),
                           rseek_A=RSEEK_A, tol_bins=args.tol_bins))

    # --- draw, rejecting harmonic or near-coincident pairs -----------------
    draws = []
    while len(draws) < ninj:
        d = pop.draw(rng, dt, args.min_fwhm_samples)
        f0 = 1.0 / d["period"]
        if any(harmonic_label(f0, e["f0"], T, 5 * args.tol_bins) for e in draws):
            continue                      # would be indistinguishable at scoring
        d["f0"] = f0
        d["snr"] = float(rng.choice(args.snrs))
        d["phase0"] = float(rng.random())
        draws.append(d)

    # --- build the time series ---------------------------------------------
    t0 = time.perf_counter()
    x = rng.normal(size=N)
    for d in draws:
        prof, info = MP.make_profile(d["ducy"], d["w10_w50"])
        MP.inject(x, dt, d["period"], d["phase0"], prof, d["snr"])
        d["profile"] = dict(tau=info["tau"], ped=info["ped"],
                            ratio_got=info["ratio"], ducy_got=info["ducy"],
                            exact=info["exact"])
        rec["injections"].append(d)
    stem = os.path.join(args.workdir, f"mc{idx:07d}_DM00.00")
    x.astype(np.float32).tofile(stem + ".dat")
    with open(stem + ".inf", "w") as fh:
        fh.write(INF_TEMPLATE.format(stem=os.path.basename(stem), N=N, dt=dt))
    rec["timing"]["generate"] = time.perf_counter() - t0

    try:
        _search_all(rec, draws, stem, args, tools, T)
    finally:
        if not args.keep:
            for f in glob.glob(stem + "*"):
                try:
                    os.remove(f)
                except OSError:
                    pass
    rec["t_end"] = time.time()
    return rec


def _search_all(rec, draws, stem, args, tools, T):
    inj = [dict(f0=d["f0"]) for d in draws]
    tol = args.tol_bins
    # **Run every PRESTO tool with cwd = the work directory, and give it BARE
    # NAMES.**  `rednoise` writes `<stem>_red.inf` into the CURRENT DIRECTORY
    # rather than beside its input, so with the driver launched from the repo
    # the .fft landed in the work directory and its .inf did not -- and
    # `coherent_search` then died with "The .inf file ... was not found" on every
    # single realisation while the other three codes carried on, which reads as a
    # 0% detection fraction rather than as a crash.  It also littered the repo
    # root with one stray file per realisation.
    wd = os.path.dirname(stem)
    base = os.path.basename(stem)

    # --- FFT + de-redden ---------------------------------------------------
    # `rednoise` also NORMALISES the powers to mean 1 (verified: 1.0004 on white
    # noise), which is what our analytic sigma requires.  accelsearch gets the
    # raw .fft -- it does its own local normalisation.
    t, _, _, rc = run([tools["realfft"], base + ".dat"], cwd=wd)
    rec["timing"]["realfft"] = t
    t, _, _, rc = run([tools["rednoise"], base + ".fft"], cwd=wd)
    rec["timing"]["rednoise"] = t
    # Belt and braces: if a PRESTO build still does not produce it, the search
    # cannot run, and a missing .inf must not be allowed to look like a miss.
    if not os.path.isfile(stem + "_red.inf") and os.path.isfile(stem + ".inf"):
        shutil.copyfile(stem + ".inf", stem + "_red.inf")

    # --- prepfold, one fold per injection at the KNOWN period ---------------
    pf = []
    t_pf = 0.0
    for i, d in enumerate(draws):
        o = f"{stem}_pf{i}"
        cmd = [tools["prepfold"], "-f", repr(d["f0"]), "-nosearch", "-fine",
               "-noxwin", "-nopdsearch", "-o", os.path.basename(o)]
        # PRESTO picks its own bin count for short periods (it cannot exceed
        # P/dt); 128 is the right fixed choice for everything slower.
        if not d["msp"]:
            cmd += ["-n", "128"]
        cmd += [base + ".dat"]
        dtm, _, _, rc = run(cmd, cwd=wd)
        t_pf += dtm
        best = glob.glob(o + "*.bestprof")
        if rc != 0 or not best:
            pf.append(None)
            continue
        sig, prof = read_bestprof(best[0])
        s1, ducy = snr1(prof) if prof else (float("nan"), None)
        pf.append(dict(chi2_sigma=sig, snr1=s1, ducy=ducy, nbins=len(prof)))
    rec["timing"]["prepfold"] = t_pf
    rec["results"]["prepfold"] = pf          # targeted: no false-alarm column

    # --- accelsearch --------------------------------------------------------
    t, _, _, rc = run([tools["accelsearch"], "-numharm", "16", "-zmax", "0",
                       "-sigma", str(args.accel_sigma), base + ".fft"], cwd=wd)
    rec["timing"]["accelsearch"] = t
    ac = parse_accel(stem + "_ACCEL_0")
    hits, fa = score(ac, inj, T, tol)
    rec["results"]["accelsearch"] = dict(hits=hits, false=fa_summary(fa), ncand=len(ac))

    # --- rseek, config A ----------------------------------------------------
    def rseek(cfg):
        return run([tools["rseek"], "--Pmin", repr(cfg["Pmin"]), "--Pmax", repr(cfg["Pmax"]),
                    "--bmin", str(cfg["bmin"]), "--bmax", str(cfg["bmax"]),
                    "--smin", str(args.rseek_smin), "-f", "presto", base + ".inf"], cwd=wd)
    t, out, _, rc = rseek(RSEEK_A)
    rec["timing"]["rseek_A"] = t
    rs = parse_rseek(out) if rc == 0 else []
    hits, fa = score(rs, inj, T, tol)
    rec["results"]["rseek_A"] = dict(hits=hits, false=fa_summary(fa), ncand=len(rs),
                                     ok=(rc == 0))

    # --- rseek, deep tiling, on a subset ------------------------------------
    if args.deep_every > 0 and (rec["index"] % args.deep_every) == 0:
        allc, tt, ok = [], 0.0, True
        for cfg in RSEEK_B:
            t, out, _, rc = rseek(cfg)
            tt += t
            ok &= (rc == 0)
            allc += parse_rseek(out) if rc == 0 else []
        rec["timing"]["rseek_B"] = tt
        hits, fa = score(allc, inj, T, tol)
        rec["results"]["rseek_B"] = dict(hits=hits, false=fa_summary(fa),
                                         ncand=len(allc), ok=ok)
        rec["config"]["rseek_B"] = RSEEK_B

    # --- coherent_search, BARE DEFAULTS ------------------------------------
    out_f = stem + ".cohout"
    cmd = [tools["julia"], f"--project={REPO}", f"-t{args.coh_threads}",
           os.path.join(REPO, "bin", "coherent_search.jl"),
           "--threshold", str(args.threshold), "--noprogress",
           "--ncands", str(args.ncands), "-o", out_f, stem + "_red.fft"]
    t, _, err, rc = run(cmd)
    rec["timing"]["coherent"] = t
    ch = parse_cohout(out_f) if rc == 0 else []
    hits, fa = score(ch, inj, T, tol)
    rec["results"]["coherent"] = dict(hits=hits, false=fa_summary(fa), ncand=len(ch),
                                      ok=(rc == 0))
    if rc != 0:
        rec["results"]["coherent"]["stderr"] = err[-2000:]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def which(name, override, extra=()):
    if override:
        return override
    p = shutil.which(name)
    if p:
        return p
    for d in extra:
        c = os.path.join(d, name)
        if os.path.isfile(c):
            return c
    raise SystemExit(f"cannot find {name}; pass --{name}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--nreal", type=int, default=10, help="realisations to run (this worker's share is 1/nworkers of them)")
    ap.add_argument("--start", type=int, default=0, help="first realisation index")
    ap.add_argument("--workers", type=int, default=0, help="spawn this many worker processes")
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--master-seed", type=int, default=20260828)
    ap.add_argument("--injections", type=int, default=6)
    ap.add_argument("--noise-every", type=int, default=10,
                    help="every Nth realisation gets NO injections (false-alarm calibration); 0 disables")
    ap.add_argument("--deep-every", type=int, default=3,
                    help="every Nth realisation also gets the deep rseek tiling (4.05x its cost); 0 disables")
    ap.add_argument("--nsamp", type=int, default=1 << 24)
    ap.add_argument("--dt", type=float, default=60.0e-6)
    ap.add_argument("--snrs", type=float, nargs="+", default=[6, 7, 8, 9, 10, 11])
    ap.add_argument("--min-fwhm-samples", type=float, default=3.0)
    ap.add_argument("--threshold", type=float, default=6.0, help="coherent_search S/N floor")
    ap.add_argument("--ncands", type=int, default=500)
    ap.add_argument("--rseek-smin", type=float, default=6.0)
    ap.add_argument("--accel-sigma", type=float, default=1.0)
    ap.add_argument("--tol-bins", type=float, default=3.0)
    ap.add_argument("--coh-threads", type=int, default=1)
    ap.add_argument("--workdir", default=None, help="scratch for the ~200 MB of transient files per realisation (default: /dev/shm if present)")
    ap.add_argument("--tpa", default=None, help="TPA table_1.csv; without it the recorded quantiles are used")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--nothreadpin", action="store_true",
                    help="do NOT pin the child thread pools to 1 (see below)")
    ap.add_argument("--julia", default="julia")
    ap.add_argument("--presto-bin", default=None, help="directory holding realfft/rednoise/prepfold/accelsearch")
    ap.add_argument("--rseek", default=None)
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)

    # Pin every child's thread pool to 1, inherited through the environment.
    #
    # The parallelism here is one single-threaded worker per core, so a child
    # that helpfully spawns a pool per process is pure contention.  Measured on
    # bla0 (48 cores, 96 threads): rseek's numpy/BLAS ran **96 threads per
    # worker** at ~240% CPU, and 48 of those put the load average at 350.
    #
    # `compare/compare_riptide.py` records that pinning riptide's pools changed
    # its wall clock by less than the run-to-run scatter -- but that was ONE
    # process on a 4-core laptop, which is the opposite regime.  Do not read
    # that note as covering this case.
    if not args.nothreadpin:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                  "BLIS_NUM_THREADS"):
            os.environ[v] = "1"

    if args.workers > 0:
        # Strip `--workers N` as a PAIR.  Filtering out every argument equal to
        # `str(args.workers)` also deletes an unrelated option that happens to
        # share the value -- `--workers 6 --injections 6` would have silently
        # lost the 6 from `--injections` and taken its default.
        base = list(argv if argv is not None else sys.argv[1:])
        rest, skip = [], False
        for a in base:
            if skip:
                skip = False
                continue
            if a == "--workers":
                skip = True
                continue
            if a.startswith("--workers="):
                continue
            rest.append(a)
        procs = []
        for w in range(args.workers):
            cmd = [sys.executable, os.path.abspath(__file__)] + rest + \
                  ["--nworkers", str(args.workers), "--worker", str(w)]
            procs.append(subprocess.Popen(cmd))
        rc = 0
        for p in procs:
            rc |= p.wait()
        return rc

    extra = [args.presto_bin] if args.presto_bin else []
    tools = dict(julia=args.julia,
                 rseek=which("rseek", args.rseek, extra),
                 realfft=which("realfft", None, extra),
                 rednoise=which("rednoise", None, extra),
                 prepfold=which("prepfold", None, extra),
                 accelsearch=which("accelsearch", None, extra))

    workdir = args.workdir or ("/dev/shm" if os.path.isdir("/dev/shm") else "/tmp")
    workdir = os.path.join(workdir, f"mcsim_{os.getpid()}")
    os.makedirs(workdir, exist_ok=True)
    args.workdir = workdir

    tpa = MP.load_tpa(args.tpa) if args.tpa else None
    pop = MP.Population(tpa=tpa)

    outfile = os.path.join(args.outdir, f"mc_{os.uname().nodename}_{args.worker:03d}.jsonl")
    done = set()
    if os.path.isfile(outfile):
        with open(outfile) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["index"])
                except Exception:
                    pass
    # Seeds depend ONLY on the realisation index, so workers need no
    # coordination, a rerun reproduces the same noise, and two workers that
    # overlap produce identical rows rather than silently different ones.
    root = np.random.SeedSequence(args.master_seed)

    idxs = [args.start + i for i in range(args.nreal)
            if (args.start + i) % args.nworkers == args.worker]
    idxs = [i for i in idxs if i not in done]
    print(f"[worker {args.worker}] {len(idxs)} realisations -> {outfile}", flush=True)

    try:
        with open(outfile, "a", buffering=1) as fh:
            for n, idx in enumerate(idxs):
                seed = np.random.SeedSequence(entropy=root.entropy, spawn_key=(idx,))
                t0 = time.perf_counter()
                try:
                    rec = realisation(idx, args, pop, tools, seed)
                except Exception as exc:      # one bad realisation must not end the run
                    rec = dict(index=idx, error=repr(exc), host=os.uname().nodename)
                fh.write(json.dumps(rec) + "\n")
                print(f"[worker {args.worker}] {n + 1}/{len(idxs)} idx={idx} "
                      f"{time.perf_counter() - t0:.1f}s", flush=True)
                # ABORT on a search that failed in the FIRST realisation.  A code
                # that never runs contributes no candidates, which is
                # indistinguishable from a code that detects nothing -- it shows
                # up as a 0% detection fraction, not as a crash.  That happened
                # once already (rednoise writes `_red.inf` into the CWD, so
                # coherent_search could not find it) and it would otherwise have
                # burned a whole night.  Better to stop now than to produce a
                # confident, entirely wrong table in the morning.
                if n == 0:
                    dead = [m for m in ("rseek_A", "rseek_B", "coherent")
                            if m in rec.get("results", {})
                            and not rec["results"][m].get("ok", True)]
                    if dead or "error" in rec:
                        sys.stderr.write(
                            f"\n*** ABORTING: {dead or rec.get('error')} failed on the "
                            f"first realisation.  Fix that before running a batch -- a "
                            f"failing search reads as 0% detection, not as an error.\n")
                        for m in dead:
                            sys.stderr.write(rec["results"][m].get("stderr", "")[-1200:] + "\n")
                        return 2
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
