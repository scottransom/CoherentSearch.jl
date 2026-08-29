#!/usr/bin/env python3
"""
mc_simulate.py -- detection-efficiency Monte Carlo for CoherentSearch.jl.

Injects von-Mises-cored pulsars drawn from the MeerKAT TPA width population into
white noise, hands the SAME noise realisation to four codes, and records what
each one recovers.  Design settled in `docs/monte_carlo.md` and
`docs/Summary_and_Future_Work.md` 3.2; the population and profile model live in
`mc_profiles.py`.

  prepfold          folds at the KNOWN period (`-nosearch`) -- the reference a
                    reader will want, scored BOTH by its chi-squared sigma (the
                    standard) and by an snr1 boxcar on its .bestprof profile (so
                    there is one column in the same statistic as the searches).
                    On the injection-free realisations it folds at periods drawn
                    from the same population, which gives both of those a
                    MEASURED null instead of a nominal cut.
  accelsearch       -numharm 16 -zmax 0, the standard incoherent harmonic sum, on
                    the raw .fft AND on the de-reddened one
  rseek             riptide's FFA; config A always, the deep 4-range tiling on a
                    subset (measured 4.05x the cost)
  coherent_search   this repository: the shipped defaults, plus a low-frequency
                    DEEP tier every realisation and a full-band deep arm on a
                    subset (see COH_ARMS)

**Injected S/N is continuous and the duty cycle is stratified.**  Run 1 used six
integer S/N values and the bare TPA population; the first spent all its resolution
on six points when what the paper wants is a fitted S/N at 50% detection, and the
second drew only 757 injections below 0.5% duty -- the one place we lose.  Duty is
now over-sampled there and every injection carries the importance `weight` that
undoes it, so population-weighted numbers are unchanged and the narrow-duty cells
are several times better determined.

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
import signal
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_profiles as MP
import mc_model as MM

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Our shipped defaults, and the rseek configuration that MATCHES their coverage.
# nharms 60 x maxdecim 6 above hifreq 125 Hz reaches 750 Hz, so rseek's Pmin is
# 1/750 -- and at bmin 20 that is above its own tsamp*bmin floor for dt = 60 us.
# Getting this wrong (Pmin = 1/hifreq) makes us search 6x its band and is the
# mistake `compare_riptide.py`'s docstring records having made once already.
COH_LOFREQ, COH_HIFREQ, COH_NHARMS, COH_MAXDECIM = 0.1, 125.0, 60, 6

# The coherent arms.  `every` is how often each one runs (1 = every realisation);
# `--deep-coh-every` overrides the third.
#
#   coherent       the shipped defaults, the arm everything else is measured
#                  against.
#   coherent_tier  section 4's proposal: a DEEP fold restricted to low frequency.
#                  `nharms x maxdecim` is unchanged, so `hifreq 5` x 12 still
#                  reaches 60 Hz; what changes is that a fundamental below 5 Hz is
#                  folded into 240 bins instead of 120, which is where the whole
#                  duty < 0.5% deficit lives.  It is a separate INVOCATION, so it
#                  carries its own false-alarm tail -- which is the point: a deep
#                  tier has to pay for its own trials, and `mc_analyze` forms
#                  `coh+tier` by concatenating the two candidate lists, so the
#                  union's threshold is measured and not assumed.
#   coherent_deep  the same depth over the WHOLE band, on a subset, purely to
#                  measure section 4's prediction that a blanket deep search is a net
#                  LOSS (2-4x the trials, and worse at 5-12% duty because the
#                  shallowest rung becomes H=20 instead of H=10).
COH_ARMS = {
    "coherent":      dict(lofreq=0.1, hifreq=125.0, nharms=60,  maxdecim=6,  every=1),
    "coherent_tier": dict(lofreq=0.1, hifreq=5.0,   nharms=120, maxdecim=12, every=1),
    "coherent_deep": dict(lofreq=0.1, hifreq=62.5,  nharms=120, maxdecim=12, every=5),
}

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
    """`period freq width ducy dm snr` -> candidates, carrying the FOLD DEPTH.

    `rseek` does not print `b`, but it prints `width` and `ducy = width / b`, so
    `b` is exact arithmetic on two printed columns.  It matters: without it an
    rseek miss at narrow duty cannot be attributed between the downsampling
    sawtooth (which sets how deep it folded) and the `bins_min` width bank (which
    caps its widest boxcar at 6 regardless of depth).
    """
    out = []
    for line in text.splitlines():
        f = line.split()
        if len(f) != 6 or f[0].startswith("period"):
            continue
        try:
            w = int(f[2])
            ducy = float(f[3].rstrip("%")) / 100.0
            out.append(Cand(float(f[1]), float(f[5]), ducy,
                            {"width": w,
                             "nbins": int(round(w / ducy)) if ducy > 0 else None}))
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


def snr1(prof, sigma=None):
    """riptide's `snr1` on a folded profile: peak over the zero-mean unit-L2
    boxcar bank, with a robust baseline.  Returns `(snr1, width, sigma_used)`.

    `sigma` overrides the per-profile MAD.  Use it: the MAD carries ~11% scatter
    on the 31-128 bins prepfold picks, and `snr1` is exactly `1/sigma-hat`, so it
    puts a scatter term on prepfold's column comparable to everything else in it
    -- and prepfold is the CEILING the searches are measured against, so noise
    there is not neutral.  `mc_model.prepfold_sigma` computes it in closed
    form from the fold's own drizzle weights.

    **No band-limited correction is applied here** -- prepfold folds in the time
    domain, so this is the right `snr1`.  What its bins are NOT is independent
    (the drizzle correlates neighbours), and that correction is applied at
    analysis time from the `(nbins, dt_per_bin, w)` recorded beside this value;
    see `mc_model.drizzle_boxcar_corr`.  It is left out of the recorded number on
    purpose, so the raw statistic stays in the file and a revision of the
    correction does not mean re-running a week of compute.
    """
    p = np.asarray(prof, dtype=float)
    n = len(p)
    p = p - np.median(p)
    sd = sigma if sigma else 1.4826 * np.median(np.abs(p - np.median(p)))
    if not sd or sd <= 0:
        return float("nan"), None, float("nan")
    c = np.concatenate([p, p]).cumsum()
    c = np.concatenate([[0.0], c])
    best, bestw = -np.inf, 1
    for w in boxcar_widths(n):
        s = c[w:w + n] - c[0:n]
        d = w / n
        v = s.max() / (sd * math.sqrt(w * (1.0 - d)))
        if v > best:
            best, bestw = v, w
    return float(best), bestw, float(sd)


def score_bestprof(sigma_chi2, prof, f0, args, keep_profile=False):
    """One prepfold fold, scored every way the analysis needs.

    Records BOTH sigma estimates and the RAW `snr1`, plus `(nbins, dt_per_bin,
    w)`.  Two reasons, both learned from run 1: the drizzle correction is applied
    at analysis time (so revising it does not mean re-folding a week of data), and
    the MAD it used to divide by is noisy: measured over 144 null folds, the two
    sigmas agree in the MEDIAN to 0.5% and scatter by **11%** about it (prepfold
    picks 31 to 128 bins, and a MAD on 31 points is poor).  `snr1` is exactly
    `1/sigma-hat`, so that 11% landed on every value in the column the searches
    are measured against.  Recording both lets the run itself settle which to
    quote rather than the question being decided in advance.
    """
    if not prof:
        return None
    n = len(prof)
    dpb = (1.0 / f0) / n / args.dt          # samples per profile bin
    s_ana = MM.prepfold_sigma(n, dpb, args.nsamp)
    s1_mad, w_mad, s_mad = snr1(prof)
    s1_ana, w_ana, _ = snr1(prof, sigma=s_ana)
    out = dict(chi2_sigma=sigma_chi2, nbins=n, dt_per_bin=dpb,
               snr1=s1_ana, w=w_ana, ducy=(w_ana / n if w_ana else None),
               snr1_mad=s1_mad, w_mad=w_mad,
               sigma_mad=s_mad, sigma_analytic=s_ana)
    if keep_profile:
        out["prof"] = [round(float(v), 4) for v in prof]
    return out


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


def fa_summary(fa, cap=800):
    """False alarms, as a count and as the sorted statistics of the top few.

    Keeping the top values (not just a count at one threshold) is what lets the
    analysis build a false-alarm RATE curve per code without re-running anything.

    **800, not run 1's 200.**  `rseek_B` reports a median of 432 candidates and up
    to 503 on a 2^24 file at `--smin 6`, so 200 truncated its tail on **100% of
    realisations** and its measured rate curve was a ceiling below 6.20.  That
    never touched a threshold anyone used -- the loosest is 7.25 -- but it is a
    censoring WE imposed, on the one arm that costs 121 s to produce, and the
    storage is a few kB per realisation.  A truncated list reads as a rate
    ceiling rather than as missing data, which is the kind of quiet wrong answer
    that is hard to spot later, so the cheap fix is the right one.
    """
    s = sorted((c.stat for c in fa), reverse=True)
    return dict(n=len(s), top=[round(v, 3) for v in s[:cap]], truncated=len(s) > cap,
                floor=(round(s[-1], 3) if s else None))


def add_rednoise(x, dt, fcorner, alpha, rng):
    """Add a power-law red-noise component to `x` in place.

    `fcorner` is where the red power equals the white: `P(f) = (fcorner/f)^alpha`
    for `f > 0`, so the added variance is roughly `fcorner^alpha * T^(alpha-1)`
    and only values well below `1/sqrt(T)` are sane -- at `alpha = 2` and
    `T = 1006 s`, `fcorner = 0.01 Hz` adds ~10% of the white variance and
    `fcorner = 1 Hz` adds a thousand times it.

    OFF by default and untouched by run 2 -- everything measured so far is pure
    white noise, which is the regime where our analytic sigma is exactly right and
    where riptide's per-profile sigma-hat has no compensating advantage, so a
    red-noise run is a SEPARATE study whose numbers must not be pooled with the
    white ones (the FAP calibration and the paired-noise design both have to be
    redone per subset).  The flag exists so that study is one
    argument away rather than a fresh code change.
    """
    n = len(x)
    F = np.fft.rfft(rng.normal(size=n))
    f = np.fft.rfftfreq(n, dt)
    g = np.zeros_like(f)
    g[1:] = (fcorner / f[1:]) ** (0.5 * alpha)
    F *= g
    x += np.fft.irfft(F, n)
    return x


def one_in(idx, n):
    """True on 1-in-`n` realisations, spread EVENLY over the workers.

    `idx % n` looks right and is not.  Worker `w` of `W` takes the indices with
    `idx % W == w`, so whenever `n` divides `W` -- 15 workers with
    `--deep-every 5`, or run 1's 48 with 3 -- `idx % n` is CONSTANT inside a
    worker: the entire subset lands on `W/n` of the workers, those workers are
    then the slow ones (the deep tiling is 121 s), and the finished subset comes
    out far under 1-in-n.  **Run 1 asked for 1-in-3 and finished with 14.4%**
    (2,197 of 15,212), which is exactly this and was read as normal attrition.

    Hashing the index decorrelates the selector from the worker stride while
    staying a pure function of the index, so a rerun still reproduces the same
    subset and workers still need no coordination.

    **It has to be a real mixer, not a multiply.**  A plain `idx * K` preserves
    the low bits -- `K mod 4 == 1` for the usual Knuth constant, so `(idx*K) % 4
    == idx % 4` and the whole failure comes straight back for every power-of-two
    `n`.  Caught by `test_mc.py` at `W = 20, n = 4`, where the per-worker counts
    were `0..300` against an ideal 75.  This is splitmix64's finaliser, which
    mixes every bit into every other.
    """
    if n <= 0:
        return False
    if n == 1:
        return True
    z = (idx + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return ((z ^ (z >> 31)) % n) == 0


# ---------------------------------------------------------------------------
# One realisation
# ---------------------------------------------------------------------------
def realisation(idx, args, pop, tools, seedseq):
    rng = np.random.default_rng(seedseq)
    N, dt = args.nsamp, args.dt
    T = N * dt
    empty = one_in(idx, args.noise_every)
    ninj = 0 if empty else args.injections

    rec = dict(index=idx, host=os.uname().nodename, t_start=time.time(),
               N=N, dt=dt, T=T, empty=empty, injections=[], timing={}, results={},
               config=dict(coh=COH_ARMS, rseek_A=RSEEK_A, tol_bins=args.tol_bins,
                           threshold=args.threshold, trials=tools.get("trials"),
                           strat=args.strat, rednoise=args.rednoise))

    # On an injection-free realisation, draw the SAME number of pulsars anyway --
    # not to inject them, but to give prepfold somewhere to fold.  Folding pure
    # noise at periods from the real population is what turns `snr1` and
    # `chi2_sigma` from nominal cuts into measured nulls.
    null_draws = []
    if empty:
        while len(null_draws) < args.injections:
            d = pop.draw(rng, dt, args.min_fwhm_samples)
            d["f0"] = 1.0 / d["period"]
            null_draws.append(d)

    # --- draw, rejecting harmonic or near-coincident pairs -----------------
    draws = []
    while len(draws) < ninj:
        d = pop.draw(rng, dt, args.min_fwhm_samples)
        f0 = 1.0 / d["period"]
        if any(harmonic_label(f0, e["f0"], T, 5 * args.tol_bins) for e in draws):
            continue                      # would be indistinguishable at scoring
        d["f0"] = f0
        # **Continuous, not a six-point grid.**  Run 1's integers bracketed the
        # curve well (15% at 6, 99.6% at 11) but spent all their resolution on six
        # points, and what the paper wants is the S/N at 50% detection -- which is
        # a FIT, and fits better to a continuous covariate.
        d["snr"] = (float(rng.uniform(*args.snr_range)) if args.snrs is None
                    else float(rng.choice(args.snrs)))
        d["phase0"] = float(rng.random())
        draws.append(d)

    # --- build the time series ---------------------------------------------
    t0 = time.perf_counter()
    x = rng.normal(size=N)
    if args.rednoise:
        add_rednoise(x, dt, args.rednoise, args.rednoise_alpha, rng)
    for d in draws:
        prof, info = MP.make_profile(d["ducy"], d["w10_w50"])
        MP.inject(x, dt, d["period"], d["phase0"], prof, d["snr"])
        d["profile"] = dict(tau=info["tau"], ped=info["ped"],
                            ratio_got=info["ratio"], ducy_got=info["ducy"],
                            exact=info["exact"])
        # The section-4 band-limited efficiency of the arm we ship, for THIS pulse
        # shape at THIS frequency: what fraction of the injected S/N a 120-bin
        # fold truncated at the data's own Nyquist can recover.  ~1 ms, and it
        # lets the analysis normalise recovery out and see what is left.  The
        # profile is already built here, so this costs no extra `make_profile`.
        A = MM.profile_harmonics(prof, COH_NHARMS)
        hmax = min(COH_NHARMS, int(math.floor(0.5 / (dt * d["f0"]))))
        d["model_eff"] = MM.ladder_efficiency(A, COH_NHARMS, COH_MAXDECIM, hmax)[0]
        rec["injections"].append(d)
    stem = os.path.join(args.workdir, f"mc{idx:07d}_DM00.00")
    x.astype(np.float32).tofile(stem + ".dat")
    with open(stem + ".inf", "w") as fh:
        fh.write(INF_TEMPLATE.format(stem=os.path.basename(stem), N=N, dt=dt))
    rec["timing"]["generate"] = time.perf_counter() - t0

    try:
        _search_all(rec, draws, null_draws, stem, args, tools, T,
                    keep_prof=(args.keep_profiles > 0
                               and (empty or one_in(idx, args.keep_profiles))))
    finally:
        if not args.keep:
            for f in glob.glob(stem + "*"):
                try:
                    os.remove(f)
                except OSError:
                    pass
    rec["t_end"] = time.time()
    return rec


def _search_all(rec, draws, null_draws, stem, args, tools, T, keep_prof=False):
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
    # On an INJECTION-FREE realisation the same folds are done at periods drawn
    # from the same population.  That is what gives `snr1` and `chi2_sigma` a
    # MEASURED null, so prepfold can join the FAP-matched table as a proper
    # ceiling instead of sitting beside it at a nominal cut that means nothing.
    pf, t_pf = [], 0.0
    for i, d in enumerate(draws if draws else null_draws):
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
        pf.append(score_bestprof(sig, prof, d["f0"], args, keep_profile=keep_prof))
    rec["timing"]["prepfold"] = t_pf
    # Targeted: no false-alarm column of its own, so the empty realisations'
    # folds are filed separately -- `mc_analyze` reads them as prepfold's null.
    rec["results"]["prepfold_null" if not draws else "prepfold"] = pf

    # --- accelsearch, on BOTH the raw and the de-reddened FFT ---------------
    # Run 1 gave accelsearch the raw `.fft` while `coherent_search` got
    # `_red.fft`, which is an asymmetry a referee will find: `rednoise` also
    # normalises the powers, and accelsearch's own local normalisation is not the
    # same operation.  Running both settles it from inside the study instead of
    # by argument.  The two arms write `<base>_ACCEL_0` and `<base>_red_ACCEL_0`,
    # which is why each is parsed from its own input's stem rather than a fixed
    # name.
    for name, src in (("accelsearch", base + ".fft"),
                      ("accelsearch_red", base + "_red.fft")):
        if name == "accelsearch_red" and not args.accel_red:
            continue
        t, _, _, rc = run([tools["accelsearch"], "-numharm", "16", "-zmax", "0",
                           "-sigma", str(args.accel_sigma), src], cwd=wd)
        rec["timing"][name] = t
        ac = parse_accel(os.path.join(wd, src[:-4] + "_ACCEL_0"))
        hits, fa = score(ac, inj, T, tol)
        rec["results"][name] = dict(hits=hits, false=fa_summary(fa, args.fa_top), ncand=len(ac),
                                    ok=(rc == 0))

    # --- rseek, config A ----------------------------------------------------
    def rseek(cfg):
        return run([tools["rseek"], "--Pmin", repr(cfg["Pmin"]), "--Pmax", repr(cfg["Pmax"]),
                    "--bmin", str(cfg["bmin"]), "--bmax", str(cfg["bmax"]),
                    "--smin", str(args.rseek_smin), "-f", "presto", base + ".inf"], cwd=wd)
    t, out, _, rc = rseek(RSEEK_A)
    rec["timing"]["rseek_A"] = t
    rs = parse_rseek(out) if rc == 0 else []
    hits, fa = score(rs, inj, T, tol)
    rec["results"]["rseek_A"] = dict(hits=hits, false=fa_summary(fa, args.fa_top), ncand=len(rs),
                                     ok=(rc == 0))

    # --- rseek, deep tiling, on a subset ------------------------------------
    if one_in(rec["index"], args.deep_every):
        allc, tt, ok = [], 0.0, True
        for cfg in RSEEK_B:
            t, out, _, rc = rseek(cfg)
            tt += t
            ok &= (rc == 0)
            allc += parse_rseek(out) if rc == 0 else []
        rec["timing"]["rseek_B"] = tt
        hits, fa = score(allc, inj, T, tol)
        rec["results"]["rseek_B"] = dict(hits=hits, false=fa_summary(fa, args.fa_top),
                                         ncand=len(allc), ok=ok)
        rec["config"]["rseek_B"] = RSEEK_B

    # --- coherent_search: the shipped defaults, plus the deeper arms --------
    # Three SEPARATE invocations, not one: each arm is a different `SearchParams`,
    # and -- more to the point -- each must carry its OWN false-alarm tail, so a
    # deeper configuration pays for its own trials instead of being handed the
    # default arm's threshold.  The ~2 s of Julia start-up per arm is ~4% of a
    # realisation and buys a threshold that is measured rather than assumed.
    for name, cfg in COH_ARMS.items():
        every = args.deep_coh_every if name == "coherent_deep" else cfg["every"]
        if every != 1 and not one_in(rec["index"], every):
            continue
        out_f = f"{stem}_{name}.cohout"
        cmd = [tools["julia"], f"--project={REPO}", f"-t{args.coh_threads}",
               os.path.join(REPO, "bin", "coherent_search.jl"),
               "--threshold", str(args.threshold), "--noprogress",
               "--lofreq", repr(cfg["lofreq"]), "--hifreq", repr(cfg["hifreq"]),
               "--nharms", str(cfg["nharms"]), "--maxdecim", str(cfg["maxdecim"]),
               "--ncands", str(args.ncands), "-o", out_f, stem + "_red.fft"]
        t, _, err, rc = run(cmd)
        rec["timing"][name] = t
        ch = parse_cohout(out_f) if rc == 0 else []
        hits, fa = score(ch, inj, T, tol)
        rec["results"][name] = dict(hits=hits, false=fa_summary(fa, args.fa_top), ncand=len(ch),
                                    ok=(rc == 0))
        if rc != 0:
            rec["results"][name]["stderr"] = err[-2000:]


# ---------------------------------------------------------------------------
# Trial counts, so cost can be normalised per unit of searching
# ---------------------------------------------------------------------------
def trial_counts(args, path):
    """Statistics each code evaluates on one realisation.  Cached in `path`.

    Computed ONCE, in the parent, and handed to every worker: it is a property of
    the configuration and the file length, not of a realisation.  Without it the
    cost column can only say "we are 6x faster than rseek_B", which is a statement
    about two implementations; with it the comparison can be made per trial, which
    is a statement about two algorithms.

    `rseek`'s count is MEASURED (riptide's `ffa_search` returns the periodogram it
    built) on a shorter series and scaled: at fixed `tsamp` and period range the
    downsampling ladder is identical and the number of shifts is linear in the
    sample count, so the scaling is exact and a 2^20 probe costs a second instead
    of a minute and a gigabyte.  Ours is analytic and exact.  accelsearch's is an
    ESTIMATE (`numharm x N/2`) and marked as one -- its sigma is already
    trials-corrected, so the number is context, not a normaliser.
    """
    if os.path.isfile(path):
        try:
            with open(path) as fh:
                return json.load(fh)
        except Exception:
            pass
    out = {}
    T = args.nsamp * args.dt
    for name, cfg in COH_ARMS.items():
        # Fundamental trials x rungs: `(hifreq - lofreq) * T * nharms / hidr`
        # fundamentals, each scored at every decimation in the ladder.
        nfund = (cfg["hifreq"] - cfg["lofreq"]) * T * cfg["nharms"] / 0.5
        nk = len([k for k in range(1, cfg["maxdecim"] + 1)
                  if cfg["nharms"] // k >= 2])
        out[name] = int(round(nfund * nk))
    out["accelsearch"] = int(16 * args.nsamp / 2)
    out["accelsearch_red"] = out["accelsearch"]
    try:
        import numpy as _np
        from riptide import TimeSeries, ffa_search
        nprobe = 1 << 20
        scale = args.nsamp / nprobe
        ts = TimeSeries.from_numpy_array(
            _np.random.default_rng(0).normal(size=nprobe).astype(_np.float32),
            tsamp=args.dt)
        for name, cfgs in (("rseek_A", [RSEEK_A]), ("rseek_B", RSEEK_B)):
            tot = 0
            for c in cfgs:
                _, pg = ffa_search(ts, period_min=c["Pmin"], period_max=c["Pmax"],
                                   bins_min=c["bmin"], bins_max=c["bmax"])
                tot += len(pg.periods) * len(_np.asarray(pg.widths))
            out[name] = int(round(tot * scale))
    except Exception as exc:                 # riptide not importable, or it moved
        out["rseek_note"] = f"not counted: {exc!r}"
    try:
        tmp = path + f".{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh, indent=1)
        os.replace(tmp, path)                # atomic, so a reader never sees half
    except OSError:
        pass
    return out


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
    ap.add_argument("--deep-every", type=int, default=5,
                    help="every Nth realisation also gets the deep rseek tiling "
                         "(121 s, the single largest cost, and already losing by "
                         "7.3 points); 0 disables")
    ap.add_argument("--deep-coh-every", type=int, default=5,
                    help="every Nth realisation also gets the FULL-BAND deep coherent "
                         "arm (nharms 120, maxdecim 12, hifreq 62.5) -- the arm that "
                         "measures whether a blanket deep search is the net loss "
                         "section 4 predicts; 0 disables")
    ap.add_argument("--accel-red", action="store_true", default=True)
    ap.add_argument("--no-accel-red", dest="accel_red", action="store_false",
                    help="do NOT also run accelsearch on _red.fft (run 1 gave it "
                         "only the raw .fft while we got the de-reddened one)")
    ap.add_argument("--keep-profiles", type=int, default=10,
                    help="store prepfold's actual profile on every Nth realisation "
                         "(and on every injection-free one), so section 5's correction "
                         "can be re-derived without re-folding; 0 disables")
    ap.add_argument("--strat", type=float, nargs=3, default=[0.005, 0.5, 0.25],
                    metavar=("D0", "ALPHA", "QMIN"),
                    help="duty-cycle stratification: keep a draw of duty d with "
                         "probability clip((D0/d)^ALPHA, QMIN, 1) and weight it 1/q. "
                         "The default raises the fraction below 0.5%% duty from 1.5%% "
                         "to 3.7%% at 0.85 effective sample size.  Pass '0 0 1' for "
                         "the unstratified population")
    ap.add_argument("--rednoise", type=float, default=0.0,
                    help="add power-law red noise with this corner frequency in Hz "
                         "(0 = off, which is what run 2 uses).  Sane values are "
                         "~0.003-0.05 at T = 1006 s; see add_rednoise for why")
    ap.add_argument("--rednoise-alpha", type=float, default=2.0)
    ap.add_argument("--nsamp", type=int, default=1 << 24)
    ap.add_argument("--dt", type=float, default=60.0e-6)
    ap.add_argument("--snrs", type=float, nargs="+", default=None,
                    help="inject at these DISCRETE S/N values (run 1's six integers); "
                         "omit for the continuous default")
    ap.add_argument("--snr-range", type=float, nargs=2, default=[5.5, 11.5],
                    metavar=("LO", "HI"))
    ap.add_argument("--min-fwhm-samples", type=float, default=3.0)
    ap.add_argument("--threshold", type=float, default=5.5,
                    help="coherent_search reporting floor.  5.5, not run 1's 6.0: "
                         "the false-alarm rate curve is FLAT below this by "
                         "construction, and the low-frequency tier arm searches "
                         "6.4x fewer trials than the default arm, so its own "
                         "matched threshold lands near 6.0 (measured 6.05 in a "
                         "short-T smoke run) -- at 6.0 it would have been floor-"
                         "limited and read as better than it is.  Costs nothing: "
                         "the gated exact rescan fires on ~1e-6 of trials either "
                         "way")
    ap.add_argument("--ncands", type=int, default=2000,
                    help="candidates coherent_search may report.  A THIRD cap that "
                         "censors the same way as --threshold and --fa-top, so it "
                         "is kept well clear: at --threshold 5.5 the full-band deep "
                         "arm produces ~200 per realisation at T = 1006 s, and "
                         "reporting more costs only the writing of them")
    ap.add_argument("--rseek-smin", type=float, default=6.0,
                    help="rseek's reporting floor.  LEFT AT 6.0 deliberately: its "
                         "matched thresholds are 7.95 and 8.05, its rate at 6.5 is "
                         "already 20-48 per realisation, and lowering it multiplies "
                         "a candidate list that is already ~170 (config A) and ~430 "
                         "(config B) long for a region of the curve no threshold "
                         "ever reaches")
    ap.add_argument("--fa-top", type=int, default=800,
                    help="false-alarm statistics stored per code per realisation")
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
    # Once, in the PARENT, before any worker starts: it is a property of the
    # configuration and the file length, and 15 workers racing to write the same
    # cache file is how a half-written JSON gets read.
    trial_counts(args, os.path.join(args.outdir, "trials.json"))

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
    tools["trials"] = trial_counts(args, os.path.join(args.outdir, "trials.json"))

    # A plain SIGTERM (a `pkill`, or the batch system) does not raise in Python,
    # so the `finally` that removes the scratch directory never runs and each
    # killed worker leaks ~400 MB of /dev/shm.  Two restarts during setup left 96
    # orphaned directories behind.  Turning the signal into SystemExit makes the
    # cleanup path run on the way out.
    def _bye(signum, frame):
        raise SystemExit(f"terminated by signal {signum}")
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _bye)
        except (ValueError, OSError):
            pass

    workdir = args.workdir or ("/dev/shm" if os.path.isdir("/dev/shm") else "/tmp")
    workdir = os.path.join(workdir, f"mcsim_{os.getpid()}")
    os.makedirs(workdir, exist_ok=True)
    args.workdir = workdir

    tpa = MP.load_tpa(args.tpa) if args.tpa else None
    strat = tuple(args.strat) if args.strat and args.strat[0] > 0 else None
    args.strat = list(strat) if strat else None
    pop = MP.Population(tpa=tpa, strat=strat)

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
                    dead = [m for m in ("rseek_A", "rseek_B", "accelsearch",
                                        "accelsearch_red", "coherent",
                                        "coherent_tier", "coherent_deep")
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
