#!/usr/bin/env python3
"""
mc_analyze.py -- combine and summarise mc_simulate.py output.

Combining runs is `cat`: every line is a self-contained realisation carrying its
index, so this globs the .jsonl files, de-duplicates by (host-independent) index,
and tabulates.

    mc_analyze.py mcout/                       # the whole report, FAP-matched at 1e-2
    mc_analyze.py mcout/ --by ducy --fap 1e-3
    mc_analyze.py mcout/ --sections roc,paired,cost
    mc_analyze.py mcout/ --weight flat         # flat in log duty, not TPA-weighted

WHAT THIS DOES NOT DO IMPLICITLY, and why (each of these was a wrong answer once):

  * **Compare codes at a common nominal threshold.**  Ours and rseek's statistics
    are single-trial, accelsearch's sigma is already trials-corrected, and
    prepfold's is a chi-squared -- so "S/N 6" means four different things.  Every
    threshold here comes from each code's own measured false-alarm rate.  Run 1
    had them at 8.00 / 7.95 / 8.05 / 7.05 for the SAME rate.
  * **Compare a code that ran on a subset against one that ran on everything.**
    Cross-code cells are restricted to the realisations every compared code saw
    (`--no-common` turns that off).  Run 1's report put `rseek_B`'s 1-in-3 subset
    beside a `coherent` column computed from 7x more data and flagged it only
    with a footnote.
  * **Treat a harmonic detection as a miss.**  Detections at f/2, 2f, 3f/2 ... are
    real recoveries; rseek and accelsearch do not collapse the family and we do,
    so scoring them as misses would penalise the codes that report them.
  * **Put error bars on injections.**  Six injections share one noise
    realisation, so they are not independent.  Every interval here is bootstrapped
    by REALISATION.
  * **Match a threshold over a mixture of noise levels.**  The false-alarm rate
    is a function of the red-noise knee, so a cut matched over a red run belongs
    to none of the levels in it -- `rseek`'s 0.01/realisation cut runs 7.95 at
    knee < 0.5 Hz and 285 above 15.  Every threshold is matched per red-noise bin
    (`--match knee`, the default), or per (bin, f0 band).  On records with no red
    noise the modes are identical, so a white run is unaffected.
  * **Count a chance coincidence as a detection.**  `score()` claims a candidate
    within `tol_bins` of ANY ratio n/m <= 8 of f0 and then removes it from the
    false-alarm list, so a flooding code's coincidences enter as detections where
    no matched threshold can see them: at knee > 15 Hz, f0 5-20 Hz, only 16% of
    `rseek`'s hits were within 0.1 bin of their target and its detection fraction
    ROSE with knee.  `--hit-tol` (0.5 bins) scores those as misses; `--sections
    hits` reports what went and what is left.
  * **Read a cell with no threshold as a detection fraction of zero.**  A method
    that produced no cut in a cell (an older run with no per-band tails, say) has
    made no measurement there, and its rows leave the denominator.
  * **Quote prepfold's `snr1` raw.**  PRESTO's `fold()` drizzles each sample
    across the bins it covers and so correlates them; `snr1` assumes independence
    and reads up to 20% high in the MSP band, where prepfold is the ceiling
    column.  `mc_model.drizzle_boxcar_corr` is applied wherever `nbins`,
    `dt_per_bin` and the winning width are recorded.
"""

from __future__ import annotations

import argparse
import functools
import glob
import json
import math
import os
import sys
from collections import defaultdict
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_model as MM

# Every method that can appear, in report order.  `coh+tier` is DERIVED here (the
# union of the default search and the low-frequency deep tier), not recorded.
# `coherent_meas` and `coherent_rawmeas` answer run 3's 'measured or analytic?'
# against the always-on `coherent` (analytic, _red.fft).  There is deliberately no
# analytic-on-raw arm: that is a usage error, not a configuration.
METHODS = ("prepfold_chi2", "prepfold_snr1", "accelsearch", "accelsearch_red",
           "rseek_A", "rseek_B", "coherent", "coherent_tier", "coherent_deep",
           "coherent_meas", "coherent_rawmeas", "coh+tier")
SEARCHES = ("accelsearch", "accelsearch_red", "rseek_A", "rseek_B",
            "coherent", "coherent_tier", "coherent_deep",
            "coherent_meas", "coherent_rawmeas", "coh+tier")
RECORDED = ("accelsearch", "accelsearch_red", "rseek_A", "rseek_B",
            "coherent", "coherent_tier", "coherent_deep",
            "coherent_meas", "coherent_rawmeas")
# The union arm: a candidate list is the two arms' lists concatenated, which is
# what a tiered search would actually report.
UNION = {"coh+tier": ("coherent", "coherent_tier")}
# Statistics that are the same quantity (riptide's snr1), so their VALUES may be
# compared and not only their detection fractions.
SNR1_LIKE = ("prepfold_snr1", "rseek_A", "rseek_B", "coherent", "coherent_tier",
             "coherent_deep", "coherent_meas", "coherent_rawmeas",
             "coh+tier")

BINS = {
    "snr":  [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5],
    "ducy": [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.50],
    "f0":   [0.1, 1.0, 5.0, 20.0, 100.0, 200.0, 400.0, 1000.0],
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load(paths, with_profiles=False):
    """Every realisation, de-duplicated by index WITHIN each run.

    **A run is a directory**, and the de-duplication key is `(directory, index)`.
    Run 2 and run 3 share their indices on purpose -- same injections in the same
    white noise, red added -- so a key of `index` alone kept whichever run sorted
    first and silently dropped the other: `mc_analyze run2 run3 --sections knee`,
    the command the README gives for the paired white row, returned all of run 2
    as `white` and NO red bins at all.  Each record carries its run as `run`.
    The key is the directory rather than the path argument because duplicates
    also occur INSIDE a run -- run 3 was restarted from 24 workers to 15, which
    re-partitioned the index space across worker files -- and a shell-expanded
    file list must still collapse those.

    The false-alarm tails are held as numpy arrays.  They are ~1500 floats per
    record (pooled plus per-band) and as Python floats they were most of the
    memory: ~5 GB for 44k run-3 realisations, which would not scale to the
    full run, let alone to run 2 and run 3 together.  float64, so every value is
    exactly the parsed one and a report is unchanged to the byte.

    `with_profiles` keeps prepfold's stored profiles.  It is OFF by default and
    that is not an oversight: a stored profile is 128 floats, so at one realisation
    in one they are 40% of the file and -- much worse -- several GB of Python
    float objects once parsed.  Only `sec_drizzle` needs them, and it is the one
    section that asks.
    """
    recs, seen = [], {}
    files, patches = [], []
    for p in paths:
        files += glob.glob(os.path.join(p, "*.jsonl")) if os.path.isdir(p) else glob.glob(p)
    for f in sorted(files):
        run = os.path.dirname(os.path.abspath(f))
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if "error" in r:
                    continue
                r["run"] = run
                # Per record, as it is parsed: converting after the whole run is
                # read leaves every Python list alive at once, which is the peak.
                for d in (r.get("results") or {}).values():
                    _tails_to_arrays(d)
                # A PATCH row carries one arm re-run on the same realisation --
                # `mc_simulate --arms accel` regenerates the noise from the index
                # and re-scores only accelsearch.  Hold them and merge after, so
                # a patch file sorting before its parent still lands.
                if r.get("patch"):
                    patches.append(r)
                    continue
                if (run, r.get("index")) in seen:
                    continue
                seen[(run, r["index"])] = r
                if not with_profiles:
                    for k in ("prepfold", "prepfold_null"):
                        for d in (r.get("results", {}).get(k) or []):
                            if d:
                                d.pop("prof", None)
                recs.append(r)
    for pr in patches:
        base = seen.get((pr["run"], pr["index"]))
        if base is None:                       # patched a realisation we do not have
            continue
        base.setdefault("results", {}).update(pr.get("results", {}))
        base.setdefault("timing", {}).update(pr.get("timing", {}))
    recs.sort(key=lambda r: r["index"])
    return recs


def _tails_to_arrays(d):
    """Replace one arm's stored false-alarm tails (pooled and per band) with arrays."""
    f = d.get("false") if isinstance(d, dict) else None
    if not isinstance(f, dict):
        return
    f["top"] = np.asarray(f.get("top") or [], dtype=float)
    for b in (f.get("bands") or {}).values():
        b["top"] = np.asarray(b.get("top") or [], dtype=float)


def _merge_union(res, name):
    """Synthesise a union arm's `results` entry from its components.

    Hits merge by taking the stronger statistic; false-alarm tails concatenate,
    which is exactly what a search running both configurations would report.  The
    merge is only honest ABOVE the higher of the two tails' truncation floors --
    `saturation()` prints those, and the FAP grid used here never goes low enough
    to reach them.
    """
    parts = [res.get(m) for m in UNION[name]]
    if any(p is None for p in parts):
        return None
    n = max(len(p["hits"]) for p in parts)
    hits = []
    for i in range(n):
        best = None
        for p in parts:
            h = p["hits"][i] if i < len(p["hits"]) else None
            if h and (best is None or h["stat"] > best["stat"]):
                best = h
        hits.append(best)
    top = _desc(np.concatenate([np.asarray(p["false"]["top"], dtype=float)
                                for p in parts]))[:1600]
    floors = [p["false"]["floor"] for p in parts if p["false"].get("floor") is not None]
    false = dict(n=sum(p["false"]["n"] for p in parts), top=top,
                 truncated=any(p["false"].get("truncated") for p in parts),
                 floor=(min(floors) if floors else None))
    if all(p["false"].get("bands") is not None for p in parts):
        bands = {}
        for lab in {k for p in parts for k in p["false"]["bands"]}:
            sub = [p["false"]["bands"][lab] for p in parts if lab in p["false"]["bands"]]
            fl = [b["floor"] for b in sub if b.get("floor") is not None]
            bands[lab] = dict(n=sum(b["n"] for b in sub),
                              top=_desc(np.concatenate(
                                  [np.asarray(b["top"], dtype=float) for b in sub]))[:1600],
                              truncated=any(b.get("truncated") for b in sub),
                              floor=(min(fl) if fl else None))
        false["bands"] = bands
    return dict(hits=hits, ncand=sum(p["ncand"] for p in parts),
                ok=all(p.get("ok", True) for p in parts), false=false)


def _desc(a):
    return np.sort(a)[::-1]


def rows(recs, dt=None, hit_tol=None):
    """One row per injection: every method's statistic, plus what it needs to be
    interpreted.  A method that did not RUN on a realisation leaves its key
    ABSENT (not nan), so a denominator can count what a method was actually
    given.

    A hit further than `hit_tol` Fourier bins from its target is scored as a miss
    (see `HIT_TOL`); `float("inf")` reproduces the scoring as recorded."""
    if hit_tol is None:
        hit_tol = HIT_TOL
    out = []
    for r in recs:
        if r.get("empty"):
            continue
        res = r["results"]
        for name in UNION:
            u = _merge_union(res, name)
            if u:
                res[name] = u
        _dt = r.get("dt", dt) or 60e-6
        T = r.get("T") or (r["N"] * _dt if r.get("N") else None)
        red = r.get("rednoise") or {}
        tol = (r.get("config") or {}).get("tol_bins")
        for i, inj in enumerate(r["injections"]):
            row = dict(index=r["index"], inj=i, run=r.get("run"), tol_bins=tol,
                       fknee=red.get("fknee"), sigma_got=red.get("sigma_got"),
                       snr=inj["snr"], ducy=inj["ducy"],
                       f0=inj["f0"], msp=inj["msp"], w10_w50=inj["w10_w50"],
                       weight=float(inj.get("weight", 1.0)), dt=_dt,
                       ducy_got=(inj.get("profile") or {}).get("ducy_got", inj["ducy"]),
                       model_eff=inj.get("model_eff"))
            pf = (res.get("prepfold") or [None] * (i + 1))[i]
            if pf:
                row["prepfold_chi2"] = pf["chi2_sigma"]
                nb = pf.get("nbins") or 0
                # dt_per_bin: recorded by run 2, reconstructible for run 1.
                dpb = pf.get("dt_per_bin")
                if dpb is None and nb:
                    dpb = (1.0 / inj["f0"]) / nb / _dt
                w = pf.get("w")
                if w is None and nb and pf.get("ducy"):
                    w = max(1, int(round(pf["ducy"] * nb)))
                row["prepfold_nbins"], row["prepfold_dpb"], row["prepfold_w"] = nb, dpb, w
                s1 = pf.get("snr1")
                row["prepfold_snr1_raw"] = s1
                row["prepfold_snr1"] = (
                    s1 * _drizzle_corr(nb, dpb, w)
                    if (s1 is not None and nb and dpb and w) else s1)
                row["prepfold_ducy"] = pf.get("ducy")
            valid = {}
            for m in RECORDED:
                d = res.get(m)
                if d is None:
                    continue
                h = d["hits"][i] if i < len(d["hits"]) else None
                off = _hit_offset(h, inj["f0"], T)
                if off is not None:
                    row[m + "_off"] = off
                    if off > hit_tol:
                        # A coincidence, not a detection: kept for the diagnostics
                        # (`sec_hits`), scored as a miss everywhere else.
                        row[m + "_junk"] = h["stat"]
                        h = None
                valid[m] = h
                row[m] = h["stat"] if h else float("nan")
                row[m + "_harm"] = h["harmonic"] if h else None
                row[m + "_ducy"] = h.get("ducy") if h else None
                if h and h.get("width") and h.get("ducy"):
                    row[m + "_b"] = int(round(h["width"] / h["ducy"]))
                if h and h.get("nharm"):
                    row[m + "_nharm"] = h["nharm"]
            for name, parts in UNION.items():
                if name in res:
                    # From the parts' VALID hits: a junk hit in one part must not
                    # outrank a real one in the other just by being stronger.
                    h = None
                    for p in parts:
                        x = valid.get(p)
                        if x and (h is None or x["stat"] > h["stat"]):
                            h = x
                    row[name] = h["stat"] if h else float("nan")
                    row[name + "_harm"] = h["harmonic"] if h else None
                    row[name + "_ducy"] = h.get("ducy") if h else None
            out.append(row)
    return out


# How far a hit may sit from its target, in the CANDIDATE's own Fourier bins,
# and still count as a detection.  `score()` in the driver accepts anything
# within `tol_bins` (3.0) of ANY ratio n/m <= 8 of f0, and a candidate it claims
# leaves the false-alarm list -- so where a code emits a flood of candidates,
# coincidences become "detections" that no matched threshold can see.  Under red
# noise rseek does exactly that at slow periods: at knee > 15 Hz and f0 5-20 Hz
# only 16% of its hits lie within 0.1 bin of their target, the rest are spread
# across the window at labels 1/8, 1/7, 1/6 with a median statistic of 25, and
# its detection fraction there ROSE with knee (80.5%, against 65.7% at knee < 0.5).
# Real hits are tight: on run 2's white noise, above each code's matched cut, the
# 99th percentile offset is 0.15-0.23 bins and <= 0.07% lie beyond 0.5, for every
# code.  So 0.5 keeps essentially every real detection and passes ~1/6 of a
# uniformly spread coincidence; `sec_hits` estimates what is left of those.
#
# The one thing this cannot undo: the driver stores ONE hit per injection,
# fundamental first and then strongest, so a junk candidate at the fundamental
# ratio can have displaced a real one.  Rejecting it scores a miss the code may
# not have made.  That errs AGAINST the flooding code, and only where its matched
# cut is already far above a real pulsar's statistic -- see `sec_hits`.
HIT_TOL = 0.5

# Below this many detections in a cell, `sec_hits` quotes counts rather than a
# percentage of them.
_HITS_MIN_DET = 20


def _hit_offset(h, f0, T):
    """`|freq - ratio * f0|` in Fourier bins, or None where it cannot be computed."""
    if not h or h.get("freq") is None or not T:
        return None
    try:
        ratio = float(Fraction(h["harmonic"]))
    except (ValueError, ZeroDivisionError, TypeError):
        return None
    return abs(h["freq"] - ratio * f0) * T


def present(rws, pool=METHODS):
    return [m for m in pool if any(m in r for r in rws)]


# ---------------------------------------------------------------------------
# Weighting and the common subset
# ---------------------------------------------------------------------------
def apply_weights(rws, scheme):
    """`tpa` keeps the importance weights the sampler recorded (so a stratified
    run still estimates the real population); `flat` re-weights to flat in log
    duty, which is the honest way to report a per-duty comparison that is not an
    artefact of where the population happens to sit."""
    if scheme == "tpa":
        return
    if scheme == "flat":
        lo, hi, nb = -3.0, -0.3, 18
        idx = np.clip(((np.log10([max(r["ducy"], 1e-3) for r in rws]) - lo)
                       / (hi - lo) * nb).astype(int), 0, nb - 1)
        cnt = np.bincount(idx, minlength=nb).astype(float)
        for r, j in zip(rws, idx):
            r["weight"] = 1.0 / cnt[j] if cnt[j] else 0.0
        s = sum(r["weight"] for r in rws)
        for r in rws:
            r["weight"] *= len(rws) / s
        return
    raise SystemExit(f"unknown --weight {scheme}")


def common(rws, methods):
    """Restrict to injections every one of `methods` saw.  Without this a method
    running on a `--deep-every` subset is compared against columns built from
    several times more data, and the difference reads as sensitivity."""
    ms = [m for m in methods if m not in ("prepfold_chi2", "prepfold_snr1")]
    return [r for r in rws if all(m in r for m in ms)]


def split_coverage(rws, ms, frac=0.95):
    """Split methods into those that ran on (nearly) everything and those that
    ran on a subset.

    The subset arms (`rseek_B` 1-in-10, the three sigma arms 1-in-3) so intersecting
    EVERY table down to what all eight methods saw would throw away 80% of the
    run to make one comparison fair.  The report instead prints two blocks: the
    always-run methods over the whole set, and every method over the subset they
    share.  Both are honest; neither is the run-1 mistake of putting a 1-in-N
    column beside a full one in the same row.
    """
    n = max(len(rws), 1)
    full = [m for m in ms if sum(1 for r in rws if m in r) >= frac * n]
    return full, [m for m in ms if m not in full]


def ess(rws):
    w = np.array([r["weight"] for r in rws], dtype=float)
    return float(w.sum() ** 2 / max(np.sum(w * w), 1e-30))


# ---------------------------------------------------------------------------
# False alarms and thresholds
# ---------------------------------------------------------------------------
def fa_rates(recs, method, thresholds, empty_only=False, band=None):
    """False alarms per realisation at each threshold, from the stored top-N tails.

    Counted on EVERY realisation, not only the injection-free ones: a false alarm
    is a candidate matching no injection at any simple harmonic ratio, which is
    well defined either way, and run 1 verified the two agree to <= 0.10 in
    threshold at FAP 1e-2 -- so the larger sample is free.

    `band` restricts to one `false["bands"]` entry, i.e. to the false alarms a
    code produced in one frequency band.  A realisation the method RAN on still
    counts in the denominator when it made no false alarm in that band -- that is
    a rate of zero, not a missing measurement.  Records written before run 3's
    per-band tails have no `bands` key at all, and those realisations are skipped
    rather than counted as zero, so an old run reads as "no data" instead of "no
    false alarms".
    """
    n = 0
    counts = np.zeros(len(thresholds))
    for r in recs:
        if empty_only and not r.get("empty"):
            continue
        res = r["results"]
        d = res.get(method) or (_merge_union(res, method) if method in UNION else None)
        if not d or not d.get("ok", True):
            continue
        if band is None:
            tail = d["false"]["top"]
        else:
            bands = d["false"].get("bands")
            if bands is None:
                continue                        # pre-run-3 record: no band split
            tail = (bands.get(band) or {}).get("top", [])
        n += 1
        if not len(tail):
            continue
        top = np.asarray(tail, dtype=float)
        counts += (top[:, None] >= np.asarray(thresholds)[None, :]).sum(axis=0)
    return (counts / n if n else counts), n


def saturation(recs, method):
    """Where a false-alarm rate curve stops being a rate and becomes a ceiling.

    There are TWO censoring mechanisms and they are easy to confuse:

      * **the stored top-N cap**, which is OURS.  `rseek_B`'s 200-entry tail was
        saturated in 100% of run 1's realisations, so its rate curve below 6.20
        was a cap.  Run 2 stores 800 and this should read 0%.
      * **each code's own reporting floor** -- `--rseek-smin`, our `--threshold`,
        accelsearch's sifting cut.  Below it the curve is FLAT by construction,
        because no candidate below it was ever emitted.  Run 1's report showed
        `rseek_A` at 161.96 for cuts 5.0, 5.5 AND 6.0 and said nothing; that is
        one number printed three times, not three measurements.

    Neither ever touched a threshold run 1 actually used -- the loosest was 6.30
    against floors of 4.8 and 6.0 -- but a rate curve with a flat censored region
    in it should say which part is data.
    """
    nt, n, cap_floors, rep_floors = 0, 0, [], []
    for r in recs:
        res = r["results"]
        d = res.get(method) or (_merge_union(res, method) if method in UNION else None)
        if not d or not len(d["false"]["top"]):
            continue
        n += 1
        if d["false"].get("truncated"):
            nt += 1
            cap_floors.append(d["false"]["top"][-1])
        else:
            rep_floors.append(d["false"].get("floor", d["false"]["top"][-1]))
    if not n:
        return None
    cap = float(np.median(cap_floors)) if cap_floors else float("-inf")
    rep = float(np.median(rep_floors)) if rep_floors else float("-inf")
    return dict(n=n, frac=nt / n, cap_floor=cap, report_floor=rep,
                floor=max(cap, rep))


# The grid every matched threshold is read off.  It runs well past anything a
# calibrated statistic reaches, on purpose: under red noise `rseek`'s statistic
# is not calibrated at all -- its 0.01/realisation cut is 23 at a 0.5-2 Hz knee
# and 280 above 15 Hz -- and a grid that stopped at 30 returned `inf`, which the
# tables then printed beside a detection fraction of 0.0% as though the code had
# been scored.  It had not been: no threshold existed on the grid.  Past 30 the
# spacing coarsens, since nothing at that end is a close call.
FA_GRID = np.concatenate([np.arange(3.0, 30.0, 0.05),
                          np.arange(30.0, 100.0, 0.5),
                          np.arange(100.0, 1000.0, 5.0)])


def fa_curves(recs, methods):
    """The false-alarm rate curve of every method, computed ONCE.

    Every threshold in the report -- the matched cut, and the seven points of the
    ROC -- is read off these, so scanning the records once per method rather than
    once per method per FAP is the difference between a report that takes seconds
    and one that takes minutes.
    """
    return {m: fa_rates(recs, m, FA_GRID)[0] for m in methods}


def pick_thresholds(curves, fap):
    """Per method, the lowest threshold whose false-alarm rate is <= `fap` per
    realisation.  This is the only thing that makes the columns comparable."""
    out = {}
    for m, rate in curves.items():
        ok = np.nonzero(rate <= fap)[0]
        out[m] = float(FA_GRID[ok[0]]) if len(ok) else float("inf")
    return out


# ---------------------------------------------------------------------------
# Detection fractions, bootstrapped by realisation
# ---------------------------------------------------------------------------
def _cols(rws, m):
    """`(stat, weight, ran, realisation-id)` as arrays.

    Pulling the columns out and letting numpy do the arithmetic is what keeps the
    whole report at seconds rather than minutes on ~10^5 injections; the loops
    this replaced were the report's entire cost.
    """
    v = np.array([r.get(m, np.nan) for r in rws], dtype=float)
    w = np.array([r["weight"] for r in rws], dtype=float)
    ok = np.array([m in r for r in rws], dtype=bool)
    idx = np.array([r["index"] for r in rws], dtype=np.int64)
    return v, w, ok, idx


def _det(rws, m, t):
    if not rws:
        return float("nan"), 0.0
    v, w, ok, _ = _cols(rws, m)
    if not ok.any():
        return float("nan"), 0.0
    # A cell with NO cut (nan -- e.g. a run that stored no per-band tails) is a
    # missing measurement, not a miss.  Counting those rows in the denominator
    # read as a detection fraction of zero, which is the artifact this whole
    # per-cell machinery exists to remove.
    tv = _tv(rws, t)
    ok = ok & ~np.isnan(tv)
    d = np.where(np.isnan(v), False, v >= tv) & ok
    den = float(np.sum(w * ok))
    return (float(np.sum(w * d)) / den if den else float("nan")), den


def boot_det(rws, m, t, nboot, rng):
    """Detection fraction with a bootstrap-by-REALISATION standard error.

    Six injections share one noise realisation, so resampling INJECTIONS would
    understate the error.  Resampling realisations means summing the numerator
    and denominator per realisation first, which also makes the bootstrap a pair
    of matrix products rather than 200 passes over the rows.
    """
    p, wsum = _det(rws, m, t)
    if nboot <= 0 or not np.isfinite(p) or not rws:
        return p, float("nan"), wsum
    v, w, ok, idx = _cols(rws, m)
    tv = _tv(rws, t)
    ok = ok & ~np.isnan(tv)            # no cut in this cell: no measurement
    d = (np.where(np.isnan(v), False, v >= tv) & ok).astype(float)
    uniq, inv = np.unique(idx, return_inverse=True)
    num = np.bincount(inv, weights=w * d, minlength=len(uniq))
    den = np.bincount(inv, weights=w * ok, minlength=len(uniq))
    pick = rng.integers(0, len(uniq), size=(nboot, len(uniq)))
    n = num[pick].sum(axis=1)
    dd = den[pick].sum(axis=1)
    vals = np.divide(n, dd, out=np.full(nboot, np.nan), where=dd > 0)
    return p, float(np.nanstd(vals)), wsum


def mcnemar(rws, m1, m2, t1, t2):
    """Discordant pairs.  `1054 vs 181` says far more than `71.6% vs 64.3%`, and
    paired noise is exactly what the design bought."""
    n01 = n10 = both = neither = 0
    for r, u1, u2 in zip(rws, _tv(rws, t1), _tv(rws, t2)):
        if m1 not in r or m2 not in r:
            continue
        a = np.isfinite(r[m1]) and r[m1] >= u1
        b = np.isfinite(r[m2]) and r[m2] >= u2
        if a and b:
            both += 1
        elif a:
            n10 += 1
        elif b:
            n01 += 1
        else:
            neither += 1
    n = n10 + n01
    # Normal approximation to the exact binomial; n is in the hundreds here.
    z = (abs(n10 - n01) - 1) / math.sqrt(n) if n > 0 else 0.0
    p = math.erfc(z / math.sqrt(2.0)) if n > 0 else 1.0
    return dict(both=both, only1=n10, only2=n01, neither=neither, z=z, p=p)


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------
def _fmt(v, n=1):
    return "--" if not np.isfinite(v) else f"{v:.{n}f}"


def sec_header(recs, rws, args, book=None):
    nempty = sum(1 for r in recs if r.get("empty"))
    print(f"{len(recs)} realisations ({nempty} injection-free), {len(rws)} injections")
    print(f"weighting: {args.weight}   effective sample size {ess(rws):.0f} injections")

    # Run 3 carries a per-realisation red-noise knee, and the false-alarm rate is
    # a function of it -- so a threshold matched over the whole run is matched to
    # a MIXTURE of noise levels and belongs to none of them.  `--match pooled`
    # does exactly that.  Refuse to be quiet about it.
    red = [r["rednoise"] for r in recs if r.get("rednoise")]
    nwhite = len(recs) - len(red)
    if red and book is not None and book.match != "pooled":
        ks = sorted(x["fknee"] for x in red)
        print(f"\n  {len(red)} of {len(recs)} realisations carry RED NOISE (knee "
              f"{ks[0]:.2f}-{ks[-1]:.2f} Hz, median {ks[len(ks)//2]:.2f})"
              + (f", {nwhite} are white" if nwhite else "") + ".")
        print(f"  Every threshold is matched per {book.match.replace(',', ' x ')} cell, so "
              "each table is a mixture\n  of cells each cut at its own rate -- read "
              "the `knee` section for the split.")
        n = sum(1 for r in rws for m in RECORDED if r.get(m + "_junk") is not None)
        print(f"  {n} recorded hits sit more than {args.hit_tol:g} bins from their "
              "target and are scored as misses\n  (chance coincidences; `hits` "
              "section).\n")
    elif red:
        ks = sorted(x["fknee"] for x in red)
        print(f"\n  *** {len(red)} of {len(recs)} realisations carry RED NOISE "
              f"(knee {ks[0]:.2f}-{ks[-1]:.2f} Hz, median {ks[len(ks)//2]:.2f}).")
        print("  *** Every threshold below is matched over the POOLED run, i.e. over a")
        print("  *** mixture of noise levels, and is therefore correct for none of them.")
        print("  *** Split by knee before quoting anything: the red-noise study needs")
        print("  *** per-bin thresholds, and the white run is its own zero point.\n")
    cov = {m: sum(1 for r in rws if m in r) for m in present(rws, SEARCHES)}
    print("coverage (injections each method ran on): " +
          "  ".join(f"{m} {v}" for m, v in cov.items()))


def sec_cost(recs, rws, thr, args):
    """Sensitivity and cost belong on one axis or either one can be gamed."""
    tk = defaultdict(list)
    for r in recs:
        for k, v in r.get("timing", {}).items():
            tk[k].append(v)
    print("\n--- cost ---")
    print("median wall per realisation (s):  " +
          "  ".join(f"{k} {np.median(v):.1f}" for k, v in
                    sorted(tk.items(), key=lambda kv: -np.median(kv[1]))))
    tr = {}
    for r in recs:
        for k, v in (r.get("config", {}).get("trials") or {}).items():
            tr.setdefault(k, v)
    sub = common(rws, present(rws, SEARCHES)) if args.common else rws
    if not sub:
        return
    print(f"\n{'method':>15} {'det%':>7} {'s/real':>8} {'det/CPU-s':>10} "
          f"{'trials':>12} {'det% per 1e6 trials':>20}")
    for m in present(sub, SEARCHES):
        t = thr.get(m, thr["_default"])
        p, _ = _det(sub, m, t)
        # A union arm costs both its parts; a subset method costs what it costs
        # per realisation it ran on, not amortised over the ones it skipped.
        keys = UNION.get(m, (m,))
        wall = sum(np.median(tk[k]) for k in keys if k in tk)
        n = tr.get(m) or (sum(tr.get(k, 0) for k in keys) or None)
        print(f"{m:>15} {100 * p:7.1f} {wall:8.1f} "
              f"{(p / wall if wall else float('nan')):10.4f} "
              f"{(f'{n:,}' if n else '--'):>12} "
              f"{(100 * p / (n / 1e6) if n else float('nan')):20.3f}")
    print("  (det/CPU-s is detection fraction per second of that method's own wall clock;\n"
          "   a code that is 6x slower must be 6x better to be on the same line)")


def sec_falarm(recs, args, thr=None):
    grid = np.array([5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0])
    print("\n--- false alarms per realisation (candidates matching no injection) ---")
    print(f"{'':>15} " + " ".join(f"{t:>7.1f}" for t in grid) +
          f"   {'floor':>6}   n")
    warn = []
    for m in present_recs(recs, SEARCHES):
        rate, n = fa_rates(recs, m, grid)
        if not n:
            continue
        s = saturation(recs, m) or {}
        floor = s.get("floor", float("-inf"))
        # A cut at or below the floor is one number repeated, not a measurement.
        cells = [f"{v:>6.2f}{'c' if t <= floor else ' '}" for v, t in zip(rate, grid)]
        print(f"{m:>15} " + " ".join(cells) + f"   {floor:>6.2f}   {n}")
        if s.get("frac", 0) > 0.01:
            warn.append(f"    {m}: {100 * s['frac']:.0f}% of stored tails hit the "
                        f"top-N cap, so the curve is a CEILING below "
                        f"{s['cap_floor']:.2f} -- raise --fa-top")
        tm = _tmin(thr.get(m, np.nan)) if thr else np.nan
        if thr and np.isfinite(tm) and tm <= floor + 0.15:
            warn.append(f"    {m}: its matched threshold {tm:.2f} is within 0.15 of "
                        f"its reporting floor {floor:.2f} -- FLOOR-LIMITED, lower the "
                        f"code's own reporting cut and re-run before quoting it")
    print("  ('c' = at or below that code's reporting floor, where the curve is flat by\n"
          "   construction: no candidate below it was ever emitted.  It is one number\n"
          "   repeated, not a measurement.)")
    for w in warn:
        print(w)


def present_recs(recs, pool):
    out = []
    for m in pool:
        for r in recs:
            if m in r["results"] or (m in UNION and all(p in r["results"] for p in UNION[m])):
                out.append(m)
                break
    return out


def sec_roc(recs, rws, book, args, rng):
    """Detection fraction vs false-alarm rate.  That the ordering is invariant
    from FAP 1 to 1e-3 is itself the result: it does not hinge on one threshold.

    Under `--match knee` every point is matched per red-noise bin, so a cell's
    threshold is a set of cuts, not a number: it prints `@knee`, and the `knee`
    section lists them."""
    faps = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    full, part = split_coverage(rws, present(rws, SEARCHES))
    for ms in ([full] if not part else [full, full + part]):
        sub = common(rws, ms) if args.common else rws
        print(f"\n--- ROC: detection fraction vs false alarms/realisation "
              f"({len(sub)} injections{', the subset every arm ran' if len(ms) > len(full) else ''}) ---")
        print(f"{'FAP':>8} " + " ".join(f"{m:>16}" for m in ms))
        for f in faps:
            t = book.at(f)
            cells = []
            for m in ms:
                p, _ = _det(sub, m, t.get(m, np.inf))
                tt = t.get(m, float("nan"))
                at = (f"@{tt:4.2f}" if not isinstance(tt, Cut)
                      else ("@knee" if tt.kind == "knee" else "@k,band"))
                cells.append(f"{100 * p:10.1f}% {at}")
            print(f"{f:>8g} " + " ".join(f"{c:>16}" for c in cells))


def sec_table(rws, thr, args, rng, key=None):
    full, part = split_coverage(rws, present(rws, METHODS))
    edges = BINS[key] if key else None
    ttl = f"detection fraction by {key}" if key else "detection fraction, overall"
    for ms in ([full] if not part else [full, full + part]):
        sub = common(rws, ms) if args.common else rws
        deep = len(ms) > len(full)
        print(f"\n--- {ttl} ({len(sub)} injections"
              f"{', the subset every arm ran' if deep else ''}) ---")
        print(f"{'':>14} " + " ".join(f"{m:>16}" for m in ms) + f"   {'N':>7}")
        groups = [("all", sub)] if not key else [
            (f"{lo:g}-{hi:g}", [r for r in sub if lo <= r[key] < hi])
            for lo, hi in zip(edges[:-1], edges[1:])]
        for name, g in groups:
            if not g:
                continue
            cells = []
            for m in ms:
                p, se, _ = boot_det(g, m, thr.get(m, thr["_default"]), args.boot, rng)
                cells.append(f"{100 * p:8.1f}+-{100 * se:4.1f}" if np.isfinite(se)
                             else f"{100 * p:13.1f}  ")
            print(f"{name:>14} " + " ".join(f"{c:>16}" for c in cells) + f"   {len(g):>7}")
        print("  (+- is a bootstrap over REALISATIONS, not injections: six "
              "injections share one noise realisation)")


def sec_pairs(rws, thr, args):
    """McNemar on the paired noise -- the design's whole point."""
    ms = present(rws, SEARCHES)
    if args.ref not in ms:
        return
    print(f"\n--- discordant pairs against {args.ref} (same injection, same noise) ---")
    print(f"{'vs':>16} {'both':>8} {args.ref[:9]+'-only':>16} {'other-only':>12} "
          f"{'z':>7} {'p':>9}   N")
    for m in ms:
        if m == args.ref:
            continue
        g = common(rws, [args.ref, m])
        r = mcnemar(g, args.ref, m, thr.get(args.ref, 6.0), thr.get(m, 6.0))
        print(f"{m:>16} {r['both']:>8} {r['only1']:>16} {r['only2']:>12} "
              f"{r['z']:>7.1f} {r['p']:>9.2e}   {len(g)}")


def sec_decompose(rws, thr, args):
    """Split the advantage into (a) threshold and (b) paired recovery.

    They have different causes and different lessons, and quoting only the sum
    hides that we can WIN a cell while recovering a LOWER statistic in it.
    """
    # Only against codes whose threshold was MEASURED from their own false-alarm
    # rate.  A "threshold gap" against prepfold, which folds at the known period
    # and has no false-alarm column unless the run measured one, is not a
    # quantity -- run 1's report would have printed `nan` for it.
    ms = [m for m in present(rws, SNR1_LIKE)
          if m != args.ref and _tfinite(thr.get(m, np.nan))]
    if args.ref not in present(rws, SNR1_LIKE) or not _tfinite(thr.get(args.ref, np.nan)):
        return
    print(f"\n--- advantage decomposition: {args.ref} vs each snr1-comparable code ---")
    for m in ms:
        g = common(rws, [args.ref, m])
        if not g:
            continue
        tm, tr = thr.get(m, np.nan), thr.get(args.ref, np.nan)
        # Per-cell cuts: the gap is per ROW (each injection against the cuts of
        # its own cell), and quoted as its median.
        percell = isinstance(tm, Cut) or isinstance(tr, Cut)
        if percell:
            gap = _tv(g, tm) - _tv(g, tr)
            gid = {id(r): x for r, x in zip(g, gap)}
            fg = gap[np.isfinite(gap)]
            dthr = float(np.median(fg)) if len(fg) else float("nan")
            print(f"\n  {args.ref} vs {m}:  threshold gap {dthr:+.2f}, median over "
                  f"injections (theirs {tm:.2f}, ours {tr:.2f}, per cell)")
        else:
            dthr = tm - tr
            print(f"\n  {args.ref} vs {m}:  threshold gap {dthr:+.2f} "
                  f"(theirs {thr.get(m, float('nan')):.2f}, ours {thr.get(args.ref, float('nan')):.2f})")
        print(f"  {'duty':>10} {'n':>7} {'median dstat':>13} {'effective margin':>18}")
        for lo, hi in zip(BINS["ducy"][:-1], BINS["ducy"][1:]):
            cell = [r for r in g if lo <= r["ducy"] < hi
                    and np.isfinite(r[args.ref]) and np.isfinite(r[m])]
            if len(cell) < 20:
                continue
            d = np.median([r[args.ref] - r[m] for r in cell])
            em = (np.median([r[args.ref] - r[m] + gid[id(r)] for r in cell])
                  if percell else d + dthr)
            print(f"  {f'{lo:g}-{hi:g}':>10} {len(cell):>7} {d:>13.2f} {em:>18.2f}")
        # Counterfactual: how much of the win survives if we are held to their cut?
        p_ours, _ = _det(g, args.ref, thr.get(args.ref, 6.0))
        p_theirs_cut, _ = _det(g, args.ref,
                               _tv(g, tm) if percell else thr.get(m, 6.0))
        p_them, _ = _det(g, m, thr.get(m, 6.0))
        print(f"  counterfactual: {args.ref} at its own cut {100 * p_ours:.1f}%, "
              f"at {m}'s cut {100 * p_theirs_cut:.1f}%, against {m}'s {100 * p_them:.1f}%"
              f"  -> threshold is {100 * (p_ours - p_theirs_cut):.1f} of the "
              f"{100 * (p_ours - p_them):.1f} point gap")
    print("\n  (paired dstat is DETECTED-BY-BOTH only, so it is not a detection "
          "fraction;\n   a code can recover a lower statistic and still win on "
          "threshold and variance)")


def sec_scatter(rws, args):
    """Each statistic's own null scatter at fixed injected S/N.

    This is where `--sigma analytic` shows up, and it appeared nowhere in run 1's
    output.  Restricted to a near-100%-detection cell so max-selection bias
    cannot masquerade as low variance.
    """
    # Each method over the injections IT saw: this is a statement about one
    # statistic's own variance, not a paired comparison, so intersecting down to
    # the 1-in-5 subset arms would only cost precision.
    ms = present(rws, SNR1_LIKE)
    sub = [r for r in rws if r["snr"] >= 10.5 and 0.04 <= r["ducy"] < 0.16]
    print(f"\n--- recovered-statistic scatter at injected S/N >= 10.5, duty 4-16% "
          f"(n = {len(sub)}, each method over what it ran) ---")
    print(f"{'':>16} {'median':>8} {'sd':>8} {'IQR':>8} {'det%':>7}  "
          "(sd ~ 1.00 means a unit-variance statistic)")
    for m in ms:
        v = np.array([r[m] for r in sub if m in r], dtype=float)
        f = np.isfinite(v)
        if f.sum() < 30:
            continue
        q1, q3 = np.percentile(v[f], [25, 75])
        print(f"{m:>16} {np.median(v[f]):8.2f} {np.std(v[f]):8.3f} {q3 - q1:8.2f} "
              f"{100 * f.mean():7.1f}")


def sec_recovery(rws, args):
    """Median recovered/injected by duty, and the model's prediction beside it."""
    ms = present(rws, SNR1_LIKE)
    sub = rws
    print("\n--- median recovered/injected S/N by duty (DETECTED-ONLY: biased high "
          "where detection is low; each method over what it ran) ---")
    print(f"{'duty':>12} {'n':>7} " + " ".join(f"{m:>14}" for m in ms) + f" {'model':>8}")
    for lo, hi in zip(BINS["ducy"][:-1], BINS["ducy"][1:]):
        g = [r for r in sub if lo <= r["ducy"] < hi]
        if len(g) < 20:
            continue
        cells = []
        for m in ms:
            v = np.array([r[m] / r["snr"] for r in g if m in r and np.isfinite(r[m])])
            cells.append(f"{np.median(v):14.3f}" if len(v) >= 10 else f"{'--':>14}")
        e = [r["model_eff"] for r in g if r.get("model_eff")]
        print(f"{f'{lo:g}-{hi:g}':>12} {len(g):>7} " + " ".join(cells) +
              (f" {np.median(e):8.3f}" if e else f" {'--':>8}"))
    if not any(r.get("model_eff") for r in sub):
        print("  (no model_eff recorded -- run 2 stores it per injection; "
              "`--model` computes it here instead)")


def sec_model(rws, thr, args):
    """Overlay the section-4 band-limited efficiency model on the measurement.

    It reproduced run 1's recovery to a few percent, so a DEPARTURE from it is a
    bug signal -- which is the only reason it is worth carrying in the analysis
    rather than in a notebook.  The predicted detection fraction assumes the
    statistic is unit-variance (which section 2b measured it to be, to three digits),
    so `P(det) = Phi(eff * snr - threshold)`.
    """
    m = args.ref
    if m not in present(rws, SNR1_LIKE):
        return
    sub = [r for r in rws if m in r and r.get("model_eff")]
    if not sub:
        return
    t = thr.get(m, 6.0)
    print(f"\n--- section-4 model vs measurement for {m} (threshold {t:.2f}) ---")
    print(f"{'duty':>12} {'n':>7} {'model eff':>10} {'meas eff':>10} "
          f"{'pred det%':>10} {'meas det%':>10}")
    for lo, hi in zip(BINS["ducy"][:-1], BINS["ducy"][1:]):
        g = [r for r in sub if lo <= r["ducy"] < hi]
        if len(g) < 20:
            continue
        eff = np.array([r["model_eff"] for r in g])
        snr = np.array([r["snr"] for r in g])
        w = np.array([r["weight"] for r in g])
        pred = np.sum(w * 0.5 * np.array([math.erfc((tt - e * s) / math.sqrt(2.0))
                                          for e, s, tt in zip(eff, snr, _tv(g, t))])) / w.sum()
        v = np.array([r[m] / r["snr"] for r in g if np.isfinite(r[m])])
        meas, _ = _det(g, m, t)
        print(f"{f'{lo:g}-{hi:g}':>12} {len(g):>7} {np.median(eff):10.3f} "
              f"{(np.median(v) if len(v) >= 10 else float('nan')):10.3f} "
              f"{100 * pred:10.1f} {100 * meas:10.1f}")
    print("  (measured efficiency is detected-only and so reads HIGH where "
          "detection is low;\n   the predicted vs measured DETECTION columns are "
          "the ones to compare)")


def sec_prepfold(rws, args):
    """prepfold's `snr1`, before and after the drizzle correction, against `f0`.

    Held at FIXED duty so the profile-shape loss is flat across the rows and the
    only thing moving is `dt_per_bin`.  Raw, the column RISES with frequency and
    crosses 1.0 -- an inflated ceiling in the MSP band, which is the worst place
    for one.  If a future run ever shows the corrected column rising again, the
    drizzle model has stopped describing what `fold()` does.
    """
    sel = [r for r in rws if 0.05 <= r["ducy"] < 0.20 and r.get("prepfold_snr1")]
    if len(sel) < 200:
        return
    print("\n--- prepfold snr1 / injected S/N at duty 5-20%, raw vs drizzle-corrected ---")
    print(f"{'f0 (Hz)':>14} {'n':>7} {'nbins':>6} {'dt/bin':>8} {'raw':>7} "
          f"{'corrected':>10} {'ref':>7}")
    ref = args.ref if args.ref in present(rws, SEARCHES) else None
    for lo, hi in ((0.1, 1), (1, 5), (5, 20), (20, 100), (100, 200),
                   (200, 300), (300, 400), (400, 1000)):
        g = [r for r in sel if lo <= r["f0"] < hi]
        if len(g) < 20:
            continue
        raw = np.median([r["prepfold_snr1_raw"] / r["snr"] for r in g])
        cor = np.median([r["prepfold_snr1"] / r["snr"] for r in g])
        rv = [r[ref] / r["snr"] for r in g if ref and np.isfinite(r.get(ref, np.nan))]
        print(f"{f'{lo:g}-{hi:g}':>14} {len(g):>7} "
              f"{np.median([r['prepfold_nbins'] for r in g]):6.0f} "
              f"{np.median([r['prepfold_dpb'] for r in g]):8.2f} {raw:7.3f} "
              f"{cor:10.3f} {(np.median(rv) if rv else float('nan')):7.3f}")
    print("  (raw is what `snr1` gives if the profile bins are assumed independent;\n"
          "   PRESTO's fold() drizzles each sample across the bins it covers, so they "
          "are not)")


def _pooled_corr(profs, widths):
    """Measured `sigma_bin / sd(S_w)` for a pool of NOISE profiles of one shape.

    This is exactly the factor `drizzle_boxcar_corr` returns, measured instead of
    modelled.  `S_w` is the zero-mean unit-L2 boxcar sum, evaluated at EVERY
    phase of every profile and pooled: phases within one profile are correlated,
    which costs precision but not accuracy, and profiles are independent.
    """
    P = np.asarray(profs, dtype=float)
    # Pooled mean, NOT per-profile: subtracting each profile's own mean removes a
    # degree of freedom from `nbins` and biases `sd_bin` low by `1/(2 nbins)` --
    # 0.4% at 128 bins, which is the same size as the effect at large `dt/bin`
    # and would have been read as a defect in the model.  `S_w` is a zero-sum
    # template, so it is invariant to any per-profile offset anyway.
    P = P - P.mean()
    n = P.shape[1]
    sd_bin = float(P.std())
    out = {}
    c = np.concatenate([P, P], axis=1).cumsum(axis=1)
    c = np.concatenate([np.zeros((len(P), 1)), c], axis=1)
    tot = P.sum(axis=1, keepdims=True)
    for w in widths:
        if w >= n:
            continue
        Sw = c[:, w:w + n] - c[:, 0:n]
        # (S_w - delta*S_tot) / sqrt(w(1-delta)) is `sum_j h_j P_j` exactly.
        t = (Sw - (w / n) * tot) / math.sqrt(w * (1.0 - w / n))
        out[w] = sd_bin / float(t.std())
    return out, sd_bin


def sec_drizzle(recs, args):
    """Measure prepfold's inter-bin correlation, against the model that corrects it.

    Section 5's correction rests on an ASSUMPTION about what PRESTO's `fold()` does
    -- that it spreads each sample across the bins its finite duration covers, in
    proportion to the overlap.  The stored profiles are what let that be checked
    rather than believed, and the injection-free realisations are the right data
    for it: they are pure noise folded at periods from the real population, so
    every number below is a direct measurement of the noise covariance the
    statistic divides by.

    If the measured column ever departs from the model column, the drizzle model
    has stopped describing `fold()` -- and every prepfold S/N in the study is
    then wrong by that factor, silently, in the direction that flatters it.
    """
    cells = defaultdict(list)
    for r in recs:
        if not r.get("empty"):
            continue
        for d in (r["results"].get("prepfold_null") or []):
            if d and d.get("prof") and d.get("nbins") and d.get("dt_per_bin"):
                # Bin dt_per_bin by half-decade: the correction is smooth in it,
                # and it is drawn continuously so no two folds share a value.
                cells[(d["nbins"], round(math.log10(d["dt_per_bin"]) * 2) / 2)].append(d["prof"])
    if not cells:
        print("\n--- prepfold drizzle check: no stored null profiles "
              "(--keep-profiles 0, or no injection-free realisations) ---")
        return
    print("\n--- prepfold inter-bin correlation: MEASURED vs the section-5 model ---")
    print(f"{'nbins':>6} {'dt/bin':>8} {'nprof':>6} {'sd/sqrt(C0)':>12} {'+-':>6}   "
          + "  ".join(f"w={w:<2d} meas/model" for w in (1, 2, 4, 8)))
    for (nb, ldpb), profs in sorted(cells.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        if len(profs) < 40:
            continue
        dpb = 10.0 ** ldpb
        meas, sd_bin = _pooled_corr(profs, (1, 2, 4, 8))
        # The per-bin sd itself: sqrt(C0 * rotations) on unit-variance data.
        pred_sd = MM.prepfold_sigma(nb, dpb, recs[0].get("N"))
        cells_txt = []
        for w in (1, 2, 4, 8):
            if w not in meas:
                cells_txt.append(f"{'--':>14}")
                continue
            mod = MM.drizzle_boxcar_corr(nb, dpb, w)
            cells_txt.append(f"{meas[w]:6.3f}/{mod:<6.3f} ")
        # A width-`w` boxcar has only ~nbins/w independent phases per profile, so
        # the wide columns are much noisier than the profile count suggests.
        err = 1.0 / math.sqrt(2 * len(profs) * max(nb / 8.0, 1.0))
        print(f"{nb:>6} {dpb:>8.2f} {len(profs):>6} {sd_bin / pred_sd:>12.4f} "
              f"{err:>6.3f}   " + "  ".join(cells_txt))
    print("  (sd/sqrt(C0) is the measured per-bin noise over the ANALYTIC one -- 1.000\n"
          "   means the closed-form sigma is right, and `+-` is its rough 1-sigma.\n"
          "   meas/model is the boxcar correction: they must agree, or `fold()` is not\n"
          "   drizzling the way `mc_model.fold_covariance` assumes and every prepfold\n"
          "   S/N is off by the gap.  Read the SMALL dt/bin rows: that is where the\n"
          "   correction is large (0.82-0.89, i.e. the MSP band) and so where being\n"
          "   wrong would matter.  At dt/bin > 30 the correction is ~1 either way.)")


# ---------------------------------------------------------------------------
# Red noise: everything, per knee bin
# ---------------------------------------------------------------------------
# Bin edges in Hz.  Chosen to straddle the measured knees (Parkes 1.6-1.8,
# GBT GUPPI 6.7, Arecibo 31.5) rather than to divide the sampled range evenly,
# so a row can be read against a real telescope.
KNEE_EDGES = (0.1, 0.5, 2.0, 6.0, 15.0, 50.0)
SIGMA_EDGES = (0.0, 0.1, 0.5, 2.0, 6.0, 1e9)


def knee_bins(recs, by="knee"):
    """`(label, [records])` per red-noise bin, with the injection-free... no:
    with the WHITE realisations first when any are present.

    Run 2's records carry `rednoise: null`, so pointing this at run 2 and run 3
    together puts the paired white control in the first row -- which is the
    comparison the whole red-noise study is built on, and it costs nothing to
    make it a row rather than a separate report.
    """
    edges = KNEE_EDGES if by == "knee" else SIGMA_EDGES
    key = "fknee" if by == "knee" else "sigma_got"
    out, white = [], [r for r in recs if not r.get("rednoise")]
    if white:
        out.append(("white", white))
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = [r for r in recs
               if r.get("rednoise") and lo <= (r["rednoise"].get(key) or 0.0) < hi]
        if sub:
            hi_s = "inf" if hi >= 1e8 else f"{hi:g}"
            out.append((f"{lo:g}-{hi_s}", sub))
    return out


def fa_band_labels(recs):
    """The frequency bands the run stored its false-alarm tails in, in order.

    Read off the records rather than hardcoded, so the analysis follows the
    driver.  Returns `[]` for a run written before per-band tails existed.
    """
    edges = None
    for r in recs:
        e = (r.get("config") or {}).get("fa_bands")
        if e:
            edges = e
            break
    if not edges:
        seen = set()
        for r in recs:
            for v in (r.get("results") or {}).values():
                if isinstance(v, dict) and isinstance(v.get("false"), dict):
                    seen |= set(v["false"].get("bands") or {})
        if not seen:
            return []
        def lo_of(lab):
            return float(lab.split("-")[0])
        return sorted(seen, key=lo_of)
    out = []
    for i in range(len(edges) - 1):
        hi = edges[i + 1]
        out.append(f"{edges[i]:g}-" + ("inf" if not np.isfinite(hi) else f"{hi:g}"))
    return out


def band_range(lab):
    lo, hi = lab.split("-")
    return float(lo), (float("inf") if hi == "inf" else float(hi))


def prepfold_null_thresholds(recs, labs, fap):
    """prepfold's matched cut per frequency band, from its null folds.

    prepfold folds at a KNOWN period, so its null has to come from the
    injection-free realisations -- and under red noise that null is entirely a
    function of the fold PERIOD.  Measured at a knee of 8-50 Hz, the null `snr1`
    median runs 2.8 at P < 5 ms to 44.9 above 2 s.  Pooling those gave one cut of
    191 for every period, and both prepfold columns then read 0.0% at every
    frequency INCLUDING the millisecond band, where its folds are clean.  That is
    the same pooling error `rseek` suffers across its candidate list, one level
    down.

    The fold period is not stored directly; `nbins * dt_per_bin * dt` is it
    exactly, which is why those three are recorded.

    `fap` is per REALISATION and each null realisation contributes several folds
    per band, so the per-FOLD tail probability wanted is `fap` divided by the
    folds per realisation in that band.  A cell can only resolve a rate its
    sample reaches: the returned `censored` flag marks a band where the
    requested rate needs fewer than `_NULL_MIN_ABOVE` folds above the cut, and
    those cells are never quoted as matched.
    """
    per_band = {lab: [] for lab in labs}
    nreal = 0
    for r in recs:
        pn = r["results"].get("prepfold_null")
        if not r.get("empty") or not pn:
            continue
        nreal += 1
        dt = r.get("dt") or 60e-6
        for d in pn:
            if not d:
                continue
            nb, dpb = d.get("nbins"), d.get("dt_per_bin")
            if not nb or not dpb:
                continue
            f0 = 1.0 / (nb * dpb * dt)
            for lab in labs:
                lo, hi = band_range(lab)
                if lo <= f0 < hi:
                    for key, col in (("chi2_sigma", "prepfold_chi2"),
                                     ("snr1", "prepfold_snr1")):
                        # `snr1` drizzle-corrected per fold, exactly as the
                        # injected folds are: the cut and the statistic it is
                        # compared against must be the same quantity.
                        v = _null_snr1(d) if key == "snr1" else d.get(key)
                        if v is not None and np.isfinite(v):
                            per_band[lab].append((col, v))
                    break
    out = {}
    for lab in labs:
        byc = defaultdict(list)
        for col, v in per_band[lab]:
            byc[col].append(v)
        for col, vals in byc.items():
            v = np.sort(np.asarray(vals))[::-1]
            folds_per_real = len(v) / max(nreal, 1)
            per = fap / max(folds_per_real, 1e-9)
            k = int(round(per * len(v)))
            censored = k < _NULL_MIN_ABOVE
            k = max(k, _NULL_MIN_ABOVE)
            out[(col, lab)] = (float(v[k - 1]) if len(v) >= k else float("inf"),
                               censored, len(v))
    return out


# A matched cut read off fewer than this many null folds above it is a number the
# sample cannot resolve, and is reported as censored rather than as a threshold.
_NULL_MIN_ABOVE = 3


def matched_threshold(recs, method, fap, band=None):
    """The exact cut at `fap` false alarms per realisation, from the tails.

    `pick_thresholds` reads its cut off `FA_GRID`, which is right for the pooled
    tables (one scan builds every FAP in the report).  Per band it is the wrong
    shape: the same scan would run once per (knee, band, method) cell, ~7x the
    pooled cost on a seventh of the data each time.  The cut wanted is just an
    order statistic of the pooled tail -- the `round(fap * nreal)`-th largest --
    which is exact rather than rounded to the grid, and one sort instead of a
    thousand comparisons per record.

    Returns `(threshold, nreal, ncand)`; the threshold is `-inf` when the band
    holds fewer candidates than the rate asks for, which means the cut is set by
    the code's reporting floor and not by these data.
    """
    vals, n = [], 0
    for r in recs:
        res = r["results"]
        d = res.get(method) or (_merge_union(res, method) if method in UNION else None)
        if not d or not d.get("ok", True):
            continue
        if band is None:
            tail = d["false"]["top"]
        else:
            bands = d["false"].get("bands")
            if bands is None:
                continue
            tail = (bands.get(band) or {}).get("top", [])
        n += 1
        vals.append(np.asarray(tail, dtype=float))
    if not n:
        return float("nan"), 0, 0
    vals = np.concatenate(vals)
    k = max(1, int(round(fap * n)))
    if len(vals) < k:
        return float("-inf"), n, len(vals)
    v = np.partition(vals, -k)[-k:]
    return float(v.min()), n, len(vals)


def band_floor(recs, method, lab):
    """The code's own reporting floor, as seen INSIDE one frequency band.

    Below it a rate curve is flat by construction -- no candidate below it was
    ever emitted -- so a matched cut that lands there is not a measurement, and a
    detection fraction counted above it is not either.  A band in which a code
    made no false alarm at all has no floor of its own; fall back to its pooled
    one, which is the same number measured on more candidates.
    """
    inb, pooled = [], []
    for r in recs:
        res = r["results"]
        d = res.get(method) or (_merge_union(res, method) if method in UNION else None)
        if not d or not isinstance(d.get("false"), dict):
            continue
        f = d["false"].get("floor")
        if f is not None:
            pooled.append(f)
        b = (d["false"].get("bands") or {}).get(lab)
        if b and b.get("floor") is not None:
            inb.append(b["floor"])
    src = inb or pooled
    return float(np.median(src)) if src else float("-inf")


# ---------------------------------------------------------------------------
# Matched thresholds per CELL -- what makes the pooled sections valid under red noise
# ---------------------------------------------------------------------------
# `pooled`: one cut per method over the whole run (run 2's report).
# `knee`: one per red-noise bin -- what a pipeline tuned to one observation's
#   noise level applies, and the default.
# `knee,band`: one per (red-noise bin, f0 band) -- rescues a code whose noise is
#   not stationary ACROSS its own output (rseek), at the price of making the
#   false-alarm rate per band.
# On records with no red noise all three are the same thing, so a white run's
# report is unchanged by the choice.
MATCH = ("pooled", "knee", "knee,band")
PF_COLS = ("prepfold_chi2", "prepfold_snr1")


def knee_label(row, by="knee"):
    """The `knee_bins` label of a row's realisation; `white` without red noise."""
    if row.get("fknee") is None:
        return "white"
    edges = KNEE_EDGES if by == "knee" else SIGMA_EDGES
    v = (row["fknee"] if by == "knee" else row.get("sigma_got")) or 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= v < hi:
            return f"{lo:g}-" + ("inf" if hi >= 1e8 else f"{hi:g}")
    return None


def band_label(f0, labs):
    for lab in labs:
        lo, hi = band_range(lab)
        if lo <= f0 < hi:
            return lab
    return None


class Cut:
    """One method's matched threshold, one value per CELL of the rows it meets.

    `kind` says what a cell is: `knee` (the realisation's red-noise bin) or
    `knee,band` (that, and the injection's f0 band).  A row in a cell with no cut
    gets `missing` -- `inf` for a search, i.e. no threshold there means no
    detection, and the nominal cut for prepfold, as in the pooled report.
    """

    def __init__(self, kind, cuts, knee_by, labs, missing=np.inf):
        self.kind, self.cuts, self.knee_by, self.labs = kind, cuts, knee_by, labs
        self.missing = missing

    def key(self, r):
        k = knee_label(r, self.knee_by)
        return k if self.kind == "knee" else (k, band_label(r["f0"], self.labs))

    def vec(self, rws):
        return np.array([self.cuts.get(self.key(r), self.missing) for r in rws],
                        dtype=float)

    def finite(self):
        return [v for v in self.cuts.values() if np.isfinite(v)]

    def __format__(self, spec):
        v = self.finite()
        if not v:
            return "--"
        lo, hi = min(v), max(v)
        spec = spec or ".2f"
        return format(lo, spec) if lo == hi else f"{lo:{spec}}-{hi:{spec}}"


def _tv(rws, t):
    """A threshold as one value per row, whatever form it came in."""
    if isinstance(t, Cut):
        return t.vec(rws)
    if np.ndim(t) == 0:
        return np.full(len(rws), float(t))
    return np.asarray(t, dtype=float)


def _tfinite(t):
    return bool(t.finite()) if isinstance(t, Cut) else bool(np.isfinite(t))


def _tmin(t):
    if isinstance(t, Cut):
        v = t.finite()
        return min(v) if v else float("inf")
    return t


@functools.lru_cache(maxsize=1 << 20)
def _drizzle_corr(nbins, dt_per_bin, w):
    """`MM.drizzle_boxcar_corr`, memoised on its EXACT arguments.

    Exact, not quantised: the same correction has to be applied to prepfold's
    null folds (which set its threshold) and to its injected folds (the
    statistic compared against that threshold), and rounding the two differently
    would put the mismatch straight back.  The cache matters because the report
    asks for eight false-alarm rates and re-corrects the same null folds for
    each.
    """
    return MM.drizzle_boxcar_corr(nbins, dt_per_bin, w)


def _null_snr1(d):
    """A null fold's `snr1`, drizzle-corrected the way `rows()` corrects the
    injected folds.

    Without this the cut was read off RAW `snr1` and applied to corrected
    values, so prepfold was held to a threshold up to ~12% too high wherever the
    correction bites -- the MSP band, where the correction is 0.83-0.89 and where
    prepfold is the ceiling column everything else is measured against.  It
    biased prepfold's own column low, which is the flattering direction for us,
    and it was in run 2's numbers too.
    """
    v = (d or {}).get("snr1")
    nb, dpb, w = d.get("nbins"), d.get("dt_per_bin"), d.get("w")
    if v is None or not (nb and dpb and w):
        return v
    return v * _drizzle_corr(nb, dpb, w)


def prepfold_nulls(recs):
    """prepfold's null statistics from the injection-free realisations' folds.

    `snr1` is drizzle-corrected per fold (see `_null_snr1`); `chi2_sigma` is
    prepfold's own chi-squared and takes no such correction.
    """
    nulls, nreal = defaultdict(list), 0
    for r in recs:
        pn = r["results"].get("prepfold_null")
        if not r.get("empty") or not pn:
            continue
        nreal += 1
        for d in pn:
            for k, col in (("chi2_sigma", "prepfold_chi2"), ("snr1", "prepfold_snr1")):
                v = _null_snr1(d) if k == "snr1" else (d or {}).get(k)
                if v is not None and np.isfinite(v):
                    nulls[col].append(v)
    return nulls, nreal


def prepfold_pooled_thresholds(recs, fap):
    """prepfold's matched cut over every fold period at once -- right on white noise.

    `fap` is per REALISATION, and each null realisation contributed
    `len(v)/nreal` folds, so the per-FOLD tail probability that gives that rate
    is fap divided by the folds per realisation.  Using the plain `1 - fap`
    quantile instead would quote a threshold several times too low and hand
    prepfold a free advantage in the one table where it is the ceiling everyone
    else is measured against.
    """
    nulls, nreal = prepfold_nulls(recs)
    out = {}
    for col, v in nulls.items():
        if len(v) < 200:
            continue
        per = fap / max(1.0, len(v) / max(nreal, 1))
        if per < 1.0:
            out[col] = float(np.quantile(v, 1.0 - per))
    return out


class CutBook:
    """Every method's matched cut at ANY false-alarm rate, under one `--match`.

    Built once and asked for as many rates as the report wants (the ROC asks for
    seven), so the per-group rate curves and per-band tails are scanned once.
    """

    def __init__(self, recs, match="knee", knee_by="knee", default=6.0):
        if match not in MATCH:
            raise SystemExit(f"--match {match}: choose from {MATCH}")
        self.recs, self.knee_by, self.default = recs, knee_by, default
        red = any(r.get("rednoise") for r in recs)
        self.match = match if red else "pooled"
        self.labs = fa_band_labels(recs)
        if self.match == "knee,band" and not self.labs:
            raise SystemExit("--match knee,band needs per-band false-alarm tails, "
                             "which these records do not carry")
        self.groups = (knee_bins(recs, knee_by) if self.match != "pooled"
                       else [(None, recs)])
        self._curves, self._tails = {}, {}
        # prepfold cells whose null sample cannot resolve the requested rate:
        # their cut is the lowest rate the sample CAN resolve, so the detection
        # fraction beside it is a lower bound.  Counted, and reported by `main`.
        self.censored = set()

    def _group_curves(self, glab, sub):
        if glab not in self._curves:
            self._curves[glab] = fa_curves(sub, present_recs(sub, SEARCHES))
        return self._curves[glab]

    def _band_cut(self, glab, sub, m, lab, fap):
        """`sec_band`'s cut: the band's order statistic, floored at the code's own
        reporting floor inside the band (below it the curve is flat by
        construction).  `nan` where the method stored no per-band tail."""
        key = (glab, m, lab)
        if key not in self._tails:
            vals, n = [], 0
            for r in sub:
                res = r["results"]
                d = res.get(m) or (_merge_union(res, m) if m in UNION else None)
                if not d or not d.get("ok", True):
                    continue
                bands = d["false"].get("bands")
                if bands is None:
                    continue
                n += 1
                vals.append(np.asarray((bands.get(lab) or {}).get("top", []), dtype=float))
            v = _desc(np.concatenate(vals)) if vals else np.zeros(0)
            self._tails[key] = (v, n, band_floor(sub, m, lab))
        v, n, fl = self._tails[key]
        if not n:
            return float("nan")
        k = max(1, int(round(fap * n)))
        t = float(v[k - 1]) if len(v) >= k else float("-inf")
        return fl if t <= fl else t

    def at(self, fap):
        if self.match == "pooled":
            out = pick_thresholds(self._group_curves(None, self.recs), fap)
            out.update(prepfold_pooled_thresholds(self.recs, fap))
            return out
        cuts = defaultdict(dict)
        for glab, sub in self.groups:
            if self.match == "knee":
                for m, t in pick_thresholds(self._group_curves(glab, sub), fap).items():
                    cuts[m][glab] = t
            else:
                for m in present_recs(sub, SEARCHES):
                    for lab in self.labs:
                        cuts[m][(glab, lab)] = self._band_cut(glab, sub, m, lab, fap)
            # prepfold's null is a function of the fold PERIOD under red noise
            # (null snr1 median 2.8 below 5 ms, 44.9 above 2 s at knee 8-50 Hz),
            # so a red bin is always matched per band; white is pooled, as in run 2.
            if glab == "white":
                for col, t in prepfold_pooled_thresholds(sub, fap).items():
                    for lab in (self.labs or [None]):
                        cuts[col][(glab, lab)] = t
            else:
                for (col, lab), (t, cens, _) in prepfold_null_thresholds(
                        sub, self.labs, fap).items():
                    cuts[col][(glab, lab)] = t
                    if cens:
                        self.censored.add((col, glab, lab))
        out = {}
        for m, c in cuts.items():
            pf = m in PF_COLS
            out[m] = Cut("knee,band" if (pf or self.match == "knee,band") else "knee",
                         c, self.knee_by, self.labs,
                         missing=(self.default if pf else np.inf))
        return out


def sec_band(recs, args, rng):
    """Matched threshold and detection fraction per FREQUENCY band.

    THE POINT OF THE SECTION, and it is the same argument as `sec_knee` one axis
    over: a threshold matched over a code's whole output is meaningful only where
    that code's noise is stationary across the output.  Ours is -- we search a
    whitened FFT and the false-alarm tail is flat in knee -- but `rseek` emits one
    candidate list from 1.33 ms to 10 s, and its time-domain dereddening is a
    high-pass at `1/rmed_width` that cannot reach red noise above ~0.25 Hz.  At a
    knee of 8-50 Hz its slow trials throw false alarms to S/N 128 while its fast
    folds stay clean: a real S/N-10 pulsar above 100 Hz reads 9.15 and is reported
    on 98% of injections.  One pooled cut is set by the junk and buries them, and
    run 3's pooled table read 0.0% at EVERY frequency as a result.

    So: cut inside the band, count detections inside the band.  The rate is
    `--fap` false alarms per realisation IN THAT BAND, so the pooled rate over
    `nbands` bands is up to `nbands x fap` -- that is deliberate, because the
    question here is "at equal false-alarm rate here, who detects more here?".
    The pooled table is still the operational number (one threshold is what a
    pipeline really applies), and both belong in the report.
    """
    labs = fa_band_labels(recs)
    if not labs:
        print("\n--- no per-band false-alarm tails in these records: "
              "re-run the driver to record them ---")
        return
    groups = knee_bins(recs, args.knee_by) if any(r.get("rednoise") for r in recs) \
        else [("all", recs)]
    print(f"\n--- per FREQUENCY band, every threshold matched INSIDE the band at "
          f"{args.fap:g} false alarms/realisation ---")
    pf_cols = ("prepfold_chi2", "prepfold_snr1")
    for glab, sub in groups:
        pfthr = prepfold_null_thresholds(sub, labs, args.fap)
        rws_all = rows(sub, hit_tol=args.hit_tol)
        apply_weights(rws_all, args.weight)
        ms = present_recs(sub, SEARCHES)
        ms = [m for m in ms if any(m in r for r in rws_all)]
        ms += [c for c in pf_cols if any(c in r for r in rws_all)]
        det, thr, note, nrow = {}, {}, {}, {}
        for lab in labs:
            lo, hi = band_range(lab)
            rws = [r for r in rws_all if lo <= r["f0"] < hi]
            nrow[lab] = len(rws)
            for m in ms:
                if m in pf_cols:
                    t, censored, nnull = pfthr.get((m, lab), (float("inf"), True, 0))
                    note[(m, lab)] = "c" if censored else ""
                else:
                    t, n, _ = matched_threshold(sub, m, args.fap, band=lab)
                    fl = band_floor(sub, m, lab)
                    if not n:
                        note[(m, lab)] = "-"
                    elif t <= fl:
                        # The band produced too few false alarms to set a cut, so
                        # the binding constraint is the code's own reporting
                        # floor.  Quote THAT and mark it: `rseek` makes no false
                        # alarm above 20 Hz at any knee, and a cut of 3.00 read
                        # off the bottom of the grid would count every candidate
                        # it ever emitted as a detection.
                        t, note[(m, lab)] = fl, "c"
                    else:
                        note[(m, lab)] = ""
                thr[(m, lab)] = t
                det[(m, lab)] = boot_det(rws, m, t, args.boot, rng)[:2] if rws \
                    else (float("nan"), float("nan"))
        title = f"  knee {glab} Hz" if glab != "all" else "  all realisations"
        print(f"\n{title}   (injections per band: " +
              " ".join(f"{lab} {nrow[lab]}" for lab in labs) + ")")
        head = f'{"method":>17} ' + " ".join(f"{l:>13}" for l in labs)
        for what in ("det", "thr"):
            print(f"\n  {'detection %' if what == 'det' else 'matched threshold'}")
            print(head)
            for m in ms:
                cells = []
                for lab in labs:
                    if what == "det":
                        pval, se = det[(m, lab)]
                        cells.append("            ." if not np.isfinite(pval)
                                     else f"{100 * pval:8.1f}+-{100 * se:3.1f}")
                    else:
                        t = thr[(m, lab)]
                        mark = note.get((m, lab), "")
                        cells.append("            ." if not np.isfinite(t)
                                     else f"{t:12.2f}{mark or ' '}")
                print(f"{m:>17} " + " ".join(cells))
        print("  ('c' on a search = the band made too few false alarms to set a cut, so the\n"
              "   code's own reporting floor is quoted instead and its detection fraction\n"
              "   is an upper bound;\n"
              "   'c' on prepfold = its null sample cannot resolve a rate this low in this band,\n"
              "   so the cut shown is the lowest rate it CAN resolve and the detection\n"
              "   fraction beside it is therefore a lower bound -- `--fap 0.1` resolves\n"
              "   these on a night's data, where 1e-2 needs roughly ten times more;\n"
              "   '-' = the run stored no per-band tail for this method)")


def sec_paired(rws, thr, args, rng):
    """Run 3 against run 2: the SAME injection, in the SAME white noise, red added.

    THE POINT OF THE SECTION.  Red noise draws off its own RNG stream, so at a
    given index the population draws, the injected S/N, the phases and the white
    noise are bit-for-bit run 2's.  The two runs are therefore a PAIRED sample,
    and the degradation can be taken injection by injection, with the population
    scatter cancelling instead of being averaged over.  Give the command both run
    directories.

    Threshold-free where it can be -- the median paired statistic needs no cut at
    all -- and matched per cell where a detection has to be counted: a white row
    against the white cut, a red row against its own knee bin's.

    The last block is what Lazarus et al. (2015) can be compared against: their
    factor 1.1-2 in Smin at P = 0.1-2 s (f0 0.5-10 Hz) and DM > 150.  Quote their
    HIGH-DM figure -- their DM dependence is RFI confusability, and this study
    models red noise only.
    """
    white = {(r["index"], r["inj"]): r for r in rws if r.get("fknee") is None}
    red = [r for r in rws if r.get("fknee") is not None]
    if not white or not red:
        print("\n--- paired white/red: needs BOTH runs on the command line, e.g.\n"
              "    mc_analyze.py /data/mc/run2 /data/mc/run3 ---")
        return
    pairs = [(white[k], r) for r in red
             for k in ((r["index"], r["inj"]),) if k in white]
    if not pairs:
        print("\n--- paired white/red: the two runs share no realisation index ---")
        return
    bad = sum(1 for w, r in pairs
              if abs(w["f0"] - r["f0"]) > 1e-12 or abs(w["snr"] - r["snr"]) > 1e-12)
    # EVERY method, not just the snr1-comparable ones: a paired difference
    # compares a code to ITSELF across the two runs, so the cross-code
    # comparability that `SNR1_LIKE` exists for is beside the point here.
    # Restricting it dropped both accelsearch arms, which are the field's
    # standard and the reason the comparison is interesting at all.
    ms = present(rws, METHODS)
    print(f"\n--- paired white/red: {len(pairs)} injections present in both runs "
          f"({len(red)} red rows, {len(white)} white) ---")
    if bad:
        print(f"  *** {bad} pairs disagree about the injection itself -- the runs are "
              "NOT paired.\n  *** Red noise must draw off `rng_for_rednoise`; check "
              "the driver before reading on.")
    else:
        print("  (every pair carries the identical injection, so the pairing holds)")

    klabs = _row_knee_labels([r for _, r in pairs], args.knee_by)
    fedges = BINS["f0"]
    fcols = list(zip(fedges[:-1], fedges[1:]))

    def cell(sel, m):
        v = [r[m] - w[m] for w, r in sel
             if np.isfinite(w.get(m, np.nan)) and np.isfinite(r.get(m, np.nan))]
        return (float(np.median(v)), len(v)) if len(v) >= 20 else (float("nan"), len(v))

    print(f"\n  median paired (red - white) statistic for {args.ref}, "
          "both runs recovering it")
    print(f'{"knee":>10} ' + " ".join(f"{f'{lo:g}-{hi:g}':>14}" for lo, hi in fcols))
    for kl in klabs:
        cells = []
        for lo, hi in fcols:
            d, n = cell([(w, r) for w, r in pairs
                         if knee_label(r, args.knee_by) == kl and lo <= r["f0"] < hi],
                        args.ref)
            cells.append(f"{'.':>14}" if not np.isfinite(d) else f"{d:+8.2f} ({n:>4d})")
        print(f"{kl:>10} " + " ".join(cells))

    print("\n  median paired (red - white) statistic per method, over all f0")
    print(f'{"method":>17} ' + " ".join(f"{l:>14}" for l in klabs))
    for m in ms:
        cells = []
        for kl in klabs:
            d, n = cell([(w, r) for w, r in pairs
                         if knee_label(r, args.knee_by) == kl], m)
            cells.append(f"{'.':>14}" if not np.isfinite(d) else f"{d:+6.2f}({n:>6d})")
        print(f"{m:>17} " + " ".join(cells))
    print("  (both-recovered only, so it is a shift in the statistic, not a "
          "detection fraction;\n   a pair the red run lost entirely cannot appear "
          "here -- that is the next block)")

    print("\n  paired detection: white-only (LOST to red) / red-only (gained), "
          "each at its own cell's cut")
    print(f'{"method":>17} ' + " ".join(f"{l:>14}" for l in klabs))
    for m in ms:
        cells = []
        for kl in klabs:
            sel = [(w, r) for w, r in pairs if knee_label(r, args.knee_by) == kl
                   and m in w and m in r]
            if len(sel) < 20:
                cells.append(f"{'.':>14}")
                continue
            tw = _tv([w for w, _ in sel], thr.get(m, thr["_default"]))
            tr = _tv([r for _, r in sel], thr.get(m, thr["_default"]))
            a = np.array([np.isfinite(w.get(m, np.nan)) and w[m] >= u
                          for (w, _), u in zip(sel, tw)])
            b = np.array([np.isfinite(r.get(m, np.nan)) and r[m] >= u
                          for (_, r), u in zip(sel, tr)])
            cells.append(f"{int((a & ~b).sum()):>6d} /{int((~a & b).sum()):>6d}")
        print(f"{m:>17} " + " ".join(cells))

    print(f"\n  injected S/N at 50% detection, and red/white ratio -- {args.ref}")
    print(f'{"knee":>10} {"S/N(50%)":>10} {"vs white":>9}   per f0 band, ratio to white:')
    wl = [w for w, _ in pairs]
    s50_w, _ = _logistic_s50(wl, args.ref, thr.get(args.ref, thr["_default"]))
    for kl in klabs:
        sel = [(w, r) for w, r in pairs if knee_label(r, args.knee_by) == kl]
        s50, _ = _logistic_s50([r for _, r in sel], args.ref,
                               thr.get(args.ref, thr["_default"]))
        cells = []
        for lo, hi in fcols[:5]:
            sb = [(w, r) for w, r in sel if lo <= r["f0"] < hi]
            a, _ = _logistic_s50([r for _, r in sb], args.ref,
                                 thr.get(args.ref, thr["_default"]))
            b, _ = _logistic_s50([w for w, _ in sb], args.ref,
                                 thr.get(args.ref, thr["_default"]))
            cells.append(f"{lo:g}-{hi:g}: " +
                         ("--" if not (np.isfinite(a) and np.isfinite(b) and b)
                          else f"{a / b:.2f}"))
        print(f"{kl:>10} {_fmt(s50, 2):>10} "
              f"{(f'{s50 / s50_w:.2f}x' if np.isfinite(s50) and np.isfinite(s50_w) and s50_w else '--'):>9}"
              f"   " + "  ".join(cells))
    print(f"  (white S/N(50%) = {_fmt(s50_w, 2)}.  A ratio is a degradation FACTOR in "
          "sensitivity, which is\n   what Lazarus et al. (2015) quote as 1.1-2 at "
          "P = 0.1-2 s for PALFA at DM > 150.\n   `--` where the fit did not reach 50% "
          "inside the injected 5.5-11.5 band.)")


def sec_hits(rws, thr, args):
    """Chance coincidences: recorded hits too far from their target to be the signal.

    THE POINT OF THE SECTION.  The driver's `score()` claims a candidate for an
    injection if it lies within `tol_bins` (3.0) of ANY ratio n/m <= 8 of f0, and
    a claimed candidate is removed from the false-alarm list -- so where a code
    emits a flood of candidates, coincidences enter as DETECTIONS and no matched
    threshold can see them.  Run 3's `rseek` does this at slow periods: at knee
    > 15 Hz, f0 5-20 Hz, its detection fraction ROSE with knee (80.5% against
    65.7% at knee < 0.5) while the median hit offset was 8 bins and the median
    hit statistic 25, at labels 1/8, 1/7 and 1/6.

    `--hit-tol` (default 0.5 bins) rejects those; this is what it cost and what
    it left behind.  A coincidence is uniform across the tolerance window, so the
    rejected hits ABOVE the cut in the sideband measure the density, and
    `hit_tol / (tol_bins - hit_tol)` of that is what remains inside the accepted
    window.  Real hits are nothing like uniform: on run 2's white noise, above
    each code's matched cut, the 99th-percentile offset is 0.15-0.23 bins.

    The one thing neither the cut nor this estimate can undo: the driver stores
    only the BEST hit per injection (fundamental first, then strongest), so a
    junk candidate at the fundamental ratio can have displaced a real one, and
    rejecting it scores a miss the code may not have made.  `displaced` counts
    the rejections that were above the cut -- an upper bound on that loss, and
    the same number as the coincidences the cut removed from the detections.
    """
    ms = [m for m in present(rws, SEARCHES) if any(r.get(m + "_off") is not None
                                                   for r in rws)]
    if not ms:
        print("\n--- chance coincidences: no hit frequencies recorded ---")
        return
    tol = np.median([r["tol_bins"] for r in rws if r.get("tol_bins")]) \
        if any(r.get("tol_bins") for r in rws) else 3.0
    if not np.isfinite(args.hit_tol) or args.hit_tol >= tol:
        print(f"\n--- chance coincidences: --hit-tol {args.hit_tol:g} does not cut "
              f"inside the driver's {tol:g}-bin tolerance, so every recorded hit "
              "stands ---")
        return
    groups = [(lab, [r for r in rws if knee_label(r, args.knee_by) == lab])
              for lab in _row_knee_labels(rws, args.knee_by)]
    frac = args.hit_tol / (tol - args.hit_tol)
    print(f"\n--- chance coincidences: hits more than {args.hit_tol:g} of the "
          f"driver's {tol:g} tolerance bins from their target ---")
    labs = [lab for lab, _ in groups]
    head = f'{"method":>17} ' + " ".join(f"{l:>14}" for l in labs)
    for what in ("rejected % of recorded hits",
                 "displaced: rejections above the cut, % of detections",
                 "estimated coincidences REMAINING, % of detections"):
        print(f"\n  {what}")
        print(head)
        for m in ms:
            cells = []
            for lab, g in groups:
                t = _tv(g, thr.get(m, thr["_default"]))
                nhit = sum(1 for r in g if r.get(m + "_off") is not None)
                rej = [r[m + "_junk"] for r in g if r.get(m + "_junk") is not None]
                above = sum(1 for r, u in zip(g, t)
                            if np.isfinite(r.get(m, np.nan)) and r[m] >= u)
                disp = sum(1 for r, u in zip(g, t)
                           if r.get(m + "_junk") is not None and r[m + "_junk"] >= u)
                if not nhit:
                    cells.append(f"{'.':>14}")
                elif what.startswith("rejected"):
                    cells.append(f"{100 * len(rej) / nhit:11.1f}   ")
                elif above < _HITS_MIN_DET:
                    # A ratio to a handful of detections is not a measurement.
                    # `rseek`'s cut in these cells is 170-315, so it detects
                    # almost nothing and the percentage read 900% once; the two
                    # counts say the same thing without pretending to precision.
                    cells.append(f"{f'{disp} of {above}':>14}")
                elif what.startswith("displaced"):
                    cells.append(f"{100 * disp / above:11.1f}   ")
                else:
                    cells.append(f"{100 * disp * frac / above:11.2f}   ")
            print(f"{m:>17} " + " ".join(cells))
    print("  (a coincidence is uniform in offset, so the sideband above the cut "
          f"measures its density\n   and {frac:.2f}x of it remains inside the "
          "accepted window -- that last row is what the\n   detection fractions "
          "still carry.  `displaced` is the upper bound on real hits the cut\n"
          "   threw away, because only the best hit per injection was stored.\n"
          f"   `d of n` = fewer than {_HITS_MIN_DET} detections in the cell, so "
          "the counts are quoted instead\n   of a ratio to them; '.' = the method "
          "recorded no hit there.)")


def _row_knee_labels(rws, by):
    """The knee labels present among rows, white first, then in bin order."""
    seen = {knee_label(r, by) for r in rws}
    edges = KNEE_EDGES if by == "knee" else SIGMA_EDGES
    order = ["white"] + [f"{lo:g}-" + ("inf" if hi >= 1e8 else f"{hi:g}")
                         for lo, hi in zip(edges[:-1], edges[1:])]
    return [l for l in order if l in seen]


def sec_knee(recs, args, rng):
    """Matched threshold and detection fraction per red-noise bin.

    THE POINT OF THE SECTION: the false-alarm rate is a function of the knee, so
    a threshold matched over the pooled run is matched to a MIXTURE of noise
    levels and is correct for none of them.  Every cut below is measured INSIDE
    its own bin, which is the only thing that makes these columns comparable to
    each other -- exactly the argument that makes the codes comparable in the
    first place, applied one level down.

    Read the `white` row as the zero point (give this both run directories), and
    read ACROSS a row for the degradation.  Lazarus et al. (2015) measured a
    factor 1.1-2 in Smin at P = 0.1-2 s and DM > 150 for PALFA -- quote their
    HIGH-DM figure, because their DM dependence is RFI confusability and this
    study models red noise only.
    """
    groups = knee_bins(recs, args.knee_by)
    if len(groups) < 2:
        print("\n--- no red-noise bins to split (records carry no `rednoise`) ---")
        return
    unit = "knee, Hz" if args.knee_by == "knee" else "realised sigma_red/sigma_w"
    print(f"\n--- per red-noise bin ({unit}), every threshold matched INSIDE its bin "
          f"at {args.fap:g} false alarms/realisation ---")

    ms, det, thr, warn, band_det = [], {}, {}, defaultdict(dict), {}
    for lab, sub in groups:
        curves = fa_curves(sub, present_recs(sub, SEARCHES))
        t = pick_thresholds(curves, args.fap)
        rws = rows(sub, hit_tol=args.hit_tol)
        apply_weights(rws, args.weight)
        # The same bin, matched per (knee x f0 band) instead: `rseek` emits one
        # candidate list over its whole range and its noise is not stationary
        # across it, so the two matchings answer different questions and the
        # report gives both.  See `sec_band` for the cells themselves.
        tb = {}
        if fa_band_labels(sub):
            tb = CutBook(sub, "knee,band", args.knee_by, args.threshold).at(args.fap)
        for m in present(rws, SEARCHES):
            if m not in ms:
                ms.append(m)
            p, se, _ = boot_det(rws, m, t.get(m, float("inf")), args.boot, rng)
            det[(m, lab)] = (p, se)
            thr[(m, lab)] = t.get(m, float("inf"))
            if m in tb:
                band_det[(m, lab)] = boot_det(rws, m, tb[m], args.boot, rng)[:2]
        for v in (r.get("results", {}) for r in sub):
            for arm, d in v.items():
                if isinstance(d, dict) and d.get("sigma_warn"):
                    warn.setdefault(arm, {})
                    warn[arm][lab] = warn[arm].get(lab, 0) + 1

    labs = [l for l, _ in groups]
    head = f'{"method":>17} ' + " ".join(f"{l:>14}" for l in labs)
    tabs = [("detection % (bootstrap se over realisations)", det, "det"),
            ("matched threshold", thr, "thr")]
    if band_det:
        tabs.insert(1, ("detection % with every cut matched per (this bin x f0 band) "
                        "instead", band_det, "det"))
    for title, tab, fmt in tabs:
        print(f"\n  {title}")
        print(head)
        for m in ms:
            cells = []
            for l in labs:
                if fmt == "det":
                    v = tab.get((m, l))
                    cells.append("             ." if v is None or not np.isfinite(v[0])
                                 else f"{100 * v[0]:8.1f}+-{100 * v[1]:4.1f}")
                else:
                    v = tab.get((m, l))
                    cells.append("             ." if v is None or not np.isfinite(v)
                                 else f"{v:14.2f}")
            print(f"{m:>17} " + " ".join(cells))
    print(f'\n{"realisations":>17} ' + " ".join(f"{len(s):>14d}" for _, s in groups))
    if warn:
        print("\n  the search's own sigma guard (analytic disagreeing with measured "
              "by >10%), per arm")
        print(head)
        for arm in sorted(warn):
            print(f"{arm:>17} " +
                  " ".join(f"{warn[arm].get(l, 0):>14d}" for l in labs))
        print("  (NOT a red-noise diagnostic, and it was read as one once.  On run 3 it "
              "fires only on\n   `coherent_tier`, at a rate FLAT in knee, and it fires "
              "the same way on pure white noise:\n   the guard's third sample point is "
              "the LAST chunk, a stub in a narrow band (57 trials\n   spanning 0.24 "
              "Fourier bins), so its measured sigma scatters +-7% against a 10%\n"
              "   tolerance over 12 rungs.  In the full chunks the analytic sigma is "
              "right to 0.3-1%, so\n   no result here is biased.  A guard firing on an "
              "arm that searches the WHOLE band, or a\n   rate that climbs with knee, "
              "would be the real thing.)")


def sec_duty(rws, args):
    """Duty recovery as bias + scatter, against the realised `ducy_got`."""
    ms = [m for m in present(rws, SEARCHES) if any(r.get(m + "_ducy") for r in rws)]
    if not ms:
        return
    print("\n--- recovered duty vs realised duty (median log10 ratio, and its MAD) ---")
    print(f"{'duty':>12} " + " ".join(f"{m:>18}" for m in ms))
    for lo, hi in zip(BINS["ducy"][:-1], BINS["ducy"][1:]):
        cells = []
        for m in ms:
            v = np.array([math.log10(r[m + "_ducy"] / max(r["ducy_got"], 1e-9))
                          for r in rws if lo <= r["ducy"] < hi and r.get(m + "_ducy")])
            cells.append(f"{np.median(v):+8.2f} +-{1.4826 * np.median(np.abs(v - np.median(v))):5.2f}"
                         if len(v) >= 20 else f"{'--':>18}")
        print(f"{f'{lo:g}-{hi:g}':>12} " + " ".join(f"{c:>18}" for c in cells))
    print("  (a boxcar bank quantises this; +0.30 is a factor of 2 too wide)")


def sec_2d(rws, thr, args):
    """2-D maps.  In the TPA population duty and frequency are strongly
    correlated, so a 1-D marginal cannot separate "we lose at narrow duty" from
    "we lose below 2 Hz"."""
    m = args.ref
    if m not in present(rws, SEARCHES):
        return
    sub = [r for r in rws if m in r]          # one method: no intersection needed
    t = thr.get(m, 6.0)
    for xk, yk in (("ducy", "f0"), ("ducy", "snr")):
        xe, ye = BINS[xk], BINS[yk]
        print(f"\n--- {m}: detection% over ({xk} x {yk}) ---")
        print(f"{yk + r' \ ' + xk:>14} " +
              " ".join(f"{lo:g}-{hi:g}".rjust(11) for lo, hi in zip(xe[:-1], xe[1:])))
        for ylo, yhi in zip(ye[:-1], ye[1:]):
            cells = []
            for xlo, xhi in zip(xe[:-1], xe[1:]):
                g = [r for r in sub if xlo <= r[xk] < xhi and ylo <= r[yk] < yhi]
                if len(g) < 15:
                    cells.append(f"{'.':>11}")
                    continue
                p, _ = _det(g, m, t)
                cells.append(f"{100 * p:6.1f}({len(g):>4d})")
            print(f"{f'{ylo:g}-{yhi:g}':>14} " + " ".join(cells))
    print("  ('.' = fewer than 15 injections in the cell)")


def sec_s50(rws, thr, args):
    """Injected S/N at 50% detection: one number per code, at matched FAP.

    Fitted, not interpolated, because the S/N grid can be coarse (run 1 used six
    integers).  Two-parameter logistic by IRLS -- no scipy, and a weighted fit so
    importance weights carry through.
    """
    ms = present(rws, SEARCHES)
    sub = rws
    print("\n--- injected S/N at 50% detection (logistic fit, matched FAP, "
          "each method over what it ran) ---")
    print(f"{'method':>16} {'S/N(50%)':>10} {'slope':>8}   per duty band:")
    bands = [(0.0, 0.01), (0.01, 0.04), (0.04, 0.16), (0.16, 0.5)]
    for m in ms:
        t = thr.get(m, thr["_default"])
        s50, b = _logistic_s50(sub, m, t)
        cells = []
        for lo, hi in bands:
            g = [r for r in sub if lo <= r["ducy"] < hi]
            v, _ = _logistic_s50(g, m, t)
            cells.append(f"{lo:g}-{hi:g}: {_fmt(v, 2)}")
        print(f"{m:>16} {_fmt(s50, 2):>10} {_fmt(b, 2):>8}   " + "  ".join(cells))


def _logistic_s50(rws, m, t, iters=25):
    x = np.array([r["snr"] for r in rws if m in r], dtype=float)
    w = np.array([r["weight"] for r in rws if m in r], dtype=float)
    v = np.array([r[m] for r in rws if m in r], dtype=float)
    if len(x) < 50:
        return float("nan"), float("nan")
    if isinstance(t, Cut):
        t = t.vec([r for r in rws if m in r])
    y = (np.nan_to_num(v, nan=-1e9) >= t).astype(float)
    if y.sum() < 10 or y.sum() > len(y) - 10:
        return float("nan"), float("nan")
    # The fit must be CONSTRAINED ON BOTH SIDES of 50% by measured points.  With
    # detection stuck below half across the whole injected band the logistic is
    # free to put its midpoint anywhere, and it did: accelsearch read S/N(50%)
    # 10.0 at knee 6-15 and then 7.57 at 15-50 -- sensitivity IMPROVING as the
    # noise got worse.  The range check below is not enough, because such a
    # midpoint can still land inside 5.5-11.5.  `coherent_tier` did the same
    # where it is scored outside its own 5 Hz band.
    seen = []
    for a, b in zip(BINS["snr"][:-1], BINS["snr"][1:]):
        sel = (x >= a) & (x < b)
        if sel.sum() >= 20:
            seen.append(float(np.average(y[sel], weights=w[sel])))
    if not seen or min(seen) >= 0.5 or max(seen) < 0.5:
        return float("nan"), float("nan")
    X = np.column_stack([np.ones_like(x), x])
    beta = np.array([-8.0, 1.0])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
        W = w * p * (1 - p) + 1e-12
        g = X.T @ (w * (y - p))
        H = X.T @ (X * W[:, None])
        try:
            beta = beta + np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
    if beta[1] <= 0.05:
        return float("nan"), float(beta[1])
    s50 = float(-beta[0] / beta[1])
    # The grid brackets 5.5-11.5; anything outside it is an extrapolation from a
    # curve that never reached 50%, and quoting it as a number invites it being
    # read as a measurement.  Run 1's `rseek_A` at duty < 1% fitted -5.93.
    lo, hi = float(x.min()) - 0.5, float(x.max()) + 0.5
    return (s50 if lo <= s50 <= hi else float("nan")), float(beta[1])


def sec_harm(rws):
    ms = present(rws, SEARCHES)
    print("\n--- detections at a harmonic ratio rather than the fundamental ---")
    for m in ms:
        tot = sum(1 for r in rws if r.get(m + "_harm"))
        nh = sum(1 for r in rws if r.get(m + "_harm") not in (None, "1"))
        if tot:
            print(f"{m:>16} {nh:>7} of {tot} matched candidates "
                  f"({100 * nh / tot:.1f}%)")
    print("  (counted AS detections: rseek and accelsearch do not collapse the "
          "f/2, 2f, 3f/2 family and we do)")


# ---------------------------------------------------------------------------
# The section-4 model, computed here when the driver did not record it
# ---------------------------------------------------------------------------
def add_model(rws, nharms, maxdecim, quiet=False):
    """Fill `model_eff` for rows that do not carry it (i.e. run-1 output).

    Cached on a rounded `(ducy, W10/W50)` grid: `make_profile` solves for the
    scattering tail and costs ~10 ms, which at 82,014 injections is 14 minutes,
    and the model is smooth in both arguments well below the rounding used here.
    """
    todo = [r for r in rws if not r.get("model_eff")]
    if not todo:
        return
    if not quiet:
        print(f"computing the section-4 efficiency model for {len(todo)} injections "
              "(rounded profile cache) ...", file=sys.stderr, flush=True)
    import mc_profiles as MP
    prof_cache, eff_cache = {}, {}
    for r in todo:
        key = (round(math.log10(max(r["ducy"], 1e-6)), 2), round(r["w10_w50"], 1))
        A = prof_cache.get(key)
        if A is None:
            prof, _ = MP.make_profile(10 ** key[0], key[1], nph=1 << 12)
            A = MM.profile_harmonics(prof, max(nharms, 4))
            prof_cache[key] = A
        # Harmonics past the data's own Nyquist are zero rows, so the fold is
        # truncated there and the model must know it.  Capped at `nharms`: above
        # that it changes nothing, which collapses every injection below
        # ~139 Hz (at dt = 60 us) onto one cache entry.
        hmax = min(nharms, int(math.floor(0.5 / (r["dt"] * r["f0"]))))
        ek = key + (hmax,)
        e = eff_cache.get(ek)
        if e is None:
            e = MM.ladder_efficiency(A, nharms, maxdecim, hmax)[0]
            eff_cache[ek] = e
        r["model_eff"] = e


# ---------------------------------------------------------------------------
SECTIONS = ("header", "cost", "falarm", "roc", "table", "pairs", "decompose",
            "scatter", "recovery", "model", "prepfold", "drizzle", "duty", "2d",
            "s50", "harm", "hits", "knee", "band", "paired")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--fap", type=float, default=1e-2,
                    help="give each code the threshold at which it makes this many "
                         "false alarms per realisation (default 1e-2)")
    ap.add_argument("--threshold", type=float, default=6.0,
                    help="nominal cut for methods with no false-alarm column (prepfold)")
    ap.add_argument("--knee-by", default="knee", choices=("knee", "sigma"),
                    help="bin the `knee` section by the DRAWN knee frequency or by "
                         "the REALISED sigma_red/sigma_w.  The latter is the finer "
                         "covariate -- at fixed knee the realised level spans "
                         "0.55-1.50x, because the red variance is dominated by a "
                         "few low-frequency bins")
    ap.add_argument("--match", default="knee", choices=MATCH,
                    help="where every threshold is matched: over the pooled run, "
                         "per red-noise bin (default), or per red-noise bin x f0 "
                         "band.  Identical on records without red noise")
    ap.add_argument("--hit-tol", type=float, default=HIT_TOL,
                    help="a hit further than this many Fourier bins from its "
                         "target is a chance coincidence, scored as a miss "
                         "(default %(default)s; `inf` scores as recorded)")
    ap.add_argument("--by", default="snr,ducy,f0",
                    help="comma-separated binning axes for the detection tables")
    ap.add_argument("--ref", default="coherent",
                    help="the method the paired and decomposition sections compare against")
    ap.add_argument("--weight", choices=("tpa", "flat"), default="tpa",
                    help="tpa keeps the sampler's importance weights (the real "
                         "population); flat re-weights to flat in log duty")
    ap.add_argument("--boot", type=int, default=200,
                    help="bootstrap resamples, BY REALISATION (0 disables)")
    ap.add_argument("--common", action="store_true", default=True)
    ap.add_argument("--no-common", dest="common", action="store_false",
                    help="do NOT restrict cross-code cells to the realisations "
                         "every compared code ran (run 1's mistake; here for "
                         "reproducing it deliberately)")
    ap.add_argument("--sections", default="all",
                    help="comma-separated subset of: " + ",".join(SECTIONS))
    ap.add_argument("--model", action="store_true",
                    help="compute the section-4 efficiency model for injections that "
                         "do not carry one (slow on a big run-1 set)")
    ap.add_argument("--nharms", type=int, default=60)
    ap.add_argument("--maxdecim", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    want = SECTIONS if args.sections == "all" else tuple(args.sections.split(","))
    bad = [s for s in want if s not in SECTIONS]
    if bad:
        raise SystemExit(f"unknown section(s) {bad}; choose from {SECTIONS}")

    recs = load(args.paths, with_profiles=("drizzle" in want))
    if not recs:
        raise SystemExit("no realisations found")
    rws = rows(recs, hit_tol=args.hit_tol)
    apply_weights(rws, args.weight)
    if args.model:
        add_model(rws, args.nharms, args.maxdecim)
    rng = np.random.default_rng(args.seed)

    # Every threshold in the report comes from here.  prepfold folds at the KNOWN
    # period, so it has no false-alarm column unless the run folded the
    # injection-free realisations at random periods too (runs 2 and 3 do); where
    # it did, prepfold joins the matched table as a proper ceiling.
    book = CutBook(recs, args.match, args.knee_by, args.threshold)
    thr = book.at(args.fap)
    thr["_default"] = args.threshold
    nulls, _ = prepfold_nulls(recs)

    if "header" in want:
        sec_header(recs, rws, args, book)
    print(f"\nthresholds at {args.fap:g} false alarms per realisation: " +
          ", ".join(f"{k} {v:.2f}" for k, v in sorted(thr.items()) if k != "_default"))
    if book.match != "pooled":
        print(f"  (matched per {'red-noise bin' if book.match == 'knee' else 'red-noise bin x f0 band'}"
              f" -- each is the range over cells; the `knee` and `band` sections list them."
              f"\n   prepfold is matched per (red-noise bin, f0 band) either way: its null"
              f" depends on the fold period.)")
        if book.censored:
            ncell = sum(len(v.cuts) for k, v in thr.items()
                        if k in PF_COLS and isinstance(v, Cut))
            print(f"  ({len(book.censored)} of {ncell} prepfold cells cannot resolve "
                  f"{args.fap:g}/realisation from their null folds: the cut shown is the "
                  f"lowest\n   rate that sample CAN resolve, so prepfold reads as a LOWER "
                  "bound there.  `--fap 0.1` needs ~10x less data.)")
    if any(k.startswith("prepfold") for k in nulls):
        print(f"  (prepfold_* are matched too, from {sum(len(v) for v in nulls.values())} "
              f"folds of the injection-free realisations at random periods.  Its "
              f"'trials'\n   are the folds per realisation -- it is a TARGETED fold, so "
              "that is a floor on\n   what a search would pay, which is what makes it a "
              "ceiling and not a competitor.)")
    else:
        print(f"  (prepfold_* keep the nominal {args.threshold:g}: this run did not fold "
              "the injection-free\n   realisations, so they have no measured null -- read "
              "them as a reference, not a\n   FAP-matched column)")

    if "cost" in want:
        sec_cost(recs, rws, thr, args)
    if "falarm" in want:
        sec_falarm(recs, args, thr)
    if "roc" in want:
        sec_roc(recs, rws, book, args, rng)
    if "table" in want:
        for key in [k.strip() for k in args.by.split(",") if k.strip()]:
            if key not in BINS:
                raise SystemExit(f"--by {key}: choose from {sorted(BINS)}")
            sec_table(rws, thr, args, rng, key)
        sec_table(rws, thr, args, rng)
    if "pairs" in want:
        sec_pairs(rws, thr, args)
    if "decompose" in want:
        sec_decompose(rws, thr, args)
    if "scatter" in want:
        sec_scatter(rws, args)
    if "recovery" in want:
        sec_recovery(rws, args)
    if "model" in want:
        sec_model(rws, thr, args)
    if "prepfold" in want:
        sec_prepfold(rws, args)
    if "drizzle" in want:
        sec_drizzle(recs, args)
    if "duty" in want:
        sec_duty(rws, args)
    if "2d" in want:
        sec_2d(rws, thr, args)
    if "s50" in want:
        sec_s50(rws, thr, args)
    if "harm" in want:
        sec_harm(rws)
    if "hits" in want:
        sec_hits(rws, thr, args)
    if "knee" in want:
        sec_knee(recs, args, rng)
    if "band" in want:
        sec_band(recs, args, rng)
    if "paired" in want:
        sec_paired(rws, thr, args, rng)
    return 0


if __name__ == "__main__":
    sys.exit(main())
