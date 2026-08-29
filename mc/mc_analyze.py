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
  * **Quote prepfold's `snr1` raw.**  PRESTO's `fold()` drizzles each sample
    across the bins it covers and so correlates them; `snr1` assumes independence
    and reads up to 20% high in the MSP band, where prepfold is the ceiling
    column.  `mc_model.drizzle_boxcar_corr` is applied wherever `nbins`,
    `dt_per_bin` and the winning width are recorded.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_model as MM

# Every method that can appear, in report order.  `coh+tier` is DERIVED here (the
# union of the default search and the low-frequency deep tier), not recorded.
METHODS = ("prepfold_chi2", "prepfold_snr1", "accelsearch", "accelsearch_red",
           "rseek_A", "rseek_B", "coherent", "coherent_tier", "coherent_deep",
           "coh+tier")
SEARCHES = ("accelsearch", "accelsearch_red", "rseek_A", "rseek_B",
            "coherent", "coherent_tier", "coherent_deep", "coh+tier")
RECORDED = ("accelsearch", "accelsearch_red", "rseek_A", "rseek_B",
            "coherent", "coherent_tier", "coherent_deep")
# The union arm: a candidate list is the two arms' lists concatenated, which is
# what a tiered search would actually report.
UNION = {"coh+tier": ("coherent", "coherent_tier")}
# Statistics that are the same quantity (riptide's snr1), so their VALUES may be
# compared and not only their detection fractions.
SNR1_LIKE = ("prepfold_snr1", "rseek_A", "rseek_B", "coherent", "coherent_tier",
             "coherent_deep", "coh+tier")

BINS = {
    "snr":  [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5],
    "ducy": [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.50],
    "f0":   [0.1, 1.0, 5.0, 20.0, 100.0, 200.0, 400.0, 1000.0],
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load(paths):
    recs, seen = [], set()
    files = []
    for p in paths:
        files += glob.glob(os.path.join(p, "*.jsonl")) if os.path.isdir(p) else glob.glob(p)
    for f in sorted(files):
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if "error" in r or r.get("index") in seen:
                    continue
                seen.add(r["index"])
                recs.append(r)
    recs.sort(key=lambda r: r["index"])
    return recs


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
    top = sorted((v for p in parts for v in p["false"]["top"]), reverse=True)[:1600]
    return dict(hits=hits, ncand=sum(p["ncand"] for p in parts),
                ok=all(p.get("ok", True) for p in parts),
                false=dict(n=sum(p["false"]["n"] for p in parts), top=top,
                           truncated=any(p["false"].get("truncated") for p in parts)))


def rows(recs, dt=None):
    """One row per injection: every method's statistic, plus what it needs to be
    interpreted.  A method that did not RUN on a realisation leaves its key
    ABSENT (not nan), so a denominator can count what a method was actually
    given."""
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
        for i, inj in enumerate(r["injections"]):
            row = dict(index=r["index"], snr=inj["snr"], ducy=inj["ducy"],
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
                    s1 * MM.drizzle_boxcar_corr(nb, dpb, w)
                    if (s1 is not None and nb and dpb and w) else s1)
                row["prepfold_ducy"] = pf.get("ducy")
            for m in RECORDED:
                d = res.get(m)
                if d is None:
                    continue
                h = d["hits"][i] if i < len(d["hits"]) else None
                row[m] = h["stat"] if h else float("nan")
                row[m + "_harm"] = h["harmonic"] if h else None
                row[m + "_ducy"] = h.get("ducy") if h else None
                if h and h.get("width") and h.get("ducy"):
                    row[m + "_b"] = int(round(h["width"] / h["ducy"]))
                if h and h.get("nharm"):
                    row[m + "_nharm"] = h["nharm"]
            for name in UNION:
                if name in res:
                    d = res[name]
                    h = d["hits"][i] if i < len(d["hits"]) else None
                    row[name] = h["stat"] if h else float("nan")
                    row[name + "_harm"] = h["harmonic"] if h else None
                    row[name + "_ducy"] = h.get("ducy") if h else None
            out.append(row)
    return out


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

    The subset arms (`rseek_B`, `coherent_deep`) are 1-in-5, so intersecting
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
def fa_rates(recs, method, thresholds, empty_only=False):
    """False alarms per realisation at each threshold, from the stored top-N tails.

    Counted on EVERY realisation, not only the injection-free ones: a false alarm
    is a candidate matching no injection at any simple harmonic ratio, which is
    well defined either way, and run 1 verified the two agree to <= 0.10 in
    threshold at FAP 1e-2 -- so the larger sample is free.
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
        n += 1
        top = np.asarray(d["false"]["top"], dtype=float)
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
        if not d or not d["false"]["top"]:
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


FA_GRID = np.arange(3.0, 30.0, 0.05)


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
    d = np.where(np.isnan(v), False, v >= t) & ok
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
    d = (np.where(np.isnan(v), False, v >= t) & ok).astype(float)
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
    for r in rws:
        if m1 not in r or m2 not in r:
            continue
        a = np.isfinite(r[m1]) and r[m1] >= t1
        b = np.isfinite(r[m2]) and r[m2] >= t2
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


def sec_header(recs, rws, args):
    nempty = sum(1 for r in recs if r.get("empty"))
    print(f"{len(recs)} realisations ({nempty} injection-free), {len(rws)} injections")
    print(f"weighting: {args.weight}   effective sample size {ess(rws):.0f} injections")
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
        if thr and np.isfinite(thr.get(m, np.nan)) and thr[m] <= floor + 0.15:
            warn.append(f"    {m}: its matched threshold {thr[m]:.2f} is within 0.15 of "
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


def sec_roc(recs, rws, curves, args, rng):
    """Detection fraction vs false-alarm rate.  That the ordering is invariant
    from FAP 1 to 1e-3 is itself the result: it does not hinge on one threshold."""
    faps = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    full, part = split_coverage(rws, present(rws, SEARCHES))
    for ms in ([full] if not part else [full, full + part]):
        sub = common(rws, ms) if args.common else rws
        print(f"\n--- ROC: detection fraction vs false alarms/realisation "
              f"({len(sub)} injections{', the subset every arm ran' if len(ms) > len(full) else ''}) ---")
        print(f"{'FAP':>8} " + " ".join(f"{m:>16}" for m in ms))
        for f in faps:
            t = pick_thresholds(curves, f)
            cells = []
            for m in ms:
                p, _ = _det(sub, m, t.get(m, np.inf))
                cells.append(f"{100 * p:10.1f}% @{t.get(m, float('nan')):4.2f}")
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
          if m != args.ref and np.isfinite(thr.get(m, np.nan))]
    if args.ref not in present(rws, SNR1_LIKE) or not np.isfinite(thr.get(args.ref, np.nan)):
        return
    print(f"\n--- advantage decomposition: {args.ref} vs each snr1-comparable code ---")
    for m in ms:
        g = common(rws, [args.ref, m])
        if not g:
            continue
        dthr = thr.get(m, np.nan) - thr.get(args.ref, np.nan)
        print(f"\n  {args.ref} vs {m}:  threshold gap {dthr:+.2f} "
              f"(theirs {thr.get(m, float('nan')):.2f}, ours {thr.get(args.ref, float('nan')):.2f})")
        print(f"  {'duty':>10} {'n':>7} {'median dstat':>13} {'effective margin':>18}")
        for lo, hi in zip(BINS["ducy"][:-1], BINS["ducy"][1:]):
            cell = [r for r in g if lo <= r["ducy"] < hi
                    and np.isfinite(r[args.ref]) and np.isfinite(r[m])]
            if len(cell) < 20:
                continue
            d = np.median([r[args.ref] - r[m] for r in cell])
            print(f"  {f'{lo:g}-{hi:g}':>10} {len(cell):>7} {d:>13.2f} {d + dthr:>18.2f}")
        # Counterfactual: how much of the win survives if we are held to their cut?
        p_ours, _ = _det(g, args.ref, thr.get(args.ref, 6.0))
        p_theirs_cut, _ = _det(g, args.ref, thr.get(m, 6.0))
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
        pred = np.sum(w * 0.5 * np.array([math.erfc((t - e * s) / math.sqrt(2.0))
                                          for e, s in zip(eff, snr)])) / w.sum()
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
    y = (np.nan_to_num(v, nan=-1e9) >= t).astype(float)
    if y.sum() < 10 or y.sum() > len(y) - 10:
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
            "scatter", "recovery", "model", "prepfold", "duty", "2d", "s50", "harm")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--fap", type=float, default=1e-2,
                    help="give each code the threshold at which it makes this many "
                         "false alarms per realisation (default 1e-2)")
    ap.add_argument("--threshold", type=float, default=6.0,
                    help="nominal cut for methods with no false-alarm column (prepfold)")
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

    recs = load(args.paths)
    if not recs:
        raise SystemExit("no realisations found")
    rws = rows(recs)
    apply_weights(rws, args.weight)
    if args.model:
        add_model(rws, args.nharms, args.maxdecim)
    rng = np.random.default_rng(args.seed)

    curves = fa_curves(recs, present_recs(recs, SEARCHES))
    thr = pick_thresholds(curves, args.fap)
    thr["_default"] = args.threshold
    # prepfold folds at the KNOWN period, so it has no false-alarm column unless
    # the run folded the injection-free realisations at random periods too (run 2
    # does).  Where it did, prepfold joins the matched table as a proper ceiling.
    nulls = defaultdict(list)
    nreal_null = 0
    for r in recs:
        pn = r["results"].get("prepfold_null")
        if not r.get("empty") or not pn:
            continue
        nreal_null += 1
        for d in pn:
            for k, col in (("chi2_sigma", "prepfold_chi2"), ("snr1", "prepfold_snr1")):
                v = (d or {}).get(k)
                if v is not None and np.isfinite(v):
                    nulls[col].append(v)
    for col, v in nulls.items():
        if len(v) < 200:
            continue
        # `fap` is per REALISATION, and each null realisation contributed
        # `len(v)/nreal_null` folds, so the per-FOLD tail probability that gives
        # that rate is fap divided by the folds per realisation.  Using the plain
        # `1 - fap` quantile instead would quote a threshold several times too low
        # and hand prepfold a free advantage in the one table where it is the
        # ceiling everyone else is measured against.
        per = args.fap / max(1.0, len(v) / max(nreal_null, 1))
        if per < 1.0:
            thr[col] = float(np.quantile(v, 1.0 - per))

    if "header" in want:
        sec_header(recs, rws, args)
    print(f"\nthresholds at {args.fap:g} false alarms per realisation: " +
          ", ".join(f"{k} {v:.2f}" for k, v in sorted(thr.items()) if k != "_default"))
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
        sec_roc(recs, rws, curves, args, rng)
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
    if "duty" in want:
        sec_duty(rws, args)
    if "2d" in want:
        sec_2d(rws, thr, args)
    if "s50" in want:
        sec_s50(rws, thr, args)
    if "harm" in want:
        sec_harm(rws)
    return 0


if __name__ == "__main__":
    sys.exit(main())
