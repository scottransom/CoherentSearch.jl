#!/usr/bin/env python3
"""
mc_analyze.py -- combine and summarise mc_simulate.py output.

Combining runs is `cat`: every line is a self-contained realisation carrying its
index, so this just globs the .jsonl files, de-duplicates by (host-independent)
index, and tabulates.

Two things it deliberately does NOT do implicitly:

  * **compare codes at a common nominal threshold.**  Ours and rseek's statistics
    are single-trial, accelsearch's sigma is already trials-corrected, and
    prepfold's is a chi-squared -- so "S/N 6" means four different things.  Pass
    `--fap` to pick each code's threshold from its own measured false-alarm rate
    instead, which is the only comparison that means anything.
  * **treat a harmonic detection as a miss.**  Detections at f/2, 2f, 3f/2 ...
    are real recoveries of the signal; rseek and accelsearch do not collapse the
    family and we do, so scoring them as misses would penalise the codes that
    report them.  They are counted, and reported separately.

    mc_analyze.py mcout/                     # summary at the default threshold
    mc_analyze.py mcout/ --fap 1e-2          # thresholds matched on false-alarm rate
    mc_analyze.py mcout/ --by ducy           # detection fraction vs duty cycle
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

METHODS = ("prepfold_chi2", "prepfold_snr1", "accelsearch", "rseek_A", "rseek_B", "coherent")
SEARCHES = ("accelsearch", "rseek_A", "rseek_B", "coherent")   # the blind ones


def load(paths):
    recs, seen = [], set()
    files = []
    for p in paths:
        files += glob.glob(os.path.join(p, "*.jsonl")) if os.path.isdir(p) else glob.glob(p)
    for f in files:
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
    return recs


def rows(recs):
    """One row per injection per method: the statistic, or nan for no match."""
    out = []
    for r in recs:
        if r.get("empty"):
            continue
        res = r["results"]
        for i, inj in enumerate(r["injections"]):
            row = dict(index=r["index"], snr=inj["snr"], ducy=inj["ducy"],
                       f0=inj["f0"], msp=inj["msp"], w10_w50=inj["w10_w50"])
            pf = res.get("prepfold", [None] * (i + 1))[i]
            row["prepfold_chi2"] = pf["chi2_sigma"] if pf else float("nan")
            row["prepfold_snr1"] = pf["snr1"] if pf else float("nan")
            row["prepfold_ducy"] = (pf or {}).get("ducy")
            for m in SEARCHES:
                d = res.get(m)
                if d is None:
                    # The method did not RUN on this realisation (rseek_B only
                    # runs on a --deep-every subset).  Leave the key ABSENT, so
                    # the denominator below counts only what it was given: a nan
                    # would be indistinguishable from "ran and found nothing" and
                    # would divide a subset method's detections by every
                    # injection -- which read as 11.4% for rseek_B against
                    # rseek_A's 81.6% purely from the 1-in-3 duty cycle.
                    continue
                h = d["hits"][i] if i < len(d["hits"]) else None
                row[m] = h["stat"] if h else float("nan")
                row[m + "_harm"] = h["harmonic"] if h else None
                row[m + "_ducy"] = h.get("ducy") if h else None
            out.append(row)
    return out


def fa_rates(recs, method, thresholds):
    """False alarms per realisation at each threshold, from the top-N tails.

    Uses EVERY realisation, not only the injection-free ones: a false alarm is a
    candidate matching no injection at any harmonic ratio, which is well defined
    whether or not the realisation had signals in it.  The empty ones are the
    clean subset and are reported separately by `--empty-only`.
    """
    n = 0
    counts = np.zeros(len(thresholds))
    for r in recs:
        d = r["results"].get(method)
        if not d or not d.get("ok", True):
            continue
        n += 1
        top = np.array(d["false"]["top"], dtype=float)
        for j, t in enumerate(thresholds):
            counts[j] += float((top >= t).sum())
    return (counts / n if n else counts), n


def pick_thresholds(recs, fap, methods):
    """Per method, the lowest threshold whose false-alarm rate is <= `fap` per
    realisation.  This is what makes the codes comparable at all."""
    grid = np.arange(3.0, 25.0, 0.05)
    out = {}
    for m in methods:
        rate, n = fa_rates(recs, m, grid)
        if n == 0:
            continue
        ok = np.nonzero(rate <= fap)[0]
        out[m] = float(grid[ok[0]]) if len(ok) else float("inf")
    return out


def table(rws, thr, key=None, edges=None, label=""):
    methods = [m for m in METHODS if any(m in r for r in rws)]
    print(f"\n{'':>16} " + " ".join(f"{m:>13}" for m in methods) + f"   {'N':>6}")
    groups = [("all", rws)]
    if key:
        groups = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            g = [r for r in rws if lo <= r[key] < hi]
            groups.append((f"{lo:g}-{hi:g}", g))
    for name, g in groups:
        if not g:
            continue
        cells = []
        for m in methods:
            t = thr.get(m, thr.get("_default", 6.0))
            # Denominator is the injections this METHOD saw, not all of them.
            v = np.array([r[m] for r in g if m in r], dtype=float)
            if len(v) == 0:
                cells.append(f"{'--':>12} ")
                continue
            det = int(np.nansum(v >= t))
            cells.append(f"{100.0 * det / len(v):10.1f}%{'*' if len(v) < len(g) else ' '}")
        print(f"{label}{name:>{16 - len(label)}} " + " ".join(cells) + f"   {len(g):>6}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--threshold", type=float, default=6.0)
    ap.add_argument("--fap", type=float, default=None,
                    help="instead of one nominal threshold, give each code the threshold at which it makes this many false alarms per realisation")
    ap.add_argument("--by", choices=("snr", "ducy", "f0"), default="snr")
    args = ap.parse_args(argv)

    recs = load(args.paths)
    if not recs:
        raise SystemExit("no realisations found")
    rws = rows(recs)
    nempty = sum(1 for r in recs if r.get("empty"))
    print(f"{len(recs)} realisations ({nempty} injection-free), {len(rws)} injections")

    # timing, per realisation
    tk = {}
    for r in recs:
        for k, v in r.get("timing", {}).items():
            tk.setdefault(k, []).append(v)
    print("\nmedian wall per realisation (s):  " +
          "  ".join(f"{k} {np.median(v):.1f}" for k, v in sorted(tk.items(),
                                                                 key=lambda kv: -np.median(kv[1]))))

    # false-alarm rates
    grid = np.array([5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0])
    print("\nfalse alarms per realisation (candidates matching no injection):")
    print(f"{'':>14} " + " ".join(f"{t:>7.1f}" for t in grid))
    for m in SEARCHES:
        rate, n = fa_rates(recs, m, grid)
        if n:
            print(f"{m:>14} " + " ".join(f"{v:>7.2f}" for v in rate) + f"   (n={n})")

    if args.fap is not None:
        thr = pick_thresholds(recs, args.fap, SEARCHES)
        thr["_default"] = args.threshold
        # prepfold is a targeted fold, not a search: it has no false-alarm
        # column, so it keeps the nominal threshold and is a reference, not a
        # competitor.
        print(f"\nthresholds matched at {args.fap} false alarms per realisation: " +
              ", ".join(f"{k} {v:.2f}" for k, v in sorted(thr.items()) if k != "_default"))
    else:
        thr = {"_default": args.threshold}
        print(f"\nNOMINAL threshold {args.threshold} for every code -- these statistics are "
              "NOT the same quantity;\nuse --fap to compare them fairly.")

    edges = dict(snr=[5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5],
                 ducy=[0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.50],
                 f0=[0.1, 1.0, 5.0, 20.0, 100.0, 400.0, 1000.0])[args.by]
    print(f"\ndetection fraction by {args.by}:")
    table(rws, thr, args.by, edges)
    print("\noverall:")
    table(rws, thr)

    print("  (* = this method ran on only a subset of realisations, e.g. rseek_B "
          "under --deep-every;\n     its column is a fraction of what IT saw, not of "
          "every injection)")
    nh = sum(1 for r in rws for m in SEARCHES
             if r.get(m + "_harm") not in (None, "1"))
    print(f"\n{nh} detections were at a harmonic ratio rather than the fundamental "
          "(counted as detections; rseek/accelsearch do not collapse the family and we do)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
