#!/usr/bin/env python3
"""
mc_quicklook.py -- one page of diagnostic plots from mc_simulate.py output.

Zeroth-order: enough to see whether a run looks sane before anyone builds a
result on it.  Not the paper's figures.

    mc_quicklook.py mcout/ -o quicklook.png
    mc_quicklook.py mcout/ --fap 1e-2 -o quicklook.png

Panels, and what each is for:

  1  detection fraction vs injected S/N     -- must rise monotonically for every
                                               code; if it does not, something is
                                               wrong with the injection or scoring
  2  detection fraction vs duty cycle       -- the axis the study exists to measure
  3  detection fraction vs spin frequency   -- exposes band edges and the Nyquist
                                               knee, where our S/N used to inflate
  4  recovered vs injected S/N              -- only for the codes that report the
                                               SAME statistic (ours, rseek, and the
                                               snr1 computed on prepfold's profile);
                                               the diagonal is the ideal filter
  5  recovered vs true duty cycle           -- a boxcar bank quantises this, so
                                               expect steps, not a clean diagonal
  6  false alarms per realisation vs cut    -- the only thing that makes the codes
                                               comparable at all
  7  injected population (P, duty)          -- against the TPA median relation, so
                                               a sampler bug is visible at a glance
  8  wall clock per realisation by stage    -- cost has to be reported with
                                               sensitivity or either one is gamed

**A dashed vertical line marks each code's matched-FAP threshold** when `--fap` is
given.  Without it every code is cut at the same nominal value, which is NOT a
fair comparison -- ours and rseek's statistics are single-trial, accelsearch's
sigma is trials-corrected and prepfold's is a chi-squared.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_analyze as MA
import mc_profiles as MP

# One colour per method, used everywhere so the panels can be read together.
COLOURS = {"prepfold_chi2": "#8c8c8c", "prepfold_snr1": "#3b7dd8",
           "accelsearch": "#d9822b", "accelsearch_red": "#b06010",
           "rseek_A": "#c1272d", "rseek_B": "#7b3fa0",
           "coherent": "#1a9850", "coherent_tier": "#0d6e33",
           "coherent_deep": "#66bd63", "coh+tier": "#054d21"}
SNR1_LIKE = ("prepfold_snr1", "rseek_A", "coherent", "coh+tier")


def frac(rws, m, thr, key, edges):
    """Detection fraction of method `m` in bins of `key`, with the denominator
    counting only injections that method actually saw (rseek_B runs on a
    subset)."""
    x, y, n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        v = np.array([r[m] for r in rws if m in r and lo <= r[key] < hi], dtype=float)
        if len(v) == 0:
            continue
        x.append(math_mid(lo, hi, key))
        y.append(float(np.nansum(v >= thr)) / len(v))
        n.append(len(v))
    return np.array(x), np.array(y), np.array(n)


def math_mid(lo, hi, key):
    return math_gmean(lo, hi) if key in ("ducy", "f0") else 0.5 * (lo + hi)


def math_gmean(a, b):
    return float(np.sqrt(max(a, 1e-12) * b))


def panel_frac(ax, rws, methods, thr, key, edges, xlabel, logx=False):
    for m in methods:
        x, y, n = frac(rws, m, thr.get(m, thr["_default"]), key, edges)
        if len(x) == 0:
            continue
        # Binomial error bars: with a few hundred injections per cell the scatter
        # is large enough that a bare line invites over-reading.
        err = np.sqrt(np.clip(y * (1 - y), 1e-6, None) / n)
        ax.errorbar(x, 100 * y, yerr=100 * err, marker="o", ms=3, lw=1.4,
                    capsize=2, color=COLOURS.get(m), label=m)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("detected (%)")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.25)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-o", "--out", default="quicklook.png")
    ap.add_argument("--threshold", type=float, default=6.0)
    ap.add_argument("--fap", type=float, default=None,
                    help="give each code the threshold at which it makes this many false alarms per realisation")
    ap.add_argument("--methods", default=None,
                    help="comma-separated subset to plot (run 2 has ten, which is "
                         "too many curves for one panel to be read)")
    args = ap.parse_args(argv)

    recs = MA.load(args.paths)
    if not recs:
        raise SystemExit("no realisations found")
    rws = MA.rows(recs)
    methods = [m for m in MA.METHODS if any(m in r for r in rws)]
    if args.methods:
        methods = [m for m in methods if m in args.methods.split(",")]

    if args.fap is not None:
        thr = MA.pick_thresholds(MA.fa_curves(recs, MA.present_recs(recs, MA.SEARCHES)),
                                 args.fap)
        thr["_default"] = args.threshold
        cut = f"matched at {args.fap:g} false alarms/realisation"
    else:
        thr = {"_default": args.threshold}
        cut = f"nominal {args.threshold:g} for every code (NOT the same statistic)"

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(f"MC quick-look — {len(recs)} realisations, {len(rws)} injections, "
                 f"cut: {cut}", fontsize=13)

    panel_frac(axes[0, 0], rws, methods, thr, "snr",
               [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5], "injected S/N")
    axes[0, 0].set_title("detection vs injected S/N")
    axes[0, 0].legend(fontsize=7, loc="lower right")

    panel_frac(axes[0, 1], rws, methods, thr, "ducy",
               [0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.5],
               "FWHM duty cycle", logx=True)
    axes[0, 1].set_title("detection vs duty cycle")

    panel_frac(axes[0, 2], rws, methods, thr, "f0",
               [0.1, 0.5, 2.0, 8.0, 30.0, 120.0, 400.0, 1000.0],
               "spin frequency (Hz)", logx=True)
    axes[0, 2].set_title("detection vs spin frequency")

    # --- recovered vs injected S/N, only for the comparable statistics -------
    ax = axes[0, 3]
    # Binned, not grouped by exact value: injected S/N is CONTINUOUS as of run 2,
    # so grouping on equality would give one point per injection.
    edges = np.arange(5.5, 12.01, 0.5)
    for m in [m for m in SNR1_LIKE if m in methods]:
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            v = np.array([r[m] for r in rws
                          if m in r and lo <= r["snr"] < hi], dtype=float)
            v = v[np.isfinite(v)]
            if len(v) >= 10:
                xs.append(0.5 * (lo + hi))
                ys.append(np.median(v))
        ax.plot(xs, ys, "o-", ms=4, color=COLOURS.get(m), label=m)
    lim = [5, 13]
    ax.plot(lim, lim, "k--", lw=1, label="ideal (y = x)")
    ax.set_xlabel("injected S/N")
    ax.set_ylabel("median recovered statistic")
    ax.set_title("recovered vs injected (snr1-comparable only)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # --- recovered vs true duty --------------------------------------------
    ax = axes[1, 0]
    for m in ("coherent", "coherent_tier", "rseek_A"):
        if m not in methods:
            continue
        x = np.array([r["ducy"] for r in rws if r.get(m + "_ducy")], dtype=float)
        y = np.array([r[m + "_ducy"] for r in rws if r.get(m + "_ducy")], dtype=float)
        if len(x):
            ax.plot(x, y, ".", ms=3, alpha=0.35, color=COLOURS.get(m), label=m)
    ax.plot([1e-3, 0.5], [1e-3, 0.5], "k--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("true FWHM duty"); ax.set_ylabel("recovered boxcar duty")
    ax.set_title("duty recovery (boxcar bank quantises this)")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)

    # --- false alarms -------------------------------------------------------
    ax = axes[1, 1]
    grid = np.arange(5.0, 12.01, 0.25)
    for m in MA.SEARCHES:
        rate, n = MA.fa_rates(recs, m, grid)
        if n:
            ax.semilogy(grid, np.clip(rate, 1e-3, None), lw=1.5,
                        color=COLOURS.get(m), label=f"{m} (n={n})")
    if args.fap is not None:
        ax.axhline(args.fap, color="k", ls=":", lw=1)
        for m in MA.SEARCHES:
            if m in thr and np.isfinite(thr[m]):
                ax.axvline(thr[m], color=COLOURS.get(m), ls="--", lw=0.8)
    ax.set_xlabel("statistic cut"); ax.set_ylabel("false alarms / realisation")
    ax.set_title("false-alarm rate (what makes codes comparable)")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)

    # --- injected population ------------------------------------------------
    ax = axes[1, 2]
    P = np.array([1.0 / r["f0"] for r in rws])
    D = np.array([r["ducy"] for r in rws])
    msp = np.array([r["msp"] for r in rws], dtype=bool)
    ax.plot(P[~msp], D[~msp], ".", ms=3, alpha=0.35, color="#444444", label="slow (TPA)")
    ax.plot(P[msp], D[msp], ".", ms=3, alpha=0.35, color="#d9822b", label="MSP (prior)")
    pp = np.logspace(-2.7, 0.9, 50)
    ax.plot(pp, 10 ** (MP.TPA_SLOPE * np.log10(pp) + MP.TPA_INTERCEPT), "k-", lw=1.5,
            label="TPA median")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("period (s)"); ax.set_ylabel("FWHM duty")
    ax.set_title("injected population vs the TPA relation")
    ax.legend(fontsize=7); ax.grid(alpha=0.25)

    # --- cost ---------------------------------------------------------------
    ax = axes[1, 3]
    tk = {}
    for r in recs:
        for k, v in r.get("timing", {}).items():
            tk.setdefault(k, []).append(v)
    keys = sorted(tk, key=lambda k: -np.median(tk[k]))
    ax.barh(range(len(keys)), [np.median(tk[k]) for k in keys], color="#4c72b0")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("median wall clock per realisation (s)")
    tot = sum(np.median(tk[k]) for k in keys)
    ax.set_title(f"cost per realisation (total {tot:.0f} s)")
    ax.grid(alpha=0.25, axis="x")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}  ({len(recs)} realisations, {len(rws)} injections)")
    if args.fap is None:
        print("NOTE: every code cut at the same nominal value, which is not a fair "
              "comparison.  Re-run with --fap 1e-2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
