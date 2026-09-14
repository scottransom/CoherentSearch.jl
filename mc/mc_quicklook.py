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

**A SECOND page is written whenever the records carry red noise**
(`<out>_red.png`), because every panel on it is cut inside a red-noise bin and
putting that on the same axes as a pooled cut would be two different cuts on one
plot.  Its panels:

  1  detection vs knee, matched per knee bin  -- the degradation, at the cut a
                                                 pipeline tuned to one
                                                 observation would apply
  2  the same, matched per (knee x f0 band)   -- what a code whose noise is not
                                                 stationary across its own
                                                 output (rseek) can recover
  3  paired (red - white) statistic vs f0     -- run 3 against run 2 on the SAME
                                                 injection; needs both run
                                                 directories on the command line
  4  S/N at 50% detection vs knee             -- one number per code per bin
  5  false-alarm tail per knee bin            -- ours is flat in knee, rseek's is
                                                 not, and that is the whole story
                                                 of why its cut moves
  6  detection vs f0 per knee bin             -- red noise eats the slow end
                                                 first
  7  hit offset distributions                 -- real hits are tight, chance
                                                 coincidences are flat across
                                                 the tolerance window
  8  the drawn (knee, sigma_red, alpha)       -- a sampler check
"""

from __future__ import annotations

import argparse
import math
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
           "rseek_A": "#c1272d", "rseek_B": "#7b3fa0", "rseek_W": "#e8734a",
           "coherent": "#1a9850", "coherent_tier": "#0d6e33",
           "coherent_deep": "#66bd63", "coh+tier": "#054d21",
           "coherent_meas": "#2b8cbe", "coherent_rawmeas": "#7fcdbb"}
# `rseek_W` is riptide on a PRESTO-whitened series, so its statistic is the same
# snr1 as `rseek_A`'s and the two belong on the same axes: that pair IS the
# preprocessing-vs-FFA comparison run 4 exists to make.
SNR1_LIKE = ("prepfold_snr1", "rseek_A", "rseek_W", "coherent", "coh+tier")
# One colour per red-noise bin, dark (quiet) to bright (worst).
KNEE_COLOURS = ["#000000", "#2166ac", "#4393c3", "#f4a582", "#d6604d", "#b2182b"]


def frac(rws, m, thr, key, edges):
    """Detection fraction of method `m` in bins of `key`, with the denominator
    counting only injections that method actually saw (rseek_B runs on a
    subset).

    `thr` may be one cut or a per-cell one (`mc_analyze.Cut`), which is what
    makes these panels legible under red noise: a cut matched over a mixture of
    noise levels belongs to none of them."""
    x, y, n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rws if m in r and lo <= r[key] < hi]
        if not sel:
            continue
        v = np.array([r[m] for r in sel], dtype=float)
        tv = MA._tv(sel, thr)
        keep = ~np.isnan(tv)           # no cut in the cell: no measurement
        if not keep.any():
            continue
        v, tv = v[keep], tv[keep]
        x.append(math_mid(lo, hi, key))
        y.append(float(np.nansum(v >= tv)) / len(v))
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


def _mark_white(ax, x):
    """Label the paired WHITE control, which has no knee to be plotted at.

    It sits at the left edge as the zero point of every knee axis; without this
    it reads as a measurement at 0.05 Hz, which it is not.
    """
    ax.axvline(x, color="#888888", lw=0.8, ls=":")
    ax.annotate("white\n(run 2)", (x, 0.02), xycoords=("data", "axes fraction"),
                fontsize=6, color="#555555", ha="center", va="bottom")


def red_page(recs, rws, thr, book, args, methods):
    """The red-noise page: everything that has the knee as an axis.

    Separate from the white page on purpose -- every panel here needs a
    threshold matched INSIDE a red-noise bin, and overlaying that on the
    pooled-cut panels would put two different cuts on one axis.
    """
    klabs = MA._row_knee_labels(rws, args.knee_by)
    red_labs = [l for l in klabs if l != "white"]
    kx = {l: (0.05 if l == "white" else math_gmean(*MA.band_range(l)))
          for l in klabs}
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    unit = "red-noise knee (Hz)" if args.knee_by == "knee" else "realised sigma_red/sigma_w"

    # 1, 2 -- detection vs knee, matched per knee and then per (knee x band).
    for ax, mode, ttl in ((axes[0, 0], "knee", "matched per knee bin"),
                          (axes[0, 1], "knee,band", "matched per (knee x f0 band)")):
        try:
            t = MA.CutBook(recs, mode, args.knee_by, args.threshold).at(args.fap)
        except SystemExit:
            ax.set_axis_off()
            continue
        for m in methods:
            if m not in t:
                continue
            xs, ys, es = [], [], []
            for l in klabs:
                g = [r for r in rws if MA.knee_label(r, args.knee_by) == l and m in r]
                if len(g) < 30:
                    continue
                p, _ = MA._det(g, m, t[m])
                xs.append(kx[l])
                ys.append(100 * p)
                es.append(100 * math.sqrt(max(p * (1 - p), 1e-6) / len(g)))
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", ms=3, lw=1.4, capsize=2,
                            color=COLOURS.get(m), label=m)
        ax.set_xscale("log")
        ax.set_xlabel(unit)
        ax.set_ylabel("detected (%)")
        ax.set_ylim(-3, 103)
        ax.set_title(f"detection vs knee, {ttl}")
        ax.grid(alpha=0.25)
        if "white" in klabs:
            _mark_white(ax, kx["white"])
    axes[0, 0].legend(fontsize=7, loc="lower left")

    # 3 -- the paired degradation: run 3 against run 2, same injection.
    ax = axes[0, 2]
    white = {(r["index"], r["inj"]): r for r in rws if r.get("fknee") is None}
    pairs = [(white[k], r) for r in rws if r.get("fknee") is not None
             for k in ((r["index"], r["inj"]),) if k in white]
    m = args.ref
    if pairs:
        edges = [0.1, 1.0, 5.0, 20.0, 100.0, 1000.0]
        for l, c in zip(red_labs, KNEE_COLOURS[1:]):
            xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                d = [r[m] - w[m] for w, r in pairs
                     if MA.knee_label(r, args.knee_by) == l and lo <= r["f0"] < hi
                     and np.isfinite(w.get(m, np.nan)) and np.isfinite(r.get(m, np.nan))]
                if len(d) >= 20:
                    xs.append(math_gmean(lo, hi))
                    ys.append(float(np.median(d)))
            if xs:
                ax.plot(xs, ys, "o-", ms=4, lw=1.4, color=c, label=f"knee {l}")
        ax.axhline(0.0, color="k", lw=1, ls=":")
        ax.set_xscale("log")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "give BOTH run directories\nfor the paired comparison",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
    ax.set_xlabel("spin frequency (Hz)")
    ax.set_ylabel(f"median paired (red - white) {m}")
    ax.set_title("paired degradation (both runs recovering)")
    ax.grid(alpha=0.25)

    # 4 -- S/N at 50% detection against knee.
    ax = axes[0, 3]
    for m in methods:
        xs, ys = [], []
        for l in klabs:
            g = [r for r in rws if MA.knee_label(r, args.knee_by) == l and m in r]
            if len(g) < 200:
                continue
            s50, _ = MA._logistic_s50(g, m, thr.get(m, thr["_default"]))
            if np.isfinite(s50):
                xs.append(kx[l])
                ys.append(s50)
        if xs:
            ax.plot(xs, ys, "o-", ms=4, lw=1.4, color=COLOURS.get(m), label=m)
    ax.set_xscale("log")
    ax.set_xlabel(unit)
    ax.set_ylabel("injected S/N at 50% detection")
    ax.set_title("sensitivity vs knee (lower is better)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # 5 -- the false-alarm tail per knee bin: ours flat, rseek's not.
    ax = axes[1, 0]
    grid = np.arange(5.0, 40.01, 0.25)
    groups = dict(MA.knee_bins(recs, args.knee_by))
    for m, ls in ((args.ref, "-"), ("rseek_A", "--"), ("accelsearch", ":")):
        for l, c in zip(klabs, KNEE_COLOURS):
            sub = groups.get(l)
            if not sub:
                continue
            rate, n = MA.fa_rates(sub, m, grid)
            if n:
                ax.semilogy(grid, np.clip(rate, 1e-4, None), ls, lw=1.2, color=c,
                            label=f"{m} {l}")
    ax.axhline(args.fap, color="k", ls=":", lw=1)
    ax.set_xlabel("statistic cut")
    ax.set_ylabel("false alarms / realisation")
    ax.set_title(f"false-alarm tail per knee bin ({args.ref} solid, rseek_A dashed,\n"
                 "accelsearch dotted)")
    ax.legend(fontsize=5, ncol=2)
    ax.grid(alpha=0.25)

    # 6 -- where in frequency the loss happens, per knee bin.
    ax = axes[1, 1]
    for l, c in zip(klabs, KNEE_COLOURS):
        g = [r for r in rws if MA.knee_label(r, args.knee_by) == l]
        if len(g) < 200:
            continue
        x, y, n = frac(g, args.ref, thr.get(args.ref, thr["_default"]), "f0",
                       [0.1, 0.5, 2.0, 8.0, 30.0, 120.0, 400.0, 1000.0])
        if len(x):
            ax.plot(x, 100 * y, "o-", ms=3, lw=1.4, color=c, label=f"knee {l}")
    ax.set_xscale("log")
    ax.set_xlabel("spin frequency (Hz)")
    ax.set_ylabel("detected (%)")
    ax.set_ylim(-3, 103)
    ax.set_title(f"{args.ref}: detection vs f0, per knee bin")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # 7 -- hit offsets: how the chance coincidences were found.
    ax = axes[1, 2]
    tol = np.median([r["tol_bins"] for r in rws if r.get("tol_bins")]) \
        if any(r.get("tol_bins") for r in rws) else 3.0
    bins = np.linspace(0, tol, 40)
    # The worst knee bin for each code that floods, plus the reference code at the
    # quietest bin to show what a clean offset distribution looks like.
    series = []
    if red_labs:
        hi, lo = red_labs[-1], red_labs[0]
        series = [(args.ref, hi, "-", 1.0), ("rseek_A", hi, "--", 1.0),
                  ("accelsearch", hi, ":", 1.0), (args.ref, lo, "-", 0.4)]
    for m, l, ls, al in series:
        v = np.array([r[m + "_off"] for r in rws
                      if r.get(m + "_off") is not None
                      and MA.knee_label(r, args.knee_by) == l], dtype=float)
        if len(v) > 50:
            ax.hist(np.clip(v, 0, tol), bins=bins, histtype="step", ls=ls,
                    color=COLOURS.get(m), lw=1.3, alpha=al, density=True,
                    label=f"{m} knee {l}")
    ax.axvline(args.hit_tol, color="k", lw=1.2, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("hit offset from target (Fourier bins)")
    ax.set_ylabel("density")
    ax.set_title("real hits are tight; coincidences are flat")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.25)

    # 8 -- the drawn population, as a sampler check.
    ax = axes[1, 3]
    red = [r["rednoise"] for r in recs if r.get("rednoise")]
    if red:
        k = np.array([x["fknee"] for x in red])
        s = np.array([x.get("sigma_got") or np.nan for x in red])
        al = np.array([x["alpha"] for x in red])
        sc = ax.scatter(k, s, c=al, s=4, alpha=0.5, cmap="viridis")
        fig.colorbar(sc, ax=ax, label="spectral index alpha")
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("drawn knee (Hz)")
    ax.set_ylabel("realised sigma_red / sigma_white")
    ax.set_title("the red-noise population that was drawn")
    ax.grid(alpha=0.25)

    nred = sum(1 for r in recs if r.get("rednoise"))
    fig.suptitle(f"MC red-noise page — {nred} red realisations of {len(recs)}, "
                 f"thresholds matched per {book.match.replace(',', ' x ')} at "
                 f"{args.fap:g} false alarms/realisation, hit tolerance "
                 f"{args.hit_tol:g} bins", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


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
    ap.add_argument("--match", default="knee", choices=MA.MATCH,
                    help="where thresholds are matched; identical on records "
                         "without red noise")
    ap.add_argument("--knee-by", default="knee", choices=("knee", "sigma"))
    ap.add_argument("--hit-tol", type=float, default=MA.HIT_TOL,
                    help="a hit further than this many Fourier bins from its "
                         "target is a chance coincidence, scored as a miss")
    ap.add_argument("--ref", default="coherent",
                    help="the method the red-noise page follows in detail")
    ap.add_argument("--out-red", default=None,
                    help="the red-noise page (default: <out> with _red before the "
                         "suffix).  Written whenever the records carry red noise")
    args = ap.parse_args(argv)

    recs = MA.load(args.paths)
    if not recs:
        raise SystemExit("no realisations found")
    rws = MA.rows(recs, hit_tol=args.hit_tol)
    methods = [m for m in MA.METHODS if any(m in r for r in rws)]
    if args.methods:
        methods = [m for m in methods if m in args.methods.split(",")]

    book = None
    if args.fap is not None:
        book = MA.CutBook(recs, args.match, args.knee_by, args.threshold)
        thr = book.at(args.fap)
        thr["_default"] = args.threshold
        cut = (f"matched at {args.fap:g} false alarms/realisation"
               + (f", per {book.match.replace(',', ' x ')} cell"
                  if book.match != "pooled" else ""))
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
            # A per-cell cut is a set of numbers, not a line on a pooled axis:
            # the red page plots those against the knee instead.
            if m in thr and not isinstance(thr[m], MA.Cut) and np.isfinite(thr[m]):
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

    # The red-noise page, whenever there is red noise to plot.  Its own file
    # because every panel on it is cut INSIDE a red-noise bin, and putting that
    # on the same axes as a pooled cut would be two cuts on one plot.
    if any(r.get("rednoise") for r in recs):
        out_red = args.out_red or (os.path.splitext(args.out)[0] + "_red"
                                   + (os.path.splitext(args.out)[1] or ".png"))
        rfig = red_page(recs, rws, thr, book, args, methods)
        rfig.savefig(out_red, dpi=110)
        print(f"wrote {out_red}  (red-noise page)")

    if args.fap is None:
        print("NOTE: every code cut at the same nominal value, which is not a fair "
              "comparison.  Re-run with --fap 1e-2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
