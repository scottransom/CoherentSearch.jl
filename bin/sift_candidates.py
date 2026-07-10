#!/usr/bin/env python3
"""
sift_candidates.py -- cross-observation sifting of CoherentSearch.jl candidates.

Analogue of PRESTO's ACCEL_sift.py / stacksift.py, but for the plain-text
candidate lists written by ``bin/coherent_search.jl`` and for a search that has
been run over *many observations* (epochs) as well as many DMs.

Each input file holds the candidates from ONE (observation, DM) pair.  The file
name encodes both, e.g. ``NGC6624_03L_DM87.10.txt`` -> obs="NGC6624_03L",
DM=87.10.  The columns are::

    rank   S/N   Frequency(Hz)   Period(ms)   #Harm

The sifting proceeds in three stages:

  1. Per observation, collapse candidates across DM into "detections": groups of
     rows at (nearly) the same frequency but different DM.  For each detection we
     record the S/N-vs-DM behaviour -- where it peaks, how many DMs it spans,
     whether the peak sits on the edge of the searched DM range, etc.

  2. Across observations, link detections at (nearly) the same frequency into
     "signals".  The frequency-match tolerance is fractional, so it is
     independent of the (unknown) observation length T.

  3. Collapse simple harmonic relations, score every signal for pulsar-likeness,
     and rank.

Nothing here is specific to any particular source or DM range: the DM grid, the
observation set, and the frequency scale are all learned from the input files.

Pulsar-likeness rewards signals that (a) persist across observations, (b) have a
consistent, *non-edge* DM of peak S/N, (c) are detected at >= 2 neighbouring DMs
(but usually not the whole searched range), and (d) have a consistent spin
frequency.  Broadband / edge-peaked / single-DM detections are flagged and
down-ranked, never silently dropped (use --min-score to trim the tail).

Author: written for Scott Ransom's CoherentSearch.jl.
"""

from __future__ import annotations

import argparse
import glob
import html
import math
import os
import re
import sys
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------
# Tunable scoring weights.  Edit these to change how the ranking is composed;
# every term is reported in the per-candidate breakdown (--verbose), so the
# ranking stays auditable.  (PRESTO's sifting module uses the same module-level
# convention.)
# ----------------------------------------------------------------------------
W_PERSIST = 3.0      # per observation the signal appears in
W_STRENGTH = 3.0     # x log2(sum of per-obs peak S/N)
W_DMEXTENT = 2.0     # x min(median #DMs per obs, cap)   -- being DM-extended
DMEXTENT_CAP = 6.0
W_DMPEAK = 6.0       # bonus for a tight, consistent peak-DM across observations
W_FREQ = 5.0         # bonus for a tight (isolated-pulsar) frequency across obs
P_EDGE = 4.0         # penalty x fraction of obs whose peak DM is at the grid edge
P_BROADBAND = 2.0    # extra penalty x frac of obs that are full-span AND edge-peaked
P_SINGLEDM = 5.0     # penalty x fraction of obs detected at a single DM (no neighbour)

# A real pulsar repeats at a consistent barycentric frequency: fractionally
# ~1e-6 (isolated) up to ~1e-4 (a binary whose period differs slightly per day).
# A chance agglomeration of per-observation noise picks, linked only because they
# happen to fall inside the (loose) cross-obs window, is frequency-INCOHERENT.
# This scale sets how tightly a signal must repeat to earn full persistence /
# strength credit: at dffrac=FREQ_COH_SCALE the coherence factor is ~0.37.
FREQ_COH_SCALE = 1e-4

# Cap on how many of the strongest signals are cross-checked for harmonic
# relations. Harmonics of the noise tail are irrelevant (only the top handful of
# signals are ever displayed), and an all-pairs check over every signal is
# O(N**2) -- ruinous when a tight tolerance yields tens of thousands of them.
HARMONIC_TOPK = 500

# A signal must peak in the interior (see edge test) unless its period exceeds
# this, in which case full-DM-span coverage is not held against it.  Long-period
# pulsars legitimately smear across a wide DM range.
DEFAULT_LONG_PERIOD_MS = 100.0

# Palette for per-observation SVG traces (colour-blind-friendly-ish, cycles).
_PALETTE = [
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#76b7b2",
    "#edc948", "#ff9da7", "#9c755f", "#bab0ac", "#1b9e77", "#d95f02",
]


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------
@dataclass
class Row:
    """A single candidate line at one (obs, DM)."""
    snr: float
    freq: float
    period_ms: float
    numharm: int


def parse_name(path, regex):
    """Return (obs, dm) parsed from a file's basename, or None if it doesn't match."""
    m = regex.search(os.path.basename(path))
    if not m:
        return None
    gd = m.groupdict()
    try:
        return gd["obs"], float(gd["dm"])
    except (KeyError, ValueError):
        return None


def read_rows(path, min_snr):
    """Read candidate rows from one file, keeping those with S/N >= min_snr."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                snr = float(parts[1])
                freq = float(parts[2])
                period_ms = float(parts[3])
                numharm = int(float(parts[4]))
            except ValueError:
                continue
            if freq <= 0.0 or snr < min_snr:
                continue
            rows.append(Row(snr, freq, period_ms, numharm))
    return rows


# ----------------------------------------------------------------------------
# Fractional-frequency clustering
# ----------------------------------------------------------------------------
def cluster_by_freq(items, freq_of, tol):
    """Greedy 1-D clustering of items by fractional frequency.

    Items are sorted by frequency; a new cluster starts whenever the gap to the
    running (S/N-weighted) cluster mean exceeds ``tol`` (fractional).  Linkage to
    the centroid -- not to the previous point -- prevents a long chain of
    barely-touching points from drifting arbitrarily far apart.
    """
    ordered = sorted(items, key=freq_of)
    clusters = []
    cur, cur_sumf, cur_n = [], 0.0, 0
    for it in ordered:
        f = freq_of(it)
        if cur and abs(f - cur_sumf / cur_n) > tol * (cur_sumf / cur_n):
            clusters.append(cur)
            cur, cur_sumf, cur_n = [], 0.0, 0
        cur.append(it)
        cur_sumf += f
        cur_n += 1
    if cur:
        clusters.append(cur)
    return clusters


# ----------------------------------------------------------------------------
# Stage 1: per-observation detections (collapse across DM)
# ----------------------------------------------------------------------------
@dataclass
class Detection:
    obs: str
    # peak-S/N row and its DM
    freq: float = 0.0
    period_ms: float = 0.0
    numharm: int = 0
    best_snr: float = 0.0
    best_dm: float = 0.0
    hits: dict = field(default_factory=dict)   # dm -> (snr, freq)
    # derived (filled by finalize)
    dms: list = field(default_factory=list)
    ndms: int = 0
    dm_min: float = 0.0
    dm_max: float = 0.0
    peak_at_edge: bool = False
    has_neighbour: bool = False
    has_gaps: bool = False
    full_span: bool = False

    def finalize(self, dm_index, dm_grid):
        self.dms = sorted(self.hits)
        self.ndms = len(self.dms)
        self.dm_min, self.dm_max = self.dms[0], self.dms[-1]
        self.peak_at_edge = self.best_dm in (dm_grid[0], dm_grid[-1])
        self.full_span = self.ndms == len(dm_grid)
        idx = [dm_index[d] for d in self.dms]
        diffs = [b - a for a, b in zip(idx[:-1], idx[1:])]
        self.has_neighbour = any(d == 1 for d in diffs)
        self.has_gaps = any(d > 1 for d in diffs)


def build_detections(obs, rows, within_tol, dm_index, dm_grid):
    """Collapse one observation's (row, DM) candidates into per-frequency detections.

    ``rows`` is a list of (dm, Row).  Rows are clustered by frequency; within a
    cluster we keep the strongest hit per DM, and the cluster's peak-S/N hit
    defines the detection's frequency / period / numharm.
    """
    clusters = cluster_by_freq(rows, lambda dr: dr[1].freq, within_tol)
    dets = []
    for cl in clusters:
        det = Detection(obs=obs)
        for dm, row in cl:
            prev = det.hits.get(dm)
            if prev is None or row.snr > prev[0]:
                det.hits[dm] = (row.snr, row.freq)
            if row.snr > det.best_snr:
                det.best_snr = row.snr
                det.best_dm = dm
                det.freq = row.freq
                det.period_ms = row.period_ms
                det.numharm = row.numharm
        det.finalize(dm_index, dm_grid)
        dets.append(det)
    return dets


# ----------------------------------------------------------------------------
# Stage 2: cross-observation signals
# ----------------------------------------------------------------------------
@dataclass
class Signal:
    freq: float = 0.0
    period_ms: float = 0.0
    dets: dict = field(default_factory=dict)   # obs -> Detection (best per obs)
    # derived
    n_obs: int = 0
    sum_snr: float = 0.0
    max_snr: float = 0.0
    freq_frac_spread: float = 0.0
    freq_coherence: float = 0.0
    dm_med: float = 0.0
    dm_mad_steps: float = 0.0
    median_ndms: float = 0.0
    frac_edge: float = 0.0
    frac_single: float = 0.0
    frac_broadband: float = 0.0
    score: float = 0.0
    components: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    klass: str = ""
    harmonic_of: object = None    # Signal this is a harmonic of, if any
    harm_str: str = ""

    def finalize(self, dm_step, long_period_ms):
        obs_dets = list(self.dets.values())
        self.n_obs = len(obs_dets)
        snrs = [d.best_snr for d in obs_dets]
        self.sum_snr = sum(snrs)
        self.max_snr = max(snrs)
        # representative freq/period = strongest observation
        best = max(obs_dets, key=lambda d: d.best_snr)
        self.freq = best.freq
        self.period_ms = best.period_ms
        freqs = [d.freq for d in obs_dets]
        fmean = sum(freqs) / len(freqs)
        self.freq_frac_spread = (max(freqs) - min(freqs)) / fmean if fmean else 0.0
        peak_dms = sorted(d.best_dm for d in obs_dets)
        self.dm_med = peak_dms[len(peak_dms) // 2]
        mad = _median([abs(d - self.dm_med) for d in peak_dms])
        self.dm_mad_steps = mad / dm_step if dm_step else 0.0
        self.median_ndms = _median([d.ndms for d in obs_dets])
        self.frac_edge = sum(d.peak_at_edge for d in obs_dets) / self.n_obs
        self.frac_single = sum(not d.has_neighbour for d in obs_dets) / self.n_obs
        self.frac_broadband = sum(
            d.full_span and d.peak_at_edge and d.period_ms < long_period_ms
            for d in obs_dets) / self.n_obs


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def build_signals(detections, cross_tol, dm_step, long_period_ms):
    """Link per-observation detections into cross-observation signals."""
    clusters = cluster_by_freq(detections, lambda d: d.freq, cross_tol)
    signals = []
    for cl in clusters:
        sig = Signal()
        for det in cl:
            cur = sig.dets.get(det.obs)
            if cur is None or det.best_snr > cur.best_snr:
                sig.dets[det.obs] = det
        sig.finalize(dm_step, long_period_ms)
        signals.append(sig)
    return signals


# ----------------------------------------------------------------------------
# Stage 3: harmonic relations
# ----------------------------------------------------------------------------
_HARM_RATIOS = [(n, 1) for n in range(2, 17)] + [
    (1, 2), (3, 2), (5, 2), (2, 3), (4, 3), (5, 3), (3, 4), (5, 4)]


def mark_harmonics(signals, tol):
    """Flag weaker signals that are harmonically related to a stronger one.

    Sorted strongest-first (by sum S/N x observations); each remaining signal is
    tested against every accepted fundamental for integer and small-ratio
    frequency relations.  Harmonics are annotated and later down-ranked, but kept
    in the output.
    """
    order = sorted(signals, key=lambda s: s.sum_snr * s.n_obs,
                   reverse=True)[:HARMONIC_TOPK]
    funds = []
    for sig in order:
        for f in funds:
            hs = _harmonic_relation(f.freq, sig.freq, tol)
            if hs:
                sig.harmonic_of = f
                sig.harm_str = hs
                break
        if sig.harmonic_of is None:
            funds.append(sig)


def _harmonic_relation(f_fund, f_test, tol):
    """Return a label if f_test is a simple harmonic of f_fund, else ''."""
    for num, den in _HARM_RATIOS:
        ratio = num / den
        if abs(f_test - f_fund * ratio) < tol * f_fund * ratio:
            return "%d/%d" % (num, den) if den != 1 else "x%d" % num
    return ""


# ----------------------------------------------------------------------------
# Scoring & classification
# ----------------------------------------------------------------------------
def score_signal(sig, min_dms):
    c = {}
    # Frequency coherence: 1 for a signal that repeats to << FREQ_COH_SCALE, ~0.37
    # at exactly FREQ_COH_SCALE (a slightly-different-per-day binary), ~0 for an
    # incoherent noise agglomeration.  It gates persistence & strength so a pile
    # of unrelated per-obs noise picks can't win on observation count alone.
    coh = math.exp(-(sig.freq_frac_spread / FREQ_COH_SCALE) ** 2)
    sig.freq_coherence = coh
    c["persist"] = W_PERSIST * sig.n_obs * (0.2 + 0.8 * coh)
    c["strength"] = W_STRENGTH * math.log2(max(sig.sum_snr, 1.0)) * (0.3 + 0.7 * coh)
    c["dm_extent"] = W_DMEXTENT * min(sig.median_ndms, DMEXTENT_CAP)
    # tight, consistent peak DM across observations (only meaningful for n_obs>=2)
    if sig.n_obs >= 2:
        c["dm_consist"] = W_DMPEAK * math.exp(-(sig.dm_mad_steps / 3.0) ** 2)
    else:
        c["dm_consist"] = 0.0
    # tight frequency => isolated pulsar bonus; binaries (~1e-4) get a small bonus
    c["freq_consist"] = W_FREQ * coh
    c["edge"] = -P_EDGE * sig.frac_edge
    c["broadband"] = -P_BROADBAND * sig.frac_broadband
    c["single_dm"] = -P_SINGLEDM * sig.frac_single
    sig.components = c
    sig.score = sum(c.values())
    if sig.harmonic_of is not None:
        sig.score -= 100.0     # sink harmonics below genuine fundamentals

    # human-readable flags
    flags = []
    if sig.n_obs == 1:
        flags.append("single-obs")
    if sig.frac_edge > 0.5:
        flags.append("edge-peaked")
    if sig.frac_broadband > 0.0:
        flags.append("broadband")
    if sig.frac_single > 0.5:
        flags.append("no-DM-neighbour")
    if sig.dm_mad_steps > 3.0 and sig.n_obs >= 2:
        flags.append("DM-inconsistent")
    if all(d.full_span for d in sig.dets.values()):
        flags.append("full-DM-span")
    if sig.harmonic_of is not None:
        flags.append("harm(%s of %.6fHz)" % (sig.harm_str, sig.harmonic_of.freq))
    sig.flags = flags

    # coarse class
    if sig.harmonic_of is not None:
        sig.klass = "harmonic"
    elif sig.frac_edge > 0.5 or sig.frac_broadband > 0.5:
        sig.klass = "RFI-like"
    elif (sig.n_obs >= 3 and sig.dm_mad_steps <= 3.0 and sig.frac_edge == 0.0
          and sig.median_ndms >= min_dms):
        sig.klass = "PULSAR?"
    elif sig.n_obs >= 2 and sig.frac_edge < 0.5 and sig.median_ndms >= min_dms:
        sig.klass = "candidate"
    else:
        sig.klass = "weak"


# ----------------------------------------------------------------------------
# Reporting -- text
# ----------------------------------------------------------------------------
def fmt_dm_list(dms, decimals):
    return ",".join(("%.*f" % (decimals, d)).rstrip("0").rstrip(".") for d in dms)


def write_text(signals, opts, out):
    obs_all = sorted({o for s in signals for o in s.dets})
    p = lambda *a: print(*a, file=out)
    p("# CoherentSearch.jl candidate sift")
    p("# %d input files, %d observations, %d DMs (%.*f - %.*f)"
      % (opts["nfiles"], len(obs_all), len(opts["dm_grid"]),
         opts["dm_decimals"], opts["dm_grid"][0],
         opts["dm_decimals"], opts["dm_grid"][-1]))
    p("# within_tol=%.1e  cross_tol=%.1e  min_snr=%.1f  min_dms=%d  min_obs=%d"
      % (opts["within_tol"], opts["cross_tol"], opts["min_snr"],
         opts["min_dms"], opts["min_obs"]))
    p("# observations: %s" % ", ".join(obs_all))
    p("#")
    p("# %-5s %-9s %14s %12s %5s %5s %8s %8s %8s %9s %7s  %s"
      % ("rank", "class", "freq(Hz)", "P(ms)", "#obs", "#har",
         "sumS/N", "maxS/N", "peakDM", "dMAD(st)", "dffrac", "flags"))
    for i, s in enumerate(signals, 1):
        numharm = int(round(_median([d.numharm for d in s.dets.values()])))
        p("%-7d %-9s %14.9f %12.4f %5d %5d %8.1f %8.1f %8.*f %9.2f %7.1e  %s"
          % (i, s.klass, s.freq, s.period_ms, s.n_obs, numharm,
             s.sum_snr, s.max_snr, opts["dm_decimals"], s.dm_med,
             s.dm_mad_steps, s.freq_frac_spread, ";".join(s.flags)))
        if opts["verbose"]:
            comp = "  ".join("%s=%+.1f" % (k, v) for k, v in s.components.items())
            p("        score=%.1f  [%s]" % (s.score, comp))
            for o in sorted(s.dets):
                d = s.dets[o]
                p("        %-14s peakDM=%.*f  S/N=%.1f  #DM=%d  DMrange=[%.*f,%.*f]%s"
                  % (o, opts["dm_decimals"], d.best_dm, d.best_snr, d.ndms,
                     opts["dm_decimals"], d.dm_min, opts["dm_decimals"], d.dm_max,
                     "  EDGE" if d.peak_at_edge else ""))
    p("#")
    p("# %d signals shown (of %d total; use --min-score / --top to adjust)."
      % (len(signals), opts["ntotal"]))


# ----------------------------------------------------------------------------
# Reporting -- HTML/SVG (self-contained, no external dependencies)
# ----------------------------------------------------------------------------
def _svg_snr_vs_dm(sig, dm_grid, obs_colors, width=380, height=200):
    """One S/N-vs-DM panel: a polyline per observation over the searched DM grid."""
    ml, mr, mt, mb = 42, 10, 14, 30
    pw, ph = width - ml - mr, height - mt - mb
    dmin, dmax = dm_grid[0], dm_grid[-1]
    smax = max(d.best_snr for d in sig.dets.values())
    smax = max(smax, 1.0) * 1.08
    smin = 0.0

    def X(dm):
        return ml + (dm - dmin) / (dmax - dmin or 1) * pw

    def Y(s):
        return mt + (1 - (s - smin) / (smax - smin)) * ph

    out = ['<svg viewBox="0 0 %d %d" class="snr">' % (width, height)]
    # axes
    out.append('<line class="ax" x1="%g" y1="%g" x2="%g" y2="%g"/>'
               % (ml, mt + ph, ml + pw, mt + ph))
    out.append('<line class="ax" x1="%g" y1="%g" x2="%g" y2="%g"/>'
               % (ml, mt, ml, mt + ph))
    # y ticks
    for frac in (0.0, 0.5, 1.0):
        yv = smin + frac * (smax - smin)
        y = Y(yv)
        out.append('<line class="grid" x1="%g" y1="%g" x2="%g" y2="%g"/>'
                   % (ml, y, ml + pw, y))
        out.append('<text class="tick" x="%g" y="%g" text-anchor="end">%.0f</text>'
                   % (ml - 4, y + 3, yv))
    # x ticks (ends + middle)
    for dm in (dmin, 0.5 * (dmin + dmax), dmax):
        out.append('<text class="tick" x="%g" y="%g" text-anchor="middle">%.1f</text>'
                   % (X(dm), mt + ph + 14, dm))
    out.append('<text class="axlab" x="%g" y="%g" text-anchor="middle">DM</text>'
               % (ml + pw / 2, height - 2))
    # traces
    for o in sorted(sig.dets):
        d = sig.dets[o]
        col = obs_colors[o]
        pts = " ".join("%g,%g" % (X(dm), Y(d.hits[dm][0])) for dm in d.dms)
        out.append('<polyline class="tr" points="%s" stroke="%s"/>' % (pts, col))
        # mark the peak DM
        out.append('<circle cx="%g" cy="%g" r="2.6" fill="%s"/>'
                   % (X(d.best_dm), Y(d.best_snr), col))
    out.append("</svg>")
    return "".join(out)


def _legend(obs_colors):
    items = []
    for o in sorted(obs_colors):
        items.append('<span class="lg"><i style="background:%s"></i>%s</span>'
                     % (obs_colors[o], html.escape(o)))
    return '<div class="legend">%s</div>' % "".join(items)


def build_html_body(signals, opts):
    obs_all = sorted({o for s in signals for o in s.dets})
    obs_colors = {o: _PALETTE[i % len(_PALETTE)] for i, o in enumerate(obs_all)}
    dm_grid = opts["dm_grid"]
    dd = opts["dm_decimals"]
    esc = html.escape
    parts = []
    parts.append('<h1>CoherentSearch.jl candidate sift</h1>')
    parts.append('<p class="meta">%d files &middot; %d observations &middot; '
                 '%d DMs (%.*f&ndash;%.*f) &middot; within_tol=%.0e '
                 'cross_tol=%.0e &middot; min_snr=%.1f</p>'
                 % (opts["nfiles"], len(obs_all), len(dm_grid), dd, dm_grid[0],
                    dd, dm_grid[-1], opts["within_tol"], opts["cross_tol"],
                    opts["min_snr"]))
    parts.append(_legend(obs_colors))

    # summary table
    parts.append('<div class="twrap"><table><thead><tr>'
                 '<th>#</th><th>class</th><th>freq (Hz)</th><th>P (ms)</th>'
                 '<th>#obs</th><th>sum S/N</th><th>max S/N</th><th>peak DM</th>'
                 '<th>DM MAD (steps)</th><th>&Delta;f/f</th><th>flags</th>'
                 '</tr></thead><tbody>')
    for i, s in enumerate(signals, 1):
        kcls = s.klass.replace("?", "").replace("-", "").lower()
        parts.append('<tr class="k-%s"><td>%d</td><td><b>%s</b></td>'
                     '<td class="mono">%.9f</td><td class="mono">%.3f</td>'
                     '<td>%d</td><td>%.1f</td><td>%.1f</td>'
                     '<td class="mono">%.*f</td><td>%.2f</td>'
                     '<td class="mono">%.1e</td><td class="fl">%s</td></tr>'
                     % (kcls, i, esc(s.klass), s.freq, s.period_ms, s.n_obs,
                        s.sum_snr, s.max_snr, dd, s.dm_med, s.dm_mad_steps,
                        s.freq_frac_spread, esc("; ".join(s.flags))))
    parts.append("</tbody></table></div>")

    # per-candidate S/N-vs-DM cards for the top signals
    ntop = min(opts["html_top"], len(signals))
    parts.append('<h2>S/N vs DM for the top %d signals</h2>' % ntop)
    parts.append('<p class="meta">Each line is one observation; the dot marks '
                 'its peak DM. A real pulsar peaks at a consistent, interior DM '
                 'across observations.</p>')
    parts.append('<div class="cards">')
    for i, s in enumerate(signals[:ntop], 1):
        parts.append('<div class="card">')
        parts.append('<div class="ctitle"><span class="rk">#%d</span> '
                     '%.6f Hz &middot; %.3f ms<br><span class="ck k-%s">%s</span> '
                     '&middot; %d obs &middot; peak DM %.*f</div>'
                     % (i, s.freq, s.period_ms,
                        s.klass.replace("?", "").replace("-", "").lower(),
                        esc(s.klass), s.n_obs, dd, s.dm_med))
        parts.append(_svg_snr_vs_dm(s, dm_grid, obs_colors))
        if s.flags:
            parts.append('<div class="cfl">%s</div>' % esc("; ".join(s.flags)))
        parts.append('</div>')
    parts.append("</div>")
    return "\n".join(parts)


_HTML_STYLE = """
:root {
  color-scheme: light dark;
  --bg: #ffffff; --fg: #1a1c22; --muted: #5b6472; --line: #e4e4ea;
  --panel: #fafafb; --head: #f3f4f7; --grid: #e8e8ee; --axis: #9aa0ad;
  --pulsar: #3a8a2f; --candidate: #3a6ea5; --dim: #7d8695; --flag: #a1455b;
  --pulsar-row: rgba(58,138,47,.15); --candidate-row: rgba(58,110,165,.10);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16181d; --fg: #e6e6ea; --muted: #98a0ad; --line: #2c2f38;
    --panel: #1c1f26; --head: #23262e; --grid: #2c2f38; --axis: #565d6b;
    --pulsar: #6fce5f; --candidate: #6ea3d6; --dim: #8a92a0; --flag: #d98098;
    --pulsar-row: rgba(58,138,47,.22); --candidate-row: rgba(58,110,165,.20);
  }
}
:root[data-theme="light"] {
  --bg: #ffffff; --fg: #1a1c22; --muted: #5b6472; --line: #e4e4ea;
  --panel: #fafafb; --head: #f3f4f7; --grid: #e8e8ee; --axis: #9aa0ad;
  --pulsar: #3a8a2f; --candidate: #3a6ea5; --dim: #7d8695; --flag: #a1455b;
  --pulsar-row: rgba(58,138,47,.15); --candidate-row: rgba(58,110,165,.10);
}
:root[data-theme="dark"] {
  --bg: #16181d; --fg: #e6e6ea; --muted: #98a0ad; --line: #2c2f38;
  --panel: #1c1f26; --head: #23262e; --grid: #2c2f38; --axis: #565d6b;
  --pulsar: #6fce5f; --candidate: #6ea3d6; --dim: #8a92a0; --flag: #d98098;
  --pulsar-row: rgba(58,138,47,.22); --candidate-row: rgba(58,110,165,.20);
}
body { font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif;
       margin: 0 auto; max-width: 1150px; padding: 24px; line-height: 1.4;
       color: var(--fg); background: var(--bg); }
h1 { font-size: 1.5rem; margin: 0 0 4px; text-wrap: balance; }
h2 { font-size: 1.15rem; margin: 28px 0 6px; }
.meta { color: var(--muted); font-size: .85rem; margin: 2px 0 12px; }
.mono, .snr text { font-variant-numeric: tabular-nums;
       font-family: ui-monospace, "SF Mono", Menlo, monospace; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 14px; margin: 8px 0 4px;
          font-size: .8rem; }
.lg { display: inline-flex; align-items: center; gap: 5px; }
.lg i { width: 12px; height: 12px; border-radius: 2px; display: inline-block; }
.twrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: .82rem; margin: 6px 0 4px; }
th, td { padding: 4px 8px; text-align: right; border-bottom: 1px solid var(--line);
         white-space: nowrap; }
th { position: sticky; top: 0; background: var(--head); font-weight: 600; }
th:nth-child(2), td:nth-child(2), .fl { text-align: left; }
.fl { color: var(--flag); font-size: .76rem; white-space: normal; }
tr.k-pulsar td { background: var(--pulsar-row); }
tr.k-candidate td { background: var(--candidate-row); }
tr.k-rfilike td, tr.k-harmonic td, tr.k-weak td { color: var(--dim); }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr));
         gap: 14px; }
.card { border: 1px solid var(--line); border-radius: 8px; padding: 10px 10px 4px;
        background: var(--panel); }
.ctitle { font-size: .82rem; margin-bottom: 4px; }
.rk { font-weight: 700; }
.ck { font-weight: 600; }
.cfl { color: var(--flag); font-size: .74rem; margin: 2px 2px 6px; }
.k-pulsar { color: var(--pulsar); } .k-candidate { color: var(--candidate); }
.k-rfilike, .k-harmonic, .k-weak { color: var(--dim); }
svg.snr { width: 100%; height: auto; display: block; }
.snr .ax { stroke: var(--axis); stroke-width: 1; }
.snr .grid { stroke: var(--grid); stroke-width: 1; }
.snr .tr { fill: none; stroke-width: 1.5; opacity: .9; }
.snr .tick, .snr .axlab { fill: var(--muted); font-size: 9px; }
"""


def write_html_standalone(signals, opts, path):
    body = build_html_body(signals, opts)
    doc = ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
           "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
           "<title>Candidate sift</title>\n<style>%s</style></head>\n<body>\n%s\n"
           "</body></html>\n" % (_HTML_STYLE, body))
    with open(path, "w") as fh:
        fh.write(doc)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def gather_files(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.txt"))))
        else:
            g = sorted(glob.glob(p))
            files.extend(g if g else [p])
    # de-dup, preserve order
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Cross-observation sifting of CoherentSearch.jl candidate files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("paths", nargs="+",
                    help="candidate .txt files, globs, and/or directories")
    ap.add_argument("--regex",
                    default=r"(?P<obs>.+?)_DM(?P<dm>[0-9]+(?:\.[0-9]+)?)",
                    help="regex with named groups 'obs' and 'dm' matched against "
                         "each file's basename")
    ap.add_argument("--within-tol", type=float, default=1e-5,
                    help="fractional freq tolerance for grouping DMs within one obs")
    ap.add_argument("--cross-tol", type=float, default=1e-5,
                    help="fractional freq tolerance for linking detections across "
                         "observations. The default (1e-5) cleanly isolates repeaters "
                         "at a constant barycentric frequency (isolated pulsars). To "
                         "also link a binary whose period differs slightly from day to "
                         "day, raise to ~1e-4 -- but note a clean isolated core can "
                         "then be absorbed into a looser noise-contaminated family, so "
                         "compare both runs.")
    ap.add_argument("--min-snr", type=float, default=0.0,
                    help="ignore candidate rows below this S/N")
    ap.add_argument("--min-dms", type=int, default=2,
                    help="min #DMs a good signal should span (per obs, median)")
    ap.add_argument("--min-obs", type=int, default=1,
                    help="drop signals seen in fewer than this many observations")
    ap.add_argument("--long-period-ms", type=float, default=DEFAULT_LONG_PERIOD_MS,
                    help="periods above this are exempt from the full-DM-span penalty")
    ap.add_argument("--min-score", type=float, default=None,
                    help="only report signals scoring at least this")
    ap.add_argument("--top", type=int, default=60,
                    help="report at most this many signals")
    ap.add_argument("--html-top", type=int, default=24,
                    help="number of S/N-vs-DM panels in the HTML")
    ap.add_argument("--dm-decimals", type=int, default=2)
    ap.add_argument("-o", "--output", default=None, help="text report file (default stdout)")
    ap.add_argument("--html", default=None, help="write a self-contained HTML summary here")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print per-observation detail and score breakdown")
    args = ap.parse_args(argv)

    regex = re.compile(args.regex)
    files = gather_files(args.paths)
    if not files:
        ap.error("no input files found")

    # parse everything; collect the DM grid and per-obs (dm,row) lists
    by_obs = {}          # obs -> list of (dm, Row)
    dm_set = set()
    nfiles = 0
    for path in files:
        info = parse_name(path, regex)
        if info is None:
            print("# skipping (name did not match --regex): %s" % path, file=sys.stderr)
            continue
        obs, dm = info
        rows = read_rows(path, args.min_snr)
        if not rows:
            continue
        nfiles += 1
        dm_set.add(dm)
        by_obs.setdefault(obs, []).extend((dm, r) for r in rows)

    if not by_obs:
        ap.error("no candidates parsed from the input files")

    dm_grid = sorted(dm_set)
    dm_index = {dm: i for i, dm in enumerate(dm_grid)}
    dm_step = _median([b - a for a, b in zip(dm_grid[:-1], dm_grid[1:])]) or 1.0

    # stage 1: detections per observation
    detections = []
    for obs, rows in by_obs.items():
        detections.extend(
            build_detections(obs, rows, args.within_tol, dm_index, dm_grid))

    # stage 2: link across observations
    signals = build_signals(detections, args.cross_tol, dm_step, args.long_period_ms)

    # stage 3: harmonics + scoring
    mark_harmonics(signals, args.cross_tol)
    for s in signals:
        score_signal(s, args.min_dms)

    signals = [s for s in signals if s.n_obs >= args.min_obs]
    if args.min_score is not None:
        signals = [s for s in signals if s.score >= args.min_score]
    signals.sort(key=lambda s: s.score, reverse=True)
    ntotal = len(signals)
    signals = signals[:args.top]

    opts = dict(nfiles=nfiles, dm_grid=dm_grid, dm_decimals=args.dm_decimals,
                within_tol=args.within_tol, cross_tol=args.cross_tol,
                min_snr=args.min_snr, min_dms=args.min_dms, min_obs=args.min_obs,
                verbose=args.verbose, ntotal=ntotal, html_top=args.html_top)

    if args.output:
        with open(args.output, "w") as fh:
            write_text(signals, opts, fh)
        print("# wrote text report to %s" % args.output, file=sys.stderr)
    else:
        write_text(signals, opts, sys.stdout)

    if args.html:
        write_html_standalone(signals, opts, args.html)
        print("# wrote HTML summary to %s" % args.html, file=sys.stderr)


if __name__ == "__main__":
    main()
