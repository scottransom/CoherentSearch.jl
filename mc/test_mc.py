#!/usr/bin/env python3
"""
test_mc.py -- pins for the Monte Carlo's two models and its sampler.

    python3 mc/test_mc.py            # from the repo root (finds julia if present)

These are cheap and they exist because each one guards something that would fail
SILENTLY and in a plausible-looking direction:

  * the width bank is a TRANSCRIPTION of `ladder_boxcar_widths` in src/search.jl,
    so a change there must show up here as a disagreement rather than as a model
    that quietly describes a search we no longer run;
  * `fold_covariance`'s closed form is used for every `dpb >= 1`, which is every
    prepfold fold in the study -- if it drifted from the weight accumulation, the
    prepfold column would move and nothing would complain;
  * the stratified sampler must leave population-weighted numbers UNCHANGED.  A
    broken weight is the one bug in this study that makes every headline number
    wrong while every plot still looks reasonable.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_model as MM
import mc_profiles as MP

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAIL = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'}  {name}{('   ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(name)


# --- 1. the ladder, against the Julia it transcribes ------------------------
def test_ladder():
    jl = """
    using CoherentSearch
    for (nh,md) in ((60,6),(120,12),(240,24))
      ks = decimation_set(nh,md); p = SearchParams(nharms=nh, decimations=ks)
      for k in ks
        nb = 2*fld(nh,k)
        print(nh," ",md," ",k," ",nb," "); println(join(ladder_boxcar_widths(nb,k,p),","))
      end
    end
    """
    try:
        out = subprocess.run(["julia", f"--project={REPO}", "-e", jl],
                             capture_output=True, text=True, timeout=300)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"skip  ladder vs julia ({exc.__class__.__name__})")
        return
    if out.returncode:
        print("skip  ladder vs julia (julia failed):", out.stderr.strip()[-200:])
        return
    bad = []
    for line in out.stdout.splitlines():
        f = line.split()
        nh, md, k, nb = (int(v) for v in f[:4])
        jw = [int(v) for v in f[4].split(",")]
        pw = MM.ladder_boxcar_widths(nb, k, MM.decimation_set(nh, md))
        if jw != pw:
            bad.append((nh, md, k, nb, jw, pw))
    check("ladder_boxcar_widths matches src/search.jl", not bad, str(bad[:2]))


# --- 2. the drizzle covariance ----------------------------------------------
def test_drizzle():
    def numeric(nbins, dpb, nrot=4096):
        L = 1.0 / dpb
        nspan = int(math.ceil(L)) + 1
        n = nrot * nbins
        s = (np.arange(n, dtype=float) * L) % nbins
        b0 = np.floor(s)
        W = np.stack([np.clip(np.minimum(s + L, b0 + j + 1) - np.maximum(s, b0 + j),
                              0, None) / L for j in range(nspan)])
        norm = nbins * (n / (nbins * dpb))
        return np.array([float(np.sum(W[:nspan - d] * W[d:])) / norm
                         for d in range(nspan)])

    # Compared on the CORRECTION, not on C1/C0: the raw ratio is degenerate at a
    # commensurate `dpb`.  At `dpb = 271` exactly, every sample starts at an exact
    # multiple of 1/271 of a bin, so none of them straddles a boundary and the
    # accumulation returns `C1 = 0` -- a property of that float grid, not of a
    # fold, whose period is never commensurate with `dt`.  The closed form is the
    # equidistributed average, which is the physical one.
    worst, where = 0.0, None
    for nb, dpb in ((128, 271.37), (128, 12.913), (128, 4.0907), (64, 1.3907),
                    (64, 1.0409)):
        C = MM.fold_covariance(nb, dpb)
        num = numeric(nb, dpb)
        for w in (1, 2, 4, 9, 19):
            h = MM._boxcar_template(nb, w)
            def corr(c):
                v = c[0] * float(np.dot(h, h))
                for d in range(1, min(len(c), nb)):
                    v += 2.0 * c[d] * float(np.dot(h, np.roll(h, d)))
                return math.sqrt(c[0] / v)
            d = abs(corr(C) - corr(num)) / corr(num)
            if d > worst:
                worst, where = d, (nb, dpb, w)
    check("fold_covariance closed form == weight accumulation", worst < 0.01,
          f"worst {worst:.4f} at {where}")

    # A one-bin boxcar cannot feel bin-to-bin correlation, so its correction must
    # be ~1 no matter how hard the drizzle bites; wide ones must fall.  Getting
    # this backwards (a flat sqrt(DOF_corr)) over-corrects narrow pulses by ~20%.
    c1 = MM.drizzle_boxcar_corr(64, 1.04, 1)
    c19 = MM.drizzle_boxcar_corr(64, 1.04, 19)
    check("drizzle correction is width-dependent", 0.99 < c1 < 1.01 and c19 < 0.87,
          f"w=1 {c1:.3f}, w=19 {c19:.3f}")
    check("drizzle correction vanishes for a well-sampled fold",
          abs(MM.drizzle_boxcar_corr(128, 271.0, 9) - 1.0) < 0.002)


# --- 3. the efficiency model ------------------------------------------------
def test_model():
    prof, _ = MP.make_profile(0.05, MP.VONMISES_W10_W50, nph=1 << 12)
    A = MM.profile_harmonics(prof, 240)
    check("profile_harmonics is unit-normalised",
          abs(2 * float(np.sum(np.abs(MM.profile_harmonics(prof, 1 << 11)) ** 2)) - 1.0) < 1e-6)

    e = {}
    for nh, md in ((60, 6), (120, 6), (120, 12), (240, 24)):
        e[(nh, md)] = MM.ladder_efficiency(A[:nh], nh, md, hmax_data=10 ** 6)[0]
    check("efficiency <= 1 everywhere", all(0 < v <= 1.0 for v in e.values()),
          str({k: round(v, 3) for k, v in e.items()}))
    # Section 4's structural result: at broad duty, nharms 120 with the SAME maxdecim
    # LOSES, because the shallowest rung becomes H=20 instead of H=10 and broad
    # pulses want the shallow rungs.  Doubling maxdecim with it recovers that.
    check("nharms 120 alone loses at 5% duty", e[(120, 6)] < e[(60, 6)] - 0.005,
          f"{e[(120, 6)]:.3f} vs {e[(60, 6)]:.3f}")
    check("120/12 recovers it", e[(120, 12)] >= e[(60, 6)] - 0.002,
          f"{e[(120, 12)]:.3f} vs {e[(60, 6)]:.3f}")

    # ... and at narrow duty the deep folds are what carries it.
    pn, _ = MP.make_profile(0.002, MP.VONMISES_W10_W50, nph=1 << 13)
    An = MM.profile_harmonics(pn, 240)
    a = MM.ladder_efficiency(An[:60], 60, 6, hmax_data=10 ** 6)[0]
    b = MM.ladder_efficiency(An[:120], 120, 12, hmax_data=10 ** 6)[0]
    check("harmonic truncation dominates at 0.2% duty", a < 0.65 and b > a + 0.15,
          f"60/6 {a:.3f} -> 120/12 {b:.3f}")

    # The data Nyquist must bite -- for a pulse whose power actually reaches past
    # it.  Note what this does NOT say: for a BROAD pulse the cap is a free
    # low-pass filter and the efficiency goes UP (0.982 at hmax 20 against 0.967
    # unlimited, at 5% duty), because the rows past the last filled harmonic are
    # zero and `_boxcar_shape!` stops counting their noise.  That is the
    # band-limited normalisation of 1b8fed9 working as designed, and it is why our
    # detection fraction was flat in frequency in run 1 rather than falling.
    hi = MM.ladder_efficiency(An[:60], 60, 6, hmax_data=10 ** 6)[0]
    lo = MM.ladder_efficiency(An[:60], 60, 6, hmax_data=8)[0]
    check("data Nyquist truncates a NARROW pulse", lo < hi - 0.1,
          f"0.2% duty: hmax 8 {lo:.3f} vs unlimited {hi:.3f}")
    broad = MM.ladder_efficiency(A[:60], 60, 6, hmax_data=20)[0]
    check("...and is a free low-pass for a broad one", broad > e[(60, 6)],
          f"5% duty: hmax 20 {broad:.3f} vs unlimited {e[(60, 6)]:.3f}")


# --- 4. the stratified sampler ----------------------------------------------
def test_strat():
    rng = np.random.default_rng(7)
    plain = MP.Population()
    strat = MP.Population(strat=(0.005, 0.5, 0.25))
    a = [plain.draw(rng, 60e-6) for _ in range(30000)]
    b = [strat.draw(rng, 60e-6) for _ in range(30000)]
    da = np.array([d["ducy"] for d in a])
    db = np.array([d["ducy"] for d in b])
    wb = np.array([d["weight"] for d in b])
    raw_a, raw_b = (da < 0.005).mean(), (db < 0.005).mean()
    wtd = wb[db < 0.005].sum() / wb.sum()
    ess = wb.sum() ** 2 / (wb ** 2).sum() / len(wb)
    check("stratification oversamples narrow duty", raw_b > 2 * raw_a,
          f"{100 * raw_a:.2f}% -> {100 * raw_b:.2f}%")
    # The whole point: the weights must put it back.  Binomial sd on 30k draws at
    # p ~ 0.015 is ~0.07%, so 0.3% is several sigma of slack and still tight.
    check("importance weights restore the population",
          abs(wtd - raw_a) < 0.003, f"weighted {100 * wtd:.2f}% vs true {100 * raw_a:.2f}%")
    check("effective sample size is not wrecked", ess > 0.75, f"ESS/N {ess:.3f}")


# --- 5. the 1-in-N selector -------------------------------------------------
def test_one_in():
    import mc_simulate as MS
    for W, n in ((15, 5), (48, 3), (15, 10), (20, 4)):
        cnt = [sum(1 for i in range(6000) if i % W == w and MS.one_in(i, n))
               for w in range(W)]
        ideal = 6000 / W / n
        check(f"one_in spreads 1-in-{n} over {W} workers",
              min(cnt) > 0.6 * ideal and max(cnt) < 1.4 * ideal,
              f"{min(cnt)}..{max(cnt)} vs ideal {ideal:.0f}")
    check("one_in(., 1) is every realisation",
          all(MS.one_in(i, 1) for i in range(100)))
    check("one_in(., 0) is never", not any(MS.one_in(i, 0) for i in range(100)))


if __name__ == "__main__":
    test_ladder()
    test_drizzle()
    test_model()
    test_strat()
    test_one_in()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED: {FAIL}")
        sys.exit(1)
    print("all pins green")
