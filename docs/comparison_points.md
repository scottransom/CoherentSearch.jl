# Comparison points for the paper

What the Monte Carlo actually establishes about the four codes, and the
methodological points that have to travel with each number or it will be
misread. Written 2026-09-11, after the first full red-noise analysis.

**Provenance of every number below unless stated otherwise:** `mc_analyze.py
/data1/mc/run2 /data1/mc/run3 --fap 0.1` on **fitzroy**, 2026-09-11, code at
`222c787`. That is **120,006 realisations / 648,336 injections**: run 2 (white,
76,105) plus the eiger run-3 snapshot (red, 43,901 of a planned ~396,000), with
**237,258 paired injections**. Run 3 is ~11% complete, so every red number here
is provisional in precision but not in sign. Re-check before quoting.

## 0. The one comparison rule

Statistics are not comparable across codes — ours and riptide's are
single-trial, accelsearch's sigma is already trials-corrected, prepfold's is a
chi-squared. **Fix the empirical false-alarm rate, let each code choose its own
threshold, count detections.** A code then wins on two separable things: how low
its threshold can sit (noise-tail shape) and how much signal it puts above it
(filter match).

**Under red noise that rule needs a second clause: the rate must be matched
where the noise is stationary.** Three axes, all measured, all now handled by
`--match` and the `knee`/`band` sections: the red-noise knee, the frequency
band, and (for prepfold) the fold period. A cut matched over a mixture belongs
to none of its parts and reads as a detection fraction of zero — which looks
measured and is not.

## 1. riptide's break under red noise is PREPROCESSING, not its filter

Matched cut at 0.1 false alarms/realisation, per knee bin (Hz):

| | white | 0.1–0.5 | 0.5–2 | 2–6 | 6–15 | 15–50 |
|---|---|---|---|---|---|---|
| `coherent` | 6.75 | 6.75 | 6.70 | 6.70 | 6.70 | 6.75 |
| `accelsearch` | 7.20 | 7.20 | 7.20 | 7.20 | 7.20 | 7.20 |
| `rseek_A` | 7.55 | 7.55 | 18.85 | 59.00 | 140.00 | **250.00** |
| `rseek_B` | 7.65 | 7.75 | 15.25 | 35.50 | 96.50 | **205.00** |

Ours is flat because we search a PRESTO-whitened FFT; accelsearch's is flat
because of its own local power normalisation. riptide detrends in the time
domain with a running median, which is a high-pass at `1/rmed_width` and cannot
whiten above ~0.25 Hz — and no width can without eating the signal (30 Hz would
need a 30 ms window). It is running as its authors intend (`rseek` defaults to
4 s; the example pipeline uses 5 s), so **this is not a harness error and must
not be presented as one.**

Detection fraction, per knee / per (knee × f0 band):

| | white | 0.1–0.5 | 0.5–2 | 2–6 | 6–15 | 15–50 |
|---|---|---|---|---|---|---|
| `coherent` | 76.3 | 76.1 / 84.2 | 75.2 / 82.6 | 69.7 / 77.6 | 61.6 / 69.1 | 48.1 / 54.5 |
| `coh+tier` | 77.6 | 77.5 / 83.9 | 75.6 / 82.4 | 70.4 / 77.4 | 62.1 / 69.1 | 49.0 / 54.4 |
| `rseek_A` | 50.2 | 50.2 / 71.4 | **0.0** / 65.9 | 0.0 / 54.4 | 0.0 / 50.8 | 0.0 / 36.9 |
| `rseek_B` | 71.3 | 69.9 / 80.0 | **0.0** / 71.7 | 0.0 / 61.9 | 0.0 / 46.1 | 0.0 / 32.4 |
| `accelsearch` | 41.6 | 41.3 / 57.7 | 39.4 / 55.1 | 36.4 / 50.7 | 32.1 / 44.9 | 26.3 / 35.3 |
| `accelsearch_red` | 38.5 | 38.4 / 56.5 | 36.8 / 54.0 | 33.9 / 49.4 | 29.7 / 43.3 | 24.4 / 34.1 |
| `coherent_tier` | 51.6 | 51.6 / 55.8 | 49.8 / 54.1 | 44.7 / 49.0 | 36.8 / 40.6 | 22.3 / 26.4 |

**Both columns belong in the paper.** Per knee is what a pipeline tuned to one
observation applies, and it is where riptide goes to zero. Per (knee × band)
gives it back its clean fast folds — and is the fairer statement about its
*search*. Publishing only the first invites "you broke riptide"; only the
second hides the operational cost of an uncalibrated statistic.

**The accelsearch/riptide ordering FLIPS with that choice** — per knee,
accelsearch beats riptide at every knee ≥ 0.5 Hz; per band, riptide beats
accelsearch below ~6 Hz knee (71.4 vs 57.7 at 0.1–0.5). Both statements are
mostly about preprocessing, not about the FFA versus the harmonic sum. Ours is
above both under either matching.

## 2. Chance coincidences: a scoring hazard a matched FAP CANNOT catch

`score()` claims a candidate for an injection if it lies within `tol_bins`
(3.0) of ANY ratio n/m ≤ 8 of f0, and **a claimed candidate is removed from the
false-alarm list.** So where a code floods, its junk enters as *detections* and
no false-alarm matching can see it.

Measured at knee > 15 Hz, f0 5–20 Hz (5-file subset, `--fap 1e-2`):

| | hits within 0.1 bin | median offset | median stat | commonest labels |
|---|---|---|---|---|
| `rseek_A` | **16%** | 8.1 bins | 25.2 | 1/8, 1/7, 1/6 |
| `coherent` | 93% | 0.008 bins | 7.1 | fundamental |

An S/N-8 injection cannot read 25. The symptom was `rseek_A` **detecting 80.5%
in that cell against 65.7% at knee < 0.5 — detection RISING with red noise.**
With the 0.5-bin cut it reads 13.9% and is monotone.

Rejected fraction of recorded hits, per knee (full run):

| | white | 0.1–0.5 | 0.5–2 | 2–6 | 6–15 | 15–50 |
|---|---|---|---|---|---|---|
| `rseek_A` | 0.9 | 2.4 | 18.2 | 26.6 | 39.1 | **55.0** |
| `accelsearch` | 0.3 | 5.7 | 15.8 | 21.2 | 24.5 | 29.0 |
| `accelsearch_red` | 0.4 | 0.5 | 0.5 | 0.5 | 0.7 | **0.8** |
| `coherent` | 0.8 | 0.9 | 1.0 | 1.9 | 3.7 | 7.8 |
| `coherent_rawmeas` | — | 1.4 | 6.7 | 17.6 | 28.5 | 41.0 |

Note `accelsearch_red` (the de-reddened input) is the *cleanest* arm in the
study on this measure — the same mechanism as its flat threshold.

**Run 2 is unaffected, so the white results stand**: above every code's matched
cut the 99th-percentile hit offset is 0.15–0.23 bins and ≤ 0.07% lie beyond 0.5.
Residual contamination after the cut, estimated from the uniform sideband, is
**0.00% of detections for every coherent arm and ~0.01% for accelsearch**.

**The residual caveat to state honestly:** the driver stores only the best hit
per injection (fundamental first, then strongest), so a junk candidate at the
fundamental ratio can have displaced a real one, and rejecting it scores a miss
the code may not have made. The `hits` section's `displaced` row bounds it; it
is 0.0% for every coherent arm, and non-trivial only in cells where riptide
detects almost nothing anyway.

## 3. Red noise costs us 1.0–1.4x in Smin, which is BELOW Lazarus

`coherent`, injected S/N at 50% detection, paired against run 2 (white = 6.84):

| knee (Hz) | S/N(50%) | ratio | 0.1–1 Hz | 1–5 | 5–20 | 20–100 |
|---|---|---|---|---|---|---|
| 0.1–0.5 | 6.86 | 1.00x | 1.02 | 1.00 | 1.00 | 1.00 |
| 0.5–2 | 6.86 | 1.00x | 1.12 | 1.01 | 0.99 | 0.99 |
| 2–6 | 7.16 | 1.05x | **1.37** | 1.11 | 1.01 | 0.99 |
| 6–15 | 7.62 | 1.11x | — | 1.31 | 1.08 | 1.01 |
| 15–50 | 8.68 | **1.27x** | — | — | 1.34 | 1.11 |

Lazarus et al. (2015) measure **1.1–2 at P = 0.1–2 s for PALFA at DM > 150**.
We land at or below the bottom of that range, which is the right side of it:
**their factor includes RFI confusability and we model red noise only.** A
result above their range would have needed explaining.

The paired median (red − white) statistic localises the loss: for `coherent` it
runs −0.06 (knee 0.1–0.5, f0 0.1–1 Hz) to **−3.01** (knee 15–50, f0 0.1–1) and
is **0.00 above 100 Hz at every knee.** Red noise eats the slow end and nothing
else.

## 4. `coherent_rawmeas`: you cannot skip `rednoise`

Measured sigma on the RAW `.fft` — the "can I skip the de-reddening step?" arm.
Per knee: **76.3 → 32.5 → 8.5 → 2.8 → 1.2%**, against `coherent`'s 76.1 → 48.1.
Its matched cut inflates 6.85 → 11.75 to hold the false-alarm rate. This is a
negative result about *our own* default path, which is worth stating plainly.

`coherent_meas` (measured sigma on the whitened file) tracks the analytic
default to within ~1% at every knee, so the analytic/measured half of the 2×2 is
answered: **they agree, and the analytic one is free.**

## 5. Cost, which has to be quoted with sensitivity

fitzroy medians per realisation, full run: `rseek_B` 179.3 s, `coherent_deep`
75.3, `rseek_A` 41.9, **`coherent` 29.1**, `coherent_meas`/`rawmeas` 18.5,
`prepfold` 8.6, `coherent_tier` 7.8, `generate` 7.2, `accelsearch_red` 2.3,
`accelsearch` 2.2. accelsearch is **13x cheaper than us** and that belongs
beside its 41.6% vs our 76.3%.

## 6. Traps — read before quoting any of this

* **Do not quote the pooled tables from a combined run2+run3 load.** They
  average a 76k-white plus 44k-red mixture whose composition is just how far
  run 3 has got. Thresholds are right per cell; the marginal is meaningless.
  Read `knee`, `band` and `paired`.
* **`coherent_tier` is scored outside its own band** in the pooled and S50
  panels — it searches below 5 Hz only, so at high knee it is charged for
  injections it never covered. Restrict it to its band for any figure.
* **S/N(50%) is only reported where the data bracket 50%.** A logistic fit to a
  code whose detection never reaches half across the injected 5.5–11.5 band is
  unconstrained and will still return a number inside that range: accelsearch
  read 10.0 at knee 6–15 and then **7.57** at 15–50, i.e. sensitivity improving
  as the noise worsened, and `coherent_tier` did the same outside its band.
  `_logistic_s50` now requires a measured S/N bin below 50% and one at or above
  it, and blanks the cell otherwise. **A blank there means "this code does not
  reach 50% detection in this cell", which is itself the result** — not missing
  data. With the guard, accelsearch's per-knee S/N(50%) is monotone as it must
  be — **9.12 / 9.23 / 9.49 / 9.94 / 10.54** over knee 0.1–0.5 … 15–50 (9.49 on
  white), at 41% falling to 27% detection — and the cells that blank are
  `accelsearch_red` and `coherent_tier` at knee 15–50 (24% and 22% detection)
  and `rseek_A` everywhere above 0.5 Hz (0%).
* **`prepfold_chi2` has a cut of 0.00 in at least one cell.** `chi2_sigma`
  floors at zero on noise folds, so there every candidate clears the cut and
  the column is degenerate. `prepfold_snr1` is the comparable column.
* **prepfold's cut is drizzle-corrected as of `222c787`**, matching its
  statistic. The pooled cut barely moved (185.52 → 185.46) but per band it is
  **0.880 at 200–400 Hz** and 0.882 above 400, so prepfold's MSP-band columns
  were biased LOW — including in run 2. Anything quoted before 2026-09-11 has
  the old cut.
* **The `sigma_warn` row is not a red-noise diagnostic.** It fires only on
  `coherent_tier`, flat in knee, and identically on white noise: the guard's
  last-chunk sample is a stub in a narrow band. A guard firing on a whole-band
  arm, or a rate climbing with knee, would be the real thing.
* **accelsearch's run-2 repair looks like a hole and is not.** Only 10,384 of
  76,105 realisations have `mcpatch_accel_*` rows, but after the merge
  **76,100 of 76,105 records have candidates** — the patches were a top-up. Do
  not re-panic about this; it cost half an hour on 2026-09-11.
* **prepfold is a reference, not a competitor** — a targeted fold at the known
  period with no trials penalty. It is the ceiling, and at `--fap 1e-2` many of
  its per-cell nulls cannot resolve the rate and come back as lower bounds;
  **`--fap 0.1` resolves them on the data in hand**, which is why this run used
  it.
* **One observation is not a sensitivity measurement.** The PM0063 numbers in
  `CLAUDE.md` (our 12.30 vs rseek's 11.80, and rseek's 12.60 vs our 11.84 in
  `matched`) are single detections whose extreme-value scatter dwarfs the
  differences. This study is what settles relative sensitivity.

## 7. Deliberately absent, and the paper should say so

* **No RFI.** riptide has no zapping stage, so FFT → zap → iFFT would be a step
  *we* impose on it. Revisit only if a referee asks.
* **No whitened-rseek arm.** Per-band matching already makes its fast end a
  fair comparison, and a one-off check showed whitening restores riptide
  exactly: 172 candidates on white, 917 on red, ~170 on the same red data
  whitened, with the 200 Hz pulsar back at 11.1 against 11.2 and its top false
  alarm back at 7.5 from 155.8.
* **`coherent_deep` is parked** — run 2 measured it at 70.6% against 71.0% for
  2.6x the cost.

## 8. Still to do before the paper's red-noise section is final

1. Finish run 3 (~396k realisations planned; 43,901 in hand) and re-run this
   analysis. Signs and orderings should not move; precision and the
   per-(knee × band) prepfold cells will.
2. Restrict `coherent_tier`'s curves to its own band in every figure.
3. Decide whether the headline red-noise figure is per knee, per band, or both
   panels side by side (the `knee` section prints both).
