# What the simulations show

A plain-language draft of the paper's simulation-results sections, followed by
the numbers behind each statement and the caveats that have to travel with them.
Section 1 is meant to be readable by someone who has never run a pulsar search;
everything after it is for us.

**Provenance.** `mc_analyze.py /data1/mc/run2 /data1/mc/run3 --fap 0.1` on
fitzroy, 2026-09-14, against the **completed** run 3 (`report_v4_fap0.1.txt`).
**160,976 realisations / 869,574 injections**: run 2 (white noise, 76,105) plus
run 3 (red noise, 84,871), with **411,078 paired injections**. Band-matched
numbers are the same data with `--match knee,band`
(`report_v4_band_fap0.1.txt`). Analysis code `e0be51d`, i.e. with the three
defects of §9 fixed.

**Eiger was stopped on 2026-09-14 at 84,871 red realisations of a planned
396,000, deliberately.** Going from 43,901 to 82,370 red realisations — a
factor 1.88 — moved every headline cell by **≤ 0.6 points** and reversed no
ordering, while finishing the run would have taken ~21 more days. The numbers
below are the final ones.

**One cell to understand before reading the tables.** `coherent` at knee 15–50
reads 47.9 here against 48.7 in the snapshot, and that is not sampling: its
matched cut stepped 6.70 → 6.75, one grid step. Detection is steep in threshold
at high knee, where the recovered-S/N distribution is compressed against the cut.
**Per-cell matched cuts quantise these fractions at the 0.05 grid, so ±0.5-point
jitter between runs is expected in those cells and is not a change in the
science.**

---

## 1. The findings, in plain sentences

**1. Our search finds more pulsars than the other search codes, at the same
false-alarm rate.** On clean data it recovers 76% of the injected pulsars where
riptide's fast-folding search recovers 50% and PRESTO's `accelsearch` 42%.

**2. The advantage is biggest for narrow pulses.** Pulsars whose pulse covers
less than 1% of a rotation are the hardest case, and there the other two codes
almost vanish: we find 56% of them, riptide 5%, `accelsearch` 2%. For fat pulses
(a sixth of a rotation or more) the gap narrows but does not close.

**3. Most of our margin comes from a better-behaved noise tail, not from a
better filter.** Our statistic's threshold can sit lower for the same number of
false alarms, and that alone accounts for about five sixths of the gap to
riptide. A code wins a search either by putting more signal above the line or by
being able to draw the line lower; ours mostly does the second.

**4. Slow "red" noise is what breaks the fast-folding search, and it is the
cleaning step, not the algorithm.** riptide removes slow drifts by sliding a
median filter along the time series. A 4-second window cannot remove wiggles
faster than about a quarter of a hertz, and no window can without eating the
pulsar. PRESTO removes them in the frequency domain instead, across the whole
spectrum. So when the noise wanders faster than that, riptide's candidate list
fills with junk at slow periods, and one threshold for the whole list buries
everything: its detection rate goes to **zero**. Give it its own threshold in
each frequency band and it recovers to 71% (mild red noise) and 37% (severe) —
against our 84% and 54% under the same treatment. **This is riptide running the
way its authors intend, and the paper must say so.**

**5. Red noise costs us little, and only at low frequencies.** The pulsar
brightness we need for a 50% detection rises by a factor of 1.00, 1.00, 1.05,
1.12, 1.27 as the noise gets worse. Above 100 Hz the loss is exactly zero at
every noise level. Published measurements of the same effect for a real survey
quote a factor of 1.1–2, so we sit at or below the bottom of that range — as we
should, because they include radio interference and we model only red noise.

**6. You cannot skip the de-reddening step.** Running our own search on data
that has not been whitened drops us from 76% to 1% as the noise worsens. This is
a negative result about our own default recipe, and it is worth stating plainly.

**7. Computing the noise level from theory works as well as measuring it, and
it is free.** The two agree to about 1% at every noise level.

**8. We cost less than riptide and much more than `accelsearch`.** Per
simulated observation on the same machine: ours 17 s, riptide 28 s,
`accelsearch` 1.5 s. So we are ~1.6x cheaper than riptide and ~11x dearer than
`accelsearch`, whose 42% has to be read next to our 76%.

**9. Our candidate lists are also cleaner.** Three separate measures: far fewer
of our "detections" are accidents (0.3% against riptide's 16% in the worst
noise); we report the true spin frequency rather than a harmonic of it four
times less often than the others (2.6% against 7–9%); and the pulse width we
report is close to the real one, where riptide's is up to ten times too wide for
the narrowest pulses.

**10. There is still headroom.** `prepfold`, which is *told* the right period
and so is a ceiling rather than a competitor, reaches 93% where we reach 72% on
the same injections. About two thirds of that gap is the price of not knowing
the period in advance — the threshold a blind search has to set.

---

## 2. The setup, in plain terms

Each simulated observation is 1006 s of noise sampled every 60 µs. Into each we
inject six fake pulsars drawn from a realistic population: spin frequency 0.1 to
1000 Hz, pulse widths from the MeerKAT Thousand-Pulsar-Array sample, and a
brightness (injected S/N) between 5.5 and 11.5, which straddles the detection
threshold of every code. Four codes then run on the **identical** data:

| | what it is |
|---|---|
| `coherent` | ours: coherent harmonic summing in the Fourier domain |
| `accelsearch` | PRESTO's standard Fourier-domain search |
| `rseek_A` / `rseek_B` | riptide's fast folding algorithm (`_B` folds 6x deeper) |
| `prepfold` | PRESTO folding at the KNOWN period — a ceiling, not a competitor |

Half the observations also carry red noise: slow wandering of the baseline, with
a "knee" frequency between 0.1 and 50 Hz saying how fast the wander is. Run 3
repeats run 2's injections in run 2's noise with that wander added, so every
red/white comparison is paired on the same pulsar and the same noise.

---

## 3. The one rule that makes codes comparable

A "S/N of 8" means four different things here: ours and riptide's are
single-trial, `accelsearch`'s is already corrected for its trials, `prepfold`'s
is a chi-squared. Comparing at a common nominal threshold is meaningless.

**So: fix the empirical false-alarm rate, let each code pick whatever threshold
delivers it, and count detections.** A code then wins on two separable things —
how low its threshold can sit (the shape of its noise tail) and how much signal
it puts above it (how well its filter matches the pulse).

**Under red noise the rule needs a second clause: the rate must be matched where
the noise is stationary.** A threshold matched over a mixture belongs to none of
its parts and reads as a detection fraction of zero — which looks measured and
is not. Three axes, all found in run 3: the red-noise knee, the frequency band,
and (for `prepfold`) the fold period.

---

## 4. riptide under red noise: preprocessing, not the filter

Matched cut at 0.1 false alarms per realisation, per knee bin (Hz):

| | white | 0.1–0.5 | 0.5–2 | 2–6 | 6–15 | 15–50 |
|---|---|---|---|---|---|---|
| `coherent` | 6.75 | 6.75 | 6.70 | 6.70 | 6.70 | 6.75 |
| `accelsearch` | 7.20 | 7.20 | 7.20 | 7.20 | 7.20 | 7.20 |
| `rseek_A` | 7.55 | 7.55 | 18.85 | 59.50 | 140.00 | **250.00** |
| `rseek_B` | 7.65 | 7.75 | 14.55 | 35.50 | 91.50 | **200.00** |

Ours is flat because we search a PRESTO-whitened FFT; `accelsearch`'s is flat
because of its own local power normalisation. riptide's climbs by a factor of 33
because its time-domain running median is a high-pass at `1/rmed_width` and
cannot whiten above ~0.25 Hz. It is running as its authors intend (`rseek`
defaults to 4 s; the example pipeline uses 5 s), so **this is not a harness
error and must not be presented as one.**

Detection fraction, per knee / per (knee × f0 band):

| | white | 0.1–0.5 | 0.5–2 | 2–6 | 6–15 | 15–50 |
|---|---|---|---|---|---|---|
| `coherent` | 76.3 | 76.2 / 84.2 | 75.2 / 82.7 | 69.8 / 77.6 | 61.3 / 68.7 | 47.9 / 54.5 |
| `coh+tier` | 77.6 | 77.5 / 83.9 | 75.7 / 82.4 | 70.5 / 77.4 | 61.9 / 68.7 | 48.9 / 54.4 |
| `rseek_A` | 50.2 | 50.2 / 71.5 | **0.0** / 65.6 | 0.0 / 55.0 | 0.0 / 50.8 | 0.0 / 36.8 |
| `rseek_B` | 71.3 | 69.2 / 79.6 | **0.1** / 71.1 | 0.0 / 61.6 | 0.1 / 45.6 | 0.0 / 30.9 |
| `accelsearch` | 41.6 | 41.3 / 57.7 | 39.6 / 55.2 | 36.4 / 50.9 | 31.9 / 44.6 | 26.3 / 35.3 |
| `accelsearch_red` | 38.5 | 38.3 / 56.4 | 36.2 / 54.2 | 33.9 / 49.5 | 29.5 / 43.1 | 24.4 / 34.0 |
| `coherent_tier` | 51.6 | 52.0 / 55.6 | 49.8 / 54.1 | 44.6 / 49.1 | 36.7 / 40.6 | 22.5 / 26.5 |

**Both columns belong in the paper.** Per knee is what a pipeline tuned to one
observation applies, and it is where riptide goes to zero. Per (knee × band)
gives it back its clean fast folds and is the fairer statement about its
*search*. Publishing only the first invites "you broke riptide"; only the second
hides the operational cost of an uncalibrated statistic.

**The two columns are not on the same scale and must never be read against each
other.** The band-matched column allows `--fap` false alarms *in each of seven
bands*, so up to 7x the pooled rate. It also has **no white zero point**: run 2
stored no per-band false-alarm tails, so its band-matched row is blank by
construction, not by measurement.

**The `accelsearch`/riptide ordering FLIPS with that choice** — per knee,
`accelsearch` beats riptide at every knee ≥ 0.5 Hz; per band, riptide beats
`accelsearch` below ~6 Hz knee (71.5 vs 57.7 at 0.1–0.5). Both statements are
mostly about preprocessing, not about the FFA versus the harmonic sum. Ours is
above both under either matching.

### Where our margin comes from

The report's counterfactual splits it: scored at *riptide's* threshold we would
detect 38.0% against its 31.0%, so of the 41.1-point gap, **34.1 points are the
threshold** and 7 are the filter. Against `prepfold` the same calculation runs
the other way: at `prepfold`'s threshold we reach 86.2% against its 92.9%, so
**14.1 of the 20.8-point gap is the threshold** a blind search must pay.

---

## 5. Chance coincidences: a scoring hazard a matched FAP cannot catch

The scorer claims a candidate for an injection if it lies within 3 Fourier bins
of **any** simple ratio n/m ≤ 8 of the true frequency, and a claimed candidate is
removed from the false-alarm list. So where a code floods, its junk enters as
*detections* and no false-alarm matching can see it. The symptom in run 3 was
`rseek_A` detecting **more** as the red noise got worse, which is impossible.

`--hit-tol 0.5` scores a hit further than half a bin from its target as a miss.
What remains afterwards, estimated from the uniform sideband, as a percentage of
each code's detections — **band-matched, i.e. measured where riptide actually
detects**:

| | white | 0.1–0.5 | 0.5–2 | 2–6 | 6–15 | 15–50 |
|---|---|---|---|---|---|---|
| `coherent` | — | 0.03 | 0.04 | 0.08 | 0.16 | 0.27 |
| `accelsearch` | — | 0.29 | 0.35 | 0.40 | 0.50 | 0.66 |
| `accelsearch_red` | — | 0.05 | 0.07 | 0.05 | 0.07 | 0.09 |
| `rseek_A` | — | 0.31 | 2.44 | 4.34 | 7.22 | **16.01** |
| `rseek_B` | — | 0.07 | 0.10 | 0.32 | 0.72 | 1.30 |

(White is blank because this matching has no white cut at all — run 2 stored no
per-band tails, so no detections are scored there. The white result is the
closing paragraph of this section instead: above every code's matched cut the
99th-percentile hit offset is 0.15–0.23 bins and the residual is 0.00% for every
coherent arm.)

**This is the one number that qualifies finding 4's rehabilitation of riptide.**
Its band-matched 36.8% at the worst knee still carries ~16% chance coincidences;
ours carries 0.27%. Note also `accelsearch_red` — the de-reddened input — is the
cleanest arm in the study, the same mechanism as its flat threshold.

**The residual caveat to state honestly:** the driver stores only the *best* hit
per injection, so a junk candidate at the fundamental ratio can have displaced a
real one, and rejecting it scores a miss the code may not have made. The
`displaced` row bounds that, and for `rseek_A` it reaches **80% of detections**
at the worst knee. Storing every hit rather than the best is the only thing that
would close this, and it needs a re-run (§10).

**Run 2 is unaffected, so the white results stand**: above every code's matched
cut the 99th-percentile hit offset is 0.15–0.23 bins, ≤ 0.07% lie beyond 0.5,
and the residual is 0.00% for every coherent arm.

---

## 6. Red noise costs us 1.0–1.3x in Smin, below the published figure

`coherent`, injected S/N at 50% detection, paired against run 2 (white = 6.84):

| knee (Hz) | S/N(50%) | ratio | 0.1–1 Hz | 1–5 | 5–20 | 20–100 |
|---|---|---|---|---|---|---|
| 0.1–0.5 | 6.85 | 1.00x | 1.02 | 1.00 | 1.00 | 1.00 |
| 0.5–2 | 6.87 | 1.00x | 1.12 | 1.01 | 0.99 | 0.99 |
| 2–6 | 7.15 | 1.05x | **1.37** | 1.11 | 1.01 | 0.99 |
| 6–15 | 7.63 | 1.12x | — | 1.32 | 1.08 | 1.01 |
| 15–50 | 8.68 | **1.27x** | — | — | 1.34 | 1.12 |

Lazarus et al. (2015) measure **1.1–2 at P = 0.1–2 s for PALFA at DM > 150**.
Quote their high-DM figure: their DM dependence is RFI confusability and we
model red noise only, so landing at or below the bottom of their range is the
right side of it. A result above it would have needed explaining.

The paired median (red − white) statistic localises the loss: for `coherent` it
runs −0.06 (knee 0.1–0.5, f0 0.1–1 Hz) to **−3.12** (knee 15–50, f0 0.1–1) and
is **0.00 above 100 Hz at every knee.** Red noise eats the slow end and nothing
else.

---

## 7. The two sigma questions, answered

**`coherent_rawmeas` — can I skip `rednoise`?** No. Measured sigma on the raw
`.fft`, per knee: **77.3 → 32.4 → 8.5 → 3.0 → 1.0%**, against `coherent`'s
76.2 → 47.9. Its matched cut inflates 6.80 → 11.80 just to hold the false-alarm
rate. A negative result about our own default path, worth stating plainly.

**`coherent_meas` — should I measure sigma or compute it?** It makes no
difference: measured sigma on the whitened file tracks the analytic default to
within ~1% at every knee (76.5 / 74.4 / 69.2 / 60.9 / 47.7 against
76.2 / 75.2 / 69.8 / 61.3 / 47.9). The analytic one is free, so use it.

---

## 8. Cost, which has to be quoted with sensitivity

**Per host, because run 2 ran on fitzroy and run 3 on eiger** — the pooled
median mixes two machines and a ratio taken from it would be part hardware:

| median s per realisation | `rseek_B` | `coherent_deep` | `rseek_A` | `coherent` | `prepfold` | `coherent_tier` | `accelsearch` |
|---|---|---|---|---|---|---|---|
| eiger (run 3) | 121.3 | — | 27.5 | **16.9** | 5.5 | 5.0 | 1.5 |
| fitzroy (run 2) | 179.8 | 75.3 | 42.3 | **29.4** | 8.8 | 8.0 | 2.3 |

Both hosts agree on the ratios that matter: we are **1.4–1.6x cheaper than
`rseek_A`** and **11–13x more expensive than `accelsearch`**. That last number
belongs next to `accelsearch`'s 41.6% against our 76.3%.

---

## 9. Three defects fixed on 2026-09-14 — anything quoted before then

All three had the same shape: a number that looked measured and was not.

* **`prepfold` read 0.0% at 100–200 Hz in every cell.** That band is the gap
  between the two injected populations, so its null folds land there at **0.089
  per realisation** — under the 0.1 the rate asks for — and the cut came back
  `inf`, which the detection counter scored as "every row a miss". It now
  returns `nan` and the cell prints blank with the fold rate quoted. It biased
  the ceiling column low, i.e. in the direction that flattered us — and not only
  in the band table: the pooled `by f0` row for 100–200 Hz read **45.3 / 47.7%**
  before the fix and reads **94.5 / 99.6%** after, a ~50-point correction to the
  reference column in that band.
* **Four "the subset every arm ran" tables printed rows of `nan`, and the whole
  cost table silently vanished.** `coherent_deep` ran only in run 2 and
  `coherent_meas`/`rawmeas` only in run 3, so no injection was seen by every
  arm. The blocks now name the non-overlapping pair, and the cost table scores
  each subset arm against the always-run arms instead of disappearing.
* **Under `--match knee,band` every white realisation scored zero.** Run 2 has
  no per-band tails, so white rows fell in cells with no cut and `Cut.missing`
  handed them `inf`. A cell we cannot cut is a hole in the measurement and now
  prints as one.

`test_mc.py` pins all three (`test_missing_cut`).

---

## 10. Traps — read before quoting any of this

* **Do not quote the pooled tables from a combined run2+run3 load.** They
  average a 76k-white plus 85k-red mixture whose composition is just how far
  run 3 got. Thresholds are right per cell; the marginal is meaningless. Read
  `knee`, `band` and `paired`. (The by-duty and by-f0 tables are pooled this
  way: read their *shape*, not their level.)
* **Band-matched and per-knee numbers are not comparable** — see §4.
* **`coherent_tier` is scored outside its own band** in the pooled and S50
  panels — it searches below 5 Hz only, so at high knee it is charged for
  injections it never covered. Restrict it to its band for any figure.
* **S/N(50%) is only reported where the data bracket 50%.** A logistic fit to a
  code that never reaches half across the injected 5.5–11.5 band is
  unconstrained and will still return a number inside it. A blank means "this
  code does not reach 50% detection in this cell", which is itself the result.
* **`prepfold_chi2` has a cut of 0.00 in at least one cell.** `chi2_sigma`
  floors at zero on noise folds, so there every candidate clears the cut and the
  column is degenerate. `prepfold_snr1` is the comparable column.
* **`prepfold`'s cut is drizzle-corrected as of `222c787`.** Per band it is
  0.880 at 200–400 Hz, so `prepfold`'s MSP-band columns were biased low before
  that — including in run 2.
* **The `sigma_warn` row is not a red-noise diagnostic.** It fires only on
  `coherent_tier`, flat in knee, and identically on white noise: the guard's
  last-chunk sample is a stub in a narrow band.
* **`accelsearch`'s run-2 repair looks like a hole and is not.** Only 10,384 of
  76,105 realisations have `mcpatch_accel_*` rows, but after the merge 76,100 of
  76,105 have candidates — the patches were a top-up. It cost half an hour on
  2026-09-11.
* **`prepfold` is a reference, not a competitor** — a targeted fold at the known
  period with no trials penalty. At `--fap 1e-2` many of its per-cell nulls
  cannot resolve the rate; `--fap 0.1` resolves them on the data in hand, which
  is why this run uses it.
* **One observation is not a sensitivity measurement.** The PM0063 numbers in
  `CLAUDE.md` are single detections whose extreme-value scatter dwarfs the
  differences. This study is what settles relative sensitivity.

---

## 11. Deliberately absent, and the paper should say so

* **No RFI.** riptide has no zapping stage, so FFT → zap → iFFT would be a step
  *we* impose on it. Revisit only if a referee asks.
* **No whitened-rseek arm.** Per-band matching already makes its fast end a fair
  comparison, and a one-off check showed whitening restores riptide exactly: 172
  candidates on white, 917 on red, ~170 on the same red data whitened, with the
  200 Hz pulsar back at 11.1 against 11.2 and its top false alarm back at 7.5
  from 155.8. **Note this is n = 1**, and §5 is the reason it might be worth
  more than that.
* **`coherent_deep` is parked** — run 2 measured it at 70.6% against 71.0% for
  2.6x the cost.

---

## 12. Still open, in value order

1. ~~Re-run the analysis on the completed run 3~~ — **done 2026-09-14**; every
   number above comes from `report_v4_*`. Signs and orderings did not move.
2–4. **Run 4 — implemented 2026-09-14, waiting to be launched.** One pass over
   run 2's white indices closes all three of the gaps that no re-analysis can:
   the missing **per-band tails** on the white side (§4's absent zero point); the
   **displaced-hit** bound of §5, now that `--hits-per-inj` records the
   runners-up and the analysis takes the best candidate that is close enough;
   and **`rseek_W`**, riptide on a PRESTO-whitened time series with its own
   running median switched off, which turns §11's n = 1 check into a measurement.
   Measured on eiger at 15 workers: **92.1 s a realisation** for
   `rseek,rseekw,coherent`, and ~107 s with `accel` and `rseek_B` added so that
   every cell of the band-matched white row is filled — **~12,100 a day, so ~24k
   realisations (~145k injections) over two days.** Load costs 1.70x against the
   54.3 s solo figure, and it is contention rather than throttling: the clock
   *rises* 1572 → 1975 MHz under load. See `mc/README.md`.
5. **Restrict `coherent_tier`'s curves to its own band in every figure.**
6. **Decide the headline red-noise figure**: per knee, per band, or both panels
   side by side.
