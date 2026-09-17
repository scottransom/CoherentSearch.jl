# What the simulations show

A plain-language draft of the paper's simulation-results sections, followed by
the numbers behind each statement and the caveats that have to travel with them.
Section 1 is meant to be readable by someone who has never run a pulsar search;
everything after it is for us.

**Provenance.** `mc_analyze.py /data1/mc/run2 /data1/mc/run3 --fap 0.1` on
fitzroy, against the **completed** run 3, **re-run 2026-09-16 with run 4's
patches merged** (`report_v4_fap0.1.txt`, analysis at `79e08e5`; the white
band-matched cells of §4 and the cost lines of §8 come from a `--sections
knee` / `--sections cost` re-run with the two fixes below).
**160,976 realizations / 869,574 injections**: run 2 (white noise, 76,105) plus
run 3 (red noise, 84,871), with **411,078 paired injections**. Band-matched
numbers are the same data with `--match knee,band`
(`report_v4_band_fap0.1.txt`).

**Run 4** (eiger, 2026-09-14 → 16) re-ran `accelsearch`, `rseek_A` (and
`rseek_B` one in ten), the new `rseek_W`, and `coherent` on **22,368 of run 2's
76,105 white realizations**, as patch records in run 2's directory. It gives the
white side per-band false-alarm tails and runner-up hits. It regenerates
run 2's noise from the index, so the re-run arms should reproduce run 2, and
**no headline number below moved** (`rseek_B` white 71.3 → 71.4, on 99,720
injections instead of 87,564). **It covered white noise only**, which limits
what it closes (§5, §11).

**Checked record by record** (2026-09-16, fitzroy; hits and the head of the
false-alarm tail, per re-run arm, against run 2's record, with run 2's
`accelsearch` repair patches applied):

| arm | identical | differ |
|---|---|---|
| `rseek_A` | 22,368 | 0 |
| `rseek_B` | 436 | 0 (1,777 ran only in run 4) |
| `accelsearch` | 22,367 | 1 (tail) |
| `coherent` | 22,362 | 6 (a candidate count ±1, a tail, one hit) |
| `coherent_tier` | 22,354 | 14 (e.g. one S/N 7.12 → 7.11) |
| **`accelsearch_red`** | **13,802** | **8,566** |

- **Every search arm is reproducible across the two hosts except
  `accelsearch_red`.** The handful of `coherent` differences are at the
  last-digit level, consistent with host-specific FFTW plans.
- **`accelsearch_red` differs on 38% of records**, in hits and in the tail,
  and still on 31% when compared to 2 decimals. So this is not rounding. It is
  **not the PRESTO build date**: fitzroy's PRESTO was rebuilt on 2026-09-10 for
  the accelsearch repair, and repaired and unrepaired records differ at the
  same rate. Plain `accelsearch` on the raw `.fft` is 100% reproducible, and so
  is `coherent`, which reads the very same `_red.fft`. **Cause unidentified.**
  The leading hypothesis, untested, is SMR's: eiger's PRESTO was built with
  `-Dc_args=-march=native` and fitzroy's probably was not, which is enough for a
  small floating-point inconsistency in `rednoise`'s output. `coherent` (S/N to
  2 decimals, on a fixed trial grid) would not see a difference that small, and
  something in `accelsearch`'s candidate pipeline amplifies it.
  Its white detection fraction is unchanged (38.5 in both reports), so no
  number here moves. But it is the one arm whose per-record output depends on
  where it ran. The merge applies fitzroy's repair patch last, so the analysis
  uses fitzroy's `accelsearch_red` wherever one exists and eiger's elsewhere.

**Two analysis defects found on 2026-09-16, both with the same shape as §9's**
(fixed in the commit that added this paragraph, pinned in `test_mc.py`):

* **The white cell of the band-matched knee row was the per-knee number,
  relabelled.** `sec_knee` builds a `CutBook` per knee bin, and a book holding
  no red-noise records downgraded itself to pooled matching. Before run 4
  the white bin had no band tails, so the cell printed blank. After run 4 it
  printed the POOLED white fraction (76.3 for `coherent`) under the
  band-matched heading. The real value is in §4.
* **The per-host cost lines filed run 4's eiger timings under fitzroy**,
  because a patch's timings were merged into its run-2 parent and attributed
  to the parent's host. `rseek_W`, which ran only on eiger, printed on the
  fitzroy line. §8 is from the fixed re-run.

**Eiger was stopped on 2026-09-14 at 84,871 red realizations of a planned
396,000, deliberately.** Going from 43,901 to 82,370 red realizations — a
factor 1.88 — moved every headline cell by **≤ 0.6 points** and reversed no
ordering, while finishing the run would have taken ~21 more days. The numbers
below are the final ones.

**One cell to understand before reading the tables.** `coherent` at knee 15–50
reads 47.9 here against 48.7 in the snapshot, and that is not sampling: its
matched cut stepped 6.70 → 6.75, one grid step. Detection is steep in threshold
at high knee, where the recovered-S/N distribution is compressed against the cut.
**Per-cell matched cuts quantize these fractions at the 0.05 grid, so ±0.5-point
jitter between runs is expected in those cells and is not a change in the
science.**

---

## 1. The findings, in plain sentences

**1. Our search finds more pulsars than the other search codes, at the same
false-alarm rate.** On clean data it recovers 76% of the injected pulsars where
riptide's fast-folding search recovers 50% and PRESTO's `accelsearch` 42%.

**2. The advantage is biggest for narrow pulses — against the riptide
configuration matched to our band.** Pulsars whose pulse covers less than 1% of
a rotation are the hardest case, and there the other two codes almost vanish: at
a duty cycle of 0.5–1% we find 56%, `rseek_A` 5%, `accelsearch` 2%. For fat
pulses (a sixth of a rotation or more) the gap narrows but does not close.

**SMR was right to flag this: those numbers are `rseek_A` only, and `rseek_B` is
what riptide's authors would point at a narrow pulse.** `rseek_B` folds six times
deeper for ~6x our runtime, and a deeper fold is exactly what a narrow pulse
needs. There is no `rseek_B` entry in that table because the per-duty detection
table covers only the arms that ran on every realization and `rseek_B` ran on one
in ten. The statistic that does exist for it is the logistic fit of injected S/N
at 50% detection, and below 1% duty that reads **8.96 for `rseek_B` against our
8.25** — about 0.7 in S/N, not a factor of ten in detection fraction.
**So the paper must never quote 56 against 5 without naming `rseek_A` and saying
what it is matched to.** A `rseek_B` per-duty row is obtainable — `mc_analyze.py
--no-common` includes the subset arms — but it is a 36-minute, 13 GB pass over
the records on fitzroy, so it has not been run.

**3. Most of our sensitivity benefit comes from a better-behaved noise
tail, not from a better filter.** Our statistic's threshold can sit lower,
and therefore be more sensitive, for the same number of false alarms, and
that alone accounts for about five sixths of the gap to riptide. A code
wins a search either by putting more signal above the threshold or by being
able to move the threshold lower; ours mostly does the second.

**4. Slow "red" noise is what breaks the fast-folding search, and it is
the cleaning step, not the algorithm.** riptide removes slow drifts by
sliding a median filter along the time series. A 4-second window cannot
remove wiggles faster than about a quarter of a hertz, and no window can
without eating the pulsar, eventually. PRESTO removes them in the
frequency domain instead, across the whole spectrum. So when the noise
wanders faster than that, riptide's candidate list fills with junk at slow
periods, and one threshold for the whole list buries everything: its
detection rate effectively goes to **zero** since it is swamped by false
positives. Give it its own threshold in each frequency band and it
recovers to 71% (mild red noise) and 37% (severe) — against our 84% and
54% under the same treatment. (On clean data, under that same per-band
treatment, it is 72% against our 84%.) **This is riptide running the way
its authors intend, and the paper must say so.**

**5. Red noise costs us little, and only at low frequencies.** The pulsar
brightness we need for a 50% detection rises by a factor of 1.00, 1.00,
1.05, 1.12, 1.27 as the knee frequency rises through the bins 0.1–0.5, 0.5–2,
2–6, 6–15 and 15–50 Hz. Above 100 Hz the loss is exactly zero at every noise
level. Published measurements of the same effect for a real survey quote a
factor of 1.1–2, so we sit at or below the bottom of that range — as we
should, because they include radio interference and we model only red
noise.

**6. You cannot skip the de-reddening step.** Running our own search on
data that has not been whitened drops us from 76% to 1% as the noise
worsens. This is a negative result about our own default recipe, and it
must be stated plainly. We should note that it doesn't mean that you need
to use exactly PRESTO's `rednoise` whitening, but the FFT must be normalized.

**7. Computing the noise level from theory (given the normalized input
Fourier amplitudes) works as well as measuring it, and it is free.** The
two agree to about 1% at every noise level.

**8. We cost less than riptide and much more than `accelsearch`.** Per
simulated observation on the same machine: ours 17 s, riptide 28 s,
`accelsearch` 1.5 s. So we are ~1.6x cheaper than riptide and ~11x more expensive than
`accelsearch`, whose 42% has to be read next to our 76%.

**9. Our candidate lists are also cleaner and more accurate.** Three
separate measures: far fewer of our "detections" are accidents (0.3%
against riptide's 16% in the worst noise); we report the true spin
frequency rather than a harmonic of it four times less often than the
others (2.6% against 7–9%); and the pulse width we report is close to the
real one, where riptide's is up to ten times too wide for the narrowest
pulses.

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

Half the observations also carry red noise: slow wandering of the
baseline, with a "knee" frequency between 0.1 and 50 Hz saying how fast
the wander is (i.e. where the rednoise level drops to match the whitenoise
level). Run 3 repeats run 2's injections in run 2's noise with that wander
added, so every red/white comparison is paired on the same pulsar and the
same noise.

---

## 3. The one rule that makes codes comparable

A "S/N of 8" does not mean the same thing in any two of these codes.
**SMR's suspicion was right and the sentence that used to stand here was wrong:
we do not use `accelsearch`'s trials-corrected sigma.** `parse_accel` reads the
`ACCEL_0` file through `presto.sifting.candlist_from_candfile`, which recomputes
`candidate_sigma(opt_ipow, 1, 1)` from the optimized harmonic powers and hands
back a **single-trial** sigma; the trials-corrected value printed in the file is
never read. So three of the four statistics — ours, riptide's and
`accelsearch`'s — are single-trial, and `prepfold`'s is a chi-squared test
converted to an equivalent Gaussian significance. The code was always right here
and only the prose was wrong.

Even so, comparing at a common nominal threshold is meaningless. Three
single-trial sigmas are not interchangeable when the codes search different
numbers of trials, over differently-shaped noise, with statistics whose tails
differ. That is what the empirical matching below exists to fix.

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

**Everything in this section is binned on two axes, so define them once.**

- **The knee frequency** is where the injected red noise crosses the white floor:
  below it the red noise dominates, above it the spectrum is flat. A knee of
  15–50 Hz is severe, 0.1–0.5 Hz mild, and `white` is a realization with no red
  noise at all. The bins are 0.1–0.5, 0.5–2, 2–6, 6–15 and 15–50 Hz.
- **The f0 band** is the *injected pulsar's own spin frequency*, in seven bins:
  0–1, 1–5, 5–20, 20–100, 100–200, 200–400 and above 400 Hz. It is **not** a
  range the measurement is averaged over, and it is **not** a restriction of the
  search — every code always searches its whole band. It says only which
  injections a cell counts, and which of a code's false alarms set that cell's
  threshold.

**Every threshold quoted below is chosen so that all codes make the same number
of false alarms, not a threshold anyone would pick a priori.** Matching per knee
gives one threshold per code per knee bin, from that code's false alarms in the
realizations with that knee. Matching per (knee × f0 band) gives one threshold
per cell: it comes from the false alarms whose *candidate* frequency fell in that
band, and it is applied to the injections whose *f0* fell in the same band.

**The two matchings are not on the same scale and must never be read against
each other.** Band matching allows `--fap` false alarms in each of seven bands,
so up to 7x the rate the per-knee matching allows.

Matched cut at 0.1 false alarms per realization, per knee bin (Hz) — i.e. the
S/N threshold each code has to be given for all of them to produce the same
false-alarm rate:

| | white | 0.1–0.5 | 0.5–2 | 2–6 | 6–15 | 15–50 |
|---|---|---|---|---|---|---|
| `coherent` | 6.75 | 6.75 | 6.70 | 6.70 | 6.70 | 6.75 |
| `accelsearch` | 7.20 | 7.20 | 7.20 | 7.20 | 7.20 | 7.20 |
| `rseek_A` | 7.55 | 7.55 | 18.85 | 59.50 | 140.00 | **250.00** |
| `rseek_B` | 7.65 | 7.75 | 14.55 | 35.50 | 91.50 | **200.00** |

Ours is flat because we search a PRESTO-whitened FFT; `accelsearch`'s is
flat because of its own local power normalization. riptide's climbs by a
factor of 33 because its time-domain running median is a high-pass at
`1/rmed_width` and cannot whiten above ~0.25 Hz. It is running as its
authors intend (`rseek` defaults to 4 s; the example pipeline uses 5 s),
so **this is not a testing or simulation error and must not be presented
as one.**

Detection fraction, per knee / per (knee × f0 band):

| | white | 0.1–0.5 | 0.5–2 | 2–6 | 6–15 | 15–50 |
|---|---|---|---|---|---|---|
| `coherent` | 76.3 / **84.2** | 76.2 / 84.2 | 75.2 / 82.7 | 69.8 / 77.6 | 61.3 / 68.7 | 47.9 / 54.5 |
| `coh+tier` | 77.6 / 83.9 | 77.5 / 83.9 | 75.7 / 82.4 | 70.5 / 77.4 | 61.9 / 68.7 | 48.9 / 54.4 |
| `rseek_A` | 50.2 / **72.5** | 50.2 / 71.5 | **0.0** / 65.6 | 0.0 / 55.0 | 0.0 / 50.8 | 0.0 / 36.8 |
| `rseek_B` | 71.4 / **80.7** | 69.2 / 79.6 | **0.1** / 71.1 | 0.0 / 61.6 | 0.1 / 45.6 | 0.0 / 30.9 |
| `accelsearch` | 41.6 / 57.9 | 41.3 / 57.7 | 39.6 / 55.2 | 36.4 / 50.9 | 31.9 / 44.6 | 26.3 / 35.3 |
| `accelsearch_red` | 38.5 / 56.6 | 38.3 / 56.4 | 36.2 / 54.2 | 33.9 / 49.5 | 29.5 / 43.1 | 24.4 / 34.0 |
| `coherent_tier` | 51.6 / 55.6 | 52.0 / 55.6 |
| `rseek_W` | 49.0 / 71.3 | — | — | — | — | — | 49.8 / 54.1 | 44.6 / 49.1 | 36.7 / 40.6 | 22.5 / 26.5 |

**Both columns belong in the paper.** Per knee is what a pipeline tuned to
one observation applies, and it is where riptide goes to zero. Per (knee ×
band) gives it back its clean fast folds and is the fairer statement about
its *search*. Publishing only the first invites "you are misusing
riptide"; only the second hides the operational cost of an uncalibrated
statistic.

(The reminder that these two columns are on different scales is now at the top
of this section, where it is read before the numbers.)

**The band-matched row now has a white zero point** (run 4; the white cell of
the report printed before 2026-09-16 is wrong, see the provenance note). It
agrees with the mildest knee to about a point for every code (`coherent` 84.2
against 84.2, `rseek_A` 72.5 against 71.5, `rseek_B` 80.7 against 79.6), as a
zero point should. Two things follow:

- **Per-band matching narrows the white gap to riptide, but does not close it.** 
  Against `rseek_A` it goes from 26.1 points to **11.7**, and against the
  deep `rseek_B` from 4.9 to **3.5**. Every code gains from the 7x looser rate,
  and riptide's matched configuration gains most (+22.3 against our +7.9). The
  per-band table below shows where. This is **not** its slow folds on white
  noise: there its three slowest bands sit at its reporting floor, and its
  highest per-band cuts are in the fast bands (7.1–7.3).
- **Red-noise degradation can now be read along the band-matched row too.**
  `coherent` 84.2 → 54.5 across the knees; `rseek_A` 72.5 → 36.8; `rseek_B`
  80.7 → 30.9. riptide's deep configuration loses more than we do and falls
  below us at every knee, and below its own matched configuration past 6 Hz.

**The `accelsearch`/riptide ordering FLIPS with that choice** — per knee,
`accelsearch` beats riptide at every knee ≥ 0.5 Hz; per band, riptide beats
`accelsearch` below ~6 Hz knee (71.5 vs 57.7 at 0.1–0.5). Both statements are
mostly about preprocessing, not about the FFA versus the harmonic sum. Ours is
above both under either matching.

### White noise, per frequency band: the zero point run 4 added

Detection %, every cut matched inside its band at 0.1 false alarms per
realization. That allows up to 7x the pooled rate over the whole list, so these
numbers are NOT comparable with the per-knee white column. The cuts come from
the 22,368 run-4 realizations that store band tails, applied to all 76,105:

| f0 band (Hz) | 0–1 | 1–5 | 5–20 | 20–100 | 100–200 | 200–400 | 400+ |
|---|---|---|---|---|---|---|---|
| injections | 72,963 | 100,681 | 84,505 | 49,893 | 7,814 | 82,748 | 12,474 |
| `coherent` | 85.9 | **86.0** | **84.6** | **80.8** | **83.1** | **83.9** | 84.6 |
| `rseek_B` | **87.0** | 84.5 | 82.8 | 75.2 | 69.1 | 77.4 | **86.2** |
| `rseek_A` | 58.4 | 66.6 | 73.4 | 77.3 | 69.5 | 76.9 | 85.3 |
| `rseek_W` | 56.3 | 65.8 | 72.5 | 76.6 | 67.4 | 75.6 | 83.6 |
| `accelsearch` | 32.6 | 48.7 | 60.3 | 60.2 | 70.4 | 68.3 | 63.3 |
| `prepfold_snr1` (ceiling) | 99.3 | 99.8 | 99.9 | 100.0 | — | 100.0 | 100.0 |

(Bootstrap errors: ±0.1–0.4 for most cells; `rseek_B` ±0.3–1.4, since it ran on
one realization in ten; 100–200 Hz ±0.4–1.4.)

- **This is the honest white-noise statement of where riptide stands, and it is
  closer than the per-knee 76 vs 50 suggests.** Given its own cut in each
  band, riptide's deep configuration **beats us below 1 Hz** (87.0 vs 85.9) and
  **edges us above 400 Hz** (86.2 vs 84.6, about 1.7σ). The matched
  configuration `rseek_A` is within a point of us above 400 Hz (85.3 vs 84.6).
  We lead everywhere between, by up to **14 points at 100–200 Hz** against
  `rseek_B` and by 27 points below 1 Hz against `rseek_A`.
- **Per-band matching loosens every code's cut, and riptide gains more from
  it.** `coherent` goes from 6.75 pooled to 5.75–6.56 per band (76.3 → 84.2
  overall), and `rseek_A` from 7.55 to 6.1–7.3 (50.2 → 72.5). Below 20 Hz,
  `rseek_A`'s 6.1s are its reporting floor: it made too few false alarms there
  to set a cut, so its three slowest cells are upper bounds. That is §4's red-noise
  argument, already present at a smaller scale without any red noise.
- **Cost still has to sit next to this.** `rseek_B` is the configuration that
  matches us per band, and it costs ~6–7x our runtime (§8).
- **`prepfold` 100–200 Hz is still blank.** Run 4 re-used run 2's seeds, so it
  regenerated the same too-thin null folds, as predicted.

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
| `coherent` | 0.02 | 0.03 | 0.04 | 0.08 | 0.16 | 0.27 |
| `accelsearch` | 0.04 | 0.29 | 0.35 | 0.40 | 0.50 | 0.66 |
| `accelsearch_red` | 0.05 | 0.05 | 0.07 | 0.05 | 0.07 | 0.09 |
| `rseek_A` | 0.15 | 0.31 | 2.44 | 4.34 | 7.22 | **16.01** |
| `rseek_B` | 0.07 | 0.07 | 0.10 | 0.32 | 0.72 | 1.30 |
| `rseek_W` | 0.16 | — | — | — | — | — |

(The white column is new with run 4, whose band tails give white a band-matched
cut; before it, the column was blank. The red columns are unchanged.)

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

**Run 4 stores the runners-up, but only on white noise, so this caveat still
stands for every red column.** On white the bound is small anyway: displaced
0.8% of `rseek_A`'s detections and 0.1% of ours, band-matched, and run 4's
records (29% of the white set) can now *rescue* a displaced real hit rather
than only bound it. Closing the red side needs runners-up on run 3's indices,
i.e. a red top-up with `--hits-per-inj`.

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
Quote their high-DM figure: their DM dependence is likely confusion with RFI and we
model red noise only, so landing at or below the bottom of their range is the
right side of it. A result above it would have needed explaining.

The paired median (red − white) statistic localizes the loss: for `coherent` it
runs −0.06 (knee 0.1–0.5, f0 0.1–1 Hz) to **−3.12** (knee 15–50, f0 0.1–1) and
is **0.00 above 100 Hz at every knee.** Red noise eats the slow end and nothing
else.

---

## 7. The two sigma questions, answered

**`coherent_rawmeas` — can I skip `rednoise`?** No.

**What the arm is** (answering SMR's question: it is the first of the two
readings, and the `--sigma measured` is deliberate). It runs our own search on
the raw `realfft` output, never passed through `rednoise`, with
`--sigma measured`. There is no analytic-on-raw arm on purpose: `realfft` emits
an un-normalized FFT — unit-variance noise gives mean Fourier power `N`, not 1 —
so analytic sigma on a raw file is wrong by `sqrt(N)`. That is a usage error
rather than a configuration worth measuring, and the search's own guard catches
it, at ratios of 3.5e-5 to 3.7e-4 on all three smoke realizations. The measured
MAD adapts to any overall scale, so `rawmeas` is the **well-posed** form of the
question: it gives "skip `rednoise`" the best noise estimator we have, and it
still collapses.

Per knee: **77.3 → 32.4 → 8.5 → 3.0 → 1.0%**, against `coherent`'s 76.2 → 47.9.
Its matched cut inflates 6.80 → 11.80 just to hold the false-alarm rate. A
negative result about our own default path, worth stating plainly.

**What fails is whitening, not scaling, and the paper has to say which.** The run
does not separate the two, so this is a reading of the mechanism rather than a
measurement: a single scale factor cannot fix a red spectrum, because a trial
fundamental at `r` sums harmonics out to `60r` and one profile therefore mixes
Fourier bins whose noise powers differ by orders of magnitude. No single sigma,
measured or computed, is right for all of them, and the red bins dominate the
profile and manufacture candidates. `rednoise` divides each bin by a running
median of the *local* power, which makes the mean Fourier power 1 at every
frequency — normalization and whitening in one step. So SMR's note on headline 6
is right, and the requirement to state is that **the input FFT be locally
normalized**; PRESTO's `rednoise` is one way to get there, not the only one.

**`coherent_meas` — should I measure sigma or compute it?** It makes no
difference: measured sigma on the whitened file tracks the analytic default to
within ~1% at every knee (76.5 / 74.4 / 69.2 / 60.9 / 47.7 against
76.2 / 75.2 / 69.8 / 61.3 / 47.9). The analytic one is free, so use it.

---

## 8. Cost, which has to be quoted with sensitivity

**Per host, because run 2 ran on fitzroy and run 3 on eiger** — the pooled
median mixes two machines and a ratio taken from it would be part hardware:

| median s per realization | `rseek_B` | `coherent_deep` | `rseek_W` | `rseek_A` | `coherent` | `prepfold` | `coherent_tier` | `accelsearch` |
|---|---|---|---|---|---|---|---|---|
| eiger (runs 3 + 4) | 121.8 | — | 29.7 | 27.8 | **17.0** | 5.5 | 5.0 | 1.5 |
| fitzroy (run 2) | 179.8 | 75.3 | — | 42.2 | **29.4** | 8.8 | 8.0 | 2.3 |

(From the 2026-09-16 re-run with the per-host fix. Each timing counts under
the host that produced it. The eiger row now includes run 4's white patches:
84,871 + 22,368 realizations, both at 15 workers. It moved by ≤0.5 s against
run 3 alone: 121.3 / 27.5 / 16.9. The fitzroy row lost the 22,368 timings run 4
overwrote for its arms and moved by 0.1 s.)

Both hosts agree on the ratios that matter: we are **1.4–1.6x faster than
`rseek_A`** and **11–13x slower than `accelsearch`**. That last number
belongs next to `accelsearch`'s 41.6% against our 76.3%. **`rseek_W` costs 7%
more than `rseek_A`** on eiger (29.7 against 27.8 s, including the `realfft` →
`rednoise` → `realfft -inv` round trip). And `rseek_B`, the configuration that
comes within 3.5 points of us under per-band matching (§4), costs **7.2x** our
runtime on eiger and 6.1x on fitzroy.

---

## 9. Three defects fixed on 2026-09-14 — anything quoted before then

All three had the same shape: a number that looked measured and was not.

* **`prepfold` read 0.0% at 100–200 Hz in every cell.** That band is the gap
  between the two injected populations, so its null folds land there at **0.089
  per realization** — under the 0.1 the rate asks for — and the cut came back
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
* **Under `--match knee,band` every white realization scored zero.** Run 2 has
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
  76,105 realizations have `mcpatch_accel_*` rows, but after the merge 76,100 of
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
* **A whitened-rseek arm exists on WHITE noise only.** Run 4 added `rseek_W`
  (riptide on a PRESTO-whitened time series, its own running median off) on
  22,368 white realizations. There it reads **49.0% against `rseek_A`'s 50.2%**
  at the same 7.55 cut, and 1–2 points below `rseek_A` in every white band
  (§4). (`rseek_A`'s figure is over all 76,105 white realizations and
  `rseek_W`'s over run 4's 29% of them, so this is a comparison of samples, not
  a paired one.) So the whitening round trip costs riptide about a point and changes
  nothing else, which is the control the arm needed. **On red noise it is still
  unmeasured**: the one-off check (172 candidates on white, 917 on red, ~170 on
  the same red data whitened; the 200 Hz pulsar back at 11.1 against 11.2, its
  top false alarm back at 7.5 from 155.8) remains **n = 1**. Per-band matching
  already makes riptide's fast end a fair comparison, and §5 is the reason a
  red `rseek_W` might still be worth running.
* **`coherent_deep` is not worth running** — run 2 measured it at 70.6% against 71.0% for
  2.6x the cost.

---

## 12. Still open, in value order

1. ~~Re-run the analysis on the completed run 3~~ — **done 2026-09-14**; every
   number above comes from `report_v4_*`. Signs and orderings did not move.
2. ~~Run 4~~ — **done 2026-09-16**, 22,368 white realizations (planned ~20–24k).
   It closed **the white zero point** (§4's per-band white table, and the white
   cell of the band-matched knee row) and gave `rseek_W` its white control
   (§11). It did **not** close the red side of either of its other two goals,
   because it ran on white indices only:
3. **A red top-up, if either red gap matters for the paper:** run 3's indices
   with `--arms rseek,rseekw` and `--hits-per-inj`. That would (a) replace §5's
   red displaced-hit *bound* (up to 80% of `rseek_A`'s detections) with a
   measurement, and (b) turn §11's n = 1 whitened-riptide check into one.
   Both are statements about riptide, not about us, and per-band matching
   already gives riptide a fair fast end. So this is optional.
4. **`prepfold` at 100–200 Hz stays blank** unless realizations beyond run 2's
   seeds (or a larger `--fap`) are run. That is a hole in the ceiling column,
   not in any comparison.
5. **Restrict `coherent_tier`'s curves to its own band in every figure.**
6. **Decide the headline red-noise figure**: per knee, per band, or both panels
   side by side.
