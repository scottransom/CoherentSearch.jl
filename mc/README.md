# Detection-efficiency Monte Carlo

Design: `../docs/monte_carlo.md` and `../docs/Summary_and_Future_Work.md` §3.2.
Population and profile model: `mc_profiles.py`. Driver: `mc_simulate.py`.
Combining and tabulating: `mc_analyze.py`.

## Running it

```sh
PIXI=/data1/environments/pixiPSR/.pixi/envs/default/bin   # fitzroy; bla0 and the laptop differ

# run 2, on fitzroy: 15 of its 40 logical CPUs, leaving the desktop usable
mc/launch_fitzroy.sh /data1/mc/run2

# or by hand, on any host: N independent workers partitioning the index space
$PIXI/python mc/mc_simulate.py --outdir mcout --nreal 100000 --workers 48 \
    --presto-bin $PIXI --rseek $PIXI/rseek --tpa /path/to/table_1.csv

# the whole report (combining runs is `cat`; this globs *.jsonl)
$PIXI/python mc/mc_analyze.py mcout/
$PIXI/python mc/mc_analyze.py mcout/ --sections roc,pairs,decompose --fap 1e-3
$PIXI/python mc/mc_analyze.py mcout/ --weight flat      # flat in log duty

# one page of diagnostic plots
$PIXI/python mc/mc_quicklook.py mcout/ --fap 1e-2 -o quicklook.png \
    --methods prepfold_snr1,rseek_A,rseek_B,coherent,coh+tier
```

`mc_analyze.py` matches false-alarm rates at `--fap 1e-2` by default. A nominal
cut is not a comparison: ours and rseek's statistics are single-trial,
accelsearch's sigma is already trials-corrected and prepfold's is a chi-squared,
and run 1 measured the four sitting at 8.10 / 7.95 / 8.05 / **7.05** for the same
rate. `mc_quicklook.py` still takes `--fap` explicitly and warns without it.

## The pieces

| file | what it is |
|---|---|
| `mc_simulate.py` | the driver: generate, inject, run every code, record |
| `mc_profiles.py` | the population sampler and the profile model (the astronomy) |
| `mc_model.py` | the band-limited efficiency model and prepfold's drizzle correction |
| `mc_analyze.py` | combining and the whole report |
| `mc_quicklook.py` | one page of diagnostic plots |
| `test_mc.py` | pins for both models, the sampler and the subset selector |

## Red noise (run 3)

Run 2 was pure white noise. The `coherent` arms have always searched
`_red.fft` — run 1's asymmetry was the *opposite* way round (accelsearch got the
raw `.fft` while we got the de-reddened one), which is why `accelsearch_red`
exists. So run 2's `coherent` column already **is** the deployed configuration:
analytic sigma on a whitened file. What run 2 could not do is separate the two
choices a user has, because on white noise `rednoise` has nothing to remove.

```sh
# run 3: the same population, with a red-noise knee drawn per realisation
mc/mc_simulate.py --outdir /data/mc/run3 --rednoise-knee 0.1 50 ...
```

**The knee is the axis; the spectral index is a nuisance parameter.** Fitting
`P(r) = 1 + (r/r_knee)^-alpha` to four real dedispersed observations:

| observation | T | f_knee | alpha | sigma_red/sigma_w |
|---|---|---|---|---|
| PALFA / Arecibo | 269 s | 31.5 Hz | 1.88 | 4.5 |
| Terzan 5 / GBT GUPPI | 4915 s | 6.7 Hz | 1.95 | 6.0 |
| PM0063 / Parkes MB | 2097 s | 1.8 Hz | 0.86 | 0.07 |
| Parkes 70cm | 158 s | 1.6 Hz | 0.23 | 0.05 |

Where red noise matters at all the index is **1.9–2.0** — a random walk, which is
what gain drift and atmospheric opacity produce — while the amplitude spans
**eight orders of magnitude**. So `alpha ~ N(2.0, 0.3)` truncated to [1.2, 2.8],
and the knee log-uniform over 0.1–50 Hz carries the variation. Deriving `alpha`
from a drawn amplitude, the obvious alternative, would make it wander over
exactly the range the measurements say it does not.

The knee range reaching tens of Hz is corroborated independently by Lazarus
et al. (2015): PALFA's measured degradation sets in at `P ~ 100 ms` (10 Hz), and
PRESTO's `rednoise` grows its block size to 100 bins above 6 Hz "where there is
little to no coloured noise". Their **factor 1.1–2 at `P = 0.1–2 s`, DM > 150**
is the number run 3's `coherent` arm should reproduce — quote their high-DM
figure, because their DM dependence is RFI confusability, and we model red noise
only.

### The sigma x rednoise 2x2

The standard procedure is "make a `_red.fft`, zap it, run `coherent_search.jl`",
so the choice that matters is **measured or analytic noise scale**. Run 3 measures
it, against the always-on `coherent`:

| arm | sigma | input | how often |
|---|---|---|---|
| `coherent` | analytic | `_red.fft` | every — the shipped, deployed config |
| `coherent_meas` | measured | `_red.fft` | `--sigma-every` (3) |
| `coherent_rawmeas` | measured | `.fft` | `--sigma-every` (3) |

All three run on the same realisations, so every comparison is paired.

**There is deliberately no analytic-on-raw arm.** `realfft` emits an
un-normalised FFT — unit-variance noise gives mean Fourier power `N`, not 1 — and
it is `rednoise` that normalises, so analytic sigma on a raw file is off by
`sqrt(N)` and is a *usage error*, not a configuration worth a column. It was
tried: the search's own guard caught it on 3 of 3 smoke realisations at ratios
3.5e-5 to 3.7e-4, and the wrong sigma also made the search 2.5x slower (the gate
stops rejecting and the exact rescan runs on everything). The interesting case —
normalised but still red — has no PRESTO tool that produces it, so
`coherent_rawmeas` is the well-posed "can I skip `rednoise`?" arm: the MAD adapts
to any normalisation. `sigma_warn` and `sigma_ratio_seen` are still recorded on
every arm, since that guard is what protects a real user from the same mistake.

**`coherent_deep` is parked and `rseek_B` is thinned to 1-in-10.** Run 2 measured
the deep arm at 70.6% against the default's 71.0% for 2.6x the cost, and answered
what the deep tiling was for; `--deep-coh-every 5` brings it back. With the new
arms the run costs **146.5 s per realisation against run 2's 152.6 (0.96x)**, so
the 2x2 is paid for out of the questions run 2 closed.

**Things that are deliberate here too:**

* **Injected S/N stays defined against the WHITE floor.** `inject` normalises
  analytically against unit-variance noise, so red noise *eats* S/N rather than
  moving the axis under it. Renormalising against the realised variance would
  make run 2 useless as a zero point.
* **Red noise draws off a SEPARATE RNG stream** (`rng_for_rednoise`), so
  switching it on perturbs nothing else: at a given index the population draws,
  the injected S/N, the phases and the white noise are bit-for-bit run 2's.
  **Run 2 is therefore a paired control for run 3**, not an independent sample —
  verified end to end. Sharing one stream would have destroyed that silently,
  and the damage would have looked like ordinary Monte Carlo scatter.
* **`sigma_ratio` is an expectation; `sigma_got` is what the realisation got.**
  The red variance is dominated by a few exponential low-frequency bins, so it is
  a few-DOF random variable: the mean *variance* matches the closed form to
  0.3–1.1%, but the realised *sd* spans 0.55–1.50x it (5th–95th). That scatter is
  physical, so both are recorded and `sigma_got` is the finer covariate. It also
  means `--rednoise-cap` caps the *expected* wander and individual realisations
  will exceed it.
* **The cap is a backstop, not a design driver.** `sigma_red <= 30` (~100x peak
  baseline wander) rejects ~6.5% of pairs, all of it the "high knee AND steep
  index" corner no telescope produces — at 10 Hz it admits `alpha = 2.0`
  (`sigma_red` 4.5) and refuses 2.5 (40.3). It induces a mild knee/alpha
  correlation (mean alpha 2.00 below 8 Hz, 1.86 above 20), which is why both are
  recorded rather than assumed independent.
* **Thresholds are matched WITHIN a knee bin.** The false-alarm rate is
  knee-dependent, so a threshold matched over the pooled run belongs to none of
  the levels in it — the same argument that makes the *codes* comparable, one
  level down. `mc_analyze.py --sections knee` does the split; every other section
  still pools, and the header warns loudly whenever the records carry red noise.

  ```sh
  # give it BOTH run directories: run 2 becomes the `white` row, i.e. the zero point
  mc_analyze.py /data/mc/run2 /data/mc/run3 --sections knee
  mc_analyze.py /data/mc/run3 --sections knee --knee-by sigma
  ```

  `--knee-by sigma` bins on the *realised* `sigma_got` rather than the drawn
  knee, which is the finer covariate for the reason above. The section also
  reports how often the search's own sigma guard fired per bin — on a whitened
  file it should be silent, and where it is not, `rednoise` left a residual and
  the analytic default is reporting inflated S/N.

  **Still to write, deliberately:** the degradation curve (S/N at 50% detection
  against knee) and the comparison to Lazarus's factor 1.1–2. Those depend on
  what the data actually looks like, so they get written against data rather than
  guessed at now.
* **No RFI, on purpose.** Frequency-domain zapping is what a real pipeline does,
  but riptide has no zapping stage — it detrends in the time domain — so
  FFT → zap → iFFT would be a preprocessing step *we* impose on it, unfair either
  way. White and red are the two noise types always present and native to both
  pipelines. Revisit only if a referee asks.

**One bug this work fixed:** `add_rednoise`'s docstring claimed the added
variance was `fcorner^alpha * T^(alpha-1)` — short by `2*dt*zeta(alpha)`, a
factor of ~5000 at run-3 size — and the CLI help repeated it as "sane values are
~0.003-0.05 Hz". Those knees are ~100x below any real observation and add ~2% of
the white sigma, so **a red-noise run configured from that advice would have
measured nothing.** The code was always right; only the guidance was wrong.
`test_mc.py` pins the number so it cannot come back.

`--tpa` wants the MeerKAT TPA supplementary `table_1.csv`
(`stab2775_supplemental_file.zip`). Without it the sampler falls back to the
recorded quantiles of that table, which is close but does not carry the real
(period, width, W10/W50) joint distribution.

## What it does per realisation

White noise, `N = 2^24`, `dt = 60 µs` (`T = 1006 s`), 6 injected pulsars drawn
from the TPA width population at continuous injected S/N over 5.5–11.5, then the
**same** realisation handed to every code. One JSON object per realisation.

| arm | configuration | how often |
|---|---|---|
| `prepfold` | `-nosearch` at the known period; on the injection-free realisations, at random periods from the same population (its measured null) | every |
| `accelsearch` | `-numharm 16 -zmax 0` on the raw `.fft` | every |
| `accelsearch_red` | the same on `_red.fft` | every |
| `rseek_A` | `bmin 20 / bmax 120`, matching our coverage exactly | every |
| `rseek_B` | the deep 4-range tiling, 4.05x the cost | 1-in-5 |
| `coherent` | the shipped defaults (`nharms 60`, `maxdecim 6`, `hifreq 125`) | every |
| `coherent_tier` | `nharms 120 maxdecim 12` below 5 Hz — §4's proposal | every |
| `coherent_deep` | `nharms 120 maxdecim 12` over the whole band | 1-in-5 |

`mc_analyze.py` also forms **`coh+tier`**, the union of the first two coherent
arms: hits merged by the stronger statistic, false-alarm tails concatenated. That
is what a tiered search would actually report, and because each arm ran as its own
invocation the union's threshold is *measured*, not assumed — a deep tier has to
pay for its own trials.

Seeds come from the realisation index alone, so workers need no coordination,
a rerun reproduces the same noise, and appending to an existing output file
skips indices already present.

## Things that are deliberate

* **Injected S/N is defined against a ZERO-MEAN unit-L2 template.** riptide's
  `generate_signal` normalises the von Mises including its DC component, but
  every search removes the baseline; that convention makes 8% of the "injected"
  S/N unrecoverable in principle at 10% duty and **30% at 30% duty**, right
  along the main axis of the study. Verified exact: the sampled template comes
  out mean `-4e-19`, L2 `9.000000` for `snr=9`.
* **Statistic values are recorded, not booleans**, plus injection-free
  realisations (`--noise-every`). Ours and rseek's thresholds are single-trial,
  accelsearch's sigma is already trials-corrected, prepfold's is a chi-squared —
  a common nominal threshold means nothing, and only a matched empirical
  false-alarm rate does. `mc_analyze.py --fap` is what makes the columns
  comparable; without it the table prints a warning.
* **False alarms are counted on every realisation**, not just the empty ones —
  a candidate matching no injection at any simple harmonic ratio. The top **800**
  statistics are stored, so a rate curve can be built at analysis time.
* **Two things censor that curve, and only one of them is ours.** The stored
  top-N cap is ours: at run 1's 200 it truncated `rseek_B` on **100%** of
  realisations, making its rate a ceiling below 6.20 (it reports a median of 432
  candidates). 800 covers it. The other is each code's own reporting floor —
  `--rseek-smin 6.0`, our `--threshold`, accelsearch's sifting cut at ~4.8 —
  below which the curve is *flat by construction*. Run 1's report printed
  `rseek_A` at 161.96 for cuts 5.0, 5.5 and 6.0 and said nothing; that is one
  number three times. `mc_analyze.py` now marks those cells `c` and warns if any
  matched threshold lands within 0.15 of a floor.
* **Our reporting floor is 5.5, not 6.0, and that is because of the tier arm.**
  `coherent_tier` searches 6.4x fewer trials than the default arm, so its own
  matched threshold sits near 6.0 (measured 6.05 on a short-`T` smoke run) — at a
  6.0 floor it would have been floor-limited and read as better than it is. It
  costs nothing: the gated exact rescan fires on ~1e-6 of trials either way.
  **rseek's floor is left at 6.0 on purpose** — its matched thresholds are 7.95
  and 8.05, its rate at 6.5 is already 20–48 per realisation, and lowering it
  would multiply an already ~430-long candidate list over a region of the curve
  no threshold ever reaches.
* **Harmonic detections count as detections, and are flagged.** rseek and
  accelsearch do not collapse the `f/2, 2f, 3f/2 …` family and we do, so scoring
  them as misses would penalise the codes that report them.
* **Scattered pulsars are kept** (`Sflag` in the TPA table). The paper excludes
  them from its width fits; they have median duty 3.71% against 2.15%, so
  dropping them biases the population narrow, which *flatters* a deep harmonic
  sum.
* **Profiles are a von Mises core plus wings solved to the drawn W10/W50.** A
  single von Mises is pinned at ratio 1.823 and cannot represent ~35% of the
  measured population, all on the broad-winged side — and broad wings carry less
  high-harmonic power, so getting them wrong would flatter us. Below the
  scattering limit `ln10/ln2 = 3.32` the wings are a one-sided exponential tail;
  above it they are a broad pedestal component, because no amount of scattering
  can exceed that ratio. Unreachable combinations are recorded
  (`profile.exact = false`), never silently clipped.
* **rseek gets two configurations.** `RSEEK_A` (`bmin 20 / bmax 120`,
  `Pmin = 1/750`) matches our default coverage exactly. `RSEEK_B` is the deep
  4-range tiling, measured at **4.05x** the cost (188.6 s against 46.6 s), run on
  a subset via `--deep-every`. Note a *literal* riptide-pipeline tiling is
  **worse** than A at short periods — narrow bins ranges pin `b` near 22, where
  `bmin 20 / bmax 120` lets it climb to the sampling limit (measured 9.6 vs 11.8
  on the same 271 Hz pulsar) — which is why `RSEEK_B`'s first range is wide.
* **`prepfold` is a reference, not a competitor.** It folds at the known period
  with no trials penalty. Its chi-squared sigma is the standard people fold
  with; the `snr1` boxcar computed from its `.bestprof` is the same statistic as
  the searches, so there is one directly comparable column.
* **prepfold's bins are NOT independent, and run 1's numbers were inflated
  because of it.** `fold()` drizzles each finite-duration sample across every
  profile bin it covers, which correlates neighbouring bins; `snr1` divides by
  `σ√(w(1−δ))` as if they were independent, and the study can see it. At fixed
  duty, prepfold's recovered/injected ratio *rose 11% with spin frequency and
  crossed 1.0* — an inflated ceiling, in the MSP band, which is the worst place
  for one. `mc_model.drizzle_boxcar_corr` is exact linear algebra on the fold
  weights and flattens it (200–300 Hz: 1.009 → 0.858, against 0.902 at low
  frequency). It is **width-dependent**: a one-bin boxcar cannot feel
  bin-to-bin correlation at all, so a flat `√DOF_corr` would over-correct narrow
  pulses by ~20%. The correction is applied at ANALYSIS time, from the `(nbins,
  dt_per_bin, w)` recorded beside the raw value, so revising it never means
  re-folding a week of compute.
* **prepfold's σ is computed, not measured.** The MAD carries ~11% scatter on the
  31–128 bins prepfold picks, and `snr1` is exactly `1/σ̂`, so it put a scatter
  term on the reference column comparable to everything else in it. Both are
  recorded.
* **prepfold's profiles are stored** — 1-in-10 realisations and **every**
  injection-free one — so the drizzle model can be *checked* rather than
  believed. `mc_analyze.py --sections drizzle` measures the inter-bin covariance
  from those null folds and prints it beside the model. Verified against 210 real
  `prepfold` folds of pure noise: at `dt_per_bin = 1.35`, where the correction is
  a 10% effect, measured and modelled agree to **0.1–2%** (0.928/0.928,
  0.896/0.897 at w = 2, 4), and the analytic σ is right to ~2% at every
  `dt_per_bin`. The empties are what that check needs and they are already at
  100%; `--keep-profiles 1` stores every realisation's, for ~290 MB over a
  40k-realisation run.
* **The duty cycle is stratified and every injection carries an importance
  weight.** Run 1 drew 757 injections below 0.5% duty out of 82,014, and that is
  the one place we lose. The default `--strat 0.005 0.5 0.25` raises that
  fraction from 1.5% to 3.7% at 0.85 effective sample size, and `weight = 1/q`
  makes every population-weighted number identical to the unstratified one.
* **1-in-N subsets are selected by a HASH of the realisation index, not by
  `idx % N`.** Worker `w` of `W` takes `idx % W == w`, so when `N` divides `W`
  the whole subset lands on `W/N` of the workers — which are then the *slow*
  ones. Run 1 asked for 1-in-3 deep tilings and finished with **14.4%** (2,197 of
  15,212); that was this, and it was read as ordinary attrition.

## Cost

Run-1 medians, measured on `bla0` (48 workers, `N = 2^24`), and the run-2
estimate for fitzroy — scaled by 1.35, which is roughly the per-core ratio and is
the number the projection below stands or falls on:

| stage | bla0 (run 1) | fitzroy (run 2, est.) | how often |
|---|---|---|---|
| `rseek_B`, deep tiling | 121.1 s | 163 s | 1-in-5 → 33 s |
| `rseek_A` | 29.2 s | 39 s | every |
| `coherent` | 20.2 s | 27 s | every |
| `coherent_deep`, full band | — | 54 s | 1-in-5 → 11 s |
| `prepfold` x6 | 5.8 s | 8 s | every |
| generate + inject | 4.5 s | 6 s | every |
| `coherent_tier`, below 5 Hz | — | 6 s | every |
| `accelsearch` x2 | 1.7 s | 4.5 s | every |
| `realfft` + `rednoise` | 0.8 s | 1.1 s | every |

**~137 s per realisation**, so 15 workers give ~9,400 realisations (~56,000
injections) per day. The 1-in-5 arms cost 44 s of that between them, i.e. a third
of the run goes to the two configurations that exist to *test* a claim rather
than to make one.

**Memory is the constraint on worker count**, not CPU: rseek's deep ranges peak
at ~1.7 GB RSS, and each realisation holds ~200 MB of transient files in
`--workdir` (default `/dev/shm`). At 15 workers that is ~3 GB of scratch against
fitzroy's 94 GB `/dev/shm`, plus up to ~25 GB of RSS if every worker hits a deep
range at once, against 187 GB. Comfortable. At 48 workers on a smaller box it is
not — raise `--deep-every`, or lower `--workers`.

## Failure modes this has already hit

* **`rednoise` writes `<stem>_red.inf` into the CURRENT DIRECTORY**, not beside
  its input. Every PRESTO tool is therefore run with `cwd` set to the work
  directory. Before that fix `coherent_search` died with "The .inf file ... was
  not found" on every realisation while the other three codes carried on — which
  reads as a **0% detection fraction, not as a crash** — and the driver littered
  the repo root with one stray file per realisation. `mc_simulate.py` now aborts
  if any search fails on a worker's *first* realisation, precisely so this class
  of thing cannot burn a night again.
* **A method that runs on only a subset must not be divided by every injection.**
  `rseek_B` runs under `--deep-every`, and counting its detections against all
  injections made it read 11.4% where it is really 95%. `mc_analyze.py` now
  restricts every cross-code cell to the realisations all compared codes ran, and
  says so in the header; run 1's report only footnoted it with a `*`, and its
  headline table still put `rseek_B`'s 1-in-3 subset beside a `coherent` column
  built from 7x more data.
* **`idx % N` selects a 1-in-N subset that lands on `W/N` of the workers.**
  See the `one_in` note above. Run 1's deep tiling was configured 1-in-3 and
  finished at 14.4%. The symptom is a subset fraction well under `1/N` with no
  errors anywhere, which reads as attrition and is not.
* **Buffered output hides a slow script.** Every analysis command here prints as
  it goes; run them with `python -u`, and never pipe into `head`/`tail` while
  waiting, or the first output you see is the last.
