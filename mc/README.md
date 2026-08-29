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
  a candidate matching no injection at any simple harmonic ratio. The top 200
  statistics are stored, so a rate curve can be built at analysis time. rseek at
  `--smin 6` reports ~150 candidates per file, which is why the cap is 200 and
  not 20.
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
* **prepfold's σ is computed, not measured.** A 128-point MAD has ~7% sampling
  error and `snr1` is exactly `1/σ̂`, so the MAD alone put a scatter term on the
  reference column comparable to everything else in it. Both are recorded.
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
