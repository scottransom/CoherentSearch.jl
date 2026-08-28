# Detection-efficiency Monte Carlo

Design: `../monte_carlo.md` and `../Summary_and_Future_Work.md` §3.2.
Population and profile model: `mc_profiles.py`. Driver: `mc_simulate.py`.
Combining and tabulating: `mc_analyze.py`.

## Running it

```sh
PIXI=/data1/environments/pixiPSR/.pixi/envs/default/bin   # or the laptop's

# fill a machine: N independent workers partitioning the realisation index space
$PIXI/python mc/mc_simulate.py --outdir mcout --nreal 100000 --workers 48 \
    --presto-bin $PIXI --rseek $PIXI/rseek --tpa /path/to/table_1.csv

# look at what came out (combining runs is `cat`; this globs *.jsonl)
$PIXI/python mc/mc_analyze.py mcout/ --fap 1e-2 --by ducy
```

`--tpa` wants the MeerKAT TPA supplementary `table_1.csv`
(`stab2775_supplemental_file.zip`). Without it the sampler falls back to the
recorded quantiles of that table, which is close but does not carry the real
(period, width, W10/W50) joint distribution.

## What it does per realisation

White noise, `N = 2^24`, `dt = 60 µs` (`T = 1006 s`), 6 injected pulsars drawn
from the TPA width population, then the **same** realisation handed to four
codes: `prepfold -nosearch` at the known period, `accelsearch -numharm 16
-zmax 0`, `rseek` (riptide's FFA), and `coherent_search.jl` at its bare
defaults. One JSON object per realisation.

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
  the searches, so there is one directly comparable column. Plain `snr1` is
  correct there with no band-limited correction — prepfold folds in the time
  domain, so its phase bins really are independent.

## Cost

Measured on the laptop, one worker, `N = 2^24`:

| stage | wall |
|---|---|
| `rseek_B` (deep tiling, subset only) | 177 s |
| `rseek_A` | 51 s |
| `coherent_search` `-t 1` | 20 s |
| generate + inject 6 pulsars | 11 s |
| `prepfold` x6 | 4 s |
| `accelsearch` | 1 s |
| `realfft` + `rednoise` | 0.7 s |

So ~96 s per realisation without the deep tiling and ~270 s with it. At
`--deep-every 3` the average is ~160 s, i.e. **6 injections per 160 s per
worker**.

**Memory is the constraint on worker count**, not CPU: rseek's deep ranges peak
at ~1.7 GB RSS, and each realisation holds ~200 MB of transient files in
`--workdir` (default `/dev/shm`). At 48 workers that is ~10 GB of scratch plus up
to ~80 GB of RSS if every worker hits a deep range at once. Lower
`--deep-every`, or `--workers`, if that is tight.
