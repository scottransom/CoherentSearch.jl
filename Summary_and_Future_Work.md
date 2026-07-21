# CoherentSearch.jl — Summary and Future Work

This document summarizes the current state of the Julia port of the Python
`coherent_search` package. Phase 1 delivered a correct, oracle-validated,
multi-threaded search; Phase 2 (now implemented) reorganized the search around
independent fundamental-frequency *chunks* and made the hot loop
allocation-free with cached FFTW plans and interpolation kernels.

---

## 1. What exists today

A working, well-tested, multi-threaded coherent harmonic-summing search that is
numerically validated against the original Python code, and an optimised search
path built on cached plans, per-harmonic interpolation tuning, and chunk-level
parallelism.

### Components

| File | Role |
|------|------|
| `src/fourierinterp.jl` | Fourier interpolation kernels (Eqn. 30 of [astro-ph/0204349](https://arxiv.org/pdf/astro-ph/0204349)) |
| `src/fileio.jl` | `mmap`'d PRESTO `.fft` reader + `.inf` metadata parser |
| `src/search.jl` | Reference `block_metrics` + optimised chunk-parallel `search` |
| `bin/coherent_search.jl` | ArgParse command-line front-end |
| `test/` | 40 unit tests (kernel golden values, indexing, irfft convention, optimised-vs-reference equivalence, detection) |
| `crossval/` | Python-as-oracle accuracy + speed cross-validation |

### Key design decisions

- **Indexing is isolated and audited.** The 0-based (Python, half-open slices)
  → 1-based (Julia, inclusive ranges) translation lives in one documented
  helper, `nearby_fourier_bin_range`, with the original Python slice arithmetic
  written out in comments.

- **A simple reference path is kept alongside the optimised one.**
  `block_metrics(ft, rfund, params)` is a self-contained, allocating
  implementation that mirrors the Python algorithm one-to-one. It is left
  unoptimised on purpose so it stays easy to audit, and it is what the Python
  oracle is pinned to at machine precision. The optimised `search` is then
  pinned to *it*.

- **FFT conventions verified, not assumed.** `np.fft.fft`/`ifft` and Julia's
  `fft`/`ifft` share the same normalization, so the FFT-correlation interpolator
  ports directly. The one subtlety — `np.fft.irfft` vs FFTW's `c2r` handling of
  the DC and Nyquist bins' imaginary parts — is checked by a dedicated test;
  both ignore those imaginary parts, so the coherent fold matches.

### Verification status

- **All 40 unit tests pass** (the original kernel/indexing/IO tests plus a new
  `test_search.jl` covering the optimised path).
- **Accuracy cross-validation** (`crossval/crossval_accuracy.jl`) runs the
  original Python `coherent_search` as an oracle and agrees to **~3e-16 relative
  on the `finterp_FFT` kernel** and **~8e-16 relative end-to-end** (the full
  coherent-fold metric) on the bundled 10.0123 Hz test pulsar. This is the
  primary guard that the indexing and FFT conventions are correct.
- **The optimised search path is pinned to that oracle.** With per-harmonic
  tuning disabled (`align=false`) the new chunk/plan-caching path reproduces the
  reference `block_metrics` to **~6e-16 relative** — i.e. it is numerically
  identical to the oracle-validated reference, so the performance work provably
  did not change results.

### How to run things

```sh
# Tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Search (use -t auto for all cores)
julia --project=. -t auto bin/coherent_search.jl FILE.fft --lofreq 0.1 --hifreq 100

# Cross-validation (COHERENT_PYTHON / COHERENT_FFT configurable)
julia --project=crossval        crossval/crossval_accuracy.jl FILE.fft
julia --project=crossval -t auto crossval/crossval_speed.jl   FILE.fft
```

---

## 2. The optimised search (implemented)

Following `coherent_search_design.md`, the production search is structured as
three nested loops, with parallelism at the *outermost* one:

- **Loop #1 — chunks (parallel).** The fundamental-frequency range is cut into
  chunks of `Nprof` trial fundamentals (`blocksize`, default 2048). Chunks are
  independent and are distributed round-robin across `nthreads()` tasks with
  `Threads.@spawn`; each task owns one private `Workspace`, so there is no shared
  mutable state and no `threadid()` indexing (robust under task migration).
- **Loop #2 — harmonics.** For each harmonic `h`, one Fourier interpolation fills
  row `h+1` of an `(nharms+1) × Nprof` complex amplitude array `ftprofs`.
- **Loop #3 — profiles.** A single *batched* complex→real transform inverts all
  `Nprof` profiles at once (`plan_brfft(ftprofs, 2*nharms, 1)`), then a
  width-sensitive S/N metric (see §3) is read off each profile column.

### What makes it fast

- **Plans built once, executed many times.** FFTW *planning* is not thread-safe,
  so every plan (`plan_fft`, `plan_bfft`, `plan_brfft`) is built single-threaded
  while the workspaces are constructed, before the parallel region. The hot loop
  only *executes* prebuilt plans on a workspace's private buffers via `mul!`.
- **Cached interpolation kernels.** The FFT'd sinc/phase kernel
  (`finterp_fft_coeffs`) depends only on `(numbetween, m, fftlen)`, so it is
  precomputed once per harmonic and shared read-only across threads. The old
  per-call recomputation (an extra FFT plus transcendentals on *every* harmonic
  of *every* block) is gone. The `1/fftlen` inverse-FFT normalization is folded
  into the cached kernel, so the loop can use an unnormalized `bfft`.
- **Allocation-free hot loop.** `fill_chunk_profiles!` on a warm workspace
  allocates ~2 KB total (just the per-harmonic `@view` headers), independent of
  `Nprof` — versus the old path that allocated several arrays per harmonic per
  call. No per-chunk garbage means no GC pauses serializing the threads.
- **Batched inverse FFT.** The `Nprof` profiles are short (`2*nharms` points);
  one batched `brfft` amortizes FFTW overhead far better than `Nprof` tiny calls.
  (Every part of the S/N metric except its linear signal term is scale-invariant,
  and that term folds the missing `1/Nbins` irfft normalization into its `scale`
  argument, so the unnormalized transform is used directly — see §3.)

### Per-harmonic `numbetween` (the `align` option)

The trial fundamentals are stepped by `deltar = hidr/nharms` bins, so harmonic
`h` is sampled every `deltar_h = hidr·h/nharms` bins — finer at low harmonics.
`harmonic_numbetween` sizes each harmonic's interpolation oversampling to its
own `deltar_h` (`= nharms/(hidr·h)`, e.g. 64 at `h=1` down to the floor at high
`h`), never going below `numbetween` (the accuracy floor). The result is that
the low harmonics — which carry most of a pulsar's power — get a much finer,
more accurate interpolation grid: harmonic-1 amplitudes match a `numbetween=256`
reference to **~1e-15 at the aligned `nb=64`** vs **~1e-2 at the fixed `nb=16`**.
Each harmonic also gets its own `fftlen`, sized to span a full chunk in a single
transform (no tiling at the default chunk size). `align=false` (`--noalign`)
falls back to a single fixed `numbetween`, which is the configuration used to
prove bit-level equivalence with the reference.

> **Caveat worth keeping in mind.** Aligning `numbetween` to `deltar_h` is an
> *accuracy* lever at low harmonics, not a throughput lever at high ones: linear
> interpolation between finterp grid points needs the grid finer than the ~1-bin
> response curvature regardless of how coarse `deltar_h` is, which is why the
> floor exists and why high harmonics stay at `numbetween`. On the *nonlinear*
> width-sensitive S/N metric the end-to-end difference from a fixed grid is order
> ~1% and not strictly monotonic (per-harmonic errors partially cancel); the win
> is real and large at the *amplitude* level, where it physically belongs.

### Measured behavior

- **Thread scaling** on a 1–50 Hz search of the test file: 7.5 s → 4.1 s → 2.3 s
  at 1 / 2 / 4 threads (**~3.3× on 4 cores**), with an identical candidate count
  at every thread count (deterministic and correct).
- **Detection** of the bundled 10.0123 Hz pulsar is recovered at
  `10.0123125 Hz`, independent of chunk size.

### Performance optimization (2026-07)

A profiling pass (`bench/`: `microbench.jl`, `profile_search.jl`,
`median_bench.jl`) found the hot-loop cost was **not** where the design narrative
assumed. On the heavy `--metric sd2 --maxdecim 6` config (`nharms=60`), single-
thread self-time split as: **profile-median `sort!` ~41%**, FFTW ~30%, everything
else ~29% — while `uniform_linear_interp`, despite ~1.5 billion calls, was ~0%.
Two changes followed, each guarded by the `align=false` machine-precision gate
and a byte-identical full-run candidate diff:

- **Quickselect median** (`_median!`/`_select!` in `search.jl`) replacing the
  radix `sort!` in `_profile_snr`, plus fusing the copy/argmax passes and
  special-casing `w^pexp` for `pexp ∈ {0.5, 1}`. The median ran ~1e8× on short,
  *cold* profile columns where radix sort mispredicts; quickselect for just the
  two central order statistics is ~2× and returns the identical value. Median
  bucket 41% → 7.5%.
- **Type-stable `Workspace`** (`Workspace{S,B,D}` with a concrete
  `Dict{Int,FFTScratch{…}}` / `Vector{DecimBuf{…}}`). The abstract containers had
  made `sc.fwd`/`sc.bwd`/`db.brfftplan` `::Any`, so every hot-loop `mul!`
  dispatched dynamically and boxed its result (`fill_harmonic_row!` 64–80 B/call
  → 0 B).

Net: warm single-thread search **~1.6× faster** (0.1–5 Hz band, 20.1 → 12.6 s),
results provably unchanged.

**Investigated and rejected:** sizing the interpolation `fftlen` to smooth
`2·3·5·7` numbers instead of the next power of two. Per large size it looks ~2×
faster, but the total ceiling is only ~3% (mixed-radix per-point cost cancels the
smaller size), and smooth sizes need ~60 distinct lengths whose `MEASURE`
planning is ~0.7 s *each*; `ESTIMATE` avoids that but executes slower, making it a
net regression. **`next_pow_of_2` + `MEASURE` is already ≈ FFTW's best case** — so
the throughput-tuning/tiling items below are lower-value than expected.

---

## 3. Next steps

> **Status: feature-complete.** The search, detection metric, candidate
> de-duplication, harmonic decimation, and candidate profile plots are all
> implemented, tested, and oracle-validated. The primary focus now shifts from
> features to **performance**: careful profiling of the hot loop (interpolation,
> batched inverse FFTs, allocation and memory-bandwidth behavior under threading)
> and acting on what it finds. The throughput-tuning and tiling items below are
> the concrete starting points for that work.

- **The big remaining lever: `ComplexF32` interpolation.** The `.fft` amplitudes
  are already `ComplexF32`; the interpolation pipeline widens to `ComplexF64`.
  Keeping it `ComplexF32` would ~halve the bandwidth of the two dominant buckets
  (the interp FFTs *and* the full-length `spec .* coeffs` multiply). It breaks the
  `Float64` oracle pins (~1e-7 vs ~1e-15), so it needs a deliberate precision-mode
  design and fresh validation against injected fake pulsars — a separate effort,
  not a drop-in. This is the most promising throughput item.
- **Rethink FFT-correlation vs. direct interpolation.** The FFT computes ~16k
  fine-grid points per harmonic to then linear-interp only the ~`Nprof` we need; a
  cached-coefficient *direct* `O(m)` interpolation of just those points is ~7×
  fewer arithmetic ops in principle. An architectural change to `interp_tile!` /
  `fill_harmonic_row!` — higher risk, potentially a large win. Profile first.
- **`fftlen` sizing is settled (do not revisit).** Sweeping `fftlen`/`numbetween`
  or smooth (`2·3·5·7`) sizes was investigated (see §2) and rejected:
  `next_pow_of_2` + `MEASURE` is already ≈ FFTW's best case. Tiling a chunk into
  smaller overlapping transforms only matters if a future capped `fftlen` is
  forced (e.g. a memory-constrained `ComplexF32` mode); `fill_harmonic_row!`
  leaves room for the tile loop, but it is not a throughput win on its own.
- **Candidate de-duplication (implemented).** `remove_duplicates` (wired to the
  default; `--noremove` now disables it) collapses the run of adjacent trial
  fundamentals a single signal lights up down to its strongest member: sort by
  Fourier frequency `r`, group where consecutive `r` fall within `dr_tol` bins
  (`--drtol`, default 1.0 — one bin is `1/T` Hz, far finer than the spacing of
  distinct sources yet comfortably wider than the sub-bin coherent cluster),
  keep the max-metric candidate per group. On the test band this turns ~32k
  above-threshold trials into the single 10.0123 Hz candidate.
- **Harmonically-related de-duplication (implemented).** `remove_harmonics` (wired
  to the default; `--noharmremove` disables it, `--numharm` sets the max harmonic)
  collapses the `f/2`, `2f`, `3f/2`, … family a real signal produces — a distinct
  problem from the near-identical collapse above, and one made especially
  prominent by harmonic decimation (whose subharmonic folds report genuinely
  different Fourier frequencies `r`). Candidates are visited strongest-metric
  first; each is kept unless its `r` is a small-integer ratio `n/m` (both ≤
  `--numharm`) of an already-kept stronger one, tested as `|m·r_hi − n·r_lo| ≤
  tol·m` (a bin-scale tolerance on the shared comb that does not tighten
  spuriously at high harmonic number). On a band spanning `f/3 … f` with
  decimation the ~10-member family collapses to the single strongest survivor.
  *Worth noting:* the survivor is whichever member scored highest, which may be a
  subharmonic fold (e.g. `f/3` summed with all 60 harmonics can outscore the
  direct `f` fold with 20) rather than the true fundamental — reporting the
  physical fundamental of each family is a further refinement, as is threshold
  comparability across differing harmonic counts.
- **Statistically meaningful detection metric (implemented).** The old
  peak/|trough| ratio is replaced by a width-sensitive metric ported from the
  Python `snr_metric` (`_profile_snr` / the public `snr_metrics`):

      metric = sum_on(prof - median) / rms / width^pexp

  The **signal** sums the excess over the median across the *on-pulse* set — the
  bins above `xsignal·(peak - median)` (`--xsignal`, default 0.2) — so it is a
  stable measure of pulsed flux that does not grow with `nbins` and adds up
  multi-component pulses (two peaks with a valley) that a boxcar would miss.
  `rms = 1/sqrt(2*ngoodbins+1)` with `ngoodbins = min(Nyquist/r̄, nharms)` per
  chunk. The **width** penalty is selectable (`--metric`, `--pexp`):
    - `:non` — `width = N_on`, the count of on-pulse bins (a duty-cycle penalty).
      `pexp=1/2` is the calibrated matched filter (equivalent-σ); larger `pexp`
      suppresses high-duty-cycle signals (broad or many-toothed RFI) while
      leaving narrow pulses — even widely separated multi-component/interpulse —
      alone, since it keys on *how many* bins are lit, not *where*. **Default.**
    - `:sd2` — `width = Σd²`, the summed squared modular phase distance of the
      on-pulse bins from the peak. Penalises phase *spread*; larger `pexp`
      down-weights scattered profiles harder, but also genuine wide doubles.

  Empirically (equal matched-filter S/N templates), `:non` cleanly separates
  narrow pulsars from sawtooth/broad RFI and is stable across `nharms`, whereas
  `:sd2` mis-ranks interpulse pulsars below many-toothed RFI — hence `:non` is
  the default. The hot loop keeps its unnormalised batched `brfft`: median,
  argmax, on-pulse set and width are scale-invariant, so only the linear
  `signal` term needs the `1/Nbins` factor (folded into the `scale` argument).
  Oracle-pinned two ways — the reconstructed **profiles** match numpy to ~2e-16
  (FFT conventions), and *both* width penalties run on *identical* profiles match
  the shipped Python `snr_metric` to ~2e-16 (the port itself, isolated from the
  metric's discontinuous on-pulse threshold). *Still worth doing:* validate
  against injected fake pulsars of varying width, and sweep `pexp` on real data.

- **Boxcar matched-filter metric `:boxcar` (implemented) — fixes the `√nbins`
  disease at the source.** Rather than *normalise away* the non-analytic noise
  floor (the `--normalize` route below), `--metric boxcar` replaces the
  data-adaptive on-pulse selection with a *fixed* bank of top-hat filters, whose
  noise statistics are analytic. It correlates each profile with boxcars of
  geometric width (`wₖ₊₁ = max(⌊1.5·wₖ⌋, wₖ+1)`, riptide's recurrence, capped at
  `boxcar_maxfrac·nbins`, default 0.3) via the prefix-sum "strided differences" of
  Morello et al. 2020 (MNRAS 497, 4654, §5.4): one exclusive prefix sum of the
  median-subtracted, phase-tiled profile, then every boxcar sum is a two-index
  difference. The reported metric is the peak matched-filter S/N,
  `max_{w,p} (Σ_{i=p}^{p+w-1}(Pᵢ − med)) / (σ·√w)`. Because the widths are chosen
  a priori, a width-`w` boxcar over white noise is `N(0, w·σ²)`, so `/√w` makes
  every (phase, width) trial exactly `N(0,1)` — the peak over trials follows
  analytic extreme-value statistics with a known, ~`nbins`-flat trials factor, and
  there is *no* `√nbins` floor to correct. It is scale-free (a ratio of two
  linear-in-amplitude quantities), so the unnormalised hot-loop `brfft` and the
  normalised reference `irfft` give the identical value — the `align=false`
  equivalence pin extends to it unchanged — and neither `ngoodbins` nor the
  `scale` factor enters. `xsignal`/`pexp` are unused.
  - **The noise `σ` is estimated once per `(block, k)`**, not per profile:
    `1.4826·MAD` over a strided subsample (`_BOXCAR_SIGMA_SAMPLES = 8192` bins) of
    the block. A per-*profile* MAD (only `nbins` samples) has `~0.76/√nbins`
    relative error — ~17% at `nbins=20` — which multiplies straight into every
    S/N and re-inflates the small-`nbins` tail (measured: it flipped the FAP=1e-4
    drift to run *up* with `k`, 5.9→9.5). Pooling thousands of block bins drops
    `σ̂`'s variance below 1%, restoring the clean per-trial `N(0,1)`; it is also
    cheaper (one MAD per block vs one per profile) and, being median-based and
    pooled, immune to the rare signal/RFI bin. The subsample indices depend only
    on `(nbins, n)` and enter only through the `excess/σ` ratio, so the pins hold.
  - **Measured (`PM0063…red.fft`, 1–20 Hz, `maxdecim 6`, 4.78M trials/k):** the
    FAP=1e-4 threshold is now **flat across decimations** — 5.28 (k=1, 120 bins) →
    5.02 (k=6, 20 bins), a 5% spread, vs `:non`'s 9.82→5.90 (67%) and the
    per-profile-MAD boxcar's 5.86→9.52. The std is tight and stable (0.46→0.61 vs
    `:non` ~0.71). A single `--threshold` finally means one consistent false-alarm
    rate for every `k`. The residual mean drift (2.89→2.12) is just the analytic
    expected-max-over-phase-trials growth (does not move the detection threshold),
    and the ~2× low-f/high-f drift that remains is the *same* red-noise structure
    for every `k` (per-`k` FAP=1e-4 min ~4.9–5.1), cleanly separated from
    decimation. **This is the preferred path** and largely obviates `--normalize`'s
    motivation; the frequency (red-noise) drift is the only thing left for a
    per-`f` threshold, and it is now `k`-independent. *Still worth doing:*
    injected-signal width/S/N validation, a semi-analytic trials-factor →
    equivalent-σ map (the analytic EVD makes this tractable now), and deciding
    whether `:boxcar` becomes the default (retiring `:non`/`--normalize`).

- **Threshold calibration across metric / `pexp` / decimation (to investigate).**
  The metric's numeric scale is *not* comparable across `--metric` or `--pexp`,
  so a fixed `--threshold` means different things in each configuration. On the
  test pulsar, the same signal reads ~28 at `:non`/`pexp=0.5`, ~20 at
  `:non`/`pexp=1.0`, and ~39 at `:sd2`/`pexp=0.5`; only `:non`/`pexp=1/2` is a
  calibrated equivalent-σ, and even that is single-trial (no trials factor). We
  need to work out how the detection threshold should be set for each
  metric/`pexp` — ideally derive (or empirically fit, from pure-noise runs) the
  false-alarm rate vs. threshold for each configuration so a single "sigma"-like
  knob has a consistent meaning, and fold in the number of independent trials
  searched. Until then, `--threshold` must be re-tuned by hand whenever
  `--metric` or `--pexp` changes.
  - **`nbins`-dependence of the noise floor — confirmed, and it bites under
    decimation.** The pure-noise metric is *not* `nbins`-independent as the
    `snr_metric` docstring hoped: with `:non`/`pexp=0.5` its whole distribution
    scales as `√nbins = √(2·Hk)` (mean, min, max all shift up together; the std
    stays ~constant). Measured on `PM0063…red.fft`, 5–30 Hz: mean metric / √nbins
    ≈ 0.61–0.63, flat across `k=1..6`, so the raw mean runs 6.87 (k=1, 120 bins)
    → 2.74 (k=6, 20 bins). Because harmonic decimation folds `nbins = 2·⌊nharms/k⌋`
    (fewer bins at higher `k`), a single `--threshold` is systematically biased
    toward the low-`k` (many-bin) decimations — at `threshold=6` the k=1 *median*
    already sits above threshold, flooding the candidate list from one decimation
    while k=5/6 contribute almost nothing. The cause is the adaptive on-pulse set:
    under noise `N_on ∝ nbins`, and summing `N_on` selected noise excesses gives a
    "signal" whose fluctuation grows as `√N_on ∝ √nbins`, which the `width^0.5 =
    N_on^0.5` penalty does *not* cancel (it cancels the *count*, not the
    selection-induced bias). A proper fix is a per-`nbins` (equivalently per-`k`)
    threshold, or renormalising the metric by its measured pure-noise mean/σ at
    each `nbins` so a single sigma-like threshold is comparable across
    decimations. **Diagnose with `--metricstats`** (see below) before changing the
    metric.

- **`--metricstats` diagnostic (implemented).** `--metricstats` reports the metric
  distribution over *every* trial (not just those above threshold), for every
  harmonic decimation, without perturbing the candidate results (verified
  byte-identical, and read-only by construction). Two complementary views are
  collected into a `MetricStats` sink (`search(...; metricstats=ms)`):
  - **Per-`k` histograms** (`MetricHistogram`, one streaming pass, bounded memory:
    a fixed `[lo,hi)` linear histogram plus over/underflow and exact
    `total/sum/sumsq/min/max` accumulators). These give the *exact* global
    moments and *empirical* per-`k` quantiles (`hist_quantile`) — hence per-`k`
    false-alarm thresholds. The `stderr` summary tabulates, for each `k`, the
    metric value at single-trial FAP = 1e-1 … 1e-5, which is the directly
    actionable view: on `PM0063…red.fft` (5–30 Hz) the FAP=1e-4 threshold runs
    9.78 (k=1, 120 bins) → 5.91 (k=6, 20 bins), so a single `--threshold` picks a
    wildly different false-alarm rate per decimation. The histograms are written
    to `<stem>_metrichist.txt` for offline fitting. (The default range `[0,64)` is
    sized for a *normalised* FFT; a signal-/RFI-dominated or un-normalised input
    overflows it, which the summary flags — moments stay exact, only quantiles are
    range-limited. Range/resolution are `MetricStats` keyword args.)
  - **Per-block, per-decimation stats** (`BlockMetricStats`: min/median/mean/std/max
    per processed block) written to `<stem>_metricstats.txt`, with the per-block
    `ngoodbins` and searched frequency range so the frequency dependence of the
    floor (red-noise excess at low `f`, the Nyquist `ngoodbins` rolloff at high
    `f`) is visible.
  Collection allocates only per-task buffers/histograms and is off by default.
  - **Frequency-windowed histograms (implemented).** The band is now split into
    `nwin` log-spaced *searched-spin-frequency* windows per `k` (each `k`'s band
    is `k×` the base band; `MetricStats.nwin`, default 16), giving a
    `MetricHistogram` per `(k, window)` (`ms.whists`, and the band-wide per-`k`
    `ms.hists` are just their merge; `metricstats_windows` tabulates the
    per-window rows). Each block, being narrow, is assigned whole to the window
    of its centre frequency, so windowing costs one `searchsortedlast` per
    `(block, k)` — nothing per trial. This resolves the frequency dependence the
    band-wide histogram averages over: on `PM0063…red.fft` (0.5–50 Hz, `:non`,
    k=1) the empirical FAP=1e-4 threshold runs ~12.5 at 0.5–0.9 Hz (red-noise
    residual in the tail) → ~9.7 mid-band → 8.2 at 37–50 Hz, and the top window's
    mean drops as the `ngoodbins` Nyquist rolloff sets in (only ~20 of 60
    harmonics fit below Nyquist at 50 Hz). Written per `(k, window)` to
    `<stem>_metricfap.txt` (thresholds) and `<stem>_metrichist.txt` (raw
    histograms); the `stderr` summary adds a FAP=1e-4-vs-frequency drift line per
    `k`. These per-`(k, f)` empirical quantiles are exactly the substrate the
    dynamic normalisation path needs.
  The per-`(k, f)` normalisation is now wired into detection via `--normalize`
  (see the threshold-calibration item below); a pure-noise-simulation calibration
  to give the normalised significance an absolute equivalent-σ meaning is the
  remaining step.

- **Threshold-calibration plan — hybrid; in-situ half implemented.** The agreed
  direction: (1) *dynamic, in-situ* per-`(k, frequency)` normalisation, measured
  from the search data itself so it absorbs the real data's normalisation,
  red-noise residual, and Nyquist rolloff that a static table cannot know;
  (2) *offline pure-noise simulation* to give that normalised statistic an
  absolute FAP/equivalent-σ (trials factor folded in) and validate the in-situ
  estimator against ideal noise.
  - **(1) `--normalize` (implemented).** A two-pass search: pass 1 measures the
    per-`(k, frequency window)` noise (the `--metricstats` machinery), pass 2
    builds a [`MetricNorm`](@ref) and thresholds on the normalised significance
    `z = (M − loc)/scale` instead of the raw metric (recording `z` as the
    candidate metric, which also makes the cross-`k` `remove_harmonics` ranking
    comparable). `loc` is the window's noise median and `scale` its upper-side
    robust spread `q(0.8413) − median` (Gaussian-calibrated, taken from the
    noise bulk so tail signals/RFI don't bias it), with a per-`k` band-wide
    fallback for sparse/degenerate windows. Verified on `PM0063…red.fft`
    (5–30 Hz, threshold 6): raw gives ~100 candidates dominated by one
    decimation (94/100 at `Hk=30`), while `--normalize` gives 6 spanning `k =
    1,4,5,6` — the `√nbins` + frequency flood is gone — with the true 7.1187 Hz
    pulsar still ranked first. Assumes a normalised input.
    *Limitation:* `z` is only a true equivalent-σ where the noise is Gaussian;
    the right-skewed metric makes `z` an over-estimate deep in the tail, so a
    fixed `z` threshold is *comparable* across `(k, f)` but not yet an absolute
    σ — that is what (2) fixes.
    - **The ~2× runtime penalty is *definitely* not acceptable long-term** and
      must be worked on — running the entire interpolate/profile/metric pipeline
      twice, just so pass 2 knows the pass-1 noise floor, is the wrong shape. The
      intended fix, once (2) exists: use the **absolute calibration as the base
      `loc`/`scale`** (a function of `nbins`/`ngoodbins`, i.e. of `k` and
      frequency, from the simulation + semi-analytic Nyquist rolloff) and only
      *perturb* it with a cheap in-situ measurement — so no second full pass is
      needed. The perturbation could come from a **sub-sampled** measuring pass,
      or ideally from the **current block's own statistics** in a *single* pass
      (normalise each trial against its block's measured median/scale, computed
      from the profiles already in hand — no re-interpolation, no re-`irfft`).
      That likely wants a **larger `blocksize`** so each block holds enough
      trials for a stable per-block median/scale (and enough tail for the deep
      quantile the threshold needs); the block would then be the natural
      frequency window, superseding the separate windowing. The base calibration
      keeps the per-block estimate honest where a block is signal-/RFI-heavy or
      too short. This is the preferred end state: single-pass, self-calibrating,
      no 2× tax.
  - **(2) pure-noise simulation (next).** Fit the noise distribution's absolute
    FAP-vs-`z` tail from Monte-Carlo pure noise, handling the `ngoodbins` Nyquist
    rolloff semi-analytically (it enters the metric only through
    `invrms = √(2·ngoodbins+1)` plus the reduced harmonic count), then map `z` to
    a true equivalent-σ and validate that the in-situ `loc`/`scale` match ideal
    noise. The `--metricstats` per-`(k, f)` histograms are the validation data.
    Besides the absolute σ, this yields the **base `loc`/`scale` numbers** the
    single-pass scheme above needs to escape the 2× penalty.

- **Default metric produces many non-pulsar-like false positives (to
  investigate; may change defaults).** On real data the current defaults
  `--metric non --pexp 0.5` empirically generate *many* more false-positive
  candidates than `--metric sd2` at a comparable threshold. Crucially, a large
  fraction of the `non` false positives are not merely marginal — their
  reconstructed profiles (now easy to eyeball via the candidate profile plots)
  look like **random noise**, with no narrow, low-duty-cycle pulse of the kind
  most real pulsars show. In other words the `N_on^p` duty-cycle penalty at
  `pexp=0.5` appears to let broad, noise-like profiles through too readily. This
  is distinct from (but entangled with) the threshold-calibration item above:
  even at a fixed false-alarm *rate*, the *character* of the survivors differs
  between penalties. Action items: (1) quantify the false-positive rate and the
  profile "pulsar-likeness" of survivors for `non` vs `sd2` across `pexp` on
  pure-noise and real data; (2) reconsider whether the shipped defaults should
  move to `sd2` and/or a larger `pexp` (a stronger width penalty suppresses
  broad/noise-like profiles); (3) consider an explicit profile-shape / narrowness
  discriminant as a post-detection cut. Until this is settled the defaults are
  provisional — `sd2` is worth trying on real searches.

- **Cheap multi-frequency search by harmonic decimation (implemented).** Starting
  from a large, composite `nharms` (default 60 when enabled), the full harmonic
  amplitude stack for each fundamental is re-used to fold at 2×, 3×, … that
  frequency *almost for free*: taking every `k`-th interpolated harmonic and
  running a shorter batched `irfft` yields the fold at `k·rf` with
  `Hₖ = ⌊nharms/k⌋` harmonics. Enabled with `--maxdecim k` (default 1 = off);
  each candidate now reports its frequency, **period** (`1/f`), and the number of
  harmonics summed — which identifies the decimation (`k = nharms ÷ nharm`). The
  full bookkeeping and the derivation that decimation stays *correctly sampled*
  (each `k`'s top harmonic still steps by ≤ `hidr`, and the base input-FFT read
  depth already covers every `k`) live in `decimation_design.md`. Two properties
  fell out cleanly: the caveats about re-striding the input FFT / `deltar` /
  `numbetween` turned out **not** to bite (top-harmonic sampling and read depth
  are preserved automatically), and cross-`k` detections of the *same* frequency
  share an `r`, so the existing near-identical `remove_duplicates` already
  collapses them. Guarded by a machine-precision test that each decimation pass
  reproduces the *native* `Hₖ`-harmonic fold (transitively oracle-pinned via
  `reference_profiles`) plus a detection test recovering the bundled 10.0123 Hz
  pulsar via `k=2` and `k=3`. The `f`, `f/2`, `3f/2`, … family that decimation
  makes prominent (its subharmonic folds report genuinely different `r`, so the
  near-identical dedup does not touch them) is now collapsed by the
  **harmonically-related de-duplication** above. *Still open:* threshold
  comparability now has `Hₖ` as an extra axis alongside `--metric`/`--pexp`.

- **Profile plots for the best candidates (implemented).** For the reported
  survivors, `candidate_profile` (`src/candidate.jl`) reconstructs the actual
  pulse profile by the brute-force, high-accuracy path anticipated here: one wide
  (`m=64`) exact `fourier_interp` per harmonic at the candidate's exact
  frequencies (`r·h`), packed into a harmonic stack and inverted with a plain
  `irfft` — no throughput-tuned approximation, since it runs on only a handful of
  candidates. It is pinned to the search's independent `reference_profiles` path
  (matched kernel `m`, fine grid) to ~1e-4, guarding indexing/FFT convention.
  Each profile is folded at the **full `--nharms` depth** regardless of the
  decimation factor `k` that found the candidate (a `k=3` detection summed only
  `⌊nharms/k⌋` harmonics; its profile still uses all `nharms`), so it much more
  closely matches a true time-domain fold at the candidate period. Harmonics that
  would cross the Nyquist frequency are omitted rather than zero-padded — the fold
  stops at the first such harmonic and inverts the `H ≤ nharms` available ones to
  `2H` bins — so fast candidates simply get fewer bins. `rotate_to_peak`
  circularly shifts each profile so its peak sits at phase 0.5. The
  `CandidatePlots` helper (`bin/plotting.jl`, CairoMakie) lays the profiles out in
  a `ncols×nrows` grid (default 3×5) on US-Letter portrait pages, written one PNG
  per page (`<stem>_NN.png`, zero-padded so pages sort) with the full grid
  geometry reserved even on a partly filled last page (so every panel is the same
  size), each panel captioned with the full candidate text-line (index, S/N,
  frequency, period, harmonic count, decimation `k`) and each page with a metadata
  banner. Plotting runs by default from the CLI (`--noplot` disables,
  `--plotstem/--plotcols/--plotrows` configure) and can be regenerated later from
  a saved candidate file with `bin/plot_candidates.jl`. CairoMakie is a project
  dependency but is loaded *lazily* — only `bin/plotting.jl` imports it — so
  `using CoherentSearch`, `Pkg.test`, and the cross-validation never pay for it.
  *Still worth doing:* optionally overlay the metric's measured on-pulse width /
  baseline.

- **`Distributed.jl` backend** reusing the same chunk abstraction, for
  cluster-scale searches across nodes.
- **Broader real-data validation** beyond the single artificial test pulsar.

---

## 4. Summary

Phase 1 delivered a correct, parallel, well-tested foundation whose numerical
results are pinned to the Python implementation at machine precision. Phase 2
turned that foundation into performance: a chunk-parallel driver with one
private workspace per task, FFTW plans and interpolation kernels built once and
reused, a batched inverse FFT for the profiles, and per-harmonic interpolation
tuning that sharpens the low harmonics. The hot loop is allocation-free, the
search scales ~3.3× on 4 cores, and — guarded the whole way by the Python oracle
and an `align=false` equivalence test — the results are provably unchanged.
