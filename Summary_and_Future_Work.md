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

### Re-baseline under the `:boxcar` default + `--maxdecim 6` (2026-07-21)

The 2026-07-09 split above was taken on `--metric sd2`. Two later changes — making
`:boxcar` the default detection metric and `--maxdecim 6` the standard config —
reshaped the hot loop enough to warrant a fresh profile. The `bench/` harness was
updated to match (`metric=:boxcar`, a clean **5–30 Hz** mid-band that avoids the
low-frequency red-noise candidate flood, and boxcar-aware buckets:
`_profile_boxcar`, `_block_sigma`). The self-time aggregator was also fixed to
attribute Base leaf frames (`getindex`/`setindex!`/`*`) to their **nearest
enclosing `search.jl` frame** rather than dumping them into an uninformative ~56%
"other" — the boxcar width-scan and the interp `spec.*coeffs` multiply both inline
down to Base arithmetic, so a leaf-only classifier mis-attributes them.

Single-thread, warm, `PM0063…red.fft` 5–30 Hz, `nharms=60`, six decimations
(68.6 s, 22 candidates, 150k samples):

| bucket | self-time |
|--------|-----------|
| FFTW | 32.5% |
| median-select (`_median!`/`_select!`) | 28.4% |
| boxcar-metric (prefix-sum + width×phase scan) | 24.9% |
| interp (`spec.*coeffs` multiply + gather) | 6.8% |
| decim (gather + short brfft) | 3.3% |
| `uniform_linear_interp` | 2.4% |
| block-sigma (non-median part of `_block_sigma`) | 1.3% |
| other | 0.3% |

Grouped: **interpolation/FFT pipeline ≈ 45%** (FFTW + interp + decim + uniform),
**detection metric ≈ 55%** (median + boxcar-scan + block-sigma). **This inverts
the old baseline**, where interp/FFT dominated and the metric was ~12%. Two new
costs, both consequences of the `:boxcar` default under decimation:

- **Per-profile baseline median = 28.4% (now the largest non-FFT bucket).**
  `:boxcar` subtracts a per-profile median baseline from *every* profile, and it
  runs that at *every* one of the seven decimation passes (`k=1…6`). Decimation is
  "cheap multi-frequency" only on the *interpolation* side — the 60-harmonic stack
  is interpolated once per chunk and re-folded — but each `k` pays the **full
  metric cost** on its own profiles. So the metric's share grows with `maxdecim`
  while the interp's does not; at the default `maxdecim 6` the metric already wins.
- **Boxcar width×phase scan = 24.9%.** `_profile_boxcar`'s inner loop is an
  `O(nbins × nwidths)` strided max-reduction over the prefix sum (e.g. 120 phases ×
  9 widths at `k=1`) — pure sequential FP, and it did not exist in the old profile.

**Reprioritised levers (supersedes the ComplexF32-first ordering in §3):**

1. **`@turbo` the boxcar width×phase scan (24.9%, easy, low-risk).** The inner
   loop is a strided max-reduction — ideal for `LoopVectorization` (which works on
   1.12). `max` is exactly associative, so vectorising it returns the identical
   `Float64` and the `align=false` machine-precision pin holds trivially. Plausible
   2–4× on a quarter of the runtime. **Best first move.**
2. **Cut the per-profile median (28.4%, medium).** The `_median!` quickselect is
   already tuned, but boxcar now calls it ~7× more (once per profile per
   decimation). Options: a branchless small-`n` median network for the handful of
   common `nbins` (20/24/30/40/60/120 — `SortingNetworks.jl` is broken on 1.12, so
   hand-rolled), or SIMD selection. Note the baseline is intrinsically per-profile
   (each profile has its own DC level), so it cannot be pooled per-block the way
   `_block_sigma` pools σ without changing results.
3. **`ComplexF32` interpolation (still the biggest *single* bucket at 32.5%, but no
   longer the top lever).** Same precision-mode caveat as before; now it addresses
   ~⅓ of runtime, not the old ~⅔.

The takeaway: with decimation as the default, **detection-metric cost scales with
`maxdecim` while interpolation is amortised**, so metric optimisation is the
highest-leverage work — the opposite of the pre-decimation conclusion.

### Metric optimisations landed (2026-07-21) — 68.6 s → 45.5 s (1.51×)

Both metric levers above were implemented, each validated by the full test suite
and a byte-identical candidate-file diff:

1. **Vectorised the boxcar width×phase scan.** Restructured `_profile_boxcar`'s
   inner loop to a per-width max-reduction over the strided prefix-sum difference
   `psum[p+w]−psum[p]` (the monotone `invsw` lifts out: `max_p(dₚ·invsw) =
   (max_p dₚ)·invsw`). **Plain `@simd` + `max` auto-vectorises** (verified IR:
   `vector.body`, `<N×double>` loads, vectorised `fmax` reduce) — **no
   `LoopVectorization` dependency**, so start-up/precompile cost is unchanged.
   Result-preserving to the bit. Bucket 24.9% → 18.5%, ~5% end-to-end.
2. **Made the per-profile median mostly disappear.** Two parts:
   - **Sorting-network median for short profiles.** A Batcher odd-even mergesort
     network (branch-free `min`/`max` compare-exchanges) beats quickselect for the
     small `nbins` decimation produces — measured ~2.0× at n=20, ~1.75× at n=30,
     ~1.28× at n=60, crossing to a loss by n=120 — so it is used for `nbins ≤
     _MED_NET_MAX (=64)` and quickselect for the base pass. A full sort's two
     central order statistics are identical to quickselect's, so the value is
     bit-for-bit unchanged.
   - **Adaptive zero-baseline gate (the big one).** The profile spectrum's DC bin
     is held at zero, so **every profile's mean is exactly 0** (verified `|mean|/σ
     ≈ 5e-17`). The median only departs from 0 when a real pulse skews the profile,
     and the baseline error propagates into the metric as `−δ·√w/σ` — negligible
     for the narrow pulses we target, largest for wide boxcars. So `_profile_boxcar`
     first scans against a **zero** baseline (no median); because a positive pulse
     has median ≤ 0, that `m₀` is a lower bound on the true metric, with the gap
     bounded by `|med|·√wₘₐₓ/σ` (empirically ≤ 1.23). Only if `m₀ ≥ threshold −
     boxcar_medmargin` (default 2.0) does it pay for the exact median and rescan.
     On real data this computes the median for only ~1% of trials (the rest are
     pure noise, safely below threshold) while keeping every candidate exact. The
     metricstats/normalize/reference paths force the exact median (`medcut = −∞`).
   Together: **median-select 31.2% → 5.5%** (≈20.4 s → 2.5 s, ~8×), full run
   **65.3 s → 45.5 s**.

**New split (5–30 Hz, single-thread, 45.5 s):** FFTW **49.7%**, boxcar-metric
23.6%, interp 10.1%, median-select 5.5%, decim 4.9%, uniform 3.7%, block-σ 1.9%.
Grouped: interp/FFT ≈ 68%, metric ≈ 31% — the balance has **flipped back**, and
**FFTW is now unambiguously the top cost.** The next lever is therefore the
`ComplexF32` interpolation (§3), which Scott notes is precision-safe (PRESTO
interpolates at `ComplexF32`), so the work there is re-pinning tests, not a
numerical risk. *Caveat on the gate:* `boxcar_medmargin` is the safety dial — it
must exceed `|med|·√wₘₐₓ/σ` for every threshold-crossing trial; 2.0 clears the
observed 1.23 with headroom, but widen it if broader real-data validation ever
shows a near-threshold broad-signal miss.

### Direct `O(m)` interpolation replaces the FFT correlation (2026-08-08)

The 2026-07-21 re-baseline left **FFTW at 49.7%** and named `ComplexF32`
interpolation as the next lever. Profiling the FFT bucket properly first turned
up something better, and it retires three earlier conclusions at once.

**Where the FFT time actually goes.** Of `fill_chunk_profiles!` (9.9 ms per
2048-trial chunk, `nharms=60`, single thread), the 60 per-harmonic interpolation
transforms are **76%** and the base batched profile `brfft` only **4%** (the
other six `brfft`s, one per decimation pass, run outside the chunk fill). So the
transform cost is overwhelmingly interpolation, not folding — which is why
attacking the interpolation is what moves the FFTW bucket.

**The kernel's coefficients are a real vector times a complex scalar.** With
`offsets = dr - j` and integer `j`, the `(-1)^j` in `sin(pi(dr-j))` cancels the
one in `cispi(dr-j)`, so Eqn. 30's coefficients collapse to

    coeff_j = sinc(dr-j)*cispi(dr-j) = A(dr)/(dr - j),   A = sin(pi dr) cispi(dr)/pi

and since `fourier_interp` forms `sum conj(coeff_j) bin_j` with `1/(dr-j)` real,

    amp(r) = conj(A(dr)) * sum_j  bin_j / (dr - j)

One point is `m` **real**-times-complex FMAs (4 flops), not complex ones (8), it
reads only `m` consecutive bins (L1-resident), and it vectorises cleanly on
de-interleaved real/imaginary planes. Verified against `fourier_interp` at
4e-16.

**Only `q = 2*nharms` distinct `dr` values occur in a whole search.** Global
trial `t` sits at `r_t = r_lo + t*pnum/q` for `lodr = pnum/q` (exactly `1/120` at
the defaults), so harmonic `h` sees `frac(h*r_lo) + i/q` and nothing else — for
every chunk, forever. The `m` reciprocals and `A(dr)` are therefore tabulated
**once per harmonic at plan time** and indexed by an integer residue that
advances by a fixed step per trial. The inner loop has no transcendentals, no
divisions, and no `mod`. Both the residue and the integer bin index are carried
in exact integer arithmetic, so there is no drift across a long search and the
"trial lands exactly on a bin" corner is handled explicitly rather than by
`floor(r + 1e-15)`.

**Measured** (`PM0063…red.fft`, `nharms=60`, `Nprof=2048`, single thread), summed
over the 60 harmonic rows of one chunk: FFT-correlation + linear interpolation
**7.77 ms** → direct with the phase table **1.32 ms**, i.e. **~5.9x**.
`bench/interp_bench.jl` measures the two as pure throughput: at the production
settings **57–74 M points/s** for direct+table against **6–7.3 M points/s** for
FFT+linear, ~9x.

**End to end** (5–30 Hz, `--maxdecim 6`, `:boxcar`, `--threshold 6`, single
thread, warm wisdom, wall clock including ~2 s of Julia start-up):

| interpolator | wall | candidates |
|---|---|---|
| `--interp fft` | 70.1 s | 159 above threshold → 22 after dedup |
| `--interp direct` | **42.7 s** | 159 above threshold → 22 after dedup |

**1.64x** overall (1.68x excluding start-up). The candidate lists match: the same
22 survivors at the same frequencies and harmonic counts, metrics differing by
~1% (the linear-interpolation error being removed), with only a couple of
rank swaps among near-ties.

**Was the `N log N` intuition wrong?  No — it just never applies here.**
`bench/interp_bench.jl` separates the two effects. Counting only the points the
FFT actually produces, FFT correlation **beats** direct summation, and by more as
the kernel widens — exactly the reasoning the FFT-correlation design was built
on:

| m | FFT, per *grid* point | direct+table, per point | |
|---|---|---|---|
| 8 | 82.0 | 69.3 | FFT 1.18x |
| 16 | 82.7 | 69.0 | FFT 1.20x |
| 32 | 79.9 | 53.4 | FFT 1.50x |
| 64 | 66.1 | 48.0 | FFT 1.38x |
| 128 | 68.2 | 25.3 | FFT 2.69x |

(M points/s, `N=2048`, single thread.) What the search cannot do is *use* those
points. The fine grid has to be `numbetween` times finer than the trial spacing
so the linear interpolation that follows is accurate — not merely fine enough to
contain the trials — so at the production settings **7 of every 8 grid points are
computed and discarded**, and the linear interpolation that reads the eighth is
itself not free. Counted per point actually wanted:

| m | FFT+linear | direct+table | |
|---|---|---|---|
| 8 | 6.9 | 53.0 | direct 7.7x |
| 16 | 7.3 | 80.8 | direct 11.0x |
| 32 | 6.5 | 60.3 | direct 9.3x |
| 64 | 5.9 | 41.4 | direct 7.1x |
| 128 | 5.7 | 26.4 | direct 4.6x |

The oversampling is not a tuning mistake that could be dialled away, either: the
fine grid is anchored to integer Fourier bins while the trials sit at an
arbitrary sub-bin offset `frac(r0)`, so the two only coincide if the kernel is
rebuilt per chunk to absorb that offset — a third transform per harmonic per
chunk, which costs more than it saves. Direct summation has no grid to align.

**And it removes an approximation that was costing real sensitivity.** The FFT
path's final linear interpolation between fine-grid points is wrong by

| h | nb | max rel. amplitude error vs the exact kernel |
|---|----|---------------------------------------------|
| 1 | 120 | 8.4e-4 |
| 8 | 16 | 2.7e-2 |
| 30 | 16 | 6.1e-2 |
| 60 | 16 | 5.0e-2 |

so the coherent sum has been adding 60 harmonics whose amplitudes are up to ~5%
off. The direct path matches the exact kernel at ~1e-10 (1e-15 where `dr` is
exactly representable). This is the "wart" the `numbetween`/`align` machinery
was managing rather than removing.

**What this retires.**

- **The `ComplexF32` interpolation item is moot, and this was measured, not
  assumed.** Its purpose was to halve the bandwidth of transforms that no longer
  run. The direct path already reads the mmap's `ComplexF32` bins natively, so
  the only remaining question was whether to carry the *weights* and the
  *accumulator* in `Float32` too. Measured over the 60 harmonic rows:

  | inner loop | time | vs `Float64` | error vs `Float64` |
  |---|---|---|---|
  | `Float64` planes + weights | 1.32 ms | — | — |
  | `Float32` planes, `Float64` weights/accumulator | 1.20 ms | 1.11x | **0** (bit-identical) |
  | `Float32` throughout | 0.98 ms | 1.35x | 2.1e-7 |

  The middle row is free — `Float32 → Float64` is exact, so the sum is
  bit-identical while reading half the bytes — and is now what the `Workspace`
  planes use. Full `Float32` buys only a further 1.2x on ~20% of runtime (~3%
  end-to-end) in exchange for giving up 9 digits, because the loop is
  FMA-throughput-bound rather than bandwidth-bound. **Not worth it; not
  implemented.** That closes the "biggest remaining lever" item as a
  measurement rather than a project.
- **HPK (and the `fft_tests/HPK_JULIA_HANDOFF.md` investigation) drops in
  priority.** Its §7 prerequisite #1 was "profile the FFT length histogram and
  the true FFT fraction"; done, and FFTW falls from **49.7% to 12.9%** — what is
  left is the batched profile `brfft`, once per decimation pass, on short
  (`2·Hₖ`-point) transforms with a 2048-wide batch. A 1.5x there is ~4%
  end-to-end, which does not justify a proprietary binary-only dependency, a C++
  shim, the GPL-vs-no-redistribution tension, or the AUI/NRAO legal review — and
  batched short real transforms are not the ground HPK's published wins are on
  (§4 of the handoff is 1-D complex, batch 1). Keep the handoff document; revisit
  only if the direct path is ever backed out. KFR stays ruled out on speed.
- **"`fftlen` sizing is settled — do not revisit" was wrong**, and is fixed
  below.

### Smooth `fftlen` sizing (2026-08-08) — reopened, measured in situ, re-rejected

The §2 note above rejected smooth `2·3·5·7` lengths on two grounds: a "~3%"
ceiling, and `MEASURE` planning ~60 lengths at 0.7 s each. The second ground is
genuinely dead — the wisdom persistence landed later the same month means those
plans are measured once, ever. The first deserved re-measuring, because the real
argument for smooth sizing was never "mixed-radix beats radix-2 per point": it is
that **we choose the length**, and `next_pow_of_2` throws away a **mean 1.38x
(worst 1.98x)** in transform length across the harmonic schedule, which is pure
loss.

**Per-transform, that argument holds.** Re-measured in Julia at `ComplexF64` with
`MEASURE`, summed over the distinct lengths the search asks for, smooth is
**1.26x** faster.

**In the actual search, it does not.** Same band and config as the table above,
`--interp fft`, single thread, wall clock:

| sizing | distinct lengths | mean padding | wall |
|---|---|---|---|
| `:pow2` | 4 | 1.382 | **66.6 s** |
| `:smooth`, merge tol 1.25 | 8 | 1.115 | 66.2 s |
| `:smooth`, merge tol 1.10 | 16 | 1.047 | 67.8 s |
| `:smooth`, merge tol 1.60 | 4 | 1.259 | 71.6 s |
| `:smooth`, merge tol 2.50 | 2 | 1.582 | 76.6 s |

So the best smooth configuration is within noise of `:pow2` and the rest are
worse — a **wash at best**, against a claimed 1.26x. **`:pow2` stays the
default.** Two reasons the isolated benchmark overstates the case, both of which
it is structurally unable to see:

- **Smooth lengths are not uniformly good.** Compare rows 1 and 4: the *same*
  number of distinct lengths and *less* padding, yet 7.5% slower. Per-size ratios
  against the power of two ranged from **0.69x to 2.25x** in the isolated
  measurement — the `3^k`-heavy lengths (6561, 13122, 15309) are markedly *worse*
  than padding to a power of two — and "the smallest smooth number ≥ what I need"
  sometimes lands on exactly those. Summing over distinct sizes hides that a bad
  pick can be assigned to a harmonic that runs every chunk.
- **It times transforms whose buffers are already resident.** In the search they
  are not: `:pow2` gives 60 harmonics only **4** `FFTScratch` buffer sets that
  stay warm across a chunk, while a tight smooth schedule spreads them over 8–16
  sets and 6 MB. Padding and buffer reuse pull against each other, which is why
  the sweep is not monotonic in either.

**The old verdict was right; its stated reasons were not.** That distinction is
worth keeping, because the reason it gave ("`next_pow_of_2` + `MEASURE` is
already ≈ FFTW's best case") is false — the padding really is wasted — and would
have discouraged the measurement that produced the useful finding.

`SearchParams.fftsizing` keeps `:smooth` available (`--fftsizing smooth`), with
lengths within `FFTLEN_MERGE_TOL` (1.10) merged upward so it cannot explode the
plan and buffer count on a many-core node. It only affects `--interp fft`.

### New profile (2026-08-08, `:direct` default)

Same warm single-thread sampling profile as the 2026-07-21 baseline (5–30 Hz,
`nharms=60`, six decimations, `:boxcar`), with a bucket for the direct
interpolator added to `bench/profile_search.jl`:

| bucket | self-time |
|--------|-----------|
| boxcar-metric (prefix-sum + width×phase scan) | 44.6% |
| interp (direct `O(m)`) | 18.7% |
| FFTW (the batched profile `brfft`s only) | 12.9% |
| median-select | 10.1% |
| decim (gather + short brfft) | 9.1% |
| block-sigma | 3.7% |
| fill-chunk (other) / other | 0.8% |

Grouped: **detection metric ≈ 58%, interpolation/FFT pipeline ≈ 41%** — the
balance has flipped back to the metric, as it was before the 2026-07-21 metric
work, and for the same structural reason: **metric cost scales with `maxdecim`
while interpolation is amortised across the decimations.** The next lever is
therefore the boxcar width×phase scan again (44.6%, already `@simd`-vectorised
once), not anything on the FFT side.

### `--verbose`: the interpolation plan is now inspectable

`harmonic_plan_report` (CLI `--verbose`) prints, per harmonic: `nb`, `m`,
`deltar_h`, the input bins the chunk spans, the fine-grid length actually needed,
the `fftlen` chosen and its padding factor, the fine-grid oversampling
`nb*deltar_h`, and whether linear interpolation is required (`no` under
`:direct`; `yes(fixed)` when the trial step is commensurate with the grid so one
constant weight pair serves every trial; `yes` otherwise). The summary totals
fine-grid points computed against trials wanted — which is how the 8x
oversampling at high harmonics became visible in the first place.

---

## 3. Next steps

> **Status: feature-complete.** The search, detection metric, candidate
> de-duplication, harmonic decimation, and candidate profile plots are all
> implemented, tested, and oracle-validated. The primary focus now shifts from
> features to **performance**: careful profiling of the hot loop (interpolation,
> batched inverse FFTs, allocation and memory-bandwidth behavior under threading)
> and acting on what it finds. The throughput-tuning and tiling items below are
> the concrete starting points for that work.

- **Metric cost dominates again — 58% (see the 2026-08-08 profile in §2).** With
  the direct interpolator in, the split is boxcar-metric 44.6%, direct interp
  18.7%, FFTW 12.9%, median-select 10.1%, decim 9.1%, block-σ 3.7%. The
  highest-leverage item is the **boxcar width×phase scan (44.6%)**, which has
  already had one `@simd` pass; it scales with `maxdecim` while interpolation is
  amortised, so it will keep growing relative to everything else. The three
  FFT-side bullets that used to live here are done or closed:
  - *`ComplexF32` interpolation* — **closed by measurement, not implemented.**
    The direct path reads `ComplexF32` bins natively and its planes are now
    `Float32` (bit-identically); carrying the weights and accumulator in
    `Float32` too buys 1.2x on ~19% of runtime for 9 digits. See §2.
  - *Rethink FFT-correlation vs. direct interpolation* — **done**, and it was the
    large win the bullet hoped for (1.64x end-to-end). See §2.
  - *`fftlen` sizing* — reopened, re-measured in situ, and re-rejected; `:pow2`
    stays the default but for different reasons than originally recorded. See §2.
    Tiling a chunk into smaller overlapping transforms remains unmotivated.

- **Kadane's algorithm as a fast (approximate) boxcar metric — to investigate.**
  The boxcar width×phase scan is now the single largest bucket (44.6%), and it is
  an `O(nbins × nwidths)` search over a *fixed, a-priori* width bank (9 widths at
  `nbins=120`, falling to 5 at `nbins=20` under decimation). The [`loki`
  code](/home/sransom/git/loki) — a new Fast Folding Algorithm implementation
  that has the same "score a huge batch of folded profiles with boxcars" problem
  — replaces that scan with a **maximum-subarray (Kadane) sweep**, which finds the
  best contiguous run of bins in a *single* `O(nbins)` pass with no prefix sum.
  Worth understanding and probably worth trying. Its mechanics
  (`lib/kadane.cpp`, `include/loki/detection/kadane.hpp`):

  - **Bias-subtracted Kadane.** Plain max-subarray always prefers longer runs,
    because the `1/√w` normalisation is not inside the recurrence. Running Kadane
    on `x_j − mean − b` instead makes the constant `b` a per-bin length penalty,
    so the maximising window is a proxy for the best `w`. A small bank of biases
    then spans a range of characteristic widths — loki ships **three**
    (`1.42, 0.76, 0.41`, `lib/configs.cpp`), all updated in one pass over the
    bins, versus our 5–9 widths.
  - **Wrapped pulses come free from a simultaneous min-subarray.** The complement
    of the minimum-sum window is the maximum-sum *circular* window, so tracking
    both in the same sweep handles a pulse straddling phase 0 — which our scan
    handles by phase-tiling the prefix sum instead.
  - **Different normalisation.** loki scores
    `sum · √(nbins / (w·(nbins−w)))` rather than our `sum/(σ·√w)`; that is the
    right factor when the baseline is estimated from the *same* profile, and it
    is worth adopting (or at least understanding) independently of Kadane.
  - **Vectorised across profiles, not within one.** loki transposes a batch of
    profiles and runs the sweep SIMD-wide over the batch. **This may matter more
    than Kadane itself for us:** our scan vectorises along phase *within* one
    profile, and under decimation `nbins` falls to 20, where the inner loops are
    far too short to fill a vector. We already hold `Nprof = 2048` profiles in a
    contiguous `(nbins, Nprof)` array, so the batch axis is right there.

  **The catch, and the reason this is an "approximation".** The window width
  Kadane selects is *data-dependent*, whereas the entire justification for the
  `:boxcar` metric (§2) is that its widths are fixed a priori, making each
  (phase, width) trial exactly `N(0,1)` with an analytic, `nbins`-flat trials
  factor. A data-selected width reintroduces precisely the selection bias that
  made `:non`'s noise floor scale as `√nbins` — so a Kadane-scored metric would
  need its own threshold calibration, and the flat-across-decimations property
  that `:boxcar` was adopted for cannot be assumed to survive. Also, with a finite
  bias bank the recovered window is not guaranteed optimal, so the score is a
  *lower bound* on the true best-boxcar S/N.

  **That lower-bound property is exactly what makes it attractive here, though,**
  because `_profile_boxcar` already has a two-tier structure built for it: the
  adaptive zero-baseline gate computes a cheap lower bound and only pays for the
  exact version when the trial comes within `boxcar_medmargin` of threshold. A
  Kadane sweep could slot in as (or ahead of) that cheap stage, with the exact
  fixed-width scan run only on survivors — **which keeps the reported metric, its
  calibration and every candidate exactly as they are today**, and confines
  Kadane to deciding what not to bother scoring properly. That framing avoids the
  calibration problem entirely and is how I would try it first; a
  `--metric kadane` that reports the Kadane score directly is the more invasive
  option and should wait on a `--metricstats` study of its noise distribution.

  Concretely, in order: (1) port the bias-subtracted sweep (plus min-subarray for
  wrapping) and check on real data how often its score brackets the exact metric,
  and how tight the bound is — the gate needs a *guaranteed* margin, as
  `boxcar_medmargin` does; (2) microbenchmark it against `_profile_boxcar` at
  `nbins ∈ {20, 30, 60, 120}`, both as-is and vectorised across profiles, since
  the operation count only falls ~2–3× and the per-bin work is higher (several
  compare/selects per bias versus one subtract-and-max), so the win is not
  obvious on paper; (3) separately, try the cross-profile SIMD transpose on the
  *existing* exact scan — it may capture much of the benefit with no
  approximation and no calibration risk at all.

- **Start-up latency: persist FFTW wisdom (implemented, 2026-07-21).** A short
  search spent several seconds *before* the hot loop planning every distinct
  transform with `FFTW.MEASURE`, re-timed on every process start (building all
  plans for the standard `maxdecim 6` config measured at **4.3 s cold**). FFTW's
  plan cache is serialisable, so `search` now `import_wisdom!`s before planning and
  `export_wisdom!`s after (`src/wisdom.jl`; `wisdom=false` / `--nowisdom` disables,
  `--wisdomfile`/`$COHERENT_WISDOM` set the path, default per-host under the depot).
  With wisdom present the same planning is **13.7 ms (~315×)**. The default
  `MEASURE` path is **byte-identical** to cold planning (verified: candidate files
  match), so this is a free start-up win. `bin/prime_wisdom.jl` / `prime_wisdom`
  optionally do a one-time `FFTW.PATIENT` pass whose better plans a later `MEASURE`
  run reuses directly — *caveat:* PATIENT may pick a different algorithm than
  MEASURE, perturbing results at the ~1e-16 level (harmless for detection, but not
  bit-identical to a MEASURE run; the oracle pins' tolerances absorb it). The
  residual Julia load/precompile latency (~1 s) is the separate, heavier sysimage /
  `PackageCompiler` question, worth a look only if it becomes the bottleneck.
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
      alone, since it keys on *how many* bins are lit, not *where*. Was the
      default; superseded by `:boxcar` (see below).
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
    decimation. **This is now the default metric** (`SearchParams.metric` /
    `--metric boxcar`) and largely obviates `--normalize`'s motivation; the
    frequency (red-noise) drift is the only thing left for a per-`f` threshold, and
    it is now `k`-independent. `:non`/`:sd2` remain available. *Still worth doing:*
    injected-signal width/S/N validation, a semi-analytic trials-factor →
    equivalent-σ map (the analytic EVD makes this tractable now), and eventually
    retiring `:non`/`--normalize` once `:boxcar` is validated on more surveys.

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

- **`:non`/`:sd2` produce many non-pulsar-like false positives (largely
  superseded by the `:boxcar` default).** This item motivated the `:boxcar`
  switch and is mostly of historical interest now; re-evaluate it for `:boxcar`.
  On real data the *former* defaults `--metric non --pexp 0.5` empirically
  generate *many* more false-positive candidates than `--metric sd2` at a
  comparable threshold. Crucially, a large
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
