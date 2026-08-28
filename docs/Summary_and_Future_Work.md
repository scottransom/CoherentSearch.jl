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
  chunks of `Nprof` trial fundamentals (`blocksize`, default 2048 on the CPU —
  the GPU backend defaults to 65536, see `gpu_design.md` §4.13). Chunks are
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
     **(Superseded 2026-08-24: there is no median baseline any more, so the gate
     is the whole metric and the second pass is a `Float64` re-score. See §3.4.)**
   Together: **median-select 31.2% → 5.5%** (≈20.4 s → 2.5 s, ~8×), full run
   **65.3 s → 45.5 s**.

**New split (5–30 Hz, single-thread, 45.5 s):** FFTW **49.7%**, boxcar-metric
23.6%, interp 10.1%, median-select 5.5%, decim 4.9%, uniform 3.7%, block-σ 1.9%.
Grouped: interp/FFT ≈ 68%, metric ≈ 31% — the balance has **flipped back**, and
**FFTW is now unambiguously the top cost.** The next lever is therefore the
`ComplexF32` interpolation (§3), which Scott notes is precision-safe (PRESTO
interpolates at `ComplexF32`), so the work there is re-pinning tests, not a
numerical risk. *Caveat on the gate (superseded — see §3.4):* `boxcar_medmargin` was the safety dial — it
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

### Cross-profile SIMD on the boxcar gate (2026-08-09) — 1.26x end-to-end

The 44.6% boxcar bucket above was `@simd`-vectorised along **phase**, which is
the wrong axis. Phase is `nbins` long, and `nbins = 120` only for the base pass:
under decimation it falls to 20, where the prefix sum is a serial add chain, the
max-reduction cannot fill a vector, and the horizontal reduce at the end of each
of the 5–9 width iterations costs more than the reduction it closes. The **batch**
axis is the one that is always long — a chunk holds `Nprof = 2048` profiles,
already sitting contiguously as the columns of `profs`.

So `_boxcar_gate!` transposes a tile of `B` profiles into a `(B, nbins)` buffer
and runs the identical recurrence with `b` innermost. Every lane is a different
profile: the prefix sum's serial dependency becomes `B`-wide, and the phase scan
becomes a plain element-wise `max` with no horizontal reduce at all. Measured per
2048-profile chunk, summed over the six decimations (`bench/boxcar_bench.jl`):

| gate kernel | total | vs current |
|---|---|---|
| scalar per column, phase-SIMD, `Float64` (was) | 3457 µs | 1.00x |
| batched cross-profile, `Float64`, `B=32` | 1248 µs | 2.77x |
| batched cross-profile, `Float32`, `B=64` | 855 µs | 4.05x |

End to end (5–30 Hz, `nharms=60`, `maxdecim 6`, single thread, median of 3
alternating runs against a `HEAD` worktree): **41.7 s → 33.1 s, 1.26x**, with a
**byte-identical candidate file**.

**Two things this measurement establishes, both non-obvious:**

- **`B` must be a compile-time constant.** With `B` an ordinary runtime `Int` the
  inner loops do not unroll and batching is *slower than the scalar path* it
  replaces — 2119 µs vs 1179 µs at `nbins=120, B=4`. Only passing `Val(B)` turns
  it into a win. `bench/boxcar_bench.jl` keeps the runtime-`B` variant in the
  table specifically so this cannot be "simplified" away unnoticed.
- **The win is concentrated exactly where it should be.** It is smallest at the
  base `nbins=120` (2.2x), where phase-SIMD already worked, and largest at the
  decimated `nbins=20–30` (3.3–4.0x), where it never did. Since metric cost
  scales with `maxdecim` while interpolation is amortised, this grows with the
  standard config rather than shrinking.

**`Float32`, applied only where it is provably free.** The tile is `Float32`
while `profs` stays `Float64` — the conversion rides along in the transpose that
must happen anyway, and it doubles the AVX lane count over the whole gate. This
is sound *because it is a gate*: `_boxcar_gate!` computes the zero-baseline lower
bound that ~99% of trials return from, and any trial reaching `medcut` is
re-scored by the unchanged exact `Float64` path. Measured on real profiles the
`Float32` bound differs from the `Float64` one by at most **~3e-6** in metric
units (6e-7 on `PM0063…red.fft`, at `nbins` 120 and 20 alike; 2e-6 on the bundled
test file), against the `boxcar_medmargin = 2.0` slack the rescue already
reserves — a factor of ~1e6. It therefore cannot change which
trials get scored exactly, hence cannot change the candidate list, which the
byte-identical candidate file confirms. It does rely on the profile mean being 0
(DC is held at zero), which keeps the prefix sum from drifting away from the
boxcar sums it must resolve.

**The gate had no test; it does now.** (Names below predate 2026-08-24:
`boxcar_medmargin` → `boxcar_gatemargin`, `medcut` → `exactcut`, and the "exact
median path" is now just the `Float64` path — see §3.4.) The adaptive
zero-baseline gate landed
without an equivalence pin, so nothing was checking the property the whole
optimisation rests on. `test/test_search.jl` adds two: a full `search` against
one with `boxcar_medmargin = Inf` (which forces `medcut = -Inf`, i.e. the exact
median path for every trial) must produce byte-identical candidates; and the
batched gate is compared directly to the scalar gate on real profiles, with the
error required to stay three orders under `boxcar_medmargin`. Both cover the old
gate and the new batching.

**New profile, and it moves the target off the metric entirely.** Same warm
single-thread sampling profile (5–30 Hz, `nharms=60`, six decimations, 34k
samples):

| bucket | self-time | was (2026-08-08) |
|--------|-----------|------------------|
| interp (direct `O(m)`) | 22.5% | 18.7% |
| **boxcar-gate (batched)** | **20.0%** | — |
| FFTW (batched profile `brfft`s) | 19.8% | 12.9% |
| median-select | 15.6% | 10.1% |
| decim (gather + short brfft) | 13.0% | 9.1% |
| block-sigma | 5.7% | 3.7% |
| boxcar-metric (exact rescue only) | 1.2% | 44.6% (whole metric) |
| fill-chunk (other) / other | 2.3% | 0.8% |

The metric fell from 44.6% to **21.2%** (gate 20.0 + rescue 1.2) and everything
else rose proportionally. The 1.2% rescue bucket independently confirms the gate
is working: only **98 of 12288 trials per chunk (0.8%)** ever reach `medcut` and
pay for a median.

**But `median-select` at 15.6% is not the metric — it is `_block_sigma`, and that
is now the largest metric-side cost.** `_median!` has two callers, and the
profiler's nearest-leaf classifier charges both to `median-select`; with the
rescue down at 1.2%, almost all of it must be the two MAD passes in
`_block_sigma`. Measured directly rather than inferred, per chunk summed over the
six decimations:

| | µs / chunk |
|---|---|
| `_block_sigma` (robust σ̂) | **949** |
| `boxcar_metrics!` (gate + rescue) | 1064 |
| — of which the batched gate | 912 |
| — of which the exact rescue | 152 |

So the noise-scale estimate now costs **0.89x the entire detection metric it
normalises**, ~20% of the whole search. Worse, it is *worst where the profiles
are cheapest*: at `nbins=20` (k=6) it is **213 µs against the gate's 60 µs**, 3.5x
the metric, because `_BOXCAR_SIGMA_SAMPLES = 8192` is a flat cap independent of
`nbins` and the strided subsample gather gets more expensive as the columns get
shorter. This was invisible before — at 3.7% + a share of a 10.1% bucket it read
as noise next to a 44.6% metric. **It is the next target, ahead of Kadane**, and
it looks cheap: the estimate only needs sub-percent accuracy (that was the whole
argument for pooling it per block), so a smaller subsample, a stride that is
friendlier at small `nbins`, or reusing one σ̂ across decimations are all on the
table. None of it touches the candidate list's definition, only the accuracy of a
scale factor that is already deliberately approximate.

### `_block_sigma` (2026-08-09) — 2.90x, and a corrected diagnosis

Acting on the item above: **839 → 289 µs per chunk, bit-identical σ̂ at every
decimation**, and **1.08x end-to-end** on top of the batched gate (candidate file
byte-identical). Two changes, and the interesting part is that the obvious
suspect was the smaller one.

- **The gather was reconstructing an index the array already had (1.12x).** It
  walked a strided subsample computing `i = (t-1) % nbins + 1`,
  `j = (t-1) ÷ nbins + 1` — but `M` is column-major with `size(M,1) == nbins`, so
  that `(i, j)` *is* linear index `t`. Indexing linearly gathers the identical
  samples in the identical order while dropping two hardware integer divisions
  per sample. I predicted this was ~all of the cost (16k `idiv`s × ~26 cycles ≈
  140 µs, which matched the measured total almost exactly). **It was ~12%.** The
  latency figure was the wrong one to reason from: the divisions are independent
  across iterations, so they pipeline at throughput, not latency. A
  `size(M,1) == nbins` guard now enforces the assumption the linear indexing
  rests on, since violating it would silently corrupt σ̂ and hence every metric.
- **The real cost was branch misprediction in `_select!` (2.58x).** Splitting the
  function showed the two 8192-sample quickselects were **93%+** of it (gather
  7–10 µs, abs pass 1.5 µs, medians 115–160 µs). ~65–80 µs to select from 8192
  doubles is ~12 cycles/element — the Lomuto partition's `if v[jj] <= pivot`
  mispredicts ~50% of the time on noise. Swapping *unconditionally* and advancing
  the pivot index by the comparison (`i += (x <= pivot)`) is branch-free; it
  costs two extra stores per element and saves the miss. Selection returns an
  order statistic, which is unique, so every median is bit-for-bit unchanged —
  verified against the old partition across continuous, heavily-tied, all-equal
  (Lomuto's worst case) and pre-sorted inputs at ten sizes.

**The branchless partition is size-gated (`_SELECT_BRANCHLESS_MIN = 256`)**,
because it is *not* a uniform win: **3.62x at n=8192** but **0.94x at n=120**, the
per-trial profile median of `:non`/`:sd2` that runs ~1e8 times. The extra stores
only pay once the range is long enough for misprediction to dominate. With the
gate, n=120 measures 1.32x rather than a 6% regression, so both metrics improve.

| | µs / chunk |
|---|---|
| `_block_sigma`, as committed at `3d385e1` | 839 |
| + linear-index gather | 747 |
| + branchless partition (both) | **289** |

**Lesson, and it is the second time on this function:** a plausible mechanism
with an order-of-magnitude estimate that *matches the measured total* is still
not evidence. Splitting the function into its four phases took one benchmark and
overturned the diagnosis.

**What is deliberately *not* done: a fully `Float32` profile stage.** Carrying
`ftprofs` as `ComplexF32` and folding with a single-precision `brfft` is the
remaining 1.46x on the metric kernel (855 µs vs 1248 µs) and would also cut the
12.9% FFTW bucket, so it is the largest single item left. But unlike the gate it
changes *reported* results, at ~3e-7 relative — physically irrelevant against a
metric quoted to three digits, yet it would force the `align=false` equivalence
pins down from machine precision to ~1e-6. That is a weakening of the stated
correctness discipline, not a free win, and it is a judgement call rather than a
measurement. Left open on purpose.

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
  highest-leverage item was the **boxcar width×phase scan (44.6%)**, and it has
  now had its second pass: **cross-profile SIMD batching (2026-08-09) took the
  gate 2.77x and the whole search 1.26x** with a byte-identical candidate file
  (see §2). What that leaves on the metric side, in order:
  - *Fully `Float32` profile stage* — implemented on `float32-profiles`, rebased
    onto master 2026-08-15, and now **swept over thread count, which is what the
    contradictory earlier verdicts were missing** (§3.1 below). It is 0.88x at
    `-t 1` and 1.14x at `-t 16`; the accuracy price is unchanged (reported
    metrics move ~3e-7, `align=false` pins loosen from machine precision to
    ~1e-6). Still unmerged, now for a different and better-understood reason.
  - ***`_block_sigma`* — done (2026-08-09), 2.90x (839 → 289 µs/chunk), 1.08x
    end-to-end, bit-identical. Almost all of it was branch misprediction in the
    quickselect, not the integer division the first diagnosis blamed. See §2.
  - *Kadane* (below) is now a smaller target than it looked: it competes against
    a gate that is 2.77x faster than the one the bullet was written against, and
    its own §3 step (3) — "try the cross-profile SIMD transpose on the existing
    exact scan first, it may capture much of the benefit with no approximation
    and no calibration risk" — **is what was just done, and it did.**

  The three FFT-side bullets that used to live here are done or closed:
  - *`ComplexF32` interpolation* — **closed by measurement, not implemented.**
    The direct path reads `ComplexF32` bins natively and its planes are now
    `Float32` (bit-identically); carrying the weights and accumulator in
    `Float32` too buys 1.2x on ~19% of runtime for 9 digits. See §2.
  - *Rethink FFT-correlation vs. direct interpolation* — **done**, and it was the
    large win the bullet hoped for (1.64x end-to-end). See §2.
  - *`fftlen` sizing* — reopened, re-measured in situ, and re-rejected; `:pow2`
    stays the default but for different reasons than originally recorded. See §2.
    Tiling a chunk into smaller overlapping transforms remains unmotivated.

### 3.1 Thread scaling, and the deployment model that decides what to optimise

`bench/thread_scaling.jl` is the standard measurement: it re-invokes itself once
per thread count (Julia fixes `nthreads` at process start), times only the *warm
in-process* `search`, fits Amdahl, and plots speedup against the ideal line
alongside a CPU-seconds panel. Excluding start-up matters — the same run scales
6.7x at `-t 20` on whole-process wall clock and 9.9x with the ~1.4 s fixed cost
removed.

Master, 20-core workstation, PM0063 at the riptide bench config
(`--ncands 300 --threshold 6.3`), medians of 3:

| threads | 1 | 2 | 4 | 8 | 16 | 20 |
|---|---|---|---|---|---|---|
| wall (s) | 31.9 | 18.3 | 9.4 | 5.4 | 3.3 | 3.2 |
| speedup | 1.00 | 1.74 | 3.39 | 5.95 | 9.59 | 9.88 |
| CPU-s | 31.7 | 36.3 | 36.8 | 41.5 | 48.0 | 51.8 |

**9.88x at `-t 20` (49% efficiency), Amdahl `s = 0.059`.** The efficiency loss is
not serial-fraction-shaped so much as bandwidth-shaped: CPU-seconds for
*identical work* inflate 63% across the sweep, and the `float32-profiles` branch
— which changes nothing but the width of the profile arrays — reaches **13.6x
(68%)** with only 34% CPU inflation. The parallel decomposition is fine; the
memory system is the limit.

**But thread scaling may be the wrong axis for production.** A real search runs
over many DMs, and that parallelism is usually taken by scheduling one
*single-threaded* process per DM, one per core. Under that model the figure that
governs throughput is **`-t 1` CPU-seconds**, and the whole speedup curve above
is irrelevant — every core is already busy with its own DM. The two models rank
optimisations differently, and `Float32` is the case in point: 1.14x at `-t 16`,
0.88x at `-t 1`. Committing to one would be premature; what is *not* premature is
recording that the choice exists, because several past decisions here were made
against an unstated assumption of multi-threaded deployment.

#### Where Float32's extra `-t 1` cost actually is (2026-08-15)

Profiled both arms at `-t 1` over the *same* band the A/B used. **Note the
default profiling band does not reproduce the regression**: `profile_search.jl`
starts at 5 Hz, and over 5–30 Hz the two arms are within 0.6% of each other. The
A/B band starts at 0.1 Hz, where red noise floods candidates past the gate and
the exact-median rescue path actually runs. Profile the band you benchmarked.

Absolute samples over 0.1–33.3 Hz:

| bucket | master | f32 | Δ |
|---|---|---|---|
| boxcar-gate (batched) | 23990 | 23869 | −0.5% |
| interp (direct O(m)) | 14835 | 16105 | **+8.6%** |
| FFTW (batched brfft) | 12149 | 13660 | **+12.4%** |
| decim (gather) | 9509 | 8869 | −6.7% |
| median-select | 3710 | 4250 | +14.6% |
| block-sigma | 1093 | 540 | −50.6% |
| boxcar-metric | 825 | 1351 | +63.8% |

`Float32` wins where expected — `_block_sigma` halves, the gate and the decim
gather improve — and loses in two places that need explaining:

- **interp is not a failed SIMD widening.** `src/directinterp.jl` is *byte
  identical* between the arms: `W` is `Matrix{Float64}`, the `re`/`im` planes are
  already `Float32` on master, and `sre`/`sim` accumulate in `Float64`. The only
  change is the destination — `ftprofs[hrow,k] = A[p]*complex(sre,sim)` now
  narrows into a `ComplexF32` matrix. So this is a store-side effect in an
  otherwise unchanged loop, and *widening the `m`-sum to 8 lanes was never part
  of this branch.* (That separate experiment is the 7%-slower one, and the reason
  is that at `m=16` the `m`-sum is only ~23% of interp cost while the horizontal
  reduce — paid once per trial — gets a shuffle stage deeper at 8 lanes.)
- **FFTW is not the transform getting slower.** Timed on its own, the batched
  `brfft` at `(61, 2048) -> (120, 2048)` is **422 µs in `Float32` vs 483 µs in
  `Float64`, i.e. 1.14x *faster*** — the opposite sign to the in-situ bucket. So
  the cost is an interaction, not the kernel, and optimising the transform would
  be aimed at the wrong thing. **This is the third time an isolated benchmark has
  failed to predict this workload** (smooth `fftlen`, `_block_sigma`'s `idiv`,
  now this); the standing lesson applies — split the phases *in situ*.

Next step is to split the chunk fill into "interp writes `ftprofs`" and "brfft
reads `ftprofs`" with separate timers, since both regressions sit on the same
array and a single cause (the narrowed strided scatter leaving the array in a
state the transform reads back more slowly) would explain both.

#### The split, done — and it dissolved the "FFTW" bucket (2026-08-16)

The phase timers now live in `search.jl` (`phase_reset!` / `phase_times`, always
on, ~0.03% of runtime) rather than in a profiler's bucket classifier, and they
immediately contradicted the table above. The "FFTW +12.4%" bucket was **two
transforms of opposite sign**:

| phase | f64 | f32 | Δ |
|---|---|---|---|
| base `brfft` (61→120) | 0.64 | 0.50 | **−22%** |
| decimated `brfft` (k=2…6) | 0.84 | 1.28 | **+52%** |

The base transform behaves exactly as the isolated benchmark said it would; only
the decimated ones regress, and the smaller the transform the worse it got
(k=2 +15%, k=3 +61%, k=4 +93%, k=5 +94%, k=6 +37%). Two other corrections fell
out at the same time:

- **"The whole regression lives below 5 Hz" was wrong** — an artefact of profile
  attribution, not a property of the search. `interp` regresses **+20% at
  5–13 Hz** just as it does at 0.1–8 Hz. Band choice changes the *mix* (the
  exact-median rescan), not the interp penalty.
- **`fill!(ws.ftprofs, 0)` was hiding inside the `interp` bucket** and is 0.29x
  in `Float32` — a real win that was being netted against a real loss.

Four candidate mechanisms for the decimated transform were then tested and all
four are **wrong**, which is worth recording because each is the obvious guess:

1. *Alignment of the leading dimension.* `Hk+1` is odd for four of the five `k`,
   so `ComplexF32` columns start 8-byte-aligned where `ComplexF64` columns are
   always 16-byte-aligned. But `k=4` gives `Hk+1 = 16` — perfectly 64-byte
   aligned in `Float32` — and it was the *worst* of the five. Padding the
   leading dimension would fix nothing.
2. *The transposed layout.* `(Nprof, Hk+1)` transforming along dim 2 gives FFTW a
   batch stride of 1, which is supposed to be its best case. It is **2–3x
   slower** at every size, in both precisions. The current layout is already the
   right one for the transform.
3. *The data.* FFTW is data-independent apart from subnormals; timed on genuine
   chunk contents versus `randn` of the same shape, the two agree to <1%.
4. *Cold caches.* Flushing 32 MB between calls makes `Float32` **relatively
   better** (0.70–0.82x), not worse.

#### What it actually was: the decimation gather (2026-08-16) — 1.12x…1.26x

`decim_pass!` copied every `k`-th row of `ftprofs` into a compact
`(Hₖ+1, Nprof)` buffer and transformed that. The copy read exactly the elements
the transform then read again — **pure duplicated traffic**, 14% of single-thread
runtime in its own right, and the thing that made the `Float32` arm look bad
because it doubled the pressure on the array whose narrowing was supposed to
relieve it.

The decimated stack does not need to be built at all: rows `1, k+1, 2k+1, …` of
`ftprofs` *are* the stack, DC row included (the search never writes it), so
`DecimBuf.src` is a stride-`k` view and FFTW takes the stride. Measured
standalone over `k = 2…6` at `nharms = 60, Nprof = 2048`, against gather +
contiguous transform:

| | gather+transform | strided view | |
|---|---|---|---|
| `Float64` | 1417 µs | 1038 µs | **1.36x** |
| `Float32` | 1267 µs | 792 µs | **1.60x** |

End to end on PM0063 at the riptide bench config, warm in-process `search`,
median of 3, **candidate files byte-identical to master**:

| threads | master | strided | speedup |
|---|---|---|---|
| 1 | 30.14 s | 26.99 s | **1.12x** |
| 4 | 9.98 s | 7.92 s | **1.26x** |
| 8 | 5.59 s | 4.46 s | **1.25x** |
| 16 | 3.36 s | 2.66 s | **1.26x** |

It also deletes `Σₖ (Hₖ+1)·Nprof` complex words from every workspace — most of
the per-thread footprint — which is why the win *grows* with thread count.

#### Which changes the `Float32` verdict (2026-08-16)

`SearchParams.precision` (`:f64` default, `:f32`) now carries the profile-stage
width as a type parameter on `Workspace{…,P}`/`DecimBuf{…,P}`, so **both widths
are compiled into one build** and A/B in one process — no branch switching, and
none of the ~24 s precompile that once produced a confidently wrong result. With
the gather gone:

| threads | `:f64` | `:f32` | f32 vs f64 |
|---|---|---|---|
| 1 | 26.99 s | 32.81 s | **0.82x** |
| 4 | 7.92 s | 8.64 s | 0.92x |
| 8 | 4.46 s | 4.53 s | 0.98x |
| 16 | 2.66 s | 2.64 s | **1.01x** |

**The threaded win is gone.** `Float32` never made the search faster; it relieved
a bandwidth problem, and that problem has now been fixed at its source. It is a
clear loss single-threaded and a wash at 16 threads, so **do not merge
`float32-profiles`** — the deployment-model question of §3.1 no longer decides
anything. The knob stays in tree because it costs nothing and makes the
measurement repeatable.

#### Fully-`Float32` interpolation: 1.64x slower, and now understood

Narrowing `DirectPlan`'s `W`/`A` and the accumulators to `Float32` — the last
piece of "make everything `Float32`" — is **1.64x slower** than the `Float64`
sum (1533 → 2482 µs per chunk), against the 1.09x that narrowing only the *store*
into `ftprofs` costs. The mechanism is the same one that killed the earlier
8-lane widening experiment, and it is not about bandwidth at all: at `m = 16` the
`Float32` sum is *exactly one* 16-lane vector, so there is a single FMA with no
instruction-level parallelism followed by a **four**-stage cross-lane reduce,
where `Float64` gets two independent 8-lane accumulators and a three-stage
reduce. **The per-trial horizontal reduction is what this loop pays for**, and
every attempt to narrow or widen the `m`-axis makes it worse.

That is also the standing lesson for the next interp optimisation: the win is not
a wider vector along `m`, it is *removing the reduce* by vectorising across
trials — the same move that gave the boxcar gate 2.77–4.05x. The structure is
there for it (`base_adv = 0` for every harmonic when `hidr < 1`, so the bin
window slides by 0 or 1 bin per trial, and the weight table is periodic with
period `P` in trial index, so a trial-ordered transposed table `Wt[P, m]` makes
consecutive trials contiguous). The obstacle is that a same-window run is only
~`q/h` trials long, so it vectorises well for low harmonics and barely at all for
`h > 15`; a version that handles the ±1 window slide with a permute rather than a
gather is the piece that has not been prototyped.

> **Done 2026-08-22 — and the obstacle in that last sentence was illusory.** See
> "The interpolator, across trials" below: framing the group as a matrix-vector
> product against an *extended* window removes the need for a same-window run
> entirely, so it wins at every harmonic including `h = 59`.

#### Where the single-thread time is now

`Float64`, PM0063, 0.1–33.3 Hz, `-t 1`, 29.5 s of accounted phase time:

| phase | s | % |
|---|---|---|
| decim-metric (σ̂ + boxcar, k=2…6) | 8.43 | 29% |
| interp (direct O(m)) | 7.15 | 24% |
| decim-brfft (k=2…6) | 5.18 | 18% |
| gate+metric (k=1) | 5.19 | 18% |
| base brfft | 2.56 | 9% |
| block-sigma, zeroing, candidate loops | 0.94 | 3% |

The boxcar metric work is **47%** across the two metric rows — the largest single
target left, and larger than the interpolation it was long assumed to sit behind.

#### The metric, taken apart (2026-08-22) — 1.13x on the metric, ~1.05x end to end

The 47% above was one number covering four different things, so the first step
was to split it. **`bench/metric_bench.jl`** does that, on the production
`Workspace`/`DecimBuf` buffers holding a genuine chunk: per fold depth it times
`_block_sigma`, the tile transpose, the prefix+width scan, the whole
`_boxcar_gate!`, and the full `boxcar_metrics!` (gate + exact rescan), and prints
the fraction of trials that clear `medcut`. On master that read, per chunk
(µs, `nharms=60`, `Nprof=2048`, `maxdecim=6`):

| | σ̂ | transpose | scan | gate | metric | rescan |
|---|---|---|---|---|---|---|
| total, k=1…6 | 369 | 455 | 410 | 944 | 1072 | 0.05–2.3% |

Two things in that table were not what the code's own comments claimed. The
transpose is **48% of the batched gate**, not the "~20%" the section comment in
`search.jl` said. And `_block_sigma` — billed to `block-sigma` for `k=1` but
hidden inside `decim-metric` for `k=2…6` — is **26% of all metric work**, which
no phase timer had ever shown.

**Landed: `_BC_BATCH` 32 → 64 (10.8% of the metric).** Interleaved over all six
fold depths on real chunk data, `B = 32/48/64/96/128` gives
**692/628/575/572/577 µs** per chunk: 1.20x from 32 to 64, then flat, so 64 is
the knee. This is **byte-identical** output — the batched kernel's per-profile
arithmetic and operation order do not depend on `B`, only the vector width the
compiler sees does — and it was verified as such at every `B` in the sweep.
Note what it disproves: at `B = 64` the `tile` + `psT` pair is **63 KB**, which
overflows the 32 KB L1 that `B = 32`'s 34 KB nearly fits, and it is *faster*
anyway. "Size the tile to L1" was the wrong model.

**Landed: `_block_sigma` subsamples whole profiles (3.2% of the metric).** The
old flat stride-`N÷cap` walk of the linear index stepped 240 B at
`nbins=120, n=2048`, so it pulled a fresh cache line per sample and used 8 of its
64 bytes — **2.7 MB per chunk** over the six depths, comparable to the gate's own
read of the same arrays. Taking `cap÷nbins` whole columns, evenly spaced, is
contiguous: **0.4 MB** for the same 8192 samples. It is also a better sample.
The flat stride visits bins `1, s+1, 2s+1, …` of every column, so whenever
`s | nbins` it sees only `nbins/s` **distinct phases** — four of them, at five of
the six default fold depths (`k=1,2,3,5,6`; only `k=4`'s `s=7` is coprime with
its `nbins=30`). σ̂ could be decided outright by a per-phase artefact. Whole
columns sample every phase.

This is oracle-safe by construction: `snr_metrics` defaults to
`sigma_samples = typemax(Int)`, so the estimator the Python oracle is pinned to
takes the `N <= cap` branch and never subsamples at all.

**The measurements, and the gap between them.** In isolation the two changes take
the metric from 1441 to 828 µs per chunk — **1.74x**. In situ they do not:

| arm | metric phases, % of accounted total | metric time vs master |
|---|---|---|
| master | 36.0% | 1.000 |
| σ̂ only | 35.3% | 0.968 |
| `B=64` only | 33.4% | 0.892 |
| both | 32.9% | 0.870 |

(`-t 1`, two interleaved rounds, PM0063 at 0.1–33.3 Hz. Wall clock could not
resolve this — ±10% run-to-run with visible within-round thermal drift, against a
5% effect — so the arms are compared by each run's *own* metric-phase fraction,
which is drift-robust because the untouched phases drift with it. At `-t 4` the
same comparison gives 33.3% → 30.4%, i.e. **12.7%**, so the wider tile costs
nothing threaded despite doubling the gate scratch to ~160 KB per workspace.)

So: **13.0% off the metric, ~1.05x end to end** — against 1.74x in the
microbenchmark. The reason is the same one this document keeps recording: the
isolated bench re-runs on data it has just left hot, while in the search both
`profs` and `dprofs` are read once by a phase that is competing for L3 with the
transform that wrote them. The parts of the win that were bandwidth (most of σ̂'s)
largely evaporate; the part that was vector width (`B`) survives.

**Four things that did not work, all measured:**

1. **Fusing the transpose away** — building `psT` straight from `profs` and
   deleting the `tile` entirely. A wash at `B=32` (664 vs 644 µs over all folds)
   and worse at `B=64`. The tile write/read is not the cost.
2. **Full fusion** — prefix sum and every width in one pass over the profile,
   with per-width running accumulators. **1.33x slower**: `nwidths × B/8` live
   accumulator vectors plus the rolling prefix rows exceed the 16 AVX2 registers.
3. **Blocking or reordering the transpose** — 4/8/16-bin row blocks, or a
   column-major walk: **0.56x/0.71x/0.85x/1.03x** vs the current loop, on the
   laptop, at `B=64`. **This entry was right about its own configuration and
   wrong as a generalisation — see §3.3.** All three of those block the *phase*
   axis, which is not the axis that matters; blocking the *profile* axis by 8 is
   **3.51x** on the workstation (and a small in-situ win on the laptop). The
   "~15.8 GB/s L3 bandwidth wall" this entry inferred was not a wall: at `B=128`
   the workstation ran the same loop at 4.7 GB/s against the 21 GB/s that host
   delivers at this footprint.
4. **A cheap upper bound to skip whole profiles.** Tempting because the gate only
   has to *not underestimate* by more than `boxcar_medmargin`, so a profile that
   provably cannot reach `medcut` needs no scan at all. But after the ladder
   pruning the bank is `[1,2,3,4,6]`, all narrow: for Gaussian noise at
   `nbins=120` even the tightest cheap bound (the sum of the `w` largest bins,
   `≈14.5σ` at `w=6`) is `5.9` against `medcut = 4.0`. No bound is tight enough
   to reject anything.

**Where the metric time is now** (isolated, per chunk, `B=64`): transpose 303 µs,
width scan 228, σ̂ 210, rescan + writeback 87. The scan runs at >1 vector op/cycle
and what is left of σ̂ is two quickselects. (The transpose is called a bandwidth
wall here; it was not — see §3.3.)

**Candidates are not byte-identical, by design** — σ̂ moved. At the riptide bench
config the same three candidates come out at the same frequencies, with S/N
12.97 → 13.27, 7.83 → 7.76, 7.33 → 7.22, and the pulsar's winning ladder rung
moving `k=6` → `k=4` (`H=10` → `H=15`, a near-tie between adjacent rungs).

**Chasing that +2.3% turned up something worth knowing about every S/N this code
reports.** It is not the new sampling being better or worse — it is that the
per-chunk σ̂ is *itself* a random variable with ~1% error, and reported S/N is
exactly inversely proportional to it. Measured over 24 chunks against the exact
all-bins σ̂ (245,760 samples), at 8192 samples:

| fold | flat stride: bias / rms / max | whole columns: bias / rms / max |
|---|---|---|
| k=1 (120 bins) | +0.14% / 1.50% / 3.48% | −0.05% / 1.08% / 2.47% |
| k=2 (60) | −0.47% / 1.39% / 2.69% | −0.25% / 1.15% / 2.58% |
| k=3 (40) | +0.15% / 1.22% / 2.26% | +0.27% / 1.15% / 2.22% |
| k=4 (30) | −0.09% / 1.24% / 3.09% | −0.11% / 0.88% / 1.97% |
| k=5 (24) | +0.26% / 0.97% / 2.24% | −0.11% / 0.88% / 1.89% |
| k=6 (20) | +0.17% / 1.13% / 2.18% | −0.06% / 1.03% / 2.87% |

Both are unbiased; whole columns are modestly *less* noisy at every depth. In the
one chunk that holds the 7.1185 Hz pulsar the draws happened to go the other way
— at `k=4`, flat stride landed at 1.0004 of exact and whole columns at 0.9809, so
the new number is ~1.9% high and the old one was ~0.04% high. Scoring the pulsar
against the exact σ̂ puts it at **≈13.0**, with master's 12.97 (at `k=6`, where
its draw was +0.8%) and the new 13.27 straddling it.

**So the last two digits of "12.97 vs riptide's 11.80" are σ̂ estimation noise,
and always were.** That claim is safe — the gap is 10%, the jitter is 1% — but a
detection-efficiency Monte Carlo (§3.2) comparing S/N distributions at the
percent level needs to either use the exact σ̂ or model this term. Raising
`_BOXCAR_SIGMA_SAMPLES` is the knob (rms falls as `1/√cap`), and the contiguous
gather makes the *traffic* affordable now, but the two quickselects are `O(cap)`:
16384 samples would take rms to ~0.8% and give back most of this section's win.

#### Follow-up, same day: σ̂ folds about the structural zero — another 12.2%

`_block_sigma` was spending two quickselects: one to find the median, one for the
MAD about it. But the location is *known*. DC is held at zero, so every profile's
bin mean is exactly 0 and the pooled distribution is symmetric about it, and
`1.4826 × median(|x|)` is the same scale estimator with the location supplied
rather than estimated — **one `_median!` instead of two**, and statistically the
better of the pair, since it spends no degree of freedom on a location it has.
Measured, the sample median sits within **±0.03σ** of zero at every fold depth
and σ̂ moves by ≤0.6%, against its own ~1% sampling error.

Isolated, σ̂ goes 245 → 149 µs per chunk. In situ, over three interleaved rounds,
the metric share of accounted time goes **32.83% → 30.04%** — i.e. **12.2% off
the metric, ~1.04x end to end**. Candidates barely move: 13.27 → 13.27,
7.76 → 7.76, 7.22 → 7.21.

**Cumulative for the day: metric share 36.0% → 30.0%, 23.7% off the metric,
~1.09x end to end at `-t 1`.** Interpolation (~34%) is now the largest single
phase.

**The divergence from the oracle is named, not swallowed.** This *does* change the
un-subsampled branch, which is the one `snr_metrics` uses and the one
`crossval/crossval_accuracy.jl` pins Python against — the reason it was held back
until Scott took the call on 2026-08-22 ("push speed, accept a bit of divergence,
we already know the algorithm is right from the earlier detailed tests"). Rather
than loosen the crossval tolerance from 1e-9 to ~1e-2 to hide it, the estimator
is selectable:

- `_block_sigma(...; center = :median)` computes the classic MAD about the sample
  median — exactly what upstream `snr_metric` does;
- `snr_metrics` forwards it as `sigma_center`, defaulting to `:zero` like the
  search;
- `crossval_accuracy.jl` passes `:median`, so the metric pin **stays at
  1.365e-16** and still covers the three separately fallible pieces it was written
  for: the width bank, the per-profile median baseline, and the scan arithmetic.

Loosening the tolerance instead would have let a real bug in any of those three
hide under the 1e-2 that this one deliberate difference costs.
`test/test_search.jl` pins `:zero` against `:median` at every fold depth, and
pins the *premise* separately — that the pooled median really sits at zero —
because if that ever stopped holding, `:zero` would be silently biased and every
reported S/N with it.

#### The interpolator, across trials (2026-08-22) — 1.56x on interp, 1.14x end to end

§3.1 had this move planned and had also recorded what looked like a blocking
obstacle: the bin window `b` is constant only for runs of ~`q/h` trials, which is
120 at `h = 1` but 2 at `h = 60`, so a run-based vectorisation would help low
harmonics and nothing else — and every harmonic costs the same. **That obstacle
was an artefact of insisting on a constant window.**

Write the group's sum with the window index substituted, `j = δₖ + i`, where
`δₖ = bₖ - b₀` is the `k`-th trial's bin offset within the *group's* window:

    sreₖ = Σᵢ W[i, pₖ]·re[bₖ + i]  =  Σⱼ Wx[k, j]·re[b₀ + j],   Wx[k, j] = W[j - δₖ, pₖ]

Now every trial in the group reads the *same* contiguous `m+Δ` slice of the
planes, and the group is one `(V, m+Δ)` matrix-vector product. `re[b₀+j]` is a
broadcast scalar; `Wx[:, j]` is a contiguous column; the accumulator has one lane
per trial and stays live across the whole group. **No gather, and no horizontal
reduce** — which was the thing this loop was actually paying for.

`Wx` depends only on the residue the group *starts* at, of which there are
`ngrp = q ÷ gcd(V·s, q)`, so it is tabulated once per harmonic at plan time.

**Why it wins everywhere.** The price is `(m+Δ)/m` wasted FMAs, because `Wx` is
zero off each row's `m` nonzeros, and `Δ ≈ V·h/q` grows with harmonic. That is
1.06x at `h=1` and 1.5x at `h=59` — and the kernel still wins at the worst
harmonic by 1.7x. Isolated, per 2048-trial chunk:

| h | 1 | 7 | 16 | 30 | 41 | 59 | 60 |
|---|---|---|---|---|---|---|---|
| per-trial (µs) | 13.6 | 14.4 | 14.1 | 14.4 | 14.5 | 14.8 | 14.4 |
| trials-axis, V=16 | 6.7 | 7.4 | 7.2 | 7.4 | 8.3 | 8.7 | 7.9 |
| speed-up | 2.03x | 1.95x | 1.96x | 1.95x | 1.74x | 1.70x | 1.82x |

**`V = 16`, and both benchmarks agree for once.** Isolated `V = 8/16/32/64` gives
121/99/107/157 µs; in situ the interp share is 28.5/25.4/30.7% for `V = 8/16/32`.
Below 16 there are too few lanes to hide FMA latency; above it `Δ` and the table
footprint (0.65/1.5/3.5/9.3 MB over 60 harmonics) run away. `V` is a **`const`,
not a plan field**, because the accumulators are an `NTuple{V}` and a runtime `V`
stops the `ntuple`s unrolling — the same trap `_BC_BATCH` documents.

**A closure cost 2000x.** With the `ntuple` bodies written inline they captured
`o`, a local assigned inside the loop, and Julia boxed it: 24 ms against 12 µs.
They are top-level `@inline` helpers taking everything as arguments. If this
kernel ever reads as inexplicably slow, look there before anywhere else.

**Bit-exact chunk invariance survives, and is now actually tested for it.**
Groups are anchored to the **global** trial index, so which group a trial belongs
to — and hence its exact arithmetic — cannot depend on where chunk boundaries
fall. A chunk's first and last groups may hang off either end; they are computed
*in full* and masked on store, rather than falling back to a second kernel that
would sum in a different order. That is why the plane buffers are widened to the
group range and zero-filled outside the file. The **range guard stays on the true
trial range**, so where a harmonic gives up (off the end of the amplitudes, or
past Nyquist) is unchanged — widening that guard would have silently dropped a
whole harmonic for a chunk at the top of the band. The invariance pin now runs at
chunk sizes 128/100/37/51 and asserts that three of them really do straddle
groups; `worst == 0.0` still holds.

**Measured in situ**, interleaved, PM0063 at 0.1–33.3 Hz:

| | interp (s) | interp share | interp vs old |
|---|---|---|---|
| `-t 1` before | 4.93 | 34.8% | 1.00 |
| `-t 1` after | 3.02 | 25.5% | **0.64** |
| `-t 4` before | 8.26 | 28.9% | 1.00 |
| `-t 4` after | 5.41 | 20.6% | **0.64** |

Identical at both thread counts, so the table growing 395 KB → 1.45 MB
(shared read-only across threads) costs nothing at 4. Wall clock resolved this
one without help: `-t 1` 13.89 → 11.64 s, `-t 4` 7.19 → 6.64 s.
**~1.14x end to end.** 460/460 tests, crossval unchanged, and the `.cohout` is
**byte-identical** — the ~5e-16 of resummation moved no reported S/N.

**This reopens `Float32` weights.** The 2026-08-16 verdict (1.64x *slower*) rested
entirely on the per-trial cross-lane reduce, which no longer exists; and `Float32`
would now halve a table that has grown 3.7x. Untested — see §3.1's next-steps.

#### Retirements (2026-08-16)

Two paths were removed outright rather than left as options, because both had
become strictly dominated and both were still costing something to carry:

- **The production `:fft` interpolator** (`interp_tile!`, `fill_harmonic_row!`,
  `FFTScratch`, `HarmonicPlan`/`build_harmonic_plans`/`harmonic_plan_report`,
  `harmonic_numbetween`, and the `interp`/`fftsizing`/`align` parameters with
  their CLI flags). It had been superseded on 2026-08-08 by the direct kernel —
  3.8× slower and ~1e-2 accurate against ~1e-10 — and survived only as "the
  machine-precision equivalence gate". That role is now filled better: the FFT
  correlation lives on in `fourierinterp.jl` and `reference_profiles(...;
  kernel=:fft)`, which is what the Python oracle actually pins, while the
  end-to-end gate runs `chunk_metrics` against `block_metrics(...;
  kernel=:direct)` — **the exact kernel on both sides, agreeing at 8.4e-16**.
  The old gate agreed at 7e-16 too, but only because both sides shared an
  interpolator carrying a ~1e-2 error, so it could not have caught that
  interpolator being wrong. Dropping `FFTScratch` also removed the `S` type
  parameter and the `scratch::Dict` from `Workspace`, along with `hplans` from
  `fill_chunk_profiles!`, `_search_region!`, `_plans!` and `SearchCache`.

- **The `:non`/`:sd2` on-pulse metrics.** Upstream replaced `snr_metric` with the
  boxcar matched filter as its only metric, so these had no oracle left; they had
  no user either, `:boxcar` having been the default since it was written. Their
  removal took `_profile_snr`, `xsignal`, `pexp` and a branch out of the hot loop
  and out of `decim_pass!`, `chunk_metrics` and `block_metrics`.

`src/` went 3798 → 3514 lines (`search.jl` 2281 → 1995) with candidate output
byte-identical and warm `-t 1` wall clock unchanged in an interleaved A/B (29.4 s
vs 28.6 s, pre vs post, on a machine with background load).

One subtlety worth keeping: the pooled block `σ̂` is an exact MAD in Python and a
`_BOXCAR_SIGMA_SAMPLES` subsample in the production search. `snr_metrics` now
defaults to the exact estimator (oracle-faithful) and takes `sigma_samples`, which
`block_metrics` sets to the production value (equivalence-faithful). Previously
both were the same accidental constant, which would have started lying silently
the moment either moved.

Open question, and the natural next piece of work: **how should large-scale
searches actually be driven?** The candidates are (a) one single-threaded
process per DM, maximising throughput and letting the batch scheduler handle
parallelism; (b) one multi-threaded process per DM, minimising latency per DM
and amortising start-up; (c) the existing multi-file mode, which already shares
`SearchCache` (hplans + workspaces) across files in one invocation and costs
~1.2 s marginal per file against ~15 s for a separate invocation — but which
currently shares that cache across *files*, not across DMs of the same file, and
whose `dplans` are per-file because they depend on `r_lo`.

**(c) turns out to need no work at all — it already spans DMs.** `_plans!` keys
cache reuse on `cache.params === params && cache.Nprof == Nprof` and never
consults the file, so `hplans` and the `FFTW.MEASURE` workspaces are shared
across an arbitrary file list. And the `dplans` caveat is vacuous over a
dedispersion plan: neighbouring DMs are sums of the same channels with different
delays, so they share `N` and `dt` exactly — only the per-sample noise differs —
which makes `T`, `r_lo = lofreq*T`, the bin spacing and Nyquist identical, and
therefore `dplans` identical too. It is immaterial regardless:
`build_direct_plans` is 0.29 ms at `nharms=60`, against a ~1.1 s marginal
per-file cost. Measured over ten NGC6624 `16L` DMs (T=26459 s), narrow band,
`-t 8`: 1 file 2.91 s, 10 files 12.88 s (1.11 s marginal each) versus 29.1 s as
ten invocations — 2.3x. So `coherent_search.jl ...DM??.??_red.fft` over a whole
DM range is the supported path today.

**Nor does it need the files to match.** Everything cached is a pure function of
`(params, Nprof)` — `build_harmonic_plans` never sees the file, and
`direct_window_size` is `ceil((Nprof-1)*hidr) + m + 4` — while everything
file-scaled (`r_lo`, `dplans`, the trial ranges, the `ft.N÷2` Nyquist guard) is
recomputed per `search` call. A single invocation over PM0063 (T=2097 s) and
NGC6624 (T=26459 s), a 12.6x span in `T`, writes `.cohout` files byte-identical
to running each alone. So mixed `N`/`dt` is correct and costs 0.29 ms per file,
not a cache rebuild; a warning claiming the cache must be rebuilt would be wrong.

The silent hazard in a heterogeneous glob is elsewhere: when a harmonic passes
`ft.N÷2`, `fill_harmonic_row_direct!` returns early leaving that row of
`ftprofs` zero, with no error or warning. That is deliberate — it is how the
search degrades at the top of the band — but a band chosen for a finely-sampled
file will quietly lose harmonics on a coarser-sampled one. A mixed-`dt`
diagnostic, if ever wanted, belongs there rather than on the cache.

One caveat for that mode, and it is not the obvious one: **pass `--noplot`.**
Beyond CairoMakie's 9 s load, `main` defers plotting to a pass after every
search, holding `(ft, cands, stem)` in `toplot` — so each input's mmap stays
live for the whole run. At 50 NGC6624 files that pins 69 GB of mmap at once.
With `--noplot` each `FFTFile` becomes collectable as its iteration ends.

That leaves (a) versus (b) as the real open question, and it is the same
`-t 1`-versus-threaded axis that decides the `Float32` merge.

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

### 3.2 Detection-efficiency Monte Carlo vs riptide (for the paper)

**Planned, not started (noted 2026-08-16).** Everything measured against riptide
so far is *one* real pulsar in *one* observation, plus timing. A paper needs a
population: a large Monte Carlo of injected Gaussian-shaped pulses across the
full frequency range and a realistic spread of duty cycles, at noise levels
giving overall S/N ≈ 9–13, measuring **detection fraction as a function of
(frequency, duty cycle, true S/N) for both codes, alongside the compute cost of
getting it.** Sensitivity and cost have to be reported together — either one
alone is trivially gamed by changing how much searching you do.

Design points, all of which are already-settled results elsewhere in this repo
and should not be re-derived:

- **Gaussian pulses, never boxcars.** A boxcar-shaped *signal* has harmonic
  content at every `h`, which is the best possible case for a deep harmonic sum
  and inverts conclusions about which fold depth wins. Boxcars are the *filter*,
  used because their noise statistics are tractable. This mistake was made once
  already, in the `(k, W)` redundancy analysis, and was caught only because it
  contradicted a measured detection. Include scattered (exponential tail,
  τ ≈ 0.5–2 × duty), two-component and interpulse profiles — those are the
  realistic harmonic-rich cases, and they are what makes the ladder pay.
- **Duty cycles** spanning at least 1–30%, log-spaced. 30% is `boxcar_maxfrac`
  and the top of riptide's `ducy_max`; below ~1% the fold resolution, not the
  filter, is the limit.
- **Our reported S/N carries a ~1% per-chunk σ̂ jitter (measured 2026-08-22, see
  §3.1).** `_block_sigma` estimates the noise scale from 8192 subsampled bins per
  `(chunk, k)`, and the metric is exactly `1/σ̂`, so any single reported S/N is
  good to ~1% rms and ~3% worst case. Detection *fraction* at a threshold is
  barely affected (the jitter is small against the S/N 9–13 spread), but any
  plot or claim that compares S/N *values* between the codes at the percent level
  must either raise `_BOXCAR_SIGMA_SAMPLES` for the MC runs or model the term.
- **Run BOTH comparison configurations, and say which is which.**
  `compare/compare_riptide.py --preset bench` matches *coverage* (0.1–200 Hz both
  sides) but not work — we fold everything below `hifreq` six times and riptide
  folds it once, 2.81x the profiles. `--preset matched` matches *work* to a few
  percent by running one fold depth per side. The first answers "which code
  searches an observation better", the second "which implementation is faster".
  The harness prints the full work accounting (profiles, folded bins, boxcar
  bins×widths, and `dr` vs frequency) before it times anything — quote it.
- **The frequency grids are already matched exactly and this is worth stating in
  the paper**: riptide's FFA emits trials at `dr = 1/b` Fourier bins (one phase
  bin of drift across `T`, verified against `pgram.freqs` to 2e-4), and our
  decimation-`k` pass steps by `k·hidr/nharms = 1/nbins` at `hidr = 0.5`. Same
  rule, not a coincidence, and it means no tuning knob is quietly doing the work.
- **Two known asymmetries to disclose rather than paper over.** (a) riptide
  builds its boxcar bank once from `bins_min` and reuses it for every profile, so
  at `b = 120` it only reaches 5% duty — it is width-limited on broad pulses, and
  on the reference observation reports the ~10%-duty pulsar at `w = 6`, its
  maximum. (b) `bmin 20 / bmax 120` is a factor of 6 where riptide's own
  docstring asks for ~10% and its example pipeline uses 240/260 and 480/520; that
  is forced if one invocation must reach 200 Hz, but it is not the regime riptide
  is written for. A fair population study should probably also run riptide the
  way its pipeline does — several narrow-`bins` ranges tiling the band — and
  report that as a third configuration.
- **The S/N statistics are now the SAME statistic, full stop — settled
  2026-08-24, see §3.4.** This bullet used to record the `√(1−duty)` discrepancy
  as an open item; it is closed. Our metric is riptide's `snr1` exactly, verified
  against the riptide binary at 1.4e-7 (its own `Float32`), so the paper can plot
  one against the other with no correction factor at all. **The correction this
  bullet used to prescribe would have been wrong** — §3.4 says why.
  Detection fraction at fixed injected S/N and the recovered duty cycle remain
  the primary observables, but the S/N scales are now directly comparable rather
  than merely reconcilable.
  - **What this does NOT fix is the `Hₖ` threshold-comparability item** — the
    per-`(k, frequency)` drift of the noise floor recorded in §3. Measured on
    PM0063 after the change (8.36M trials per rung), the FAP=1e-4 metric runs
    4.855 at `k=6` to 5.127 at `k=1`; before it ran 4.988 to 5.164. The spread
    across the ladder barely moved, so `--normalize` is still the answer there,
    not the metric definition.
- **Injection mechanics.** We read `.fft`, riptide reads `.dat`; inject in the
  time domain once and hand each code its own view of the same realisation, so
  the noise is common and the comparison is paired rather than independent.
  `../coherent_search/examples/` has the generator lineage, and riptide's
  `libffa.generate_signal` already builds von Mises pulses with a documented
  amplitude → expected-S/N convention worth reusing so "injected S/N" means the
  same thing to both codes.
- **Also fixes the known fixture problem** recorded in `CLAUDE.md`: the bundled
  `harmonics_hi.fft` test pulsar is far too bright and has harmonic content past
  harmonic 60, which is why the default search reports it at `11f`. A Monte Carlo
  at S/N 9–13 is the realistic regime that fixture is not.

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

### 3.3 The tile transpose (2026-08-22) — 1.36x on the workstation, from a loop nest

`bench/metric_bench.jl` on the workstation put the boxcar gate's tile transpose at
**1516 µs per chunk, 69% of the metric and ~31% of the whole search** — the
largest single item, larger than any FFT phase. It is now **421 µs (3.60x)**, and
the fix is one loop nest: **block the profile axis by 8**.

**The diagnosis that had blocked this was wrong twice over.** The recorded verdict
(§3.1, dead end 3) was "an L3 bandwidth wall, not a code defect", and the brief
that reopened it blamed **write-scatter**. Neither holds:

* It is not a bandwidth wall. At `B = 128` the workstation ran the transpose at
  **4.7 GB/s** while that host delivers **21 GB/s** of copy at the same 2 MB
  footprint (`bench/transpose_bench.jl`), and the rate is flat in size, which is
  the signature of an access-pattern limit rather than a bandwidth one.
* It is not write-scatter. In `_bc_transpose!` the inner loop is over `b` with `i`
  fixed, so the **write** `tile[(i-1)*B + b]` is a contiguous 512 B run and the
  **read** `profs[i, j0+b]` is the stride-`nbins` gather. The kernel's own comment
  had this right; the brief inverted it, and the "16 Float32 = one cache line"
  write blocking it proposed is measurably *worse* on both hosts.

What the loop actually asks for is `B = 128` **concurrent strided read streams**,
and that is what the two hosts disagree about. `bench/tile_shape_bench.jl` sweeps
the block width `BJ` (µs per chunk summed over the default `k = 1…6` ladder,
`Nprof = 2048`, `Float64` profiles):

| `BJ` | 4 | 8 | 16 | 32 | 64 | 128 (shipped) |
|---|---|---|---|---|---|---|
| i7-10510U | 354 | 325 | 377 | 372 | 354 | **295** |
| Xeon Silver 4114 | 464 | **431** | 1840 | 1764 | 1765 | 1514 |

On the Xeon everything from 16 up is one flat plateau at the unblocked cost and 8
is **3.51x** below it (2.40x in `Float32`); on the laptop the ordering is inverted
and 8 is 0.91x (1.01x in `Float32`). Blocking the *phase* axis as well (4x8, 8x8)
is worse than blocking `b` alone on both hosts, which is why the earlier
phase-blocked attempts all read as losses.

**In situ the laptop's predicted 0.91x loss does not appear — it is a small win.**
Three interleaved rounds per host, `-t 1`, PM0063 at 0.1–33.3 Hz, comparing each
run's metric share of accounted time:

| | metric share, before | after | accounted time |
|---|---|---|---|
| i7-10510U | 35.66% | 34.81% | 10.79 → 10.50 s (1.03x) |
| Xeon Silver 4114 | 48.93% | 31.60% | 19.41 → 14.32 s (**1.36x**) |

The direction flip on the laptop is the isolated-vs-in-situ gap running the other
way for once: the microbench re-reads `profs` hot, while the search reads it right
after the `brfft` that wrote it, so cutting 128 read streams to 8 helps there too.
**On the workstation the metric itself is 2.10x** (9.49 → 4.53 s) and is no longer
the largest phase — `decim-brfft`, which has still never been looked at, is.

**Candidates are byte-identical on both hosts** (21 of them down to threshold 6),
which is the right gate here: `BJ` is a loop nest, not a method, so anything else
would be a bug. `test/test_search.jl` pins the tile bit-for-bit across
`BJ ∈ {1,2,4,8,16,32,64,128}` at two tile offsets, and asserts `_BC_BATCH` stays
divisible by `_BC_TR_BJ`.

**Two ideas that this beat, both measured, so they need not be re-tried:**

* **FFTW's guru rank-0 r2r transpose** (the PRESTO trick,
  `~/src/presto/tests/test_transpose.c`: `howmany_rank = 2`, `rank = 0`, a pure
  strided copy that FFTW cache-blocks). It is genuinely good — 561 µs on the
  workstation against the shipped loop's 1510 — but `BJ = 8` gets 431 µs with no
  plan, no buffer and no ccall. On the laptop it is 0.86x. `bench/transpose_bench.jl`
  and `bench/gate_layout_bench.jl` keep it, together with the whole-chunk
  `profsT` layout it enables (1.79x on the workstation gate, **0.73x on the
  laptop**, and it costs 4.8 MB per workspace).
* **The guru profile-major `brfft`** (`bench/guru_transpose_probe.jl`, prototyped
  the day before): plan the `c2r` with output stride `Nprof` so the transform
  writes the gate's tile layout and the transpose disappears. Swept across the
  whole ladder in both precisions (`bench/guru_brfft_ladder.jl`) it is **0.78x
  (`:f64`) / 0.99x (`:f32`) on the laptop**. The probe's 1.25–1.50x came from
  comparing against `copyto!(tile, transpose(Yd))`, a naive whole-array transpose
  that is ~2.7x slower than the `_bc_transpose!` that actually ships — **the
  baseline was wrong, not the measurement.** Always benchmark against the shipped
  kernel, not against the obvious way to write it.
* **Fusing the transpose into the prefix sum** was re-tested at `B = 128` (it was
  a wash at 32 and worse at 64): **0.98x**. Still not it.

---

### 3.4 The metric IS riptide's `snr1` (2026-08-24) — the median baseline is gone

**What changed.** `_boxcar_scan`, the batched gate, `boxcar_best_width` and the
Python oracle's `snr_metric` all now correlate each profile against the width-`w`
boxcar made **zero-mean and unit-L2** — riptide's `cpp/snr.hpp:snr1`:

```
h = sqrt((n-w)/(n*w))     on the w on-pulse bins
b = w/(n-w)*h             subtracted from all n bins
snr = ((h+b)*max_p S_w(p) - b*S_tot) / sigma
    = (max_p S_w(p) - d*S_tot) / (sigma*sqrt(w*(1-d))),      d = w/n
```

The per-profile **median baseline is gone**, and with it `_boxcar_exact`,
`_baseline_median!`, `_median_net!`, `_batcher_pairs`, `_MED_NET_MAX`,
`_NO_MEDPAIRS`, and the `medbuf`/`medpairs` fields of `Workspace` and `DecimBuf`.
`boxcar_medmargin` (2.0) became `boxcar_gatemargin` (0.01): the second pass is no
longer a median rescue but a `Float64` re-score of the `Float32` gate, so the
margin only has to clear the gate's ~3e-6 rounding. Note that makes the margin
**two-sided** — the gate is no longer a lower *bound*, just a lower-precision
evaluation of the same number.

**Verification.** Against the riptide binary on identical profiles with identical
sigma: **1.4e-7** on real PM0063 folds, **2.0e-7** on pure noise — riptide's own
`Float32` accumulation. `test/test_search.jl` pins `snr_metrics` against a
longhand `snr1` at machine precision, pins the template's zero-mean/unit-L2
properties, pins unit variance per (phase, width) on 20k noise realisations, and
pins baseline invariance of `_boxcar_scan`. The Python oracle moved in step, so
`crossval_accuracy.jl` still reads **1.416e-16** on the metric. 548 tests, up
from 462.

**Why the median had to go, and why the obvious fix would have been wrong.**

* The prescription in §3.2 was "a deterministic per-width factor in
  `_boxcar_scan`". **That would have over-corrected.** `ours = sqrt(1-d) x riptide`
  is exact for the *zero-baseline gate*, which is where the analysis had been
  done — but `_boxcar_scan` runs with the **median** baseline for every reported
  candidate, and the median's own variance already compensates part of the
  factor. Measured per-(phase,width) noise sd at `nbins = 120`: 1.001 (`w=1`) to
  **0.951** (`w=28`), not to 0.876. Applying `1/sqrt(1-d)` on top gives 1.005 to
  1.086, turning a bias against broad pulses into a bias for them (12% at
  `nbins = 20`). **The analysis had been done on the gate; the shipped path was
  not the gate.**
* **The median recovered nothing.** DC is held at zero, so `S_off == -S_w`
  identically: the true off-pulse level is `-S_w/(n-w)`, and subtracting it is
  *exactly* a `1/(1-d)` rescale of `S_w`. Verified to 9 decimals against an
  oracle off-pulse baseline. The pulse's DC power was never lost when the profile
  was forced to zero mean — the constraint had already put all of it into `S_w`,
  where the boxcar picks it up.
* **What the *sample* median actually did was make the normalisation depend on
  the source.** It sits near 0 in noise and near the off-pulse level for a bright
  pulse, so the ratio to riptide's statistic drifted **0.981 -> 1.065** with
  injected amplitude at 5% duty (0.90 -> 1.03 at 15% duty). A statistic whose
  scale depends on the signal has no calculable false-alarm rate at any
  threshold, which is a worse problem than the one §3.2 set out to fix.
* **At matched false-alarm rate the zero-mean template is strictly better**
  (Gaussian pulses, FAP 1e-3, `nbins = 120`, detection fraction at amplitude 8):

  | ducy | 0.02 | 0.05 | 0.10 | 0.20 | 0.30 |
  |---|---|---|---|---|---|
  | median baseline, `/sigma*sqrt(w)` | 0.999 | 0.993 | 0.893 | 0.142 | 0.010 |
  | riptide's template | 0.999 | 0.998 | **0.971** | **0.274** | **0.019** |

**Effect on real data (PM0063, `-t 4`, threshold 6, 0.1-33.3 Hz).** The 7.1185 Hz
pulsar reads **13.27 -> 12.30** (`k = 4` both times, ducy 10%), and the empirical
noise floor falls with it — FAP=1e-5 goes 6.367 -> 6.167 at `k=1` and 5.681 ->
5.514 at `k=4`, from 8.36M trials per rung. Candidates >= 6.0 go 21 -> 7, but
**that is mostly the rescale**: at matched FAP the counts are comparable, and
saying otherwise would overclaim. The genuine gains are the broad-duty row of the
table above, and having a threshold whose FAP is calculable at all.

**On the residual difference from `rseek` — and why it is NOT evidence about
sensitivity.** `--preset matched`, one fold depth per side: rseek **12.60**
(`w=13`, ducy 10.3%) vs ours 12.10 -> **11.84** (`H=65`, ducy 14.6%).

What that licenses is a *decomposition*, not a conclusion: since the metric is
now provably identical given identical profiles, whatever difference exists on
any given observation comes from the **profile estimator** (our 65-harmonic
coherent Fourier fold against riptide's time-domain FFA fold) and/or sigma-hat,
and no longer from the S/N definition. That statement holds at any sample size.

**Its magnitude does not.** This is one pulsar in one observation — a single
noise realization, whose extreme-value scatter on a single detection is far
larger than the ~0.8 separating the two numbers, and larger again than the ~1%
sigma-hat jitter recorded in §3.1. **Do not read a sensitivity difference out of
it, in either direction, and do not open a bug hunt on the strength of it.**
Relative sensitivity is exactly what §3.2's injection Monte Carlo is for: many
realizations, paired noise, detection fraction against injected S/N and duty
cycle. Until that runs, the honest statement is that the two codes' S/N values
are now the same statistic and the per-observation difference is un-interpreted.

**Cost.** ~8% off the metric phase: `bench/metric_bench.jl` reads 1092 -> 950 us
per chunk, after correcting the ~6% whole-machine drift visible in the
*untouched* sigma-hat (193.4 -> 181.0) and transpose (437.6 -> 410.8) columns —
the removed median rescan, against ~1% added to the scan by the `d*S_tot` term.
End to end that is inside this host's run-to-run scatter (`-t 1`: 17.45 s ->
17.34 s, median of 3, ~15% spread), so **do not quote a wall-clock speedup for
this change.**

**One API note.** `boxcar_widths` now caps at `nbins-1` (riptide's
`check_trial_widths` requires `w < bins`) and rejects `nbins < 2`: the zero-mean
unit-L2 template of a full-width boxcar is the zero vector, and `sqrt(w*(1-w/n))`
would be 0. At the default `maxfrac = 0.3` the cap only bites for `nbins <= 3`.

---

### 3.5 The noise scale is computed, not measured (2026-08-24)

`--sigma analytic` is the default as of 2026-08-24. This closed one of the
longest-standing soft spots in the metric and made the search ~1.07x faster as a
side effect, which is the opposite of the usual trade here.

**The derivation.** The search is only meaningful on a normalised `.fft`, and
that assumption already fixes the fold's noise. Mean Fourier power 1 means the
real and imaginary part of each amplitude have variance ½, so the hot loop's
unnormalised `brfft` of a stack holding harmonics `1 … H` with DC held at zero
gives, at every phase bin,

```
var(P_j) = 4·(½)·(harmonics below the profile's Nyquist bin) + (½)·(that bin)
         =>  sigma = sqrt(2·nlow + 0.5·nnyq)
```

the last term halved because the transform keeps only the real part of that bin.
That is `sqrt(nbins)` times `sqrt(1 − 3/(4H))`.

**Two details that are not optional.**

1. **The `sqrt(1 − 3/(4H))` factor.** 0.6% at `H = 60`, **3.8% at the `H = 10`
   of a `k = 6` fold**. Dropping it biases the shallow folds against the deep
   ones, which is exactly the cross-decimation bias §3.4's metric exists to
   remove — it would have quietly reintroduced it one rung at a time.
2. **The fill count, not the stack length.** Harmonics past Nyquist are zero rows
   and carry no noise. `fill_harmonic_row_direct!` now returns whether it filled
   the row, `fill_chunk_profiles!` records that in `ws.filled`, and
   `_analytic_sigma` counts what is there. Using `H` at the top of the band
   overestimates sigma and silently suppresses fast candidates — the failure
   would look like a sensitivity limit, not a bug.

**It is faster.** Laptop, PM0063 0.1–33.3 Hz, 7 interleaved reps, read by
metric-phase *share* (the wall clock scattered 8.6–12.0 s across reps while the
share held to ±0.2%, and one absolute phase table showed every phase dropping
24–31%, including ones that cannot have changed — the drift trap again):

| | measured | analytic | |
|---|---|---|---|
| metric share, `-t 1` | 27.00% | 22.86% | −15.3% |
| wall clock, `-t 1` | 8.91 s | 8.29 s | 1.075x |
| metric share, `-t 4` | 26.99% | 23.09% | −14.5% |
| wall clock, `-t 4` | 4.14 s | 3.94 s | 1.053x |

**And it is more accurate, which is the better half.** Against the *exact*
all-bins pooled MAD on PM0063, over four frequency windows and all six fold
depths (`bench/toy_vs_production.jl`):

| | vs exact sigma-hat | spread |
|---|---|---|
| analytic closed form | 0.9918–1.0217 | **3.0%** |
| production's 8192-sample subsample | 0.9806–1.0335 | 5.4% |

**The closed form is closer to the truth than the estimator it replaced.**
Reported S/N is exactly `1/sigma-hat`, so the subsample's ~1% sampling error was
landing on every candidate — the term §3.2 was going to have to model or avoid.
It is gone, and §3.2's injection Monte Carlo can now compare S/N at the percent
level directly.

**The one residual bias is predicted, constant, and left uncorrected.** The
`m`-bin kernel keeps `S_m ≈ 1 − 0.203/m` of the *noise* power along with the
signal, so the analytic scale runs `0.203/(2m)` high. Measured on synthetic
normalised white noise: 1.0086 / 1.0053 / 1.0035 at `m = 16 / 32 / 64` against
1.0064 / 1.0032 / 1.0016 predicted. Smaller than the error it replaces, so
correcting it would be false precision.

**The assumption fails silently and in the dangerous direction.** The measured
scale tracks whatever the amplitudes are; the analytic one keeps insisting on
unit variance. So a normalisation error *inflates* every S/N and fills the
candidate list with noise, rather than emptying it — the failure mode that looks
like success. `_sigma_sanity_check` therefore scores three chunks both ways
before the detection pass and warns above 10% disagreement, naming both numbers,
for ~0.1% of the runtime. On the un-normalised `harmonics_hi.fft` it fires at a
factor of ~1000. It **warns** rather than switching estimator, because changing
the statistic behind the user's back would be worse than the problem.

**`--sigma measured` is still right for some data**, and this is not a hedge: if
the noise level varies with Fourier frequency — residual red noise, an RFI comb,
a `rednoise` pass that did not take — the MAD adapts and the closed form cannot.
That trades a ~1% estimation error against an unmodelled bias, and on a
badly-behaved observation the bias wins.

**Consequences to know about.** Candidates move ~1–2% and near-threshold ones
churn (PM0063 at threshold 6: the pulsar 12.30 → 12.11, the 0.2603 Hz candidate
7.32 → 7.37, 7 candidates either way with one swap at 6.0–6.1), so a `.cohout`
diff across this change is *not* expected to be empty. `chunk_metrics` still
forces `_block_sigma`, since its job is to equal `block_metrics`, so every
equivalence pin is untouched; `test_search.jl`'s decimation-vs-native-fold pin
had to say `sigma=:measured` explicitly for the same reason, and was off by
~900x until it did.

**Confirmed on both hosts (2026-08-24), which is not the usual outcome here.**
fitzroy (Xeon Silver 4114) gives a metric share of 26.92% → 23.08% at `-t 1`
(14.3% off the metric, 1.067x wall) and 25.92% → 22.34% at `-t 20` (13.8%),
against the laptop's 15.3% and 14.5%. 747/747 tests and crossval at 3.885e-16 /
1.435e-16 / 1.416e-16 there. The `-t 20` wall clock is *not* usable — a ~1 s run
on a desktop carrying Chrome and Zoom scattered 0.93–1.36 s — but the share held
to ±0.2% across seven reps.

Better still, the **accuracy** table came out digit-for-digit identical on the
two hosts, given the same input file: every `(k, window)` ratio, both summary
rows, all of it. That is the expected result — σ is a property of the data and
of deterministic code, not of the CPU — which is precisely why it was worth
checking. A host-dependent number there would have been a bug, not a measurement.

**Where this came from.** It was worked out and validated in
`bin/toy_coherent_search.jl` — the simple reference implementation written for
the paper's pseudo-code figure — before being brought into production. That is
the second time the toy has paid for itself in the same session.

### 3.6 The boxcar's noise scale on a band-limited profile (2026-08-28)

**Found while designing the §3.2 Monte Carlo, and it was in production the whole
time.** `snr1` divides the width-`w` boxcar sum by `σ·√(w(1−δ))`, which is right
only when the profile's phase bins are **independent**. riptide's are: it folds in
the time domain, so each bin sums a disjoint set of samples. Ours are not — our
profile is the unnormalised `brfft` of a harmonic stack with DC held at zero and
every row past the last filled harmonic left zero, i.e. **band-limited**, and
band-limited noise is correlated between bins. The `1.4e-7` agreement with the
`rseek` binary recorded in §3.4 was measured by handing both codes *the same
profile*, which is exactly the test that cannot see this.

**Symptom, on pure noise.** Same trial count per 1 Hz band, `--maxdecim 1`,
`dt = 60 µs` so harmonic 60 crosses Nyquist at 138.9 Hz. Peak S/N, and trials
above 5, before → after:

| band | before | after | trials ≥ 5 |
|---|---|---|---|
| 20–21 Hz | 5.32 | 5.28 | 24 → 18 |
| 100–101 Hz | 5.59 | 5.55 | 26 → 24 |
| 271–272 Hz | **7.57** | **5.56** | 4112 → 11 |
| 500–501 Hz | **9.09** | **5.01** | 33116 → 1 |

**The fix is exact, not fitted.** Writing the boxcar as a filter, its response at
profile harmonic `j` is the Dirichlet kernel `D_j = sin(πjw/nbins)/sin(πj/nbins)`,
and each filled harmonic contributes independent real and imaginary parts of
variance 1/2, so in the units `_analytic_sigma` counts

    var(S_w) = 2·Σ_{j filled, j < nbins/2} |D_j|²  +  (1/2)·|D_{nbins/2}|²

with the last term present only when the profile's own Nyquist harmonic is
filled. `_boxcar_shape!` builds `σ/√(var(S_w))` per width; `_boxcar_scan` then
forms `S_w · invsigma · shape[i]`. The `δ·S_tot` term contributes nothing to the
variance because DC is held at zero, making `S_tot ≡ 0` identically.

Validated against 100k noise profiles built through the shipped `brfft`, at
`nbins ∈ {120, 40, 20}` × six fill counts: **predicted and measured agree inside
the ~0.2% Monte-Carlo error in all 60 cells.**

**It fixes two biases, and only the first was being looked for.**

* **Nyquist truncation**, above. Inflation ≈ `√(nbins/2H)`: measured
  per-(phase,width) sd 1.40 at `H = 30`, 1.90 at `H = 16`, **2.48 at `H = 10`**.
* **Fold depth, with nothing truncated at all.** At `H = nbins/2` the exact
  formula does *not* reduce to `√(w(1−δ))`; it leaves ≈ `1 + 3/(4·nbins)` — 1.006
  at `nbins = 120` but **1.040 at the `nbins = 20` of a `k = 6` rung**. Every
  ladder rung carried a noise floor ~3.4% different from its neighbours, biasing
  **which fold depth wins** — the one comparison the boxcar metric and
  `_analytic_sigma`'s own `√(1−3/(4H))` term exist to keep honest.

**Measured end to end** (injected file, `dt = 60 µs`, defaults, so nothing is
truncated and this is purely the second bias). Two injections at zero-mean
matched-filter S/N 9:

| | before | after | Δ |
|---|---|---|---|
| 271.234 Hz, `k=6` (20 bins) | 10.01 | 9.68 | −3.3% |
| 3.7124 Hz pulsar | 10.00 at **`k=5`** (24 bins) | 9.77 at **`k=2`** (60 bins) | −2.3%, **rung moved** |
| 608 Hz noise, `k=5` | 6.74 | 6.53 | −3.1% |
| 33.1 / 70.7 Hz noise, `k=1` | 6.39 / 6.23 | 6.35 / 6.19 | −0.6% |

The rung change is the point: with the rungs finally on the same footing, a
5%-duty pulse is won by a 30-harmonic fold rather than a 12-harmonic one, which
is the physically right answer. Candidates above 6 went 18 → 13, and
`total_above_threshold` 248 → 220.

**What this does to §3.4's headline.** "Our S/N **is** riptide's `snr1`" is no
longer literally true of the production path, and should be stated as: both codes
use the same zero-mean unit-L2 boxcar matched filter, and ours additionally
normalises by the exact covariance of *its own* fold. On a fully-sampled profile
the two differ by the `1 + 3/(4·nbins)` above. `snr_metrics` still computes plain
`snr1` by default — that is what the Python oracle is pinned to and what
`test_search.jl`'s longhand-`snr1` test asserts — and takes `nfilled` to select
the production normalisation, which is what `block_metrics` passes so the
equivalence gate keeps comparing like with like.

**Pins.** Oracle unchanged at 3.847e-16 / 2.131e-16 / **1.393e-16**; suite 819
passing (up from 752, the new testset being the noise simulation, the
`H = nbins/2` residual, rung-`k` indexing, and `_refresh_shape!`'s cache).

**Cost is nil.** The table is `O(Hk · nwidths)` transcendentals, cached on the
fill count in `_refresh_shape!` — and because failure to fill is monotone in
harmonic number the filled set is always a prefix, so the count determines it.
In a normal search it rebuilds a handful of times in total.

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
