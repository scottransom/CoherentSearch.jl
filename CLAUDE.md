# CoherentSearch.jl — working notes for Claude

A pure-Julia port of the Python `coherent_search` package (a sibling repo at
`../coherent_search`): a coherent harmonic-summing pulsar search over PRESTO
`.fft` files, using fast complex Fourier interpolation. The port is restructured
for multi-threaded performance and is numerically pinned to the Python original.

## Workflow (solo developer, in active development)

- **Commit straight to `master`. No branches, no PRs, and you do not need to ask
  before committing** — Scott is the only developer and this code is pre-release.
- **Always run the test suite before committing** (`Pkg.test()`), and prefer a
  real end-to-end check when results could move (see the equivalence gate below).
- End commit messages with the `Co-Authored-By: Claude Opus 4.8` trailer.
- Scott is a pulsar astronomer and the author of PRESTO — pitch at expert level;
  be concise and don't over-explain domain basics.

## Architecture essentials

- `src/fourierinterp.jl` — Fourier interpolation kernels (Eqn. 30 of
  astro-ph/0204349). Heavily indexing-tested (0-based Python ↔ 1-based Julia).
- `src/fileio.jl` — mmap'd PRESTO `.fft` reader + `.inf` parser. Amplitudes are
  `ComplexF32`; element 1 packs DC.re + Nyquist.im.
- `src/search.jl` — the core. **Two paths that must agree:**
  - a simple *reference* path (`block_metrics`/`reference_profiles`) kept
    deliberately unoptimised and pinned to the Python oracle at ~1e-15;
  - an *optimised* production `search`: chunk-parallel (`@spawn`, one private
    `Workspace` per task), per-harmonic cached FFTW plans + interpolation
    kernels, a batched inverse FFT, and harmonic decimation for cheap
    multi-frequency search.
- `src/candidate.jl`, `bin/plotting.jl` — per-candidate profile reconstruction
  and CairoMakie plots (loaded lazily; ordinary runs/tests never pay for it).

**Correctness discipline (do not break this):** every optimisation must keep the
oracle/equivalence pins green. The key guard is the `align=false` test in
`test/test_search.jl`, which pins the optimised path to `block_metrics` at
machine precision. A change that alters the median/metric value fails it. When in
doubt, also diff a full-run candidate file against a known-good one.

## Commands

```sh
# Tests (the correctness gate)
julia --project=. -e 'using Pkg; Pkg.test()'

# Search (-t auto for all cores)
julia --project=. -t auto bin/coherent_search.jl FILE.fft --lofreq 0.1 --hifreq 100
# Heavy multi-frequency config (the standard perf test):
julia --project=. -t 4 bin/coherent_search.jl --threshold 6 --metric sd2 \
      --maxdecim 6 -o out.txt --noplot FILE.fft

# Cross-validation against the Python oracle
julia --project=crossval        crossval/crossval_accuracy.jl FILE.fft
julia --project=crossval -t auto crossval/crossval_speed.jl   FILE.fft
```

## Performance work (current focus)

The project is feature-complete; the active focus is profiling and speeding up
the hot loop. See `Summary_and_Future_Work.md` (§3) for the roadmap.

- **Bench harness lives in `bench/`** (own env, dev-deps only):
  `microbench.jl` (per-bucket timings), `profile_search.jl` (warm sampling
  profile with a bucket-aggregated self-time table), `median_bench.jl`.
  Run single-threaded (`-t 1`) for clean profile attribution; warm up before
  timing to exclude JIT. Example FFT for longer runs:
  `PM0063_034C1_DM445.0_red.fft`.
- **Done (2026-07):** quickselect median in `_profile_snr` (was 41% of runtime →
  7.5%) and a type-stable `Workspace{S,B,D}` (killed hot-loop dynamic `mul!`
  dispatch) — together ~1.6× warm single-thread, results unchanged. See §2 of
  `Summary_and_Future_Work.md`.
- **Settled — do not revisit:** smooth (`2·3·5·7`) `fftlen` sizing. Investigated
  and rejected; `next_pow_of_2` + `MEASURE` is already ≈ FFTW's best case (~3%
  ceiling, and smooth needs ~60 sizes whose MEASURE planning is ~0.7 s each).
- **Biggest remaining lever:** `ComplexF32` interpolation (halve bandwidth on the
  interp FFTs + the `spec.*coeffs` multiply), but it breaks the `Float64` oracle
  pins → needs a precision-mode design + injected-signal validation. Second:
  direct `O(m)` interpolation of only the needed points instead of FFT-correlation
  over a full fine grid. Profile before committing to either.

## Environment gotchas (Julia 1.12)

- `SortingNetworks.jl` and `StatProfilerHTML`'s HTML writer are **broken on
  1.12** (method-overwriting precompile / `Core.MethodInstance` field change).
  Use `partialsort!`/hand-rolled selection and the profiler's text/bucket
  summary instead. `LoopVectorization.@turbo` works.
- FFTW *planning* is not thread-safe: build all plans single-threaded before the
  parallel region; only *execute* them (via `mul!`) inside it.
