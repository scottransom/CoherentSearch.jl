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
- `src/directinterp.jl` — **the default interpolator** (`--interp direct`).
  Evaluates Eqn. 30 exactly at each trial: the coefficients factor as
  `A(dr)/(dr-j)` with *real* `1/(dr-j)`, and only `q = 2·nharms` distinct `dr`
  occur in a whole search, so the weights are tabulated once per harmonic and
  indexed by an exact integer residue. No FFT, no fine grid, no linear interp.
  ~3.8× faster than the FFT path *and* ~1e-10 vs its ~1e-2 accuracy.
  `--interp fft` keeps the old FFT-correlation path (the Python original's
  method) as a fallback and as the machine-precision equivalence gate.
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
oracle/equivalence pins green. There are now two, one per interpolator:

- **`:fft`** — the `align=false` tests in `test/test_search.jl` pin the optimised
  path to `block_metrics` at machine precision. They must be run with
  `interp=:fft, fftsizing=:pow2`, which is exactly what `block_metrics` does.
- **`:direct`** — pinned to `fourier_interp` (the exact Eqn.-30 kernel, itself
  oracle-pinned to Python at ~3e-16) at ~1e-8, plus a bit-exact chunk-invariance
  test. This is a *stronger* statement than the `:fft` pin, since it is agreement
  with the exact kernel rather than with an approximating path.

The two interpolators cannot agree bit-for-bit — `:fft` carries a
linear-interpolation error — so a `:direct` change is checked by candidate-list
comparison (same survivors, metrics within ~1%), not byte-identical diff.

## Commands

```sh
# Tests (the correctness gate)
julia --project=. -e 'using Pkg; Pkg.test()'

# Search (-t auto for all cores)
julia --project=. -t auto bin/coherent_search.jl FILE.fft --lofreq 0.1 --hifreq 100
# Heavy multi-frequency config (the standard perf test):
julia --project=. -t 4 bin/coherent_search.jl --threshold 6 --metric sd2 \
      --maxdecim 6 -o out.txt --noplot FILE.fft
# Per-harmonic interpolation plan (fftlen, numbetween, m, padding, linterp)
julia --project=. bin/coherent_search.jl --verbose ... FILE.fft
# The old FFT-correlation interpolator (fallback / equivalence gate)
julia --project=. bin/coherent_search.jl --interp fft --fftsizing pow2 ... FILE.fft

# Cross-validation against the Python oracle
julia --project=crossval        crossval/crossval_accuracy.jl FILE.fft
julia --project=crossval -t auto crossval/crossval_speed.jl   FILE.fft
```

## Performance work (current focus)

The project is feature-complete; the active focus is profiling and speeding up
the hot loop. See `Summary_and_Future_Work.md` (§3) for the roadmap.

- **Bench harness lives in `bench/`** (own env, dev-deps only):
  `microbench.jl` (per-bucket timings, both interpolators), `profile_search.jl`
  (warm sampling profile with a bucket-aggregated self-time table),
  `median_bench.jl`, `interp_bench.jl` (interpolator throughput in points/sec vs
  `m`, grid oversampling and request size, + plots). Run single-threaded (`-t 1`)
  for clean profile attribution; warm up before timing to exclude JIT. Example
  FFT for longer runs: `PM0063_034C1_DM445.0_red.fft`.
- **Done (2026-07):** quickselect median in `_profile_snr` (was 41% of runtime →
  7.5%) and a type-stable `Workspace{S,B,D}` (killed hot-loop dynamic `mul!`
  dispatch) — together ~1.6× warm single-thread, results unchanged. See §2 of
  `Summary_and_Future_Work.md`.
- **Done (2026-08-08):** direct `O(m)` interpolation replacing the FFT
  correlation (`src/directinterp.jl`, `--interp direct`, now the default). The
  interp FFTs — 76% of the chunk fill, ~50% of runtime — are gone; **1.64×
  end-to-end** and ~1e-10 accuracy where the FFT path had ~1e-2.
- **Smooth `fftlen` sizing: re-examined, still rejected — but measure in situ,
  not per size.** Per transform it really is 1.26× (`next_pow_of_2` wastes a mean
  1.38× in length, and we choose the length). In a real search it is a wash to
  −7%: specific smooth lengths (`3^k`-heavy: 6561, 13122, 15309) are *worse* than
  the power of two, and `:pow2`'s 4 shared `FFTScratch` sets stay cache-warm
  where 8–16 smooth ones do not. `fftsizing=:pow2` stays the default; `:smooth`
  is available. **Lesson: an isolated per-size FFT benchmark does not predict
  this workload.**
- **`ComplexF32` interpolation is largely moot.** It existed to halve the
  bandwidth of transforms that no longer run; the direct path already reads
  `ComplexF32` bins natively and accumulates in `Float64`. What remains is one
  isolated question — `Float32` weights/planes in the direct inner loop — worth
  measuring, not architecting.
- **HPK / KFR (see `~/programming/fft_tests/HPK_JULIA_HANDOFF.md`): deferred.**
  Its own prerequisite was to profile the FFT fraction first; done, and FFTW now
  falls to the ~4% batched profile `brfft`. 1.5× on 4% does not justify a
  proprietary binary-only dependency. Revisit only if `:direct` is backed out.

## Environment gotchas (Julia 1.12)

- `SortingNetworks.jl` and `StatProfilerHTML`'s HTML writer are **broken on
  1.12** (method-overwriting precompile / `Core.MethodInstance` field change).
  Use `partialsort!`/hand-rolled selection and the profiler's text/bucket
  summary instead. `LoopVectorization.@turbo` works.
- FFTW *planning* is not thread-safe: build all plans single-threaded before the
  parallel region; only *execute* them (via `mul!`) inside it.
