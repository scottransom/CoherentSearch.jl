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
- End commit messages with a `Co-Authored-By: Claude <model>` trailer naming the
  model that actually wrote the commit (e.g. `Claude Opus 5`). Commits through
  2026-08-09 say `Opus 4.8`, which was current then — don't copy that trailer
  forward once the model has moved on.
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
- `src/cli.jl` — the ArgParse driver, `CoherentSearch.main`. **In the package,
  not in `bin/`, on purpose**: as a top-level script, inferring and codegen'ing
  `main` cost ~4.7 s on every run. `bin/coherent_search.jl` is a shim.
  Searches *many* `.fft` files per invocation, sharing one `SearchParams` and
  one `SearchCache`, and defers all plotting to a single pass at the end.

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
# Many files in ONE invocation (start-up + plans paid once; one .cohout each)
julia --project=. -t auto bin/coherent_search.jl *_red.fft --noplot --threshold 8
# Production sysimage (build takes minutes; freezes src/ — do not use while editing)
julia --project=sysimage sysimage/build_sysimage.jl
julia --sysimage sysimage/coherent_search.so --project=. -t auto bin/coherent_search.jl ...
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

# Head-to-head against riptide's FFA (the real external bar)
python3 compare/compare_riptide.py --repeat 3 --threads 4 FILE.fft
```

## The riptide bar (read this before optimising)

**The point of the performance work is to beat riptide's FFA** (`rseek`, cloned
at `../riptide`), not the Python original. Run
`python3 compare/compare_riptide.py FILE.fft` — it is an *occasional* benchmark
(~5 min at `--preset bench --repeat 3`), not a dev-loop tool.

Measured 2026-08-11 on `PM0063_034C1_DM445.0_red.fft`, `--preset bench`, both
covering **0.1–200 Hz in 120…20 bins**, 4-core i7-10510U:

| | wall (s) | cores |
|---|---|---|
| `rseek` | 24.4 | 1.02 |
| ours `-t 1` | 29.3 | 1.01 |
| ours `-t 4` | 22.3 | 3.23 |

**~1.2–1.4x slower single-threaded, but we detect more strongly**: the 7.1185 Hz
pulsar at S/N 12.97 vs riptide's 11.80, plus two candidates it does not report.
riptide's two extra entries are the `f/2` and `2f` of the pulsar, which it does
not filter and we collapse.

- **Match total frequency COVERAGE, not the trial range.** Both codes hit the
  same sampling wall — riptide needs `P >= tsamp*bins` and downsamples to stay
  in `[bmin, bmax]`; our `k`-fold of `nharms/k` harmonics needs its top harmonic
  under Nyquist, the same inequality — and both climb it by folding into fewer
  bins. So `nharms = bmax/2`, `maxdecim = bmax/bmin`, and **`hifreq =
  (1/Pmin)/maxdecim`**, decimation carrying coverage the rest of the way.
  Setting `hifreq = 1/Pmin` instead makes us search 6x riptide's band and
  reports a bogus 2.1x deficit. **This mistake was made once already** — the
  first version of this section quoted 2.1x from exactly that mismatch.
- **Decimation is a real advantage; configure it to be used.** `--maxdecim 6`
  against `--bmin 20` is the matched pair, not `--maxdecim 1`.
- **riptide's BLAS threading is a red herring**: pinning every thread pool to 1
  moved its median wall clock 4.68 -> 4.54 s over 6 interleaved pairs (nothing,
  against ~8% scatter) while dropping CPU 114% -> 99%. Left enabled by default.
- Threading is ours alone, not a like-for-like win (riptide has no OpenMP).
- This laptop throttles (`scaling MHz: 67%`); the **threaded** number is the
  least reliable, since back-to-back heavy runs clock the CPU down. Quote `-t 1`.

## Environment gotchas (Julia 1.12)

- `SortingNetworks.jl` and `StatProfilerHTML`'s HTML writer are **broken on
  1.12** (method-overwriting precompile / `Core.MethodInstance` field change).
  Use `partialsort!`/hand-rolled selection and the profiler's text/bucket
  summary instead. `LoopVectorization.@turbo` works.
- FFTW *planning* is not thread-safe: build all plans single-threaded before the
  parallel region; only *execute* them (via `mul!`) inside it.
