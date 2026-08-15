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
- `bin/sift_candidates.py` — cross-observation/cross-DM candidate sifter (a
  PRESTO `ACCEL_sift` analogue; pure stdlib, no numpy). Reads the `.cohout`/`.txt`
  candidate files, parses DM from the filename, and learns everything else from
  the inputs. Three stages: per-obs collapse across DM → link across obs by
  *fractional* frequency (T-independent) → harmonic collapse + pulsar-likeness
  score. Text report plus a self-contained HTML/SVG summary (`--html`).
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

# Cross-validation against the Python oracle.  Needs an interpreter that can
# `import coherent_search`: /home/sransom/python_venvs/pixiPSR/.pixi/envs/default/bin/python
# (override with $COHERENT_PYTHON).  Default test data is the sibling repo's
# examples/harmonics_hi.fft — a 10.0123456789123 Hz fake pulsar, T=1000 s
# (override with $COHERENT_FFT).  Both paths are machine-specific; re-point them
# on a new host.
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

- **Start-up is NOT what the comparison measures**, and the harness proves it
  per run: rseek 1.41 s start-up + 22.93 s searching, ours 1.48 s + 28.78 s, so
  the pure-compute ratio (1.26x) matches the wall ratio (1.24x). Our fixed cost
  is measured by re-running the same command over a near-empty band.
  riptide's `find_peaks` is 7.0 s (31% of its compute) and is a separate pass
  doing the candidate work we do inline — do **not** compare us against its
  `ffa_search` (15.6 s) alone.
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

## Performance work (current focus)

The project is feature-complete; the active focus is profiling and speeding up
the hot loop. See `Summary_and_Future_Work.md` (§3) for the roadmap.

- **Bench harness lives in `bench/`** (own env, dev-deps only):
  `microbench.jl` (per-bucket timings, both interpolators), `profile_search.jl`
  (warm sampling profile with a bucket-aggregated self-time table),
  `median_bench.jl`, `interp_bench.jl` (interpolator throughput in points/sec vs
  `m`, grid oversampling and request size, + plots), `thread_scaling.jl`
  (speedup vs threads against the ideal line and an Amdahl fit, plus the
  CPU-seconds panel; CSV + PNG). Run single-threaded (`-t 1`)
  for clean profile attribution; warm up before timing to exclude JIT. Example
  FFT for longer runs: `PM0063_034C1_DM445.0_red.fft`.
- **`bench/thread_scaling.jl` is the standard scaling check** (it replaced the
  old `scaling.jl`, which timed one thread count per invocation and left the
  analysis to you). It re-invokes itself as a worker per thread count — Julia
  fixes `nthreads` at process start — and each worker times only the *warm
  in-process* `search`. **Excluding start-up is the point:** on the whole-process
  wall clock the same run scales 6.7x at `-t 20`, and 9.9x once the ~1.4 s fixed
  cost is out. Measured on master, 20-core workstation, PM0063 at the riptide
  bench config: 1/2/4/8/16/20 threads → 31.9/18.3/9.4/5.4/3.3/3.2 s, i.e. **9.9x
  at `-t 20` (49% efficiency), Amdahl `s = 0.059`.** CPU-seconds inflate 63%
  over the same span, which is the memory-stall term, not a code defect.
- **Done (2026-07):** quickselect median in `_profile_snr` (was 41% of runtime →
  7.5%) and a type-stable `Workspace{S,B,D}` (killed hot-loop dynamic `mul!`
  dispatch) — together ~1.6× warm single-thread, results unchanged. See §2 of
  `Summary_and_Future_Work.md`.
- **Done (2026-08-08):** direct `O(m)` interpolation replacing the FFT
  correlation (`src/directinterp.jl`, `--interp direct`, now the default). The
  interp FFTs — 76% of the chunk fill, ~50% of runtime — are gone; **1.64×
  end-to-end** and ~1e-10 accuracy where the FFT path had ~1e-2.
- **Done (2026-08-09): default `m` 32 → 16, plus a `--m` flag.** Justified by the
  Monte Carlo in `../coherent_search/examples/interp_accuracy_vs_m.md`: the
  recovered signal-power fraction is `S_m = Σ sinc²(dr−k)`, so the loss is
  `≈ 0.203/m` averaged over sub-bin offset — 1.27% at `m=16` vs 0.63% at `m=32`,
  against the ~6.5% the `hidr=0.5` grid already costs at the top harmonic.
  Confirmed end to end: the 7.1185 Hz pulsar's metric moves 12.034 → 12.001
  (−0.27%, predicted ≈0.3%); at `m=8` it is 11.894 (−1.16%, predicted ≈0.95%).
  **`m` is not where the remaining time is.** The interp bucket is 18.8% at
  `m=32` → 15.0% at `m=16`, i.e. **~3.6% end-to-end** — halving `m` cuts interp
  time only ~19%, not 50%, because the direct interpolator is dominated by fixed
  per-point cost rather than the `m`-term sum (`bench/interp_bench.csv`
  `m_sweep`: 1.49e-4 s at `m=8` → 4.44e-4 s at `m=128` for 2048 points, i.e.
  `t ≈ 1.47e-4 + 2.4e-7·m`). Below `m=16` the accuracy cost rises faster than the
  time falls, so 16 is the knee.
  Note end-to-end wall-clock cannot resolve this: run-to-run scatter is ~9%.
- **Done (2026-08-09): cross-profile SIMD on the `:boxcar` gate — 1.26x
  end-to-end**, byte-identical candidates. The width×phase scan vectorised along
  *phase*, which is only 120 long at k=1 and 20 under decimation; the batch axis
  (`Nprof=2048` profiles, the columns of `profs`) is always long. `_boxcar_gate!`
  transposes a `B`-profile tile to `(B, nbins)` and runs the same recurrence with
  `b` innermost — the prefix sum's serial chain becomes `B`-wide and the phase
  max needs no horizontal reduce. Gate kernel **2.77x** (`Float64`) / **4.05x**
  (`Float32` tile, what ships). Two traps: **`B` must be a `Val`** — with a
  runtime `Int` batching is *slower* than the scalar path it replaces — and the
  `Float32` is sound only *because it is a gate* (error ≤7e-7 vs the
  `boxcar_medmargin=2.0` slack, so it cannot move which trials get scored
  exactly). `bench/boxcar_bench.jl` re-measures both axes; two new pins in
  `test/test_search.jl` cover the gate, which previously had none.
- **Done (2026-08-09): `_block_sigma` 839 → 289 µs/chunk (2.90x), 1.08x
  end-to-end**, bit-identical σ̂. (a) The strided gather recomputed `(i,j)` from a
  linear index the array already had — `size(M,1)==nbins`, so `M[i,j]` *is*
  `M[t]`; a guard now enforces that. Worth only 1.12x. (b) The real cost was
  **branch misprediction in `_select!`**: two 8192-sample quickselects were 93%+
  of the function, and a branchless Lomuto (`i += (x <= pivot)`, unconditional
  swap) is 3.62x at n=8192. It is **size-gated at `_SELECT_BRANCHLESS_MIN=256`**
  because it is 0.94x at n=120 — the `:non`/`:sd2` per-trial median that runs
  ~1e8 times. Selection returns a unique order statistic, so all medians are
  bit-identical (verified vs the old partition on tied/all-equal/sorted inputs).
  **Note the profiler charges `_median!` to `median-select` regardless of caller**,
  which is why `_block_sigma` read as 3.7% while really being ~20%.
- **Done (2026-08-11): start-up, not the search, was dominating short runs.**
  Measured on `PM0063_034C1_DM445.0_red.fft`, `-t 1 --hifreq 20 --nharms 32
  --noplot`: 15.65 s wall for **1.36 s** of actual searching. Phase breakdown —
  boot + `using` 0.14 s (loading was never the problem), inferring `main` 4.7 s,
  ArgParse's first `parse_args` 1.6 s, first `search` call 5.9 s. A warm second
  `main()` in the same process is **0.05 s**, which is both the floor and the
  marginal cost of an extra file. Three changes, all measured:
  - **`PrecompileTools` workload** at the bottom of `src/CoherentSearch.jl` (a
    miniature end-to-end search + a `main` call): 15.65 → 8.13 s. Costs +3.4 s
    of re-precompilation per `src/` edit (6.05 → 9.47 s).
  - **CLI moved into the package** (`src/cli.jl`), so `main` and the ArgParse
    table are cached too: → **2.39 s**, candidates byte-identical. **6.5x.**
  - **Multi-file runs + `SearchCache`** (reuses hplans/workspaces across files;
    `dplans` still per-file, they depend on `r_lo`): 3 files in 4.83 s, i.e.
    ~1.2 s marginal per file vs 15.65 s for a separate invocation.
  **The biggest single fixed cost left is plotting, which is on by default:**
  `using CairoMakie` alone is 9.0 s. Plotting is now deferred to one pass after
  all searches, so a batch pays it once, and bulk runs should use `--noplot`.
- **`SearchCache` is already DM-safe — glob a whole DM range into one
  invocation** (`... NGC6624_16L_DM??.??_red.fft`). `_plans!` keys reuse on
  `cache.params === params && cache.Nprof == Nprof` and **never consults the
  file**, so hplans and the `FFTW.MEASURE` workspaces are shared across any file
  list. A dedispersion plan gives neighbouring DMs identical `N` and `dt`, hence
  identical `T`, `r_lo = lofreq*T` and Nyquist — only the noise values differ —
  so the `dplans` caveat above is vacuous across such a group *and* immaterial
  anyway: `build_direct_plans` is **0.29 ms** at nharms=60. Measured on ten
  NGC6624 `16L` DMs (T=26459 s), narrow band, `-t 8`: 1 file 2.91 s, 10 files
  **12.88 s = 1.11 s marginal each**, against 29.1 s for ten separate
  invocations (2.3x).
- **Mixing files with *different* `N`/`dt` in one invocation is also correct, and
  needs no guard.** Everything cached is a pure function of `(params, Nprof)`:
  `build_harmonic_plans` never sees the file, and `direct_window_size` is
  `ceil((Nprof-1)*hidr) + m + 4`. Everything file-scaled — `r_lo = lofreq*ft.T`,
  `dplans`, the trial ranges, the `Nhalf = ft.N÷2` Nyquist guard — is recomputed
  per `search` call. **Verified, not just read:** one invocation over PM0063
  (T=2097 s) and NGC6624 (T=26459 s), a 12.6x span, produces `.cohout` files
  byte-identical to running each alone. So a heterogeneous glob costs 0.29 ms
  extra per file, not a cache rebuild — do not add a warning saying otherwise.
- **The genuinely silent case is Nyquist, not the cache.** When a harmonic runs
  past `ft.N÷2`, `fill_harmonic_row_direct!` returns early and leaves that row of
  `ftprofs` **zero** — no error, no warning. That is deliberate (it is how the
  search degrades at the top of the band), but it means a band chosen for a
  finely-sampled file will quietly lose harmonics on a coarser-sampled one in the
  same glob. If a diagnostic is ever wanted for mixed-`dt` runs, this is where it
  belongs.
- **But use `--noplot` for such a run, for a second reason beyond CairoMakie's
  9 s.** With plotting on, `main`'s deferral pushes `(ft, cands, stem)` into
  `toplot` and plots only after every search, so **every input's mmap stays live
  to the end of the run** — 50 NGC6624 files is 69 GB of mmap pinned at once
  (`FFTFile` mmaps the whole `.fft`; `src/fileio.jl`). With `--noplot` each `ft`
  is collectable as its iteration ends.
- **The PackageCompiler sysimage (`sysimage/`) is worth it *only* for plotting
  runs — this was projected as the big win and measured as almost nothing.**
  Warm: `--noplot` 2.3 s with it vs 2.4 s without (nothing — the precompile
  workload already took that), but with plots 7.4 s vs 18.7 s (**2.5x**). It
  removes CairoMakie's load and nothing else, because Julia's boot is only
  ~0.2 s and the search is already cached. Costs: ~28 min to build, 1.14 GB, and
  the first run after a build/reboot is **23 s** paging the image in (vs 2.3 s
  warm) — an occasional single search is *slower* with it than without.
  Do not use it while editing `src/`: a sysimage freezes the code it was built
  from and will silently run the old search.
- **Thread scaling cannot be measured on the laptop — use the workstation
  (which is where it has now been measured; see `bench/thread_scaling.jl`
  above, 9.9x at `-t 20`).** On the laptop, at the riptide bench config,
  `-t 1/2/4/8` gave 29.8/19.7/15.8/15.1 s: only 1.88x from 4 threads. That was
  the machine, not the code. CPU-seconds for *identical work* inflate
  30 → 38 → 54 → 82 s, and the clock falls 2400 MHz (`-t 1`) → 1800 MHz
  (`-t 4`, exactly the i7-10510U base) on 4 physical cores that also carry the
  desktop's own ~2 cores of load. Clock-normalised, 4 threads deliver ~2.6x
  (~66% efficiency). **Measure CPU-seconds and the clock before blaming the code
  for poor scaling** — the workstation numbers make the same point in the other
  direction, since its CPU-seconds still inflate 63% across the sweep.
- **Chunk→thread assignment is whole-chunk round-robin** (`c = t; c += nt` in
  `_search_region!`), never per-trial, which is what lets
  `fill_harmonic_row_direct!` load each harmonic's bin window *once per chunk*
  (0.3 KB at h=1 to 8.1 KB at h=60, 251 KB total at nharms=60/Nprof=2048) and
  have every trial read it back from L1. **Do not shrink `--blocksize` to
  relieve cache pressure** — tried, and it is *worse* at every thread count
  (2048/512/256 → 29.3/31.9/37.0 s at `-t 1`): per-chunk fixed costs (batched
  `brfft` efficiency, `_block_sigma`'s 8192-sample subsample) dominate. The
  per-thread workspace is 3.78 MB vs 8 MB shared L3, which *looked* like the
  scaling culprit and is not.
- **Next target: Kadane** — whose own first-listed step (cross-profile SIMD on
  the exact scan) is now done, so it is a smaller prize than the write-up assumes.
  Branch `float32-profiles` narrows the profile stage to `Float32` through a
  `ProfT`/`CProfT` alias pair at the top of `src/search.jl` — set them back to
  `Float64`/`ComplexF64` and it is master, which is what makes it cheap to A/B.
  Rebased onto master 2026-08-15 (so it has the start-up work); 423/423 tests
  pass and candidates stay byte-identical to master.
- **The `Float32` profile stage is a *thread-count-dependent* win, and at `-t 1`
  it is a loss (2026-08-15).** Both earlier verdicts were right about their own
  configuration and wrong as generalisations. Interleaved A/B, two git worktrees
  so neither arm invalidates the other's precompile cache, PM0063 at the riptide
  bench config plus `--ncands 300 --threshold 6.3`:

  | threads | master | f32 | f32 vs master | master CPU-s | f32 CPU-s |
  |---|---|---|---|---|---|
  | 1 | 31.1 s | 35.2 s | **0.88x** | 30.6 | 34.7 |
  | 2 | 18.4 s | 19.9 s | 0.93x | 33.6 | 37.0 |
  | 4 | 11.1 s | 11.1 s | 1.00x | 37.3 | 37.4 |
  | 8 | 6.99 s | 6.80 s | 1.03x | 42.3 | 40.9 |
  | 16 | 5.10 s | 4.49 s | **1.14x** | 51.4 | 45.0 |
  | 20 | 4.64 s | 4.28 s | 1.08x | 55.5 | 47.7 |

  The CPU-seconds give the mechanism: `Float32` does **13% more** CPU work at
  `-t 1`, and wins only by *stalling less* once cores contend for bandwidth —
  master's CPU-seconds inflate 81% from 1→20 threads, f32's only 37%. Crossover
  is ~4 threads. Scott's `-t 8` NGC6624 runs (227 → 217.5 s, 4.2%, distributions
  non-overlapping) sit exactly on this curve, as does the original laptop 1.05x
  at `-t 4`. So it is not a scatter artefact and never was — it is a different
  point on a curve nobody had swept.
- **Which makes the merge decision depend on how searches are actually
  deployed.** A production search over many DMs usually gets its parallelism for
  free by running one *single-threaded* process per DM, in which case throughput
  is governed by `-t 1` CPU-seconds — where `Float32` is 13% **worse**. Do not
  merge `float32-profiles` on the strength of the threaded number alone; decide
  the deployment model first (see §3.1 of `Summary_and_Future_Work.md`). The open
  optimisation is to find and remove f32's extra `-t 1` CPU cost, which would
  make it a win on both axes. Diagnosed so far (§3.1): it is **not** the
  transform (the `Float32` batched `brfft` is 1.14x *faster* timed alone) and
  **not** a failed SIMD widening (`src/directinterp.jl` is byte-identical between
  the arms — the `m`-sum is `Float64` on both). Both regressions sit on
  `ftprofs`, whose store narrowed to `ComplexF32`.
- **Profile the band you benchmarked.** `bench/profile_search.jl` defaults to
  5–30 Hz, and over that band the two Float32 arms differ by 0.6% — the whole
  effect lives below 5 Hz, where red noise pushes trials past the boxcar gate
  into the exact-median rescan. Pass `FILE.fft 33.3333 0.1` to match the riptide
  bench config.
  Keep the two Float32 results distinct: the *profile stage* is the above;
  adding Float32 to the *interpolation* on top was 7% *slower* than master and
  is still not understood (CPU/`m=16`-specific; revisit for a GPU port).
- **Measurements in these docs are pinned to the config current when taken —
  re-check before reusing one.** Three projections have now been wrong because a
  default moved out from under a recorded number: the `1.35x` for a fully-`Float32`
  direct inner loop (measured at `m=32`, default is now 16), the `1.46x` for a
  `Float32` metric kernel (measured against a `Float64` gate, but the shipped gate
  already uses a `Float32` tile), and the smooth-`fftlen` ceiling. When quoting a
  figure from `Summary_and_Future_Work.md`, check what `m`, `nharms`, `maxdecim`,
  metric and interpolator it was taken under, and re-measure if any have moved.
  The same staleness applies to instructions that name a version or model — say
  what the thing *is* ("the model that wrote the commit"), not today's value.
- **Twice now, a plausible mechanism with an order-of-magnitude estimate that
  *matched the measured total* has been wrong** (smooth `fftlen`; `_block_sigma`'s
  `idiv`). Split the function and measure the phases before optimising one.
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
