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
  **Reference only** — `fourier_interp` (exact, per point) and `finterp_fft`
  (FFT-correlation onto a fine grid, the Python original's method) are what
  `reference_profiles` and the cross-validation use. Nothing here runs in a
  search.
- `src/directinterp.jl` — **the interpolator.** Evaluates Eqn. 30 exactly at each
  trial: the coefficients factor as `A(dr)/(dr-j)` with *real* `1/(dr-j)`, and
  only `q = 2·nharms` distinct `dr` occur in a whole search, so the weights are
  tabulated once per harmonic and indexed by an exact integer residue. No FFT, no
  fine grid, no linear interp. ~3.8× faster than the FFT-correlation path it
  replaced *and* ~1e-10 vs its ~1e-2 accuracy.
- `src/fileio.jl` — mmap'd PRESTO `.fft` reader + `.inf` parser. Amplitudes are
  `ComplexF32`; element 1 packs DC.re + Nyquist.im.
- `src/search.jl` — the core. **Two paths that must agree:**
  - a simple *reference* path (`block_metrics`/`reference_profiles`) kept
    deliberately unoptimised and pinned to the Python oracle at ~1e-15;
  - an *optimised* production `search`: chunk-parallel (`@spawn`, one private
    `Workspace` per task), cached per-harmonic interpolation tables, a batched
    inverse FFT, and harmonic decimation for cheap multi-frequency search.
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
oracle/equivalence pins green. There are three, and they chain:

- **Python oracle** (`crossval/crossval_accuracy.jl`, needs the sibling repo and
  a Python that can import it): `finterp_fft` at 3.9e-16, `reference_profiles`
  at 1.4e-16, and `snr_metrics` (the boxcar matched filter) at 7.0e-17. Run it
  after touching anything in `fourierinterp.jl`, `reference_profiles`, or the
  metric. **The oracle prefers the sibling repo's `src/` over any installed copy
  and prints the path it used** — a stale non-editable install once pinned this
  comparison to a superseded metric with no visible sign.
- **End-to-end equivalence**: `chunk_metrics` (the whole optimised machinery)
  against `block_metrics(...; kernel=:direct)`, at **8.4e-16**. Both sides
  evaluate the exact Eqn.-30 kernel — the reference point by point, the
  production path through its tabulation — so the residual is rounding, not
  method. The same testset also asserts the `kernel=:fft` reference is orders of
  magnitude further away, which is the accuracy the production interpolator buys.
- **Interpolator**: `fill_harmonic_row_direct!` against `fourier_interp` at
  ~1e-11, plus a **bit-exact** chunk-invariance test (a chunk starting at global
  trial `t0` must reproduce one long chunk exactly).

A change that should not move results is checked by `diff` on the `.cohout`, not
by eyeball: the production path is deterministic and chunk-invariant, so
byte-identical candidates is the normal standard.

## Commands

```sh
# Tests (the correctness gate)
julia --project=. -e 'using Pkg; Pkg.test()'

# Search (-t auto for all cores).  The bare defaults are a full blind search:
# 0.1-125 Hz fundamentals, --nharms 60 --maxdecim 6 => spin coverage to 750 Hz,
# and no plotting (--plot opts in; --noplot is accepted and ignored).
julia --project=. -t auto bin/coherent_search.jl FILE.fft
# Many files in ONE invocation (start-up + plans paid once; one .cohout each)
julia --project=. -t auto bin/coherent_search.jl *_red.fft --threshold 8
# Production sysimage (build takes minutes; freezes src/ — do not use while editing)
julia --project=sysimage sysimage/build_sysimage.jl
julia --sysimage sysimage/coherent_search.so --project=. -t auto bin/coherent_search.jl ...
# The standard perf test (this is now close to the defaults; --hifreq pins the
# band so timings stay comparable to the recorded ones):
julia --project=. -t 4 bin/coherent_search.jl --threshold 6 \
      --lofreq 0.1 --hifreq 33.3333 -o out.txt FILE.fft
# Trial grid, chunking and interpolation phase-cycle lengths
julia --project=. bin/coherent_search.jl --verbose ... FILE.fft
# Narrowed profile stage (measured a loss at every thread count -- see below)
julia --project=. bin/coherent_search.jl --precision f32 ... FILE.fft

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

That table is the **laptop**. The same command on the 20-core workstation
(`fitzroy`) gives rseek 20.4 s, ours `-t 1` 32.3 s, `-t 20` 4.6 s — i.e. **1.59x
slower**, not 1.24x. Quote the machine with the ratio; the two hosts disagree by
more than any optimisation discussed below.

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
- **The frequency grids are matched exactly, and this is not a coincidence
  (settled 2026-08-16).** riptide's FFA at base period `b` samples emits
  `m = n/b` shifts spanning periods `b`…`b+1`, so its trial step is
  `dr = n/((m−1)·b²) = 1/b` Fourier bins — **one phase bin of drift across `T`**,
  and *not a tunable*: it is what the transform produces. Our `--hidr` is the
  step of the *highest* harmonic, so the fundamental steps by `hidr/nharms` and a
  decimation-`k` pass (reporting `k·rf`, `nbins = 2·nharms/k`) steps by
  `k·hidr/nharms = 1/nbins` at `hidr = 0.5`. **Same rule, at every `k`.**
  Verified against riptide itself (`ffa_search` on random data, reading
  `pgram.freqs`/`pgram.foldbins`): measured `dr·b = 1.0001–1.0002` at
  `b = 20, 40, 60, 90, 120`. So `--hidr 0.5` with `nharms = bmax/2` *is* the
  FFA's native resolution — **there is nothing to fix in the trial grid, and
  lowering `hidr` to "be safe" would silently double the work for no reason.**
- **What is NOT matched is folds per frequency, and it is worth ~2.8x.**
  riptide folds each frequency **once**, at whatever `b` its downsampling ladder
  lands on (sawtoothing 20→120 within each cycle: `b = 36` at 3 Hz, `b = 109` at
  1 Hz, `b = 92` at 7.1 Hz — depth and resolution are what the ladder gives, not
  a choice). We fold every frequency below `hifreq` **six times**, because
  decimation `k` covers `[k·lofreq, k·hifreq]` and those overlap. On the bench
  config that is **2.81x the profiles, 2.47x the folded bins, 3.59x the boxcar
  (bins×widths) work** — measured, not estimated; `compare/compare_riptide.py`
  now prints this table before it times anything, so the ratio is never quoted
  bare. **Quote the timing ratio against the work ratio.**
- **The overlap is our harmonic-sum ladder — do NOT "optimise" it away.** It
  looks like pure redundancy below `hifreq` (a signal at 7 Hz is fully covered by
  `k=1`), and restricting each `k` to a disjoint band would cut the `k≥2` work
  2.94x — measured 31.7 s → ~22 s at `-t 1`, which would make us *faster* than
  `rseek`. It also **costs real sensitivity**: measured `--maxdecim 6` vs
  `--maxdecim 1` on PM0063, the 7.1185 Hz pulsar scores **12.97 at `k=6`
  (`H=10`, a 20-bin fold) but only 11.89 from `k=1` alone** (`H=60`). The shallow
  folds are exactly PRESTO's 1/2/4/8/16-harmonic ladder, and the boxcar scan on
  the deep fold does *not* subsume them. The headline "we detect more strongly
  than riptide" depends on them. (The `k≥2` passes cost 14.6 s of 31.7 s at
  `-t 1` — 46%, matching the phase timers' decim-brfft + decim-metric.)
- **`bench` puts riptide outside its own documented operating range, and
  `--preset matched` is the fix.** `ffa_search`'s docstring says `bins_max`
  should be "approx. 10% larger" than `bins_min`; riptide's example pipeline
  config uses 240/260 and 480/520. Our `bmin 20 / bmax 120` is a factor of 6, so
  `b` sawtooths across the whole range inside every downsampling cycle, dropping
  its mean trial density to ~43 per Fourier bin instead of ~`b`. That is *forced*
  — one `rseek` invocation cannot reach 200 Hz on this data with `bins_min > 20`
  — but it is not the regime riptide is written for. **`--preset matched`
  (`bmin 120 / bmax 130`, which derives `maxdecim = 1`, `nharms = 65`) runs one
  fold depth per side over 0.1–33.3 Hz**: same band, same `dr = 1/nbins` grid,
  and at `bins_min = 120` riptide's boxcar bank finally matches ours (9 widths,
  30% duty). Work agrees to 4–8%. Measured 2026-08-16, workstation, `-t 1`,
  interleaved: **rseek 18.35/18.33 s, ours 17.86/16.45 s — we are ~1.05–1.11x
  FASTER at equal work.** The `bench` deficit is the ladder, not the code.
  **But rseek wins the pulsar in that config: S/N 12.6 (`w=13`, ducy 10.3%) vs
  our 11.89 at the same depth.** Our 12.97 comes from the `k=6` fold, so
  "we detect more strongly" is a statement about the *ladder*, not about the
  per-fold detector.
- **riptide is width-limited at large `b`, and that is why its S/N is lower.**
  `generate_width_trials` is called with **`bins_min`**, and `rseek` hardcodes
  `ducy_max = 0.3`, so the bank is `[1,2,3,4,6]` for *every* profile regardless
  of `b`. At `b = 92` that reaches only 6/92 = 6.5% duty — and on the reference
  observation `rseek` reports the ~10%-duty pulsar at `w=6`, its maximum, ducy
  6.52%. Our `boxcar_widths` takes each profile's own `nbins` (same `fsp=1.5`,
  same `maxfrac=0.3`), so we reach 30% duty at every depth: 9 widths at 120 bins
  against riptide's 5. Some of the S/N gap is this, not the algorithm.
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
  CPU-seconds panel; CSV + PNG), `precision_ab.jl` (`:f64` vs `:f32` in **one
  process**, wall clock plus the in-situ `phase_times` split — the first thing to
  run when a phase looks suspicious), `chunkfill_bench.jl` (splits
  `fill_chunk_profiles!` into zeroing / interp / transform, and times the
  decimated transforms, on the production `Workspace`). Run single-threaded (`-t 1`)
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
  correlation (`src/directinterp.jl`). The interp FFTs — 76% of the chunk fill,
  ~50% of runtime — are gone; **1.64× end-to-end** and ~1e-10 accuracy where the
  FFT path had ~1e-2. (The `--interp fft` fallback it left behind was deleted on
  2026-08-16; see the retirement entry below.)
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
  because it is 0.94x at n=120 — the per-profile baseline median in
  `_boxcar_exact`, run once per trial that clears the gate. Selection returns a unique order statistic, so all medians are
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
  **The biggest single fixed cost left is plotting, which was on by default when
  this was measured:**
  `using CairoMakie` alone is 9.0 s. Plotting is now deferred to one pass after
  all searches, so a batch pays it once. **Plotting is off by default as of
  2026-08-16** (`--plot` opts in), for this reason and the mmap one below.
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
- **The second reason plotting is off by default, beyond CairoMakie's 9 s.**
  With plotting on, `main`'s deferral pushes `(ft, cands, stem)` into
  `toplot` and plots only after every search, so **every input's mmap stays live
  to the end of the run** — 50 NGC6624 files is 69 GB of mmap pinned at once
  (`FFTFile` mmaps the whole `.fft`; `src/fileio.jl`). Without `--plot` each `ft`
  is collectable as its iteration ends.
- **The PackageCompiler sysimage (`sysimage/`) is worth it *only* for plotting
  runs — this was projected as the big win and measured as almost nothing.**
  Warm: no-plot 2.3 s with it vs 2.4 s without (nothing — the precompile
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
- **Done (2026-08-16): the decimation gather is gone — 1.12x at `-t 1`, 1.26x at
  `-t 4…16`, candidates byte-identical.** `decim_pass!` used to copy every `k`-th
  row of `ftprofs` into a compact `(Hₖ+1, Nprof)` buffer and transform that. The
  copy read exactly the elements the transform then read again. Rows
  `1, k+1, 2k+1, …` of `ftprofs` **are** the decimated stack (DC included — the
  search never writes row 1), so `DecimBuf.src` is now a stride-`k` *view* and
  FFTW takes the stride. Standalone over `k=2…6`: **1.36x** (`Float64`) /
  **1.60x** (`Float32`) vs gather-then-transform. End to end (PM0063, riptide
  bench config, warm in-process, median of 3): `-t 1` 30.14 → 26.99 s, `-t 4`
  9.98 → 7.92, `-t 8` 5.59 → 4.46, `-t 16` 3.36 → 2.66. It also deletes
  `Σₖ (Hₖ+1)·Nprof` complex words per workspace, which is why the win grows with
  thread count.
- **Done (2026-08-16): the ladder's redundant `(k, W)` corners are pruned —
  1.07x at `-t 1`, no measurable sensitivity cost** (`ladder_boxcar_widths`).
  A `W`-bin boxcar on an `M`-bin profile weights harmonic `h` by the Dirichlet
  kernel, which depends on `W` and `M` only through the duty `δ = W/M`. Since
  `M = 2·nharms/k`, the *filter shape* is a function of `δ` alone and the fold
  only sets where the harmonic sum truncates — so `(k, 2W)` and `(2k, W)` are the
  same filter, differing only in the harmonics past the kernel's first null.
  Both corners of the grid are therefore dead weight: **`W = 1` on any fold but
  the deepest** (same duty as `W = 2` one rung deeper, at half the resolution)
  and **wide `W` on any fold but the shallowest** (same duty, but dragging in
  harmonics that carry noise and no signal). Keeping `2 ≤ W ≤ 6`, plus `W = 1`
  on the deepest fold and the full tail on the shallowest, is 25 of 38 pairs at
  the defaults: **0.65x the boxcar scan work at 0.00% modelled S/N loss.**
  - **The model must use Gaussian pulses, not boxcars.** With a boxcar-shaped
    *signal* the analysis comes out backwards (it declares the shallow folds
    dominated) because a boxcar has harmonic content at every `h`, which is the
    best possible case for a deep fold. Boxcars are the *filter* — chosen
    because their noise statistics are tractable — not a profile any pulsar has.
    Re-run over Gaussian, scattered-Gaussian (τ = 0.5–2× duty), two-component and
    interpulse profiles at 60 duty cycles, the loss is 0.00% and the retained set
    strictly contains the greedy zero-loss optimum. The same model puts a
    `k=1`-only search **8.40%** below the full ladder against the **9.1%**
    measured on the 7.1185 Hz pulsar — so it reproduces the one number we have.
  - **Measured, and much smaller than the scan-work cut suggests:** interleaved
    `-t 1` A/B with a warm-up between checkouts, PM0063 at the bench config,
    32.35/32.80/33.57 s → 30.24/30.64/31.07 s, i.e. **1.07x**. A 0.65x cut to a
    phase billed at 47% "should" have been ~1.2x; it is not, because the prefix
    sums, the median/σ̂ and the gated exact rescan inside that 47% are untouched.
    **Another projection-vs-measurement gap — the width scan is ~20% of runtime,
    not 41%.**
  - Candidates are **not** byte-identical, by design: everything above S/N 6.3 is
    unchanged (the pulsar stays at 12.97), and the five that vanish are exactly
    the pruned corners at S/N 6.0–6.3 — noise at a threshold of 6. Dropping them
    lowers the trials factor, so it slightly *improves* the statistics.
  - **The bank is derived from the decimation set, never hardcoded.** With
    `decimations == [1]` nothing is pruned, which is what keeps the equivalence
    gate byte-identical; a `k=1`-only search with a pruned bank would be 8.4%
    worse. A sparse ladder (`[1, 6]`) is also left alone — the cap of one rung
    must reach the floor of the next (`k′ ≤ 3k`) or a duty-cycle hole opens.
    `block_metrics`/`snr_metrics` gained a `widths` kwarg so the reference path
    can stand in for a pruned fold; `snr_metrics` still defaults to the full bank,
    which is what the Python oracle is pinned to.
- **In-situ phase timers are now permanent** (`phase_reset!` / `phase_times`,
  `PHASE_NAMES`; ~0.03% of runtime, one `time_ns` pair per phase per chunk).
  `bench/precision_ab.jl` prints them alongside a wall-clock A/B. Use them
  *before* reaching for a profiler bucket table: the bucket classifier lumped the
  base and decimated transforms into one "FFTW" row that was hiding a −22% and a
  +52% inside a reported +12.4%, and it charged `fill!(ws.ftprofs, 0)` (0.29x in
  `Float32`) to the `interp` bucket.
- **`SearchParams.precision` (`:f64` default, `:f32`) replaces the
  `float32-profiles` branch.** The profile-stage width is a type parameter on
  `Workspace{…,P}` / `DecimBuf{…,P}`, so both widths live in one build and A/B in
  one process — which removes the precompile confound that once produced a
  confidently wrong answer, and makes `--precision f32` a flag rather than a
  checkout.
- **`Float32` profiles: do not merge, at any thread count (2026-08-16).** The
  2026-08-15 verdict below ("a thread-count-dependent win, crossover ~4 threads")
  was measured *over the gather* and no longer holds. With the gather removed,
  `:f32` vs `:f64` is **0.82x at `-t 1`, 0.92x at `-t 4`, 0.98x at `-t 8`, 1.01x
  at `-t 16`.** `Float32` never made the search faster — it relieved a bandwidth
  problem, and that problem has been fixed at its source, so the threaded win
  went with it. The deployment-model question (§3.1) therefore no longer decides
  anything.
- **Fully-`Float32` interpolation is 1.64x SLOWER, and the reason generalises.**
  Narrowing `DirectPlan`'s `W`/`A` and the accumulators (the last piece of "make
  everything `Float32`") costs 1533 → 2482 µs per chunk, against the 1.09x that
  narrowing only the *store* into `ftprofs` costs. At `m = 16` the `Float32` sum
  is *exactly one* 16-lane vector — one FMA, no ILP, then a **four**-stage
  cross-lane reduce — where `Float64` gets two independent 8-lane accumulators and
  a three-stage reduce. **The per-trial horizontal reduce is what this loop pays
  for**; narrowing *or* widening the `m`-axis both make it worse (this is the same
  mechanism as the earlier 8-lane experiment). `build_direct_plans(WT, …)` keeps
  the knob, but the search always asks for `Float64`.
- **Four plausible mechanisms for the decimated-transform regression, all
  measured, all wrong** — record them so they are not re-guessed: (1) *leading-dimension
  alignment* — `k=4` gives `Hₖ+1 = 16`, perfectly 64-byte aligned in `Float32`,
  and was the *worst* of the five; padding fixes nothing. (2) *The transposed
  layout* `(Nprof, Hₖ+1)` transforming along dim 2, giving FFTW batch stride 1 —
  **2–3x slower** at every size in both precisions, so the current layout is
  already the right one for the transform. (3) *The data* — genuine chunk
  contents vs `randn` of the same shape agree to <1% (no subnormals). (4) *Cold
  caches* — flushing 32 MB between calls makes `Float32` relatively **better**.
- **"The whole regression lives below 5 Hz" was wrong** (it was profile
  attribution, not the search): `interp` regresses +20% at 5–13 Hz just as at
  0.1–8 Hz. Band choice changes the *mix* (how much exact-median rescan runs), not
  the interp penalty. `bench/profile_search.jl` still defaults to 5–30 Hz; pass
  `FILE.fft 33.3333 0.1` to match the riptide bench config.
- **Where the single-thread time is now** (`:f64`, PM0063, 0.1–33.3 Hz, `-t 1`,
  29.5 s of phases): decim-metric 29%, interp 24%, decim-brfft 18%, gate+metric
  18%, base brfft 9%. **The boxcar metric work is 47%** across the two metric
  rows — the largest remaining target, and larger than the interpolation it was
  long assumed to sit behind. For `interp` (24%), the identified structural cost
  is the per-trial horizontal reduce, and the way out is vectorising across
  *trials* (as the boxcar gate did for 2.77–4.05x), not a wider `m`-axis; see
  §3.1 for the periodicity that makes a trial-ordered weight table possible and
  for the part that has not been prototyped.
- **Historical (2026-08-15), superseded by the two entries above but kept because
  the *shape* of the curve was real:** with the gather still present, `Float32`
  was 0.88x at `-t 1`, 1.00x at `-t 4`, 1.14x at `-t 16`, doing 13% more CPU work
  at `-t 1` and winning only by stalling less under contention. Both of the
  *earlier* verdicts (a laptop 1.05x at `-t 4`, a "do not merge" at `-t 1`) were
  right about their own configuration and wrong as generalisations — which is the
  standing lesson, now applied a third time to the 2026-08-15 numbers themselves.
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
- **Smooth `fftlen` sizing: moot since 2026-08-16** (it sized the interpolation
  transforms, which no longer exist), but the *lesson* stands and is why the
  in-situ phase timers exist: per transform it really was 1.26×, and in a real
  search it was a wash to −7%, because specific smooth lengths (`3^k`-heavy: 6561,
  13122, 15309) are worse than the power of two and the shared `:pow2` scratch
  sets stayed cache-warm where 8–16 smooth ones did not. **An isolated per-size
  FFT benchmark does not predict this workload.**
- **`ComplexF32` interpolation is settled, not open.** It existed to halve the
  bandwidth of transforms that no longer run; the direct path already reads
  `ComplexF32` bins natively. The one remaining question — `Float32` weights in
  the direct inner loop — was measured on 2026-08-16 at **1.64× slower**; see the
  entry above for why (the per-trial horizontal reduce, not the data width).
- **HPK / KFR (see `~/programming/fft_tests/HPK_JULIA_HANDOFF.md`): deferred.**
  Its own prerequisite was to profile the FFT fraction first; done, and FFTW now
  falls to the ~4% batched profile `brfft`. 1.5× on 4% does not justify a
  proprietary binary-only dependency. Revisit only if `:direct` is backed out.

## Retired, on purpose — do not reintroduce

Both of these were removed on 2026-08-16 after a review pass; each had been
strictly dominated for a while and was still costing something to carry.

- **The production `:fft` interpolator** and everything that planned it
  (`interp_tile!`, `fill_harmonic_row!`, `FFTScratch`, `HarmonicPlan`,
  `build_harmonic_plans`, `harmonic_plan_report`, `harmonic_numbetween`, and the
  `interp`/`fftsizing`/`align` params with their CLI flags). The FFT correlation
  itself lives on in `src/fourierinterp.jl` (`finterp_fft`) and in
  `reference_profiles(...; kernel=:fft)`, which is what the Python oracle is
  pinned to — **that is a different thing from the deleted search path, despite
  the shared name.** Removing it also dropped `Workspace`'s `S` parameter and
  `scratch::Dict`, and `hplans` from `fill_chunk_profiles!`/`_search_region!`/
  `_plans!`/`SearchCache`.
- **The `:non`/`:sd2` on-pulse metrics** (`_profile_snr`, `xsignal`, `pexp`,
  `--metric`). Upstream replaced `snr_metric` with the boxcar matched filter as
  its *only* metric, so they had no oracle left, and `:boxcar` had been the
  default here since it was written.

`src/` went 3798 → 3514 lines; candidate output stayed byte-identical and warm
`-t 1` wall clock was a wash in an interleaved A/B.

**Two traps this exposed, worth remembering:**

- `prime_wisdom` builds a `Workspace` purely for its FFTW-planning side effect,
  and nothing tested it, so it silently kept calling a deleted function. It has a
  test now. When changing a constructor, grep for callers that use it only for
  effect.
- The pooled block `σ̂` is an exact MAD in Python and a `_BOXCAR_SIGMA_SAMPLES`
  subsample in the search. `snr_metrics` defaults to the exact estimator
  (oracle-faithful); `block_metrics` passes `sigma_samples` to match the
  production one (equivalence-faithful). Previously both were the same accidental
  constant, which would have started lying silently the moment either moved.

## Known follow-up: the test signal is too bright

`../coherent_search/examples/harmonics_hi.fft` (the 10.0123456789123 Hz fake
pulsar, `T = 1000 s`) is far stronger than anything a search would really face,
and its harmonic content runs well past harmonic 60. That is what makes the
default `--nharms 60` search report it at **11f rather than f**: folded at 11f it
scores 31.92 against the fundamental's 28.97, so harmonic collapse — which keeps
the strongest family member — quite correctly picks the harmonic. Real
observations do not behave this way (scattering and finite time resolution bound
the harmonic content), so this is a property of the test fixture, not a defect.

**The fix is to regenerate the fixture fainter**, in the sibling Python repo.
Before doing that, note what leans on its brightness:

- Several `test/test_search.jl` pins assert `!isempty(cands)` at `threshold=8.0`
  over 9.5–10.5 Hz; a much fainter signal could drop below and turn them into
  flaky tests rather than failing ones.
- `test_search.jl` also pins the chunk-size drift and duplicate/harmonic-collapse
  behaviour on this signal, and `crossval/` uses it as the default `COHERENT_FFT`.
  The oracle regenerates its reference arrays from whatever file it is given, so
  the cross-validation itself is indifferent to the change.

So: regenerate, then re-check those thresholds deliberately rather than nudging
them until green.

## Environment gotchas (Julia 1.12)

- `SortingNetworks.jl` and `StatProfilerHTML`'s HTML writer are **broken on
  1.12** (method-overwriting precompile / `Core.MethodInstance` field change).
  Use `partialsort!`/hand-rolled selection and the profiler's text/bucket
  summary instead. `LoopVectorization.@turbo` works.
- FFTW *planning* is not thread-safe: build all plans single-threaded before the
  parallel region; only *execute* them (via `mul!`) inside it.
