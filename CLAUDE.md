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

## The two hosts (quote the host with every number)

Scott develops on both, and they have inverted each other's conclusions
repeatedly — now with a known mechanism (see the AVX-512 entry below).

- **`foops`** (laptop) — repo at `/home/sransom/git/CoherentSearch.jl`.
  i7-10510U, Comet Lake, 4 cores / 8 threads, 8 MB L3. **No AVX-512**
  (`avx avx2 fma` only), so `--cpu-target=skylake` is a no-op here. Throttles
  under sustained load; quote `-t 1`. Pixi env (a Python that can
  `import coherent_search`, plus riptide's `rseek`):
  `/home/sransom/python_venvs/pixiPSR/.pixi/envs/default/bin/{python,rseek}`.
- **`fitzroy`** (workstation) — repo at `/data1/git/CoherentSearch.jl`, *not*
  under `~/git`. `ssh fitzroy` works unattended. Xeon Silver 4114, Skylake-SP,
  2x10 cores, 14 MB L3 per socket, nominal 2.2 GHz. **Has AVX-512.** `perf`
  needs no root (`perf_event_paranoid = 2`), and
  `core_power.lvl{0,1,2}_turbo_license` is available — that is the direct
  license-level counter, better than inferring it from `cycles/ref-cycles`.
  It is Scott's desktop: check `uptime` and top processes before timing.
  Pixi env: `/data1/environments/pixiPSR/.pixi/envs/default/bin/{python,rseek}`;
  riptide at `/data1/git/riptide`. **The system `python3` there is too old for
  `compare/compare_riptide.py`** — run it with the pixi `python`. Neither host
  has `rseek` on a non-interactive `PATH`, so pass `--rseek <pixi>/bin/rseek`
  (the script's fallback is the laptop's path only).

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
- `bin/toy_coherent_search.jl` — **the algorithm with every optimisation
  removed**, written to be read: brute-force per-point interpolation, one
  `irfft` per fold, the boxcar filter straight from its definition, plain nested
  loops. It is what the paper's pseudo-code figure describes, and its functions
  carry that figure's line numbers. Roughly **100–200x** slower than production
  and machine-dependent — two runs of the same command on the laptop gave 190.8x
  and 177.1x (~243 vs ~1.27 µs per trial fundamental) — so quote it as a range
  and give it a narrow band. It reuses the
  production candidate collapsing and output verbatim, and differs deliberately
  in two ways: the full geometric width bank rather than the ladder-pruned one,
  and it was where the analytic σ was worked out and validated first.
  `bench/toy_vs_production.jl` A/Bs the two and scans the σ comparison across
  the band; `test/test_toy.jl` pins the toy against the reference path.
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
  **The printed path does NOT tell you the sibling repo is up to date, and that
  cost an hour on 2026-08-24.** The metric check read `rel = 5.694e-02` against
  its `1e-9` tolerance on a laptop whose `../coherent_search` checkout predated
  the `snr1` metric change — the *old* `(Σ(Pᵢ − median))/(σ√w)` against our
  zero-mean unit-L2 template, which differ by exactly `√(1−δ)` (16% at the widest
  bank member, bracketing the 5.7% seen). `git pull` in the sibling repo restored
  it to **1.393e-16**. So the recorded warning about a stale *install* has a twin:
  a stale *checkout* prints exactly the path you expect. **When only the metric
  check fails and the kernel/profile checks stay at ~1e-16, suspect the sibling
  repo before the Julia, and check its log against the last metric change** —
  `sigma_center=:median` reconciles the σ̂ centring, not the template, so it will
  not paper over this. **Do not "fix" it by loosening the 1e-9.**
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
# `import coherent_search`.  `DEFAULT_PY` in the script is the WORKSTATION's
# (/data1/environments/pixiPSR/...); on the laptop set
#   COHERENT_PYTHON=/home/sransom/python_venvs/pixiPSR/.pixi/envs/default/bin/python  Default test data is the sibling repo's
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

**We now beat `rseek` single-threaded on both hosts.** Re-measured **2026-08-24**
(after the scatter fix and the `:f32` default) on
`PM0063_034C1_DM445.0_red.fft`, `--preset bench`, both covering **0.1–200 Hz in
120…20 bins**, median of 3:

| | rseek | ours `-t 1` | ratio | ours, all cores |
|---|---|---|---|---|
| i7-10510U (laptop, 4 cores) | 21.13 s | **10.05 s** | **2.13x faster** | 5.20 s `-t 4` |
| Xeon Silver 4114 (fitzroy, 20 cores) | 19.81 s | **13.45 s** | **1.46x faster** | 2.86 s `-t 20` |

(2026-08-22, for comparison: laptop 22.25 / 11.99 / 1.84x / 7.77; fitzroy
19.83 / 15.79 / 1.26x / 3.51. `rseek` itself is reproducible to ~1–5% across the
two dates, which is the scatter to judge our column against.)

Start-up split, same runs: laptop rseek 1.10 + 19.89 s against ours 0.85 +
9.01 s (**pure compute 0.45x**); fitzroy rseek 0.79 + 18.74 s against ours
1.44 + 11.89 s (**0.63x**). On both hosts the pure-compute ratio is at least as
good as the wall-clock one, so start-up is not what the comparison measures.

**And we detect more strongly**: the 7.1185 Hz pulsar at **S/N 12.30 vs
riptide's 11.80** (ducy 10.0% against its 6.5% — its width bank is built from
`bins_min` and cannot reach this pulse at the depth it folded), plus the
0.2603 Hz candidate at 7.32 that it does not report. riptide's two extra entries
are the `f/2` and `2f` of the pulsar, which it does not filter and we collapse.
Both hosts report identical candidates, as they must. **Our S/N is now riptide's
statistic exactly**, so the two columns are finally the same quantity — but this
is still one pulsar in one observation, so read §3.2 before drawing any
sensitivity conclusion from the 0.5 between them.

**This section has read "we are slower" for its whole life and no longer does —
check the date before quoting it.** The history on the laptop is 29.3 s
(2026-08-11, 1.24x slower) → 11.99 s, and on the workstation 32.3 s (1.59x
slower) → 20.89 s (1.07x slower, after the 2026-08-22 metric/interp work) →
15.79 s (1.26x faster, after the transpose fix). Quote the machine *and* the
date with the ratio; the two hosts still disagree by more than any single
optimisation below (1.84x vs 1.26x on the same code and the same data).

- **Start-up is NOT what the comparison measures**, and the harness proves it
  per run — and it now works *against* us, since ours is the larger of the two
  and we win anyway. Laptop, `bench`, 2026-08-22: rseek 0.46 s start-up +
  21.54 s searching, ours 0.95 s + 11.03 s, so the pure-compute ratio (0.51x) is
  a shade better than the wall ratio (0.54x). Our fixed cost is measured by
  re-running the same command over a near-empty band. riptide's `find_peaks` is
  9.9 s (46% of its compute) and is a separate pass doing the candidate work we
  do inline — do **not** compare us against its `ffa_search` (11.1 s) alone.
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
  interleaved: rseek 18.35/18.33 s, ours 17.86/16.45 s — ~1.05–1.11x faster at
  equal work; 1.10x after that day's work. Re-measured 2026-08-22 on the **laptop**
  after the transpose fix: **rseek 18.07 s, ours 10.07 s — 1.80x faster on wall
  clock, 1.91x on pure compute** (9.03 s against its 17.27 s). The `bench` deficit
  was never the code, and there is no longer a deficit in either preset.
  **But rseek still wins the pulsar in that config: S/N 12.60 (`w=13`, ducy 10.3%)
  vs our 12.10 (H=65, ducy 14.6%) at the same depth — 11.84 after 2026-08-24.**
  Our 13.27 comes from the decimated fold, so "we detect more strongly" is a
  statement about the *ladder*, not about the per-fold detector. Since the metric
  is now riptide's exactly, any such per-fold difference is the *profile
  estimator*, not the S/N definition — but every number in this bullet is one
  pulsar in one observation, so treat them as anecdotes and let §3.2's Monte
  Carlo decide anything about sensitivity.
- **Our S/N IS riptide's S/N as of 2026-08-24 — same statistic, not merely
  comparable.** `_boxcar_scan`, the Python oracle's `snr_metric` and riptide's
  `cpp/snr.hpp:snr1` all correlate the profile against the width-`w` boxcar made
  *zero-mean and unit-L2* (`h = √((n−w)/(nw))` on-pulse, `−b = −w/(n−w)·h` off),
  which is `(S_w − δ·S_tot)/(σ√(w(1−δ)))` with `δ = w/n`. Verified against the
  riptide binary itself on real PM0063 folds and on pure noise: **1.4e-7 and
  2.0e-7 max relative**, which is riptide's own `Float32` accumulation.
  `test_search.jl` pins it against a longhand `snr1` at machine precision.
  - **What this replaced, and why the obvious fix would have been wrong.** The
    metric used to subtract each profile's *median* and divide by `σ√w`. Against
    the zero baseline that really is `√(1−δ) ×` riptide's, exactly and
    pointwise — but `_boxcar_scan` is called with the *median* baseline for every
    reported candidate, and the median's own variance already compensates part of
    it: measured per-(phase,width) noise sd at `nbins=120` ran 1.001 (`w=1`) to
    0.951 (`w=28`), not to `√(1−δ) = 0.876`. **So the prescribed "deterministic
    per-width factor in `_boxcar_scan`" would have over-corrected by 9% at
    `nbins=120` and 12% at `nbins=20`**, turning a bias against broad pulses into
    a bias for them. Read that as another instance of the standing lesson: the
    analysis had been done on the gate, and the shipped path was not the gate.
  - **The median recovered nothing.** DC is held at zero, so `S_off ≡ −S_w`
    identically and subtracting the true off-pulse level is *exactly* a `1/(1−δ)`
    rescale of `S_w` (verified to 9 decimals). What the *sample* median does is
    slide between 0 (noise) and that level (bright pulse), so the old metric's
    ratio to riptide's drifted **0.981 → 1.065 with source brightness at 5%
    duty** — a normalisation that depended on the signal, hence no calculable
    false-alarm rate. At matched FAP the zero-mean template detects strictly
    better at every duty: equal below ~5%, **0.274 vs 0.142** detection fraction
    at 20% duty, 0.019 vs 0.010 at 30% (Gaussian pulses, FAP 1e-3, `nbins=120`).
  - **Cost on real data, measured:** PM0063's 7.1185 Hz pulsar reads 13.27 → 12.30
    (bench config, `k=4` both), and the empirical noise floor falls with it
    (FAP=1e-5 at `k=4`: 5.681 → 5.514; at `k=1`: 6.367 → 6.167, from 8.36M trials
    per rung). Candidates ≥ 6.0 go 21 → 7 — **mostly the rescale, not a real
    reduction in false alarms**; at matched FAP the counts are comparable.
  - **The residual difference from `rseek` is n=1 — do not read sensitivity into
    it.** `--preset matched`, same depth: rseek 12.60 (w=13), ours 12.10 → 11.84
    (H=65). What follows is only a *decomposition*: the metric is now provably
    identical on identical profiles, so whatever difference exists comes from the
    **profile estimator** (our coherent Fourier fold vs riptide's time-domain FFA
    fold) and/or σ̂, never from the S/N definition. **The size of it means nothing
    — one pulsar, one noise realisation, and the single-detection extreme-value
    scatter is much larger than the 0.8 between them.** Relative sensitivity is
    what §3.2's injection Monte Carlo settles; do not start a bug hunt on this.
  - Worth ~8% of the metric phase (1092 → 950 µs/chunk, after correcting the ~6%
    whole-machine drift visible in the untouched σ̂ and transpose columns): the
    median rescan is gone, against ~1% added to the scan by the `δ·S_tot` term.
    End to end that is inside this host's run-to-run scatter.
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

**Planned: a detection-efficiency Monte Carlo against riptide, for a paper**
(`Summary_and_Future_Work.md` §3.2). Injected Gaussian pulses over the full band
and a realistic duty-cycle range at S/N 9–13, scoring detection fraction *and*
compute cost for both codes. The design constraints are already settled by the
work recorded here — Gaussian pulses not boxcars, both the `bench` and `matched`
riptide configurations, paired noise realisations, and duty cycle rather than
S/N as the cross-code observable — so read §3.2 before starting rather than
re-deriving them.

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
  decimated transforms, on the production `Workspace`), `metric_bench.jl` (the
  same treatment for the metric: σ̂ / transpose / width scan / gate / gate+rescan,
  per fold depth, plus the fraction of trials clearing `exactcut`). Run single-threaded (`-t 1`)
  for clean profile attribution; warm up before timing to exclude JIT. Example
  FFT for longer runs: `PM0063_034C1_DM445.0_red.fft`.
- **`bench/thread_scaling.jl` is the standard scaling check** (it replaced the
  old `scaling.jl`, which timed one thread count per invocation and left the
  analysis to you). It re-invokes itself as a worker per thread count — Julia
  fixes `nthreads` at process start — and each worker times only the *warm
  in-process* `search`. **Excluding start-up is the point:** on the whole-process
  wall clock the same run scales 6.7x at `-t 20`, and 9.9x once the ~1.4 s fixed
  cost is out. Measured on the 20-core workstation, PM0063 at the riptide bench
  config, 1/2/4/8/16/20 threads:

  | | 1 | 2 | 4 | 8 | 16 | 20 | best | Amdahl `s` |
  |---|---|---|---|---|---|---|---|---|
  | before 2026-08-22 | 31.9 | 18.3 | 9.4 | 5.4 | 3.3 | 3.2 | 9.9x @20 | 0.059 |
  | after the laptop pass | 22.7 | 12.8 | 6.6 | 4.0 | 2.5 | 2.5 | **9.0x @16** | 0.063 |
  | **2026-08-24** (scatter fix + `:f32` default) | **11.58** | 6.51 | 3.44 | 1.93 | 1.42 | **1.29** | **9.0x @20** | 0.0646 |

  **The 2026-08-24 row is 1.96x the previous one at `-t 1` and 1.94x at `-t 20`**
  — the AVX-512 scatter fix plus `:f32` becoming the default. Scaling is
  unchanged (`s` 0.063 → 0.0646 against a serial baseline that is twice as fast),
  which is the good outcome: the win was to the parallel part as well as the
  serial one. CPU-seconds still inflate 62% across the sweep. The plot lives at
  `docs/thread_scaling.png` and is embedded in the README; regenerate both
  together, since `bench/thread_scaling.png` is gitignored.

  **The laptop pass is worth 1.40x at `-t 1` here** (better than the ~1.24x it
  measured on the laptop) **and 1.26x at `-t 20`** — the win shrinks with thread
  count because it removed work that was parallel. Scaling itself barely moved
  (`s` 0.059 → 0.063 against a faster serial baseline), so **the two scratch
  growths did NOT cost anything at 20 threads**: the boxcar gate's tile doubling
  (`_BC_BATCH` 32 → 64, since 64 → 128) and the interpolator's weight table
  (395 KB → 1.45 MB, shared read-only). CPU-seconds inflate 62% over the span,
  the same memory-stall term as before, not a code defect.
  - **RESOLVED 2026-08-24: `-t 20` IS faster than `-t 16` again** (1.29 vs
    1.42 s, a clean 9% on the full sweep). The 2026-08-22 reading of 2.54 vs
    2.52 s was inside that day's scatter, and the `:f32`/AVX-512 investigation
    separately failed to reproduce its own `-t 20` anomaly. This is still a
    dual-socket box (2×10 cores, 14 MB L3 *per socket*) and the marginal core
    past 16 threads is worth less than a linear one — 8.15x → 9.00x for a 25%
    thread increase — but it is not negative. **Do not re-quote the old claim.**
  - The desktop carries ~2 cores of its own load (Chrome, Zoom), which is worth
    remembering before reading too much into the top of the curve.
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
  `boxcar_gatemargin=0.01` slack, so it cannot move which trials get re-scored
  in `Float64`). `bench/boxcar_bench.jl` re-measures both axes; two new pins in
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
- **Done (2026-08-22): the metric, split and cut — 13.0% off the metric phases,
  ~1.05x end-to-end.** `bench/metric_bench.jl` breaks the 47% "metric" into σ̂ /
  transpose / width scan / gate / rescan per fold depth, and two of its numbers
  contradicted the code's own comments: the tile transpose is **48% of the gate**
  (the section comment said ~20%), and `_block_sigma` is **26% of all metric
  work** — invisible because it is billed to `block-sigma` at `k=1` but hidden
  inside `decim-metric` for `k=2…6`. Two changes landed:
  - **`_BC_BATCH` 32 → 64 → 128** (64 on the laptop, raised to 128 on
    2026-08-22 after the workstation sweep), worth 10.8% of the metric there and
    another 6.9% here, **byte-identical** either way (the per-profile arithmetic
    does not depend on `B`). Laptop, interleaved over all six depths:
    `B = 32/48/64/96/128` → 692/628/575/572/577 µs per chunk, so 64 is its knee
    and 128 is 0.3% behind. Workstation, in situ metric share:
    48.96/48.55/48.68/48.03/**45.52**% for the same five, plus 47.67/45.42% at
    192/256 — everything from 32 to 96 is one flat plateau and only 128 moves it.
    So 128 costs the laptop ~0.3% and buys the workstation ~4% end to end.
    **It disproves the L1 intuition twice over**: at `B=64` the tile+prefix pair is
    63 KB, overflowing the 32 KB L1 that `B=32` nearly fits, and wins anyway — and
    at `B=128` it is 126 KB and wins again.
    - **This sweep is the cleanest example of why to read shares, not seconds.**
      The absolute times across `B = 32…96` differ by 6% while the *share* is
      flat to 0.5%: that 6% is whole-machine drift between invocations, on a host
      whose within-run rep-to-rep scatter is only ±1%. Reading the absolute
      column would have produced a confident ranking of pure noise.
  - **`_block_sigma` subsamples whole profiles**, evenly spaced, instead of a flat
    stride through the linear index — 2.7 MB → 0.4 MB of gather per chunk, worth
    3.2%. It is also a *better* sample: the flat stride hit only `nbins/s`
    distinct phase bins whenever `s | nbins`, i.e. **four**, at five of the six
    default depths. Oracle-safe because `snr_metrics` defaults to
    `sigma_samples = typemax(Int)` and so takes the un-subsampled branch.
  - **The isolated bench said 1.74x; in situ it is 1.13x** on the metric. Same
    lesson as the smooth-`fftlen` and `_block_sigma` `idiv` cases: the
    microbenchmark re-runs on hot data, the search reads `profs`/`dprofs` once
    while competing with the transform that wrote them, so the *bandwidth* half
    of the win evaporates and only the *vector-width* half survives.
  - **Wall clock cannot resolve a 5% change on this laptop** (±10%, with
    within-round thermal drift). Compare arms by each run's own metric-phase
    *fraction* of accounted time — the untouched phases drift with it, so the
    ratio is drift-robust. It gives 36.0% → 32.9% at `-t 1` and 33.3% → 30.4% at
    `-t 4`, i.e. the same 13%, so the doubled gate scratch costs nothing threaded.
  - **Candidates are not byte-identical** — σ̂ moved. Same three candidates at the
    same frequencies; S/N 12.97 → 13.27, 7.83 → 7.76, 7.33 → 7.22, and the
    pulsar's winning rung moves `k=6` → `k=4` (a near-tie between adjacent rungs).
  - **That +2.3% is not the new sampling being better or worse — the per-chunk σ̂
    is itself ~1% noisy, and reported S/N is exactly `1/σ̂`.** Measured over 24
    chunks against the exact all-bins σ̂: both schemes are unbiased, with rms
    1.0–1.5% (flat stride) vs 0.9–1.2% (whole columns), i.e. the new one is
    modestly *better* at every depth — it just drew unluckily in the pulsar's own
    chunk (0.9809 of exact at `k=4`, against the old scheme's 1.0004). Against the
    exact σ̂ the pulsar is **≈13.0**, which 12.97 and 13.27 straddle.
    **So the last two digits of "12.97 vs riptide's 11.80" are σ̂ noise, and
    always were.** The 10% gap is safe; a percent-level S/N comparison (the §3.2
    Monte Carlo) is not, and must use the exact σ̂ or model this term.
  - **Four measured dead ends — do not re-guess them:** fusing the transpose away
    (a wash at `B=32`, worse at 64; re-tested at `B=128` and 0.98x); full fusion of
    prefix + widths (1.33x *slower*, AVX2 register pressure); blocking or
    reordering the transpose (0.56–1.03x — **overturned on 2026-08-22: all three
    of those blocked the PHASE axis, and blocking the PROFILE axis by 8 is 3.51x
    on the workstation.** The "~15.8 GB/s L3 wall" that justified the verdict was
    not a wall at all; see the transpose entry below); and a cheap upper bound to
    skip profiles outright (after ladder pruning the bank is all narrow, and the
    tightest cheap bound sits at 5.9 against the then-`medcut = 4.0`; the cut is
    now `threshold − 0.01`, which makes such a bound useless outright).
- **Done (2026-08-22, follow-up): σ̂ folds about the structural zero — another
  12.2% off the metric, 1.04x end-to-end.** DC is held at zero, so every profile's
  bin mean is *exactly* 0 and the pooled distribution is symmetric about it;
  `1.4826 × median(|x|)` is then the same scale estimator with the location known
  rather than estimated. **One `_median!` instead of two** (σ̂ 245 → 149 µs per
  chunk isolated), and statistically the better of the pair. Measured
  interleaved over three rounds, metric share 32.83% → 30.04% of accounted time.
  Candidates barely move (13.27 → 13.27, 7.76 → 7.76, 7.22 → 7.21).
  - **This is a deliberate divergence from the Python oracle, and it is named,
    not hidden.** `_block_sigma(...; center = :median)` still computes the classic
    MAD-about-the-sample-median that upstream `snr_metric` does, `snr_metrics`
    forwards it as `sigma_center`, and `crossval/crossval_accuracy.jl` passes
    `:median` — so the metric pin stays at **1.365e-16** (re-verified 2026-08-24
    at 1.393e-16) and still covers the
    width bank, the boxcar template and the scan arithmetic. Loosening
    that tolerance to ~1e-2 to swallow the difference would have been the wrong
    trade: it would hide a real bug in any of those three.
  - `test/test_search.jl` pins `:zero` against `:median` (≤2% at every fold depth,
    against σ̂'s own ~1% sampling error) *and* pins the premise — that the pooled
    median really sits at zero — because if that ever stopped holding, `:zero`
    would be silently biased and every reported S/N with it.
  - **Cumulative for the day: metric share 36.0% → 30.0%, i.e. 23.7% off the
    metric and ~1.09x end-to-end at `-t 1`.**
- **Done (2026-08-22): the interpolator vectorises across TRIALS — 1.56x on the
  interp phase, ~1.14x end-to-end, candidates byte-identical.** This is §3.1's
  planned move, and the obstacle recorded there ("a same-window run is only ~`q/h`
  trials long, so it vectorises well for low harmonics and barely at all for
  `h > 15`") **turned out not to apply**, because the formulation never needs a
  constant-`b` run. Substituting `j = δₖ + i` rewrites a group of `V` consecutive
  trials as a matrix-vector product `sreₖ = Σⱼ Wx[k,j]·re[b₀+j]` against one
  contiguous slice of the bin planes, with `Wx[k,j] = W[j-δₖ, pₖ]` depending only
  on the residue the group *starts* at — so it is tabulated at plan time.
  `re[b₀+j]` is then a broadcast scalar and `Wx[:,j]` a contiguous column:
  **no gather and no horizontal reduce**, every lane a different trial.
  - **It wins at every harmonic**, which the run-based approach could not:
    1.7x at `h = 59` and 1.8x at `h = 60`, 2.0x at low `h`. The cost is
    `(m+Δ)/m` wasted FMAs (`Wx` is zero off each row's `m` nonzeros,
    `Δ ≈ V·h/q`) — 1.06x at `h=1`, 1.5x at `h=59` — bought back many times over.
  - **`V = DIRECT_GROUP_V = 16`, and it is a `const`, not a plan field**: the
    accumulators are an `NTuple{V}`, and with a runtime `V` the `ntuple`s do not
    unroll. Same trap `_BC_BATCH` documents. Isolated sweep `V = 8/16/32/64` →
    121/99/107/157 µs; in situ `V = 8/16/32` → interp share 28.5/25.4/30.7%.
    **Both agree on 16**, which is not the usual outcome here.
  - **A closure cost 2000x.** Writing the `ntuple` bodies inline made them capture
    an assigned-in-loop local, which Julia boxes. They are top-level `@inline`
    helpers taking everything as arguments. If this kernel ever reads as
    inexplicably slow, check that first.
  - **Bit-exact chunk invariance is preserved, and now actually tested.** Groups
    are anchored to the **global** trial index; a chunk's two end groups may hang
    off either side, and are computed in full and masked on store rather than
    falling back to a second kernel — which is why the plane buffers are widened
    to the *group* range and zero-filled outside the file. The range guard stays
    on the **true** trial range, so where a harmonic gives up (past the
    amplitudes, or past Nyquist) is unchanged. `test_search.jl` now runs the
    invariance pin at chunk sizes 128/100/37/51 and asserts three of them really
    do straddle groups.
  - Measured `-t 1`: interp 4.93 → 3.02 s, share 34.8% → 25.5%. `-t 4`: share
    28.9% → 20.6%. **Both are 36% off the interp**, so the table growing
    395 KB → 1.45 MB (shared read-only) costs nothing threaded at 4.
  - 460/460 tests, crossval unchanged (3.8e-16 / 2.1e-16 / 1.4e-16), `.cohout`
    byte-identical — the ~5e-16 of resummation moved no reported S/N.
- **Done (2026-08-22): the tile transpose was 69% of the metric and ~31% of the
  whole search on the workstation; it is now 421 µs against 1516 (3.60x), for
  1.36x end-to-end there and ~1.03x on the laptop, candidates byte-identical.**
  The fix is one loop nest: `_bc_transpose!` blocks the **profile** axis by
  `_BC_TR_BJ = 8`. See `Summary_and_Future_Work.md` §3.3.
  - **Two recorded diagnoses were wrong, and both had blocked the work.** (a) It
    was not "an L3 bandwidth wall": at `B = 128` the workstation ran it at
    **4.7 GB/s** against the **21 GB/s** that host delivers at the same 2 MB
    footprint, flat in size — the signature of an access-pattern limit.
    (b) It is **read-gather, not write-scatter**: the inner loop is over `b` with
    `i` fixed, so the write `tile[(i-1)*B+b]` is a contiguous 512 B run and the
    read `profs[i, j0+b]` is the stride-`nbins` gather. Cache-line-filling *write*
    blocks are therefore the wrong end, and measure worse on both hosts.
  - **What matters is how many concurrent strided read streams the loop asks for,
    and the two hosts disagree about it by 3.5x.** µs per chunk over `k = 1…6`,
    `Float64` (`bench/tile_shape_bench.jl`): at `BJ = 4/8/16/32/64/128` the laptop
    gives 354/325/377/372/354/**295** and the Xeon 464/**431**/1840/1764/1765/1514.
    Everything from 16 up is one flat plateau on the Xeon; only 8 (or 4) escapes it.
    Blocking the *phase* axis too (4x8, 8x8) is worse than blocking `b` alone on
    both — which is why every earlier phase-blocked attempt read as a loss.
  - **The laptop's predicted 0.91x loss does not happen in situ; it is a small
    win.** Interleaved, 3 rounds, metric share of accounted time: laptop
    35.66% → 34.81% (10.79 → 10.50 s), workstation 48.93% → **31.60%**
    (19.41 → 14.32 s). The microbench re-reads `profs` hot; the search reads it
    straight after the `brfft` that wrote it. **The metric is no longer the
    largest phase on the workstation** — `decim-brfft` is.
  - **The guru profile-major `brfft` (`bench/guru_transpose_probe.jl`) is a DEAD
    END, and its 1.25–1.50x was a baseline error.** It compared against
    `copyto!(tile, transpose(Yd))`, a naive whole-array transpose ~2.7x slower
    than the `_bc_transpose!` that ships. Swept over the whole ladder in both
    precisions (`bench/guru_brfft_ladder.jl`) it is **0.78x (`:f64`) / 0.99x
    (`:f32`)** on the laptop. Benchmark against the shipped kernel, not against
    the obvious way to write it.
  - **FFTW's guru rank-0 r2r transpose is real and still lost.** The PRESTO trick
    (`~/src/presto/tests/test_transpose.c`: `rank = 0`, `howmany_rank = 2`, a pure
    strided copy FFTW cache-blocks) does the whole chunk in 561 µs on the
    workstation against the old loop's 1510 — but `BJ = 8` does it in 431 with no
    plan, no buffer and no `ccall`. Kept in `bench/transpose_bench.jl` and
    `bench/gate_layout_bench.jl` along with the whole-chunk `profsT` layout it
    enables (1.79x on the workstation gate, **0.73x on the laptop**, +4.8 MB per
    workspace). Fusing the transpose into the prefix sum, re-tested at `B = 128`:
    **0.98x**.
  - **`--precision f32` halves the read side** and is already measured to help
    exactly these phases (`gate+metric` −8.1%, `decim-metric` −5.4% at `-t 1`).
  - **Kadane's algorithm is still not the next move**, despite being the obvious
    idea. With the transpose fixed the metric now splits (workstation, µs per
    chunk over `k = 1…6`) transpose **421**, width scan **346**, σ̂ **185** of
    1068 — so the scan is 32% of the metric and the metric is 32% of runtime,
    putting Kadane's ceiling at ~4x on ~10% of runtime, before any of the
    tail-calibration cost.
    Worse, **plain Kadane maximises the SUM, and our statistic is `sum/(σ√w)`** —
    a different objective, needing a max-density/normalised-segment algorithm
    whose serial dependency chain vectorises *worse* than the present scan, which
    is already SIMD across `_BC_BATCH = 128` profiles.
- **Done (2026-08-24): the noise scale is COMPUTED, not measured — `--sigma
  analytic` is the default. 15% off the metric phases, 1.075x end-to-end at
  `-t 1`, and it is *more* accurate than the estimator it replaced.** The search
  is only meaningful on a normalised `.fft`, and that assumption already fixes
  the fold's noise: mean power 1 means Re and Im each have variance 1/2, so the
  hot loop's unnormalised `brfft` of `H` harmonics with DC at zero gives
  `σ = sqrt(2·nlow + 0.5·nnyq)` — `sqrt(nbins)` times `sqrt(1 − 3/(4H))`.
  - **That correction is not decoration: 0.6% at `H=60` but 3.8% at the `H=10`
    of a `k=6` fold.** Dropping it would bias the shallow folds against the deep
    ones — precisely the cross-decimation bias the boxcar metric exists to fix.
  - **The fill count, not the stack length, is what enters.** Harmonics past
    Nyquist are zero rows and carry no noise. `fill_harmonic_row_direct!` now
    returns a `Bool` and `fill_chunk_profiles!` records it in `ws.filled`, so
    `_analytic_sigma` counts what is actually there. Using `Hk` instead would
    overestimate σ at the top of the band and silently suppress fast candidates.
  - **Measured, laptop, PM0063 0.1–33.3 Hz, 7 interleaved reps:** metric share
    27.00% → 22.86% at `-t 1` (1.075x wall) and 26.99% → 23.09% at `-t 4`
    (1.053x). **Read the shares, not the seconds** — the wall clock scattered
    8.6–11.98 s across reps while the share held to ±0.2%, and one phase table
    showed *every* phase dropping 24–31% including ones that cannot have changed.
  - **It is closer to the truth than the estimator it replaced.** Against the
    exact all-bins pooled MAD on PM0063 (`bench/toy_vs_production.jl`, four
    frequency windows × six fold depths): analytic/exact spans **0.9918–1.0217
    (3.0%)**, while production's own 8192-sample subsampling spans
    **0.9806–1.0335 (5.4%)**. Reported S/N is exactly `1/σ̂`, so that ~1%
    sampling error was landing on every candidate — the term CLAUDE.md already
    flagged as something §3.2's Monte Carlo would have to model. It is gone.
  - **The one bias is the interpolation truncation, and it is predicted:** the
    `m`-bin kernel keeps `S_m ≈ 1 − 0.203/m` of the noise power along with the
    signal, so analytic runs `0.203/(2m)` high — 0.64% at `m=16`. Measured on
    synthetic normalised noise: 1.0086 / 1.0053 / 1.0035 at `m = 16/32/64`
    against 1.0064 / 1.0032 / 1.0016 predicted. Constant, and smaller than the
    error it replaced, so it is left uncorrected.
  - **Its one assumption fails SILENTLY and in the dangerous direction**, which
    is why `_sigma_sanity_check` exists: measured σ̂ tracks whatever the
    amplitudes are, while analytic keeps insisting on unit variance, so a
    normalisation error *inflates* every S/N and fills the candidate list with
    noise. `search` scores three chunks both ways and warns above 10%
    disagreement (~0.1% of runtime). On the un-normalised `harmonics_hi.fft` it
    fires at a factor of ~1000. **It warns rather than switching** — silently
    changing estimator would be a surprise.
  - **`--sigma measured` is still the right answer when the noise level varies
    with Fourier frequency** (residual red noise, an RFI comb, a `rednoise` pass
    that did not take): the MAD adapts and the closed form cannot. That trades a
    ~1% estimation error against an unmodelled bias, and on a badly-behaved
    observation the bias wins.
  - **Candidates move by ~1–2%, and near-threshold ones churn.** PM0063 at
    threshold 6: the 7.1185 Hz pulsar 12.30 → 12.11, the 0.2603 Hz candidate
    7.32 → 7.37; 7 candidates either way, with one swap at 6.0–6.1. Do not
    expect a `.cohout` diff to be empty across this change.
  - **`chunk_metrics` still forces `_block_sigma`**, because its job is to equal
    `block_metrics`, which measures. The equivalence pins are therefore
    untouched; `test_search.jl`'s decimation-vs-native-fold pin had to say
    `sigma=:measured` explicitly for the same reason (and was off by ~900x until
    it did, on the un-normalised fixture).
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
- **`Float32` profiles (`--precision f32`): the "do not merge at any thread count"
  verdict is DEAD, and the crossover is back (2026-08-22, workstation).** Measured
  `:f32` vs `:f64` in one process at the riptide bench config: **0.965x at `-t 1`,
  1.030x at `-t 4`, 1.173x at `-t 8`, 1.167x at `-t 16`, 1.337x at `-t 20`** —
  against the 2026-08-16 figures of 0.82/0.92/0.98/1.01 for the same four points.
  Every point improved and the crossover sits between 1 and 4 threads.
  - **RE-MEASURED 2026-08-24 on an idle fitzroy, and the ratios above do NOT
    reproduce — re-baseline before quoting them.** Same config, `precision_ab.jl`,
    5 reps: **0.875x at `-t 1`, 0.981x at `-t 4`, 1.384x at `-t 16`, 1.029x at
    `-t 20`.** The shape is the same (loses at 1 thread, wins big threaded) but
    every number moved, and `-t 20` is now well below `-t 16` — this is a
    dual-socket box and 20 threads crosses sockets, so confirm that point before
    quoting it.
  - **The recorded "mechanism" below is WRONG for the two metric rows.** It read:
    "the metric's bandwidth-bound half gets cheaper (`decim-metric` −5.4%,
    `gate+metric` −8.1%, `zero-ftprofs` −62%, base `brfft` −18%) while
    `decim-brfft` (+23%) and `interp` (+27%) get worse." Re-measured, the
    non-metric rows roughly hold (`zero-ftprofs` −60%, `interp` +32%,
    `decim-brfft` +16%, base `brfft` −30%) but **the metric rows invert:
    `decim-metric` +33.8% and `gate+metric` +46.1%.** Checked at `5bc46ad`, i.e.
    *before* the 2026-08-24 metric change, so this is not a regression from that
    work — the record was already wrong or was taken on different code.
  - **The `:f32` gate penalty is real, reproducible, and UNEXPLAINED. Five
    hypotheses are dead; do not re-guess them.** Measured on the production
    buffers with a genuine chunk (`bench/metric_bench.jl`, now precision-aware):

    | | transpose | scan | sum | gate (interleaved) |
    |---|---|---|---|---|
    | `:f64` | 415.0 | 352.3 | 767.3 | **765.8** — additive |
    | `:f32` | 314.8 | 336.4 | 651.2 | **1037.4** — +386 µs |

    Both kernels are individually *faster* in `:f32`; alternating them, as
    `_boxcar_gate!` does, `:f32` costs 36% more. Ruled out: (1) something about
    the shipped function — a hand-written copy of its loop reproduces it exactly;
    (2) a type instability — zero allocations in both arms; (3) `_BC_TR_BJ` being
    tuned in `:f64` — swept inside the *interleaved* loop, `BJ=8` is optimal for
    both and `:f32` is worse at every value (2/4/8/16/32/64/128 → f64
    948/824/**781**/2207/2179/2159/1922, f32 1395/1122/**1040**/1942/1907/1899/1907);
    (4) type-based alias analysis (tile and profs both `Float32` defeating TBAA) —
    a 2×2 on tile type × profile type does not fit that pattern (f64 profs
    767/1191 for a `Float32`/`Float64` tile, f32 profs 1046/1109), though it does
    confirm the shipped `Float32` tile is right; (5) small-transform or bandwidth
    effects — the sum of the parts is *cheaper* in `:f32`, so the cost is created
    by the interleaving itself. **`perf` is not available on fitzroy**, so the
    hardware counters that would settle it could not be read. Next step if anyone
    resumes this: bisect the interleaved loop by stubbing each kernel, or get
    counter access.
  - **`Float32` interpolation *weights* do not fix the `interp` penalty** — tried,
    on the theory that it was a `ComplexF64`→`ComplexF32` store conversion. It is
    still +31% with `Float32` weights, so that hypothesis is wrong and the cause
    is unidentified. Combining the two is also *worse* than either alone at
    `-t 1` (0.915x), because `B = 128` and the `Float32` weights had already taken
    some of what `:f32` profiles was buying.
  - **Still not the default**, because it loses at `-t 1` and the deployment model
    that matters for real searches is one single-threaded process per DM (§3.1).
    But `--precision f32` is now the right flag for a `-t 16`-and-up run, which it
    was not on 2026-08-16. Note the `-t 16` win of **1.384x is achieved while
    carrying the unexplained gate penalty above** — so that penalty is a reason
    `:f32` is not better still, not a reason to avoid it.
  - **Do not confuse this with the `Float32` interpolation *weights*, which ARE
    the default** (7361279). Two separate knobs; the profile-stage `--precision`
    has never been anything but `:f64`.
- **FIXED (2026-08-24): the AVX-512 downclocking is gone, in source, and
  `--precision f32` is now a win at `-t 1` on BOTH hosts. The cause was ONE
  instruction — `vscatterqps` in `_bc_transpose!` — not a diffuse width problem.**
  With `profs` already `Float32`, `BJ` being a compile-time constant fully
  unrolls the `b` loop, exposing `i` as innermost; LLVM then vectorises *that*
  (reads down a column are contiguous) and writes the tile with a **512-bit
  scatter**. Scatter is microcoded *and* a heavy AVX-512 op, so it alone dropped
  the core to turbo licence level 2 and taxed every other phase in the search.
  The fix is a second `_bc_transpose!` method for `profs::AbstractMatrix{T}`
  (same type as the tile) that gathers `BJ` values into an `NTuple` and stores it
  as one 32-byte aggregate, which LLVM cannot turn back into a scatter.
  - **Measured on fitzroy, `-t 1`, stock `native`, candidates byte-identical:**
    `core_power.lvl2_turbo_license` **57.5% of all cycles → 0**, effective clock
    **2.06 → 2.84 GHz**, and `precision_ab.jl` reads `:f64` 14.33 s vs `:f32`
    **11.88 s (1.206x)** — i.e. `:f32` on `native` now equals the 11.92 s that
    previously needed `--cpu-target=skylake`. Every phase Δ flipped negative
    (`interp` +34.6% → **−1.4%**, `gate+metric` +51.7% → **−4.0%**,
    `decim-metric` +46.4% → **−13.1%**, `decim-brfft` +15.3% → **−25.6%**).
  - **The laptop agrees, and always would have**: `foops` has no AVX-512, so it
    never emitted the scatter. Re-measured there `-t 1`, `:f32` is **1.190x**
    (native) / 1.221x (`--cpu-target=skylake`, a no-op there) — against
    fitzroy-under-AVX2's 1.189x. **The two hosts were never really disagreeing
    about `:f32`; one of them was running a scatter.** The recorded
    "do not merge at any thread count" verdict is dead on both.
  - **`--cpu-target` is no longer needed for anything.** It stays useful only as
    a diagnostic.
  - **`:f32` is still not the default**, but the argument against it ("it loses
    at `-t 1`") is now gone on both hosts — that is a decision to take
    deliberately, since `:f32` candidates are not bit-identical to `:f64`.
  - **Do NOT study fitzroy's codegen on the laptop.** `--cpu-target=skylake-avx512`
    on `foops` reports **zero** `zmm` for these kernels — the host has no AVX-512
    and Julia's JIT will not emit what it cannot run. It looks like a clean
    answer and is an artifact. On fitzroy, `native`, `skylake-avx512` and
    `native,prefer-256-bit` all emit the identical scatter, so this was never a
    `prefer-vector-width` defaulting problem either.
  - `bench/avx512_probe.jl` is a **standalone** reproducer (no package, no data,
    no `perf`): it reports the CPU's AVX-512 features, whether the shipped kernel
    emits a scatter, both kernels' throughput, and — the useful part on a host
    without `perf` — times a fixed *scalar* loop alternated with each kernel, so
    a licence downclock shows up as unrelated scalar code getting slower. It has
    been run on all three hosts; send it anywhere a new CPU needs classifying.
  - **`core_power.lvl{0,1,2}_turbo_license` is the right instrument, better than
    `cycles/ref-cycles`.** It names the licence level directly instead of leaving
    you to infer it from a clock ratio. It also *cleared FFTW* without an
    assumption: FFTW ignores `--cpu-target`, yet the `skylake` arm showed zero
    lvl2 cycles, so the 512-bit code had to be Julia's.
  - **What did NOT work, so it is not re-guessed:** dropping `@simd`, hand-
    unrolling with `@nexprs`, making the `b` trip count opaque, and hoisting `i`
    innermost all still scatter — LLVM re-derives it from the loop nest. Only
    making the *store* a single aggregate stops it. `ntuple`-gather with a
    *looped* store scatters again at `BJ >= 8`; a `VecElement` tuple store is
    zmm-free only at `BJ = 4` and is 4x slower.
  - **`BJ = 8` is still right for both methods** (fitzroy, µs/call, `B = 128`,
    `nbins = 120`): new kernel 11.5/**9.5**/10.4 at `BJ = 4/8/16` against the old
    nest's 12.4/9.3/23.0, and the `Float64` nest's 15.2/**11.0**/46.8. The new
    kernel is deliberately **not** used for `Float64` profs, where the widening
    already makes the scatter unprofitable and it measures 36.6 vs 11.0 µs.
  - **Confirmed on a THIRD microarchitecture, and the fix wins there for a
    different reason.** A Zen 4 laptop (Ryzen 7 7840HS, AVX-512 double-pumped at
    256 bits, no Skylake-style licensing) run by one of Scott's students,
    2026-08-24 via `bench/avx512_probe.jl`:
    - It emits the scatter **more** aggressively than fitzroy — 131 `zmm` and 24
      scatters for `Float32` profs, and it scatters the **`Float64`** path too
      (48 `zmm`, 8 scatters), which fitzroy does not.
    - It has **no licence penalty**, exactly as predicted: the probe's scalar
      neighbour reads 0.983x next to the scatter kernel against 0.970x next to
      the aggregate one, i.e. no effect either way.
    - **And the fix still wins 1.32x** on `Float32` in isolation (4.59 vs
      6.08 µs), because scatter is simply a slow instruction there.
    So: Skylake-SP pays for the scatter in *clock*, Zen 4 pays for it in
    *throughput*, and `foops` never emits it. **Three hosts, three behaviours,
    one fix** — which is a much better position than a host-specific flag.
    - It also independently confirms the dispatch: the aggregate kernel on
      `Float64` profs is **0.49x** there (14.47 vs 7.08 µs), the worst of the
      three hosts. Restricting it to `profs::AbstractMatrix{T}` is not an
      optimisation detail — using it everywhere would be a large regression.
  - **In isolation the fix is a wash on fitzroy (9.5 vs 9.3 µs); the win there is
    entirely in what it stops doing to the rest of the program.** That is the sharpest version yet
    of this file's standing warning about microbenchmarks: no per-kernel timing
    of `_bc_transpose!` could ever have found this.

- **FOUND (2026-08-24, fitzroy): every recorded `:f32` penalty was AVX-512
  license-based DOWNCLOCKING, not memory. `--cpu-target=skylake` turns `:f32`
  from 0.846x into 1.189x at `-t 1`, and the best configuration on this host is
  now `:f32` + AVX2 at 11.92 s against the shipped default's 13.50 s (1.13x) —
  single-threaded, which was the entire argument against `:f32`.**
  - **The measurement that found it, after five wall-clock hypotheses had died.**
    `perf` needs no root here (`perf_event_paranoid = 2` already allows
    user-space counters, which is all this workload is). Per gate call at `k=1`:

    | | cycles | instructions | uops | IPC | cycles/ref | effective clock |
    |---|---|---|---|---|---|---|
    | `:f64` | 1,197,983 | 1,391,085 | 2,641,773 | 1.16 | 1.327 | **2.92 GHz** |
    | `:f32` | **857,300** | **1,235,295** | **2,302,730** | **1.44** | 0.807 | **1.78 GHz** |

    `:f32` does strictly *less* work by every counter and still loses, because the
    core drops to 1.78 GHz. `ref-cycles` ticks at the nominal 2.2 GHz, so
    **`cycles/ref-cycles` IS the turbo ratio** — that one ratio is the whole
    diagnosis, and nothing in a wall-clock A/B can see it.
  - **Confirmed by removing AVX-512.** Gate µs/call, `nbuf` = distinct profile
    buffers cycled: with AVX-512, `:f64` 361.6/455.6 and `:f32` 520.3/660.1 at
    `nbuf` 1/64; with `--cpu-target=haswell`, `:f64` 335.7/434.3 and `:f32`
    342.8/**375.2**. The `:f32` penalty vanishes, and `:f64` gets *faster* too —
    AVX-512 is a net loss for both precisions here, merely catastrophic for `:f32`.
  - **Whole search, `-t 1`, phase Δ for `:f32` vs `:f64`** — every recorded
    penalty evaporates or reverses:

    | phase | AVX-512 | AVX2 (`--cpu-target=skylake`) |
    |---|---|---|
    | interp | +34.6% | **−3.5%** |
    | decim-metric | +46.4% | **−2.9%** |
    | gate+metric | +51.7% | **−3.1%** |
    | decim-brfft | +15.3% | **−25.6%** |
    | zero-ftprofs | −59.9% | −72.5% |

    Wall clock: `:f64` 13.50 s (native) / 14.18 s (skylake); `:f32` 15.96 s /
    **11.92 s**. Candidates unchanged (3) in all four arms.
  - **This closes two long-standing "cause unidentified" items.** The `interp`
    `:f32` penalty — where `Float32` *weights* were tried and did not fix it, and
    the note concluded "that hypothesis is wrong and the cause is unidentified" —
    is this. So is the `decim-brfft` `:f32` penalty. Both were the clock.
  - **The recorded claim that "LLVM emits 256-bit vectors on Skylake-SP by
    default" (used to justify ignoring AVX-512 when tuning `DIRECT_GROUP_V` and
    `_BC_BATCH`) is FALSE for these kernels** — but only for one of them, and not
    for the reason it sounds like. A codegen audit of every hot kernel (the
    interpolator's `_group_lanes`/`_group_store!`, `_bc_scan_batch!`,
    `_boxcar_scan`, `_boxcar_psum!`, `_median!`, `_select!`) finds **zero** 512-bit
    code in all of them. The *only* source was `_bc_transpose!` on `Float32`
    profs, and what it emitted was a scatter.
  - **Treat every constant tuned by wall clock on fitzroy as suspect**:
    `_BC_BATCH = 128`, `_BC_TR_BJ = 8`, `DIRECT_GROUP_V = 32` were all tuned
    *through* a variable license level, so their optima may move now the scatter
    is gone. `_BC_TR_BJ = 8` has been re-confirmed for both transpose methods
    (see the FIXED entry); `_BC_BATCH` and `DIRECT_GROUP_V` have **not** been
    re-swept and remain open. Note this only ever mattered for `:f32` — the
    `:f64` path never left licence level 1.
  - **Also re-read the four "the microbenchmark inverted in situ" entries in this
    file.** They are recorded as cache-warmth stories; some may be this instead,
    since a different instruction mix trips a different license. In this
    investigation three harnesses disagreed about the *direction* of the `:f32`
    effect for exactly that reason.
  - **CLOSED — this was the open question, and the answer was not a flag.** See
    the FIXED entry above: it is one `vscatterqps` in `_bc_transpose!`, removed in
    source. `prefer-vector-width` was a red herring; the vectoriser was choosing a
    *scatter*, not merely a wide vector, and pinning the width would not have
    stopped it. FFTW is unaffected either way (compiled C, its own runtime
    dispatch) — confirmed by the licence counters, not assumed.
  - **Threaded: measured, and the win grows with thread count exactly as
    package-wide licensing predicts.** Whole search, wall clock, candidates
    identical (3) in all sixteen arms:

    | threads | `:f64` native | `:f32` native | `:f64` AVX2 | **`:f32` AVX2** | best vs shipped default |
    |---|---|---|---|---|---|
    | 1 | 13.50 s | 15.96 s | 14.18 s | **11.92 s** | **1.13x** |
    | 4 | 4.49 s | 4.67 s | 4.44 s | **3.54 s** | **1.27x** |
    | 16 | 1.84 s | 1.51 s | 1.74 s | **1.13 s** | **1.63x** |
    | 20 | 1.75 s | 1.38 s | 1.60 s | **1.01 s** | **1.73x** |

    `:f32` + AVX2 is the best configuration at *every* thread count, `-t 1`
    included. `:f64` barely cares about the target (13.50/14.18, 1.84/1.74,
    1.75/1.60) — it is `:f32` that AVX-512 was punishing.
  - **The `-t 20` anomaly did NOT reproduce, and licensing does not explain it.**
    The earlier run read 1.384x at `-t 16` and 1.029x at `-t 20`; re-measured the
    same day these are 1.220x and 1.268x, i.e. monotone. That earlier `-t 20`
    point was scatter, not a cross-socket or licensing effect — **do not build on
    it.** For the same reason the recorded "`-t 20` is no longer faster than
    `-t 16`" needed re-checking: `:f64` here is 1.84 s at 16 and 1.75 s at 20,
    and the full 2026-08-24 sweep confirms 1.42 vs 1.29 s. That claim is now
    retired — see the scaling table above.
- **Done (2026-08-22, workstation): `Float32` interpolation weights are now the
  DEFAULT, and the old "1.64x SLOWER" verdict was not merely void but backwards.**
  That verdict blamed the per-trial cross-lane reduce; the trials-axis kernel
  deleted it, so the mechanism went with it. Re-measured in situ on the machine
  that produced it, each type at its own best `DIRECT_GROUP_V` — and the knee
  *moves*, because `Float32` doubles the lanes per register and so halves the
  accumulator count at fixed `V`:

  | WT | V=8 | V=16 | V=32 | V=64 |
  |---|---|---|---|---|
  | `Float64` | 21.2% | **17.5%** | 18.1% | 29.8% |
  | `Float32` | — | 15.5% | **13.3%** | 16.8% |

  (interp share of accounted time, PM0063 0.1–33.3 Hz `-t 1`.) **24% off the
  interp phase**, ~4% end to end, and it halves the 1.45 MB table. `V` is left a
  single const at 32: the `Float64` path is now only the reference/pin path and
  pays 3% of interp for it, which is not worth plumbing a per-type `Val`.
  - **The accuracy cost is real, tiny, and end-to-end invisible.** Worst relative
    error vs the exact `fourier_interp` goes 6.8e-11 → **2.0e-7** — five orders of
    magnitude under the ~1.27% of signal power the `m=16` truncation already
    discards and the ~6.5% the `hidr` grid costs at the top harmonic. The PM0063
    `.cohout` down to threshold 6 (21 candidates) is **byte-identical** between
    weight types, frequencies to 12 decimals included, at `m = 16, 20, 24, 32`.
  - **`m` does NOT trade against this — they are different errors.** `m` sets the
    *truncation* of the Eqn.-30 sum; the 2e-7 is *rounding* in the weights, and
    raising `m` nudges it up, not down. What `Float32` does buy is a cheaper
    kernel bin: **`m = 32` in `Float32` costs 3.44 s of interp against
    `Float64`/`m = 16`'s 3.71 s**, so halving the truncation loss (1.27% → 0.63%,
    a real ~0.3% of recovered S/N) is now available for *less* than today's
    default cost. `m` stays 16 — 0.3% of S/N is not worth 3% of wall clock in a
    blind search — but `--m 32` is the cheap option it was not before.
  - **The pins do not move, and that is the `sigma_center` pattern again.**
    2.0e-7 does not fit under the `worst < 1e-7` interpolator pin or
    `PIN_TOL = 1e-8`; those are machine-precision statements about the *exact*
    kernel, and loosening them to swallow a deliberate constant would let a real
    tabulation bug hide with it. `chunk_metrics` gained a `weights` kwarg, the
    three affected pins ask for `Float64` explicitly, and **two new pins cover the
    shipped `Float32` path** at the ~1e-6 it achieves. 462 tests, up from 460.
  - The trade is disclosed at runtime under `--verbose`, quoted against the two
    errors that dominate it so it is read in proportion.
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
- **Where the single-thread time is now — and the two machines DISAGREE about it,
  so quote the host.** `:f64`, PM0063, 0.1–33.3 Hz, `-t 1`, phase share of
  accounted time:

  | | metric | interp | decim-brfft | base brfft |
  |---|---|---|---|---|
  | i7-10510U (laptop), after 2026-08-22 | ~33% | ~25% | ~24% | ~14% |
  | Xeon Silver 4114 (fitzroy), same code | **48.8%** | 17.6% | 21.2% | 10.3% |
  | fitzroy, after the workstation pass | **45.5%** | **13.8%** | 22.8% | 10.9% |
  | fitzroy, after the transpose fix | 31.6% | 19.3% | **29.5%** | 16.0% |

  The metric used to be half the runtime on the workstation and a third on the
  laptop, with the interpolation the reverse; the transpose fix has brought the
  two hosts into rough agreement. A knob tuned on one host is still not
  automatically right on the other — that is how `_BC_BATCH` came out at 64 there
  and 128 here, and how the transpose sat unfixed for a week behind a
  laptop-derived "bandwidth wall". Re-baseline before optimising, and say which
  machine.
  - **`decim-brfft` has now been looked at (2026-08-24) and is close to its floor.
    Do not re-open it without a new idea — three have been measured and rejected.**
    `bench/decim_brfft_bench.jl` splits it, in both precisions, on the production
    `Workspace` holding a genuine chunk. Re-baselined at `-t 1`: `decim-brfft`
    29.6–30.3%, metric ~31%, interp ~19%, base `brfft` ~16%; at `-t 16`
    `decim-brfft` rises to 33% and the two transforms together are 51%.
    - **Small-transform inefficiency is NOT the cause.** Per output bin the
      *contiguous* decimated transforms cost 1.25–2.08 ns (`:f64`) against the
      base 120-bin pass's 2.048 — as good or better. FFTW is fine at these sizes.
    - **The stride is the entire excess:** 972 µs strided vs **660 µs** on a
      contiguous copy of the identical stack (1.47x; 1.22x in `:f32`). It grows
      with `k` in `:f64` (1.17x at `k=2` → 2.14x at `k=6`) because a column is
      61·16 B ≈ 15 lines and at `k ≥ 4` the 64 B stride yields exactly *one*
      useful element per line; in `:f32` the stride is half as long, 2–4 elements
      survive per line, and the cost stays flat at ~1.2x.
    - So **312 µs (`:f64`) / 127 µs (`:f32`) is the budget** any de-striding
      scheme must beat. Measured: the per-`k` `copyto!` that 51ccdf6 deleted is
      **1585 µs** (5x over — that is why it lost); blocking it by the profile
      axis, the trick that was worth 3.5x on `_bc_transpose!`, gets it only to
      **592 µs**; a **fused single pass** over the columns writing all five stacks
      is **269 µs** — 5.9x better than naive and the only one under budget, but it
      leaves just 43 µs, i.e. **1.046x on the phase, ~1.4% end-to-end**, costs
      **+3.0 MB per workspace** (bad exactly where the phase hurts most), and is a
      **net loss in `:f32`** (781 vs 701 µs). It is also already at **18.3 GB/s**
      against the ~21 GB/s this host delivers, so it will not improve.
    - **Fusing the stack writes into `fill_harmonic_row_direct!` is dead, and for
      a structural reason.** The appeal was zero extra reads — the interpolator
      holds harmonic `h` in registers and `h` belongs to stack `k` when
      `h % k == 0`. But `_group_store!` writes a *row* of a column-major matrix,
      already strided by 976 B, and the stacks are the same shape. The pure
      scatter stores alone, in the interpolator's harmonic-major order, are
      **506 µs (`:f64`) / 395 µs (`:f32`)** — 1.6x and 3.1x over budget before any
      arithmetic or transform. The same stores column-major are **147/103 µs**,
      3.4x cheaper, which is just the fused gather again. The interpolator's loop
      order is wrong for these buffers and cannot be fixed in place.
    - The remaining layout escape, `(Nprof, Hₖ+1)` transforming along dim 2, is
      **already recorded as 2–3x slower** (see the four measured dead ends above).
    - **The isolated bench does NOT track the search in `:f32`.** Per chunk
      (phase ÷ 4084 chunks): `:f64` 1041 µs in situ vs 972 isolated (1.07x), but
      `:f32` **1153 in situ vs 701 isolated (1.64x)**. So this bench's 0.721x
      `:f32` speedup is worth nothing in the real search, where `:f32` is a 10.9%
      *loss* on this phase at `-t 1`. Score precision with `bench/precision_ab.jl`
      only. **That 1.64x gap is unexplained and is now the more interesting
      thread** — see the `:f32` entry below.
  - For `interp` the structural cost *was* the per-trial horizontal reduce; that
    was fixed by vectorising across trials (a159706) and then again by
    `Float32` weights, and at 13.8% it is no longer the place to look.
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
