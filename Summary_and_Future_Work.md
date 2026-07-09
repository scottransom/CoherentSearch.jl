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

---

## 3. Next steps

> **Status: feature-complete.** The search, detection metric, candidate
> de-duplication, harmonic decimation, and candidate profile plots are all
> implemented, tested, and oracle-validated. The primary focus now shifts from
> features to **performance**: careful profiling of the hot loop (interpolation,
> batched inverse FFTs, allocation and memory-bandwidth behavior under threading)
> and acting on what it finds. The throughput-tuning and tiling items below are
> the concrete starting points for that work.

- **Throughput tuning script.** Port `examples/speed_test.py` to Julia and sweep
  the `finterp_fft` rate over both `fftlen` *and* `numbetween` to pick the
  per-harmonic sweet spots (the design's 1024 < `fftlen` < 65536 expectation),
  then feed the result back into `build_harmonic_plans`. Persisting **FFTW
  wisdom** so repeated runs skip planning is a cheap add-on here.
- **Tiling for very large chunks.** At the default `blocksize` each harmonic fits
  one transform; large `Nprof` (or small capped `fftlen`) will need the harmonic
  span split into overlapping tiles. The `fill_harmonic_row!` structure leaves
  room for this — add the tile loop when the benchmark wants a capped `fftlen`.
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

- **Threshold calibration across metric / `pexp` (to investigate).** The metric's
  numeric scale is *not* comparable across `--metric` or `--pexp`, so a fixed
  `--threshold` means different things in each configuration. On the test pulsar,
  the same signal reads ~28 at `:non`/`pexp=0.5`, ~20 at `:non`/`pexp=1.0`, and
  ~39 at `:sd2`/`pexp=0.5`; only `:non`/`pexp=1/2` is a calibrated equivalent-σ,
  and even that is single-trial (no trials factor). We need to work out how the
  detection threshold should be set for each metric/`pexp` — ideally derive (or
  empirically fit, from pure-noise runs) the false-alarm rate vs. threshold for
  each configuration so a single "sigma"-like knob has a consistent meaning, and
  fold in the number of independent trials searched. Until then, `--threshold`
  must be re-tuned by hand whenever `--metric` or `--pexp` changes.

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
