# CoherentSearch.jl — Summary and Future Work

This document summarizes the current state of the Julia port of the Python
`coherent_search` package and lays out the plan for Phase 2, which focuses on
performance: thread-local buffers for better parallel scaling and aggressive
reuse of FFTW plans.

---

## 1. What exists today (Phase 1)

A working, well-tested, multi-threaded coherent harmonic-summing search that is
numerically validated against the original Python code.

### Components

| File | Role |
|------|------|
| `src/fourierinterp.jl` | Fourier interpolation kernels (Eqn. 30 of [astro-ph/0204349](https://arxiv.org/pdf/astro-ph/0204349)) |
| `src/fileio.jl` | `mmap`'d PRESTO `.fft` reader + `.inf` metadata parser |
| `src/search.jl` | Block-parallel coherent harmonic-summing search |
| `bin/coherent_search.jl` | ArgParse command-line front-end |
| `test/` | 29 unit tests (golden values, analytic signals, indexing, irfft convention) |
| `crossval/` | Python-as-oracle accuracy + speed cross-validation |

### Key design decisions made in Phase 1

- **Indexing is isolated and audited.** The 0-based (Python, half-open slices)
  → 1-based (Julia, inclusive ranges) translation lives in one documented
  helper, `nearby_fourier_bin_range`, with the original Python slice arithmetic
  written out in comments.

- **The stateful `FourierInterpolator` was replaced with independent frequency
  blocks.** The Python version walks forward through the FFT with mutable
  `lobin`/`nextbin` state — hostile to parallelism. The Julia version exposes
  `block_metrics(ft, rfund, params)`, a self-contained function that takes a
  contiguous range of trial fundamental frequencies and returns the
  peak/|trough| metric for each, sharing no mutable state. `search()` partitions
  the frequency range into blocks and runs them under `Threads.@threads`;
  `search_block` thresholds a block's metrics into candidates. This structure
  also extends naturally to `Distributed.jl` for multi-node runs.

- **FFT conventions verified, not assumed.** `np.fft.fft`/`ifft` and Julia's
  `fft`/`ifft` share the same normalization, so the FFT-correlation interpolator
  ports directly. The one subtlety — `np.fft.irfft` vs FFTW's `c2r` handling of
  the DC and Nyquist bins' imaginary parts — is checked by a dedicated test;
  both ignore those imaginary parts, so the coherent fold matches.

### Verification status

- **All 29 unit tests pass.**
- **Accuracy cross-validation** (`crossval/crossval_accuracy.jl`) runs the
  original Python `coherent_search` as an oracle and agrees to **~3e-16 relative
  on the `finterp_FFT` kernel** and **~8e-16 relative end-to-end** (the full
  coherent-fold metric) on the bundled 10.0123 Hz test pulsar. This is the
  primary guard that the indexing and FFT conventions are correct.
- **Speed cross-validation** (`crossval/crossval_speed.jl`): the Julia kernel is
  currently **~2.1× faster single-threaded** than the Python kernel. This gap is
  almost entirely reduced call overhead and in-place operations on top of the
  same FFTW library — it is *not* yet the result of any of the Phase 2 work.

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

## 2. Where the time currently goes

The hot path, per harmonic per block, is `finterp_fft`:

1. zero a length-`fftlen` complex buffer,
2. scatter the relevant raw FFT bins into it (zero-stuffing),
3. forward FFT,
4. multiply by the cached, pre-FFT'd interpolation kernel,
5. inverse FFT,
6. slice out the valid region,

followed by a linear interp onto the exact harmonic frequencies and, once per
block, an `irfft` of the stacked harmonics.

Two things make this slower than it needs to be in the current code:

- **Every call allocates.** `finterp_fft` allocates `ftarr`, the two FFT
  outputs, and the returned slice on each of `nharms` calls per block, for every
  block. With `nharms = 32` and thousands of blocks this is a large amount of
  short-lived garbage, which both costs allocation time and pressures the GC —
  and GC pauses serialize threads, hurting parallel scaling specifically.

- **Every call re-plans (or uses generic transforms).** `finterp_fft` calls
  `fft`/`ifft`, which look up or build a plan each time rather than reusing a
  single measured plan bound to fixed-size buffers.

These are exactly the two axes Phase 2 targets.

---

## 3. Phase 2 plan — thread-local buffers + FFT-plan reuse

The goal is to make a block's inner loop **allocation-free** and **plan-stable**,
and to give each thread its own private working set so there is no contention or
false sharing between threads.

### 3.1 A per-thread `Workspace`

Introduce a `Workspace` struct that owns every mutable buffer and every FFT plan
needed to process one block, sized once from the search parameters:

```julia
struct Workspace
    fftlen::Int
    ftarr::Vector{ComplexF64}        # zero-stuffed input (reused, re-zeroed)
    spec::Vector{ComplexF64}         # forward-FFT output
    corr::Vector{ComplexF64}         # inverse-FFT output
    coeffs::Vector{ComplexF64}       # cached, pre-FFT'd interpolation kernel
    ftprofs::Matrix{ComplexF64}      # (nharms+1, L) stacked harmonics
    profs::Matrix{Float64}           # (2*nharms, L) real profiles
    fwd::FFTW.cFFTWPlan              # plan_fft!(ftarr)   — in-place
    inv::FFTW.cFFTWPlan              # plan_ifft!(spec)   — in-place
    irfftp::FFTW.rFFTWPlan          # plan_irfft(ftprofs, 2*nharms, 1)
end
```

Notes on the design:

- **Fixed `fftlen`.** Today `finterp_fft` derives `fftlen` per call from
  `(numbins + m) * numbetween`. To reuse one plan/buffer set we fix a single
  `fftlen` for the whole run (the largest a block needs, given `blocksize`,
  `numbetween`, `m`, and the top harmonic). The `numbins` actually used per
  harmonic varies, but a fixed-size transform with the unused tail left zeroed is
  both correct and plan-stable. This trades a little wasted FFT length for plan
  reuse — a good trade, since FFTW is fast on power-of-two lengths and planning
  is the expensive part.

- **In-place transforms.** `plan_fft!`/`plan_ifft!` operate in place on `ftarr`,
  removing two allocations per call. The forward transform overwrites `ftarr`,
  so we keep the zero-stuffed source pattern and re-scatter each call (cheap), or
  keep a separate `spec` buffer if we want to preserve the input.

- **Re-zeroing, not reallocating.** Between harmonics we only need to clear the
  positions that were written (the zero-stuffed bins), not the whole array — a
  targeted `fill!` over the scattered indices, or track-and-clear, avoids an
  O(fftlen) wipe per harmonic.

### 3.2 Refactor the kernel to write into a workspace

Add an in-place variant alongside the existing allocating one (which stays for
tests/clarity):

```julia
finterp_fft!(out, ws::Workspace, lobin, numbins, numbetween, ft, m)
```

It scatters into `ws.ftarr`, applies `ws.fwd`, multiplies by `ws.coeffs`,
applies `ws.inv`, and copies the valid slice into `out` (a view into
`ws.ftprofs`). No allocation. `block_metrics` then takes a `Workspace` argument
and threads it through all `nharms` harmonics and the final `irfft` (via
`mul!(ws.profs, ws.irfftp, ws.ftprofs)`).

### 3.3 Wire workspaces into the threaded driver

Allocate one `Workspace` per thread up front and index it inside the parallel
loop:

```julia
workspaces = [Workspace(params, blocksize) for _ in 1:nthreads()]
@threads for b in 1:nblocks
    ws = workspaces[threadid()]
    ...
    partials[b] = search_block!(ws, ft, rfund, params; threshold)
end
```

**Caveat to handle carefully:** `threadid()` is not a stable index under Julia's
task-migration scheduler (`:dynamic`). The robust patterns are (a) use
`@threads :static`, which pins iterations to threads and makes `threadid()`
stable, or (b) switch to a channel/task model where each task takes a workspace
from a pool. I'll prototype with `:static` (simplest, correct here since blocks
are uniform) and benchmark against a pool-based version. This is the single most
important correctness detail in Phase 2 — getting it wrong gives data races, not
just slowdowns.

### 3.4 Plan creation and thread safety

- **Plan once, before the threaded region.** FFTW *planning* is not thread-safe;
  *execution* of an already-built plan on distinct buffers is. So each
  `Workspace` builds its plans at construction time (single-threaded), and the
  parallel loop only executes them.
- **`FFTW.set_num_threads(1)`.** Keep FFTW itself single-threaded and parallelize
  at the Julia-task level. With many small independent transforms, nested FFTW
  threading contends and loses; one transform per Julia thread is the right
  granularity.
- Consider `FFTW.MEASURE` (or `PATIENT`) when building plans, and optionally
  persisting **FFTW wisdom** so repeated runs skip planning cost.

### 3.5 Expected payoff and how we'll measure it

- **Single-thread:** removing per-call allocation and re-planning should give a
  further speedup on top of the current 2.1×; the exact factor depends on how
  FFT-bound the kernel is at the chosen `fftlen`.
- **Multi-thread:** the bigger win. Eliminating GC pressure should take parallel
  scaling from "good but GC-throttled" toward near-linear in cores for the
  embarrassingly parallel block loop.

Measurement plan:

- Extend `crossval/crossval_speed.jl` with a **thread-scaling sweep** (1, 2, 4,
  8, … threads on the same band) reporting throughput (trials/s) and speedup vs
  one thread.
- Track **allocations per block** (`@allocated` / `BenchmarkTools` memory
  estimates) and assert they drop to ~0 in the hot loop.
- Keep the accuracy cross-validation green throughout — every refactor must
  still match the Python oracle to ~1e-15, so performance work cannot silently
  change results.

---

## 4. Other Phase 2+ items (lower priority)

- **Candidate de-duplication / harmonic filtering** (`--noremove` is parsed but
  not yet acting): collapse the cluster of trials around a true signal to a
  single candidate, and remove harmonically-related duplicates.
- **`Distributed.jl` backend** reusing the same block abstraction, for
  cluster-scale searches across nodes.
- **Statistically meaningful detection metric.** The current peak/|trough| ratio
  is a shape statistic, not a calibrated significance; revisit once performance
  is settled (this also differs from the documented "equivalent gaussian sigma"
  in the Python CLI help).
- **Broader real-data validation** beyond the single artificial test pulsar.

---

## 5. Summary

Phase 1 delivered a correct, parallel, well-tested foundation whose numerical
results are pinned to the Python implementation at machine precision. Phase 2
turns the parallel structure that's already in place into real performance:
per-thread workspaces to make the hot loop allocation-free, and reused,
pre-measured FFTW plans bound to fixed-size buffers — measured by a thread
scaling sweep and guarded the whole way by the existing Python-as-oracle
cross-validation.
