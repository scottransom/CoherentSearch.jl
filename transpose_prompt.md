# Next session: kill the tile transpose (the largest single item in the search)

`bench/metric_bench.jl` on fitzroy, per chunk summed over `k = 1…6`: σ̂ 181 µs,
**transpose 1516 µs**, width scan 337 µs, of a 2198 µs metric total. The metric is
~45% of runtime here, so **the transpose alone is ~31% of the whole search** —
larger than any FFT phase. It is the target.

Read first, and do not re-derive: commits `7361279`, `055b9db`, `dbe05e8`, and
the 2026-08-22 workstation entries in `CLAUDE.md` (especially the transpose entry
immediately above "In-situ phase timers are now permanent", and the four measured
dead ends just above it — one of which this work has already partly overturned).

## Why the old "don't bother" verdict does not apply

The transpose sits in the dead-end list because blocking/reordering gave
0.56–1.03x and was written off as "a ~15.8 GB/s L3 wall, not a code defect".
That is not true on this machine:

* at `k = 1` it moves 1.97 MB in + 0.98 MB out in 626.6 µs = **4.7 GB/s**, and
  the same 4.7–4.8 GB/s at *every* `k`;
* measured achievable single-core copy here is **21.1 GB/s at a 2 MB footprint**
  (10.1 GB/s at 32 MB, i.e. out of cache).

So it runs at ~22% of what the machine gives at its own footprint, and the fact
that the rate does not move with size is the signature of an **access-pattern**
limit, not a bandwidth one.

**The mechanism is write-scatter.** Reading `profs[:, j]` is contiguous (960 B per
profile). Writing `tile[(i-1)*B + b]` for a fixed profile `b`, `i` running over
phase, walks stride `B*4 = 512 B` — so 120 writes touch 120 distinct cache lines
and dirty each for one 4-byte word. That is ~1/16 of the write bandwidth used.

## Before touching anything

1. **Re-baseline.** `bench/metric_bench.jl` and `bench/precision_ab.jl --arms f64
   --reps 3` at `-t 1`. Confirm transpose ≈ 1516 µs and the ~4.7 GB/s figure.
2. **Establish the floor.** Write a standalone transpose microbench (naive vs
   blocked vs whatever FFTW gives) so there is a number to aim at — but treat it
   as an upper bound only. The isolated-vs-in-situ gap has bitten three times in
   this repo (metric bench promised 1.74x, delivered 1.13x). **Decide in situ.**
3. **Understand what the gate actually requires.** `_boxcar_gate!` transposes to
   `(B, nbins)` so the prefix-sum recurrence runs with `b` innermost — the phase
   axis carries a serial dependency and is only 20–120 long, so vectorising along
   it is not an option. Confirm that constraint still holds before assuming the
   tile layout is negotiable; the transpose is structural, so this is about making
   it cheap or making someone else do it, not deleting the requirement.

## Ideas, in the order I would try them

1. **Make the `brfft` write the tile layout directly — ALREADY PROTOTYPED AND
   MEASURED, 2026-08-22. Start here; most of the risk is gone.**

   Scott was right that FFTW's guru interface strides its *outputs*
   independently of its inputs, and FFTW.jl reaches it: `dims_howmany(X, Y, …)`
   reads `strides(X)` and `strides(Y)` separately and hands both to
   `fftw_plan_guru64_dft_c2r`. The typed two-array `rFFTWPlan` constructor
   demands `StridedArray`, which `PermutedDimsArray` is *not* (it has the right
   strides but is outside the type union), so the probe calls the guru entry
   point directly — 12 lines. It is checked in as
   **`bench/guru_transpose_probe.jl`**; run it first to reproduce.

   The plan asks for input `(nh+1, Nprof)` column-major (transform-dim stride 1,
   batch stride `nh+1`) and output **profile-major** `(Nprof, nbins)` (phase
   stride `Nprof`, profile stride 1) — which *is* the gate's tile layout.
   Measured at `k = 1`, `nbins = 120`, `Nprof = 2048`, `PATIENT`:

   | | µs |
   |---|---|
   | dense `brfft` (`PRESERVE_INPUT`, as shipped) | 491.8 |
   | guru profile-major `brfft` (`PRESERVE_INPUT`) | 662.5 (1.35x the dense plan) |
   | today's scattered transpose → `Float32` tile | 499.4 |
   | contiguous `Float64`→`Float32` narrowing | 128.5 |

   * **today = 991.2 µs**
   * **guru, `:f64` = 791.0 µs → 1.25x** (still needs the narrowing pass, but it
     is now *contiguous* — 128.5 µs against the scattered 499.4 µs)
   * **guru, `:f32` = 662.5 µs → 1.50x** — in `:f32` the guru output *is* the tile
     type, so the transpose disappears outright with nothing to replace it

   **`--precision f32` therefore stops being a marginal flag and becomes the
   point.** Idea 3 below composes with this one rather than competing.

   Three things the probe already establishes, so do not rediscover them:

   * **`PRESERVE_INPUT` is mandatory and is not free.** FFTW's c2r destroys its
     input by default (measured: max|Δ| = 53.8 without the flag), and the
     decimated passes read stride-`k` views of `ftprofs` *after* the base
     transform. FFTW.jl's `plan_brfft` sets it (`fft.jl:879`); the probe does too.
     The 1.35x above is *with* it. An earlier number without it was optimistic.
   * **The result is NOT bit-identical**: dense vs guru agree to 1.421e-14
     relative, because `PATIENT` picks a different algorithm once the output
     strides change. So **the `.cohout` byte-identity expectation below does not
     apply to this idea** — it is a different transform, not a data move. Expect
     last-digit S/N motion and check it the way the σ̂ change was checked (same
     candidates at the same frequencies, S/N moving in the last printed digit).
   * **Only `k = 1` has been probed.** The five decimated transforms already take
     a stride-`k` *input* view, so under this scheme they become strided on both
     sides. That is the case most likely to disappoint — measure it before
     committing to the design.

   Rough projection if the `k = 1` ratios hold across the ladder: the extra FFT
   cost is ~35% of ~1412 µs of transform ≈ 494 µs, against 1516 µs of transpose
   removed, so ~700 µs/chunk saved in `:f64` and ~1020 µs in `:f32` — call it
   **~14% and ~21% of total runtime**. Treat those as upper bounds: every isolated
   projection in this repo has shrunk in situ.

2. **Blocked transpose with cache-line-filling inner blocks.** The previous
   blocking attempt is recorded as 0.56–1.03x, but at `B = 32/64`; `B` is 128 now
   and the bandwidth headroom says there is room. Be specific about the block
   shape: 16 `Float32` = 64 B = exactly one cache line, so a **16 profiles × 16
   phases** inner block turns 16 scattered 4-byte writes into one full-line write.
   That is precisely the write-scatter fix, and it is not obviously what was tried.

3. **`--precision f32` halves the read side** (`profs` becomes `Float32`, the tile
   already is). Already measured to help these exact phases: `gate+metric` −8.1%,
   `decim-metric` −5.4% at `-t 1`, and a net win at ≥8 threads (1.17x at `-t 8`,
   1.34x at `-t 20`). If the transpose gets cheap, re-measure — the `:f32`
   crossover will move again, as it has four times now.

4. **Fuse the prefix sum into the transpose.** The transpose's output is read
   immediately by the prefix-sum pass. Computing the running sum while the tile is
   being written saves one full pass over the tile. Note `CLAUDE.md` records that
   *full* fusion of prefix + widths was 1.33x slower (AVX2 register pressure), so
   fuse only the one pass, and measure rather than assume the register budget.

## Gates and discipline

* `.cohout` byte-identity is the right gate for ideas 2 and 4 (pure data
  movement — anything else is a bug). **It is NOT the gate for idea 1**, which
  changes the FFTW algorithm and moves results by ~1e-14 relative; there, check
  that the candidate set and frequencies are unchanged and that S/N moves only in
  the last printed digit. Diff either way; do not eyeball.
* `Pkg.test()` (466 tests) and `julia --project=crossval
  crossval/crossval_accuracy.jl` (DEFAULT_PY is this machine's).
* **Compare arms by phase-share of accounted time, not seconds.** On this host
  within-run rep-to-rep scatter is only ±1%, but drift *between invocations* is
  ~6% — the `_BC_BATCH` sweep spread 6% across four values whose shares were flat
  to 0.5%. Interleave arms across rounds regardless.
* Quote the host. The laptop and this machine disagree about the phase split
  (metric 33% vs 45%), which is how `_BC_BATCH` came out 64 there and 128 here.
