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

1. **Make the `brfft` write the tile layout directly.** Hand `mul!` a strided
   *output view* so FFTW scatters its results into `(Nprof, nbins)`-ish order as
   part of its final pass, and the transpose ceases to exist. **This is genuinely
   untried**: dead end (2) in `CLAUDE.md` transposed the *whole* array including
   the transform axis (`(Nprof, Hₖ+1)` transforming along dim 2) and was 2–3x
   slower — a different experiment. Keeping the transform along dim 1 and changing
   only where results land is the thing to test. Do a **feasibility check first**
   (does `plan_brfft` accept the output view at all, and does PATIENT planning
   find something sane for it?), because a yes here obviates ideas 2–4.
   Note this interacts with `DecimBuf`: the decimated plans already transform a
   stride-`k` input *view*, so they would become strided on both sides.

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

* `.cohout` must be **byte-identical** — this is pure data movement, so anything
  else is a bug. Diff it, do not eyeball it.
* `Pkg.test()` (466 tests) and `julia --project=crossval
  crossval/crossval_accuracy.jl` (DEFAULT_PY is this machine's).
* **Compare arms by phase-share of accounted time, not seconds.** On this host
  within-run rep-to-rep scatter is only ±1%, but drift *between invocations* is
  ~6% — the `_BC_BATCH` sweep spread 6% across four values whose shares were flat
  to 0.5%. Interleave arms across rounds regardless.
* Quote the host. The laptop and this machine disagree about the phase split
  (metric 33% vs 45%), which is how `_BC_BATCH` came out 64 there and 128 here.
