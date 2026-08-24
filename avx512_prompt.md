# Next session: is AVX-512 downclocking costing us everywhere?

**Written 2026-08-24 on fitzroy. Start by reading the `FOUND (2026-08-24, fitzroy)`
entry in `CLAUDE.md` — it has the numbers; this file has the plan.**

## What was found

On fitzroy (Xeon Silver 4114, Skylake-SP, 2x10 cores) every recorded
`--precision f32` penalty turned out to be **AVX-512 license-based
downclocking**, not memory behaviour. Per gate call: `:f32` executes 28% fewer
cycles, 11% fewer instructions and 13% fewer uops at higher IPC (1.44 vs 1.16),
and is still slower — because the core drops from **2.92 GHz to 1.78 GHz**.

The diagnostic is one ratio. `ref-cycles` ticks at the nominal TSC frequency, so
**`cycles / ref-cycles` is the turbo ratio**. It read 1.327 (`:f64`) against
0.807 (`:f32`). `perf` needs **no root** — `perf_event_paranoid = 2` already
permits user-space counters, which is all this workload is.

Removing AVX-512 (`--cpu-target=skylake`) at `-t 1`, whole search:

| cpu-target | `:f64` | `:f32` | `:f32` vs `:f64` |
|---|---|---|---|
| native (AVX-512) | 13.50 s | 15.96 s | 0.846x |
| skylake (AVX2) | 14.18 s | **11.92 s** | **1.189x** |

Best config becomes `:f32` + AVX2 at 11.92 s against the shipped default's
13.50 s — **1.13x, single-threaded**, which was the entire argument against
`:f32`. Candidates unchanged (3) in all four arms. Per-phase, every recorded
`:f32` penalty evaporates or reverses (`interp` +34.6% -> -3.5%, `gate+metric`
+51.7% -> -3.1%, `decim-metric` +46.4% -> -2.9%, `decim-brfft` +15.3% -> -25.6%).

## Task 1 — the laptop (do this first; it is cheap and it disambiguates a lot)

**Establish whether the laptop can even exhibit this**, then decide what it means.

```sh
lscpu | grep -oE 'avx512[a-z]*' | sort -u        # empty => cannot happen here
lscpu | grep -E 'Model name|^CPU\(s\)|Core|Socket'
```

* **No AVX-512** (Comet/Ice Lake, Alder Lake+, most consumer Intel): then the
  effect is impossible there — and that is itself a finding. `CLAUDE.md` records
  the laptop and fitzroy disagreeing about `_BC_BATCH` (64 vs 128), `_BC_TR_BJ`
  (8 vs 128) and the `:f32` verdict. **Some of those "the two hosts disagree"
  entries may be one mechanism rather than host idiosyncrasy.** Re-measure the
  laptop's `:f32` verdict and see whether it now agrees with fitzroy-under-AVX2.
* **AMD Zen 4/5**: has AVX-512 but double-pumped 256-bit with no Skylake-style
  licensing. AVX-512 may be a genuine win there, which would make the right
  answer per-host rather than global. Measure, do not assume.

Then, whatever the CPU:

```sh
cd <repo>
julia --project=bench -t 1 bench/precision_ab.jl PM0063_034C1_DM445.0_red.fft \
      --arms f64,f32 --reps 5
julia --cpu-target=skylake --project=bench -t 1 bench/precision_ab.jl \
      PM0063_034C1_DM445.0_red.fft --arms f64,f32 --reps 5
```

Compare the two `f32 vs f64` ratios and the per-phase table. On a host with no
AVX-512 they should be the same to within scatter; if they are not, something
other than licensing is in play and that is worth knowing.

## Task 2 — finish fitzroy (via ssh; a sweep was left running)

A `-t 4 / 16 / 20` x `native / skylake` x `f64 / f32` sweep was started and may
not have completed. Check for it first:

```sh
ssh fitzroy 'ls -l /tmp/claude-*/**/scratchpad/cputarget_threads.txt' 2>/dev/null
```

If it is missing or partial, re-run it (each `skylake` arm pays a full package
recompile, so budget ~20 min):

```sh
ssh fitzroy 'for t in 4 16 20; do
  for tgt in native skylake; do
    echo "### -t $t $tgt"
    if [ "$tgt" = native ]; then EXTRA=""; else EXTRA="--cpu-target=$tgt"; fi
    julia $EXTRA --project=bench -t $t bench/precision_ab.jl \
       PM0063_034C1_DM445.0_red.fft --arms f64,f32 --reps 5 2>&1 |
       sed -n "/=== medians ===/,/^$/p"
  done
done'
```

**Prediction to test, not to assume:** licensing is package-wide and tightens as
more cores go active, so the AVX-512 penalty should be *larger* threaded. If so
it also explains the recorded `-t 20` anomaly (1.029x against `-t 16`'s 1.384x),
which is currently written up as a possible cross-socket effect.

**Make sure nobody is using fitzroy.** It is Scott's desktop; Chrome and Zoom
alone move these numbers by more than the effects being measured. `uptime` and
`ps -eo pcpu,comm --sort=-pcpu | head` before trusting anything.

## Task 3 — the actual fix (the open question)

`--cpu-target=skylake` is the **diagnostic instrument, not the fix**: it is
process-wide, must be passed by the user, and disables AVX-512 for code that
might want it. The targeted knob is LLVM's `prefer-vector-width=256`. Open
questions, in order:

1. Can Julia apply it at function granularity? (`@turbo` from LoopVectorization,
   which is already a bench dep and works on 1.12, may be able to pin the width.)
2. If not, is a documented `JULIA_CPU_TARGET` / `--cpu-target` recommendation
   plus a runtime warning from `src/cli.jl` acceptable? A startup check could
   detect AVX-512 + `:f32` and say so.
3. Which kernels actually emit 512-bit code? `@code_native` on
   `_bc_transpose!`, `_bc_scan_batch!`, `_group_lanes` and `_boxcar_scan`,
   grepping for `zmm`. This tells you whether one loop is responsible or it is
   diffuse, and therefore whether a surgical fix exists at all.

FFTW is unaffected either way (compiled C with its own runtime dispatch), so the
`brfft` phases will not move regardless.

## Task 4 — the constants are now suspect

`_BC_BATCH = 128`, `_BC_TR_BJ = 8` and `DIRECT_GROUP_V = 32` were **all tuned by
wall clock on fitzroy, through a variable license level.** Their optima may move
once vector width is pinned. Re-sweep each under a fixed target before trusting
them. `bench/tile_shape_bench.jl` and `bench/boxcar_bench.jl` already exist for
two of them.

Likewise, re-read the four "the microbenchmark inverted in situ" entries in
`CLAUDE.md`. They are written up as cache-warmth stories; some may be this
instead, since a different instruction mix trips a different license. During this
investigation **three harnesses disagreed about the direction of the `:f32`
effect** for exactly that reason — including one that appeared to reproduce the
in-situ result and did so by luck of instruction mix.

## Method notes, learned the hard way today

* **Verify a harness reproduces the effect before reading its counters.** Two
  perf harnesses were built and discarded because they showed `:f32` *faster*,
  the opposite of the search. At `k=1` an `:f32` `profs` is 0.98 MB and fits
  fitzroy's 1 MB L2 while the `:f64` one at 1.97 MB does not, so hammering a
  single buffer hands `:f32` an L2 advantage it never gets in the search. Cycling
  >= 16 distinct buffers reproduces the real behaviour.
* **Wall-clock bisection cannot see frequency.** Five hypotheses were killed by
  timing (loop shape, `_BC_TR_BJ`, type-based alias analysis, allocation, cache
  footprint) before the counters gave the answer in twenty minutes. When a
  timing result is inexplicable *and* harnesses disagree about its sign, reach
  for `cycles/ref-cycles` early.
* Quote the host and the date with every number. The two hosts have inverted
  each other's conclusions repeatedly, and now there is a mechanism for it.
