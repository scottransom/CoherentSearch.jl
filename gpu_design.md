# GPU support for CoherentSearch.jl — design and staging plan

**Status: planning, nothing implemented.** Written 2026-08-25 on `fitzroy`.

The optimisation phase on the CPU is closed (CLAUDE.md; `-t 1` now beats `rseek`
on every host we can test). This is the separate GPU track. It is *not* a
prerequisite for the §3.2 Monte Carlo or the paper, and it must not be allowed to
destabilise them: every CPU pin stays green throughout, because nothing in
`src/search.jl`'s existing path changes.

---

## 0. The machine, as measured today

| | |
|---|---|
| GPU | NVIDIA GeForce GTX 1080 (GP104, Pascal, **compute capability 6.1**) |
| SMs / cores | 20 SMs × 128 = 2560 CUDA cores, 1911 MHz boost |
| FP32 peak | **9.8 TFLOP/s** |
| FP64 peak | **0.31 TFLOP/s** — GP104 runs FP64 at **1/32** rate |
| FP16 | 1/64 rate — **useless**, do not consider it |
| Memory | 8 GB GDDR5X, **320 GB/s**, PCIe 3.0 x16 (~12 GB/s host↔device) |
| Driver / CUDA | 535.309.01 / **CUDA 12.2** (local toolkits present up to 12.9, but 12.2 is the driver ceiling without forward-compat) |
| Occupied | **723 MB by Xorg + desktop apps**, and `Xorg` (pid 3060) is *on this GPU* |

Two consequences of that last row, and they are the important ones:

- **The display runs on the search GPU.** Long-running kernels are subject to the
  X watchdog, and heavy occupancy makes the desktop stutter. Keep every kernel
  well under a second (chunk-sized launches are milliseconds, so this is free),
  and treat any wall-clock number taken while Scott is using the machine the same
  way `-t 20` numbers are already treated: **read shares and repeated minima, not
  single wall clocks.**
- **~7.2 GB is actually available.** That is the budget below.

**FP64 at 1/32 rate is the hard architectural constraint.** Every hot-loop
quantity must be FP32. Fortunately the code already went there: `precision = :f32`
and `Float32` interpolation weights are both defaults, and the AVX-512 work
(2026-08-24) removed the last reason they were not. **The GPU port inherits an
FP32 pipeline that is already pinned and already the shipped default** — this is a
much better starting position than it would have been three weeks ago.

---

## 0.1 Stage-0 measurements — done 2026-08-25, and they moved the plan

`bench/gpu_probe.jl` (standalone, no package, no data — send it to any GPU host).
CUDA.jl **is functional** on Julia 1.12.7 / driver 535.309 (CUDA 12.2) / sm_61,
running a 12.9 runtime under CUDA's minor-version compatibility. 7.04 GiB free.

| | measured | of peak |
|---|---|---|
| FP32 FMA kernel | **7323 GFLOP/s** | 82% of 8878 (at the reported 1.734 GHz) |
| device copy | **237 GB/s** | 74% of 320 |

**cuFFT batched C2R at the search's own fold depths** — the load-bearing number:

| `Nprof` | k=1 (120) | k=2 (60) | k=3 (40) | k=4 (30) | k=5 (24) | k=6 (20) |
|---|---|---|---|---|---|---|
| 65,536 | 0.1090 | 0.1158 | 0.1186 | 0.1207 | 0.1227 | 0.1380 |
| 262,144 | 0.1065 | 0.1092 | 0.1138 | 0.1158 | 0.1163 | 0.1270 |

(ns per output bin; **15–19× one Xeon core's FFTW**, which `decim_brfft_bench`
puts at 2.048 ns/bin. cuFFT is *not* the small-transform liability I assumed —
it barely degrades from k=1 to k=6, and `Nprof = 262144` is uniformly better.)

**Three things this changed:**

1. **The stage-0 gate as originally written was against the wrong baseline.** It
   said "if cuFFT is worse than ~2× FFTW's per-bin cost, promote the megakernel".
   cuFFT is 19× *better* than FFTW-per-core, so the gate passes — but the number
   that matters is not one core, it is **the whole 20-core socket we already
   beat riptide with**. Against that, cuFFT's 0.11 ns/bin is only ~1.9×. The
   comparison must always be GPU vs `-t 20`, never GPU vs `-t 1`.
2. **The direct-DFT megakernel (§3.3, stage 3) is dead as a cuFFT replacement.**
   Prototyped, tuned with 4 independent accumulators to break the FMA dependency
   chain, and verified correct to **2.4e-7** (Float32 machine precision, and a
   useful early confirmation that the §5 tolerances are the right ones). It runs
   at **0.41 ns/bin — 3.8× slower than cuFFT**, at both `Nprof`. And it is *not*
   FLOP-bound: 122 flops per output bin at 0.41 ns is 300 GFLOP/s against the
   7323 available, i.e. 4% — it is shared-memory/latency bound. Closing a 3.8×
   gap from there is speculative, and it would have to be closed *and* beat cuFFT
   before the fusion saving (~0.12 s of traffic) is even reachable. **Do not
   re-open this without a genuinely different kernel; the probe re-measures it in
   one command on any new card.**
3. **The realistic projection is lower than §2 first claimed.** See below.

---

## 0.2 Running the probe on a new card

`bench/gpu_probe_setup.sh` bootstraps Julia + CUDA.jl on a bare GPU host and runs
the probe. **No root, no system CUDA toolkit, nothing from this repo but the two
files** — CUDA.jl ships its own toolkit as artifacts, so only an NVIDIA *driver*
matters and a module-loaded CUDA is irrelevant (and will not be used).

```sh
scp bench/gpu_probe{.jl,_setup.sh} newhost:            # the only two files needed
ssh newhost './gpu_probe_setup.sh'
PREFIX=/scratch/$USER/gpuprobe ./gpu_probe_setup.sh    # somewhere other than $HOME
JULIA_DEPOT_PATH=/fast/local/depot ./gpu_probe_setup.sh
```

Two things that are the usual way this goes wrong on a cluster:

- **The CUDA artifacts are ~2.2 GB and land in the Julia depot.** If `$HOME` is
  quota'd or on slow NFS, set `JULIA_DEPOT_PATH` to local scratch.
- **Compute nodes are often air-gapped.** Install once on a login node sharing the
  filesystem, then run the probe on the GPU node with the same `PREFIX` and
  `JULIA_DEPOT_PATH`. The script uses whatever `julia` is already on `PATH` and
  otherwise fetches a private tarball; it deliberately does **not** use `juliaup`,
  which wants to own a shell profile.

The probe now derives each card's *theoretical* peak from its compute capability,
SM count and bus width, so every measurement is reported as a percentage and the
numbers are comparable across cards without a spec lookup. It ends with a
one-line classification to paste back.

### What to expect from the card, and the number to look at

**The diagnostic that matters is the new "% of achievable bandwidth" column on
the cuFFT table.** On the GTX 1080 cuFFT runs at **27–33%** of the bandwidth the
card actually delivers — so at these transform sizes it is **latency- and
occupancy-bound, not bandwidth-bound**. The practical consequence:

- **SM count and scheduling matter more than GB/s for this workload.** Every
  RTX 4000-class card has ≥36 SMs against the 1080's 20, so the transform stage —
  which §2 shows is ~65% of GPU time and the thing capping the whole port — should
  improve even on a card whose bandwidth barely moves.
- **"RTX 4000" is three different cards**, and the difference is large: a *Quadro*
  RTX 4000 (Turing, sm_75, 8 GB, ~7 TFLOP/s) is barely above the 1080; an
  *RTX A4000* (Ampere, sm_86, 16 GB, ~19 TFLOP/s) is ~2.5×; an *RTX 4000 Ada*
  (sm_89, 20 GB, ~27 TFLOP/s) more again — but with *less* bandwidth than the
  A4000, which would matter if cuFFT were bandwidth-bound and (per the above)
  should not be. **Don't infer from the name — run the probe and read the
  percentages.** That is exactly what it was extended to settle.
- **Memory size changes §3.2's `Nprof` sizing.** 16–20 GB comfortably holds a much
  larger chunk than the 1080's 7 GB, and an A100's 40–80 GB would swallow the
  1.4 GB `NGC6624` file whole with room for a `262144`+ workspace.

**Re-run `dft_vs_cufft` on every new card rather than trusting §0.1's verdict.**
The DFT loses 3.8× on Pascal because it is shared-memory/latency bound; a card
with a different shared-memory-to-FLOP balance could rank them differently, and
this repo's standing lesson is that two hosts invert each other's conclusions.

---

## 0.3 Second card: RTX 4000 SFF Ada — and the `Nprof` guidance INVERTS

Measured 2026-08-25 by Scott on a two-card host (the probe uses device 0 only;
nothing below involves both).

| | GTX 1080 (sm_61) | RTX 4000 SFF Ada (sm_89) | ratio |
|---|---|---|---|
| SMs / clock | 20 @ 1.734 GHz | **48 @ 1.560 GHz** | 2.16x |
| FP32 achievable | 7323 GFLOP/s (82%) | **16813 (88%)** | **2.30x** |
| bandwidth achievable | 237 GB/s (74%) | **238 GB/s (85%)** | **1.00x** |
| memory | 7.0 GiB | **19.5 GiB** | 2.8x |
| bus | 256-bit | **160-bit** | — |
| L2 | **2.0 MB** | large (see below) | — |

**This card is compute-rich and bandwidth-starved**: 2.3x the FLOPs of a 2016
consumer card and *exactly the same* memory bandwidth, because the SFF part has a
160-bit bus. Anything bandwidth-bound will not improve on it at all; anything
compute- or latency-bound will improve by ~2.2x. That is a sharper split than the
two CPU hosts ever showed, and it is the lens for every number below.

### The impossible column was the finding

The probe's "% of achievable bandwidth" read **184%, 147%, 190%, 185%** for
`k = 3…6` at `Nprof = 65536` — physically impossible for a streaming transform,
and therefore informative. The working sets bracket it exactly:

| `Nprof` = 65536 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 |
|---|---|---|---|---|---|---|
| src+dst | 60.5 MB | 30.5 MB | **20.5 MB** | 15.5 MB | 12.5 MB | 10.5 MB |
| % of DRAM copy | 53% | 84% | **184%** | 147% | 190% | 185% |

Everything at or below ~20 MB **never touches DRAM** — it is L2-resident, so the
denominator (a 256 MB device copy) is the wrong yardstick. The knee sits between
20.5 and 30.5 MB. The GTX 1080 is the control: **2.0 MB of L2, no knee anywhere,
and no row above 100%.**

**The probe now prints L2 size and sweeps `Nprof`** (`bench/gpu_probe.jl`), and
the re-run **confirms L2 = 40.0 MB** and the predicted optimum:

| whole six-rung transform stage | 16384 | **32768** | 65536 | 131072 | 262144 |
|---|---|---|---|---|---|
| RTX 4000 SFF Ada | 0.071 s | **0.071 s** | 0.101 s | 0.133 s | 0.161 s |
| GTX 1080 | 0.316 s | 0.299 s | 0.286 s | 0.280 s | **0.275 s** |

**`Nprof = 32768` gives 0.071 s — 7.2x the CPU socket's ~0.51 s**, against
`262144`'s 0.161 s. The two cards want opposite ends of the sweep, and the
spread within a single card is **2.3x**, so **`Nprof` is a per-device tuning
parameter and §3.2's "~10^5 trials" is a 1080 result.** The probe now derives
this number directly, so a new card answers it in one command.

**Read the small-`Nprof` rows carefully — `%DRAM` there measures occupancy, not
cache.** At `Nprof = 16384` the `k = 4,5,6` rungs sit at 85–93% despite working
sets of only 2.6–3.9 MB, which cannot be a cache effect: 16384 batches of a
20-bin transform is too little parallelism to fill 48 SMs. So the column means
"L2-resident" only when it is *above* 100%; below it, it conflates cache misses
with underutilisation. The optimum balances the two, which is exactly why it
needs measuring rather than deriving.

### The rungs want DIFFERENT `Nprof`, and that is worth another 1.40x

The per-rung optima are spread across the whole sweep:

| | k=1 (120) | k=2 (60) | k=3 (40) | k=4 (30) | k=5 (24) | k=6 (20) |
|---|---|---|---|---|---|---|
| best ns/bin | 0.02155 | 0.02246 | 0.01853 | 0.02349 | **0.01509** | **0.01484** |
| at `Nprof` | 16384 | 32768 | 65536 | 65536 | **131072** | **131072** |
| at the single optimum 32768 | 0.03493 | 0.02246 | 0.02277 | 0.02889 | 0.02486 | 0.02656 |

The deep fold wants a *small* batch (its 968 B per trial fills L2 fastest) and
the shallow folds want a *large* one (their 168 B per trial needs many trials
before there is enough work). One `Nprof` cannot satisfy both.

**But it does not have to.** The six rungs read the same `ftprofs`, yet nothing
forces them to be transformed in one call: each is a batched transform over
*columns*, so a rung can be run in **column sub-batches sized to its own L2
footprint** while the chunk stays whatever the interpolator wants. Taking each
rung at its own optimum gives **6.038 ns per trial against 8.445 — 1.40x — i.e.
0.051 s, or 10.1x the CPU socket.**

**This decouples the two competing pressures on `Nprof`**: the interpolator and
the launch-overhead argument want a large chunk, the transforms want L2-sized
batches. Sub-batching lets each have what it wants. It is a stage-2 design item,
not a stage-3 optimisation, because it changes how the pipeline is structured
rather than how a kernel is written.

### What it does to the projection — and the phase ranking INVERTS

| phase | GTX 1080 | RTX 4000 SFF Ada | why |
|---|---|---|---|
| transforms | 0.275 s | **0.071 s** (0.051 sub-batched) | 40 MB L2 |
| interpolation | 0.135 s | ~0.063 s | issue/latency-bound, scales with SMs x clock (2.16x) |
| metric | ~0.080 s | ~0.080 s | bandwidth-bound, and this card has **no more bandwidth** |
| **total** | ~0.49 s | **~0.21 s** (0.19 sub-batched) | |
| **vs CPU `-t 20` (1.01 s)** | ~2.1x | **~4.7x** (5.2x sub-batched) | |

**§2's "expect ~65% of GPU time in cuFFT" is a 1080 statement. On the Ada the
transform is 33% — or 26% sub-batched — and interpolation becomes the largest
single phase.** The two cards disagree about what to optimise, exactly as
`foops` and `fitzroy` have done throughout this project. Two consequences:

- **The interp register-reuse idea (§4.1) is now the highest-value kernel work**,
  not a footnote: 1.73x on ~30% of GPU time. It was worth ~0.9x of a phase on the
  1080 and is worth ~0.024 s here.
- **Nothing bandwidth-bound will improve on this card.** The metric is the phase
  to watch, and it is the one stage 2 has not written yet.

**And the host has two cards**, which in §7.2's throughput mode is **~9-10x the
20-core Xeon aggregate** — from a pair of small workstation GPUs.

### Pre-registered prediction for the RTX 2080 Super (sm_75)

Recorded **before** the run, so it is a test rather than a post-hoc story. The
2080 Super is the opposite corner from the Ada **at matched SM count**, which
isolates what the 40 MB L2 actually buys:

| | GTX 1080 | RTX 2080 Super | RTX 4000 SFF Ada |
|---|---|---|---|
| arch | Pascal sm_61 | **Turing sm_75** | Ada sm_89 |
| SMs x cores/SM | 20 x 128 | **48 x 64** | 48 x 128 |
| FP32 peak | 8.9 TF | ~11.2 TF | 19.2 TF |
| bandwidth peak | 320 GB/s | **~496 GB/s** | 280 GB/s |
| L2 | 2 MB | **~4 MB** | **40 MB** |

It is **bandwidth-rich and L2-poor** — roughly 2x the DRAM bandwidth of *either*
other card and a fortieth of the Ada's cache. So:

1. **Almost no row should read above 100%.** With ~4 MB of L2, only the very
   smallest working sets can be cache-resident. If rows go L2-resident anyway,
   the §0.3 reading is wrong.
2. **`Nprof` should prefer the LARGE end**, like the 1080 and unlike the Ada,
   because nothing fits in cache and only occupancy is left to optimise.
3. **The transform stage should land ~0.12–0.16 s** — roughly half the 1080's
   0.275 s, tracking its ~2x bandwidth, and roughly 2x the Ada's 0.071 s.

**What each outcome would mean.** If it comes in near 0.14 s, the story holds:
the transform stage is bandwidth-bound *unless* a large cache lifts it out, and
the sub-batching win of §0.3 is specifically a big-L2 phenomenon. **If it comes
in near the Ada's 0.071 s, the cache story is wrong** — raw bandwidth would be
substituting for residency, which would make cheap high-bandwidth cards the right
target and would demote the per-rung sub-batching item. Either answer is worth
the ten minutes.

A secondary reading: Turing has **64** FP32 cores per SM against Ada's 128, so if
the interpolation kernel really is issue/latency-bound (§4.1) rather than
FLOP-bound, it should scale with *SM count* and clock — i.e. land much closer to
the Ada than the 2.3x FP32 ratio between them would suggest.

### 0.4 RTX 2080 Super: prediction scored — 2 of 3, and the miss is the useful part

| | GTX 1080 | **RTX 2080 Super** | RTX 4000 SFF Ada |
|---|---|---|---|
| arch / SMs x cores | Pascal, 20 x 128 | **Turing, 48 x 64** | Ada, 48 x 128 |
| SMs x clock | 34.7 | **87.1** | 74.9 |
| FP32 achieved | 7323 (82%) | **9438 (85%)** | 16807 (88%) |
| bandwidth achieved | 237 (74%) | **431 GB/s (87%)** | 239 (85%) |
| L2 | 2 MB | **4 MB** | 40 MB |
| transform stage | 0.275 s @262144 | **0.100 s @262144** | 0.071 s @32768 |

**Confirmed:** (1) no row above 100% — the maximum is 48%, nothing is ever
L2-resident. (2) `Nprof` prefers the **large** end, monotonically
(0.123 → 0.113 → 0.106 → 0.102 → **0.100**), exactly like the 1080 and opposite
to the Ada.

**Missed:** the stage landed at **0.100 s against a predicted 0.12–0.16 s** —
1.41x better than the bandwidth-scaling argument said. The prediction assumed the
1080's 0.275 s would scale by bandwidth alone (431/237 = 1.82x → 0.151 s). It
scaled by **2.75x**, because the 2080 Super also has **2.4x the SMs**.

**That miss identifies the mechanism, and the three cards now pin it down.**
cuFFT's efficiency, as a fraction of each card's own measured DRAM copy, at the
DRAM-bound end of the sweep:

| | GTX 1080 (20 SMs) | RTX 2080 Super (48) | RTX 4000 Ada (48) |
|---|---|---|---|
| cuFFT % of own DRAM copy | ~33% | **~46%** | **~51%** |

**The two 48-SM cards reach nearly the same fraction, and the 20-SM card reaches
far less** — with those cards differing in bandwidth by 1.8x and in L2 by 10x.
So the DRAM-bound transform efficiency is set by **SM count** (concurrent memory
requests hiding latency), not by bandwidth or cache; bandwidth then sets the
scale of what that fraction is a fraction *of*. This is the same conclusion
§4.1's probe reached for the interpolation kernel by a completely different
route: **at these transform sizes the GPU is latency- and occupancy-bound, and
raw FLOPs are nearly irrelevant.**

**The pre-registered discriminator resolved cleanly in favour of the cache
story.** Per-rung sub-batching (§0.3) is worth **1.40x on the Ada and exactly
1.00x on the 2080 Super** — every 2080 Super rung has its optimum at the same
`Nprof = 262144`, because with 4 MB of L2 there is no residency to arrange. So
sub-batching is confirmed as **specifically a big-L2 optimisation**, not a
general one, and it should be implemented as a per-device policy rather than
unconditionally.

### The two modern cards are a TIE, and it is not close to the FLOPs ratio

Combining the measured transform stage with projections for the other two phases
— interpolation scaled by SMs x clock (§4.1 says issue/latency-bound) and the
metric scaled by bandwidth (bandwidth-bound):

| | transforms | interp* | metric* | total* | vs CPU `-t 20` |
|---|---|---|---|---|---|
| GTX 1080 | 0.275 | **0.136** | 0.080 | 0.491 s | 2.1x |
| **RTX 2080 Super** | 0.100 | **0.056** | 0.044 | **0.200 s** | **5.1x** |
| **RTX 4000 SFF Ada** | **0.071** | 0.063 | 0.079 | 0.213 s | 4.7x |

`*` The GTX 1080 and RTX 2080 Super interp columns are now **measured**
(§0.45); the Ada's is projected and the metric column is projected throughout.

**A 2019 consumer card ties a current workstation card on this workload, despite
having 56% of its FP32.** They win different phases: the Ada takes the transforms
on its 40 MB L2, the 2080 Super takes interpolation (2.4x the SMs x clock at 1.16x
the Ada's) and the metric (1.8x the bandwidth). Sub-batching would put the Ada
back ahead (0.193 s, 5.2x), and the Ada's 20 GB against 8 GB matters for large
files — but on raw throughput per card they are level.

**The hardware lesson for the paper, and for buying: this workload wants SMs and
bandwidth, not FLOPs.** FP32 peak across these three spans 8.9 → 11.2 → 19.2
TFLOP/s while the projected end-to-end time spans 0.49 → 0.198 → 0.213 s — i.e.
the *fastest* FP32 part is not the fastest card, and the ranking follows
SMs x clock and GB/s instead. §0.1 already measured the interpolator at 4.8% of
peak FLOPs; this is the same fact seen from the hardware side.

**Done, and the second card broke the model — see §0.45 and §0.46.** The 2080
Super fit SMs x clock to 3.5%; the Ada missed it by 26% in the *other* direction,
so the A4000 is now a two-way discriminator between cores/SM and L2 residency.

### 0.45 Interpolation measured on a second card — §4.1's verdict CONFIRMED

`bench/gpu_interp_bench.jl` on the RTX 2080 Super (real PM0063 `.fft`), against
the GTX 1080, ns per (harmonic, trial) at `Nprof = 65536`:

| | GTX 1080 | RTX 2080 Super | speedup |
|---|---|---|---|
| **measured** | 0.2695 | **0.1112** | **2.42x** |
| SMs x clock predicts | 34.7 | 87.1 | **2.51x** — off by **3.5%** |
| FP32 predicts | 7323 | 9438 | 1.29x — off by **88%** |

**That is as clean a confirmation as this project has produced.** §4.1 concluded
from a single-card probe that the interpolation kernel is issue- and
latency-bound rather than FLOP-bound; a completely independent cross-card test
now agrees to 3.5%, while the FLOP-based model is wrong by a factor of two. The
interp stage for the reference workload is **0.0558 s measured against 0.054 s
projected — 3.3%.**

Combined with §0.4's finding that DRAM-bound *transform* efficiency also tracks
SM count (33% at 20 SMs, 46–51% at 48), **both of the pipeline's compute phases
scale with SMs x clock and neither scales with FP32.** That is now a measured
property of the workload on two architectures, not an inference.

It also sharpens §0.5's test: the Ada should do interpolation in **0.1248 ns
(0.0626 s)** — identical SMs x clock to the A4000 — so **the 2080 Super should be
1.12x faster than either**, and **the Ada and the A4000 should tie to within a
few percent despite their very different bandwidth and cache.**

**`test/test_gpu.jl` is 117/117 on sm_75**, so the kernel — including the
bit-exact batch invariance — is now pinned on two architectures.

**A trap this run exposes: the "x one core" column is host-specific and must not
be carried across machines.** The 2080 Super host's CPU is much faster than
fitzroy's Xeon Silver 4114 — 3.70 vs 4.85 ns at `Nprof = 2048` (1.31x) and 4.16
vs 9.88 at 65536 (**2.38x**, its cache holding the larger plane buffers where the
Xeon's does not). So the headline "33.2x one core" there and "18.5x one core" on
fitzroy describe *different denominators*, and neither may be multiplied by 20 to
get a socket comparison for a machine it was not measured on. **Every end-to-end
ratio in this document is against fitzroy's 20-core Xeon at 1.01 s**; keep the
GPU's absolute ns and compare that.

### 0.46 Third card breaks the model — the prediction was WRONG

The Ada measured **0.0992 ns** per (harmonic, trial) against a pre-registered
**0.1248**, and it is **1.12x faster than the 2080 Super** where §0.45 predicted
the 2080 Super would be 1.12x faster. **Wrong magnitude and wrong direction.**

| | measured ns | SMs x clock | FP32 | cores/SM | L2 |
|---|---|---|---|---|---|
| GTX 1080 | 0.2695 | 34.7 | 7323 | 128 | 2 MB |
| RTX 2080 Super | 0.1112 | 87.1 | 9438 | **64** | 4 MB |
| RTX 4000 SFF Ada | **0.0992** | 74.9 | 16807 | **128** | **40 MB** |

Speedup over the GTX 1080, against the two candidate models:

| | measured | SMs x clock | FP32 |
|---|---|---|---|
| RTX 2080 Super | 2.42x | 2.51x (**3.5%**) | 1.29x (88%) |
| RTX 4000 SFF Ada | **2.72x** | 2.16x (**25.8%**) | 2.30x (18%) |

**Neither model fits all three cards.** SMs x clock nails the 2080 Super and
misses the Ada by 26%; FP32 is hopeless on the 2080 Super and merely bad on the
Ada. §0.45 called SMs x clock "a measured property of the workload on two
architectures" — **two architectures were not enough, and that claim is now
retired.** The honest statement is narrower: *FP32 peak is definitively not the
right predictor* (88% error on the 2080 Super), and SMs x clock is a good first
approximation that under-predicts the Ada.

**Two candidate explanations, and they are cleanly separable.** The Ada beats
SMs x clock in the direction of the two things it has that the 2080 Super does
not:

1. **128 FP32 cores per SM against Turing's 64.** §4.1 found the kernel
   issue-bound, and issue width per SM is exactly what cores/SM buys — so this
   would mean the truth is between the two models rather than either one.
2. **40 MB of L2 against 4 MB.** The kernel is *load-latency* bound (§4.1's probe:
   volume irrelevant, the load's existence worth 1.73x), and at `Nprof = 65536`
   the interpolator's amplitude windows total ~7.6 MB across 60 harmonics — which
   is L2-resident on the Ada and not on either other card. Lower load latency is
   precisely what would show up here.

**The A4000 now settles it, and it is a clean two-way discriminator.** It has
**identical SMs (48), clock (1.56 GHz) and cores/SM (128)** to the Ada, and
**4 MB of L2** like the 2080 Super:

- **A4000 ~= 0.0992 ns (ties the Ada)** -> explanation 1: it is cores/SM, and L2
  is irrelevant to interpolation.
- **A4000 ~= 0.1248 ns (the SMs x clock line)** -> explanation 2: it is the 40 MB
  L2, and cache residency matters to the interpolator as well as to cuFFT.

This is a better experiment than the one §0.5 was designed for, and it costs the
same single run.

### The two modern cards are a DEAD TIE

With the Ada's interp column now measured:

| | transforms | interp | metric* | total | vs fitzroy `-t 20` |
|---|---|---|---|---|---|
| GTX 1080 | 0.275 | 0.1352 | 0.080 | 0.490 s | 2.06x |
| **RTX 2080 Super** | 0.100 | 0.0558 | 0.044 | **0.1998 s** | **5.06x** |
| **RTX 4000 SFF Ada** | 0.071 | **0.0498** | 0.079 | **0.1998 s** | **5.06x** |

`*` metric is the one remaining projection; it is the phase stage 2 has not
written.

**SUPERSEDED 2026-08-25 — the metric column is now measured on all three cards
and the bandwidth model behind it is WRONG. See §4.8.** In-search, scaled to this
same reference workload: metric **0.1478 / 0.0655 / 0.0797 s**. The tie survives
(measured totals 0.354 s and 0.341 s, the Ada 1.04x ahead against a predicted
0.00x), but the *level* is 1.7-1.8x optimistic on every row, and the mechanism
splits cleanly: ~26-38% of it is phases this table never had (transpose, zero,
download, host scan), the rest is the Ada's isolated L2 transform win not
surviving the pipeline. **The Ada's 0.079 metric figure is right for the wrong
reason** — the bandwidth model predicted a 1.00x speedup over the 1080 while the
baseline it scaled was itself 1.85x too low, and the two errors cancelled.

**0.1998 s each, to four digits** — a coincidence, but a telling one. They get
there completely differently: the Ada wins transforms (40 MB L2) *and* now
interpolation, while the 2080 Super wins the metric on 1.8x the bandwidth. A 2019
consumer card and a current workstation card tie on this workload while differing
by 1.78x in FP32, 1.8x in bandwidth and 10x in L2. **That is the hardware lesson
in its strongest form: above ~48 SMs, this workload does not care much what you
buy.**

**A third host, a third CPU — the "x one core" column still must not travel.**
`hypatia` reads 2.59 ns at `Nprof = 2048` against `spare2`'s 3.70 and fitzroy's
4.85, so the same GPU number reads as 26.1x, 33.2x or 18.5x "one core" depending
only on which machine it sat in. Every end-to-end ratio in this document is
against **fitzroy's** 20-core Xeon at 1.01 s.

### 0.5 Pre-registered prediction for the RTX A4000 (sm_86)

**The A4000 against the RTX 4000 SFF Ada is very nearly a controlled experiment,
and that is rare enough to be worth exploiting.** GA104 and AD104 in these two
parts have the *same* SM count (48), the *same* cores/SM (128), the *same* boost
clock (1.56 GHz) and therefore the **same FP32 peak — 19169 GFLOP/s, to four
digits**. They differ in exactly two things:

| | RTX 4000 SFF Ada | **RTX A4000** |
|---|---|---|
| FP32 peak | 19169 GFLOP/s | **19169 GFLOP/s** (identical) |
| SMs x clock | 74.9 | **74.9** (identical) |
| L2 | **40 MB** | **4 MB** |
| bandwidth peak | **280 GB/s** | **448 GB/s** |
| memory | 20 GB | 16 GB |

So it isolates **big cache versus more bandwidth at fixed compute** — the one
question the first three cards left entangled, since they varied SM count,
bandwidth and L2 all at once.

**Predictions, recorded before the run:**

1. **Achieved bandwidth ~385 GB/s** (86% of peak; the three measured cards came
   in at 74%, 85%, 87%).
2. **No row above 100%** and **`Nprof` preferring the large end**, exactly like
   the 2080 Super — 4 MB of L2 cannot hold any rung's working set.
3. **Per-rung sub-batching worth 1.00x**, as on the 2080 Super. If it is worth
   more than ~1.05x with only 4 MB of L2, §0.3's mechanism is wrong.
4. **Transform stage ~0.112 s**, from the 2080 Super's measured 0.100 s scaled by
   bandwidth at equal SM count (431/385).
5. **Interpolation should match the Ada's, not beat it** — identical SMs x clock —
   which combined with §0.4's test makes a three-way discriminator: if interp is
   issue/latency-bound, A4000 ≈ Ada < 2080 Super; if FLOP-bound, A4000 ≈ Ada >
   2080 Super by 1.78x.

**What outcome 4 decides.** The Ada does the transform stage in **0.071 s** on
40 MB of L2 with 239 GB/s. If the A4000 needs ~0.112 s with 1.6x that bandwidth
and a tenth of the cache, then **for this stage a large L2 is worth more than
1.6x the DRAM bandwidth**, and the per-rung sub-batching item (§0.3) is the right
thing to build. If the A4000 comes in at or below 0.071 s, bandwidth substitutes
for residency after all and sub-batching should be dropped.

**Projected end-to-end, all four** (transform stage measured except the A4000;
interp and metric projected throughout):

| | transforms | interp* | metric* | total* | vs CPU `-t 20` |
|---|---|---|---|---|---|
| GTX 1080 | 0.275 | 0.136 | 0.080 | 0.491 s | 2.1x |
| RTX 2080 Super | 0.100 | **0.056** | 0.044 | **0.200 s** | **5.1x** |
| RTX 4000 SFF Ada | 0.071 | 0.063 | 0.079 | 0.213 s | 4.7x |
| RTX A4000 (predicted) | 0.112 | 0.063 | 0.049 | 0.224 s | 4.5x |

**The three modern cards are predicted to land within 13% of each other while
spanning 11.2–19.2 TFLOP/s of FP32 and 280–496 GB/s of bandwidth.** That is the
sharpest form of §0.4's hardware lesson, and if it holds it is a genuinely useful
statement for the paper: **on this workload the card barely matters above a
threshold of ~48 SMs — buy on SM count and price, not on FLOPs.**

### Two verdicts that survived, and one caution

- **The direct-DFT is beaten on all three cards, and by MORE on the newer ones**:
  3.78x (GTX 1080), 3.09x/2.87x (RTX 4000 Ada), and **7.36x/7.69x (RTX 2080
  Super)** — the widest margin yet, since the DFT gains nothing from bandwidth it
  cannot use while cuFFT does. Accuracy held at 2.4e-7 to 2.7e-7 on every card.
  **§0.1's verdict is now tested across three microarchitectures and is closed.**
- **19.5 GiB free** removes the §3.2 memory worry entirely: the 1.4 GB
  `NGC6624` file fits with room for any workspace.
- **NFS does not affect any number here** — every timing is device-side, on data
  already resident. Where it *does* bite:
  - **Start-up.** The depot must live on NFS on these machines; `/tmp` is the only
    fast local disk and is not durable. This costs package load, not compute, and
    §7.2's throughput mode amortises it across hundreds of files, so it is
    acceptable. If it ever stops being acceptable, the answer is the existing
    `sysimage/` machinery staged to `/tmp` — a sysimage is a single file, so it
    is exactly the thing a scratch disk is good for.
  - **Reading `.fft` files, which is the real risk.** At ~5x, a 32 MB file
    searches in ~0.2 s — the same order as reading 32 MB over NFS. **In
    throughput mode the limit becomes I/O, not the GPU.** Scott can stage `.fft`
    files on local disk, which settles it; where that is not possible, §7.2's
    prefetch/overlap has to hide *disk* latency rather than just PCIe, and
    `FFTFile`'s mmap is worth measuring against a bulk read — mmap page-faults
    over the network one fault at a time, which is the worst possible access
    pattern for NFS.

---

## 1. Why this search is a good GPU fit — and the one place it is not

**Good:**

1. **The output is tiny.** A whole bench-config search over PM0063 is 8,363,442
   trial fundamentals × 6 ladder rungs, and returns *a few hundred* candidates.
   Everything between the amplitudes and the threshold test is reducible on the
   device. There is no PCIe bottleneck to design around — upload the `.fft` once,
   download a candidate list.
2. **The parallelism is embarrassing and already explicit.** Trials are
   independent; the CPU already exploits that at chunk granularity with private
   workspaces and no cross-chunk state.
3. **The trials-axis interpolation kernel is *already a warp kernel*.**
   `DIRECT_GROUP_V = 32`. The group table `gW` is laid out `(V, m+Δ)`
   column-major so that lane `k` is trial `k` and `gW[:, j]` is contiguous, with
   `re[b0+j]` broadcast. That is *precisely* a CUDA warp with lane = trial,
   coalesced weight loads and a broadcast operand — **no gather, no shuffle
   reduction.** The reformulation done for AVX2 (a159706) ports to CUDA almost
   verbatim, and its exact integer/rational phase bookkeeping
   (`direct_chunk_state`, `grow`/`goff`/`gnj`) is anchored to the *global* trial
   index, so it is already batch-size invariant — which is what lets the GPU use a
   completely different chunk size without changing a single result.
4. **The plan tables are small and read-only.** 1.45 MB of `gW`/`gA` over 60
   harmonics — upload once per search, perfect for the read-only/texture path,
   shared by every block.
5. **Analytic σ removed the only sequential statistic.** As of 2026-08-24 the
   noise scale is a closed form over `ws.filled`, not a MAD over a subsample. The
   selection network / quickselect that would have been awkward on a GPU is
   **gone from the default path**. (`--sigma measured` would need a device-side
   median; treat it as a stage-4 item, not a stage-1 one.)

**Not good:**

- **The exact-`Float64` rescan.** `boxcar_metrics!` re-scores in `Float64` every
  trial that lands within `boxcar_gatemargin` of `threshold`. At 1/32 rate that is
  the one piece that must not run on this GPU. It does not need to: survivors are
  ~0.1% of trials, so **compact them on the device and rescan on the host**, which
  is both faster and keeps the rescan bit-identical to today's.
- **The transforms are small.** 120-bin C2R at k=1, down to 20 bins at k=6.
  cuFFT is fine at these sizes given a large enough batch, which is an argument
  for a large GPU chunk (§3).

---

## 2. Baseline and target

The reference workload is the one every number in CLAUDE.md is quoted against:
PM0063 at the riptide `bench` config, 0.1–33.333 Hz fundamentals, `--nharms 60
--maxdecim 6`, 8,363,442 trials per rung.

| | fitzroy wall clock |
|---|---|
| `rseek` (riptide FFA) | 19.81 s |
| ours `-t 1`, `:f32` + AVX2 | **11.92 s** |
| ours `-t 20`, `:f32` + AVX2 | **1.01 s** |

Rough arithmetic budget for that workload, to set expectations honestly:
~18 kFLOP per trial fundamental across all six rungs (interp ≈ 5.8k, six C2R
transforms ≈ 8.7k, boxcar bank ≈ 3.8k) → **~150 GFLOP total**. The 20-core Xeon
therefore achieves ~150 GFLOP/s, about 11% of its FP32 peak.

The GTX 1080 has **6.9× the Xeon's FP32 peak**. A staged (unfused) port also
moves ~28 GB of intermediate arrays, i.e. ~0.09 s at 320 GB/s, so it lands
roughly balanced between compute and bandwidth.

**Revised with the §0.1 measurements — this supersedes the estimate above.**
The transform stage is 294 output bins per trial across the six rungs, so
`2.46e9` output bins for the bench config; at the measured ~0.112 ns/bin that is
**0.275 s, and it is not reducible** (cuFFT is already 19× a core, and the DFT
alternative is 3.8× worse). The CPU spends ~51% of its `-t 20` second on the same
transforms, i.e. ~0.51 s, so **the transform stage alone is only ~1.9×**.
Interpolation (~48 GFLOP) should land at 0.03–0.05 s against the CPU's ~0.15 s,
and the metric at 0.05–0.10 s against ~0.31 s.

- **Stage-1/2 target: ~0.35–0.45 s, i.e. 2.2–2.9× the full 20-core Xeon**
  (~26–34× one core), and **transform-dominated** — expect ~65% of GPU time in
  cuFFT.
- **Fusion (stage 3) buys traffic, not transforms**: eliminating the `ftprofs`
  and `profs` round-trips is ~0.12 s, so ~1.4× on top, not the 2× I first
  guessed. Worth doing, not worth blocking on.
- **Failure signal: worse than ~1 s.** That means launch overhead dominates and
  §3.2's chunk sizing is wrong.

**So the headline is not raw speed over the CPU — it is 2–3× a dual-socket
20-core server from a 2016 consumer card, at ~180 W against 2×85 W and roughly
1/20 the purchase price.** And there is no architectural wall in the way: we
reach 82% of achievable FLOPs and cuFFT scales with the card, so an A100
(19.5 TFLOP/s, 1555 GB/s) or L40S should extend this nearly linearly. That is
what makes §7.1 the first question to answer.

**The honest headline for a paper** is *a 2016 consumer GPU against a 20-core
dual-socket server*, quoted with the same work accounting
`compare/compare_riptide.py` already prints. Do not quote GPU-vs-one-core.

---

## 3. Architecture

### 3.1 Where the CPU/GPU line goes

Everything from the amplitudes to the threshold test goes on the device:

```
host                          device
────                          ──────
FFTFile mmap  ──upload──▶  amps (ComplexF32, band-limited window)
DirectPlan[]  ──upload──▶  grow/goff/gnj/gW/gA          (1.45 MB, read-only)
                           │
                           ├─ K1  interp        → ftprofs (Nprof, nharms+1)
                           ├─ K2  cuFFT C2R ×6  → profs_k (Nprof, 2Hk)
                           ├─ K3  boxcar gate   → mvals  (Nprof, 6)
                           └─ K4  compact       → (trial, k, mval) above exactcut
                           │
     survivors ◀──download─┘   (~0.1% of trials)
Float64 rescan, candidate collapse, output  ── unchanged CPU code
```

The candidate loop, `remove_duplicates`, `remove_harmonics`, `MetricNorm`,
plotting and all I/O stay on the host and stay exactly as they are.

### 3.2 The GPU chunk is ~10⁵ trials, not 2048

This is the single most important sizing decision, and getting it wrong is the
most likely way stage 1 fails.

At `Nprof = 2048` the bench config is 4084 chunks. At ~10 kernel launches per
chunk that is ~40,000 launches; at 5 µs each, **0.2 s of pure launch overhead** —
comparable to the entire target runtime. It also leaves cuFFT batching far too
small.

Measured footprints (PM0063, `nharms = 60`, `maxdecim = 6`):

| `Nprof` | ftprofs | profs | dprofs (k=2…6) | harmonic windows | chunks |
|---|---|---|---|---|---|
| 2,048 | 1.0 MB | 0.9 MB | 1.4 MB | 0.3 MB | 4,084 |
| **65,536** | **30.5 MB** | **30.0 MB** | **43.5 MB** | **7.6 MB** | **128** |
| 262,144 | 122 MB | 120 MB | 174 MB | 30.5 MB | 32 |

`Nprof = 65536` is the natural starting point: ~112 MB of workspace against a
7.2 GB budget, 128 chunks, ~1300 launches. `262144` is also affordable and should
be swept.

`direct_window_size` scales as `(Nprof-1+2V)·hidr`, so at `Nprof = 65536` the top
harmonic spans ~32 K bins — 256 KB — and **all 60 harmonic windows together are
7.6 MB**, which means the GPU can hold every harmonic's window resident rather
than de-interleaving one at a time as the CPU does. Or skip windowing entirely and
index `amps` directly: PM0063's whole `.fft` is 32 MB. A band-limited upload
(bins `1 … ceil(nharms·r_hi)+m`) is the general answer, and matters for files like
`NGC6624_16L_DM87.40_red.fft` at 1.4 GB — that *fits*, but only just, alongside a
`262144` workspace.

**`Nprof` therefore becomes a GPU-specific parameter, not `--blocksize`.** The
existing chunk-invariance guarantee is what makes this safe: results must not
change, and there is a pin for that (§5).

### 3.3 Kernel sketches

**K1 — interpolation.** Grid `(harmonic h, trial-group g)`, one warp per
`(h, group of 32 trials)`, lane = trial. Body is `fill_harmonic_row_direct!`'s
inner loop with the `NTuple{V}` accumulator replaced by one register per lane:

```
sre = sim = 0f0
for j in 1:nj
    w   = gW[o + (j-1)*32 + lane]      # coalesced
    sre = fma(w, re[b0+j], sre)        # re[b0+j] broadcast via L1/__ldg
    sim = fma(w, im[b0+j], sim)
end
ftprofs[trial, h+1] = conj_mul(gA[lane, g], sre + im*sim)
```

Note the store is `ftprofs[trial, h+1]` — **transposed relative to the CPU**, so
consecutive lanes write consecutive addresses. That layout is what the CPU
rejected (CLAUDE.md records `(Nprof, Hₖ+1)` as *2–3× slower* for FFTW), and it is
almost certainly right here. **Do not carry CPU layout verdicts across; re-measure
both.**

**K2 — the six inverse transforms.** cuFFT batched C2R. The stride-`k` decimation
view ports directly to cuFFT's advanced data layout (`istride`, `idist`,
`inembed`), so the 2026-08-16 "no gather" win survives unchanged: the five
decimated plans read the same `ftprofs` where it lies.

**K3 — the boxcar gate.** With `profs` laid out `(trial, bin)`, one thread per
trial reads `profs[trial, i]` coalesced as `i` advances — which is the GPU form of
`_bc_transpose!`, obtained for free from the layout rather than from a tile
transpose. Each thread runs the same wrapped prefix/max recurrence, carrying a
`wmax`-entry circular buffer (~36 floats) plus 9 per-width maxima in registers.
Alternative if register pressure bites: one thread per `(trial, width)`, 9× the
prefix work but trivial registers. Measure both.

`ladder_boxcar_widths` and the `(k, W)` pruning are host-side plan data — upload
the bank per `k`.

**K4 — compaction.** Atomic append of `(trial, k, mval)` for `mval > exactcut`
into a device buffer with a capacity guard; download, then run today's `Float64`
`_profile_boxcar` on the host for exactly those trials. Preserves the two-phase
gate semantics *and* keeps the reported S/N of every candidate `Float64`-exact,
which matters for the §3.2 Monte Carlo.

**Stage 3 (later): fusion.** One block per group of trials, harmonics resident in
shared memory, the boxcar scan run without a global round-trip. That removes the
`ftprofs` and `profs` traffic (~0.12 s). **It does not remove cuFFT** — §0.1
settled that: the hand-rolled direct-DFT replacement is 3.8× slower and
latency-bound, so fusion here means fusing *around* cuFFT (interp→transform and
transform→metric), not replacing it.

### 3.4 Packaging — how CPU and GPU code coexist, and how a user switches

**A package extension**, `ext/CoherentSearchCUDAExt.jl`, with `CUDA` in
`[weakdeps]`. This is Julia's built-in mechanism for exactly this problem, and
the three properties that matter were **measured on 2026-08-25**, not assumed:

1. **A CPU-only user pays nothing — literally.** With `CUDA` declared as a weak
   dependency, `Pkg.add`/`Pkg.develop` does **not** put CUDA in the user's
   `Manifest.toml`, does not download it, does not precompile it, and does not
   load it. Verified on a scratch package: `grep 'name = "CUDA"' Manifest.toml`
   → absent, `using WeakDemo` → **0.006 s**, `CUDA loaded: false`. So
   `Pkg.test()`, the crossval, `foops`, and every CPU search are untouched, and
   the repo keeps working on machines with no GPU and no NVIDIA driver.
2. **It turns on by itself when CUDA is present.** If the user's environment has
   CUDA and the session loads it, Julia loads the extension automatically and the
   GPU backend registers itself. No flag, no rebuild, no separate package.
3. **Runtime opt-in works too, which is what `--gpu` uses.** `@eval using CUDA`
   from inside `main` triggers the extension mid-session. **Measured cost:
   5.93 s** warm — comparable to CairoMakie's 9.0 s, which is precisely why
   plotting is off by default here. So **`--gpu` is opt-in, mirroring `--plot`**,
   and a GPU search amortises that 5.9 s across every file in the invocation (the
   CLI already searches many files per run and `SearchCache` already reuses plans
   across them). A GPU sysimage would remove it, the same way `sysimage/` is worth
   2.5x for plotting runs and nothing otherwise.

The user-facing switch is therefore:

```sh
julia --project=. bin/coherent_search.jl FILE.fft          # CPU, unchanged, 0 s of CUDA
julia --project=. bin/coherent_search.jl --gpu FILE.fft    # GPU, +5.9 s once per invocation
```

and `--gpu` on a machine without CUDA fails with a message naming the fix rather
than a `MethodError`.

**The design constraint this imposes, and it is a hard one:** an extension may
only add methods on **new types**. It must never redefine a method the base
module already defined — that is method overwriting, and Julia raises
`ERROR: Method overwriting is not permitted during Module precompilation`. The
first version of the probe demo did exactly that and failed to precompile. The
shape that works:

```julia
# src/search.jl                      # ext/CoherentSearchCUDAExt.jl
abstract type Backend end            struct CUDABackend <: Backend end     # NEW type
struct CPUBackend <: Backend end     search(ft, p, ::CUDABackend) = ...    # NEW method
const _GPU_BACKEND = Ref{Any}(nothing)
                                     __init__() = CUDA.functional() &&
gpu_backend() = _GPU_BACKEND[]           (CoherentSearch._GPU_BACKEND[] = CUDABackend())
```

so the base module carries an abstract `Backend` and a `Ref` the extension
*populates* at load time. `search` then dispatches, and **every existing CPU code
path keeps its current method unchanged** — which is what keeps all three
existing pins green by construction rather than by testing.

`CUDA.jl` directly rather than `KernelAbstractions.jl`: cuFFT is needed anyway
(§0.1), the kernels want warp-level control, and the target is NVIDIA. Keep the
kernel bodies plain enough that a KA port stays cheap if AMD ever matters (§7).

---

## 4. Staging, with a gate after each stage

Each stage ends in a measurement and a decision. **Nothing merges without its
pin.**

**Stage 0 — feasibility spike. DONE 2026-08-25**, see §0.1 and
`bench/gpu_probe.jl`. CUDA.jl functional; 82% of FP32 peak and 74% of bandwidth
reachable; cuFFT excellent at all six fold depths; the DFT alternative measured
and rejected. *Gate passed, and it corrected the target (§2) and killed the
stage-3 megakernel (§3.3).*

**Stage 1 — K1 alone, against the CPU. DONE 2026-08-25.** Extension skeleton
(`src/backend.jl`, `ext/CoherentSearchCUDAExt.jl`), the interpolation kernel, and
`test/test_gpu.jl`. *Gate passed on accuracy, missed on speed — see §4.1.*

**Stage 2 — the whole chunk pipeline. DONE 2026-08-25**, including candidate
extraction and the `--gpu` CLI. *See §4.2–§4.5.*

**Stage 3 — fusion.** Megakernel / direct-DFT experiment, chunk-size sweep,
`Nprof` sweep. *Gate: measure, and be willing to keep stage 2 if fusion does not
pay — this repo has four recorded cases of an isolated bench inverting in situ.*

**Stage 4 — the parts deliberately deferred.**
`--sigma measured` (device-side median), `--normalize`'s two-pass mode,
`metricstats` histograms, multi-DM streaming, multi-GPU.

### 4.1 Stage-1 results

**Accuracy: passed, comfortably.** GPU vs CPU on the interpolated harmonic stack,
PM0063, `nharms = 60`, over five (chunk, `t0`) combinations including ones that
straddle 32-trial group boundaries: **max relative error 6.0e-8 to 9.8e-8 at
`Float32` weights**, 2.9e-8 to 4.8e-8 at `Float64` — against a pin set at 1e-5.
The `filled` flags agree exactly, so the two agree about *where a harmonic gives
up* as well as about its value.

**Batch invariance: BIT-EXACT**, at chunk sizes 1024 / 512 / 100 / 37 against one
2048-trial chunk, with a CPU-vs-CPU control alongside. This is the §5 pin that
protects the sensitivity Monte Carlo, and it is the property that lets the GPU
run at `Nprof = 262144` while the CPU runs at 2048 without a single result
moving. It holds because groups are anchored to the **global** trial index and
partial end groups are computed in full and masked on store — the CPU design,
transferred intact. `test/test_gpu.jl` pins it (117 tests).

**Speed: 18.5x one Xeon core, which is only ~0.93x the 20-core socket.**
ns per (harmonic, trial), `Float32` weights:

| | `Nprof` = 2048 | 65,536 | 262,144 |
|---|---|---|---|
| CPU `-t 1` | **4.997** | 9.713 | — |
| GPU | 0.685 | **0.270** | 0.280 |

(The CPU's 2048 is its tuned point; at 65536 its plane buffers leave cache and it
degrades 2x. `Nprof = 65536` is the GPU's knee, as §3.2 predicted.)

**The kernel is at 4.8% of this card's achievable FP32, and two plausible
explanations are now measured and dead:**

1. **Not the 64-bit division.** `direct_chunk_state` is a `mod`/`fld` of a 64-bit
   product, and GPUs have no hardware 64-bit divide. Replacing it with a host-
   supplied `(res0, qint0)` plus a 32-bit recurrence, and switching the inner
   loop from `o + (j-1)*V + lane` to running 32-bit indices, moved the time from
   0.262 to 0.270 ns — **nothing, or very slightly worse.** The change is kept
   (it is the more obviously correct code and it removes an overflow cliff) but
   it bought no speed.
2. **Not L2 bandwidth on the weight table.** The apparent traffic is ~110 B per
   (harmonic, trial) = **407 GB/s**, above the 237 GB/s device copy, which reads
   like a bandwidth wall. `bench/gpu_interp_probe.jl` varies *only* the weight
   traffic, by 32x, holding the arithmetic and the access shape fixed:
   **`real` 0.1773 vs `broadcast` 0.1766 ns — identical.** Volume is irrelevant.
   Removing the load entirely is **1.73x**, so what costs is the load's issue
   slot and latency.

**So the remaining idea worth trying is register reuse, not shared memory.** The
group residue cycles with period `ngrp`, so groups `gi` and `gi + ngrp` use the
*identical* weight block; a warp that holds one block in registers across those
repeats removes the load from the inner loop rather than making it cheaper.
**1.73x is the ceiling that buys**, and the real kernel carries a further ~1.5x
of per-group setup and store over the idealised loop. Staging weights in shared
memory would cut bandwidth we are not short of.

**What this does to the §2 projection — read this before quoting 2.2–2.9x.**
Interp measures ~0.93x the socket, not the 3–5x §2 assumed, and cuFFT measures
~1.9x. Weighting by the CPU's phase shares, a staged port on **this** card now
projects **~1.3–1.5x the 20-core Xeon**, not 2.2–2.9x. That is a real result and
it should not be dressed up: **on a GTX 1080 a good GPU port roughly ties a
dual-socket 20-core server.** It also makes the newer card decisive rather than
merely nice — §0.1 measured no architectural wall, and every ratio here is a
property of a 2016 card with 20 SMs.

### 4.2 Stage-2 results — and a layout assertion in §3.3 that was WRONG

**Equivalence, all six rungs, PM0063:** profiles **1.1e-7 to 1.9e-7**, boxcar
metric **1.2e-7 to 3.1e-7**, against a 1e-5 pin. On the 7.1185 Hz pulsar the GPU
and CPU agree on the peak value *and* the argmax at both `k=1` and `k=4`.
`test/test_gpu.jl` is **166 tests** (up from 117), pinned per rung so a
decimation bug cannot hide behind rung 1 being right; the CPU suite is 747/747,
unchanged.

**The layout mistake.** §3.3 asserted the transposed `(Nprof, nharms+1)` store
was "almost certainly right" and that the CPU's contrary verdict should not be
carried across. Transposed *is* right for the interpolator — the CPU layout makes
its store a 32-way scatter and costs **2.54–2.65x** — but §3.3 also assumed cuFFT
would want it. Measured:

| | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | all rungs |
|---|---|---|---|---|---|---|---|
| dim 1 (CPU layout) faster by | 2.02x | 1.80x | 1.76x | 1.75x | 1.68x | 1.43x | **1.84x** |

**I verified the transposed layout for *correctness* (1.7e-7 against a CPU
`irfft`) and never timed it against the alternative.** That is this file's
standing microbenchmark lesson in a new form: *checking that something works is
not evidence that it is fast.*

**Neither pure layout wins** — transposed throughout pays ~0.23 s on the
transform, CPU-layout throughout pays ~0.21 s on interpolation. So each phase
keeps the layout it wants and a **dedicated transpose-and-decimate kernel** sits
between them, staging a tile in shared memory so both the read and the write are
coalesced. It also produces the six dense stacks in one pass, which is needed
anyway: **cuFFT cannot transform a strided view** ("Illegal conversion of a
DeviceMemory to a Ptr"), so the CPU's 2026-08-16 in-place stride trick does not
port. The transpose costs **0.089 s against the strided `copyto!` gather's 0.687
— 7.7x** — so the gather the CPU deleted for being pure duplicated traffic is
even more wrong here.

**Measured pipeline, GTX 1080, scaled to the reference workload:**

| `Nprof` | interp | transpose | transform | boxcar | total | vs CPU `-t 20` |
|---|---|---|---|---|---|---|
| 16384 | 0.141 | 0.100 | 0.309 | 0.205 | 0.756 s | 1.34x |
| 65536 | 0.125 | 0.092 | 0.282 | 0.162 | 0.661 s | 1.53x |
| **131072** | **0.112** | **0.089** | **0.279** | **0.138** | **0.619 s** | **1.63x** |

(Superseded by §4.4 after the boxcar work; kept because §4.2's projection
accounting is written against it.)

**Against §0.46's projection of 0.490 s this is 26% optimistic, and the
accounting says exactly where.** Transform 0.279 vs 0.275 projected and interp
0.112 vs 0.135 were both right; the metric came in at **0.138 against 0.080** (1.7x
worse), and **the transpose phase was not in the projection at all** — it did not
exist as a concept until cuFFT's stride limitation forced it. Every end-to-end
number in §0.3–§0.46 that carries a projected metric column should be read as
~25% optimistic until the metric is re-measured per card.

### 4.3 Boxcar tuning — one idea worked, two hypotheses died

`bench/gpu_boxcar_bench.jl`, on genuine chunk profiles in the production device
buffers, summed over all six rungs and scaled to the reference workload:

| variant | B=32 | B=64 | shared/block |
|---|---|---|---|
| 1 — shared prefix, per-width scan | 0.1621 | 0.1649 | 16 KB |
| 2 — no shared, register sliding window | **1.871** | **3.116** | 0 |
| **3 — width-fused scan** | **0.1282** | 0.1305 | 16 KB |

**Variant 3 is 1.26x and bit-identical to variant 1** (0.0e+00, not a tolerance).
The scan is `max_p (psum[p+w] - psum[p])`, two shared loads per (phase, width) —
but `psum[p]` does not depend on `w`. Putting phase outside and width inside
loads it once for every width, cutting shared traffic from `2*nw` to `nw+1` per
phase (10 → 6 at the `k=1` bank). Predicted 1.67x on the scan; delivered 1.26x on
the phase, the difference being the prefix-sum pass, which is untouched. The
per-width running maxima have to be an `NTuple` with `NW` a `Val` — indexed by a
runtime `wi` they spill to local memory and the win vanishes, the same trap
`DIRECT_GROUP_V` and `_BC_BATCH` document on the CPU.

**Both of §4.2's stated hypotheses were wrong.**

1. *"32 threads per block is poor occupancy"* — B is **flat**: 0.1621 vs 0.1649
   at B=64, and 128 does not fit in 48 KB of shared. Occupancy is not the limit.
2. *"~3x off bandwidth, so there is headroom"* — there is, but not where I said.
   Variant 2 removed shared memory entirely to raise occupancy, betting the extra
   profile re-reads would be L1 hits. It is **12–19x SLOWER** (1.87 s at B=32,
   3.12 s at B=64 — and it gets *worse* with more threads, which is the opposite
   of an occupancy story). Shared memory was earning its keep: the sliding sum is
   a serial dependency on a *global* load, and ~400-cycle latency in a dependent
   chain is not something occupancy can hide at these trip counts. **Do not
   re-guess this one.**

### 4.4 Stage-2 pipeline, measured

| `Nprof` | interp | transpose | transform | boxcar | total | vs CPU `-t 20` |
|---|---|---|---|---|---|---|
| 16384 | 0.139 | 0.103 | 0.309 | 0.181 | 0.731 s | 1.38x |
| 65536 | 0.124 | 0.093 | 0.290 | 0.135 | 0.642 s | 1.57x |
| **131072** | **0.113** | **0.092** | **0.281** | **0.126** | **0.611 s** | **1.65x** |

**The transform is now 46% of GPU time and the clear next target**; interp and
the boxcar are 18% and 21%. Note this ordering is a GTX 1080 statement — §0.3
measured the Ada doing the transform stage 3.9x faster on its 40 MB L2, which
would leave the *boxcar* as that card's largest phase. Re-measure per card before
optimising, as always.

### 4.5 End to end — `--gpu` works, and the loop cost more than the kernels

```sh
julia --project=. bin/coherent_search.jl --gpu --blocksize 131072 FILE.fft
```

**Candidate lines are byte-identical to the CPU path** on PM0063 at threshold 6 —
all seven, including the 7.1185 Hz pulsar at 12.11 and the 144.91 Hz `k>1`
detection. That is better than promised: the two agree to ~2e-7, which happens to
round identically at the reported precision. **Do not read it as a guarantee** —
`--gpu` is documented as producing comparable, not bit-identical, candidates, and
a near-threshold trial could still cross either way.

**Warm in-process search, PM0063 0.1–33.3 Hz, GTX 1080 against fitzroy's 20 cores:
GPU 0.739 s vs CPU 0.959 s — 1.30x.**

**Three design points worth keeping:**

- **Candidate extraction is a download, not a device compaction.** One `Float32`
  per (trial, rung) is ~200 MB over the whole workload, ~0.02 s of PCIe, measured
  at 8.3% of loop time. Atomic compaction would save nothing and add a capacity
  guard and a failure mode.
- **There is no `Float64` rescan and none is needed.** The CPU gates in `Float32`
  and re-scores within `boxcar_gatemargin` of threshold in `Float64`; the GPU is
  `Float32` throughout and agrees to ~2e-7 — four orders inside the 0.01 margin,
  so it cannot move which trials become candidates.
- **`@eval using CUDA` inside `main` raises "method too new to be called from this
  world context"**, because the extension's `_region!` is defined after `main`
  started. `Base.invokelatest` on the GPU path is the fix, once per file.

**What actually cost the time, and none of it was a kernel.** The first working
version ran 1.536 s against 0.699 s of measured GPU work. Three rounds of
instrumentation:

1. **Six synchronous `copyto!`s per chunk** — one per rung, each blocking until
   its kernel retired. Batching them into one column-major download: 1.536 →
   1.376 s. Real, and much smaller than expected.
2. **A first instrumentation pass blamed the host scan at 90.6% (7.27 s).** That
   was the harness's own bug — an inline loop over *script globals*, i.e. type
   instability, not the shipped code, which has the same loop inside a function.
   Behind a function barrier it is **0.0000 s**. A measurement harness can have
   the bug it is looking for.
3. **The real cost was CPU work outside the loop**: `search` sized the host
   `Workspace` and `_sigma_sanity_check` to the GPU's `Nprof`. At 131072 that
   allocated hundreds of MB, planned FFTW transforms for it, and then ran three
   *CPU* chunks of 131072 trials — **~0.68 s against 0.70 s of GPU work, doubling
   the search.** The sanity check is a property of the data, so the workspace is
   now capped at 2048 on a non-CPU backend. **1.376 → 0.739 s.**

**Deferred, and they error clearly rather than silently doing something else:**
`--sigma measured` (needs a device MAD), `--normalize` (needs the two-pass
per-`(k, frequency)` statistics), `--metricstats`.

### 4.6 A large file, and a lesson about whose GPU it is

`NGC6624_16L_DM87.40_red.fft` — **1.29 GiB of amplitudes, T = 26459 s, 105.5M
trial fundamentals, 12.6x the reference workload**:

| | GTX 1080 | fitzroy `-t 20` | |
|---|---|---|---|
| warm search | **9.53 s** | 11.82 s | **1.24x** |
| ns per trial | 90 | 112 | |
| candidates | 135 | 135 | agreeing to ~1e-7 |

**The per-trial cost is flat across a 12.6x size range** (88 -> 90 ns on the GPU,
115 -> 112 on the CPU), so the pipeline scales linearly and the 1.24–1.30x is not
an artefact of the small file.

**But it produced a desktop low-memory warning at 86% device usage**, and that is
the part worth recording. Two defects, both fixed:

- **`_region!` uploaded the whole `.fft` on every call and never freed it.** Three
  searches of NGC6624 reached **5.87 GiB of 7.92**. Explicit `unsafe_free!` of
  every device buffer brings that to 3.74, and `CUDA.reclaim()` returns it to the
  *driver* rather than to CUDA.jl's pool — without which the desktop does not get
  its memory back at all. Residual after a small search is now 221 MiB, which is
  the CUDA context.
- **Nothing checked whether the search would fit.** `_check_device_memory` now
  refuses up front, naming the two knobs that help: `--blocksize` (the workspace
  scales with it) and *not* using a display GPU (the amplitudes do not). It warns
  above 60% of free memory and errors above 90%.

**The general point: on a card that drives a display, memory is shared with the
compositor, and an out-of-memory error is a far better outcome than quietly
starving the desktop.** That is why the guard errors rather than trying to
squeeze. It is also a reason to prefer `--blocksize` at the low end of §0.3's
sweep on such a card — the Ada's optimum of 32768 uses a quarter of 131072's
workspace and is *faster* there anyway.

**Still open for throughput mode:** `GPUChunk` and `GPUInterpPlan` depend only on
`(params, Nprof, r_lo)` and not on the file's contents — exactly the property that
makes the CPU's `SearchCache` safe across files — so they should be cached across
a multi-file run rather than rebuilt and freed per call. Only the amplitude
upload is genuinely per-file, and §7.2's prefetch would overlap that with the
previous file's search.

### 4.7 Classifying a new card in one command

```sh
git pull
./bench/gpu_probe_setup.sh bench/gpu_search_report.jl     # first time on a host
# then, directly:
julia --project=$PREFIX/env bench/gpu_search_report.jl FILE.fft [--cpu] [--band lo hi]
```

Prints device identity, a `--blocksize` sweep, per-phase GPU timings, the
candidates, and a pasteable summary block.

**Two things about the numbers, both built into the output.**

- **Per-phase timing serialises the GPU queue.** A wall-clock timer around a CUDA
  launch measures the launch, not the work, so each phase needs a
  `CUDA.synchronize()` around it — which changes what is being measured. So
  `gpu_timing!` is **opt-in**, off in normal runs, and the report gives a clean
  total from a pass with it *off* and the shares from a separate pass with it
  *on*. On the GTX 1080 the instrumented total reads 0.736 s against a clean
  0.896 s, so the inflation is ~20%: **read the shares, take the total from the
  sweep.** This is the GPU form of the rule the CPU's `phase_times` already
  follows, but stricter, because there the timers are ~0.03% and always on.
- **The `--cpu` arm is optional and off by default.** A CPU run on an unfamiliar
  host is not comparable to fitzroy's Xeon — three hosts have already differed by
  up to 2.4x per core (§0.45) — so it is there for a same-host ratio only, and the
  host must be quoted with it.

**Reference output, GTX 1080, PM0063:**

```
  best blocksize 131072 | 0.896 s | 107.1 ns/trial | 7 cands
  phases: zero 2.5%  interp 15.4%  transpose 12.9%  transform 38.5%
          boxcar 16.5%  download 6.5%  scan 7.6%
```

**What to look for on a new card.** The transform is 38.5% here and is the phase
the two other cards should change most — §0.3 measured the Ada doing that stage
**3.9x** faster than the 1080 on its 40 MB L2 and the 2080 Super 2.75x on
bandwidth, which would drop it to ~14-18% and make the *boxcar* the largest
phase. The blocksize row is the other one to watch: the 1080 wants the largest
chunk and the Ada should want a much smaller one, and this report finds each
card's own answer rather than assuming.

**Both of those were run on 2026-08-25 and the transform-share prediction was
WRONG — it stayed the largest phase on both cards (31.0% and 33.0%). The
blocksize prediction was right on both. See §4.8.** (The "2.8x" this paragraph
originally quoted for the Ada was a misreading of §0.3's own table, which says
0.275 -> 0.071 s = 3.9x; corrected above. It did not change the verdict — the
prediction was wrong for a structural reason, not an arithmetic one.)

### 4.8 Two more cards, end to end — and four pre-registered predictions scored

`bench/gpu_search_report.jl` on **`NGC6624_16L_DM87.40_red.fft`** (1.29 GiB,
T = 26459 s, **105,519,959 trial fundamentals**, 0.1-33.3 Hz, nharms 60,
maxdecim 6), run 2026-08-25 by Scott on two hosts. This is §4.6's large file, so
the GTX 1080's 9.53 s warm search is the same-file baseline.

| | GTX 1080 | **RTX 2080 Super** | **RTX 4000 SFF Ada** |
|---|---|---|---|
| host | fitzroy | `spare2` | `hypatia` |
| arch / SMs x cores | Pascal, 20 x 128 | Turing, **48 x 64** | Ada, **48 x 128** |
| SMs x clock | 34.7 | 87.1 | 74.9 |
| bandwidth achieved | 237 GB/s | **431** | 239 |
| L2 | 2 MB | 4 MB | **40 MB** |
| **best `--blocksize`** | 131072 | **262144** | **16384** |
| clean total | 9.53 s (warm) | **4.459 s** | **4.292 s** |
| ns per trial | 90 (warm) / 107.1 (report) | **42.3** | **40.7** |
| candidates | 135 | **135** | **135** |

**Correctness first: 135 candidates on both, and the printed top five agree with
each other and with the CPU digit for digit** (0.1699323 Hz at 7.057, 0.2160158
at 6.343, 0.2235954 at 7.806, 0.2360172 at 8.746, 0.2450950 at 6.289). That is
**sm_61, sm_75 and sm_89 all agreeing with fitzroy's 20-core CPU** on a 105.5M
trial blind search, and the candidate count is stable across a 16x span of
`--blocksize` on both cards. §5's batch-invariance pin is doing its job on three
microarchitectures.

**End to end the two 48-SM cards are 2.5-2.6x the GTX 1080 and ~2.7x fitzroy's
20-core Xeon** (11.82 s on this file, §4.6). Read the 1080 comparison as ±10-20%:
its phase shares come from a PM0063 report run and its 9.53 s from a warm
in-process search, and the report's clean total runs ~1.2x the warm number
(0.896 vs 0.739 s on PM0063). **The two new cards are exactly comparable to each
other** — same file, same harness, same day.

#### The four predictions, scored

**1. "The two modern cards are a DEAD TIE" (§0.4, §0.46) — HIT, and it is the
best-supported claim in this file.** Predicted equal to four digits (0.1998 s
each); measured **4.459 vs 4.292 s, the Ada 1.039x ahead.** A 2019 consumer card
and a current workstation card, differing by 1.78x in FP32, 1.8x in bandwidth and
**10x in L2**, land within 3.9% of each other on a real 105M-trial search. They
still get there differently, and now by *measured* phases rather than projected
ones: the Ada wins the transpose (1.83x) and the host scan; the 2080 Super wins
interpolation (1.30x) and the boxcar (1.22x); the **transforms are a dead heat**
(13.11 vs 13.43 ns/trial) despite the 10x cache. §0.4's hardware lesson stands and
is now end-to-end rather than a phase model: **above ~48 SMs this workload does
not care much what you buy.**

**2. "The transform drops from 38.5% to ~18% and the boxcar becomes the largest
phase" (§4.7) — MISS, and the construction of the prediction was the error.**

| phase | GTX 1080 (PM0063) | RTX 2080 Super | RTX 4000 SFF Ada |
|---|---|---|---|
| zero | 2.5% | 3.1% | 3.5% |
| interp | 15.4% | 12.6% | **17.0%** |
| transpose | 12.9% | **16.2%** | 9.2% |
| **transform** | **38.5%** | **31.0%** | **33.0%** |
| boxcar | 16.5% | 18.5% | **23.4%** |
| download | 6.5% | 6.2% | 6.5% |
| scan (host) | 7.6% | **12.3%** | 7.3% |

The transform is **still the largest phase on both cards**, and the boxcar is
second. The shares barely moved at all. The prediction divided the transform by
2.8x (itself a misread of §0.3's 3.9x) and implicitly held **every other phase at
1080 speed** — but the whole pipeline sped up by 2.5-2.6x, so the shares are
nearly invariant. **A share can only move if the phases scale *differently*, and
here they scale within a factor of ~1.6 of each other.** Per-phase speedup over
the 1080, in ns/trial:

| | zero | interp | transpose | transform | boxcar | download | scan | **total** |
|---|---|---|---|---|---|---|---|---|
| RTX 2080 Super | 2.04x | **3.09x** | 2.02x | **3.14x** | 2.26x | 2.65x | 1.56x | **2.53x** |
| RTX 4000 SFF Ada | 1.88x | 2.38x | **3.69x** | **3.07x** | 1.86x | 2.63x | 2.74x | **2.63x** |

So the answer to "does this change what is worth optimising next" is **no: the
transform is still the target on every card measured.** Excluding the host-side
scan, the transform is 1.4-1.7x the next-largest device phase on both.

**3. "`--blocksize` should split 262144 / 16384-32768" (§0.3, §0.4) — HIT on both
cards, and the L2 story carries from isolated cuFFT into the real pipeline.**
The search-level sweeps are monotone in opposite directions, exactly as the
standalone probe said:

| ns/trial | 16384 | 32768 | 65536 | 131072 | 262144 |
|---|---|---|---|---|---|
| RTX 2080 Super | 52.4 | 47.4 | 45.5 | 43.4 | **42.3** |
| RTX 4000 SFF Ada | **40.7** | 43.7 | 47.2 | 48.4 | 49.7 |

`--blocksize` is worth **1.24x on the 2080 Super and 1.22x on the Ada** end to
end, and the two cards want opposite ends of the range. **It is a per-device
parameter, confirmed at the search level on three cards.**

**But the *magnitude* of the Ada's L2 win does NOT carry, only its direction.**
Isolated, §0.3 measured the Ada's transform stage **3.9x** the 1080's; in-search
it is **3.07x**, while the 2080 Super's isolated 2.75x became **3.14x**. The two
48-SM cards converge to ~3.1x whatever their cache. The likely reason is that the
probe gave cuFFT the whole 40 MB L2 to itself, while in the pipeline it shares
that cache with the interpolator's amplitude windows, the transpose tile traffic
and the boxcar's profile reads. Supporting evidence from the sweep: the probe put
the Ada's knee between 32768 and 65536, but in-search 16384 already beats 32768 —
**the effective knee has moved below the bottom of the sweep, and the sweep should
be extended to 8192 and 4096 on that card.**

**4. §0.46's projected metric column, replaced by measurement** (scaled to the
PM0063 reference workload, 8.366M trials, so it is directly comparable to
§0.3-§0.5):

| reference workload (s) | transform | interp | metric | absent from the projection | total |
|---|---|---|---|---|---|
| GTX 1080 projected | 0.275 | 0.135 | 0.080 | — | 0.490 |
| GTX 1080 **measured** | 0.345 | 0.138 | **0.148** | 0.264 | **0.896** |
| 2080 Super projected | 0.100 | 0.056 | 0.044 | — | 0.200 |
| 2080 Super **measured** | 0.110 | **0.045** | **0.066** | 0.134 | **0.354** |
| Ada projected | 0.071 | 0.050 | 0.079 | — | 0.200 |
| Ada **measured** | 0.112 | 0.058 | **0.080** | 0.090 | **0.341** |

**The bandwidth model for the metric is dead.** It predicted the boxcar would
scale with GB/s: 1.82x for the 2080 Super and **1.01x** for the Ada, which has the
1080's bandwidth exactly. Measured: **2.26x and 1.86x** — the Ada gains 1.86x on
*no extra bandwidth at all*. SMs x clock (2.51x / 2.16x) over-predicts by 10-14%
but is far closer, and is consistent with §4.3's finding that the boxcar's cost is
a serial dependency chain through **shared memory**, which is per-SM and has
nothing to do with DRAM. **Every "metric scaled by bandwidth" row in §0.3-§0.5 is
wrong; the boxcar tracks SMs x clock like the other two compute phases.**

**And the Ada's 0.079 s projection was right for the wrong reason** — the model
predicted a 1.00x speedup against a 1080 baseline that was itself 1.85x too low,
and the errors cancelled to 1%. Two compensating mistakes are not a validated
model, and this is why the column had to be measured rather than spot-checked.

**Totals are 1.70-1.83x optimistic on every card**, and it decomposes cleanly:
**26% (Ada) to 38% (2080 Super) of the measured total is phases the projection
never contained** — transpose, zero, download and the host scan — which is §4.2's
recorded "26% optimistic" warning landing almost exactly on the number it
predicted. The remainder is the transform, and it is entirely the Ada's
(1.58x over projection against the 2080 Super's 1.10x). **Interpolation is the one
phase the projections got right on all three cards** (1.02x, 0.80x, 1.16x).

#### Three findings the run produced that were not predicted

**(a) The host-side scan is host-CPU-bound, and it is polluting the GPU shares.**
Candidate extraction downloads and scans on the CPU (§4.5), so its cost belongs
to the host, not the card: `spare2` 5.20 ns/trial against `hypatia` 2.97 —
**1.75x**, against those hosts' CPU search arms at **1.71x** (105.6 vs 61.9 s
`-t 1`). That match is close enough to call it settled. Consequences: the 2080
Super's 12.3% scan share says nothing about the 2080 Super, and **12.3% of a GPU
phase table being a property of the host CPU is a trap for anyone classifying a
card.** The report should split device from host phases, or at least label the
scan.

**(b) §0.46's interpolation ranking REVERSES at each card's own blocksize.**
Isolated at `Nprof = 65536`, the Ada beat the 2080 Super 1.12x (0.0992 vs
0.1112 ns, §0.46). In-search, at each card's own optimum, **the 2080 Super is
1.30x faster** (5.33 vs 6.92 ns/trial). This is not a contradiction of §0.46 — it
is blocksize. The interpolator improves monotonically with chunk size (§4.4: 0.139
-> 0.124 -> 0.113 across 16384/65536/131072 on the 1080, 1.23x), and the Ada is
being forced down to 16384 by its transform. **The Ada is paying for its
transform-optimal blocksize in every other phase**, which is precisely the
tension §0.3 predicted and named.

**(c) That tension is now quantified from the search sweep itself, and it makes
per-rung sub-batching (§0.3) the highest-value GPU work.** Anchoring the standalone
transform-sweep *shape* on each card's measured in-search transform and
subtracting it from the sweep totals:

| non-transform ns/trial | 16384 | 262144 | |
|---|---|---|---|
| RTX 2080 Super | 36.3 | **29.2** | 1.24x better at the large end |
| RTX 4000 SFF Ada | 27.3 | **19.2** | **1.42x** better at the large end |

**Everything except the transform wants the biggest chunk available, on both
cards.** Sub-batching gives each rung its own L2-sized batch while the chunk stays
large, so the Ada could have 19.2 + 13.4 = **32.7 ns/trial -> ~3.45 s, a 1.25x
end-to-end win** — well above the ~1.10x that §0.3's "1.40x on the transform
stage" alone would buy, because the real cost is the blocksize being dragged down
for everyone else. **The 2080 Super is the control and comes out at exactly
1.00x**, as §0.4 predicted, since it already runs at 262144.

*This is a model, not a measurement:* it assumes the standalone sweep's shape
carries into the search, and finding (3) above says the *magnitude* does not.
Treat 1.25x as an upper bound to be tested, not a result.

**TESTED 2026-08-25 AND REFUTED — see §4.10. The sign of the non-transform term
is wrong on the Ada.** Sub-batching now measures the transform's blocksize
dependence directly instead of importing the probe's shape, and what is left
over says the Ada's non-transform phases are **flat to slightly worse** at
262144, not 1.42x better. The 1.25x became **0.960x**. The 2080 Super row
survives (its non-transform really does improve with blocksize); the Ada row was
an artefact of anchoring on the probe. **Do not re-use this table's
`non-transform` column for the Ada.**

#### Refined pre-registration for the RTX A4000 (sm_86)

§0.5's predictions stand; these are the search-level ones the two runs above now
make testable. The A4000 has the Ada's compute (48 SMs, 1.56 GHz, 128 cores/SM)
and the 2080 Super's cache (4 MB) with 448 GB/s:

1. **`--blocksize` prefers the large end, 262144**, monotonically, like the 2080
   Super. If it prefers the small end with 4 MB of L2, §0.3's mechanism is wrong.
2. **Transform ~3.1x the 1080 in-search**, i.e. ~13.2 ns/trial — both 48-SM cards
   landed there regardless of a 10x cache difference, so this tests whether
   in-pipeline transform speed really is set by SM count alone.
3. **Boxcar ~2.2x the 1080** (SMs x clock 2.16x less the ~12% both cards missed
   it by), i.e. ~8.0 ns/trial — *not* the 1.90x that its 448 GB/s would give under
   the now-dead bandwidth model. This is the cleanest available test of finding
   (4): the A4000 and the Ada have identical SMs x clock and 1.9x different
   bandwidth.
4. **Total ~42-44 ns/trial, ~4.4-4.6 s** — i.e. a **three-way tie** with the
   other two modern cards, spanning 11.2-19.2 TFLOP/s and 239-431 GB/s.
5. **Sub-batching worth 1.00x**, as on the 2080 Super.


### 4.9 Per-rung transform sub-batching — implemented 2026-08-25, and the Ada is the test

§0.3 proposed this and §4.8 quantified it; it is now in `ext/CoherentSearchCUDAExt.jl`
(`transform!`, `_sub_cols`, `GPUChunk`'s `sub`/`tail`/`nblocks`). **The Ada has
not been re-run yet — the 1.25x below is still a prediction, and this section
exists so that run scores it rather than explains it.**

**What it does.** Each rung's inverse transform runs in contiguous column
sub-batches sized to its own L2 footprint, instead of one batch over the whole
chunk. cuFFT accepts a contiguous column range of a column-major array — that is
dense memory, not the *strided* view that §4.2 found it refuses — so this needs
no copy, no extra buffer and no kernel. Two plan sizes per rung at most (the
blocks are balanced, so the tail is within one block of `sub`).

**Why it is worth doing at all** is §4.8's decomposition, not a kernel argument:
with one batch per rung the transform and everything else fight over
`--blocksize`, and on the Ada the transform wins — dragging the chunk to 16384,
where the interpolator, transpose, boxcar and scan together pay **1.42x** what
they pay at 262144. Sub-batching lets each side have what it wants.

#### The policy is derived from L2, and it reproduces §0.3's measured optima

`_sub_cols` targets `_SUB_L2_FRACTION = 0.5` of the device's L2 per rung, from
each rung's own bytes per column (`8(H_k+1) + 8H_k` in `Float32` — 968 B at
`k=1`, 168 B at `k=6`, which are §0.3's own figures). Derived against measured,
at `Nprof = 262144`:

| | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 |
|---|---|---|---|---|---|---|
| derived, 40 MB L2 | 21664 | 42974 | 63937 | 84562 | 104857 | 124830 |
| §0.3 measured optimum | 16384 | 32768 | 65536 | 65536 | 131072 | 131072 |

**Monotone in the same direction and within ~1.3x at every rung, from a formula
fitted to none of them.** That is why the constant is a fraction of L2 rather
than a table: the same expression that reproduces the Ada also turns itself off
on the other two cards, with no device list to maintain.

**Two guards, and they are what make `:auto` safe to ship.** A rung is split only
if `_SUB_MIN_COLS = 16384` of its columns fit in the target (the residency gate),
and never into blocks below that floor. §0.3 is the source of both: at
`Nprof = 16384` the `k = 4,5,6` rungs sat at 85–93% of the DRAM copy on working
sets of 2.6–3.9 MB, which cannot be cache and must be too little work for 48 SMs.
The gate means **a 2 MB or 4 MB card declines to split at all** — measured: the
policy returns `Nprof` at every rung for both the GTX 1080 and the RTX 2080
Super, and splits at every rung on the Ada. §0.4 asked for a per-device policy;
this is one, and it derives itself.

**An explicit byte target (`gpu_subbatch!(n)`) bypasses both guards**, on
purpose. Without that, a bench could not sweep past the knee and a small-L2 card
could not exercise the split in a test — the testset would pass by doing nothing.
`:auto` and `:off` are the only two things a search ever uses.

#### Correctness: bit-exact, which is stronger than the rest of the GPU pins

Every column's transform is independent, so splitting the batch is a scheduling
change and must not move a single bit. It does not: **`chunk_profiles` is exactly
equal** (`==`, not a tolerance) between split and unsplit at `k = 1…4`, at
`n = 1024` and a deliberately ragged `n = 999`, at two targets each.
`test/test_gpu.jl` is **226 tests**, up from 166 — the new ones pin the bit-exactness,
that a split really happened, that the blocks tile the chunk exactly, and the
policy's per-device behaviour including the two guards. CPU suite 747/747.

Whole-search A/B on PM0063, GTX 1080, forced split vs unsplit at blocksize
262144: **7 candidates either way and identical to the last bit of S/N.** The
only differences anywhere in this work came from changing *blocksize* (a ragged
50001), and they are two candidates whose `r` differs in the last bits of a
`Float64` with identical S/N and `nharm` — the chunk-origin accumulation order,
pre-existing and unrelated.

#### Measured so far: the GTX 1080 control behaved as predicted

`bench/gpu_subbatch_bench.jl`. `:auto` declines to split, as designed, so the
shipped path is unchanged there. Forcing a split anyway, interleaved, 7 reps:

| blocksize 262144 | min | median |
|---|---|---|
| `:off` (unsplit) | 1.147 s | 1.170 s |
| forced, 2 MB target | 1.120 s | 1.136 s |
| forced, 1 MB target | 1.134 s | 1.169 s |

i.e. **1.02–1.03x at best, against 7–49% run-to-run scatter on a machine that is
also Scott's desktop.** Not resolvable, and consistent with the predicted 1.00x.
**A single-shot sweep first read 1.035x and a clean-looking minimum**; the
interleaved repeat cut it to ~1.02x. That is this file's standing rule about
absolute wall clocks landing again — the target *sweep* shape (204 / 145 / 134 /
131 / 138 ns/trial over 0.125–2.0x L2) is real and useful, the difference between
its floor and the unsplit arm is not.

~~The useful negative result: a forced split is not HARMFUL on a small-L2 card,
merely useless, so the gate is protecting against nothing measurable.~~
**WRONG, and overturned the same day by the RTX 2080 Super — see §4.10.** On a
quieter host the same sweep is monotone and a forced split is a real regression:
**8.7% at the `:auto` target and 34% at 0.125x L2.** The gate is worth those.
The 1080's scatter was hiding it.

#### Pre-registered, for the RTX 4000 SFF Ada

Recorded before the run. `bench/gpu_subbatch_bench.jl` on `NGC6624`, so it is
directly comparable to §4.8's 40.7 ns/trial:

1. **`:auto` splits at every rung** into the columns tabulated above. If it
   declines, `_sub_target_bytes` is reading L2 wrongly.
2. **Blocksize 262144 with `:auto` beats blocksize 16384 with `:off`** — this is
   the whole point, and the direction is the claim. §4.8's decomposition puts it
   at **~32.7 ns/trial, 1.25x**; anything from 1.10x up confirms the mechanism.
3. **Near 1.00x means the mechanism is contended away, and the item should be
   dropped, not tuned.** §4.8 already showed the Ada's isolated 3.9x transform
   win measuring 3.07x in the pipeline because cuFFT there shares its 40 MB with
   the interpolator, the transpose and the boxcar. If residency cannot be
   arranged in situ at all, that is the same finding one step further, and it is
   a real answer rather than a failure.
4. **The target sweep's minimum should sit near 0.5x L2.** If it is far off,
   `_SUB_L2_FRACTION` is the constant to change — and note the 1080's own sweep
   put its floor at **1.0x** L2, not 0.5x, so this is genuinely open.
5. **The 2080 Super stays a 1.00x control**, since `:auto` will not split it.

**The honest status of the 1.25x**: it is a model built by anchoring the
standalone transform sweep's *shape* on the measured in-search transform and
subtracting. §4.8 established that the probe's *magnitude* does not carry into
the pipeline. So 1.25x is an upper bound to be tested, and 1.10x would still be
the largest single GPU win found since stage 2.


### 4.10 Sub-batching scored — the mechanism works, the premise it was built on does not

Both cards run 2026-08-25, `bench/gpu_subbatch_bench.jl` on `NGC6624`.
**§4.9's central prediction was wrong in direction, and the reason is that
§4.8's decomposition had a sign error that only sub-batching itself could
expose.**

| ns/trial, NGC6624 | blocksize 16384 | blocksize 262144 |
|---|---|---|
| **RTX 4000 SFF Ada** `:off` | **41.0** | 49.8 |
| **RTX 4000 SFF Ada** `:auto` | 40.3 *(no split — see below)* | **42.7** |
| **RTX 2080 Super** `:off` | 53.1 | **43.5** |
| **RTX 2080 Super** `:auto` | 53.1 *(no split)* | 43.6 *(no split)* |

#### The predictions

1. **`:auto` splits at every rung on the Ada — HIT.** 13 x 20165, 7 x 37450,
   5 x 52429, 4 x 65536, 3 x 87382, 3 x 87382, working sets 14.0–18.6 MB.
2. **"Blocksize 262144 with `:auto` beats 16384 with `:off` by ~1.25x" — MISS,
   and in the wrong direction: 0.960x.** Predicted 32.7 ns/trial, measured 42.7.
3. **`:auto` at each card's own best blocksize is worth 1.000x on all three
   cards.** This is §4.9's outcome 3, and it says drop or demote, not tune.
4. **Target sweep minimum near 0.5x L2 — HIT.** The Ada reads 44.3 / **42.5** /
   42.6 / 43.9 / 48.7 at 0.125–2.0x, a flat 0.25–0.5x plateau. `_SUB_L2_FRACTION
   = 0.5` is within 0.2% of the best and needs no change.
5. **The 2080 Super is a 1.00x control — HIT exactly.** `:auto` declined to
   split at both blocksizes, and the arms agree to 0.02%.

#### The mechanism works; it is the motivation that was wrong

Sub-batching does precisely what it was designed to do. On the Ada at 262144 it
takes the transform stage from 49.8 to **42.7 ns/trial — 1.166x**, recovering
**7.1 of the 8.8 ns/trial** blocksize penalty (81% of it). The idea is sound and
the implementation delivers.

**But the Ada's optimum blocksize is 16384, where the policy does not engage at
all** — at `Nprof = 16384` every rung's L2-sized batch is already ≥ the whole
chunk, so `_sub_cols` returns `Nprof` and there is nothing to split. So the
shipped benefit is **1.000x on every card measured**: on two of them the gate
declines, and on the third the operating point is below where splitting begins.

**§4.9's premise came from §4.8 finding (c), and that finding is refuted.** It
claimed the non-transform phases are 1.42x better at 262144 than at 16384 on the
Ada — derived by subtracting a *modelled* transform (the standalone probe's
shape) from the measured sweep. Sub-batching lets that be measured instead: with
the transform's blocksize dependence removed, the Ada reads 40.3 at 16384 and
42.7 at 262144, so the non-transform phases are **flat to slightly worse** at the
large end. **The model had the sign wrong**, and it produced a confident 1.25x
from it. The 2080 Super's row survives — there the large blocksize genuinely wins
1.222x — so the two cards differ in the non-transform phases' response to
blocksize, not just the transform's. That is consistent with §0.46's second
explanation: on 40 MB of L2 a *small* chunk makes the whole pipeline resident,
not merely the transform, so the Ada wants small everywhere.

#### An accidental scatter calibration, and it is the most useful number here

Because `:auto` produces **no split at all** at blocksize 16384 on the Ada and at
either blocksize on the 2080 Super, four of the eight headline rows are *the same
code path run twice*. They read 41.0 vs 40.3 (Ada, **1.7%**) and 53.1 vs 53.1 /
43.5 vs 43.6 (2080 Super, **0.02% and 0.2%**). So this harness is reproducible to
a few tenths of a percent on `spare2` and to ~1.7% on `hypatia` — which retires
the Ada's apparent "1.017x at 16384" as noise, and sets the bar any future claim
on these hosts has to clear. **Two identical arms in a benchmark are worth the
run time.**

#### The gate earned its keep, which §4.9 doubted

§4.9 recorded, from the GTX 1080, that a forced split "is not harmful, merely
useless". **The 2080 Super overturns that on a quieter host**: forced splits are
monotone and always worse than unsplit — 58.2 / 49.4 / 47.3 / 45.3 / 45.2 against
43.5 — so at the `:auto` target the gate is preventing an **8.7% regression**, and
at 0.125x L2 a **34%** one. The 1080's ±7–49% scatter was hiding a real effect,
which is this file's standing rule landing yet again.

**And the cost is per-launch overhead, linear and measurable.** Penalty divided
by extra transform launches, over the 2080 Super's four resolvable points:

| target | blocks/chunk | extra launches | penalty | µs per launch |
|---|---|---|---|---|
| 0.125x L2 | 1206 | 483,600 | 1.551 s | **3.21** |
| 0.25x L2 | 606 | 241,800 | 0.623 s | **2.57** |
| 0.5x L2 | 306 | 120,900 | 0.401 s | **3.32** |
| 1.0x L2 | 153 | 59,241 | 0.190 s | **3.21** |

**~3.2 µs per launch, flat across a 16x span of block count.** That also explains
why the Ada tolerates its split and the 2080 Super would not: the L2-derived
policy gives the Ada **35 blocks per chunk** against the 2080 Super's would-be
306, because a 40 MB target needs an order of magnitude fewer blocks to reach.
The gate and the block count are the same fact seen twice.

#### What the code is now worth, and the decision it needs

Peak throughput on all three cards is **unchanged**. What sub-batching buys is
**robustness to a badly-chosen `--blocksize` on a big-L2 card**: the Ada's cliff
between its best and worst sweep point falls from **1.21x to 1.06x**. A user who
does not sweep — which is every user who is not classifying a card — loses much
less by guessing wrong.

**KEPT — Scott's call, 2026-08-25, and his reason is better than the one this
section first gave.** Not "it might help a future card": *there are a wide
variety of GPUs out there and very few people are willing or able to devote
resources to tuning for what they have.* On that criterion the figure of merit
is **the worst case an untuned user hits**, not the best case a tuned one
reaches — and sub-batching improves the worst case by 1.21x -> 1.06x on the one
card where the cliff is steep, at zero cost to the best case, bit-exactly and
with no flag to set. **It should still never be described as a speed win**; it is
a flattening of the `--blocksize` response curve.

**That criterion immediately indicts something bigger — see §4.11.** If the
untuned user is who we are optimising for, the first thing to look at is what
they actually get by default, and `--blocksize` defaults to **2048** on the GPU
as well as the CPU.


### 4.11 The lower sweep floor paid off, and it exposes a much bigger robustness hole

`gpu_search_report.jl` re-run on `hypatia` with the labelled phases and the
sweep extended down to 4096.

**The extended floor was worth 1.074x, for free — §4.8's guess was right.** §4.8
argued the Ada's in-search knee had moved below the bottom of the old sweep,
because cuFFT there shares its 40 MB with the rest of the pipeline. It had:

| Ada, ns/trial | 4096 | **8192** | 16384 | 32768 | 65536 | 131072 | 262144 |
|---|---|---|---|---|---|---|---|
| | 44.5 | **37.9** | 41.0 | 42.8 | 42.9 | 43.1 | 42.6 |

**8192 is a genuine interior minimum** (4096 is worse), and it takes the card
from §4.8's 40.7 to **37.9 ns/trial — 3.994 s on NGC6624**. That is the best
number any card has produced here, and it cost nothing but two more sweep rows.
**The Ada is 15.49x `hypatia`'s own single core** (61.849 s) and **~2.96x
fitzroy's 20-core Xeon** (11.82 s).

#### Device-only shares, and the boxcar finally takes the lead

| at each card's best blocksize | zero | interp | transpose | transform | boxcar |
|---|---|---|---|---|---|
| GTX 1080 (262144), device-only | 2.8% | 17.8% | 14.7% | **45.1%** | 19.5% |
| RTX 4000 Ada (8192), device-only | 5.4% | 24.3% | 14.4% | 26.3% | **29.6%** |

**§4.7's prediction that "the boxcar becomes the largest phase" has finally come
true on the Ada — by a route it did not predict.** Not because a 40 MB L2 made
the transform cheap (§4.8 measured that not carrying into the pipeline), but
because the *blocksize* fell to 8192 and because the host scan and PCIe are now
out of the denominator. Right answer, wrong mechanism, two sections apart.

**The renormalisation arithmetic checks out where it can be checked.** §4.10
predicted 44.8% device-transform for the 1080 from renormalising §4.7's raw
shares; measured **45.1%**. That is worth noting because the same run's absolute
total moved **24%** between sessions (1.115 s against §4.7's 0.896 s) on a card
that drives Scott's desktop — **the shares held to 0.3 points while the seconds
moved 24%**, which is this file's "read the shares" rule proving itself rather
than merely being asserted. **Do not update §4.7's reference total from a
desktop-card run.** The Ada's 26.3% is at blocksize 8192 and so does not score
§4.10's 38.3% prediction, which was a renormalisation of a 16384 run — different
operating point, not comparable.

#### The real robustness hole: `--blocksize` defaults to 2048 on the GPU

`--blocksize` defaults to **2048** — the CPU's tuned value — and `--gpu` does not
change it. Measured on the GTX 1080, PM0063:

| blocksize | ns/trial | vs this card's best |
|---|---|---|
| **2048 (the default)** | **224.5** | **1.65x** |
| 8192 | 166.0 | 1.22x |
| 32768 | 143.7 | 1.06x |
| 131072 | 137.0 | 1.01x |
| 262144 | 136.1 | 1.00x |

**An untuned `--gpu` run gives up 1.65x on this card**, which dwarfs the
1.21x -> 1.06x that §4.10 kept sub-batching for. And the optimum spans **8192 to
262144 — a factor of 32 — across three cards**, so there is no single constant to
move it to; the default has to be derived per device or the user has to sweep.

**A derived rule that fits all three, using the same constant as `_sub_cols`.**
The whole per-trial device footprint of a `GPUChunk` at the default parameters is
`ftprofs` 488 B + the six stacks 1224 B + the six profile buffers 1176 B + `out`
24 B = **2912 B per trial**. Taking the same `0.5 x L2` target:

| | 0.5 x L2 / 2912 | measured optimum |
|---|---|---|
| RTX 4000 Ada (40 MB) | **6868** | **8192** |
| RTX 2080 Super (4 MB) | 687 — below any floor, fall back | **262144** |
| GTX 1080 (2 MB) | 344 — below any floor, fall back | **262144** |

Same two-regime shape as the sub-batch policy, same constant, and it lands on all
three measured optima. **Not implemented — proposed.** It changes a shipped
default, so it is Scott's call, and the floor and the fallback value both want
measuring on more than three cards before being trusted.

#### A bug of mine, recorded because it is the kind that recurs

The device/host labelling in §4.8 introduced `dev = sum(...)` in
`gpu_search_report.jl`, **shadowing the `dev = CUDA.device()` set 60 lines
earlier**, so the pasteable summary block — the last thing the script prints, and
the whole point of it — died with `MethodError: no method matching name(::Float64)`
*after* a full multi-minute run on a remote host. The script had been validated on
the GTX 1080, but only as far as the phase table. **Validate a script to its last
line of output, not to the part you changed.**


---

## 5. Correctness — the fourth pin

The existing three pins (Python oracle → end-to-end equivalence → interpolator)
are untouched, because the CPU path does not change. The GPU adds a fourth rung
below them, and it is pinned at `Float32`, not at machine precision:

1. **`gpu_fill_harmonic_row` vs `fill_harmonic_row_direct!`** (`Float32` weights),
   per harmonic, per chunk offset. Tolerance ~1e-6, the same figure the shipped
   `Float32` weight path already achieves against the exact kernel.
2. **`gpu_chunk_profiles` vs `fill_chunk_profiles!`** — adds cuFFT vs FFTW.
   Tolerance ~1e-5; both are `Float32` transforms of the same input with different
   factorisations.
3. **`gpu_chunk_metrics` vs `chunk_metrics(...; weights=Float32)`** — the new
   equivalence gate, the GPU analogue of the 8.4e-16 pin. Tolerance ~1e-5.
4. **GPU batch invariance** — the direct analogue of the CPU's bit-exact
   chunk-invariance test. A search run at `Nprof = 65536` must produce the same
   candidates as one at `262144` and as one at `2048`. The global-trial-index
   anchoring makes this *achievable*; only a test makes it true. **This is the
   pin that protects §3.2 from a GPU tuning knob quietly changing results.**

**Do not loosen tolerances 1–3 to swallow a discrepancy.** This file records two
occasions where that would have hidden a real defect (the stale sibling-repo
metric; the `sigma_center` divergence), and the pattern is always the same: name
the deliberate difference, pin it separately, and leave the strict pin strict.

**End-to-end comparison is not `diff`.** The GPU path will not produce
byte-identical `.cohout`, so the standing "byte-identical or it moved" rule needs a
replacement: a small comparison tool that matches candidates by frequency and
reports S/N deltas, of the kind `--sigma analytic` already needed when it churned
near-threshold candidates. **Write that tool in stage 1, before it is needed to
adjudicate anything.**

---

## 6. Measurement discipline

This repo has an unusually well-documented history of measurement traps. All of
them apply again, and two apply *harder* on a GPU:

- **Read shares, not seconds.** The in-situ phase timers (`phase_reset!` /
  `phase_times`) are the right instrument and need GPU equivalents — CUDA events
  around each kernel, not `time_ns` around an async launch. **A wall-clock timer
  around a CUDA call measures the launch, not the work**; every GPU timing must
  either use events or `CUDA.@sync`.
- **The desktop is on this GPU.** Xorg + Chrome + Zoom are already 723 MB and 13%
  utilisation. `-t 20` numbers on fitzroy are already treated as unreliable for
  this reason; GPU numbers are worse. Check `nvidia-smi` before timing, quote
  minima over repeats, and prefer the share.
- **Microbenchmarks invert in situ.** Four recorded cases (smooth `fftlen`,
  `_block_sigma`'s gather, the metric's 1.74×→1.13×, `decim_brfft`'s 0.72×
  `Float32` "win"). A GPU kernel benched on hot, resident data with nothing else
  on the card will overstate itself the same way. Score with an in-situ A/B.
- **The CPU-tuned constants mean nothing here.** `_BC_BATCH = 128`,
  `_BC_TR_BJ = 8`, `DIRECT_GROUP_V = 32` were tuned for AVX2/AVX-512 register
  files. Only the last has a GPU meaning, and only by coincidence (V = warp size).
  Re-derive the rest.
- **Do not study fitzroy's GPU codegen anywhere else.** The direct analogue of the
  recorded AVX-512 lesson: `foops` has no NVIDIA GPU, and PTX for sm_61 is not PTX
  for sm_86.

---

## 7. Decisions — all answered 2026-08-25

1. **Is the GTX 1080 the target or the *floor*? → THE FLOOR.** RTX 4000-class
   access already secured, possibly an A100 later. Write FP32-throughout with
   sm_61 as the lower bound; quote the 1080 as "a 2016 consumer card" and the
   newer card as the real number. §0.1 shows no architectural wall (82% of
   achievable FP32, cuFFT scaling with the device), so the §2 ceiling of ~2.5x
   should rise close to linearly. Run `bench/gpu_probe_setup.sh` on each new card
   the day it lands (§0.2), and re-check §3.2's `Nprof` there.

2. **Latency or throughput? → THROUGHPUT.** Scott expects to push **hundreds of
   `.fft` files** through this, not one. That is a design input, not a footnote:
   - **The 5.9 s CUDA load and all plan construction amortise to nothing**, so the
     `--gpu` opt-in costs effectively zero in the mode that matters. The CLI
     already searches many files per invocation and `SearchCache` already keys its
     reuse on `(params, Nprof)` and **never consults the file** — so the device-side
     workspace and plan cache is the exact same pattern, and heterogeneous `N`/`dt`
     across a glob is already proven safe.
   - **Amplitude upload becomes a real, schedulable cost.** PCIe 3.0 x16 is
     ~12 GB/s, so a 32 MB PM0063-class file is ~2.7 ms (irrelevant) but a 1.4 GB
     `NGC6624`-class one is ~0.12 s (not irrelevant). **Overlap it**: pinned host
     buffers plus a second CUDA stream uploading file *N+1* while file *N* is
     searched. Free throughput, and it only exists as an option because the
     deployment model is a queue of files.
   - **The host-side `Float64` rescan and candidate collapse for file *N* overlap
     with file *N+1*'s GPU work** for the same reason. The CPU is otherwise idle
     during a GPU search, which in throughput mode is waste.
   - **Optimise for saturation, not for single-search latency.** A large `Nprof`
     (§3.2) is what fills the device; there is no reason to shrink it to make one
     file finish sooner.
   - **This retires the CPU deployment model's grip on the design.** §3.1 of
     `Summary_and_Future_Work.md` records that one-single-threaded-process-per-DM
     makes `-t 1` CPU-seconds the figure of merit. In GPU throughput mode the
     figure of merit is **files per hour on one device**, and that is what the
     paper section should report.

3. **AMD/ROCm? → NOT NOW, PLAUSIBLY LATER.** CUDA.jl directly, and cuFFT is needed
   regardless. **But keep the kernel bodies portable-shaped**: plain Julia loops,
   no inline PTX, no CUDA-only intrinsics where a generic one exists, and keep the
   launch configuration in one place. A later `KernelAbstractions.jl` port should
   then be a mechanical change to the launch layer plus swapping cuFFT for
   rocFFT — not a rewrite. Do not pay portability *costs* now; just do not build in
   gratuitous barriers.

4. **Paper scope? → A SECTION OF THE SAME PAPER, not a separate one.** So
   **CPU-vs-GPU comparison is a deliverable, not a nicety**, and it has to be as
   carefully controlled as the riptide comparison already is:
   - Same observation, same band, same `nharms`/`maxdecim`, **same candidates**
     (within the §5 tolerances) — quote the work accounting, not just the seconds.
   - Quote the CPU at **`-t 20` (all cores)**, never `-t 1`, per §0.1's corrected
     baseline. The `-t 1` number belongs only in a per-core-efficiency argument.
   - Report **throughput (files/hour)** alongside single-search wall clock, since
     §7.2 makes that the operating mode.
   - Worth reporting **energy**, given the framing in §2: `nvidia-smi` gives board
     power, RAPL gives CPU package power, and a 2016 consumer card against a
     dual-socket server is a genuinely interesting perf/W row.
   - This means stage 4's deferred items (`--sigma measured`, `--normalize`,
     `metricstats`) are **not** needed for the paper, but a **GPU-vs-CPU harness
     is** — build it in stage 2, alongside `compare/compare_riptide.py`.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Kernel launch overhead dominates | Large `Nprof` (§3.2); fuse in stage 3 |
| ~~cuFFT poor at n=20…120~~ | **Retired** — measured 15–19× one FFTW core across all six depths (§0.1) |
| Watchdog kills a kernel / desktop stalls | Chunk-sized launches are ms; keep them so |
| 8 GB is tight for 1.4 GB `.fft` + 262 K workspace | Band-limited amplitude upload; `Nprof` is a knob |
| GPU results drift from CPU results and nobody notices | The four pins of §5, plus the candidate-comparison tool written *early* |
| The GPU track destabilises the Monte Carlo | Package extension: CPU path and all CPU pins are literally unchanged |
| Transform stage is irreducible and caps the win at ~2.5× | Known and quantified (§2); the answer is a bigger card, not a better kernel |
| Effort sinks into a 2016 card's quirks | Decide question 7.1 first |

---

## 9. First concrete steps

1. ~~Stage-0 spike~~ — **done**, `bench/gpu_probe.jl`, results in §0.1. It needs
   `CUDA` in whatever env you run it under; `bench/` does not have it yet.
2. Answer §7.1 and §7.2. §7.1 got sharper: we are at 82% of this card's
   achievable FP32 with no wall in sight, so "is the 1080 the floor?" now decides
   whether the paper's number is 2–3× a server socket or something much larger.
3. Write the candidate-comparison tool (§5).
4. Then stage 1.
