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

**Stage 1 — K1 alone, against the CPU.** Port the interpolator, compare
`ftprofs` to `fill_chunk_profiles!`'s at `Float32`. *Gate: agreement ≲1e-6, and
the interp phase faster than the CPU's ~19% share suggests.*

**Stage 2 — the whole chunk pipeline (K1–K4), staged through global memory.**
This is the deliverable that gives the headline number. *Gate: the §5 pins green
and ≥3× the 20-core Xeon.*

**Stage 3 — fusion.** Megakernel / direct-DFT experiment, chunk-size sweep,
`Nprof` sweep. *Gate: measure, and be willing to keep stage 2 if fusion does not
pay — this repo has four recorded cases of an isolated bench inverting in situ.*

**Stage 4 — the parts deliberately deferred.**
`--sigma measured` (device-side median), `--normalize`'s two-pass mode,
`metricstats` histograms, multi-DM streaming, multi-GPU.

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
