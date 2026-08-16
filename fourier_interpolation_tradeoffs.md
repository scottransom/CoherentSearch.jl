# FFT-correlation vs. direct summation for Fourier interpolation

**What this is.** In August 2026 the Fourier interpolation in CoherentSearch.jl
was rewritten from FFT-correlation onto a fine grid to direct `O(m)` summation at
exactly the frequencies wanted.  (The FFT-correlation arm was kept as a runtime
option for a week and then deleted on 2026-08-16; it survives as the *reference*
kernel `finterp_fft`, which is what the Python oracle is pinned to.  Everything
below is unchanged — the comparison was made when both ran in the search.) It was 1.64× faster end to end and about six
orders of magnitude more accurate. That is a surprising result — an FFT
correlation is `O(N log N)` and direct summation is `O(N m)` — so this document
records *why* it happened, and, more usefully, **the conditions under which it
happens**, so the same question can be asked of other codes without re-deriving
everything.

The immediate motivation for writing it down is PRESTO's `accelsearch`, which is
also built on FFT-correlation Fourier interpolation. **Section 4 works out where
`accelsearch` sits in the trade-off, and the answer is that it is already on the
right side of it — by 4× to 357×.** Section 5 is therefore not "port the direct
method" but a ranked list of what *is* worth benchmarking there, and why.

Written in `CoherentSearch.jl`, where the measurements were made. If the work
moves to PRESTO, copy this file there.

---

## 1. What changed in CoherentSearch.jl

The old path (ported from the Python original, which had no choice — vectorising
a correlation in numpy means an FFT):

1. Build a uniform fine grid of `numbetween` points per Fourier bin covering the
   chunk, by FFT correlation of the zero-stuffed input with the FFT'd kernel:
   two transforms of `fftlen ≈ numbetween·(numbins + m)` points.
2. **Linearly interpolate** that grid at the trial frequencies actually wanted.

The new path evaluates Eqn. 30 of [astro-ph/0204349][r02] directly at each trial
frequency. Measured on `PM0063_034C1_DM445.0_red.fft`, `nharms=60`,
`Nprof=2048`, single thread:

| | FFT + linear | direct | |
|---|---|---|---|
| 60 harmonic rows, one chunk | 7.77 ms | 1.32 ms | 5.9× |
| full 5–30 Hz `--maxdecim 6` search | 70.1 s | 42.7 s | 1.64× |
| amplitude error vs the exact kernel | ~1e-2 | ~1e-10 | — |
| FFTW share of runtime | 49.7% | 12.9% | — |

Candidate lists were unchanged (159 above threshold → 22 survivors either way,
metrics differing ~1%). Full details in `Summary_and_Future_Work.md` §2;
implementation in `src/directinterp.jl`.

[r02]: https://arxiv.org/pdf/astro-ph/0204349

---

## 2. Why direct summation won — the two ingredients

Neither of these is a general argument for direct summation. Both are worth
checking separately in any code considering the swap.

### 2.1 The kernel's coefficients factor into a real vector times a complex scalar

For the pure sinc interpolation kernel, with `offsets = dr - j` and integer `j`,
the `(-1)^j` in `sin(π(dr-j))` cancels the one in `cispi(dr-j)`:

```
coeff_j = sinc(dr-j)·cispi(dr-j) = A(dr) / (dr - j),   A(dr) = sin(π·dr)·cispi(dr)/π
```

Since the interpolation forms `Σ_j conj(coeff_j)·bin_j` and `1/(dr-j)` is **real**,

```
amp(r) = conj(A(dr)) · Σ_j  bin_j / (dr - j)
```

One interpolated point is `m` **real**-times-complex multiply-adds — 4 flops
each, not the 8 of a complex-times-complex FMA. It also reads only `m`
consecutive input bins, so it runs out of L1 rather than streaming
`fftlen`-point arrays, and it vectorises cleanly over de-interleaved real and
imaginary planes.

> PRESTO already exploits a weaker version of this. `gen_r_response()` computes
> `response[k] = (c·sinc, s·sinc)` with `sinc = sin(r_k)/r_k`, i.e. `e^{i r_k}
> sin(r_k)/r_k`, and advances the phase by an angle-addition recurrence rather
> than calling `sin`/`cos` per point. That is the same structure. The difference
> is *where* it is used: PRESTO uses it to build a kernel cheaply (once per
> `(z,w)`), whereas the factorisation's real value is in the **application** — and
> in `accelsearch` the application is done by FFT, so it never enters the inner
> loop.

**The factorisation does not survive `z ≠ 0`.** PRESTO's `gen_z_response()` /
`gen_w_response()` build the response from Fresnel integrals (`fresnl()` in
`responses.c`), whose `k`-dependence is inside the Fresnel arguments and does not
reduce to `A/(dr-j)`. So an f-fdot correlation kernel costs the full 8 flops per
term. **Anyone tempted to port the trick to `accelsearch` should stop here:** it
applies only to the `z = w = 0` sinc case.

### 2.2 Only finitely many distinct `dr` values occur in an entire search

Global trial `t` sits at `r_t = r_lo + t·pnum/q` with `lodr = pnum/q` (exactly
`1/(2·nharms)` at the defaults), so harmonic `h` is evaluated at
`h·r_t = h·r_lo + t·h·pnum/q`, and its fractional part takes at most
`q = 2·nharms` values — `frac(h·r_lo) + i/q` — for every chunk, forever. So the
`m` reciprocals and the scalar `A(dr)` are tabulated **once per harmonic at plan
time** and indexed by an integer residue that advances by a fixed step per trial.
The inner loop then contains no transcendentals, no divisions, and no `mod`.

Without the table, direct summation still beats FFT+linear here, but by ~2×
rather than ~9×: the `m` divisions per point dominate. **This is the ingredient
most likely to be missing in another code** — it needs the evaluation points to
lie on a rational grid whose denominator is small. (`accelsearch` satisfies it
trivially and already exploits it: `ACCEL_NUMBETWEEN = 2` means exactly two
distinct sub-bin offsets, and both are baked into the `numbetween=2` kernel.)

---

## 3. The decision rule

The two ingredients above make direct summation *cheap*. What made FFT
correlation *expensive* in CoherentSearch.jl is separate, and is the part that
generalises: **most of the grid points the FFT computed were thrown away.**

Counting only the points the FFT actually produces, FFT correlation **beats**
direct summation, and by more as the kernel widens. Measured
(`bench/interp_bench.jl`, M points/s, N = 2048, single thread):

| m | FFT, per *grid* point | direct + table, per point | |
|---|---|---|---|
| 8 | 82.0 | 69.3 | FFT 1.18× |
| 16 | 82.7 | 69.0 | FFT 1.20× |
| 32 | 79.9 | 53.4 | FFT 1.50× |
| 64 | 66.1 | 48.0 | FFT 1.38× |
| 128 | 68.2 | 25.3 | FFT 2.69× |

But the fine grid has to be `numbetween` times finer than the trial spacing for
the **linear interpolation** that follows to be accurate — not merely fine enough
to contain the trials — so at the production settings 7 of every 8 grid points
were computed and discarded. Per point actually *wanted*:

| m | FFT + linear | direct + table | |
|---|---|---|---|
| 8 | 6.9 | 53.0 | direct 7.7× |
| 16 | 7.3 | 80.8 | direct 11.0× |
| 32 | 6.5 | 60.3 | direct 9.3× |
| 64 | 5.9 | 41.4 | direct 7.1× |
| 128 | 5.7 | 26.4 | direct 4.6× |

That oversampling was not a tuning mistake that could be dialled away. The fine
grid is anchored to integer Fourier bins while the trials sit at an arbitrary
sub-bin offset `frac(r0)`, so the two coincide only if the kernel is rebuilt per
chunk to absorb that offset — a third transform per harmonic per chunk, which
costs more than it saves. **Direct summation has no grid to align, which is the
actual reason it wins.**

### The rule

Let

- `m` = kernel width in input bins (the number of terms a direct evaluation sums),
- `u` = **utilisation**: the fraction of computed grid points that are consumed,
- `L` = the correlation transform length, `K` = the kernel length in grid points.

Per point *wanted*:

```
direct :  c_d · m                       c_d = 4 flops (real weights) or 8 (complex)
FFT    :  (5·L·log2 L + 6·L) / (u · (L - K))
```

Use FFT when the second is smaller. Three practical readings:

1. **Utilisation is the dominant term.** It multiplies the FFT cost directly. A
   code that computes a fine grid to feed a *later* interpolation is paying `1/u`
   for nothing, and that factor is usually 8–16.
2. **Wide kernels favour the FFT**, linearly in `m`. This is the classic
   intuition and it is correct.
3. **Amortising the forward transform matters.** If one forward data FFT feeds
   many kernels (as in `accelsearch`), drop its `5·L·log2 L` from the numerator —
   worth roughly a factor of two.

**Caveat, learned the hard way:** this is a flop model and flops are not what
these loops are limited by. In this same project, a per-transform-size benchmark
said smooth `2·3·5·7` FFT lengths were 1.26× faster than power-of-two padding;
in the actual search they were a wash to 7% *slower*, because power-of-two sizing
let 60 harmonics share four scratch buffers that stayed cache-warm. **Use the
model to decide what to measure, never to decide what to ship.**

---

## 4. Where `accelsearch` sits — it is already on the right side

`accelsearch` is the opposite regime on every axis that matters.

**Utilisation is ~1.** `subharm_fderivs_vol()` (`accel_utils.c`) keeps every
point of the valid region: `corr_uselen = fftlen - maxkernlen`, and
`calc_fftlen()` sizes each subharmonic transform so that the good region is
exactly what the search consumes. Nothing is computed and discarded.

**Kernels are wide, and grow with `z`.** `z_resp_halfwidth() = 0.55·|z| + 16`
(LOWACC; `+48` for HIGHACC), and the kernel is
`numkern = 2·numbetween·halfwidth = 4·halfwidth` grid points.

**The forward data FFT is amortised over every kernel.** The data are read,
normalised, spread and FFT'd once per `(subharmonic, r-chunk)`; then every
`(w, z)` kernel is applied as a complex multiply plus one inverse FFT. With
`numzs × numws` in the hundreds or thousands, the forward transform is free.

Applying the rule with `c_d = 8` (Fresnel kernels do not factor) and dropping the
amortised forward transform:

| z | halfwidth | numkern | fftlen (PRESTO) | good frac | FFT flops/pt | direct flops/pt | FFT wins |
|---|---|---|---|---|---|---|---|
| 0 | 16 | 64 | 2048 | 0.97 | 63 | 256 | **4.1×** |
| 50 | 43 | 172 | 2048 | 0.92 | 67 | 688 | **10.3×** |
| 100 | 71 | 284 | 2048 | 0.86 | 71 | 1136 | **16.0×** |
| 200 | 126 | 504 | 2048 | 0.75 | 81 | 2016 | **24.9×** |
| 400 | 236 | 944 | 4096 | 0.77 | 86 | 3776 | **44.0×** |
| 800 | 456 | 1824 | 10240 | 0.82 | 88 | 7296 | **82.6×** |
| 1200 | 676 | 2704 | 10240 | 0.74 | 99 | 10816 | **109.6×** |
| 2000 | 1116 | 4464 | 25600 | 0.83 | 96 | 17856 | **186.1×** |
| 4000 | 2216 | 8864 | 65536 | 0.86 | 100 | 35456 | **356.5×** |

(HIGHACC is worse still for direct: 11× at `z=0`, 113× at `z=1200`.)

> **Conclusion: do not refactor `accelsearch` to direct summation.** The FFT
> correlation is the right algorithm there by one to two orders of magnitude, and
> its margin *grows* with `z` and `w` — i.e. it is most right exactly where
> `accelsearch` spends its time. The CoherentSearch.jl result does not transfer,
> and the reason it does not transfer is precisely the difference Scott
> identified up front: all calculated values are used, and the kernels are wide.

### 4.1 Two more things the model says about existing PRESTO choices

**`fftlen_from_kernwidth()` is already close to optimal.** Minimising
`(5·L·log2 L + 6·L)/(L - K)` over candidate lengths and comparing with PRESTO's
table gives a shortfall of **1.00–1.12×** across `z = 0…4000` — the largest gap
being ~12% at `z ≈ 1200`, where the model prefers 32768 to PRESTO's 10240. Given
that the table's comment says it was measured on FFTW 3.3.7 with "max throughput
of good correlated data" as the metric — the right metric — that agreement is a
good sign, and it means re-tuning is a small, bounded win rather than a big one.
It is still worth re-deriving on current hardware and FFTW (see §5.4).

**Direct summation is already used where it belongs.** `rz_interp()` in
`rzinterp.c` says it plainly: *"It does the correlations manually. (i.e. no
FFTs)"*. Candidate refinement evaluates a handful of points, so utilisation
collapses and direct wins — which is exactly what the rule predicts. No change
needed; this is a data point that PRESTO's existing split is the correct one.

---

## 5. What *is* worth benchmarking in `accelsearch`, in priority order

The honest headline of the CoherentSearch.jl work is not "direct beats FFT". It
is that **the profile inverted the expected answer twice** — first the detection
metric turned out to dominate over interpolation, then interpolation over the
metric, then the metric again. Every refactor that paid off here was chosen from
a profile, and the one that did not pay off (smooth FFT lengths) was chosen from
a microbenchmark. So:

### 5.1 Profile it properly first (do this before anything else)

Nothing below is worth doing on the strength of the flop model alone. What is
needed is a self-time breakdown of `accelsearch` on a realistic search, split at
minimum into:

- the complex multiply loop in `subharm_fderivs_vol()`,
- the inverse FFTs,
- the forward data FFT + normalisation (`get_fourier_amplitudes`, median/local-power),
- the plane search / harmonic summing (`search_ffdotpows`, `add_ffdotpows`),
- candidate handling and I/O.

Do it at (at least) two configurations — a plain `-zmax 200` run and a big
`-zmax 1200` (and a `-wmax` run) — because the balance certainly moves with `z`,
just as ours moved with `--maxdecim`. `perf record` on the real binary is the
cheapest route; the OpenMP region needs `--call-graph dwarf` or per-thread
attribution to be meaningful.

**Prediction to test:** at large `zmax` the multiply loop is memory-bound and a
larger share of runtime than expected. Its arithmetic intensity is 6 flops per 24
bytes = **0.25 flops/byte**, and the kernel array is far too large to cache (at
`zmax=1200`, `numzs ≈ 1201` kernels × 10240 complex × 8 B ≈ **98 MB**, streamed
in full for *every* r-chunk). If that prediction holds, the ranking below is
right; if the inverse FFTs dominate instead, jump to §5.3/§5.4.

### 5.2 Kernel memory traffic and loop blocking

If §5.1 confirms the multiply loop is bandwidth-bound, the lever is not faster
arithmetic but **touching the kernel array fewer times**. Currently the loop
nest is `r-chunk` (outer) → `(w, z)` (inner), so all ~98 MB of kernels stream
from DRAM once per r-chunk.

The obvious inversion — block several r-chunks so each kernel is loaded once per
block — is *not* free: `ffdot->powers` is `numws × numzs × numrs` floats
(≈36 MB per plane at `zmax=1200`), so holding `R` planes multiplies an already
memory-constrained structure. Two variants worth measuring before committing:

- **Tile over `(w,z)` instead**, keeping a cache-resident tile of kernels while a
  few forward-FFT'd data buffers (only `fftlen × 8 B` each — cheap) stream past.
- **Fuse the multiply into the inverse FFT input stage** so `tmpdat` is never
  materialised, removing one full `fftlen`-sized write and read per `(z,w)`.

Measure the achieved bandwidth first (`perf stat -e` on memory events, or just
bytes-moved ÷ time) to see how much headroom there actually is.

### 5.3 Batch the inverse FFTs

Every inverse FFT in an r-chunk is the same length, and there are `numzs × numws`
of them. They are currently issued one at a time via `fftwf_execute_dft()` inside
an OpenMP loop. FFTW's guru/advanced interface can plan the whole set as one
batched transform, which amortises twiddle-table traffic and gives FFTW a better
shot at streaming. This is measurable **in isolation** — a standalone harness
comparing `n` separate `fftwf_execute_dft` calls against one batched plan at
`n ∈ {16, 64, 256, 1024}` and `L ∈ {2048, 4096, 10240, 25600}` — before touching
`accelsearch` at all. Cheap to test, and it interacts with §5.2 (batching also
improves the multiply loop's access pattern).

Note this cuts against the OpenMP parallelisation, which currently gets its
parallelism from the `(w,z)` loop; a batched plan would need the parallelism to
move outward (over r-chunks or over batch sub-blocks).

### 5.4 Re-derive `fftlen_from_kernwidth()` on current hardware

The table's own comment dates it to FFTW 3.3.7 on an AVX processor. The metric it
was built with — max throughput of *good correlated data*, i.e.
`(L - K)/time(L)` — is exactly right, so this is a re-measurement rather than a
redesign. The model in §4.1 says the available gain is ~1–12%, largest at large
`z`, so treat it as a tidy-up. Worth doing as a by-product of §5.3, since the
same harness measures it.

While there: `next_good_fftlen()`'s list already includes non-powers-of-two
(192, 384, 768, 1280, 5120, 7680, 10240, 12288, 15360, 25600). That is the same
question CoherentSearch.jl got wrong by measuring per size instead of in situ —
so re-measure it **end to end in `accelsearch`**, not on isolated transforms.

### 5.5 Redirect the HPK investigation here

`~/programming/fft_tests/HPK_JULIA_HANDOFF.md` evaluated HPK as a faster FFT
backend for CoherentSearch.jl and concluded it was worth prototyping. After the
direct-interpolation change, CoherentSearch.jl no longer justifies it: FFTW fell
from 49.7% to 12.9%, and what remains is *batched short real* transforms, which
are not the configuration HPK's advantage was measured on.

**`accelsearch` is a much better target, on every criterion in that document:**

- **Transform shape matches the benchmark exactly.** HPK was measured on 1-D
  complex64, out-of-place, batch 1, single thread, N ≈ 1024–65536 — which is
  precisely `accelsearch`'s inverse FFTs. The measured advantage there was
  **1.46× median on HPK-native lengths**.
- **§5 of that document — "the decisive question is whether the code chooses its
  FFT lengths or has them forced by the data" — is answered: it chooses them,
  from a table.** `next_good_fftlen()` and `fftlen_from_kernwidth()` are literally
  lookup tables, so restricting to HPK-native lengths costs nothing. That was the
  main technical risk and it evaporates.
- **The FFT fraction is plausibly large**, which was prerequisite §7.1. §5.1
  measures it. If the inverse FFTs are (say) 50% of runtime, 1.46× on them is
  ~1.3× overall — a real win, and much bigger than the 4% it would have been in
  CoherentSearch.jl.

The licensing analysis in §8 of that document (never redistribute HPK binaries or
headers; never ship a precompiled shim; ship source and let users install HPK;
flag the GPL-vs-proprietary tension and the patent clause to AUI/NRAO) applies
unchanged and is, if anything, more pressing for PRESTO, which has far more
users. **Nothing here should be started before that conversation happens.** KFR
remains ruled out on speed (median 0.48× vs FFTW).

Note the ordering dependency: if §5.2 shows the code is memory-bound in the
multiply, a faster FFT library buys less than its headline number.

### 5.6 What not to bother with

- **Direct summation in the correlation** — §4. Wrong by 4–357×.
- **Reduced precision.** PRESTO is already `fcomplex` (float32) throughout. The
  equivalent question in CoherentSearch.jl (Float32 weights/accumulator) bought
  1.2× on ~19% of runtime for nine lost digits and was declined; there is nothing
  analogous left to harvest in `accelsearch`.
- **Porting the real-weight kernel factorisation.** It applies only to the pure
  sinc (`z = w = 0`) response, and only to the *application* of the kernel, which
  `accelsearch` does by FFT. §2.1.

---

## 6. Reproducing the CoherentSearch.jl measurements

```sh
# Interpolator throughput: points/sec vs m, grid oversampling, and request size,
# for direct / direct+table / FFT+linear / FFT-per-grid-point, plus accuracy.
# Writes bench/interp_bench_{throughput,crossover}.png and bench/interp_bench.csv
julia --project=bench -t 1 bench/interp_bench.jl FILE.fft

# Trial grid, chunking and interpolation phase-cycle lengths
julia --project=. bin/coherent_search.jl --verbose ... FILE.fft

# End-to-end (the FFT-correlation arm was retired from the search on 2026-08-16;
# `bench/interp_bench.jl` still measures both, through the reference kernel)
julia --project=. -t 1 bin/coherent_search.jl --threshold 6 --maxdecim 6 \
      --lofreq 5 --hifreq 30 --noplot FILE.fft

# Hot-loop self-time by bucket
julia --project=bench -t 1 bench/profile_search.jl FILE.fft
```

The `accelsearch` cost table in §4 is a flop model with the parameters read from
`presto/src/responses.c` (`z_resp_halfwidth`, `w_resp_halfwidth`),
`presto/src/corr_prep.c` (`next_good_fftlen`, `fftlen_from_kernwidth`) and
`presto/src/accel_utils.c` (`calc_fftlen`, `init_kernel`,
`subharm_fderivs_vol`). It has not been checked against a real `accelsearch`
profile — that is §5.1, and it is the first thing to do.
