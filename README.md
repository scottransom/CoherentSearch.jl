# CoherentSearch.jl

A pure-Julia pulsar search that uses the **phase** of every harmonic, not just
its power. Working from a PRESTO-style `.fft` file, it reconstructs the pulse
profile at each trial spin frequency — Fourier-interpolating the complex
amplitudes of up to 60 harmonics of that frequency and inverse-transforming them
— and matched-filters each profile against a bank of boxcars. Keeping the
harmonic phases is what separates this from the incoherent harmonic sum most
FFT-based searches use: the profile comes back with its *shape* intact, so the
search can tell a pulsar-like pulse from an arbitrary pile of harmonic power.

What comes out is close to what a fast folding algorithm (FFA) computes, but it
is reached through one long FFT rather than through repeated partial sums. That
difference is the point. The work factors into dense, regular, independent
kernels — interpolate, transform, filter — which vectorise, thread, and port to
a GPU cleanly, none of which the FFA's recursion does easily.

In short:

- **More sensitive.** In an injection Monte Carlo over 76,105 white-noise
  realisations (plus 84,871 with red noise), with every code's threshold
  matched to the same measured false-alarm rate, we detect **76%** of
  white-noise injections. riptide's
  `rseek` detects 71% in its deepest configuration (which costs ~6x our
  runtime) and 50% in the configuration matched to our frequency coverage;
  PRESTO's `accelsearch` detects 42%. See `docs/comparison_points.md`, and read
  its caveats before quoting any of this.
- **A calculable false-alarm rate.** The input FFT is normalised, so the noise
  in every reconstructed profile is known in closed form rather than estimated,
  and the boxcar template is normalised so that each (phase, width) trial is
  `N(0,1)`. One `--threshold` therefore means one false-alarm rate at every fold
  depth and across the whole band — which is what makes it safe to lower it.
- **Resilient to red noise.** Searching a whitened FFT keeps the matched
  threshold flat — 6.75 on white noise, 6.70–6.75 out to a 50 Hz red-noise knee.
  riptide detrends in the time domain instead, and over the same range its
  matched threshold climbs from 7.6 to 250.
- **Fast.** Single-threaded it is **1.4–2.2x** as fast as `rseek` over matched
  frequency coverage on three machines. It scales ~27x across 48 cores, and the
  whole search runs on a GPU: an L40 or an A100 is ~9x a 20-core Xeon, an RTX
  A4000 ~3.3x.
- **Pinned, not eyeballed.** Every numerical result is cross-validated against
  the original Python [`coherent_search`](../coherent_search) package used as an
  independent oracle (~1e-16 relative), the optimised search is pinned against
  an unoptimised reference path inside this repo, and a change that should not
  move results is checked by `diff` on the candidate file.

`bin/toy_coherent_search.jl` is the same algorithm with **every optimisation
removed** — brute-force per-point interpolation, one inverse FFT per fold, the
boxcar filter straight from its definition, plain nested loops. It is a complete,
working search, roughly 150–250x slower than the production code, and it exists
to be *read*: it is what the paper's pseudo-code figure describes, line for line.
Start there if you want to understand the algorithm. See
[The toy search](#the-toy-search).

## References

- Fourier interpolation: Eqn. 30 of Ransom, Eikenberry & Middleditch (2002),
  <https://arxiv.org/pdf/astro-ph/0204349>
- The boxcar matched filter: Morello et al. (2020), MNRAS 497, 4654, §5.4
- PRESTO: <https://github.com/scottransom/presto>
- riptide (the FFA we benchmark against):
  <https://github.com/v-morello/riptide>

## Layout

```
src/
  CoherentSearch.jl   module + public API
  fourierinterp.jl    reference interpolation kernels (indexing-critical code)
  directinterp.jl     the production interpolator: tabulated Eqn.-30 weights
  fileio.jl           PRESTO .fft / .inf readers (mmap)
  search.jl           chunk-parallel coherent harmonic-summing search
  candidate.jl        per-candidate high-accuracy profile reconstruction
  backend.jl          CPU/GPU backend dispatch (backendtypes.jl, wisdom.jl)
  cli.jl              ArgParse command-line driver (`CoherentSearch.main`)
ext/
  CoherentSearchCUDAExt.jl  the GPU search (a weak dependency on CUDA.jl)
bin/
  coherent_search.jl  command-line entry point (a shim onto src/cli.jl)
  toy_coherent_search.jl  the same search with every optimisation removed
  parallel_search.py  fill a machine: partition a file list over cores or GPUs
  plotting.jl         CairoMakie candidate-profile plotting (loaded on demand)
  plot_candidates.jl  standalone: re-plot profiles from a saved candidate file
  sift_candidates.py  cross-observation candidate sifter (.cohout / .txt)
test/                 unit tests (golden values, analytic signals, indexing)
crossval/             Python-as-oracle accuracy + speed cross-validation
compare/              head-to-head benchmark against riptide's rseek
mc/                   the injection Monte Carlo: sensitivity vs. the other codes
bench/                microbenchmarks and phase timings (its own environment)
docs/                 design notes, the GPU log, and the measurement record
sysimage/             optional PackageCompiler sysimage for production runs
```

## Design notes

The code is built around four goals, roughly in this order: **sensitivity**, a
**calculable false-alarm rate**, **resilience to red noise**, and **speed**.
Nearly every non-obvious choice below follows from one of them.

### Sensitivity

- **Coherent harmonic summing.** Harmonic amplitudes are summed as complex
  numbers, not as powers, so the inverse transform of the stack is the actual
  pulse profile. Discarding the phases — what an incoherent sum does — throws
  away the pulse shape and with it the ability to reject a detection that is
  not pulsar-like.
- **Exact Fourier interpolation.** A pulsar almost never sits on an integer
  Fourier bin, and a harmonic's phase rotates a full turn between bins. The
  Eqn.-30 kernel is evaluated *exactly* at each trial frequency rather than
  interpolated from a precomputed fine grid, which costs nothing (the weights
  tabulate) and is worth ~1e-10 accuracy against the fine grid's ~1e-2.
- **`--nharms 60`, i.e. 120 profile bins.** Enough to resolve duty cycles below
  1%, which is where the narrowest real pulsars live.
- **A harmonic-summing ladder, not one fold depth.** Taking every `k`-th
  interpolated amplitude folds the same data at `k` times the trial frequency
  into `2·nharms/k` bins, for the price of one short inverse FFT. That covers
  fast pulsars for nearly free — and the rungs below `hifreq` overlap on
  purpose. They look redundant and are not: a wide pulse is better detected in a
  shallow fold, and restricting each `k` to a disjoint band models out ~8% of
  recovered S/N. Do not "optimise" the overlap away.
- **A boxcar bank set by each profile's own length**, geometrically spaced out
  to 30% duty cycle, so every fold depth is filtered to the same duty cycles.

### A calculable false-alarm rate

- **A normalised template.** Each boxcar is made zero-mean and unit-L2 before
  correlating, so every (phase, width) trial is `N(0,1)` under white noise and
  the distribution of the peak is analytic. This is numerically identical to
  riptide's `snr1`, so the two codes report the same quantity.
- **Held-at-zero DC.** The zero-frequency bin is forced to zero, so every
  profile's mean is *exactly* zero. There is no mean to estimate, which removes
  a term from the statistic's variance.
- **An analytic noise scale.** For a normalised input FFT the per-bin noise of
  the reconstructed profile follows from the FFT normalisation alone —
  `σ = sqrt(2·nlow + 0.5·nnyq)/nbins` — so it is computed, not measured. That is
  both faster and *more* accurate than the subsampled robust estimator it
  replaced (3.0% spread against 5.4%), and it removes a ~1% per-chunk noise term
  that used to land directly on every reported S/N.
- **Renormalised for a band-limited fold.** Profile bins from an inverse FFT of
  a truncated harmonic stack are correlated, so the textbook `snr1`
  normalisation is wrong for them. Dividing by the exact variance of the boxcar
  sum fixes two biases: a `sqrt(nbins/2H)` inflation past the Nyquist knee (on
  pure noise the peak ran 9.09 at 500 Hz against 5.32 at 20 Hz; now 5.01 and
  5.28) and a ~3.4% offset *between* ladder rungs that made shallow folds win
  spuriously.
- Together these are why the threshold can sit near 6.7 and stay there.

### Resilience to red noise

- **Whiten first.** The input is a PRESTO `rednoise`-flattened `.fft`, which
  makes the unit-variance assumption above true rather than hopeful. Codes that
  detrend in the time domain instead cannot whiten above `1/rmed_width`, and
  their false-alarm rate then depends on the red-noise knee — which is the
  single largest effect in the Monte Carlo.
- **An escape hatch, and a guard.** `--sigma measured` estimates the noise from
  the data instead, which is the right answer when the noise level varies with
  frequency (residual red noise, an RFI comb, a `rednoise` pass that did not
  take). And because the analytic assumption fails *silently* and in the
  dangerous direction — a normalisation error inflates every S/N — the search
  scores a few chunks both ways and warns when they disagree by more than 10%.
- Skipping the de-reddening step is not an option, and we measured that on our
  own default path: detection falls 76% → 1% across the knee range.

### Speed

- **Regular kernels, then vectorise along the long axis.** The interpolator
  vectorises across *trials* (a group of consecutive trials becomes a
  matrix-vector product against one contiguous slice of Fourier bins — no
  gather, no horizontal reduce), and the boxcar scan vectorises across
  *profiles*, 128 at a time, since the phase axis is only 20–120 long. Both
  choices are worth 1.5–4x on their phase.
- **Chunk-parallel, whole chunks per thread.** Trials are grouped into chunks of
  2048 and handed to tasks round-robin, each with a private workspace. That is
  what lets a harmonic's Fourier-bin window be loaded once per chunk and read
  back from L1 by every trial in it. ~9x on 20 cores, ~27x on 48.
- **A GPU extension.** The whole search runs on a CUDA card
  (`ext/CoherentSearchCUDAExt.jl`); CUDA is a weak dependency, so a CPU-only
  user downloads nothing. Eight cards have been measured, 6 to 142 SMs, all
  reporting the same candidates as the CPU.
- **Start-up is a real cost and is treated as one.** Julia's JIT once dominated
  short runs (15.6 s wall for 1.4 s of searching). A precompile workload plus
  keeping the CLI inside the package cut that to 2.4 s, and one invocation
  searches many `.fft` files while sharing plans and workspaces, so the marginal
  cost of an extra file is ~1 s.
- **Measured, not guessed.** Permanent in-situ phase timers, a `bench/`
  environment, and a long record of projections that turned out wrong live in
  `docs/Summary_and_Future_Work.md` and `docs/gpu_design.md`. Microbenchmarks
  have inverted in situ often enough here that the phase timers are the primary
  instrument.

### Implementation notes

- **Indexing.** Python is 0-based with half-open slices; Julia is 1-based with
  inclusive ranges. The translation is isolated and documented in
  `fourierinterp.jl` (see `nearby_fourier_bin_range`), and pinned by tests and
  the cross-validation to machine precision.
- **FFT conventions.** `irfft` of the stacked harmonic amplitudes matches
  numpy's `np.fft.irfft` (both ignore the imaginary parts of the DC/Nyquist
  bins); this is verified directly in the tests.
- **Two paths that must agree.** A deliberately unoptimised *reference* path
  (`block_metrics` / `reference_profiles`) is pinned to the Python oracle at
  ~1e-16, and the whole optimised machinery is pinned to that reference at
  8.4e-16. Every optimisation has to keep both green.

## Installation and first use

If you've never used Julia before, install it using either your systems package
manager, or via a method from <https://julialang.org/downloads/>.
On Linux or Mac, the following should work:

```sh
curl -fsSL https://install.julialang.org | sh
```

Clone this repo, and cd into the top-level directory. Then do:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

That will install all of the Julia requirements and compile them. That will
likely take several minutes. When it is complete, you can run the tests and
programs as described below.

## Usage

Run the CLI (use `-t auto` so Julia uses all cores):

```sh
julia --project=. -t auto bin/coherent_search.jl FILE.fft
```

The defaults are a full blind search: fundamentals `0.1–125 Hz` with
`--nharms 60`, plus `--maxdecim 6`, which carries coverage to **750 Hz** in spin
frequency — past the 716 Hz of the fastest known pulsar, with headroom — folding
120 profile bins at the low end down to 20 at the high end. Plotting is off;
pass `--plot` for it.

The profile stage runs in `Float32` by default (`--precision f32`): the
interpolated harmonic amplitudes, the batched inverse FFT and the folded
profiles the metric reads. Everything reported — candidate frequencies, the S/N
metric, the normalisation — stays `Float64`. This is worth ~1.2× at every thread
count on both development machines and costs ~1e-7 in the profiles, five orders
of magnitude under the ~1.3% of signal power the `m = 16` interpolation
truncation already discards. Pass `--precision f64` to reproduce a run made
before 2026-08-24, or when a candidate's S/N must be bit-comparable with the
reference path.

Or from Julia — the same search the CLI runs by default, spelled out. The
library primitives keep their own minimal defaults (`SearchParams()` is
`nharms = 32`, no decimation); the survey policy above lives in the CLI, so
state it explicitly when calling `search` directly:

```julia
using CoherentSearch
ft = FFTFile("FILE.fft")
params = SearchParams(nharms=60, decimations=decimation_set(60, 6), threshold=8)
cands = search(ft, params; lofreq=0.1, hifreq=125, threshold=8)
```

Each candidate reports its barycentric spin frequency, period (`1/f`), the S/N
metric, and the number of harmonics summed in the detection.

### Searching many files at once (and start-up cost)

**Pass every `.fft` file to a single invocation** rather than running the CLI
once per file:

```sh
julia --project=. -t auto bin/coherent_search.jl *_red.fft \
    --threshold 8
```

Julia compiles the search on first use, which costs ~10 s of wall-clock before
any work happens — comparable to the search itself on a short observation. One
invocation pays it once for the whole batch, and the harmonic plans, FFTW plans
and per-thread workspaces (a [`SearchCache`](src/search.jl)) are built once and
reused, so each additional file costs only its own search time. Measured on a
32 MB `.fft`, single-threaded: one file 2.4 s, three files 4.8 s — i.e. ~1.2 s
per extra file against 15.6 s for a separate invocation each.

Output naming follows from this:

| inputs | `-o` / `--outdir` | candidates go to |
|---|---|---|
| one file | neither | stdout |
| one file | `-o NAME` | `NAME` |
| one file | `--outdir D` | `D/<base>.cohout` |
| many files | neither | `<fftfile without .fft>.cohout`, beside each input |
| many files | `--outdir D` | `D/<base>.cohout` |
| many files | `-o NAME` | rejected — it would have each file overwrite the last |

`bin/sift_candidates.py` reads `.cohout` (and `.txt`) files, so a whole DM sweep
can be sifted with `sift_candidates.py <dir>`.

**Plotting is off by default, and deferred when enabled.** With `--plot`, all
searches finish first and CairoMakie is loaded *once* to plot every file's
candidates. Loading it costs ~9 s plus first-call compilation — by far the
largest fixed cost in the program — and the deferral pins every input's mmap
until the end of the run, so bulk runs should leave it off and plot later from
the saved candidate files with `bin/plot_candidates.jl`.

Searching several files from Julia? Share one `SearchCache` (and one
`SearchParams` object — reuse is keyed on its identity):

```julia
params = SearchParams(nharms=60, decimations=decimation_set(60, 6))
cache = SearchCache()
for f in files
    cands = search(FFTFile(f), params; cache=cache, lofreq=0.1, hifreq=125)
end
```

### Using a whole machine: `bin/parallel_search.py`

One invocation uses **one CPU thread per chunk-parallel worker, and exactly one
GPU**. To fill a many-core box or a multi-GPU node, run several independent
invocations over disjoint slices of the file list — which is what this does:

```sh
# 6 GPUs, one job per card
bin/parallel_search.py --gpu -j 6 NGC6624_*_red.fft -- --blocksize 262144

# a whole CPU socket, one single-threaded process per core
bin/parallel_search.py -j 20 --outdir cands *.fft -- --threshold 8

# resume after an interruption
bin/parallel_search.py --gpu -j 6 --skip-existing --outdir cands *.fft
```

Everything after `--` is passed to `bin/coherent_search.jl` unchanged. Add `-n`
to print the commands without running them.

It **partitions** the file list rather than running one process per file: the
whole point of the section above is that start-up (and, on the GPU, the ~6 s
CUDA load plus the cached device workspace and cuFFT plans) is paid once per
*invocation*, so `parallel ::: *.fft` throws all of it away. Files are dealt to
the least-loaded job by size, largest first, so a heterogeneous glob does not
leave one job running long after the others finish; `--split block` keeps each
job's slice contiguous instead.

With `--gpu` each job gets its own device through `CUDA_VISIBLE_DEVICES`
(`--gpus 0,2,3` to choose); there is no multi-GPU support *inside* a single
invocation. On the CPU, `-t 1` and one job per core is the deployment model the
performance work targets — see `docs/Summary_and_Future_Work.md` §3.1.

One sharp edge it handles for you: a job that happened to receive exactly one
file would write its candidates to **stdout** per the table above. When that can
happen and no `--outdir` was given, the launcher supplies one, so every input
gets its `.cohout` regardless of how the list was split.

### Multi-frequency search by harmonic decimation

`--maxdecim k` (default `6`; `1` disables it) additionally folds every trial
fundamental at `2×, 3×, … k×` its frequency *almost for free*, by re-using the
harmonic amplitudes already interpolated for the base fold: taking every `k`-th
harmonic and running a shorter inverse FFT yields the fold at `k·rf` with
`⌊nharms/k⌋` harmonics. This extends the search to faster pulsars (which tend to
have wider profiles and so need fewer harmonics) without paying for extra
interpolation. `nharms` defaults to a composite `60` so that `k = 2,3,4,5,6` all
give clean integer harmonic counts. The harmonic count printed for each candidate
identifies the decimation that found it (`k = nharms ÷ nharm`).

**This is what sets the top of the searched band.** `--hifreq` is the highest
*fundamental*; decimation carries coverage to `--hifreq × --maxdecim`. The
defaults (`125 × 6`) reach 750 Hz. Raising `--hifreq` alone is usually the wrong
move — it buys the same coverage at far more cost, since every extra fundamental
is a full 60-harmonic interpolation while a decimation is nearly free.

```sh
# Fundamentals 0.1–200 Hz, and via decimation spin frequencies to 1200 Hz
julia --project=. -t auto bin/coherent_search.jl FILE.fft \
    --hifreq 200 --maxdecim 6 --threshold 8
```

> **A note on the harmonic depth.** At `--nharms 60`, a signal whose harmonic
> content extends well past harmonic 60 can be detected *more* strongly at one of
> its own harmonics than at the fundamental — folding at `11f` puts the same
> absolute pulse width across a proportionally coarser phase grid, which the
> boxcar bank matches better. Harmonic collapse keeps the strongest member of the
> family, so such a signal is reported at that harmonic. It shows up on
> narrow-duty synthetic tests; on real observations, whose harmonic content is
> bounded by scattering and finite time resolution, the fundamental wins.

See `docs/decimation_design.md` for the derivation that decimation stays correctly
sampled (each `k`'s top harmonic still steps by ≤ `hidr`, and the base input-FFT
read depth already covers every `k`) and the full bookkeeping.

### The noise scale: analytic by default

The S/N metric divides by a per-bin noise scale `σ`, and as of 2026-08-24 that
scale is **computed rather than measured** (`--sigma analytic`, the default).

The search is only meaningful on a normalised `.fft` — Fourier powers with mean
1 — and that assumption already fixes the fold's noise. Mean power 1 means the
real and imaginary part of every amplitude have variance ½, so the hot loop's
unnormalised `brfft` of a stack of `H` harmonics with DC held at zero gives

```
σ = sqrt(2·nlow + 0.5·nnyq)
```

where `nlow` counts the stacked harmonics below the profile's own Nyquist bin
and `nnyq` is 1 if that bin carries data (halved because the transform keeps only
its real part). That is `sqrt(nbins)` times a `sqrt(1 − 3/(4H))` correction — 0.6%
at `H = 60` but **3.8% at the `H = 10` of a `k = 6` fold**, so it is not
decoration: omitting it would bias the shallow folds against the deep ones.
Harmonics past Nyquist are zero rows and carry no noise, so the *fill count*, not
the stack length, is what enters.

This replaced a robust MAD estimated per chunk, and it is both faster and more
accurate:

| | `--sigma measured` | `--sigma analytic` |
|---|---|---|
| metric-phase share of runtime | 27.0% | 22.9% |
| wall clock, `-t 1` | 8.91 s | **8.29 s** (1.075×) |
| wall clock, `-t 4` | 4.14 s | **3.94 s** (1.053×) |
| agreement with the exact pooled MAD | 0.981–1.033 (5.4%) | **0.992–1.022 (3.0%)** |

(PM0063, 0.1–33.3 Hz, laptop, median of 7 interleaved reps; the agreement row is
from `bench/toy_vs_production.jl` over four frequency windows and all six fold
depths.) The last row is the one that matters: **the closed form is closer to the
exact noise scale than the subsampled estimator it replaces**, which carries ~1%
sampling error straight into every reported S/N — reported S/N is exactly `1/σ`.

**When to pass `--sigma measured` instead.** The closed form has exactly one
assumption, and cannot see it fail. If the noise level varies with Fourier
frequency — residual red noise, an RFI comb, a `rednoise` pass that did not take
— the measured estimate adapts and the analytic one does not, so the analytic S/N
is inflated wherever the real variance is higher. That trade is a real estimation
error (~1%) against an unmodelled bias, and on a badly-behaved observation the
bias wins.

Because that failure is silent and inflates S/N (a candidate list full of noise
rather than an empty one), `search` **checks it**: three chunks spread across the
band are scored both ways, and a disagreement over 10% produces a warning naming
both numbers. It costs ~0.1% of the runtime. On an un-normalised input the check
fires immediately — the raw test fixture is out by a factor of ~1000.

### Candidate de-duplication

Two collapses run on the candidate list, both on by default:

- **Near-identical** (`remove_duplicates`, `--noremove`, `--drtol`): the run of
  adjacent trial fundamentals a single signal lights up, grouped by Fourier
  frequency `r` within `--drtol` bins, reduced to the strongest member.
- **Harmonically-related** (`remove_harmonics`, `--noharmremove`, `--numharm`):
  the `f/2`, `2f`, `3f/2`, … family a real signal (and its decimation folds)
  produces at genuinely different `r`. Candidates whose frequencies form a
  ratio `n/m` of small integers (up to `--numharm`) are collapsed to the
  strongest member. Decimation makes this family especially prominent, so the
  two work together.

### Candidate output format

```
#Num    'S/N'      Frequency (Hz)        Period (ms)    #Harm  Ducy(%)
1        11.92      7.118536329269    140.478316573069    32    14.06
```

**The first column is the rank, not the S/N** — so in `awk` the S/N is `$2` and
the frequency is `$3`.  (The header used to omit the rank column, which made
`$2` look like the frequency.  It is not.)

`#Harm` is the harmonic count that found the candidate (`k = nharms ÷ #Harm`
identifies the decimation). `Ducy(%)` is the duty cycle of the best-fitting
boxcar — `width / profile bins`, exactly as riptide's `rseek` defines `ducy`, so
the two searches can be compared directly. It is `-` when unmeasured.

The search's hot loop deliberately discards *which* boxcar width won (it runs
~1e8 times and only reported candidates need it), so the width is recovered
afterwards by refolding each reported candidate — see `measure_ducy`. This is
exact, not an approximation: the noise scale σ multiplies every width's score
equally and so cannot change which one wins, which is what lets the width be
recovered from an isolated profile.

### Candidate profile plots

With `--plot`, the CLI reconstructs and plots the pulse profile of every reported
candidate, one grid of panels per US-Letter portrait page, written as
zero-padded PNGs (`<stem>_01.png`, `<stem>_02.png`, …).  It is **off by default**:
CairoMakie costs ~9 s to load, and because plotting is deferred to the end of the
run it keeps every input's mmap live until then, which a bulk pipeline does not
want.  (`--noplot` is still accepted and ignored, so existing scripts keep
working.)  Each profile is folded
with a high-accuracy exact-interpolation path (independent of the throughput-
tuned search) and rotated so its peak sits at phase 0.5; the panel caption
carries the full text-line information (index, S/N, frequency, period, harmonic
count, and the decimation `k`).  Plotting loads CairoMakie lazily — searches
without `--plot`, the test suite, and the cross-validation never load it.

Every profile is folded at the **full `--nharms` harmonic depth**, regardless of
the harmonic-decimation factor `k` that found the candidate (a `k=3` detection
summed only `nharms/3` harmonics, but its profile still uses all `nharms`): this
much more closely matches a true time-domain fold of the time series at the
candidate period.  Harmonics that would exceed the Nyquist frequency are simply
omitted (not zero-padded), so a fast candidate's profile uses fewer bins.  The
`#Harm`/`k` in each caption still report what the *search* summed to detect it.

```sh
julia --project=. -t auto bin/coherent_search.jl FILE.fft -o cands.txt \
    --plot --plotstem cands --plotcols 3 --plotrows 5     # -> cands_01.png, ...
```

The same plots can be regenerated later from a saved candidate file, without
re-running the search:

```sh
julia --project=. bin/plot_candidates.jl FILE.fft cands.txt --nharms 60
```

### Progress meter

The CLI prints a chunk-completion meter to `stderr`: a text percentage by
default, a bar with `--progressbar`, or nothing with `--noprogress`. From the
library, pass `progress = :text | :bar | :none` to `search`.

## Start-up time

Julia compiles on first use, and for a short search that compilation dominated
the wall clock: 15.6 s for a run whose actual searching took 1.4 s. Two things
address it, and a third is available for production.

1. **A precompile workload** (`PrecompileTools`, at the bottom of
   `src/CoherentSearch.jl`) runs a miniature end-to-end search at package
   *build* time, so the native code is cached in the package image. The CLI
   driver lives in `src/cli.jl` rather than in `bin/` for exactly this reason —
   as a top-level script, inferring `main` alone cost ~4.7 s per run. Together
   these took the run above to **2.4 s**. The cost is ~3.4 s of extra
   precompilation after each `src/` edit.
2. **Batching files** into one invocation (see above) amortises what remains.
3. **A sysimage** (`sysimage/`) removes CairoMakie's load — and, as measured,
   nothing else:

   | | no sysimage | sysimage |
   |---|---|---|
   | no `--plot` | 2.4 s | 2.3 s |
   | with plots | 18.7 s | **7.4 s** |

   ```sh
   julia --project=sysimage -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
   julia --project=sysimage sysimage/build_sysimage.jl        # ~28 minutes
   julia --sysimage sysimage/coherent_search.so --project=. -t auto \
         bin/coherent_search.jl FILE.fft [FILE2.fft ...] [options]
   ```

   Worth it only for repeated plotting-enabled runs: the image is 1.14 GB, and
   the *first* run after building it (or after a reboot) spends ~23 s reading it
   into page cache. If you run without `--plot`, skip it.

   Use it for production runs, **not during development**: a sysimage freezes
   `src/` as of its build, so later edits are silently ignored until you rebuild
   it. Plain `julia --project=.` and `Pkg.test()` always see the live source.
   The image is specific to the machine and the Julia version.

## Comparison against riptide's FFA

The external bar for this search is [riptide](https://github.com/v-morello/riptide),
the Fast Folding Algorithm implementation. `compare/compare_riptide.py` runs
`rseek` and this code over the same observation with matched settings, times
both, and cross-matches the candidate lists:

```sh
python3 compare/compare_riptide.py --repeat 3 --threads 4 FILE.fft
```

It matches the **total frequency coverage** rather than the trial range, which
is the subtle part. Both codes are limited by the same sampling constraint —
riptide requires `P >= tsamp * bins` and downsamples to stay within
`[bmin, bmax]`; our `k`-decimated fold of `nharms/k` harmonics needs its top
harmonic below Nyquist, which is the same inequality. Both therefore reach high
frequency by folding into fewer bins, riptide by downsampling and us by harmonic
decimation. So:

```
nharms   = bmax / 2        maxdecim = bmax / bmin
fmax     = 1 / Pmin        hifreq   = fmax / maxdecim   (our fundamental range;
                                      decimation carries coverage up to fmax)
```

`--bmin 20 --bmax 120` gives `--nharms 60 --maxdecim 6`, both covering
0.1–200 Hz in 120…20 bins. `Pmin` defaults to `tsamp * bmin`, riptide's own
floor, so both run the widest band the data support.

Measured **2026-09-15** on `PM0063_034C1_DM445.0_red.fft` (T=2097 s),
`--preset bench`, both covering 0.1–200 Hz, median of 3, `OMP_NUM_THREADS=1`,
on three machines:

| | `rseek` | ours `-t 1` | like-for-like |
|---|---|---|---|
| i7-10510U (laptop, 4 cores) | 19.92 s | **9.20 s** | **2.16× faster** |
| Xeon Silver 4114 (20 cores) | 20.31 s | **12.89 s** | **1.57× faster** |
| EPYC 7413 (2×24 cores) | 9.91 s | **7.25 s** | **1.37× faster** |

The hosts differ by more than any single optimisation in the code, so quote
the machine and the date with the ratio. (On 2026-08-24 the first two read
2.13× and 1.46×, and `rseek`'s own times agree with today's to 2–6%.)

The harness also splits start-up from searching, so the obvious objection —
that this is really measuring Julia's start-up — is answered on every run. It
is not; if anything start-up works against us, since ours is the larger of the
two on every host and we win anyway.

| | start-up | searching | pure-compute ratio |
|---|---|---|---|
| `rseek` (laptop) | 0.51 s (Python import) | 19.12 s | |
| ours `-t 1` (laptop) | 0.91 s (boot + JIT + FFTW plans) | 8.17 s | **0.43×** |
| `rseek` (Xeon) | 1.07 s | 19.17 s | |
| ours `-t 1` (Xeon) | 1.36 s | 11.50 s | **0.60×** |
| `rseek` (EPYC) | 0.97 s | 8.94 s | |
| ours `-t 1` (EPYC) | 1.14 s | 6.08 s | **0.68×** |

so on every host the pure-compute ratio is at least as good as the wall-clock
one. Note that riptide's `find_peaks` is 29–35% of its compute (6.7 s on the
laptop) and is a separate pass doing candidate work we do inline; comparing
our figure against its `ffa_search` alone would be wrong.

**Single-threaded we are 1.4–2.2× faster, while doing ~2.8× the folds** — the
harness prints that work ratio before it times anything, because the two numbers
have to be read together. We fold every frequency below `hifreq` once per
decimation factor, where `rseek` folds it exactly once; that redundancy is our
harmonic-sum ladder.

**The 7.1185 Hz pulsar reads S/N 11.65 against riptide's 11.80** (at a 10.0%
duty cycle from the `k = 6` fold, against riptide's 6.5% — its width bank is
built from `bins_min`, so it cannot reach this pulse's width at the depth it
folded). Before the 2026-08-28 band-limited renormalisation (see
[Design notes](#a-calculable-false-alarm-rate)) ours read 12.30. We also find a candidate it does not (0.2603 Hz at S/N 7.32).
riptide's two extra entries are the `f/2` and `2f` of the pulsar, which it does
not filter and we collapse by default (`--noharmremove` for a like-for-like
count). All three hosts report identical candidates, as they must — the search
is deterministic.

For a pure algorithm-vs-algorithm timing at *equal* work, use `--preset matched`,
which runs one fold depth on each side and equalises the work to a few percent.

Getting this wrong is easy and expensive: setting our `hifreq` to `1/Pmin` —
the obvious-looking choice — has us search 6× riptide's band and reports us as
2.1× slower, which is an artefact of the mismatch, not a result.

The threading axis is ours alone rather than a like-for-like win: riptide's C
extension is built without OpenMP, so `rseek` cannot use more cores. Measured
with `bench/thread_scaling.jl`, which times only the *warm in-process* search so
that the fixed start-up cost does not contaminate the fit. On the 20-core
workstation (2026-08-24):

![Thread scaling on a 20-core Xeon Silver 4114](docs/thread_scaling.png)

| threads | 1 | 2 | 4 | 8 | 16 | 20 |
|---|---|---|---|---|---|---|
| wall (s) | 11.58 | 6.51 | 3.44 | 1.93 | 1.42 | 1.29 |
| speedup | 1.00× | 1.78× | 3.37× | 6.00× | 8.15× | **9.00×** |

The Amdahl fit gives a serial fraction of 0.065 (ceiling 15.5×). The right-hand
panel is the part worth reading: CPU-seconds for *identical* work inflate 62%
across the sweep, which is memory-stall and clock-throttle time, not a code
defect — and on a dual-socket box past 16 threads the marginal core is also
paying for cross-socket traffic.

On larger machines (2026-09-15/16), speedup at each thread count:

| host | physical cores | work | `-t 1` | 2 | 4 | 8 | 16 | 32 | 48 | best |
|---|---|---|---|---|---|---|---|---|---|---|
| `bla0`, 2× EPYC 7413 | 48 | NGC6624 | 77.7 s | 1.94× | 3.59× | 7.08× | 14.6× | 24.6× | **26.8×** | 26.8× @ 48 |
| OzSTAR `dave41` | 32 (allocated) | NGC6624 | 75.9 s | 1.96× | 3.58× | 6.40× | 14.4× | **24.6×** | 21.6× | 24.6× @ 32 |
| `talanah`, 2× Xeon Silver 4514Y | 32 | NGC6624 | 74.3 s | 1.98× | 3.51× | 7.18× | 11.2× | **16.3×** | 14.2× | 16.3× @ 32 |
| `eiger`, Xeon w5-3433 | 16 | PM0063 | 5.49 s | 1.95× | 3.80× | 6.53× | **10.1×** | 4.70× | 5.03× | 10.1× @ 16 |

"NGC6624" is `NGC6624_16L_DM87.40_red.fft` over 0.1–33.3 Hz (105.5M trial
fundamentals, 42 candidates); "PM0063" is the configuration of the 20-core
table above (3 candidates), where `eiger` is 2.1× faster single-threaded. Fit to
the points up to the physical core count, Amdahl's serial fraction is 0.017 on
`bla0` and 0.020 on `dave41`, against 0.065 on the 20-core Xeon. **Threads
beyond the physical cores lost on all three hosts where that was measured**, and
on `eiger` badly (CPU-seconds
4.4× those at 16 threads), so use `-t` equal to the physical core count, not
`-t auto`, which counts hyperthreads. `talanah` is the weakest of the four
past 8 threads, with CPU-seconds up 69% at 32 threads.

Production searches are often run as one single-threaded process per DM, in
which case the `-t 1` CPU-seconds column governs throughput rather than these
curves.

Reading the output: **the two S/N values are the same statistic** as of
2026-08-24 — both are the peak of riptide's zero-mean unit-L2 boxcar matched
filter (`cpp/snr.hpp:snr1`), verified against the `rseek` binary itself to
1.4e-7 on identical profiles. What still differs is the *profile* each is
computed on (our coherent Fourier fold vs riptide's time-domain FFA fold) and
the σ̂ estimate, so a residual S/N gap is a statement about the folds, not about
the detector. Duty cycles are defined identically on both sides too.

One pulsar in one observation says nothing about relative *sensitivity*, and the
single-detection scatter is much larger than the gap above; that question is
settled by the injection Monte Carlo described in
`docs/Summary_and_Future_Work.md`
§3.2, not by this table.

## The toy search

`bin/toy_coherent_search.jl` is the whole algorithm with none of the
optimisation: brute-force per-point Fourier interpolation, one `irfft` per fold,
the boxcar matched filter evaluated straight from its definition, and plain
single-threaded nested loops. It exists to be *read* — it is the code the
paper's pseudo-code figure describes, line for line, and each function carries
the figure's line numbers.

```sh
julia --project=. bin/toy_coherent_search.jl FILE.fft --lofreq 0.1 --hifreq 0.4
```

It takes the options that set the search itself (`--threshold`, `--nharms`,
`--m`, `--ncands`, `--lofreq`, `--hifreq`, `--hidr`, `--drtol`, `--maxdecim`,
`--sigma`) and writes candidates to stdout. It reuses the production candidate
collapsing and output code unchanged, because that is bookkeeping rather than
search.

**Expect roughly 150–250× slower**, so give it a narrow band. How much depends
on the machine and the band; measured at `-t 1` over 0.1–0.4 Hz of
`PM0063_034C1_DM445.0_red.fft`, two runs of the same command on the laptop gave
190.8× and 177.1×, and the 20-core Xeon gave 200.4× (~243 and ~359 µs per trial
fundamental against production's ~1.27 and ~1.79 µs). Over 0.1–3 Hz on an EPYC
7413 it was 232.3× (198.3 against 0.85 µs).

It differs from the production search in exactly two ways, both deliberate and
both documented in the file: it scans the full geometric width bank rather than
the ladder-pruned one, and it divides by an **analytic** noise scale rather than
a measured one. For a normalised input FFT the folded profile's per-bin noise is
known in closed form,

```
sigma = sqrt(2*nlow + 0.5*nnyq) / nbins
```

where `nlow` counts the stacked harmonics below the profile's own Nyquist bin
and `nnyq` is 1 if that bin carries data — about `1/sqrt(nbins)`, times a
`sqrt(1 - 3/(4H))` correction that is 0.6% at `H = 60` but 3.8% at the `H = 10`
of a `k = 6` fold. Harmonics past Nyquist are zero and carry no noise, so the
count, not the stack length, is what enters.

`bench/toy_vs_production.jl` times the two arms against each other, cross-matches
their candidate lists, and reports the analytic noise scale against the measured
one per fold depth and across the band. `test/test_toy.jl` pins the toy's
interpolation, fold and metric against the oracle-validated reference path, and
pins the analytic noise scale against synthetic normalised white noise.

## GPU support (`--gpu`)

A CUDA GPU can run the whole search, and candidates agree with the CPU path to
~2e-7 (comparable, deliberately not guaranteed bit-identical — see below).

Measured 2026-09-15/16 with `bench/paper_gpu_run.sh` on
`NGC6624_16L_DM87.40_red.fft` (T = 26459 s) over 0.1–133.3 Hz — 423M trial
fundamentals, spin coverage to 800 Hz — at each card's best `--blocksize`,
`-t 1` on the host:

| card | SMs | best `--blocksize` | wall (s) | ns/trial | vs 20-core Xeon |
|---|---|---|---|---|---|
| L40 | 142 | 32768 | **4.30** | **10.2** | **9.25x** |
| A100-SXM4-80GB | 108 | 1048576 | **4.52** | **10.7** | **8.81x** |
| RTX 4500 Ada | 60 | 8192 | 7.30 | 17.2 | 5.45x |
| RTX A4000 | 48 | 524288 | 12.07 | 28.5 | 3.30x |
| GTX 1080 | 20 | 262144 | 29.91 | 70.7 | 1.33x |
| RTX A400 | 6 | 131072 | 61.36 | 145.0 | **0.65x** |

The reference is `fitzroy`'s 2× Xeon Silver 4114 at `-t 40` on the same file
and band: 39.79 s, 94.1 ns/trial. The CPU still wins on a big enough machine: a
32-core Threadripper PRO 7975WX (`-t 64`) takes 9.14 s, and 2× EPYC 7413
(`-t 96`) take 10.02 s, faster than an RTX A4000. Against their own hosts'
physical cores, the L40 is 3.4x a 2× Xeon Silver 4514Y (14.66 s at `-t 32`) and
the RTX 4500 Ada 5.9x a 16-core Xeon w5-3433 (43.04 s at `-t 16`). All six cards give the same
285 candidates as the CPU. The only difference is the last digit of four
printed periods (one ulp of `1/f`), and even that vanishes in a band past this
file's Nyquist knee, where the output is byte-identical.

CUDA is a **weak dependency**: it is not installed unless you ask for it, and a
CPU-only user downloads nothing. The GPU code lives in a package extension
(`ext/CoherentSearchCUDAExt.jl`) that loads only when CUDA is present.

### Installing CUDA.jl

You need an NVIDIA **driver**. You do *not* need a system CUDA toolkit, `nvcc`,
or a module-loaded CUDA — CUDA.jl ships its own toolkit as artifacts and will
use those in preference to anything on the system.

**Install CUDA into a separate environment, not into this repo.** `Pkg.add`
would move `CUDA` out of `[weakdeps]` in `Project.toml` — defeating the whole
point of the extension, since every CPU-only user would then download it — and
resolve the entire CUDA dependency tree into this repo's `Manifest.toml`.

```sh
mkdir -p ~/gpuenv && cat > ~/gpuenv/Project.toml <<'TOML'
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
CoherentSearch = "b7e4a1c2-3d6f-4e8a-9c1b-2a5d8f3e6c40"
TOML
julia --project=~/gpuenv -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=~/gpuenv -e 'using CUDA; CUDA.versioninfo()'   # check it works
julia --project=~/gpuenv bin/coherent_search.jl --gpu FILE.fft
```

`bench/gpu_probe_setup.sh` does all of this for you and picks the location
itself; use it if you would rather not think about any of the above.

The artifacts are ~2.2 GB and land in the Julia depot. If `$HOME` is small or on
slow NFS, point the depot at local scratch first:

```sh
export JULIA_DEPOT_PATH=/fast/local/depot
```

**If you use more than one GPU machine, give each its own environment, and put
it next to that machine's depot.** Watch for a shared `$HOME` in particular — and
note that the same home *path* on two machines is not proof they share, nor
proof they do not. The environment holds a `Manifest.toml`, and
a Manifest pins the exact `CUDA_Runtime_jll` and artifact versions the depot has
to contain — a choice that depends on the host's driver and card. Put one
environment on a shared NFS `$HOME` and two machines will fight over it: whoever
installed last wins, and the other tries to instantiate artifacts its depot has
never seen and fails to precompile CUDA with a missing `.so`. Separate
`JULIA_DEPOT_PATH`s do **not** protect you here, because the depot is not what is
being shared.

On a cluster whose compute nodes are air-gapped, run the install on a login
node that shares the filesystem, then run on the GPU node with the same
environment and `JULIA_DEPOT_PATH` — that case is fine, because it really is one
machine's worth of hardware.

### Tune `--blocksize` for your card — it is worth up to ~1.25x over the default

**Not urgent any more, but still worth one run.** `--blocksize` (trial
fundamentals per chunk) defaults to **65536 under `--gpu`** and 2048 on the CPU;
the two defaults are 32x apart. 65536 is one constant chosen
for its worst case — it is within **1.24x** of the optimum on the cards we have
measured, spanning 6 to 142 SMs and 1 to 96 MB of L2 — and it is not a
per-device rule, because the optimum is *not* predictable from the hardware: the
A100 and the RTX 4000 Ada have the same 40 MB of L2 and want opposite ends of a
128x range.

The best value is a property of the card and spans **8192 to 1048576**. To find
yours:

```sh
julia --project=~/gpuenv bench/gpu_search_report.jl FILE.fft
```

Use one of your own `.fft` files, ideally a large one. It sweeps `--blocksize`,
prints a per-phase breakdown, and ends with a recommendation and the penalty for
not passing one. Then run searches with that value:

```sh
julia --project=~/gpuenv bin/coherent_search.jl --gpu --blocksize 8192 FILE.fft
```

The search prints the default it used, and warns if you pass `--blocksize 2048`
or less explicitly — that is the CPU's value and it costs 1.4x to 5.6x on a GPU.

**Why it varies so much, if you are curious.** Two effects pull in opposite
directions. A large L2 wants a **small** chunk, so the whole pipeline stays
resident in cache; a lot of SMs want a **big** one, because a small chunk cannot
fill them. Which wins is not predictable from a spec sheet: the RTX 4000 Ada
(40 MB L2, 48 SMs) and the RTX 4500 Ada (48 MB, 60 SMs) want **8192**, while
the A100 (40 MB, but 108 SMs) wants **1048576** and is 2.2x slower at 8192.
The L40 (96 MB, 142 SMs) sits between them at 32768. Cards with a small L2
cannot hold the working set at any chunk size, so only occupancy and launch
amortisation are left and bigger always wins.

Measured optima, and what the default costs on each:

| card | L2 | optimum | 65536 costs | 262144 costs |
|---|---|---|---|---|
| RTX 4500 Ada | 48 MB | 8192 | **1.24x** | 1.12x |
| RTX 4000 SFF Ada | 40 MB | 8192 | 1.22x | ~1.23x |
| L40 | 96 MB | 32768 | 1.21x | 1.20x |
| A100 80GB | 40 MB | 1048576 | 1.18x | 1.05x |
| RTX A4000 | 4 MB | 524288 | 1.06x | 1.01x |
| GTX 1080 | 2 MB | 262144 (sweep capped by memory) | 1.06x | 1.00x |
| RTX 2080 Super | 4 MB | 262144 | — | — |
| RTX A400 | 1 MB | 131072 (sweep capped by memory) | 1.00x | does not fit |

(The RTX 4000 SFF Ada and 2080 Super rows are from older sweeps; see
`docs/gpu_design.md`.)

**If you are on an RTX 4000 or 4500 Ada, pass `--blocksize 8192`.** On the
cards it fits, **262144 is as good as the default or better** (within 1% on the
SFF Ada) and gets the A100 and A4000 to within 1.05x. So if you do not want to sweep, `--blocksize 262144` is
a better guess than the default on anything with 40+ SMs and enough memory
(~1.1 GiB of workspace on top of your `.fft`; it does not fit a 4 GB card
alongside a 1.29 GiB file).

### What the GPU path does and does not support

| | |
|---|---|
| `--sigma analytic` | required (the default). `--sigma measured` needs a device MAD and errors out |
| `--normalize` | not supported yet; errors out |
| `--metricstats` | not supported yet; errors out |
| everything else | as on the CPU |

Each of these errors clearly rather than silently doing something different.

### Accuracy, and how the GPU is pinned

The GPU is `Float32` throughout and agrees with the CPU to **~2e-7** on profiles
and on the boxcar metric, against a pinned tolerance of 1e-5. In practice
candidate lists have come out byte-identical on real data, but that is **not
guaranteed** — a trial sitting exactly on the threshold could cross either way.

Two properties *are* guaranteed and tested:

- **Batch invariance is bit-exact.** A chunk starting at global trial `t0`
  reproduces one long chunk exactly, so `--blocksize` changes speed and nothing
  else. This is what makes tuning it safe.
- **Transform sub-batching is bit-exact**, likewise — it is a scheduling change
  only.

`test/test_gpu.jl` (300 tests) is **not** run by `Pkg.test()`, even on a GPU
host: `test/Project.toml` has no CUDA, so it skips itself there. Run it in your
CUDA environment:

```sh
julia --project=~/gpuenv -e 'using CUDA, Test; include("test/test_gpu.jl")'
```

`bench/paper_gpu_run.sh` does this, then diffs CPU against GPU candidates in a
band below and a band past the file's Nyquist knee.

### Reporting a new card

`bench/gpu_search_report.jl`'s final block is designed to be pasted back into an
issue or email. Results from cards we have not seen are genuinely useful: the
design log (`docs/gpu_design.md`) keeps per-card measurements, and the
`--blocksize`
guidance above is built from only eight GPUs so far.

If you want to bootstrap Julia and CUDA.jl on a bare GPU host with no root,
`bench/gpu_probe_setup.sh` does the whole thing and needs nothing from this repo
but itself and `bench/gpu_probe.jl`.

## Testing

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

Every GPU test is **skipped** in that environment, because CUDA is a weak
dependency and `--project=.` does not have it — the run says so rather than
passing silently. To include them, use the environment from the GPU section:

```sh
julia --project=~/gpuenv -e 'using Pkg; Pkg.test("CoherentSearch")'
```

## Cross-validation against the Python oracle

These compare directly against the original Python `coherent_search`. Point
`COHERENT_PYTHON` at an interpreter that can `import coherent_search`, and
`COHERENT_FFT` (or the first argument) at a `.fft` file:

```sh
# Accuracy: Julia must match Python to ~1e-9 relative
julia --project=crossval crossval/crossval_accuracy.jl FILE.fft

# Speed: kernel speedup + headline full-search timing
julia --project=crossval -t auto crossval/crossval_speed.jl FILE.fft
```

On the bundled `harmonics_hi.fft` test pulsar (10.0123 Hz) the accuracy check
agrees with Python to ~1e-16 relative, confirming the indexing and FFT
conventions are correct.

## Status

Kernels, file I/O, CLI, tests, and Python-oracle cross-validation are in place
and passing. The search is chunk-parallel with cached FFTW plans and
interpolation kernels, an allocation-free hot loop, a batched inverse FFT, and
exact per-trial Fourier interpolation.

The detection metric is the **peak boxcar matched filter**: each profile is
correlated with a geometric bank of top-hat widths, each made *zero-mean and
unit-L2*, and scored

```
max_{w,phase} (S_w − δ·S_tot) / (σ̂ · sqrt(w·(1−δ))),      δ = w / nbins
```

with the per-bin noise scale `σ̂` computed analytically by default (see
[The noise scale](#the-noise-scale-analytic-by-default)). Because the widths
are fixed a priori,
every (phase, width) trial is `N(0,1)` under noise, so the pure-noise
distribution is analytic and — unlike the older on-pulse sums — flat across
harmonic decimations: one `--threshold` means one false-alarm rate at every `k`.

This is **exactly riptide's `snr1`** (`cpp/snr.hpp`), verified against the
`rseek` binary itself to 1.4e-7 on real folds, which is riptide's own `Float32`
accumulation; the two codes' S/N columns are therefore the same quantity. It is
also a port of the Python `snr_metric` and is oracle-pinned to machine
precision (1.4e-16). It replaced an earlier form that subtracted each profile's
*median* and divided by `σ̂√w`: that normalisation drifted with source
brightness, so it had no calculable false-alarm rate, and at matched FAP the
zero-mean template detects strictly better at every duty cycle.

> The earlier width-penalised on-pulse metrics (`--metric non` = `N_on^p`,
> `--metric sd2` = `Σd²^p`) were retired here and upstream. On real data `non`
> produced many more false positives than `sd2`, and their noise floors scaled
> with the profile bin count, which biased a fixed threshold toward the
> low-decimation passes — the problem the boxcar metric was written to fix.

Near-identical candidates are collapsed by default (`--noremove`
disables it, `--drtol` sets the tolerance), and harmonically-related candidates
(the `f/2`, `2f`, `3f/2`, … family) are collapsed to their strongest member
(`--noharmremove`, `--numharm`). A cheap multi-frequency search by harmonic
decimation (`--maxdecim`) re-uses the interpolated harmonics to fold at integer
multiples of each fundamental, pinned by a test that every decimation pass
reproduces the native reduced-harmonic fold. A progress meter prints to stderr
(`--progressbar`, `--noprogress`).

### Interpolation

Harmonic amplitudes come from the Eqn.-30 kernel evaluated *exactly* at each
trial frequency. Factoring the coefficients as `A(dr)/(dr-j)` makes the weights
real, so a point costs `m` real multiply-adds and reads only `m` consecutive
bins; the handful of distinct `dr` values a whole search visits are tabulated
once per harmonic and indexed by exact integer arithmetic. There is no fine
grid, no `numbetween`, and no linear interpolation.

The FFT-correlation method ported from the Python original — build a uniform
fine grid of `numbetween` points per Fourier bin with two transforms, then
linearly interpolate it — survives only as the *reference* path
(`reference_profiles(...; kernel=:fft)`, `finterp_fft`), which is what the Python
oracle is pinned to. It was retired from the search because it is both slower
(~3.8× on the interpolation) and an approximation: its linear interpolation is
worth up to ~5% in amplitude at high harmonics with `numbetween=16`.

`bench/interp_bench.jl` compares the two on throughput and accuracy, and
`--verbose` prints the trial grid, chunking and interpolation phase-cycle
lengths.

See `docs/Summary_and_Future_Work.md` and `docs/decimation_design.md` for details.
