# CoherentSearch.jl

A pure-Julia pulsar search using fast complex **Fourier interpolation** and
**coherent harmonic summing** of PRESTO-style FFT files. A port of the Python
[`coherent_search`](../coherent_search) package, restructured for
multi-threaded performance.

The goal is a fast, parallel, well-tested search code. Correctness is anchored
by cross-validating every numerical result against the original Python
implementation used as an independent oracle.

## References

- Fourier interpolation: Eqn. 30 of Ransom, Eikenberry & Middleditch (2002),
  <https://arxiv.org/pdf/astro-ph/0204349>
- PRESTO: <https://github.com/scottransom/presto>

## Layout

```
src/
  CoherentSearch.jl   module + public API
  fourierinterp.jl    interpolation kernels (the indexing-critical code)
  fileio.jl           PRESTO .fft / .inf readers (mmap)
  search.jl           chunk-parallel coherent harmonic-summing search
  candidate.jl        per-candidate high-accuracy profile reconstruction
bin/
  coherent_search.jl  ArgParse command-line front-end
  plotting.jl         CairoMakie candidate-profile plotting (loaded on demand)
  plot_candidates.jl  standalone: re-plot profiles from a saved candidate file
test/                 unit tests (golden values, analytic signals, indexing)
crossval/             Python-as-oracle accuracy + speed cross-validation
```

## Design notes

- **Indexing.** Python is 0-based with half-open slices; Julia is 1-based with
  inclusive ranges. The translation is isolated and documented in
  `fourierinterp.jl` (see `nearby_fourier_bin_range`), and pinned by tests and
  the cross-validation to machine precision.
- **Parallelism.** The stateful, forward-walking `FourierInterpolator` of the
  Python version is replaced by **independent frequency blocks**
  (`block_metrics` / `search_block`). Each block owns its buffers and shares no
  mutable state, so the search scales across cores via `Threads.@threads` and
  extends naturally to `Distributed` for cluster-scale runs.
- **FFT conventions.** `irfft` of the stacked harmonic amplitudes matches
  numpy's `np.fft.irfft` (both ignore the imaginary parts of the DC/Nyquist
  bins); this is verified directly in the tests.

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
julia --project=. -t auto bin/coherent_search.jl FILE.fft \
    --lofreq 0.1 --hifreq 100 --nharms 32 --threshold 8
```

Or from Julia:

```julia
using CoherentSearch
ft = FFTFile("FILE.fft")
cands = search(ft, SearchParams(nharms=32); lofreq=0.1, hifreq=100, threshold=8)
```

Each candidate reports its barycentric spin frequency, period (`1/f`), the S/N
metric, and the number of harmonics summed in the detection.

### Multi-frequency search by harmonic decimation

`--maxdecim k` (default `1` = off) additionally folds every trial fundamental at
`2×, 3×, … k×` its frequency *almost for free*, by re-using the harmonic
amplitudes already interpolated for the base fold: taking every `k`-th harmonic
and running a shorter inverse FFT yields the fold at `k·rf` with
`⌊nharms/k⌋` harmonics. This extends the search to faster pulsars (which tend to
have wider profiles and so need fewer harmonics) without paying for extra
interpolation. When enabled, `nharms` defaults to a composite `60` so that many
`k` give clean integer harmonic counts. The harmonic count printed for each
candidate identifies the decimation that found it (`k = nharms ÷ nharm`).

```sh
# Search fundamentals 0.1–100 Hz and, via decimation, faster pulsars up to ~600 Hz
julia --project=. -t auto bin/coherent_search.jl FILE.fft \
    --lofreq 0.1 --hifreq 100 --maxdecim 6 --threshold 8
```

See `decimation_design.md` for the derivation that decimation stays correctly
sampled (each `k`'s top harmonic still steps by ≤ `hidr`, and the base input-FFT
read depth already covers every `k`) and the full bookkeeping.

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

### Candidate profile plots

By default the CLI reconstructs and plots the pulse profile of every reported
candidate, one grid of panels per US-Letter portrait page, written as
zero-padded PNGs (`<stem>_01.png`, `<stem>_02.png`, …).  Each profile is folded
with a high-accuracy exact-interpolation path (independent of the throughput-
tuned search) and rotated so its peak sits at phase 0.5; the panel caption
carries the full text-line information (index, S/N, frequency, period, harmonic
count, and the decimation `k`).  Plotting loads CairoMakie lazily — searches
that pass `--noplot`, the test suite, and the cross-validation never load it.

Every profile is folded at the **full `--nharms` harmonic depth**, regardless of
the harmonic-decimation factor `k` that found the candidate (a `k=3` detection
summed only `nharms/3` harmonics, but its profile still uses all `nharms`): this
much more closely matches a true time-domain fold of the time series at the
candidate period.  Harmonics that would exceed the Nyquist frequency are simply
omitted (not zero-padded), so a fast candidate's profile uses fewer bins.  The
`#Harm`/`k` in each caption still report what the *search* summed to detect it.

```sh
julia --project=. -t auto bin/coherent_search.jl FILE.fft -o cands.txt \
    --plotstem cands --plotcols 3 --plotrows 5     # -> cands_01.png, ...
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

## Testing

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
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
per-harmonic interpolation tuning (`--noalign` to disable). The detection metric
sums the on-pulse flux and divides by a selectable width penalty (`--metric`):
`non` = `N_on^p` (duty cycle; `p=1/2` is a calibrated matched filter, larger `p`
suppresses broad/RFI-like signals) or `sd2` = `Σd²^p` (phase spread). It is a
port of the Python `snr_metric`, oracle-pinned to machine precision for both
penalties.

> **Note (defaults under review).** On real data the default
> `--metric non --pexp 0.5` empirically produces *many* more false-positive
> candidates than `--metric sd2`, and many of those false positives do not look
> pulsar-like at all — their reconstructed profiles resemble random noise rather
> than the narrow-duty-cycle pulse most real pulsars show. This suggests the
> default CLI options may need to change (e.g. to `sd2`, and/or a larger `pexp`).
> See `Summary_and_Future_Work.md` for the follow-up.

Near-identical candidates are collapsed by default (`--noremove`
disables it, `--drtol` sets the tolerance), and harmonically-related candidates
(the `f/2`, `2f`, `3f/2`, … family) are collapsed to their strongest member
(`--noharmremove`, `--numharm`). A cheap multi-frequency search by harmonic
decimation (`--maxdecim`) re-uses the interpolated harmonics to fold at integer
multiples of each fundamental, pinned by a test that every decimation pass
reproduces the native reduced-harmonic fold. A progress meter prints to stderr
(`--progressbar`, `--noprogress`). Next: a throughput sweep to tune per-harmonic
`fftlen`/`numbetween`. See `Summary_and_Future_Work.md` and
`decimation_design.md` for details.
