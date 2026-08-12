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
  cli.jl              ArgParse command-line driver (`CoherentSearch.main`)
bin/
  coherent_search.jl  command-line entry point (a shim onto src/cli.jl)
  plotting.jl         CairoMakie candidate-profile plotting (loaded on demand)
  plot_candidates.jl  standalone: re-plot profiles from a saved candidate file
  sift_candidates.py  cross-observation candidate sifter (.cohout / .txt)
test/                 unit tests (golden values, analytic signals, indexing)
crossval/             Python-as-oracle accuracy + speed cross-validation
sysimage/             optional PackageCompiler sysimage for production runs
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

### Searching many files at once (and start-up cost)

**Pass every `.fft` file to a single invocation** rather than running the CLI
once per file:

```sh
julia --project=. -t auto bin/coherent_search.jl *_red.fft \
    --lofreq 0.1 --hifreq 100 --threshold 8 --noplot
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

**Plotting is deferred to the end of the run.** All searches finish first, then
CairoMakie is loaded *once* to plot every file's candidates. Loading it costs
~9 s plus first-call compilation — by far the largest fixed cost in the program —
so in bulk runs pass `--noplot` and plot later from the saved candidate files
with `bin/plot_candidates.jl`.

Or from Julia:

```julia
using CoherentSearch
ft = FFTFile("FILE.fft")
cands = search(ft, SearchParams(nharms=32); lofreq=0.1, hifreq=100, threshold=8)
```

Searching several files from Julia? Share one `SearchCache` (and one
`SearchParams` object — reuse is keyed on its identity):

```julia
params, cache = SearchParams(nharms=32), SearchCache()
for f in files
    cands = search(FFTFile(f), params; cache=cache, lofreq=0.1, hifreq=100)
end
```

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

### Candidate output format

```
#       'S/N'      Frequency (Hz)        Period (ms)    #Harm  Ducy(%)
1        11.92      7.118536329269    140.478316573069    32    14.06
```

`#Harm` is the harmonic count that found the candidate (`k = nharms ÷ #Harm`
identifies the decimation). `Ducy(%)` is the duty cycle of the best-fitting
boxcar — `width / profile bins`, exactly as riptide's `rseek` defines `ducy`, so
the two searches can be compared directly. It is `-` for the `non`/`sd2`
metrics, which scan no width bank.

The search's hot loop deliberately discards *which* boxcar width won (it runs
~1e8 times and only reported candidates need it), so the width is recovered
afterwards by refolding each reported candidate — see `measure_ducy`. This is
exact, not an approximation: the noise scale σ multiplies every width's score
equally and so cannot change which one wins, which is what lets the width be
recovered from an isolated profile.

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
   | `--noplot` | 2.4 s | 2.3 s |
   | with plots | 18.7 s | **7.4 s** |

   ```sh
   julia --project=sysimage -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
   julia --project=sysimage sysimage/build_sysimage.jl        # ~28 minutes
   julia --sysimage sysimage/coherent_search.so --project=. -t auto \
         bin/coherent_search.jl FILE.fft [FILE2.fft ...] [options]
   ```

   Worth it only for repeated plotting-enabled runs: the image is 1.14 GB, and
   the *first* run after building it (or after a reboot) spends ~23 s reading it
   into page cache. If you run with `--noplot`, skip it.

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

It derives our settings from riptide's: `nharms = bmax/2` and
`maxdecim = bmax/bmin` line the profile-bin spans up (`--bmin 30 --bmax 120`
→ `--nharms 60 --maxdecim 4`, both spanning 120…30 bins), and our fundamental
range is `[1/Pmax, 1/Pmin]`. It prints the derived configuration, and flags
where the match is imperfect, so the comparison stays auditable.

Measured on `PM0063_034C1_DM445.0_red.fft` (T=2097 s, 4-core i7-10510U laptop),
searching 0.1–20 Hz with 120 profile bins:

| | wall (s) | cores |
|---|---|---|
| `rseek` | 4.65 | 1.15 |
| `coherent_search -t 1` | 9.82 | 1.04 |
| `coherent_search -t 4` | 6.18 | 2.96 |

**Single-threaded, riptide's FFA is ~2.1× faster.** Both find the 7.1185 Hz
pulsar with comparable S/N (11.9 vs 12.3) and consistent duty cycle. riptide's
two other candidates are the `f/2` and `2f` harmonics of it, which we collapse
by default and it does not filter at all (`--noharmremove` for a like-for-like
count).

Two axes are ours alone rather than like-for-like wins, and are reported
separately for that reason: riptide's C extension is built without OpenMP so
`rseek` cannot use more cores, and harmonic decimation buys us 4× the frequency
coverage (0.1–80 Hz) for 1.65× the time (16.1 s at `-t 1`).

Reading the output: the two S/N values are *different statistics* (time-domain
matched filter vs coherent Fourier boxcar) and only roughly comparable. The duty
cycles are defined identically on both sides and are the quantity to compare.

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
(`--progressbar`, `--noprogress`).

### Interpolation (`--interp`)

Two interpolators are available and produce the same physics by different means:

- **`--interp direct` (default)** evaluates the Eqn.-30 kernel *exactly* at each
  trial frequency. Factoring the coefficients as `A(dr)/(dr-j)` makes the weights
  real, so a point costs `m` real multiply-adds and reads only `m` consecutive
  bins; the handful of distinct `dr` values a search visits are tabulated once.
  There is no fine grid, no `numbetween`, and no linear interpolation.
- **`--interp fft`** is the FFT-correlation method ported from the Python
  original: build a uniform fine grid of `numbetween` points per Fourier bin with
  two transforms, then linearly interpolate it at the trial frequencies. It is
  what the Python oracle and the machine-precision equivalence tests are pinned
  to. Its padded transform length is chosen by `--fftsizing`: `pow2` (the
  default, reproducing the original exactly) or `smooth` (next `2·3·5·7`-smooth —
  much less padding, but measured no faster in an actual search).

The direct path is both faster and far more accurate: the linear interpolation
in the FFT path is an approximation worth up to ~5% in amplitude at high
harmonics with the default `numbetween=16`, which the direct path removes
entirely. `--verbose` prints the per-harmonic plan (`numbetween`, `m`, transform
length and padding, fine-grid oversampling, whether linear interpolation is
needed), and `bench/interp_bench.jl` compares the two on throughput and accuracy.

See `Summary_and_Future_Work.md` and `decimation_design.md` for details.
