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
  search.jl           block-parallel coherent harmonic-summing search
bin/
  coherent_search.jl  ArgParse command-line front-end
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

Phase 1: kernels, file I/O, block-parallel search, CLI, tests, and
cross-validation are in place and passing. Next: thread-local buffer/plan reuse
for the hot loop, candidate de-duplication (`--noremove`), and broader
benchmarking.
