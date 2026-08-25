"""
    CoherentSearch

A pure-Julia pulsar search using fast complex Fourier interpolation and
coherent harmonic summing of PRESTO-style FFT files.  A port of the Python
`coherent_search` package, restructured for multi-threaded performance.

References:
  - Fourier interpolation: Eqn. 30 of https://arxiv.org/pdf/astro-ph/0204349
  - PRESTO: https://github.com/scottransom/presto
"""
module CoherentSearch

using Logging: with_logger, NullLogger

# --- PRESTO file I/O ---
# Included *before* the interpolation kernels because `fourier_interpolate`
# (the one-call, whole-file convenience wrapper at the bottom of
# `fourierinterp.jl`) dispatches on `FFTFile`.  `fileio.jl` itself depends on
# nothing but `Mmap`, so the order is free.
export FFTFile, SimpleInf, freqs
include("fileio.jl")

# --- Fourier interpolation kernels ---
export finterp_coeffs, fourier_interp, fourier_interpolate, finterp_multi,
       finterp_fft, finterp_fft_coeffs, nearby_fourier_bins,
       nearby_fourier_bin_range, next_pow_of_2, next_smooth, is_smooth
include("fourierinterp.jl")

# --- Compute-backend TYPES (the methods come after `search.jl`; see below) ---
export SearchBackend, CPUBackend, gpu_backend, has_gpu, require_gpu,
       gpu_timing!, gpu_phase_reset!, gpu_phase_times, GPU_PHASE_NAMES
include("backendtypes.jl")

# --- Search ---
export SearchParams, Candidate, search, search_block, block_metrics, coherent_profiles,
       reference_profiles, snr_metrics, chunk_ngoodbins, remove_duplicates,
       remove_harmonics, chunk_metrics,
       decimation_set, boxcar_widths, ladder_boxcar_widths, boxcar_best_width, BlockMetricStats, MetricHistogram, MetricStats,
       SearchCache,
       metricstats_summary, metricstats_windows, hist_quantile,
       MetricNorm, build_metricnorm,
       proftype, phase_reset!, phase_times, PHASE_NAMES
include("search.jl")

# --- Direct O(m) Fourier interpolation (the default production interpolator) ---
# Included after `search.jl`: its methods take `Workspace`/`SearchParams`, and
# `fill_chunk_profiles!` calls into it through an untyped `dplans` argument.
export DirectPlan, build_direct_plans, trial_grid_rational,
       finterp_direct, finterp_direct!
include("directinterp.jl")

# --- Compute-backend METHODS (after `search.jl`/`directinterp.jl`; see above) ---
export chunk_ftprofs, gpu_chunk_ftprofs,
       chunk_profiles, gpu_chunk_profiles, chunk_boxcar, gpu_chunk_boxcar
include("backend.jl")

# --- FFTW plan-wisdom persistence (faster planning / start-up) ---
export wisdom_path, import_wisdom!, export_wisdom!, prime_wisdom,
       production_params
include("wisdom.jl")

# --- Per-candidate profile reconstruction (for plotting) ---
export candidate_profile, rotate_to_peak, measure_ducy
include("candidate.jl")

# --- Command-line front-end (`bin/coherent_search.jl` is a shim onto this) ---
# `main` lives in the package, not in `bin/`, purely so the precompile workload
# below can cache it: as a top-level script it cost ~4.7 s of inference and
# codegen on *every* run, plus ~1.6 s for ArgParse's first `parse_args`.
include("cli.jl")

# ---------------------------------------------------------------------------
# Precompile workload
#
# Julia's start-up compilation, not the search, dominated a short run: 15.6 s
# wall for 1.4 s of actual searching.  Running a miniature end-to-end search
# here caches the inferred/native code in the package image, so a real run does
# not re-derive it.  Costs ~3.4 s of extra precompilation per `src/` edit and
# saves ~13 s per run.
#
# Plotting must stay out of it — CairoMakie is a ~9 s load and would be dragged
# into precompilation for no benefit — which is now simply the CLI default
# (`--plot` opts in), so the workload need not say anything.
# ---------------------------------------------------------------------------
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    # A synthetic observation, small enough to compile fast.  The in-memory
    # `FFTFile` exercises `search`; the on-disk pair exercises `main`, which must
    # read a real `.fft`/`.inf` through the CLI.
    _N = 1 << 15
    _dt = 1.0e-4
    _amps = ComplexF32[ComplexF32(0.5f0, 0.5f0) for _ in 1:(_N ÷ 2)]
    _inf = SimpleInf("synthetic.inf", "PRECOMPILE", 5.0e4, _N, _dt, 0.0)
    _ft = FFTFile("synthetic.fft", _amps, _inf, _N, _N * _dt, 1.0 / (_N * _dt),
                  true, true, real(_amps[1]), imag(_amps[1]))
    _dir = mktempdir()                       # removed when this process exits
    _fftpath = joinpath(_dir, "synthetic.fft")
    write(_fftpath, _amps)
    write(joinpath(_dir, "synthetic.inf"),
          " Number of bins in the time series          =  $_N\n" *
          " Width of each time series bin (sec)        =  $_dt\n" *
          " Dispersion measure (cm-3 pc)               =  0.0\n")
    # No `--maxdecim`: the CLI now defaults to 6, so letting the workload take
    # the default is what makes it cache `decim_pass!` — the code path every real
    # invocation runs.  `Hk` is a field, not a type parameter, so the extra
    # decimation factors cost a little workload runtime and no extra compilation.
    _argv = [_fftpath, "--noprogress", "--threshold", "1e9",
             "--nharms", "8", "--blocksize", "64", "--lofreq", "0.6",
             "--hifreq", "0.64", "--nowisdom",
             "-o", joinpath(_dir, "synthetic.cohout")]

    @compile_workload begin
        # `search` directly, then the CLI below, so both entry points are cached.
        _p = SearchParams(nharms = 8)
        _cache = SearchCache()
        # Silenced like the `main` call below: the workload's amplitudes are a
        # constant, not noise, so the analytic-sigma sanity check legitimately
        # fires — and a warning from inside precompilation would be alarming
        # noise.  Running it here is deliberate: it caches that path too.
        with_logger(NullLogger()) do
            search(_ft, _p; lobin = 200, hifreq = 210 / _ft.T, blocksize = 64,
                   threshold = 1e9, progress = :none, wisdom = false, cache = _cache)
        end
        # …then the whole CLI, so ArgParse's table and `main` are cached too.
        # Silenced: the workload's @info output would otherwise be echoed as
        # package-precompilation noise.
        with_logger(NullLogger()) do
            main(_argv)
        end
    end
end

end # module
