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

# --- Fourier interpolation kernels ---
export finterp_coeffs, fourier_interp, finterp_multi,
       finterp_fft, finterp_fft_coeffs, nearby_fourier_bins,
       nearby_fourier_bin_range, next_pow_of_2, next_smooth, is_smooth
include("fourierinterp.jl")

# --- PRESTO file I/O ---
export FFTFile, SimpleInf, freqs
include("fileio.jl")

# --- Search ---
export SearchParams, Candidate, search, search_block, block_metrics, coherent_profiles,
       reference_profiles, snr_metrics, chunk_ngoodbins, remove_duplicates,
       remove_harmonics, chunk_metrics, build_harmonic_plans, harmonic_numbetween,
       decimation_set, boxcar_widths, BlockMetricStats, MetricHistogram, MetricStats,
       metricstats_summary, metricstats_windows, hist_quantile,
       MetricNorm, build_metricnorm, harmonic_plan_report
include("search.jl")

# --- Direct O(m) Fourier interpolation (the default production interpolator) ---
# Included after `search.jl`: its methods take `Workspace`/`SearchParams`, and
# `fill_chunk_profiles!` calls into it through an untyped `dplans` argument.
export DirectPlan, build_direct_plans, trial_grid_rational,
       finterp_direct, finterp_direct!
include("directinterp.jl")

# --- FFTW plan-wisdom persistence (faster planning / start-up) ---
export wisdom_path, import_wisdom!, export_wisdom!, prime_wisdom
include("wisdom.jl")

# --- Per-candidate profile reconstruction (for plotting) ---
export candidate_profile, rotate_to_peak
include("candidate.jl")

end # module
