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
       nearby_fourier_bin_range, next_pow_of_2
include("fourierinterp.jl")

# --- PRESTO file I/O ---
export FFTFile, SimpleInf, freqs
include("fileio.jl")

# --- Search ---
export SearchParams, Candidate, search, search_block, block_metrics, coherent_profiles
include("search.jl")

end # module
