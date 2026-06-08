# Fast complex Fourier interpolation.
#
# This is a direct port of the Python `coherent_search.fourierinterp` module.
# The interpolation kernel follows Eqn. 30 of Ransom, Eikenberry & Middleditch
# (2002), https://arxiv.org/pdf/astro-ph/0204349
#
# INDEXING NOTE (the whole reason these functions are so heavily tested):
# The Python original is 0-based with half-open slices `ft[a:b]`.  Julia is
# 1-based with inclusive ranges `ft[a:b]`.  Throughout, `r_int` is kept as the
# *Python* (0-based) bin index so the translation to Julia indices is explicit
# and auditable.  A 0-based Python slice `ft[lo:hi]` becomes the 1-based Julia
# range `ft[lo+1:hi]` (same element count: hi-lo).

using FFTW
using LinearAlgebra: dot

"""
    finterp_coeffs(dr, m) -> Vector{ComplexF64}

Compute `m` Fourier interpolation coefficients for a sub-bin Fourier frequency
offset `dr` in `[0, 1)`.  Mirrors `get_finterp_coeffs` in the Python code.
"""
function finterp_coeffs(dr::Real, m::Integer)
    iseven(m) || throw(ArgumentError("m must be even"))
    (0.0 <= dr < 1.0) || throw(ArgumentError("dr must be in [0.0, 1.0)"))
    # Python: offsets = dr - np.arange(-m//2 + 1, m//2 + 1)   (length m)
    offsets = dr .- (-(m ÷ 2) + 1 : m ÷ 2)
    # np.sinc and Julia sinc are both the *normalized* sinc, sin(pi x)/(pi x).
    # np.exp(1j*pi*x) == cispi(x).
    return sinc.(offsets) .* cispi.(offsets)
end

"""
    nearby_fourier_bin_range(r, m) -> UnitRange{Int}

Return the **1-based Julia** index range of the `m` Fourier bins surrounding the
real-valued Fourier frequency `r`.  This encapsulates the 0→1 based index
translation in one auditable place.

Python original:
    r_int = int(np.floor(r + 1e-15)) + 1   # 0-based
    return ft[r_int - m//2 : r_int + m//2] # half-open
"""
function nearby_fourier_bin_range(r::Real, m::Integer)
    iseven(m) || throw(ArgumentError("m must be even"))
    r_int = floor(Int, r + 1e-15) + 1            # 0-based bin index (as in Python)
    lo0 = r_int - m ÷ 2                           # Python slice start (0-based)
    hi0 = r_int + m ÷ 2                           # Python slice stop  (exclusive)
    return (lo0 + 1):hi0                          # 1-based inclusive Julia range
end

"""
    nearby_fourier_bins(r, ft, m) -> view

The `m` complex Fourier amplitudes around real-valued frequency `r`.
"""
function nearby_fourier_bins(r::Real, ft::AbstractVector, m::Integer)
    return @view ft[nearby_fourier_bin_range(r, m)]
end

"""
    fourier_interp(r, ft, m) -> ComplexF64

Interpolated complex Fourier amplitude at a single real-valued frequency `r`.
"""
function fourier_interp(r::Real, ft::AbstractVector, m::Integer)
    r >= 0.0 || throw(ArgumentError("r must be non-negative"))
    iseven(m) || throw(ArgumentError("m must be even"))
    coeffs = finterp_coeffs(mod(r, 1.0), m)
    bins = nearby_fourier_bins(r, ft, m)
    # Python: np.dot(coeffs.conjugate(), bins).  Julia `dot` conjugates its
    # first argument, so dot(coeffs, bins) == sum(conj(coeffs).*bins).
    return dot(coeffs, bins)
end

"""
    finterp_multi(rs, ft, m; coeffs=nothing) -> Vector{ComplexF64}

Interpolate at many real-valued frequencies `rs` that must all lie between the
same pair of integer Fourier bins.  Mirrors `finterp_multi`.
"""
function finterp_multi(rs::AbstractVector, ft::AbstractVector, m::Integer; coeffs=nothing)
    iseven(m) || throw(ArgumentError("m must be even"))
    lo_rint = floor(Int, minimum(rs) + 1e-15)
    hi_rint = floor(Int, maximum(rs) + 1e-15)
    (hi_rint - lo_rint == 0) || throw(ArgumentError("rs must all be between 2 Fourier bins"))
    if coeffs === nothing
        # offsets: (len(rs), m); rows indexed by frequency, cols by bin offset.
        offsets = mod.(rs, 1.0) .- (-(m ÷ 2) + 1 : m ÷ 2)'
        coeffs = sinc.(offsets) .* cispi.(offsets)
    else
        size(coeffs) == (length(rs), m) || throw(ArgumentError("coeffs shape must be (length(rs), m)"))
    end
    bins = nearby_fourier_bins(rs[1], ft, m)
    # Python np.vecdot(coeffs, bins) conjugates the first arg per row.
    return conj.(coeffs) * collect(bins)
end

"""
    next_pow_of_2(n) -> Int

Smallest power of two ≥ `n`.  Matches the Python `next_pow_of_2`.
"""
function next_pow_of_2(n::Integer)
    n > 0 || throw(ArgumentError("n must be a positive integer"))
    return nextpow(2, n)
end

"""
    finterp_fft_coeffs(numbetween, m, fftlen) -> Vector{ComplexF64}

Precompute the FFT'd interpolation kernel used by the FFT-correlation method.
Mirrors `get_finterp_FFT_coeffs`.
"""
function finterp_fft_coeffs(numbetween::Integer, m::Integer, fftlen::Integer)
    iseven(m) || throw(ArgumentError("m must be even"))
    fftlen >= numbetween * m || throw(ArgumentError("fftlen must be >= numbetween * m"))
    fftlen == next_pow_of_2(fftlen) || throw(ArgumentError("fftlen must be a power of 2"))
    coeffarr = zeros(ComplexF64, fftlen)
    n = (numbetween * m) ÷ 2
    # Python: offsets = np.arange(numbetween*m//2) / numbetween
    offsets = collect(0:(n - 1)) ./ numbetween
    # np.exp(-1j*pi*x) == cispi(-x)
    @views coeffarr[1:n] .= sinc.(offsets) .* cispi.(-offsets)
    # Python: offsets = (-(offsets + 1/numbetween))[::-1]
    offsets2 = reverse(-(offsets .+ 1.0 / numbetween))
    @views coeffarr[(end - n + 1):end] .= sinc.(offsets2) .* cispi.(-offsets2)
    return conj.(fft(coeffarr))
end

"""
    finterp_fft(lobin, numbins, numbetween, ft, m; coeffs=nothing) -> Vector{ComplexF64}

Interpolate `numbins * numbetween` evenly spaced frequencies starting at integer
bin `lobin` using FFT-based correlation.  The returned frequencies are
`lobin .+ (0:numbins*numbetween-1) ./ numbetween`.  Mirrors `finterp_FFT`.

`lobin` is a 0-based Fourier bin number (matching PRESTO / the Python code);
`ft` is the 1-based Julia amplitude vector.
"""
function finterp_fft(lobin::Integer, numbins::Integer, numbetween::Integer,
                     ft::AbstractVector, m::Integer; coeffs=nothing)
    m2 = m ÷ 2
    numftbins = (numbins + m) * numbetween
    fftlen = next_pow_of_2(numftbins)
    if coeffs === nothing
        coeffs = finterp_fft_coeffs(numbetween, m, fftlen)
    else
        length(coeffs) == fftlen || throw(ArgumentError("coeffs length must equal fftlen"))
    end
    ftarr = zeros(ComplexF64, fftlen)
    tmplobin = lobin - m2                 # 0-based slice start
    tmphibin = lobin + numbins + m2       # 0-based slice stop (exclusive)
    # Python: ftarr[np.arange(numbins+m)*numbetween] = ft[tmplobin:tmphibin]
    # Zero-stuff the original bins every `numbetween` samples.
    src = @view ft[(tmplobin + 1):tmphibin]          # 0→1 based slice
    dest_idx = (0:(numbins + m - 1)) .* numbetween .+ 1
    ftarr[dest_idx] .= src
    corr = ifft(fft(ftarr) .* coeffs)
    # Python: corr[m2*numbetween : (m2+numbins)*numbetween]
    return corr[(m2 * numbetween + 1):((m2 + numbins) * numbetween)]
end
