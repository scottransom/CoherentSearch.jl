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
    is_smooth(n) -> Bool

Whether `n` factorises entirely into 2, 3, 5 and 7 — the radices FFTW has
dedicated codelets for, and hence the lengths it transforms most efficiently.
"""
function is_smooth(n::Integer)
    n > 0 || return false
    for f in (2, 3, 5, 7)
        while n % f == 0
            n ÷= f
        end
    end
    return n == 1
end

"""
    next_smooth(n) -> Int

Smallest `2·3·5·7`-smooth number ≥ `n`.

The FFT-correlation interpolator is free to pad its transform to *any* length ≥
the `numbetween*(numbins+m)` points it actually needs, so the padded length is
ours to choose.  `next_pow_of_2` is the simplest choice but a poor one: the
needed lengths land wherever they land, so rounding up to a power of two costs a
mean ~1.38× (worst 1.98×) in transform length across the harmonic schedule.
Smooth lengths are dense enough that the padding is a few percent, and FFTW
handles them with mixed-radix codelets at close to power-of-two per-point cost.

Gaps between smooth numbers are small in the range of interest, so the naive
upward scan is cheap; this runs once per harmonic at plan time.
"""
function next_smooth(n::Integer)
    n > 0 || throw(ArgumentError("n must be a positive integer"))
    n <= 4 && return Int(n)
    k = Int(n)
    while !is_smooth(k)
        k += 1
    end
    return k
end

"""
    finterp_fft_coeffs(numbetween, m, fftlen) -> Vector{ComplexF64}

Precompute the FFT'd interpolation kernel used by the FFT-correlation method.
Mirrors `get_finterp_FFT_coeffs`.

`fftlen` need only be ≥ `numbetween*m`: the method is a *circular* correlation,
so any padded length works and the power-of-two restriction the Python original
imposed is not an algorithmic requirement (see [`next_smooth`](@ref)).
"""
function finterp_fft_coeffs(numbetween::Integer, m::Integer, fftlen::Integer)
    iseven(m) || throw(ArgumentError("m must be even"))
    fftlen >= numbetween * m || throw(ArgumentError("fftlen must be >= numbetween * m"))
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

`fftlen` defaults to the power-of-two padding the Python original used, keeping
this function byte-identical to the oracle; pass [`next_smooth`](@ref) of
`numftbins` for the cheaper padding the production path uses.
"""
function finterp_fft(lobin::Integer, numbins::Integer, numbetween::Integer,
                     ft::AbstractVector, m::Integer; coeffs=nothing,
                     fftlen::Integer=next_pow_of_2((numbins + m) * numbetween))
    m2 = m ÷ 2
    numftbins = (numbins + m) * numbetween
    fftlen >= numftbins || throw(ArgumentError("fftlen must be >= (numbins+m)*numbetween"))
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

"""
    fourier_interpolate(ft::FFTFile, r, m=16) -> ComplexF64

The interpolated complex Fourier amplitude of `ft` at the real-valued Fourier
frequency `r` (in bins), or **exactly zero** when `r` is not representable by
the file: past the Nyquist frequency (`r ≥ N/2`), or close enough to either end
of the stored amplitudes that the `m`-bin kernel window would run off.

This is the one-call form of [`fourier_interp`](@ref): it bundles the range
checks that every caller has to make, so a search loop can be written as a plain
`fourier_interpolate(ft, h*r, m)` per harmonic.  Returning zero rather than
throwing is what the search wants — a harmonic that has run past Nyquist simply
contributes nothing to the coherent sum, which is how the search degrades
gracefully at the top of the band (`reference_profiles`,
`fill_harmonic_row_direct!` and `candidate_profile` all do the same thing, each
with its own copy of these three inequalities).

Zero is unambiguous in practice: a real amplitude is zero only on a
measure-zero set, so `iszero` on the result is a reliable "this harmonic was not
available" test.  A search that needs the count exactly can rely on
availability being *monotone* in `r` — once one harmonic drops out, every higher
one does too.

This is the brute-force, per-point kernel: it recomputes the `m` interpolation
weights for every call.  It is what `bin/toy_coherent_search.jl` and
[`candidate_profile`](@ref) use, and is *not* what the production search runs —
`src/directinterp.jl` tabulates those weights once per harmonic and indexes them
by an exact integer residue, which is the same arithmetic several times faster.
"""
function fourier_interpolate(ft::FFTFile, r::Real, m::Integer=16)
    iseven(m) || throw(ArgumentError("m must be even"))
    r >= 0 || return zero(ComplexF64)
    m2 = m ÷ 2
    # `r_int` is the 0-based bin index, as in the Python original; the kernel
    # reads the 0-based half-open slice [r_int - m2, r_int + m2).
    r_int = floor(Int, r + 1e-15) + 1
    (r_int - m2 >= 0 && r_int + m2 <= length(ft.amps) && r < ft.N / 2) ||
        return zero(ComplexF64)
    return fourier_interp(r, ft.amps, m)
end
