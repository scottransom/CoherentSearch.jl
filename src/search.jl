# Coherent harmonic-summing pulsar search.
#
# Ports the search loop of `coherent_search.coherent_search.main_cli`, but
# restructured for parallelism: the stateful, forward-walking
# `FourierInterpolator` of the Python version is replaced by *independent
# frequency blocks*.  Each block owns its own buffers and can be processed on a
# separate thread with no shared mutable state, so the search scales across
# cores.

using FFTW
using Base.Threads: @threads, nthreads

"""
    SearchParams

Tunable search parameters (defaults match the Python CLI).
"""
Base.@kwdef struct SearchParams
    nharms::Int = 32        # number of harmonics to coherently sum (power of two)
    m::Int = 32             # Fourier bins in the interpolation kernel (even)
    numbetween::Int = 16    # interpolated points between adjacent Fourier bins
    hidr::Float64 = 0.5     # Fourier-bin step at the highest harmonic
    threshold::Float64 = 8.0
end

"""
    Candidate

A detected candidate: barycentric spin frequency (Hz), the peak/|trough|
profile metric, and the fundamental Fourier frequency (bins).
"""
struct Candidate
    freq::Float64
    metric::Float64
    r::Float64
end

"""
    uniform_linear_interp(r, lobin, numbetween, amps) -> ComplexF64

Linear interpolation of the complex `amps` (sampled on the uniform fine grid
`lobin .+ (0:K-1)/numbetween`) at real-valued Fourier frequency `r`.  Equivalent
to `np.interp(r, trs, amps)`, including its clamp-to-endpoints edge behaviour.
"""
@inline function uniform_linear_interp(r::Real, lobin::Integer, numbetween::Integer,
                                       amps::AbstractVector{<:Complex})
    K = length(amps)
    p = (r - lobin) * numbetween          # 0-based fractional index into amps
    if p <= 0
        return ComplexF64(amps[1])
    elseif p >= K - 1
        return ComplexF64(amps[K])
    end
    i0 = floor(Int, p)
    f = p - i0
    @inbounds return ComplexF64(amps[i0 + 1]) * (1 - f) + ComplexF64(amps[i0 + 2]) * f
end

"""
    coherent_profiles(ftprofs, nbins) -> Matrix{Float64}

Inverse-real-FFT the stacked harmonic amplitudes (`(nharms+1, L)`, harmonics
along dim 1) into `nbins`-point real pulse profiles (`(nbins, L)`).  Matches
`np.fft.irfft(ftprofs, axis=1)` with the harmonic axis first (column-major
friendly — see the TODO in the Python original).
"""
coherent_profiles(ftprofs::AbstractMatrix{<:Complex}, nbins::Integer) =
    irfft(ftprofs, nbins, 1)

"""
    block_metrics(ft, rfund, params) -> Vector{Float64}

Compute the coherent-fold peak/|trough| metric for every trial fundamental
Fourier frequency in `rfund`.  Self-contained (own buffers, no shared mutable
state) so it is safe to call concurrently from different threads.  This is the
exact computation the Python oracle reproduces in cross-validation.
"""
function block_metrics(ft::FFTFile, rfund::AbstractVector{<:Real}, params::SearchParams)
    nh = params.nharms
    m = params.m
    nb = params.numbetween
    m2 = m ÷ 2
    L = length(rfund)
    Nhalf = ft.N ÷ 2
    namps = length(ft.amps)

    ftprofs = zeros(ComplexF64, nh + 1, L)   # row 1 is the DC term, left at 0

    for h in 1:nh
        # Frequencies of this harmonic for every trial in the block.
        rmin = rfund[1] * h
        rmax = rfund[end] * h
        lobin = floor(Int, rmin)
        hibin = ceil(Int, rmax) + 1
        numbins = hibin - lobin
        # Skip (leave zeros) if the harmonic runs past Nyquist or off either end
        # of the available amplitudes.
        (lobin >= m2 && (lobin + numbins + m2) <= namps && hibin < Nhalf) || continue

        amps = finterp_fft(lobin, numbins, nb, ft.amps, m)
        @inbounds for j in 1:L
            ftprofs[h + 1, j] = uniform_linear_interp(rfund[j] * h, lobin, nb, amps)
        end
    end

    profs = coherent_profiles(ftprofs, 2nh)

    metrics = Vector{Float64}(undef, L)
    @inbounds for j in 1:L
        mx = -Inf
        mn = Inf
        for k in 1:2nh
            v = profs[k, j]
            mx = ifelse(v > mx, v, mx)
            mn = ifelse(v < mn, v, mn)
        end
        metrics[j] = mx / abs(mn)
    end
    return metrics
end

"""
    search_block(ft, rfund, params; threshold) -> Vector{Candidate}

Search a single block of trial fundamental Fourier frequencies `rfund`,
returning the trials whose metric exceeds `threshold`.
"""
function search_block(ft::FFTFile, rfund::AbstractVector{<:Real}, params::SearchParams;
                      threshold::Real=params.threshold)
    metrics = block_metrics(ft, rfund, params)
    cands = Candidate[]
    @inbounds for j in eachindex(rfund)
        if metrics[j] > threshold
            push!(cands, Candidate(rfund[j] / ft.T, metrics[j], rfund[j]))
        end
    end
    return cands
end

"""
    search(ft, params; lofreq, hifreq, lobin, blocksize, threshold) -> Vector{Candidate}

Run the full coherent harmonic-summing search over `[lofreq, hifreq]` Hz,
parallelised across independent frequency blocks using all available threads.

The `lofreq`/`lobin` precedence matches the Python CLI: `lofreq` is used unless
`lobin` is set to something other than its default of 100.
"""
function search(ft::FFTFile, params::SearchParams=SearchParams();
                lofreq::Real=0.1, hifreq::Real=100.0, lobin::Integer=100,
                blocksize::Integer=1024, threshold::Real=params.threshold)
    lodr = params.hidr / params.nharms
    # Faithful (if brittle) port of the Python precedence rule.
    r_lo = lofreq * ft.T
    if lobin != 100
        r_lo = float(lobin)
    end
    r_hi = hifreq * ft.T

    total = max(0, floor(Int, (r_hi - r_lo) / lodr) + 1)
    total == 0 && return Candidate[]
    nblocks = cld(total, blocksize)

    partials = Vector{Vector{Candidate}}(undef, nblocks)
    @threads for b in 1:nblocks
        i0 = (b - 1) * blocksize
        n = min(blocksize, total - i0)
        rfund = r_lo .+ (i0 .+ (0:(n - 1))) .* lodr
        partials[b] = search_block(ft, rfund, params; threshold=threshold)
    end

    cands = reduce(vcat, partials; init=Candidate[])
    sort!(cands; by=c -> c.freq)
    return cands
end
