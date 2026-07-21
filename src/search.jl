# Coherent harmonic-summing pulsar search.
#
# Ports the search loop of `coherent_search.coherent_search.main_cli`, but
# restructured for parallelism and throughput.  Two layers live here:
#
#   * A simple, allocating *reference* path (`block_metrics`, `search_block`,
#     `coherent_profiles`) that mirrors the Python algorithm one-to-one.  This
#     is what the Python-as-oracle cross-validation pins to machine precision;
#     it is deliberately left unoptimised so it stays easy to audit.
#
#   * An optimised *production* path (`search`) built on the design in
#     `coherent_search_design.md`:
#       - loop #1 over independent fundamental-frequency *chunks*, parallelised
#         across threads (one private `Workspace` per task — no shared mutable
#         state, no `threadid()` indexing);
#       - loop #2 over harmonics, each filling one row of an `(nharms+1)×Nprof`
#         amplitude array via cached-plan, cached-coefficient Fourier
#         interpolation at a per-harmonic `numbetween` matched to `deltar_h`;
#       - loop #3 a single *batched* complex→real inverse FFT of all `Nprof`
#         profiles at once, then a width-sensitive S/N metric per profile.
#
# All FFTW plans and interpolation kernels are built once (single-threaded,
# since FFTW *planning* is not thread-safe) and only *executed* inside the
# parallel region (executing a prebuilt plan on distinct buffers is safe).

using FFTW
using Base.Threads: @spawn, nthreads, Atomic, atomic_add!
using LinearAlgebra: mul!

"""
    SearchParams

Tunable search parameters (defaults match the Python CLI).

`numbetween` is the *floor* on the interpolation oversampling; with `align`
set, low harmonics use a finer `numbetween` matched to their finer `deltar_h`
(see [`harmonic_numbetween`](@ref)).
"""
Base.@kwdef struct SearchParams
    nharms::Int = 32        # number of harmonics to coherently sum
    m::Int = 32             # Fourier bins in the interpolation kernel (even)
    numbetween::Int = 16    # *minimum* interpolated points between Fourier bins
    hidr::Float64 = 0.5     # Fourier-bin step at the highest harmonic
    threshold::Float64 = 8.0
    align::Bool = true      # per-harmonic numbetween matched to deltar_h
    xsignal::Float64 = 0.2  # peak fraction bounding the on-pulse signal region
    metric::Symbol = :non   # width penalty: :non (N_on^pexp) or :sd2 (Σd²^pexp)
    pexp::Float64 = 0.5     # width-penalty exponent (1/2 = calibrated for :non)
    decimations::Vector{Int} = [1]  # harmonic-decimation factors k (see decimation_design.md)
end

"""
    decimation_set(nharms, maxdecim) -> Vector{Int}

The harmonic-decimation factors `k` to search, `1:maxdecim`, keeping only those
that still leave at least two harmonics (`⌊nharms/k⌋ ≥ 2`) to fold.  `k=1` is the
ordinary search; `k>1` re-uses the interpolated harmonic amplitudes to fold at
`k·rf` almost for free (see `decimation_design.md`).
"""
decimation_set(nharms::Integer, maxdecim::Integer) =
    [k for k in 1:max(1, maxdecim) if fld(nharms, k) >= 2]

"""
    Candidate

A detected candidate: barycentric spin frequency (Hz), the width-sensitive S/N
detection metric (see [`snr_metrics`](@ref)), the fundamental Fourier frequency
(bins), and the number of harmonics `nharm` summed in the detection.  For a
decimation-`k` detection `nharm = ⌊nharms/k⌋`, so `k = nharms ÷ nharm` — i.e.
`nharm` records which decimation found the candidate.  The pulse period is
`1/freq`.
"""
struct Candidate
    freq::Float64
    metric::Float64
    r::Float64
    nharm::Int
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

# ---------------------------------------------------------------------------
# Detection metric (port of `snr_metric` from coherent_search.py)
# ---------------------------------------------------------------------------

"""
    chunk_ngoodbins(ft, nharms, rmean) -> Float64

The number of harmonics that carry real Fourier data for a chunk of trial
fundamentals whose mean Fourier frequency is `rmean`: `min(Nyquist/rmean,
nharms)`.  This sets the expected profile-noise RMS in [`_profile_snr`](@ref).
Matches `min(ft.N/2/rstosearch.mean(), args.nharms)` in the Python code.
"""
@inline chunk_ngoodbins(ft::FFTFile, nharms::Integer, rmean::Real) =
    min(ft.N / 2 / rmean, float(nharms))

# ---------------------------------------------------------------------------
# Fast exact median of a small scratch vector (quickselect).
#
# The profile median is the single hottest operation in the search: it runs
# once per trial per decimation (~1e8 times), always on a short (nbins = 20..120)
# *cold* profile column.  The default `sort!` there is a radix sort whose
# data-dependent branches mispredict badly on cold data; a Lomuto quickselect
# for just the two central order statistics is ~2.4x faster (measured, nbins=120)
# and returns the *identical* median value, so every oracle/equivalence pin is
# unaffected.  `_median!` destroys `v[1:n]` (it is scratch, never read after).
# ---------------------------------------------------------------------------

@inline _swap!(v, a, b) = (@inbounds t = v[a]; @inbounds v[a] = v[b]; @inbounds v[b] = t; nothing)

# Place the k-th smallest of v[lo:hi] at v[k] (with v[lo:k-1] all ≤ it); Lomuto
# partition, median-of-3 pivot, insertion-sort cutoff for small ranges.
@inline function _select!(v::AbstractVector, lo::Int, hi::Int, k::Int)
    @inbounds while lo < hi
        if hi - lo < 16                       # small range: insertion sort, done
            for i in lo+1:hi
                x = v[i]; j = i - 1
                while j >= lo && v[j] > x
                    v[j+1] = v[j]; j -= 1
                end
                v[j+1] = x
            end
            return
        end
        mid = (lo + hi) >>> 1                  # median-of-3 pivot into v[hi]
        v[mid] < v[lo] && _swap!(v, mid, lo)
        v[hi]  < v[lo] && _swap!(v, hi, lo)
        v[mid] < v[hi] && _swap!(v, mid, hi)
        pivot = v[hi]
        i = lo - 1
        for jj in lo:hi-1
            if v[jj] <= pivot
                i += 1; _swap!(v, i, jj)
            end
        end
        _swap!(v, i+1, hi)
        p = i + 1
        p == k && return
        p < k ? (lo = p + 1) : (hi = p - 1)
    end
end

"""
    _median!(v, n) -> Float64

Exact median of `v[1:n]` (partially reorders `v`, which must be scratch).  For
even `n` (the search always has `n = 2*nharms`) it is the mean of the two central
order statistics; for odd `n`, the middle one.  Same value a full sort yields.
"""
@inline function _median!(v::AbstractVector{Float64}, n::Int)
    half = n >>> 1
    _select!(v, 1, n, half + 1)               # upper-median order stat at v[half+1]
    isodd(n) && return @inbounds v[half + 1]
    upper = @inbounds v[half + 1]
    lower = @inbounds v[1]                     # lower median = max of the half below
    @inbounds for i in 2:half
        v[i] > lower && (lower = v[i])
    end
    return 0.5 * (lower + upper)
end

"""
    _profile_snr(profs, j, medbuf, nbins, invrms, scale, xsignal, metric, pexp) -> Float64

Width-sensitive detection metric for profile column `j` of the `(nbins × *)`
real profile matrix.  Ports `snr_metric` from `coherent_search.py`:

    metric = sum_on(prof - median) / rms / width^pexp

The **signal** `sum_on(prof - median)` is the summed excess over the median of
every *on-pulse* bin — the bins that rise above a fraction `xsignal` of the
peak-over-median height (`prof - median > xsignal*(max - median)`).  Summing
over this set (rather than the whole, zero-mean profile) keeps the signal a
stable measure of pulsed flux that does not grow with `nbins`, and it naturally
captures multi-component pulses (two narrow peaks with a valley between them
each contribute), which a single boxcar would miss.

The **width** penalty is taken over that same on-pulse set and selected by
`metric`:

  * `:non` — `width = N_on`, the *count* of on-pulse bins (a duty-cycle penalty).
    `pexp = 1/2` is the calibrated matched-filter normalisation (a true
    equivalent-σ); larger `pexp` more aggressively penalises high-duty-cycle
    signals (e.g. broad or many-toothed RFI) while leaving narrow pulses — even
    widely separated multi-component or interpulse pulsars — untouched, since it
    keys on how *many* bins are lit, not *where* they sit.
  * `:sd2` — `width = Σ d²`, the summed squared *modular* phase distance of the
    on-pulse bins from the peak (distances wrap, the profile being periodic).
    This penalises phase *spread*: larger `pexp` down-weights scattered/broad
    profiles harder, but also down-weights genuinely separated components.

`rms = 1/sqrt(2*ngoodbins+1)` is supplied as `invrms = 1/rms`, and `medbuf` is a
length-`nbins` scratch buffer (reused, not read on entry).  `scale` carries the
profile normalisation: `1` for a true (normalised) irfft, `1/nbins` for the
unnormalised `brfft` used in the hot loop.  The median, argmax, on-pulse set,
and width are all scale-invariant, so only the linear `signal` term needs
`scale` — which is why the fast path can skip the irfft normalisation entirely.
"""
@inline function _profile_snr(profs::AbstractMatrix{Float64}, j::Integer,
                              medbuf::Vector{Float64}, nbins::Int, invrms::Float64,
                              scale::Float64, xsignal::Float64, metric::Symbol, pexp::Float64)
    col = @view profs[:, j]
    half = nbins >>> 1                           # nbins is even (= 2*nharms)
    # One fused pass: copy the column into the median scratch AND find the peak
    # (first-max tie-break, as np.argmax), instead of a separate copyto! + scan.
    mx = -Inf; mxind = 1
    @inbounds for k in 1:nbins
        v = col[k]
        medbuf[k] = v
        if v > mx
            mx = v; mxind = k
        end
    end
    med = _median!(medbuf, nbins)                # quickselect (see _median!)
    peak = mx - med                              # peak height over the baseline
    sig_thr = med + xsignal * peak               # on-pulse level

    signal = 0.0
    non = 0
    sumsq = 0.0
    @inbounds for k in 1:nbins
        col[k] > sig_thr || continue             # restrict to the on-pulse set
        signal += col[k] - med
        non += 1
        d = mxind - k
        d >  half && (d -= nbins)                # wrap to the nearest periodic image
        d < -half && (d += nbins)
        sumsq += d * d
    end
    w = metric === :sd2 ? sumsq : Float64(non)   # phase-spread vs duty-cycle penalty
    w < 1.0 && (w = 1.0)                          # floor (lone peak -> width 1)
    # Width penalty w^pexp; special-case the two calibrated exponents (0.5 = the
    # default matched filter, and 1.0) to skip the generic `pow`.
    denom = pexp == 0.5 ? sqrt(w) : (pexp == 1.0 ? w : w^pexp)
    return scale * signal * invrms / denom
end

"""
    snr_metrics(profs, ngoodbins; xsignal=0.2, metric=:non, pexp=0.5) -> Vector{Float64}

Width-sensitive detection metric for every profile (column) of the `(nbins × L)`
real profile matrix `profs`, which must be a true (normalised) irfft.  Public
port of `snr_metric` from the Python `coherent_search` (note the profiles are
columns here, rows in Python).  `ngoodbins` sets the noise RMS
`1/sqrt(2*ngoodbins+1)` (see [`chunk_ngoodbins`](@ref)); `xsignal` is the peak
fraction bounding the on-pulse region; `metric` (`:non` or `:sd2`) and `pexp`
select and tune the width penalty (see [`_profile_snr`](@ref)).
"""
function snr_metrics(profs::AbstractMatrix{<:Real}, ngoodbins::Real;
                     xsignal::Real=0.2, metric::Symbol=:non, pexp::Real=0.5)
    metric in (:non, :sd2) || throw(ArgumentError("metric must be :non or :sd2, got :$metric"))
    nbins, L = size(profs)
    invrms = sqrt(2 * ngoodbins + 1)
    medbuf = Vector{Float64}(undef, nbins)
    P = profs isa Matrix{Float64} ? profs : convert(Matrix{Float64}, profs)
    return [_profile_snr(P, j, medbuf, nbins, invrms, 1.0, Float64(xsignal), metric, Float64(pexp))
            for j in 1:L]
end

# ---------------------------------------------------------------------------
# Reference path (mirrors the Python algorithm; pinned by the cross-validation)
# ---------------------------------------------------------------------------

"""
    coherent_profiles(ftprofs, nbins) -> Matrix{Float64}

Inverse-real-FFT the stacked harmonic amplitudes (`(nharms+1, L)`, harmonics
along dim 1) into `nbins`-point real pulse profiles (`(nbins, L)`).  Matches
`np.fft.irfft(ftprofs, axis=1)` with the harmonic axis first.
"""
coherent_profiles(ftprofs::AbstractMatrix{<:Complex}, nbins::Integer) =
    irfft(ftprofs, nbins, 1)

"""
    reference_profiles(ft, rfund, params) -> Matrix{Float64}

Build the `(2*nharms, L)` real coherent-fold pulse profiles (one column per
trial fundamental in `rfund`) via the simple, allocating reference path: one
`finterp_fft` per harmonic, linear interpolation onto the trial frequencies, and
a single (normalised) `irfft`.  This is the exact profile computation the Python
oracle reproduces, kept separate from the detection metric so each can be
cross-validated on its own.
"""
function reference_profiles(ft::FFTFile, rfund::AbstractVector{<:Real}, params::SearchParams)
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

    return coherent_profiles(ftprofs, 2nh)
end

"""
    block_metrics(ft, rfund, params) -> Vector{Float64}

Compute the coherent-fold detection S/N (see [`snr_metrics`](@ref)) for every
trial fundamental Fourier frequency in `rfund`.  Self-contained reference
implementation built on [`reference_profiles`](@ref); this is the exact
computation the Python oracle reproduces in cross-validation.
"""
function block_metrics(ft::FFTFile, rfund::AbstractVector{<:Real}, params::SearchParams)
    nh = params.nharms
    nbins = 2nh
    L = length(rfund)
    profs = reference_profiles(ft, rfund, params)

    rmean = sum(rfund) / L
    invrms = sqrt(2 * chunk_ngoodbins(ft, nh, rmean) + 1)
    medbuf = Vector{Float64}(undef, nbins)
    metrics = Vector{Float64}(undef, L)
    for j in 1:L
        metrics[j] = _profile_snr(profs, j, medbuf, nbins, invrms, 1.0, params.xsignal, params.metric, params.pexp)
    end
    return metrics
end

"""
    search_block(ft, rfund, params; threshold) -> Vector{Candidate}

Search a single block of trial fundamental Fourier frequencies `rfund` using
the reference [`block_metrics`](@ref), returning trials above `threshold`.
"""
function search_block(ft::FFTFile, rfund::AbstractVector{<:Real}, params::SearchParams;
                      threshold::Real=params.threshold)
    metrics = block_metrics(ft, rfund, params)
    cands = Candidate[]
    @inbounds for j in eachindex(rfund)
        if metrics[j] > threshold
            push!(cands, Candidate(rfund[j] / ft.T, metrics[j], rfund[j], params.nharms))
        end
    end
    return cands
end

# ---------------------------------------------------------------------------
# Optimised path: per-harmonic plans + per-thread workspaces + batched irfft
# ---------------------------------------------------------------------------

"""
    harmonic_numbetween(h, nharms, hidr, minnb) -> Int

Interpolation oversampling for harmonic `h`.  The candidate fundamentals are
stepped by `deltar = hidr/nharms` bins, so harmonic `h` is sampled every
`deltar_h = hidr*h/nharms` bins.  The natural grid that lands one finterp point
per candidate at this harmonic has `numbetween = nharms/(hidr*h)` (= `2*nharms`
at `h=1`, down to `2` at `h=nharms` for the default `hidr=0.5`).

We never go *below* `minnb`, so accuracy is at least as good as the fixed-
`numbetween` reference everywhere; the low harmonics (where the natural value
exceeds `minnb`) simply get a finer, more accurate grid.  The whole schedule is
a starting heuristic — the throughput/accuracy sweet spot is meant to be tuned
by the benchmark + accuracy cross-validation.
"""
@inline harmonic_numbetween(h::Integer, nharms::Integer, hidr::Real, minnb::Integer) =
    max(minnb, round(Int, nharms / (hidr * h)))

"""
    HarmonicPlan

Per-harmonic interpolation recipe, built once and shared read-only across all
threads: the harmonic number `h`, its oversampling `nb`, the FFT length
`fftlen` sized to cover a whole chunk in a single transform, and the FFT'd
interpolation kernel `coeffs` (already scaled by `1/fftlen` so the inverse can
use an unnormalised `bfft`).
"""
struct HarmonicPlan
    h::Int
    nb::Int
    fftlen::Int
    coeffs::Vector{ComplexF64}
end

"""
    build_harmonic_plans(params, Nprof) -> Vector{HarmonicPlan}

Size each harmonic's `numbetween` and `fftlen` for chunks of `Nprof` trial
fundamentals, and precompute its FFT'd interpolation kernel.  When
`params.align` is false every harmonic uses `params.numbetween` (the fixed grid
of the reference path), which makes the production path reproduce
[`block_metrics`](@ref) to machine precision — used by the test suite.
"""
function build_harmonic_plans(params::SearchParams, Nprof::Integer)
    plans = HarmonicPlan[]
    for h in 1:params.nharms
        nb = params.align ?
             harmonic_numbetween(h, params.nharms, params.hidr, params.numbetween) :
             params.numbetween
        dh = params.hidr * h / params.nharms             # deltar_h (bins/trial)
        span_max = (Nprof - 1) * dh                       # widest a chunk can span
        numbins_max = ceil(Int, span_max) + 2             # +2: linear-interp slack
        fftlen = next_pow_of_2((numbins_max + params.m) * nb)
        # Fold the inverse-FFT 1/fftlen normalisation into the kernel so the
        # hot loop can use plain (unnormalised) bfft via mul!.
        coeffs = finterp_fft_coeffs(nb, params.m, fftlen) ./ fftlen
        push!(plans, HarmonicPlan(h, nb, fftlen, coeffs))
    end
    return plans
end

"""
    FFTScratch

Reusable buffers and prebuilt forward/backward complex FFT plans for one
`fftlen`.  Harmonics that share an `fftlen` share a scratch within a workspace.
"""
struct FFTScratch{P,Q}
    ftarr::Vector{ComplexF64}   # zero-stuffed input (reused, kept zeroed)
    spec::Vector{ComplexF64}    # forward-FFT output / kernel product
    corr::Vector{ComplexF64}    # backward-FFT output (the interpolated grid)
    fwd::P                       # plan_fft(ftarr)
    bwd::Q                       # plan_bfft(spec)   (unnormalised; see coeffs)
end

function FFTScratch(fftlen::Integer)
    ftarr = zeros(ComplexF64, fftlen)
    spec  = Vector{ComplexF64}(undef, fftlen)
    corr  = Vector{ComplexF64}(undef, fftlen)
    fwd = plan_fft(ftarr; flags=FFTW.MEASURE)
    bwd = plan_bfft(spec;  flags=FFTW.MEASURE)
    fill!(ftarr, 0)   # MEASURE planning may have dirtied the buffer
    return FFTScratch(ftarr, spec, corr, fwd, bwd)
end

"""
    DecimBuf

Per-decimation-factor `k` scratch for the harmonic-decimation multi-frequency
search: the decimated amplitude stack `dftprofs` (`(Hₖ+1, Nprof)`, `Hₖ =
⌊nharms/k⌋`), the real profile array `dprofs` (`(2Hₖ, Nprof)`), a per-profile
median buffer, and the batched complex→real inverse plan.  Built once per
`Workspace`; never shared.  Row `j+1` of `dftprofs` is filled from base harmonic
`j·k` (row `j·k+1` of the workspace `ftprofs`); the DC row stays zero.
"""
struct DecimBuf{B}
    k::Int
    Hk::Int
    dftprofs::Matrix{ComplexF64}  # (Hk+1, Nprof)
    dprofs::Matrix{Float64}       # (2*Hk, Nprof)
    medbuf::Vector{Float64}       # (2*Hk,)
    brfftplan::B                   # plan_brfft(dftprofs, 2*Hk, 1)
end

function DecimBuf(k::Integer, nharms::Integer, Nprof::Integer)
    Hk = fld(nharms, k)
    dftprofs = zeros(ComplexF64, Hk + 1, Nprof)
    dprofs   = Matrix{Float64}(undef, 2Hk, Nprof)
    medbuf   = Vector{Float64}(undef, 2Hk)
    brfftplan = plan_brfft(dftprofs, 2Hk, 1; flags=FFTW.MEASURE)
    fill!(dftprofs, 0)   # MEASURE planning may have dirtied the buffer
    return DecimBuf(Int(k), Hk, dftprofs, dprofs, medbuf, brfftplan)
end

"""
    Workspace

Everything one task needs to process a chunk with zero allocation in the hot
loop: an `FFTScratch` per distinct harmonic `fftlen`, the stacked-harmonic
amplitude array `ftprofs`, the real profile array `profs`, the prebuilt batched
complex→real inverse plan, and a [`DecimBuf`](@ref) for each harmonic-decimation
factor `k > 1`.  One `Workspace` per task; never shared.
"""
# Parameterised on the *concrete* scratch (`S`), inverse-plan (`B`), and
# decimation-buffer (`D`) types so field access stays type-stable: an untyped
# `Dict{Int,FFTScratch}` / `Vector{DecimBuf}` would make `sc.fwd`/`sc.bwd`/
# `db.brfftplan` `::Any`, turning every `mul!` in the hot loop into a dynamic
# dispatch (and boxing its result).  All `cFFTWPlan{ComplexF64,…}` share one
# concrete type regardless of length, so a single `S`/`B` covers every fftlen.
struct Workspace{S<:FFTScratch, B, D<:DecimBuf}
    scratch::Dict{Int,S}
    ftprofs::Matrix{ComplexF64}   # (nharms+1, Nprof)
    profs::Matrix{Float64}        # (2*nharms, Nprof)
    medbuf::Vector{Float64}       # (2*nharms,) scratch for the per-profile median
    brfftplan::B                   # plan_brfft(ftprofs, 2*nharms, 1)
    decims::Vector{D}              # one per decimation factor k > 1
end

function Workspace(params::SearchParams, hplans::Vector{HarmonicPlan}, Nprof::Integer)
    nh = params.nharms
    # Build the Dict from concrete pairs so it infers Dict{Int,FFTScratch{P,Q}}
    # (a concrete value type) rather than the abstract Dict{Int,FFTScratch}.
    scratch = Dict(fl => FFTScratch(fl) for fl in unique(hp.fftlen for hp in hplans))
    ftprofs = zeros(ComplexF64, nh + 1, Nprof)
    profs   = Matrix{Float64}(undef, 2nh, Nprof)
    medbuf  = Vector{Float64}(undef, 2nh)
    brfftplan = plan_brfft(ftprofs, 2nh, 1; flags=FFTW.MEASURE)
    fill!(ftprofs, 0)   # MEASURE planning may have dirtied the buffer
    # A DecimBuf per k > 1 (k = 1 is the base ftprofs/profs above).  `map` (not a
    # `DecimBuf[...]` comprehension) keeps the element type the concrete
    # `DecimBuf{B}` even when empty, so `Workspace`'s `D<:DecimBuf` stays concrete.
    decims = map(k -> DecimBuf(k, nh, Nprof), filter(>(1), params.decimations))
    return Workspace(scratch, ftprofs, profs, medbuf, brfftplan, decims)
end

"""
    interp_tile!(sc, coeffs, lobin, numbins, nb, ft, m) -> view

Allocation-free FFT-correlation Fourier interpolation onto the uniform grid
`lobin .+ (0:numbins*nb-1)/nb`, writing into `sc`'s buffers and returning a view
of the valid region of `sc.corr`.  Mirrors [`finterp_fft`](@ref) exactly (the
`coeffs` carry the `1/fftlen` factor, so `bfft` reproduces `ifft`).
"""
function interp_tile!(sc::FFTScratch, coeffs::Vector{ComplexF64}, lobin::Integer,
                      numbins::Integer, nb::Integer, ft::AbstractVector, m::Integer)
    m2 = m ÷ 2
    ftarr = sc.ftarr
    nstuff = numbins + m
    # Zero-stuff the raw bins every nb samples: ftarr[i*nb+1] = ft[lobin-m2+1+i].
    @inbounds for i in 0:(nstuff - 1)
        ftarr[i * nb + 1] = ft[lobin - m2 + 1 + i]
    end
    mul!(sc.spec, sc.fwd, ftarr)
    @inbounds @. sc.spec = sc.spec * coeffs
    mul!(sc.corr, sc.bwd, sc.spec)
    # Restore zeros at exactly the positions we wrote (cheaper than a full wipe).
    @inbounds for i in 0:(nstuff - 1)
        ftarr[i * nb + 1] = 0
    end
    return @view sc.corr[(m2 * nb + 1):((m2 + numbins) * nb)]
end

"""
    fill_harmonic_row!(ws, hp, ft, params, r0, n)

Fill row `hp.h+1`, columns `1:n`, of `ws.ftprofs` with the interpolated complex
amplitude of harmonic `hp.h` at the `n` trial frequencies `r0 .+ (0:n-1)*dh`
(`dh = hidr*h/nharms`).  Leaves the row at zero if the harmonic runs off the end
of the available amplitudes or past Nyquist.
"""
function fill_harmonic_row!(ws::Workspace, hp::HarmonicPlan, ft::FFTFile,
                            params::SearchParams, r0::Real, n::Integer)
    h = hp.h
    nb = hp.nb
    m = params.m
    m2 = m ÷ 2
    Nhalf = ft.N ÷ 2
    namps = length(ft.amps)
    dh = params.hidr * h / params.nharms

    lobin = floor(Int, r0)
    rlast = r0 + (n - 1) * dh
    numbins = floor(Int, rlast) - lobin + 2     # +2 keeps the last trial strictly inside

    # Range check (per chunk): the kernel reaches m2 bins beyond [lobin, lobin+numbins).
    (lobin >= m2 && (lobin + numbins + m2) <= namps && (lobin + numbins) < Nhalf) || return

    sc = ws.scratch[hp.fftlen]
    amps = interp_tile!(sc, hp.coeffs, lobin, numbins, nb, ft.amps, m)
    @inbounds for k in 1:n
        ws.ftprofs[h + 1, k] = uniform_linear_interp(r0 + (k - 1) * dh, lobin, nb, amps)
    end
    return
end

"""
    fill_chunk_profiles!(ws, hplans, ft, params, rstart, lodr, n)

Fill `ws.ftprofs` (zeroed first) and inverse-FFT it into `ws.profs` for a chunk
of `n` trial fundamentals starting at fundamental Fourier frequency `rstart`,
stepping by `lodr` bins.  After this call `ws.profs[:, 1:n]` holds the real
coherent-fold profiles.
"""
function fill_chunk_profiles!(ws::Workspace, hplans::Vector{HarmonicPlan}, ft::FFTFile,
                              params::SearchParams, rstart::Real, lodr::Real, n::Integer)
    fill!(ws.ftprofs, 0)
    for hp in hplans
        fill_harmonic_row!(ws, hp, ft, params, hp.h * rstart, n)
    end
    # One batched complex→real transform for all Nprof profiles at once.  This
    # is an unnormalised `brfft` (= Nbins × a true irfft); the S/N metric folds
    # the missing 1/Nbins into its `scale` argument (see `_profile_snr`), so the
    # profiles here are left unnormalised.
    mul!(ws.profs, ws.brfftplan, ws.ftprofs)
    return
end

# ---------------------------------------------------------------------------
# Per-block metric statistics (opt-in diagnostic; see the CLI `--metricstats`)
# ---------------------------------------------------------------------------

"""
    BlockMetricStats

Summary statistics of the detection metric over one processed *block* of trial
fundamentals at one harmonic-decimation factor `k` (`Hk = ⌊nharms/k⌋` harmonics
summed into `nbins = 2*Hk`-bin profiles).  `ngoodbins` is the per-block noise
normalisation ([`chunk_ngoodbins`](@ref)); `flo`/`fhi` bound the *searched* spin
frequency (Hz) — for `k>1` this is `k×` the fundamental.  `n` is the number of
trials counted (decimation trials whose `k·rf` reaches Nyquist are excluded, so
`n` can be below the block size for the low-`k`, high-frequency blocks).
Collected only when [`search`](@ref) is passed a `metricstats` sink.

These exist to expose how the metric's noise floor depends on the profile bin
count.  With the default `:non`/`pexp=0.5` penalty the pure-noise metric scales
~`√nbins = √(2·Hk)`, so the low-`k` (more-bin) decimations sit at a
systematically higher floor and dominate the candidate list at a fixed
`threshold`.  Comparing the per-`k` distributions is how one should choose a
`--threshold` (and see that it is *not* comparable across decimations).
"""
struct BlockMetricStats
    block::Int
    k::Int
    Hk::Int
    nbins::Int
    ngoodbins::Float64
    flo::Float64
    fhi::Float64
    n::Int
    min::Float64
    median::Float64
    mean::Float64
    std::Float64
    max::Float64
end

# Exact min/median/mean/std/max of `v` (which is treated as scratch: `_median!`
# reorders it, so this is called last, after the linear passes).
function _block_stats(block::Integer, k::Integer, Hk::Integer, nbins::Integer,
                      ngoodbins::Real, flo::Real, fhi::Real, v::Vector{Float64})
    n = length(v)
    n == 0 && return BlockMetricStats(block, k, Hk, nbins, ngoodbins, flo, fhi,
                                      0, NaN, NaN, NaN, NaN, NaN)
    vmin = v[1]; vmax = v[1]; s = 0.0
    @inbounds for x in v
        x < vmin && (vmin = x)
        x > vmax && (vmax = x)
        s += x
    end
    mean = s / n
    ss = 0.0
    @inbounds for x in v
        d = x - mean; ss += d * d
    end
    std = n > 1 ? sqrt(ss / (n - 1)) : 0.0
    med = _median!(v, n)                        # destroys v (scratch); done last
    return BlockMetricStats(Int(block), Int(k), Int(Hk), Int(nbins), Float64(ngoodbins),
                            Float64(flo), Float64(fhi), n, vmin, med, mean, std, vmax)
end

"""
    metricstats_summary(stats) -> Vector{<:NamedTuple}

Aggregate a vector of [`BlockMetricStats`](@ref) into one row per decimation
factor `k` (sorted by `k`).  `min`/`max`/`mean`/`std` are the exact global
values across all blocks of that `k` (the global std is reconstructed from the
per-block moments); `median` is the median of the per-block medians (a robust
stand-in for the exact global median, which the per-block reduction does not
retain); `blockmax_mean` is the mean of the per-block maxima — the typical
worst-case noise excursion per block, which is what a false-alarm-driven
threshold trades against.
"""
function metricstats_summary(stats::AbstractVector{BlockMetricStats})
    ks = sort!(unique(s.k for s in stats))
    rows = map(ks) do k
        rs = filter(s -> s.k == k && s.n > 0, stats)
        isempty(rs) && return (k=k, Hk=0, nbins=0, nblocks=0, ntrials=0,
                               min=NaN, median=NaN, mean=NaN, std=NaN,
                               blockmax_mean=NaN, max=NaN)
        N = sum(s.n for s in rs)
        gmean = sum(s.mean * s.n for s in rs) / N
        # Global Σ(x-gmean)² = Σ_block[(n-1)·s² + n·(mean-gmean)²]; exact.
        ss = sum(((s.n - 1) * s.std^2 + s.n * (s.mean - gmean)^2) for s in rs)
        gstd = N > 1 ? sqrt(ss / (N - 1)) : 0.0
        blockmeds = sort!([s.median for s in rs])
        L = length(blockmeds)
        gmed = isodd(L) ? blockmeds[(L + 1) ÷ 2] :
               0.5 * (blockmeds[L ÷ 2] + blockmeds[L ÷ 2 + 1])
        (k=k, Hk=rs[1].Hk, nbins=rs[1].nbins, nblocks=length(rs), ntrials=N,
         min=minimum(s.min for s in rs), median=gmed, mean=gmean, std=gstd,
         blockmax_mean=sum(s.max for s in rs) / length(rs),
         max=maximum(s.max for s in rs))
    end
    return rows
end

"""
    decim_pass!(out, ws, db, ft, params, rstart, lodr, n; threshold, block, stats)

Harmonic-decimation multi-frequency pass for factor `db.k`: re-use the base
harmonic amplitudes already in `ws.ftprofs` to fold at `k·rf` for each of the
`n` trial fundamentals `rstart .+ (0:n-1)*lodr`.  Gathers every `k`-th base
harmonic into `db`'s compact stack, inverse-FFTs all `n` decimated profiles at
once, and appends above-`threshold` [`Candidate`](@ref)s (tagged with `Hₖ`
harmonics) to `out`.  Trials whose decimated fundamental `k·rf` reaches Nyquist
are skipped.  Nearly free relative to the interpolation the base pass paid.

When a `stats` sink (a `Vector{BlockMetricStats}`) is passed, every folded
trial's metric is also gathered and a [`BlockMetricStats`](@ref) for this
`(block, k)` is appended — the opt-in `--metricstats` diagnostic.
"""
function decim_pass!(out::Vector{Candidate}, ws::Workspace, db::DecimBuf, ft::FFTFile,
                     params::SearchParams, rstart::Real, lodr::Real, n::Integer;
                     threshold::Real=params.threshold, block::Integer=0,
                     stats::Union{Nothing,Vector{BlockMetricStats}}=nothing)
    k = db.k
    Hk = db.Hk
    nbins = 2Hk
    src = ws.ftprofs
    # Row j+1 of the decimated stack is base harmonic j*k (row j*k+1); DC stays 0.
    @inbounds for j in 1:Hk
        rowbase = j * k + 1
        @views db.dftprofs[j + 1, 1:n] .= src[rowbase, 1:n]
    end
    mul!(db.dprofs, db.brfftplan, db.dftprofs)

    rmean = rstart + (n - 1) * lodr / 2
    ngood = chunk_ngoodbins(ft, Hk, k * rmean)
    invrms = sqrt(2 * ngood + 1)
    nyq = ft.N / 2
    mbuf = stats === nothing ? nothing : Float64[]     # gather metrics if requested
    @inbounds for j in 1:n
        r_dec = k * (rstart + (j - 1) * lodr)
        r_dec < nyq || continue                       # fundamental past Nyquist
        mval = _profile_snr(db.dprofs, j, db.medbuf, nbins, invrms,
                            1.0 / nbins, params.xsignal, params.metric, params.pexp)
        mbuf === nothing || push!(mbuf, mval)
        if mval > threshold
            push!(out, Candidate(r_dec / ft.T, mval, r_dec, Hk))
        end
    end
    if mbuf !== nothing && !isempty(mbuf)
        # Valid trials are the prefix j=1..length(mbuf) (r_dec increases with j).
        flo = k * rstart / ft.T
        fhi = k * (rstart + (length(mbuf) - 1) * lodr) / ft.T
        push!(stats, _block_stats(block, k, Hk, nbins, ngood, flo, fhi, mbuf))
    end
    return
end

"""
    chunk_metrics(ft, params, rstart, n; lodr) -> Vector{Float64}

Single-threaded convenience that runs the optimised path over one chunk of `n`
trial fundamentals starting at `rstart` and returns the S/N metric for each.
With `params.align = false` this reproduces [`block_metrics`](@ref) to machine
precision; it is the bridge the test suite uses to pin the optimised path to the
oracle-validated reference.
"""
function chunk_metrics(ft::FFTFile, params::SearchParams, rstart::Real, n::Integer;
                       lodr::Real = params.hidr / params.nharms)
    nh = params.nharms
    nbins = 2nh
    hplans = build_harmonic_plans(params, n)
    ws = Workspace(params, hplans, n)
    fill_chunk_profiles!(ws, hplans, ft, params, rstart, lodr, n)
    rmean = rstart + (n - 1) * lodr / 2
    invrms = sqrt(2 * chunk_ngoodbins(ft, nh, rmean) + 1)
    return [_profile_snr(ws.profs, j, ws.medbuf, nbins, invrms, 1.0 / nbins, params.xsignal, params.metric, params.pexp) for j in 1:n]
end

"""
    search(ft, params; lofreq, hifreq, lobin, blocksize, threshold) -> Vector{Candidate}

Run the full coherent harmonic-summing search over `[lofreq, hifreq]` Hz,
parallelised across independent fundamental-frequency chunks of `blocksize`
trials each.  Each task owns a private [`Workspace`](@ref); all FFTW plans and
interpolation kernels are built once before the parallel region.

The `lofreq`/`lobin` precedence matches the Python CLI: `lofreq` is used unless
`lobin` is set to something other than its default of 100.

Candidate post-processing: near-identical clusters are collapsed by
[`remove_duplicates`](@ref) (`remove`, `dr_tol`), then harmonically-related
candidates by [`remove_harmonics`](@ref) (`harm_remove`, `numharm`, `harm_tol`).
`progress` (`:none`, `:text`, or `:bar`) prints a chunk-completion meter to
`stderr`.

If a `metricstats` vector is supplied, per-block, per-decimation
[`BlockMetricStats`](@ref) (the metric distribution over *every* trial, not just
those above `threshold`) are collected and appended to it (sorted by block then
`k`) after the search — the opt-in `--metricstats` diagnostic.  The candidate
results are identical whether or not `metricstats` is collected.
"""
function search(ft::FFTFile, params::SearchParams=SearchParams();
                lofreq::Real=0.1, hifreq::Real=100.0, lobin::Integer=100,
                blocksize::Integer=2048, threshold::Real=params.threshold,
                remove::Bool=true, dr_tol::Real=1.0,
                harm_remove::Bool=true, numharm::Integer=16, harm_tol::Real=1.0,
                progress::Symbol=:none,
                metricstats::Union{Nothing,Vector{BlockMetricStats}}=nothing)
    progress in (:none, :text, :bar) ||
        throw(ArgumentError("progress must be :none, :text or :bar, got :$progress"))
    FFTW.set_num_threads(1)   # parallelise at the Julia-task level, not inside FFTW
    lodr = params.hidr / params.nharms
    nbins = 2 * params.nharms
    # Faithful (if brittle) port of the Python precedence rule.
    r_lo = lofreq * ft.T
    if lobin != 100
        r_lo = float(lobin)
    end
    r_hi = hifreq * ft.T

    total = max(0, floor(Int, (r_hi - r_lo) / lodr) + 1)
    total == 0 && return Candidate[]
    Nprof = max(1, Int(blocksize))
    nchunks = cld(total, Nprof)

    hplans = build_harmonic_plans(params, Nprof)
    nt = max(1, min(nthreads(), nchunks))
    # Planning is not thread-safe: build all workspaces serially, here.
    workspaces = [Workspace(params, hplans, Nprof) for _ in 1:nt]

    collect_stats = metricstats !== nothing
    results = Vector{Vector{Candidate}}(undef, nt)
    statparts = collect_stats ? Vector{Vector{BlockMetricStats}}(undef, nt) : nothing
    done = Atomic{Int}(0)     # chunks completed across all tasks (for the progress meter)
    @sync for t in 1:nt
        @spawn begin
            ws = workspaces[t]
            out = Candidate[]
            stats = collect_stats ? BlockMetricStats[] : nothing
            mbuf = collect_stats ? Vector{Float64}(undef, Nprof) : nothing
            c = t
            while c <= nchunks
                i0 = (c - 1) * Nprof
                n = min(Nprof, total - i0)
                rstart = r_lo + i0 * lodr
                fill_chunk_profiles!(ws, hplans, ft, params, rstart, lodr, n)
                rmean = rstart + (n - 1) * lodr / 2
                ngood = chunk_ngoodbins(ft, params.nharms, rmean)
                invrms = sqrt(2 * ngood + 1)
                for j in 1:n
                    metric = _profile_snr(ws.profs, j, ws.medbuf, nbins, invrms, 1.0 / nbins, params.xsignal, params.metric, params.pexp)
                    collect_stats && (mbuf[j] = metric)
                    if metric > threshold
                        rf = rstart + (j - 1) * lodr
                        push!(out, Candidate(rf / ft.T, metric, rf, params.nharms))
                    end
                end
                if collect_stats
                    flo = rstart / ft.T
                    fhi = (rstart + (n - 1) * lodr) / ft.T
                    push!(stats, _block_stats(c, 1, params.nharms, nbins, ngood, flo, fhi, mbuf[1:n]))
                end
                # Harmonic-decimation multi-frequency passes (k > 1), re-using the
                # base harmonic amplitudes already in ws.ftprofs (see decimation_design.md).
                for db in ws.decims
                    decim_pass!(out, ws, db, ft, params, rstart, lodr, n;
                                threshold=threshold, block=c, stats=stats)
                end
                atomic_add!(done, 1)
                # One task owns the display (avoids interleaved \r writes); it reads
                # the shared counter so the meter reflects every task's progress.
                t == 1 && _render_progress(progress, done[], nchunks)
                c += nt
            end
            results[t] = out
            collect_stats && (statparts[t] = stats)
        end
    end
    if progress !== :none                    # clean 100% line after the parallel region
        _render_progress(progress, nchunks, nchunks)
        println(stderr)
    end
    if collect_stats
        allstats = reduce(vcat, statparts; init=BlockMetricStats[])
        sort!(allstats; by = s -> (s.block, s.k))
        append!(metricstats, allstats)
    end

    cands = reduce(vcat, results; init=Candidate[])
    ntotal = length(cands)
    @info "Search complete; post-processing candidates" total_above_threshold=ntotal
    if remove
        n0 = length(cands)
        cands = remove_duplicates(cands; dr_tol=dr_tol)
        @info "Collapsed near-identical (duplicate) candidates" removed=(n0 - length(cands)) remaining=length(cands)
    else
        sort!(cands; by=c -> c.freq)
    end
    if harm_remove
        n1 = length(cands)
        cands = remove_harmonics(cands; numharm=numharm, tol=harm_tol)
        @info "Collapsed harmonically-related candidates" removed=(n1 - length(cands)) remaining=length(cands)
    end
    return cands
end

"""
    _render_progress(mode, done, total)

Overwrite a single-line chunk-completion meter on `stderr` (`\\r`, no newline).
`mode` is `:none` (does nothing), `:text` (a percentage) or `:bar` (a bar).  The
caller prints the closing newline once, after the parallel region.
"""
function _render_progress(mode::Symbol, done::Integer, total::Integer)
    mode === :none && return
    frac = total == 0 ? 1.0 : done / total
    pct = round(Int, 100 * frac)
    if mode === :bar
        width = 40
        filled = clamp(round(Int, width * frac), 0, width)
        print(stderr, "\r  Searching [", '#'^filled, ' '^(width - filled),
              "] ", lpad(pct, 3), "%  (", done, "/", total, " chunks)")
    else
        print(stderr, "\r  Searching: ", lpad(pct, 3), "%  (", done, "/", total, " chunks)")
    end
    flush(stderr)
    return
end

"""
    remove_duplicates(cands; dr_tol=1.0) -> Vector{Candidate}

Collapse clusters of near-identical candidates — the run of adjacent trial
fundamentals that a single signal lights up — down to their strongest member.
Candidates are grouped whenever consecutive Fourier frequencies `r` (in bins,
sorted) lie within `dr_tol` of one another, and the maximum-metric candidate of
each group is kept.  One Fourier bin is `1/T` Hz, so a `dr_tol` of order a bin
is still far finer than the spacing of astrophysically distinct sources, while
comfortably spanning the sub-bin-wide coherent-response cluster.  Returns the
kept candidates sorted by frequency.
"""
function remove_duplicates(cands::AbstractVector{Candidate}; dr_tol::Real=1.0)
    isempty(cands) && return Candidate[]
    order = sortperm(cands; by=c -> c.r)
    kept = Candidate[]
    best = cands[order[1]]
    prev_r = best.r
    @inbounds for idx in @view order[2:end]
        c = cands[idx]
        if c.r - prev_r <= dr_tol
            c.metric > best.metric && (best = c)
        else
            push!(kept, best)
            best = c
        end
        prev_r = c.r
    end
    push!(kept, best)
    sort!(kept; by=c -> c.freq)
    return kept
end

"""
    _harmonically_related(r1, r2; numharm, tol) -> Bool

Whether the Fourier frequencies `r1`, `r2` (bins) are harmonics of a common
fundamental: `hi/lo ≈ n/m` for integers `1 ≤ m, n ≤ numharm`.  With the best
common fundamental `f₀ = lo/m = hi/n`, `|m·hi - n·lo|` is `m ·` the residual of
`hi` from `n·f₀`, so the test `|m·hi - n·lo| ≤ tol·m` holds `hi` to `tol` bins on
the shared comb — a bin-scale tolerance that (unlike a fixed `|m·hi - n·lo|`
bound) does not tighten spuriously at high harmonic number.
"""
@inline function _harmonically_related(r1::Real, r2::Real; numharm::Integer, tol::Real)
    lo, hi = minmax(r1, r2)
    lo > 0 || return false
    @inbounds for m in 1:numharm
        n = round(Int, m * hi / lo)             # nearest harmonic ratio hi/lo ≈ n/m
        (1 <= n <= numharm) || continue
        abs(m * hi - n * lo) <= tol * m && return true
    end
    return false
end

"""
    remove_harmonics(cands; numharm=16, tol=1.0) -> Vector{Candidate}

Collapse harmonically-related candidates — the `f/2`, `3f/2`, `2f`, … family a
single real signal lights up (made more prominent by harmonic decimation, whose
subharmonic folds report genuinely different Fourier frequencies `r`) — keeping
the strongest member of each family.  Candidates are visited strongest-metric
first; each is kept unless its `r` is [`_harmonically_related`](@ref) (up to
`numharm`, within `tol` bins) to an already-kept stronger one.  Distinct from
[`remove_duplicates`](@ref), which collapses only *near-identical* `r`; run this
after it.  Returns the kept candidates sorted by frequency.
"""
function remove_harmonics(cands::AbstractVector{Candidate}; numharm::Integer=16, tol::Real=1.0)
    isempty(cands) && return Candidate[]
    order = sortperm(cands; by=c -> c.metric, rev=true)   # strongest first
    kept = Candidate[]
    @inbounds for idx in order
        c = cands[idx]
        if !any(k -> _harmonically_related(c.r, k.r; numharm=numharm, tol=tol), kept)
            push!(kept, c)
        end
    end
    sort!(kept; by=c -> c.freq)
    return kept
end
