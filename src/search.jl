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
#         profiles at once, then a peak/|trough| metric per profile.
#
# All FFTW plans and interpolation kernels are built once (single-threaded,
# since FFTW *planning* is not thread-safe) and only *executed* inside the
# parallel region (executing a prebuilt plan on distinct buffers is safe).

using FFTW
using Base.Threads: @spawn, nthreads
using LinearAlgebra: mul!

"""
    SearchParams

Tunable search parameters (defaults match the Python CLI).

`numbetween` is the *floor* on the interpolation oversampling; with `align`
set, low harmonics use a finer `numbetween` matched to their finer `deltar_h`
(see [`harmonic_numbetween`](@ref)).
"""
Base.@kwdef struct SearchParams
    nharms::Int = 32        # number of harmonics to coherently sum (power of two)
    m::Int = 32             # Fourier bins in the interpolation kernel (even)
    numbetween::Int = 16    # *minimum* interpolated points between Fourier bins
    hidr::Float64 = 0.5     # Fourier-bin step at the highest harmonic
    threshold::Float64 = 8.0
    align::Bool = true      # per-harmonic numbetween matched to deltar_h
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
    block_metrics(ft, rfund, params) -> Vector{Float64}

Compute the coherent-fold peak/|trough| metric for every trial fundamental
Fourier frequency in `rfund`.  Self-contained reference implementation (own
buffers, no shared mutable state); this is the exact computation the Python
oracle reproduces in cross-validation.
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

Search a single block of trial fundamental Fourier frequencies `rfund` using
the reference [`block_metrics`](@ref), returning trials above `threshold`.
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
    Workspace

Everything one task needs to process a chunk with zero allocation in the hot
loop: an `FFTScratch` per distinct harmonic `fftlen`, the stacked-harmonic
amplitude array `ftprofs`, the real profile array `profs`, and the prebuilt
batched complex→real inverse plan.  One `Workspace` per task; never shared.
"""
struct Workspace{B}
    scratch::Dict{Int,FFTScratch}
    ftprofs::Matrix{ComplexF64}   # (nharms+1, Nprof)
    profs::Matrix{Float64}        # (2*nharms, Nprof)
    brfftplan::B                   # plan_brfft(ftprofs, 2*nharms, 1)
end

function Workspace(params::SearchParams, hplans::Vector{HarmonicPlan}, Nprof::Integer)
    nh = params.nharms
    scratch = Dict{Int,FFTScratch}()
    for fl in unique(hp.fftlen for hp in hplans)
        scratch[fl] = FFTScratch(fl)
    end
    ftprofs = zeros(ComplexF64, nh + 1, Nprof)
    profs   = Matrix{Float64}(undef, 2nh, Nprof)
    brfftplan = plan_brfft(ftprofs, 2nh, 1; flags=FFTW.MEASURE)
    fill!(ftprofs, 0)   # MEASURE planning may have dirtied the buffer
    return Workspace(scratch, ftprofs, profs, brfftplan)
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
    # One batched complex→real transform for all Nprof profiles at once.  The
    # 1/Nbins normalisation of a true irfft cancels in the peak/|trough| ratio,
    # so the unnormalised brfft is used directly.
    mul!(ws.profs, ws.brfftplan, ws.ftprofs)
    return
end

# peak/|trough| of column `j` of the (Nbins × *) real profile matrix.
@inline function _profile_metric(profs::Matrix{Float64}, j::Integer)
    mx = -Inf
    mn = Inf
    @inbounds for k in 1:size(profs, 1)
        v = profs[k, j]
        mx = ifelse(v > mx, v, mx)
        mn = ifelse(v < mn, v, mn)
    end
    return mx / abs(mn)
end

"""
    chunk_metrics(ft, params, rstart, n; lodr) -> Vector{Float64}

Single-threaded convenience that runs the optimised path over one chunk of `n`
trial fundamentals starting at `rstart` and returns the metric for each.  With
`params.align = false` this reproduces [`block_metrics`](@ref) to machine
precision; it is the bridge the test suite uses to pin the optimised path to the
oracle-validated reference.
"""
function chunk_metrics(ft::FFTFile, params::SearchParams, rstart::Real, n::Integer;
                       lodr::Real = params.hidr / params.nharms)
    hplans = build_harmonic_plans(params, n)
    ws = Workspace(params, hplans, n)
    fill_chunk_profiles!(ws, hplans, ft, params, rstart, lodr, n)
    return [_profile_metric(ws.profs, j) for j in 1:n]
end

"""
    search(ft, params; lofreq, hifreq, lobin, blocksize, threshold) -> Vector{Candidate}

Run the full coherent harmonic-summing search over `[lofreq, hifreq]` Hz,
parallelised across independent fundamental-frequency chunks of `blocksize`
trials each.  Each task owns a private [`Workspace`](@ref); all FFTW plans and
interpolation kernels are built once before the parallel region.

The `lofreq`/`lobin` precedence matches the Python CLI: `lofreq` is used unless
`lobin` is set to something other than its default of 100.
"""
function search(ft::FFTFile, params::SearchParams=SearchParams();
                lofreq::Real=0.1, hifreq::Real=100.0, lobin::Integer=100,
                blocksize::Integer=2048, threshold::Real=params.threshold)
    FFTW.set_num_threads(1)   # parallelise at the Julia-task level, not inside FFTW
    lodr = params.hidr / params.nharms
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

    results = Vector{Vector{Candidate}}(undef, nt)
    @sync for t in 1:nt
        @spawn begin
            ws = workspaces[t]
            out = Candidate[]
            c = t
            while c <= nchunks
                i0 = (c - 1) * Nprof
                n = min(Nprof, total - i0)
                rstart = r_lo + i0 * lodr
                fill_chunk_profiles!(ws, hplans, ft, params, rstart, lodr, n)
                @inbounds for j in 1:n
                    metric = _profile_metric(ws.profs, j)
                    if metric > threshold
                        rf = rstart + (j - 1) * lodr
                        push!(out, Candidate(rf / ft.T, metric, rf))
                    end
                end
                c += nt
            end
            results[t] = out
        end
    end

    cands = reduce(vcat, results; init=Candidate[])
    sort!(cands; by=c -> c.freq)
    return cands
end
