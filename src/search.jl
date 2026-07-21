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
    xsignal::Float64 = 0.2  # peak fraction bounding the on-pulse signal region (:non/:sd2 only)
    metric::Symbol = :boxcar  # :boxcar (matched filter; default), :non (N_on^pexp), :sd2 (Σd²^pexp)
    pexp::Float64 = 0.5     # width-penalty exponent (1/2 = calibrated for :non; :non/:sd2 only)
    boxcar_fsp::Float64 = 1.5    # :boxcar geometric width-recurrence factor (riptide default)
    boxcar_maxfrac::Float64 = 0.3  # :boxcar widest boxcar as a fraction of nbins
    boxcar_medmargin::Float64 = 2.0  # :boxcar fast path: compute the exact median baseline only
                                     # when the 0-baseline metric is within this of `threshold`
                                     # (mean≡0 since DC=0; see `_profile_boxcar`)
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

# --- Branchless sorting-network median for short profiles ------------------
# For the small `nbins` the harmonic decimations produce (2·⌊nharms/k⌋ = 20..60
# at the default nharms=60, k=2..6), a fixed Batcher odd-even mergesort network
# beats the quickselect above: its compare-exchanges are data-independent (no
# mispredicting branch) and `min`/`max` lower to branch-free `cmov`, so the whole
# sort pipelines.  Measured on cold random columns: ~2.0× at n=20, ~1.75× at
# n=30, ~1.28× at n=60, crossing over to a *loss* by n=120 (the network's
# `O(n·log²n)` compare count overtakes quickselect's `O(n)`).  So it is used only
# for `nbins ≤ _MED_NET_MAX`; the base `k=1` pass (nbins=120) keeps quickselect.
# A full sort's two central order statistics are *identical* to quickselect's, so
# the median value is bit-for-bit unchanged and every oracle/equivalence pin holds.
const _MED_NET_MAX = 64

# Compare-exchange index pairs (1-based, `(a,b)` with `a<b`) of the Batcher
# odd-even mergesort network for length `n`.  Generated once per distinct profile
# length at plan-build time (never in the hot loop); empty ⇒ caller uses quickselect.
function _batcher_pairs(n::Int)
    pairs = Tuple{Int,Int}[]
    p = 1
    while p < n
        k = p
        while k >= 1
            for j in (k % p):(2k):(n - 1 - k), i in 0:(k - 1)
                if (i + j) ÷ (2p) == (i + j + k) ÷ (2p)
                    b = i + j + k
                    b < n && push!(pairs, (i + j + 1, b + 1))
                end
            end
            k ÷= 2
        end
        p *= 2
    end
    return pairs
end

# Median of `v[1:n]` via a precomputed compare-exchange network (branchless
# min/max), then the central order statistic(s).  Same value as `_median!`.
@inline function _median_net!(v::AbstractVector{Float64}, pairs::Vector{Tuple{Int,Int}}, n::Int)
    @inbounds for (a, b) in pairs
        x = v[a]; y = v[b]
        v[a] = ifelse(x < y, x, y)                 # min
        v[b] = ifelse(x < y, y, x)                 # max
    end
    half = n >>> 1
    isodd(n) && return @inbounds v[half + 1]
    return @inbounds 0.5 * (v[half] + v[half + 1])
end

# Per-profile baseline median: network for short profiles (`pairs` non-empty),
# quickselect otherwise.  `pairs` is chosen once per pass from `nbins`.
@inline _baseline_median!(v::AbstractVector{Float64}, n::Int, pairs::Vector{Tuple{Int,Int}}) =
    isempty(pairs) ? _median!(v, n) : _median_net!(v, pairs, n)

# Shared empty network for the reference/public paths (they use quickselect).
const _NO_MEDPAIRS = Tuple{Int,Int}[]

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

# ---------------------------------------------------------------------------
# Boxcar matched-filter metric (:boxcar)
#
# A cleaner alternative to the on-pulse-selection metric above, whose adaptive
# on-pulse set gives the pure-noise metric a non-analytic ~√nbins floor (so the
# low-k decimations dominate the candidate list; see `MetricStats`).  Here we
# instead correlate each profile with a *fixed* bank of boxcar (top-hat) filters
# and report the peak matched-filter S/N — exactly PRESTO single-pulse / the
# riptide FFA (Morello et al. 2020, MNRAS 497, 4654, §5.4).  Because the widths
# are chosen a priori (not from the data), a width-w boxcar over white noise is
# N(0, w·σ²); dividing by σ√w makes every trial unit-variance regardless of w or
# nbins, so the peak over trials follows analytic extreme-value statistics with a
# known, ~nbins-flat trials factor — no √nbins floor to normalise away.
# ---------------------------------------------------------------------------

"""
    boxcar_widths(nbins; fsp=1.5, maxfrac=0.3) -> Vector{Int}

Geometric bank of boxcar widths (in profile bins) for a `nbins`-bin profile:
`w₀=1`, `wₖ₊₁ = max(⌊fsp·wₖ⌋, wₖ+1)`, truncated at `⌊maxfrac·nbins⌋` (the widest
duty cycle worth testing).  The `fsp=1.5` recurrence is riptide's default and
reproduces the hand-picked `[1,2,3,4,6,9,13,19,…]` sequence.  Always contains at
least the width-1 (single-bin) filter.
"""
function boxcar_widths(nbins::Integer; fsp::Real=1.5, maxfrac::Real=0.3)
    wmax = max(1, floor(Int, maxfrac * nbins))
    ws = Int[]
    w = 1
    while w <= wmax
        push!(ws, w)
        w = max(floor(Int, fsp * w), w + 1)
    end
    return ws
end

const _BOXCAR_SIGMA_SAMPLES = 8192     # bins subsampled per block to fix the noise σ

"""
    _block_sigma(M, nbins, n, buf) -> Float64

Robust per-bin noise scale (`1.4826 × MAD`) for one block, pooled over a strided
subsample of the `nbins × n` profile matrix `M[:, 1:n]` into `buf` (length
`_BOXCAR_SIGMA_SAMPLES`).  A block carries thousands of noise bins, so this `σ̂`
has sub-percent variance — unlike a per-profile MAD (`~0.76/√nbins`, i.e. ~17% at
`nbins=20`), whose estimation noise multiplies straight into every boxcar S/N and
inflates the small-`nbins` tail.  Median-based and pooled, so the rare signal/RFI
bin does not bias it.  Returns `0.0` for a degenerate (flat) block.

The subsample indices depend only on `(nbins, n)`, and `M` enters only through the
ratio the caller forms (`excess/σ`), so the unnormalised `brfft` and normalised
`irfft` paths yield the identical scale-free S/N (the `align=false` pin holds).
"""
function _block_sigma(M::AbstractMatrix{Float64}, nbins::Int, n::Int, buf::Vector{Float64})
    N = nbins * n
    N == 0 && return 0.0
    cap = length(buf)
    ns = 0
    if N <= cap
        @inbounds for j in 1:n, i in 1:nbins
            ns += 1; buf[ns] = M[i, j]
        end
    else
        s = N ÷ cap                               # stride ≥ 1; not a multiple of nbins in general
        @inbounds for t in 1:s:N
            ns == cap && break
            j = (t - 1) ÷ nbins + 1
            i = (t - 1) % nbins + 1
            ns += 1; buf[ns] = M[i, j]
        end
    end
    med = _median!(buf, ns)                       # destroys buf order (values preserved)
    @inbounds for t in 1:ns
        buf[t] = abs(buf[t] - med)                # MAD over the same multiset (order irrelevant)
    end
    return 1.4826 * _median!(buf, ns)
end

# Prefix sum of the profile column minus a scalar baseline `b`, tiled by one extra
# `wmax` samples so a boxcar that wraps past bin `nbins` reads real (wrapped) data:
# boxcar sum of bins p..p+w-1 (1-based) = psum[p+w] - psum[p].
@inline function _boxcar_psum!(psum::Vector{Float64}, col::AbstractVector{Float64},
                               nbins::Int, wmax::Int, b::Float64)
    psum[1] = 0.0
    @inbounds for i in 1:(nbins + wmax)
        idx = i > nbins ? i - nbins : i
        psum[i + 1] = psum[i] + (col[idx] - b)
    end
end

# Peak matched-filter S/N over the boxcar bank.  Per width, the peak is
# `(max_p boxsum) * invsw` because `invsw > 0` is monotone — so the phase scan is a
# pure max-reduction over the strided prefix-sum difference `psum[p+w] - psum[p]`
# (two contiguous, `w`-shifted loads), which `@simd` vectorises; pulling `invsw`
# out of the inner loop returns the identical `Float64`.
@inline function _boxcar_scan(psum::Vector{Float64}, widths::Vector{Int},
                              nbins::Int, invsigma::Float64)
    best = -Inf
    @inbounds for w in widths
        invsw = invsigma / sqrt(float(w))
        m = psum[1 + w] - psum[1]                  # finite seed (no -Inf in the reduction)
        @simd for p in 2:nbins
            m = max(m, psum[p + w] - psum[p])
        end
        cand = m * invsw
        cand > best && (best = cand)
    end
    return best
end

"""
    _profile_boxcar(profs, j, medbuf, psum, widths, nbins, invsigma[, medpairs, medcut]) -> Float64

Peak boxcar matched-filter S/N of profile column `j` (see the section comment).
`medbuf` (length `nbins`) is scratch for the per-profile baseline median (computed
by [`_baseline_median!`](@ref): the sorting network when `medpairs` is non-empty,
quickselect otherwise); `psum` (length `≥ nbins + widths[end] + 1`) holds the
phase-tiled prefix sum.  `invsigma = 1/σ` is the block's robust per-bin noise scale
([`_block_sigma`](@ref)) — shared across the block so its (negligible) estimation
noise does not leak into the per-trial statistic, which is then exactly `N(0,1)`
per (phase, width) under white noise.

The baseline is the profile median; the reported S/N
`max_{w,p} (Σ_{i=p}^{p+w-1}(P_i − med)) · invsigma / √w` is a ratio of two
linear-in-amplitude quantities, hence invariant to the profile's overall scale —
so the unnormalised hot-loop `brfft` and the normalised reference `irfft` yield
the identical value, and neither `ngoodbins` nor the `scale` factor is needed.

**Fast path (`medcut > -∞`).** Because the profile spectrum's DC bin is held at
zero, every profile's *mean* is 0 by construction, so the boxcar scan against a
*zero* baseline needs no median.  For a positive pulse the median is ≤ 0, so that
zero-baseline metric `m₀` is a lower bound on the true metric, with the gap bounded
by `|med|·√wₘₐₓ/σ`.  We therefore scan against 0 first and, only if `m₀ ≥ medcut`
(caller passes `threshold − boxcar_medmargin`), pay for the exact median and
rescan.  Sub-`medcut` trials — the ~99% that are pure noise — return `m₀` and never
compute a median; any trial that could cross `threshold` gets the exact value, so
the candidate list is unchanged provided `boxcar_medmargin ≥ |med|·√wₘₐₓ/σ`.
`medcut = -∞` (the default, and the metricstats/normalize/reference paths) always
computes the exact median.
"""
@inline function _profile_boxcar(profs::AbstractMatrix{Float64}, j::Integer,
                                 medbuf::Vector{Float64}, psum::Vector{Float64},
                                 widths::Vector{Int}, nbins::Int, invsigma::Float64,
                                 medpairs::Vector{Tuple{Int,Int}}=_NO_MEDPAIRS,
                                 medcut::Float64=-Inf)
    invsigma > 0 || return 0.0                    # degenerate (flat block): no detection
    col = @view profs[:, j]
    wmax = widths[end]
    if medcut > -Inf                              # fast gate: cheap zero-baseline scan first
        _boxcar_psum!(psum, col, nbins, wmax, 0.0)
        m0 = _boxcar_scan(psum, widths, nbins, invsigma)
        m0 < medcut && return m0                  # can't reach threshold — skip the median
    end
    @inbounds for i in 1:nbins
        medbuf[i] = col[i]
    end
    med = _baseline_median!(medbuf, nbins, medpairs)   # network (short) or quickselect
    _boxcar_psum!(psum, col, nbins, wmax, med)
    return _boxcar_scan(psum, widths, nbins, invsigma)
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
                     xsignal::Real=0.2, metric::Symbol=:non, pexp::Real=0.5,
                     boxcar_fsp::Real=1.5, boxcar_maxfrac::Real=0.3)
    metric in (:non, :sd2, :boxcar) ||
        throw(ArgumentError("metric must be :non, :sd2 or :boxcar, got :$metric"))
    nbins, L = size(profs)
    medbuf = Vector{Float64}(undef, nbins)
    P = profs isa Matrix{Float64} ? profs : convert(Matrix{Float64}, profs)
    if metric === :boxcar
        widths = boxcar_widths(nbins; fsp=boxcar_fsp, maxfrac=boxcar_maxfrac)
        psum = Vector{Float64}(undef, nbins + widths[end] + 1)
        sigbuf = Vector{Float64}(undef, min(nbins * L, _BOXCAR_SIGMA_SAMPLES))
        sigma = _block_sigma(P, nbins, L, sigbuf)          # one robust σ for the whole set
        invsigma = sigma > 0 ? 1.0 / sigma : 0.0
        return [_profile_boxcar(P, j, medbuf, psum, widths, nbins, invsigma) for j in 1:L]
    end
    invrms = sqrt(2 * ngoodbins + 1)
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

    medbuf = Vector{Float64}(undef, nbins)
    metrics = Vector{Float64}(undef, L)
    if params.metric === :boxcar
        widths = boxcar_widths(nbins; fsp=params.boxcar_fsp, maxfrac=params.boxcar_maxfrac)
        psum = Vector{Float64}(undef, nbins + widths[end] + 1)
        sigbuf = Vector{Float64}(undef, min(nbins * L, _BOXCAR_SIGMA_SAMPLES))
        sigma = _block_sigma(profs, nbins, L, sigbuf)
        invsigma = sigma > 0 ? 1.0 / sigma : 0.0
        for j in 1:L
            metrics[j] = _profile_boxcar(profs, j, medbuf, psum, widths, nbins, invsigma)
        end
        return metrics
    end
    rmean = sum(rfund) / L
    invrms = sqrt(2 * chunk_ngoodbins(ft, nh, rmean) + 1)
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
    bcwidths::Vector{Int}          # :boxcar width bank for 2*Hk-bin profiles
    bcpsum::Vector{Float64}        # :boxcar prefix-sum scratch (2*Hk + wmax + 1)
    bcsig::Vector{Float64}         # :boxcar per-block σ subsample scratch
    medpairs::Vector{Tuple{Int,Int}}  # sorting-network median pairs (empty ⇒ quickselect)
end

function DecimBuf(k::Integer, nharms::Integer, Nprof::Integer, params::SearchParams)
    Hk = fld(nharms, k)
    dftprofs = zeros(ComplexF64, Hk + 1, Nprof)
    dprofs   = Matrix{Float64}(undef, 2Hk, Nprof)
    medbuf   = Vector{Float64}(undef, 2Hk)
    brfftplan = plan_brfft(dftprofs, 2Hk, 1; flags=FFTW.MEASURE)
    fill!(dftprofs, 0)   # MEASURE planning may have dirtied the buffer
    bcwidths = boxcar_widths(2Hk; fsp=params.boxcar_fsp, maxfrac=params.boxcar_maxfrac)
    bcpsum   = Vector{Float64}(undef, 2Hk + bcwidths[end] + 1)
    bcsig    = Vector{Float64}(undef, min(2Hk * Nprof, _BOXCAR_SIGMA_SAMPLES))
    medpairs = 2Hk <= _MED_NET_MAX ? _batcher_pairs(2Hk) : _NO_MEDPAIRS
    return DecimBuf(Int(k), Hk, dftprofs, dprofs, medbuf, brfftplan, bcwidths, bcpsum, bcsig, medpairs)
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
    bcwidths::Vector{Int}          # :boxcar width bank for the base 2*nharms-bin profiles
    bcpsum::Vector{Float64}        # :boxcar prefix-sum scratch (2*nharms + wmax + 1)
    bcsig::Vector{Float64}         # :boxcar per-block σ subsample scratch
    medpairs::Vector{Tuple{Int,Int}}  # sorting-network median pairs (empty ⇒ quickselect; nbins=120 default)
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
    bcwidths = boxcar_widths(2nh; fsp=params.boxcar_fsp, maxfrac=params.boxcar_maxfrac)
    bcpsum   = Vector{Float64}(undef, 2nh + bcwidths[end] + 1)
    bcsig    = Vector{Float64}(undef, min(2nh * Nprof, _BOXCAR_SIGMA_SAMPLES))
    medpairs = 2nh <= _MED_NET_MAX ? _batcher_pairs(2nh) : _NO_MEDPAIRS
    # A DecimBuf per k > 1 (k = 1 is the base ftprofs/profs above).  `map` (not a
    # `DecimBuf[...]` comprehension) keeps the element type the concrete
    # `DecimBuf{B}` even when empty, so `Workspace`'s `D<:DecimBuf` stays concrete.
    decims = map(k -> DecimBuf(k, nh, Nprof, params), filter(>(1), params.decimations))
    return Workspace(scratch, ftprofs, profs, medbuf, brfftplan, bcwidths, bcpsum, bcsig, medpairs, decims)
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
    MetricHistogram

A fixed-range, fixed-width histogram of every trial's detection metric for one
decimation factor `k` (`Hk` harmonics, `nbins = 2*Hk`) within one *searched
spin-frequency window* `[flo, fhi)` Hz (window index `win`; `win = 0` marks a
per-`k` histogram merged over all windows).  Accumulated in one streaming pass.
This is the bounded-memory substrate for *exact* per-`(k, frequency)` empirical
quantiles (and thus frequency-resolved false-alarm thresholds): `counts[i]`
covers `[lo + (i-1)·w, lo + i·w)` with `w = (hi-lo)/length(counts)`;
`under`/`over` catch metric values outside `[lo, hi)`.  `total`, `sum`, `sumsq`,
`vmin`, `vmax` are kept exactly (independent of the binning), so the mean, std,
min and max are exact regardless of range — only the quantiles are resolved to
the bin width.  One per task, summed across tasks after the parallel region.
"""
mutable struct MetricHistogram
    const k::Int
    const Hk::Int
    const nbins::Int
    const win::Int               # frequency-window index (0 = merged over all windows)
    const flo::Float64           # searched-frequency window bounds (Hz)
    const fhi::Float64
    const lo::Float64
    const hi::Float64
    const invw::Float64          # 1/binwidth
    const counts::Vector{Int}
    under::Int
    over::Int
    total::Int
    sum::Float64
    sumsq::Float64
    vmin::Float64
    vmax::Float64
end

function MetricHistogram(k::Integer, Hk::Integer, win::Integer, flo::Real, fhi::Real,
                         lo::Real, hi::Real, nb::Integer)
    hi > lo || throw(ArgumentError("histogram hi ($hi) must exceed lo ($lo)"))
    nb >= 1 || throw(ArgumentError("histogram bin count must be ≥ 1"))
    return MetricHistogram(Int(k), Int(Hk), 2 * Int(Hk), Int(win), Float64(flo), Float64(fhi),
                           Float64(lo), Float64(hi), nb / (hi - lo), zeros(Int, nb),
                           0, 0, 0, 0.0, 0.0, Inf, -Inf)
end

# Log-spaced window edges over [a, b] (nwin windows), robust to a≈b / a≤0.
function _logedges(a::Real, b::Real, nwin::Integer)
    a = max(Float64(a), floatmin(Float64))
    b = max(Float64(b), a * (1 + 1e-9))
    la, lb = log10(a), log10(b)
    return [10.0^(la + (lb - la) * i / nwin) for i in 0:nwin]
end

# Window index of searched frequency `f` in sorted `edges` (length nwin+1),
# clamped to [1, nwin].  Called once per (block, k), not per trial.
@inline _window_index(edges::Vector{Float64}, f::Real) =
    clamp(searchsortedlast(edges, f), 1, length(edges) - 1)

# Merge `hs` (same binning, same k) into one fresh histogram tagged (win, flo, fhi).
function _merge_hists(hs::AbstractVector{MetricHistogram}, win::Integer, flo::Real, fhi::Real)
    h1 = first(hs)
    g = MetricHistogram(h1.k, h1.Hk, win, flo, fhi, h1.lo, h1.hi, length(h1.counts))
    for h in hs
        _hist_merge!(g, h)
    end
    return g
end

@inline function _hist_push!(h::MetricHistogram, x::Float64)
    h.total += 1
    h.sum += x
    h.sumsq += x * x
    x < h.vmin && (h.vmin = x)
    x > h.vmax && (h.vmax = x)
    if x < h.lo
        h.under += 1
    elseif x >= h.hi
        h.over += 1
    else
        @inbounds h.counts[floor(Int, (x - h.lo) * h.invw) + 1] += 1
    end
    return
end

# Sum `b` into `a` in place (identical binning assumed; both from the same run).
function _hist_merge!(a::MetricHistogram, b::MetricHistogram)
    @inbounds for i in eachindex(a.counts)
        a.counts[i] += b.counts[i]
    end
    a.under += b.under; a.over += b.over; a.total += b.total
    a.sum += b.sum; a.sumsq += b.sumsq
    a.vmin = min(a.vmin, b.vmin); a.vmax = max(a.vmax, b.vmax)
    return a
end

"""
    hist_quantile(h, q) -> Float64

The `q`-th quantile (`0 ≤ q ≤ 1`) of the metric values in `h`, linearly
interpolated within the containing bin.  Returns `h.hi` (and the caller should
treat it as a lower bound) when the quantile falls in the overflow — i.e. the
range was too small; check `h.over / h.total` against `1-q`.
"""
function hist_quantile(h::MetricHistogram, q::Real)
    h.total == 0 && return NaN
    target = q * h.total
    cum = h.under
    cum >= target && return h.lo
    w = 1 / h.invw
    @inbounds for i in eachindex(h.counts)
        c = h.counts[i]
        if cum + c >= target
            edge = h.lo + (i - 1) * w
            return c == 0 ? edge : edge + (target - cum) / c * w
        end
        cum += c
    end
    return h.hi           # in the overflow tail
end

"""
    MetricStats

Opt-in diagnostic sink passed to [`search`](@ref) (the `--metricstats` CLI
option).  Holds three complementary views of the detection metric over *all*
trials (not just those above `threshold`):

  * `blocks` — one [`BlockMetricStats`](@ref) per processed block per
    decimation, i.e. finely frequency-resolved min/median/mean/std/max.
  * `hists` — one [`MetricHistogram`](@ref) per decimation `k`, merged over the
    whole band, giving exact global moments and empirical per-`k` quantiles
    (hence per-`k` false-alarm thresholds via [`hist_quantile`](@ref)).
  * `whists` — one [`MetricHistogram`](@ref) per `(k, frequency window)`: the
    band is split into `nwin` log-spaced *searched-spin-frequency* windows per
    `k`, so the quantiles (and false-alarm thresholds) track the frequency
    dependence of the noise floor — the red-noise excess at low `f` and the
    `ngoodbins` Nyquist rolloff at high `f` — which a single band-wide histogram
    averages over.  This is the substrate the dynamic per-`(k, f)` normalisation
    needs.

Construct empty (optionally overriding the histogram range/resolution and the
window count `nwin`) and pass to `search`; it is filled after the parallel
region.  Collecting it does not change the candidate results.
"""
mutable struct MetricStats
    hist_lo::Float64
    hist_hi::Float64
    hist_nb::Int
    nwin::Int
    blocks::Vector{BlockMetricStats}
    hists::Vector{MetricHistogram}
    whists::Vector{MetricHistogram}
end
MetricStats(; hist_lo::Real=0.0, hist_hi::Real=64.0, hist_nb::Integer=3200, nwin::Integer=16) =
    MetricStats(Float64(hist_lo), Float64(hist_hi), Int(hist_nb), max(1, Int(nwin)),
                BlockMetricStats[], MetricHistogram[], MetricHistogram[])

"""
    metricstats_summary(ms; faps=(0.1, 0.01, 1e-3, 1e-4)) -> Vector{<:NamedTuple}

One row per decimation `k` (sorted by `k`) summarising [`MetricStats`](@ref) over
the whole band (from `ms.hists`).  `ntrials`/`mean`/`std`/`min`/`max` are exact
(from the histogram accumulators); `median` and the `fap` metric values are
empirical quantiles read from the per-`k` histogram — `fap[i]` is the metric
threshold whose single-trial, single-decimation false-alarm probability is
`faps[i]` (i.e. the `1-faps[i]` quantile).  `overflow` is the fraction of trials
above the histogram range (the `fap` values are unreliable once it exceeds the
smallest requested `fap`).  `nblocks` counts the contributing blocks.  See
[`metricstats_windows`](@ref) for the frequency-resolved breakdown.
"""
# One summary row from a single histogram (moments exact, quantiles binned).
function _hist_row(h::MetricHistogram, faps)
    N = h.total
    mean = N > 0 ? h.sum / N : NaN
    std = N > 1 ? sqrt(max(0.0, (h.sumsq - N * mean^2) / (N - 1))) : 0.0
    return (k=h.k, Hk=h.Hk, nbins=h.nbins, win=h.win, flo=h.flo, fhi=h.fhi, ntrials=N,
            min=(N > 0 ? h.vmin : NaN), median=hist_quantile(h, 0.5), mean=mean, std=std,
            max=(N > 0 ? h.vmax : NaN),
            fap=Tuple(hist_quantile(h, 1 - p) for p in faps),
            overflow=(N > 0 ? h.over / N : 0.0))
end

function metricstats_summary(ms::MetricStats; faps=(0.1, 0.01, 1e-3, 1e-4))
    return [merge(_hist_row(h, faps),
                  (nblocks=count(s -> s.k == h.k && s.n > 0, ms.blocks),))
            for h in ms.hists]
end

"""
    metricstats_windows(ms; faps=(0.1, 0.01, 1e-3, 1e-4)) -> Vector{<:NamedTuple}

Frequency-resolved companion to [`metricstats_summary`](@ref): one row per
`(k, frequency window)` (from `ms.whists`, sorted by `k` then window), each with
the window's searched-frequency bounds `flo`/`fhi` (Hz), exact moments, and the
empirical `fap` metric thresholds *within that window*.  Empty windows (no
trials) are dropped.  This is where the red-noise (low-`f`) and Nyquist-rolloff
(high-`f`) drift of the false-alarm threshold shows up.
"""
metricstats_windows(ms::MetricStats; faps=(0.1, 0.01, 1e-3, 1e-4)) =
    [_hist_row(h, faps) for h in ms.whists if h.total > 0]

# ---------------------------------------------------------------------------
# In-situ metric normalisation (the `--normalize` adaptive-threshold search)
# ---------------------------------------------------------------------------

const _MIN_WIN_TRIALS = 200            # below this, a window uses the per-k global loc/scale

"""
    MetricNorm

A per-`(k, searched-frequency window)` normalisation of the detection metric,
built from a first ([`MetricStats`](@ref)-collecting) pass (see
[`build_metricnorm`](@ref)) and applied on a second: the raw metric `M` of a
trial at decimation `k` and searched spin frequency `f` (Hz) becomes

    z = (M − loc(k,f)) / scale(k,f)

with `loc` the window's noise median and `scale` its upper-side robust spread
(`q(0.8413) − median`, Gaussian-calibrated to one `σ` and taken from the noise
bulk so signals/RFI in the tail do not bias it).  This makes a single threshold
mean a *consistent* noise level across every decimation and across frequency —
collapsing the `√nbins` per-`k` floor and the red-noise / Nyquist frequency
drift that `--metricstats` exposes.  `z` is a comparable *significance* (used as
the reported metric and for cross-`k` candidate ranking), but note it is only a
true equivalent-`σ` where the noise is Gaussian; the right-skewed metric makes
`z` an over-estimate deep in the tail — an absolute calibration (pure-noise
simulation, with the `ngoodbins` Nyquist rolloff handled semi-analytically) is
the intended follow-up.  Per-window estimates fall back to a per-`k` global one
where a window has too few trials (`< $(_MIN_WIN_TRIALS)`) or a degenerate scale.
"""
struct MetricNorm
    edges::Dict{Int,Vector{Float64}}   # per-k searched-frequency window edges (Hz)
    loc::Dict{Int,Vector{Float64}}     # per-k, per-window location (noise median)
    scale::Dict{Int,Vector{Float64}}   # per-k, per-window scale (upper 1σ-equivalent)
end

# Robust (loc, scale) of a single histogram; scale ≤ 0 → NaN (caller falls back).
function _loc_scale(h::MetricHistogram)
    loc = hist_quantile(h, 0.5)
    scale = hist_quantile(h, 0.8413) - loc      # upper 1σ-equivalent, noise-bulk
    return loc, (scale > 0 ? scale : NaN)
end

"""
    build_metricnorm(ms) -> MetricNorm

Build a [`MetricNorm`](@ref) from a filled [`MetricStats`](@ref): per `(k,
window)` a robust noise location/scale (see [`MetricNorm`](@ref)), with a per-`k`
band-wide fallback for sparse or degenerate windows.
"""
function build_metricnorm(ms::MetricStats)
    edges = Dict{Int,Vector{Float64}}()
    locs = Dict{Int,Vector{Float64}}()
    scales = Dict{Int,Vector{Float64}}()
    for g in ms.hists                            # one per k (band-wide fallback)
        k = g.k
        gloc, gscale = _loc_scale(g)
        isnan(gscale) && (gscale = 1.0)          # fully degenerate k: identity-ish
        wins = sort([h for h in ms.whists if h.k == k]; by = h -> h.win)
        isempty(wins) && continue
        edges[k] = vcat([h.flo for h in wins], wins[end].fhi)
        L = length(wins)
        locv = Vector{Float64}(undef, L)
        sclv = Vector{Float64}(undef, L)
        for (i, h) in enumerate(wins)
            wloc, wscale = _loc_scale(h)
            if h.total < _MIN_WIN_TRIALS || isnan(wscale)
                locv[i], sclv[i] = gloc, gscale    # fall back to the k-global estimate
            else
                locv[i], sclv[i] = wloc, wscale
            end
        end
        locs[k], scales[k] = locv, sclv
    end
    return MetricNorm(edges, locs, scales)
end

# Normalise raw metric `M` for decimation `k` at searched frequency `f` (Hz).
@inline function _normalize(norm::MetricNorm, k::Integer, f::Real, M::Real)
    e = get(norm.edges, k, nothing)
    e === nothing && return M            # no model for this k (should not happen)
    w = _window_index(e, f)
    @inbounds return (M - norm.loc[k][w]) / norm.scale[k][w]
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
trial's (raw) metric is gathered into a [`BlockMetricStats`](@ref) for this
`(block, k)`, and (if a `hist` is also passed) streamed into the per-`k`
[`MetricHistogram`](@ref) — the opt-in `--metricstats` diagnostic.  When a
[`MetricNorm`](@ref) `norm` is passed, each trial's raw metric is normalised to a
significance `z` before the `threshold` test, and `z` (not the raw metric) is
recorded in the candidate — the `--normalize` adaptive-threshold path.
"""
function decim_pass!(out::Vector{Candidate}, ws::Workspace, db::DecimBuf, ft::FFTFile,
                     params::SearchParams, rstart::Real, lodr::Real, n::Integer;
                     threshold::Real=params.threshold, block::Integer=0,
                     stats::Union{Nothing,Vector{BlockMetricStats}}=nothing,
                     hist::Union{Nothing,MetricHistogram}=nothing,
                     norm::Union{Nothing,MetricNorm}=nothing, medcut::Real=-Inf)
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
    # Valid decimated trials are the prefix j = 1..nvalid (k·rf increases with j).
    nvalid = 0
    @inbounds while nvalid < n && k * (rstart + nvalid * lodr) < nyq
        nvalid += 1
    end
    # One robust per-bin σ for this (block, k), from the valid profiles only (past-
    # Nyquist columns are partly zero-padded and would deflate it).
    invsigma = 0.0
    if params.metric === :boxcar
        sig = _block_sigma(db.dprofs, nbins, nvalid, db.bcsig)
        invsigma = sig > 0 ? 1.0 / sig : 0.0
    end
    mbuf = stats === nothing ? nothing : Float64[]     # gather metrics if requested
    @inbounds for j in 1:n
        r_dec = k * (rstart + (j - 1) * lodr)
        r_dec < nyq || continue                       # fundamental past Nyquist
        mval = params.metric === :boxcar ?
            _profile_boxcar(db.dprofs, j, db.medbuf, db.bcpsum, db.bcwidths, nbins, invsigma, db.medpairs, Float64(medcut)) :
            _profile_snr(db.dprofs, j, db.medbuf, nbins, invrms,
                         1.0 / nbins, params.xsignal, params.metric, params.pexp)
        if mbuf !== nothing
            push!(mbuf, mval)
            hist === nothing || _hist_push!(hist, mval)
        end
        fdec = r_dec / ft.T
        score = norm === nothing ? mval : _normalize(norm, k, fdec, mval)
        if score > threshold
            push!(out, Candidate(fdec, score, r_dec, Hk))
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
    if params.metric === :boxcar
        sigma = _block_sigma(ws.profs, nbins, n, ws.bcsig)
        invsigma = sigma > 0 ? 1.0 / sigma : 0.0
        return [_profile_boxcar(ws.profs, j, ws.medbuf, ws.bcpsum, ws.bcwidths, nbins, invsigma, ws.medpairs) for j in 1:n]
    end
    rmean = rstart + (n - 1) * lodr / 2
    invrms = sqrt(2 * chunk_ngoodbins(ft, nh, rmean) + 1)
    return [_profile_snr(ws.profs, j, ws.medbuf, nbins, invrms, 1.0 / nbins, params.xsignal, params.metric, params.pexp) for j in 1:n]
end

# ---------------------------------------------------------------------------
# The parallel candidate-finding region, shared by both search passes.
#
# Runs the chunk-parallel loop once over `[r_lo, r_hi]`, returning the
# above-`threshold` candidates.  If `metricstats` is set it accumulates the
# per-block / per-(k,window) diagnostics (on the *raw* metric); if `norm` is set
# each trial's metric is normalised to a significance before the threshold test
# and that significance is what the candidate records.  Both `search` passes call
# this: pass 1 with `norm=nothing`+`metricstats` (measure), pass 2 with the built
# `norm` (detect).  Workspaces are reusable scratch, so both passes share them.
# ---------------------------------------------------------------------------
function _search_region!(ft::FFTFile, params::SearchParams, hplans::Vector{HarmonicPlan},
                         workspaces::Vector{<:Workspace}, nbins::Integer,
                         r_lo::Real, r_hi::Real, lodr::Real, total::Integer,
                         Nprof::Integer, nchunks::Integer, nt::Integer;
                         threshold::Real, norm::Union{Nothing,MetricNorm},
                         metricstats::Union{Nothing,MetricStats}, progress::Symbol)
    collect_stats = metricstats !== nothing
    # :boxcar fast path: skip the exact-median baseline for trials whose zero-baseline
    # metric is > `boxcar_medmargin` below `threshold` (see `_profile_boxcar`).  Forced
    # off (exact) when collecting stats or normalising, which need every raw metric.
    medcut = (params.metric === :boxcar && !collect_stats && norm === nothing) ?
        threshold - params.boxcar_medmargin : -Inf
    # Decimation factors present (base pass = k=1, plus each Workspace DecimBuf).
    statks = collect_stats ? sort!(unique(vcat(1, [db.k for db in workspaces[1].decims]))) : Int[]
    # Log-spaced searched-frequency window edges per k (searched freq of a k-fold
    # is k·f, so k's band is k× the base band).  One histogram per (k, window).
    nwin = collect_stats ? metricstats.nwin : 0
    wedges = collect_stats ?
        Dict(k => _logedges(k * r_lo / ft.T, k * r_hi / ft.T, nwin) for k in statks) :
        Dict{Int,Vector{Float64}}()
    results = Vector{Vector{Candidate}}(undef, nt)
    statparts = collect_stats ? Vector{Vector{BlockMetricStats}}(undef, nt) : nothing
    histparts = collect_stats ? Vector{Dict{Int,Vector{MetricHistogram}}}(undef, nt) : nothing
    done = Atomic{Int}(0)     # chunks completed across all tasks (for the progress meter)
    @sync for t in 1:nt
        @spawn begin
            ws = workspaces[t]
            out = Candidate[]
            stats = collect_stats ? BlockMetricStats[] : nothing
            # Per task: a length-`nwin` vector of histograms per k, one per window.
            hists = collect_stats ?
                Dict(k => [MetricHistogram(k, fld(params.nharms, k), w,
                                           wedges[k][w], wedges[k][w + 1],
                                           metricstats.hist_lo, metricstats.hist_hi,
                                           metricstats.hist_nb) for w in 1:nwin]
                     for k in statks) :
                nothing
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
                invsigma = 0.0
                if params.metric === :boxcar
                    sig = _block_sigma(ws.profs, nbins, n, ws.bcsig)   # one robust σ per block
                    invsigma = sig > 0 ? 1.0 / sig : 0.0
                end
                # Whole (narrow) block → one window, keyed by its centre freq.
                basehist = collect_stats ?
                    hists[1][_window_index(wedges[1], rmean / ft.T)] : nothing
                for j in 1:n
                    metric = params.metric === :boxcar ?
                        _profile_boxcar(ws.profs, j, ws.medbuf, ws.bcpsum, ws.bcwidths, nbins, invsigma, ws.medpairs, medcut) :
                        _profile_snr(ws.profs, j, ws.medbuf, nbins, invrms, 1.0 / nbins, params.xsignal, params.metric, params.pexp)
                    if collect_stats
                        mbuf[j] = metric
                        _hist_push!(basehist, metric)
                    end
                    rf = rstart + (j - 1) * lodr
                    score = norm === nothing ? metric : _normalize(norm, 1, rf / ft.T, metric)
                    if score > threshold
                        push!(out, Candidate(rf / ft.T, score, rf, params.nharms))
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
                    whist = collect_stats ?
                        hists[db.k][_window_index(wedges[db.k], db.k * rmean / ft.T)] : nothing
                    decim_pass!(out, ws, db, ft, params, rstart, lodr, n;
                                threshold=threshold, block=c, stats=stats, hist=whist, norm=norm,
                                medcut=medcut)
                end
                atomic_add!(done, 1)
                # One task owns the display (avoids interleaved \r writes); it reads
                # the shared counter so the meter reflects every task's progress.
                t == 1 && _render_progress(progress, done[], nchunks)
                c += nt
            end
            results[t] = out
            if collect_stats
                statparts[t] = stats
                histparts[t] = hists
            end
        end
    end
    if progress !== :none                    # clean 100% line after the parallel region
        _render_progress(progress, nchunks, nchunks)
        println(stderr)
    end
    if collect_stats
        allstats = reduce(vcat, statparts; init=BlockMetricStats[])
        sort!(allstats; by = s -> (s.block, s.k))
        append!(metricstats.blocks, allstats)
        # For each (k, window): sum that window's histogram across tasks → whists.
        # Then merge a k's windows into one band-wide per-k histogram → hists.
        for k in statks
            kwins = MetricHistogram[]
            for w in 1:nwin
                merged = histparts[1][k][w]
                for t in 2:nt
                    _hist_merge!(merged, histparts[t][k][w])
                end
                push!(kwins, merged)
            end
            append!(metricstats.whists, kwins)
            push!(metricstats.hists,
                  _merge_hists(kwins, 0, wedges[k][1], wedges[k][end]))
        end
        sort!(metricstats.hists;  by = h -> h.k)
        sort!(metricstats.whists; by = h -> (h.k, h.win))
    end
    return reduce(vcat, results; init=Candidate[])
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

If a [`MetricStats`](@ref) is supplied as `metricstats`, the metric of *every*
trial (not just those above `threshold`) is accumulated into it — per-block,
per-decimation [`BlockMetricStats`](@ref) plus [`MetricHistogram`](@ref)s both
per-`k` and per-`(k, log-spaced searched-frequency window)` for empirical
(frequency-resolved) quantiles — the opt-in `--metricstats` diagnostic.  The
candidate results are identical whether or not `metricstats` is collected.

If `normalize` is set, the search runs in **two passes**: pass 1 measures the
per-`(k, frequency window)` noise statistics (as `metricstats` does — the same
`metricstats` sink is filled if given), pass 2 builds a [`MetricNorm`](@ref) from
them and re-runs, thresholding on the normalised significance `z` rather than the
raw metric (and recording `z` as each candidate's metric).  This makes a single
`threshold` mean a consistent noise level across every decimation and frequency —
so `threshold` is then in noise-`σ`-like units, not raw-metric units — and makes
candidate metrics comparable across decimations (improving the cross-`k`
[`remove_harmonics`](@ref) ranking).  It roughly doubles the runtime (two full
passes) and assumes the input is normalised (see [`MetricNorm`](@ref)).
"""
function search(ft::FFTFile, params::SearchParams=SearchParams();
                lofreq::Real=0.1, hifreq::Real=100.0, lobin::Integer=100,
                blocksize::Integer=2048, threshold::Real=params.threshold,
                remove::Bool=true, dr_tol::Real=1.0,
                harm_remove::Bool=true, numharm::Integer=16, harm_tol::Real=1.0,
                progress::Symbol=:none,
                metricstats::Union{Nothing,MetricStats}=nothing,
                normalize::Bool=false)
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

    norm = nothing
    if normalize
        # Pass 1/2: measure the per-(k, frequency window) noise, build the model.
        # Reuse the user's metricstats sink if given (same measurement); else an
        # internal one.  threshold=Inf skips candidate bookkeeping in this pass.
        normstats = metricstats === nothing ? MetricStats() : metricstats
        @info "Normalising: measuring per-(k,frequency) noise (pass 1/2)"
        _search_region!(ft, params, hplans, workspaces, nbins, r_lo, r_hi, lodr,
                        total, Nprof, nchunks, nt;
                        threshold=Inf, norm=nothing, metricstats=normstats, progress=progress)
        norm = build_metricnorm(normstats)
        @info "Built in-situ normalisation model; searching (pass 2/2)" windows=normstats.nwin
    end

    # Detection pass.  When normalising, stats were collected in pass 1, so pass 2
    # does not re-collect (metricstats=nothing here).
    cands = _search_region!(ft, params, hplans, workspaces, nbins, r_lo, r_hi, lodr,
                            total, Nprof, nchunks, nt;
                            threshold=threshold, norm=norm,
                            metricstats=(normalize ? nothing : metricstats), progress=progress)

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
