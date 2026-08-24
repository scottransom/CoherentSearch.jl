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
#         amplitude array by evaluating the Eqn.-30 kernel directly at the trial
#         frequencies, from a per-harmonic table of tabulated weights;
#       - loop #3 a single *batched* complex→real inverse FFT of all `Nprof`
#         profiles at once, then a width-sensitive S/N metric per profile.
#
# All FFTW plans and interpolation kernels are built once (single-threaded,
# since FFTW *planning* is not thread-safe) and only *executed* inside the
# parallel region (executing a prebuilt plan on distinct buffers is safe).

using FFTW
using Base.Threads: @spawn, nthreads, Atomic, atomic_add!
using LinearAlgebra: mul!
using Printf: @printf

# ---------------------------------------------------------------------------
# In-situ phase timers.
#
# Three separate optimisation calls in this project have been decided by an
# *isolated* micro-benchmark and turned out backwards in the real search (smooth
# `fftlen`, `_block_sigma`'s `idiv`, the `Float32` batched `brfft`).  The fix is
# to time the phases where they actually run, so this is a permanent, always-on
# facility rather than a temporary patch.
#
# Granularity is one `time_ns()` pair per phase per chunk (per `(chunk, k)` for
# the decimation phases), i.e. ~2 µs of clock reads against ~7 ms of work — 0.03%,
# below the run-to-run scatter and independent of `nharms`/`Nprof`.  Accumulators
# are per-thread, `_PHASE_STRIDE` apart so no two threads share a cache line.
# ---------------------------------------------------------------------------

const PHASE_NAMES = ("interp", "brfft", "block-sigma", "gate+metric", "cand-loop",
                     "zero-ftprofs", "decim-brfft", "decim-metric", "decim-cand-loop",
                     # Slots 10.. break `decim-brfft` out by decimation factor, so a
                     # cost that belongs to one transform size is not read as a
                     # property of "the decimated transform" in general.
                     "decim-brfft-k2", "decim-brfft-k3", "decim-brfft-k4",
                     "decim-brfft-k5", "decim-brfft-k6", "decim-brfft-k7+")
const _NPHASE = length(PHASE_NAMES)
# Slot for decimation factor `k`: 10 for k=2 … 15 for k≥7.
@inline _decim_brfft_slot(k::Integer) = 8 + min(Int(k), 7)
const _PHASE_STRIDE = 24          # ≥ _NPHASE, and 24*8 B spans whole cache lines
const _PHASE_MAXTHREADS = 512
const _PHASE_NS = zeros(Int64, _PHASE_STRIDE, _PHASE_MAXTHREADS)

# Time `expr` into phase `i` of the calling thread's slot.  `Threads.threadid()`
# is used only as an accumulator index — never to key correctness-bearing state —
# so a task migrating between threads merely splits its time across two slots.
macro phase(i, expr)
    quote
        local _t0 = time_ns()
        local _v = $(esc(expr))
        local _tid = Threads.threadid()
        @inbounds _PHASE_NS[$(esc(i)), _tid] += (time_ns() - _t0) % Int64
        _v
    end
end

"""
    phase_reset!()

Zero the [`phase_times`](@ref) accumulators.  Call before a timed search.
"""
phase_reset!() = (fill!(_PHASE_NS, 0); nothing)

"""
    phase_times() -> Vector{Pair{String,Float64}}

Accumulated seconds per hot-loop phase since the last [`phase_reset!`](@ref),
summed over threads and sorted by cost.  These are *CPU* seconds (each thread's
own clock), so under `-t N` they sum to roughly `N ×` the wall time of the
parallel region.
"""
function phase_times()
    tot = [sum(@view _PHASE_NS[i, :]) * 1e-9 for i in 1:_NPHASE]
    return sort([PHASE_NAMES[i] => tot[i] for i in 1:_NPHASE]; by = p -> -p.second)
end

# ---------------------------------------------------------------------------
# Precision of the *profile stage* — the stacked harmonic amplitudes
# (`ftprofs`), the batched inverse transform, and the folded profiles the
# detection metric reads.
#
# This is a *runtime* choice (`SearchParams.precision`), carried as the type
# parameter `P` of `Workspace{…,P}` / `DecimBuf{…,P}` so that both widths are
# compiled into one build.  That is deliberate: the two are close enough in
# wall-clock that A/B-ing them across separate builds is dominated by Julia's
# precompile cache (see the `ab-benchmark-worktrees` note), whereas in-process
# they alternate under identical conditions.
#
# Everything downstream of the metric — the reported metric, candidate
# frequencies, `MetricNorm`, the histograms — stays `Float64`.  Only the bulk
# arrays that FFTW and the metric stream through are narrowed.
# ---------------------------------------------------------------------------

"""
    SearchParams

Tunable search parameters (defaults match the Python CLI).

`m` is the number of Fourier bins summed by the interpolation kernel (must be
even).  The cost is linear in `m`, so it is worth keeping small.  The fraction of
signal power recovered is `S_m(dr) = Σ sinc²(dr − k)`, so the loss is
`1 − S_m ≈ 4·sin²(π·dr)/(π²·m)` — 0.2/m averaged over `dr`, 0.4/m worst case:
1.27% at the default `m = 16`, 0.63% at `m = 32`.  That sits well under the ~6.5%
mean loss the `hidr = 0.5` trial grid already costs at the highest harmonic,
which is why 16 rather than 32 is the default.  Note the truncation weight `S_m`
is *real and positive*, so small `m` costs amplitude but introduces **no** phase
error and cannot degrade the coherence of the harmonic sum.  Full analysis:
`../coherent_search/examples/interp_accuracy_vs_m.md`.

`numbetween` applies only to [`reference_profiles`](@ref) — the FFT-correlation
path that mirrors the Python original and is what the cross-validation pins.  The
production search evaluates the exact kernel directly and has no fine grid, so
nothing else reads it.
"""
Base.@kwdef struct SearchParams
    nharms::Int = 32        # number of harmonics to coherently sum
    m::Int = 16             # Fourier bins in the interpolation kernel (even)
    numbetween::Int = 16    # fine-grid oversampling — `reference_profiles` only
    hidr::Float64 = 0.5     # Fourier-bin step at the highest harmonic
    threshold::Float64 = 8.0
    boxcar_fsp::Float64 = 1.5    # geometric width-recurrence factor (riptide default)
    boxcar_maxfrac::Float64 = 0.3  # widest boxcar as a fraction of nbins
    boxcar_gatemargin::Float64 = 0.01  # fast path: re-score in Float64 only when the
                                       # Float32 batched gate lands within this of
                                       # `threshold` (see `boxcar_metrics!`)
    decimations::Vector{Int} = [1]  # harmonic-decimation factors k (see decimation_design.md)
    precision::Symbol = :f64   # profile-stage element type — :f64 or :f32 (see `proftype`)
end

"""
    proftype(params) -> Type{<:AbstractFloat}

Element type of the profile stage for `params.precision` (`:f64` or `:f32`); see
the precision note above [`SearchParams`](@ref).
"""
@inline function proftype(params::SearchParams)
    params.precision === :f64 && return Float64
    params.precision === :f32 && return Float32
    throw(ArgumentError("precision must be :f64 or :f32, got $(params.precision)"))
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
    # Duty cycle of the best-fitting boxcar, i.e. (best width)/(profile bins), as
    # riptide's `rseek` defines `ducy`.  NaN until measured: the hot loop's
    # `_boxcar_scan` deliberately discards which width won (it runs ~1e8 times and
    # only the reported handful of candidates need it), so this is filled in
    # afterwards by [`measure_ducy`](@ref).
    ducy::Float64
end
Candidate(freq, metric, r, nharm) = Candidate(freq, metric, r, nharm, NaN)

# ---------------------------------------------------------------------------
# Detection metric (port of `snr_metric` from coherent_search.py)
# ---------------------------------------------------------------------------

"""
    chunk_ngoodbins(ft, nharms, rmean) -> Float64

The number of harmonics that carry real Fourier data for a chunk of trial
fundamentals whose mean Fourier frequency is `rmean`: `min(Nyquist/rmean,
nharms)`.  Recorded per block in [`BlockMetricStats`](@ref) as the diagnostic
`ngoodbins`; the metric measures its own noise level and does not use
it.  Matches `min(ft.N/2/rstosearch.mean(), args.nharms)` in the Python code.
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

# Range length at or above which the *branchless* Lomuto partition wins.  It
# trades two extra stores per element for a mispredicting branch, so it pays only
# once the range is long enough for misprediction to dominate: measured on random
# doubles, **3.62x at n=8192** (`_block_sigma`'s 8192-sample MAD) but **0.94x at
# n=120** (the short per-block σ̂ subsamples, and `boxcar_best_width`'s median,
# per trial that clears the gate).
# Both partitions select the same order statistic, so the gate cannot change a
# result — it only picks the faster route.
const _SELECT_BRANCHLESS_MIN = 256

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
        # Two partitions, chosen by range length (see `_SELECT_BRANCHLESS_MIN`).
        local p::Int
        if hi - lo >= _SELECT_BRANCHLESS_MIN
            # Branchless Lomuto.  The textbook form below has a data-dependent
            # branch that mispredicts ~50% of the time on the noise-like data this
            # runs on, at ~15-20 cycles a miss — which dominated the partition.
            # Swapping *unconditionally* and advancing `i` only when the element
            # belonged left is branch-free (the compare becomes `setcc`+`add`): it
            # costs two extra stores per element and saves the misprediction.  The
            # invariant is unchanged — `v[lo:i-1] <= pivot`, `v[i:jj] > pivot` —
            # because when `x > pivot` the swap exchanges two elements that are
            # *both* already `> pivot`, merely permuting that run.
            i = lo
            for jj in lo:hi-1
                x = v[jj]
                v[jj] = v[i]
                v[i] = x
                i += (x <= pivot)
            end
            _swap!(v, i, hi)
            p = i
        else
            i = lo - 1
            for jj in lo:hi-1
                if v[jj] <= pivot
                    i += 1; _swap!(v, i, jj)
                end
            end
            _swap!(v, i+1, hi)
            p = i + 1
        end
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
@inline function _median!(v::AbstractVector{T}, n::Int) where {T<:AbstractFloat}
    half = n >>> 1
    _select!(v, 1, n, half + 1)               # upper-median order stat at v[half+1]
    isodd(n) && return @inbounds v[half + 1]
    upper = @inbounds v[half + 1]
    lower = @inbounds v[1]                     # lower median = max of the half below
    @inbounds for i in 2:half
        v[i] > lower && (lower = v[i])
    end
    return T(0.5) * (lower + upper)
end


# ---------------------------------------------------------------------------
# Boxcar matched-filter metric
#
# A cleaner alternative to the on-pulse-selection metric above, whose adaptive
# on-pulse set gives the pure-noise metric a non-analytic ~√nbins floor (so the
# low-k decimations dominate the candidate list; see `MetricStats`).  Here we
# instead correlate each profile with a *fixed* bank of boxcar (top-hat) filters
# and report the peak matched-filter S/N — exactly PRESTO single-pulse / the
# riptide FFA (Morello et al. 2020, MNRAS 497, 4654, §5.4), and, since
# 2026-08-24, *numerically* the same statistic as riptide's `snr1`
# (`cpp/snr.hpp`), verified against it at 1.4e-7 (riptide's own `Float32`).
#
# The template is the width-`w` boxcar made **zero-mean and unit-L2**: height
# `h = √((n−w)/(n·w))` on the `w` on-pulse bins, baseline `−b` with
# `b = w/(n−w)·h` on the rest.  Then `Σt = 0`, `Σt² = 1`, and the correlation
# `⟨t, P⟩/σ` has variance exactly 1 per (phase, width) under white noise — so the
# peak over trials follows analytic extreme-value statistics with a known,
# ~nbins-flat trials factor, and a fixed threshold means the same false-alarm
# rate at every width and every fold depth.
#
# Since `b = δ·(h+b)` with `δ = w/n`, that correlation is
#
#     ⟨t, P⟩ = (h+b)·(S_w − δ·S_tot),      h+b = 1/√(w(1−δ))
#
# i.e. **subtract `δ` times the profile total from each boxcar sum and divide by
# `σ√(w(1−δ))`**.  Written that way it is manifestly invariant to any constant
# baseline already removed from `P`, which is what lets the hot loop prefix-sum
# against 0 and the reference path against the mean and get the same answer.
#
# **What this replaced, and why (2026-08-24).**  The metric used to subtract each
# profile's own *median* and divide by `σ√w`.  That is riptide's statistic times
# a factor that is not constant: `√(1−δ)` under pure noise (where the median is
# ~0), rising to `1/√(1−δ)` for a bright pulse (where the median tracks the
# off-pulse level) — an 8.6% drift with source brightness at 5% duty, measured.
# The median recovers nothing the boxcar had not already seen: DC is held at
# zero, so `S_off ≡ −S_w` identically and subtracting the true off-pulse level is
# *exactly* a `1/(1−δ)` rescale of `S_w` (verified to 9 decimals).  At matched
# false-alarm rate the zero-mean template detects strictly better at every duty
# cycle — equal below ~5%, 0.27 against 0.14 detection fraction at 20% duty.
# ---------------------------------------------------------------------------

"""
    boxcar_widths(nbins; fsp=1.5, maxfrac=0.3) -> Vector{Int}

Geometric bank of boxcar widths (in profile bins) for a `nbins`-bin profile:
`w₀=1`, `wₖ₊₁ = max(⌊fsp·wₖ⌋, wₖ+1)`, truncated at `⌊maxfrac·nbins⌋` (the widest
duty cycle worth testing).  The `fsp=1.5` recurrence is riptide's default and
reproduces the hand-picked `[1,2,3,4,6,9,13,19,…]` sequence.  Always contains at
least the width-1 (single-bin) filter.

`w` is additionally capped at `nbins−1`, as riptide's `check_trial_widths` also
requires: the zero-mean unit-L2 template of a full-width boxcar is the zero
vector, so `w = nbins` has no matched filter to speak of (and would divide by
`√(w(1−w/nbins)) = 0`).  At the default `maxfrac = 0.3` the cap only ever bites
for `nbins ≤ 3`.
"""
function boxcar_widths(nbins::Integer; fsp::Real=1.5, maxfrac::Real=0.3)
    nbins >= 2 || throw(ArgumentError("boxcar_widths needs nbins ≥ 2, got $nbins"))
    wmax = min(max(1, floor(Int, maxfrac * nbins)), Int(nbins) - 1)
    ws = Int[]
    w = 1
    while w <= wmax
        push!(ws, w)
        w = max(floor(Int, fsp * w), w + 1)
    end
    return ws
end

# Widest boxcar any fold but the shallowest needs.  See `ladder_boxcar_widths`.
const _LADDER_WMAX = 6

"""
    ladder_boxcar_widths(nbins, k, params) -> Vector{Int}

The boxcar bank for the decimation-`k` fold, pruned against the *rest of the
ladder*.  Returns [`boxcar_widths`](@ref) unchanged when only one decimation is
searched — the pruning below is sound only because the other folds are there to
pick up what it drops.

**Why most of the full bank is redundant across the ladder.**  A `W`-bin boxcar
on an `M`-bin profile weights harmonic `h` of the signal by the Dirichlet kernel
`|sin(πhW/M) / sin(πh/M)|`, which depends on `W` and `M` only through the duty
cycle `δ = W/M`.  Since `M = 2·nharms/k`, the *filter shape* is a function of `δ`
alone and the fold only sets where the harmonic sum is truncated (`H = nharms/k`).
So `(k, 2W)` and `(2k, W)` are the same filter, differing only in that the deeper
fold also sums harmonics `H₂ₖ+1 … Hₖ`.  That makes the two corners of the
`(k, W)` grid dead weight:

  * **`W = 1` on any fold but the deepest.**  Its duty is matched by `W = 2` one
    rung deeper, which has the identical kernel over twice the harmonics — the
    same filter at strictly better profile resolution.
  * **Wide `W` on any fold but the shallowest.**  Same duty, but the deep fold
    drags in `nharms/k − 2·nharms/(k·W)` harmonics past the kernel's first null,
    which for a smooth pulse carry noise and no signal.

What survives is the diagonal: deep fold ↔ narrow boxcar, shallow fold ↔ wide
boxcar.  Keeping `2 ≤ W ≤ _LADDER_WMAX`, plus `W = 1` on the deepest fold and
the full tail on the shallowest, is 25 of 38 `(k, W)` pairs at the defaults —
**0.65× the boxcar scan work at 0.00% worst-case S/N loss**.

**The loss is zero for smooth pulses, which is the only case that matters.**
Verified numerically (`compare/`-adjacent Monte Carlo, 2026-08-16) over Gaussian,
scattered-Gaussian (τ = 0.5–2 × duty), two-component and interpulse profiles at
60 duty cycles from 0.6% to 33%: the pruned bank's best matched-filter
efficiency equals the full bank's everywhere, and the retained set is a strict
superset of the greedy zero-loss optimum.  A *boxcar-shaped* signal would lose
up to 8%, but a boxcar is the filter we use because its noise statistics are
tractable, not a profile any pulsar has.

The same model reproduces the measured value of the ladder itself: it puts the
`k = 1`-only search 8.40% below the full ladder, against the 9.1% measured on
the 7.1185 Hz pulsar (S/N 11.89 at `k=1` vs 12.97 at `k=6`).
"""
function ladder_boxcar_widths(nbins::Integer, k::Integer, params::SearchParams)
    ws = boxcar_widths(nbins; fsp=params.boxcar_fsp, maxfrac=params.boxcar_maxfrac)
    ks = sort(params.decimations)
    length(ks) > 1 || return ws          # no ladder: nothing else covers the drops
    # The cap is only safe if the next rung picks up where this one stops: fold
    # `k` reaches duty `_LADDER_WMAX·k/(2·nharms)` and fold `k′` starts at
    # `k′/nharms`, so we need `k′ ≤ _LADDER_WMAX·k/2`.  `decimation_set` returns
    # `1:maxdecim`, whose worst ratio is 2, but a hand-built sparse ladder could
    # leave a duty-cycle hole — prune nothing rather than open one.
    all(ks[i+1] <= (_LADDER_WMAX ÷ 2) * ks[i] for i in 1:length(ks)-1) || return ws
    kmin, kmax = first(ks), last(ks)
    pruned = filter(w -> (w >= 2 || k == kmin) && (w <= _LADDER_WMAX || k == kmax), ws)
    # A profile short enough that its whole bank is `[1]` (tiny `nharms`, or a
    # deep decimation of one) has nothing to prune: dropping width 1 would leave
    # the fold with no filter at all.  Keep the bank as it stands.
    return isempty(pruned) ? ws : pruned
end

const _BOXCAR_SIGMA_SAMPLES = 8192     # bins subsampled per block to fix the noise σ

"""
    _block_sigma(M, nbins, n, buf) -> Float64

Robust per-bin noise scale (`1.4826 × MAD`) for one block, pooled over a
subsample of the `nbins × n` profile matrix `M[:, 1:n]` into `buf` (length
`_BOXCAR_SIGMA_SAMPLES`) — whole profiles, evenly spaced across the block, so
every read is contiguous.  When `buf` is long enough for the whole matrix
(`snr_metrics`' default) no subsampling happens at all.

A block carries thousands of noise bins, so pooling beats a per-profile MAD
(`~0.76/√nbins`, i.e. ~17% at `nbins=20`) by a wide margin: estimation noise here
multiplies straight into every boxcar S/N, and a per-profile estimate inflates the
small-`nbins` tail.  Median-based and pooled, so the rare signal/RFI bin does not
bias it.  Returns `0.0` for a degenerate (flat) block.

**It is not exact, and the residual is visible in reported S/N.**  Measured over
24 chunks against the all-bins estimator, 8192 samples give an unbiased σ̂ with
**~1% rms and ~3% worst case** at every fold depth — and the metric is `1/σ̂`, so
that is the precision of any single reported S/N.  Raise
`_BOXCAR_SIGMA_SAMPLES` (rms falls as `1/√cap`) if a use needs better; the
contiguous gather below makes the traffic cheap but the two `_median!` calls are
`O(cap)`.  See `Summary_and_Future_Work.md` §3.1.

The subsample indices depend only on `(nbins, n)`, and `M` enters only through the
ratio the caller forms (`excess/σ`), so the unnormalised `brfft` and normalised
`irfft` paths yield the identical scale-free S/N.
"""
function _block_sigma(M::AbstractMatrix{T}, nbins::Int, n::Int, buf::Vector{T};
                      center::Symbol=:zero) where {T<:AbstractFloat}
    # The linear indexing below is only the intended (i, j) if the profile axis is
    # exactly `M`'s first dimension — cheap to check once per block, and a silent
    # wrong σ̂ (hence a silently wrong metric everywhere) if it ever stops holding.
    size(M, 1) == nbins ||
        throw(DimensionMismatch("_block_sigma: size(M,1)=$(size(M,1)) != nbins=$nbins"))
    N = nbins * n
    N == 0 && return zero(T)
    cap = length(buf)
    ns = 0
    # `M` is column-major with `size(M, 1) == nbins`, so linear index `t` *is*
    # element (i, j) with `i = (t-1) % nbins + 1`, `j = (t-1) ÷ nbins + 1` — which
    # is why neither branch below reconstructs (i, j) arithmetically.
    if N <= cap
        # Every bin.  This is the estimator the Python oracle is pinned to
        # (`snr_metrics` defaults to `sigma_samples = typemax(Int)`); the
        # subsampled branch below must never leak into it.
        @inbounds for t in 1:N
            buf[t] = M[t]
        end
        ns = N
    else
        # Subsample WHOLE COLUMNS, spread evenly across the block.  This replaced
        # a flat stride-`N÷cap` walk of the linear index, for two reasons.
        #
        # Traffic: at `nbins=120, n=2048` that stride is 30 doubles = 240 B, so
        # it pulled a fresh cache line for *every* sample and used 8 of its 64
        # bytes — ~2.7 MB per chunk summed over the six fold depths, comparable
        # to the gate's own read of the same arrays.  Whole profiles are
        # contiguous and use every byte: ~0.4 MB for the same 8192 samples.
        #
        # Sampling: the flat stride visits bins `1, s+1, 2s+1, …` of every
        # column, so whenever `s | nbins` it sees only `nbins/s` *distinct
        # phases* — four of them, at five of the six default fold depths
        # (k=1,2,3,5,6; only k=4's `s=7` is coprime with its `nbins=30`).  A
        # per-phase artefact could therefore decide σ̂ outright.  Whole columns
        # sample every phase.
        ncol = min(n, max(1, cap ÷ nbins))        # profiles we can afford
        @inbounds for c in 1:ncol
            j = 1 + ((c - 1) * n) ÷ ncol          # evenly spaced over 1:n
            base = (j - 1) * nbins
            take = min(nbins, cap - ns)
            take == 0 && break
            for i in 1:take
                ns += 1; buf[ns] = M[base + i]
            end
        end
    end
    # `center = :zero` (production) folds about the location the construction
    # already guarantees: the profile spectrum's DC bin is held at zero, so every
    # profile's bin mean is *exactly* 0 and the pooled distribution is symmetric
    # about it.  `1.4826 × median(|x|)` is then the same scale estimator with the
    # location known rather than estimated — one `_median!` instead of two, and
    # statistically the better of the pair (it spends no degree of freedom on a
    # location it has).  Measured on real chunks, the sample median sits within
    # ±0.03σ of zero at every fold depth and σ̂ moves by ≤0.6%, against its own
    # ~1% sampling error.
    #
    # `center = :median` is the classic MAD-about-the-sample-median, which is what
    # the Python oracle computes.  It exists for `crossval/crossval_accuracy.jl`,
    # so that the metric pin stays at 1e-9 on everything it can still check (the
    # width bank, the per-profile median baseline, the scan arithmetic) instead of
    # being loosened to hide this one deliberate difference.  Nothing else calls
    # it; `test/test_search.jl` pins the two against each other.
    if center === :median
        med = _median!(buf, ns)                   # destroys buf order (values preserved)
        @inbounds for t in 1:ns
            buf[t] = abs(buf[t] - med)            # MAD over the same multiset
        end
    else
        @inbounds for t in 1:ns
            buf[t] = abs(buf[t])
        end
    end
    return T(1.4826) * _median!(buf, ns)
end

# Prefix sum of the profile column minus a scalar baseline `b`, tiled by one extra
# `wmax` samples so a boxcar that wraps past bin `nbins` reads real (wrapped) data:
# boxcar sum of bins p..p+w-1 (1-based) = psum[p+w] - psum[p].
@inline function _boxcar_psum!(psum::Vector{T}, col::AbstractVector{T},
                               nbins::Int, wmax::Int, b::T) where {T<:AbstractFloat}
    psum[1] = zero(T)
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
@inline function _boxcar_scan(psum::Vector{T}, widths::Vector{Int},
                              nbins::Int, invsigma::T) where {T<:AbstractFloat}
    best = T(-Inf)
    @inbounds stot = psum[nbins + 1] - psum[1]     # profile total, baseline removed
    @inbounds for w in widths
        duty = T(w) / T(nbins)
        invsw = invsigma / sqrt(T(w) * (one(T) - duty))
        m = psum[1 + w] - psum[1]                  # finite seed (no -Inf in the reduction)
        @simd for p in 2:nbins
            m = max(m, psum[p + w] - psum[p])
        end
        cand = (m - duty * stot) * invsw
        cand > best && (best = cand)
    end
    return best
end

"""
    boxcar_best_width(prof; fsp=1.5, maxfrac=0.3) -> (w, ducy)

Width of the boxcar in the geometric bank that best matches the pulse profile
`prof`, and the corresponding duty cycle `w / length(prof)`.

This is the width behind a candidate's reported S/N, recovered after
the fact: `_boxcar_scan` keeps only the peak S/N because it runs once per trial
(~1e8 times a search) and only reported candidates need the width.

The result does **not** depend on the noise scale `σ`: `σ` enters every width's
score as the same multiplicative factor, so it cannot change which width wins.
That is why this can be evaluated on an isolated profile, with no access to the
block statistics the search used.  It is otherwise the identical computation —
same `1/√(w(1−δ))` normalisation, same `δ·S_tot` baseline term, same wrapped
prefix sums, same bank — so the width it returns is the one the search's own scan
maximised.  (The mean is removed before the prefix sum purely for conditioning;
the statistic is invariant to it, which is what the `prof .+ 37.0` pin asserts.)
"""
function boxcar_best_width(prof::AbstractVector{<:Real}; fsp::Real=1.5, maxfrac::Real=0.3)
    nbins = length(prof)
    nbins >= 1 || throw(ArgumentError("profile must be non-empty"))
    col = prof isa Vector{Float64} ? prof : Vector{Float64}(prof)
    widths = boxcar_widths(nbins; fsp=fsp, maxfrac=maxfrac)
    psum = Vector{Float64}(undef, nbins + widths[end] + 1)
    _boxcar_psum!(psum, col, nbins, widths[end], sum(col) / nbins)  # mean: conditioning only
    stot = psum[nbins + 1] - psum[1]
    best, bestw = -Inf, widths[1]
    @inbounds for w in widths
        duty = w / nbins
        invsw = 1.0 / sqrt(float(w) * (1.0 - duty))
        m = psum[1 + w] - psum[1]
        for p in 2:nbins
            m = max(m, psum[p + w] - psum[p])
        end
        cand = (m - duty * stot) * invsw
        # Strictly-greater keeps the *narrowest* winner on a tie, matching
        # `_boxcar_scan`'s `cand > best` over the same ascending bank.
        cand > best && (best = cand; bestw = w)
    end
    return bestw, bestw / nbins
end

# ---------------------------------------------------------------------------
# Cross-profile (batched) boxcar gate
#
# `_boxcar_scan` vectorises along *phase*, which is the wrong axis: it is
# `nbins` long, and `nbins` is 120 only for the base pass — under decimation it
# falls to 20, where neither the prefix sum (a serial add chain) nor the max
# reduction can fill a vector, and the horizontal reduce at the end of every
# (width) iteration costs more than the reduction itself.  The *batch* axis is
# the one that is always long: a chunk holds `Nprof = 2048` profiles, sitting
# contiguously as the columns of `profs`.
#
# So transpose a tile of `B` profiles into a `(B, nbins)` buffer and run the
# identical recurrence with `b` innermost.  Every lane is then a different
# profile: the prefix sum's serial dependency becomes `B`-wide, and the phase
# scan becomes a plain element-wise `max` with no horizontal reduce at all.
# `B` is a compile-time constant (`Val`) — with a runtime `B` the inner loops do
# not unroll and the batched version is *slower* than the scalar one (measured).
#
# This runs the **gate**: the same statistic the scalar path computes, in
# `Float32`.  Any trial landing within `boxcar_gatemargin` of `threshold` is
# re-scored by the `Float64` scalar path, so every reported candidate's metric is
# the `Float64` one.
#
# The tile is `Float32` (`_BC_TILE`) while `profs` stays `Float64`: the
# conversion rides along in the transpose that has to happen anyway, and it
# doubles the AVX lane count over the whole gate.  Measured on real profiles it
# moves the value by at most ~3e-6 in metric units (6e-7 on `PM0063…red.fft` at
# `nbins` 120 and 20 alike, 2e-6 on the bundled test file), i.e. ~3 orders under
# the `boxcar_gatemargin = 0.01` the `Float64` rescore reserves, so it cannot
# change which trials get rescored and hence cannot change the candidate list.
# **Note this margin is two-sided now**: before 2026-08-24 the gate omitted the
# median baseline and so was a genuine lower *bound*; today it is the same
# statistic at lower precision, and can land either side of it.  It still relies
# on the profile mean being ~0 (DC is held at zero) to keep the prefix sum from
# drifting away from the boxcar sums it must resolve.  `test/test_search.jl` pins
# both properties.
# ---------------------------------------------------------------------------

const _BC_TILE = Float32          # gate tile/accumulator type (see above)
const _BC_BATCH = 128             # profiles per SIMD tile; see bench/boxcar_bench.jl
# 128, and emphatically not the 32 that fits in L1: the tile+prefix pair is 63 KB
# at `nbins = 120` and `B = 64`, already over the 32 KB L1, and 126 KB at 128.
# Widening past L1 wins anyway, on both machines this has been measured on.  The
# batched kernel's per-profile arithmetic is independent of `B`, so this is
# byte-identical, not an approximation — verified by `.cohout` diff.
#
# The knee is machine-dependent and the two hosts disagree, so `B` is set to the
# value that is best on one and free on the other:
#   * i7-10510U (AVX2, 4 cores): 32/48/64/96/128 → 692/628/575/572/577 µs per
#     chunk.  64 is the knee; 128 is 0.3% behind it, i.e. a wash.
#   * Xeon Silver 4114 (2×10 cores; nominally AVX-512, but LLVM emits 256-bit
#     vectors on Skylake-SP by default — see `DIRECT_GROUP_V`): in situ, the metric's
#     share of accounted time is 48.96/48.55/48.68/48.03/45.52% for
#     32/48/64/96/128 and 47.67/45.42% for 192/256.  Everything from 32 to 96 is
#     one flat plateau — the *absolute* spread across those four is whole-machine
#     drift, which is why the share is the thing to read — and only 128 moves it,
#     by 6.9% off the metric (~1.04x end to end at `-t 1`, reproduced twice).
# So 128 costs the laptop ~0.3% and buys the workstation ~4%.

"""
    BoxcarBatch{T}

Scratch for the batched boxcar gate over `_BC_BATCH` profiles at a time.
`tile` and `psT` are flat vectors with a *static* row stride `B = _BC_BATCH`, so
element `(b, i)` lives at `[(i-1)*B + b]` and the `b` loops compile to plain
contiguous vector ops.  `mvals` holds one metric per trial of the chunk.
"""
struct BoxcarBatch{T}
    tile::Vector{T}          # (B, nbins)               transposed profile tile
    psT::Vector{T}           # (B, nbins + wmax + 1)    transposed prefix sums
    res::Vector{T}           # (B,)  running best over widths
    mbuf::Vector{T}          # (B,)  running best over phase, one width
    mvals::Vector{Float64}   # (Nprof,) per-trial metric
end

function BoxcarBatch(nbins::Integer, wmax::Integer, Nprof::Integer,
                     ::Type{T}=_BC_TILE, B::Integer=_BC_BATCH) where {T}
    return BoxcarBatch{T}(Vector{T}(undef, B * nbins),
                          Vector{T}(undef, B * (nbins + wmax + 1)),
                          Vector{T}(undef, B), Vector{T}(undef, B),
                          Vector{Float64}(undef, Nprof))
end

# Profiles per inner block of `_bc_transpose!`.  See the comment there: this is
# the single most host-sensitive constant in the file, and 8 is the value that is
# a large win on one machine and nearly free on the other.
const _BC_TR_BJ = 8

# Transpose profile columns `j0+1 .. j0+B` into `tile` as (B, nbins), converting
# to the tile's eltype.  Writes are contiguous (a `BJ`-wide run of one tile row);
# the READ `profs[i, j0+b]` walks profiles, so it is a stride-`nbins` gather — a
# fully-used one (every byte of every line is read by some `i`), which LLVM
# vectorises.  Note the direction: an earlier note had this as write-scatter and
# proposed cache-line-filling write blocks, which is the wrong end — the writes
# already fill lines, and blocking them is measurably worse on both hosts.
#
# **What matters is `BJ`: how many profiles are in flight at once**, i.e. how many
# concurrent strided read streams the loop asks the machine to sustain.  The
# unblocked nest asks for `B = 128` of them, and the two hosts disagree about that
# by a factor of 3.5 (µs per chunk summed over the default `k = 1…6` ladder,
# `Nprof = 2048`, `Float64` profiles; `bench/tile_shape_bench.jl`):
#
#   BJ:            4      8     16     32     64    128 (unblocked)
#   i7-10510U:   354    325    377    372    354    295
#   Xeon 4114:   464    431   1840   1764   1765   1514
#
# On the Xeon everything from 16 up is one flat plateau at the unblocked cost and
# 8 is **3.51x** faster than it (2.40x in `Float32`); on the laptop the ordering is
# inverted and 8 measures 0.91x here (1.01x in `Float32`).  Blocking the *phase*
# axis as well (the 4x8, 8x8 shapes) is slower than blocking `b` alone on both.
#
# **In situ the laptop's 0.91x does not appear — it is a small win there too.**
# Interleaved, `-t 1`, metric share of accounted time: laptop 35.66% → 34.81%
# (10.79 → 10.50 s), Xeon 48.93% → **31.60%** (19.41 → 14.32 s, i.e. **1.36x
# end-to-end**).  The microbench above re-reads `profs` hot; the search reads it
# once, right after the `brfft` that wrote it, so cutting 128 concurrent read
# streams to 8 pays off on both machines.
#
# This supersedes the recorded verdict that the transpose sat at "an L3 bandwidth
# wall": it does not.  At `B = 128` the Xeon ran it at 4.7 GB/s against the 21 GB/s
# that host actually delivers at this footprint, and the rate's independence of
# size was the signature of an access-pattern limit, which is what `BJ` fixes.
# The output is bit-identical for every `BJ` — this is a loop nest, not a method.
@inline function _bc_transpose!(tile::Vector{T}, profs::AbstractMatrix{<:AbstractFloat},
                                j0::Int, nbins::Int, ::Val{B},
                                ::Val{BJ}=Val(_BC_TR_BJ)) where {T,B,BJ}
    B % BJ == 0 || throw(ArgumentError("_BC_BATCH=$B must be a multiple of BJ=$BJ"))
    @inbounds for b0 in 0:BJ:(B - 1)
        for i in 1:nbins
            o = (i - 1) * B + b0
            @simd for b in 1:BJ
                tile[o + b] = T(profs[i, j0 + b0 + b])
            end
        end
    end
end

# The zero-baseline `_boxcar_psum!` + `_boxcar_scan` of `B` profiles at once,
# leaving each profile's peak matched-filter S/N in `res[1:B]`.  Same recurrence
# and same operation order per profile as the scalar pair; only the axis the
# vectoriser sees is different.
@inline function _bc_scan_batch!(bb::BoxcarBatch{T}, widths::Vector{Int},
                                 nbins::Int, invsigma::T, ::Val{B}) where {T,B}
    tile = bb.tile; psT = bb.psT; res = bb.res; mbuf = bb.mbuf
    wmax = widths[end]
    @inbounds for b in 1:B
        psT[b] = zero(T)
    end
    @inbounds for i in 1:(nbins + wmax)
        idx = i > nbins ? i - nbins : i        # wrap: a boxcar may straddle phase 0
        o = (i - 1) * B
        t = (idx - 1) * B
        @simd for b in 1:B
            psT[o + B + b] = psT[o + b] + tile[t + b]
        end
    end
    @inbounds for b in 1:B
        res[b] = T(-Inf)
    end
    to = nbins * B                         # offset of each lane's profile total
    @inbounds for w in widths
        duty = T(w) / T(nbins)
        invsw = invsigma / sqrt(T(w) * (one(T) - duty))
        wo = w * B
        @simd for b in 1:B                     # finite seed, as in `_boxcar_scan`
            mbuf[b] = psT[wo + b] - psT[b]
        end
        for p in 2:nbins
            o = (p - 1) * B
            @simd for b in 1:B
                d = psT[o + wo + b] - psT[o + b]
                mbuf[b] = ifelse(d > mbuf[b], d, mbuf[b])
            end
        end
        @simd for b in 1:B
            c = (mbuf[b] - duty * (psT[to + b] - psT[b])) * invsw
            res[b] = ifelse(c > res[b], c, res[b])
        end
    end
end

# Zero-baseline gate for every column `1:n`, into `bb.mvals`.  Full `B`-tiles go
# through the batched kernel; the `< B` leftover columns take the scalar one
# (both are valid lower bounds, which is all the gate needs).
function _boxcar_gate!(bb::BoxcarBatch{T}, profs::AbstractMatrix{P}, n::Int,
                       psum::Vector{P}, widths::Vector{Int}, nbins::Int,
                       invsigma::P) where {T,P<:AbstractFloat}
    B = _BC_BATCH
    vb = Val(B)
    invsig_t = T(invsigma)
    mvals = bb.mvals
    j0 = 0
    @inbounds while j0 + B <= n
        _bc_transpose!(bb.tile, profs, j0, nbins, vb)
        _bc_scan_batch!(bb, widths, nbins, invsig_t, vb)
        for b in 1:B
            mvals[j0 + b] = Float64(bb.res[b])
        end
        j0 += B
    end
    wmax = widths[end]
    @inbounds for j in (j0 + 1):n
        col = @view profs[:, j]
        _boxcar_psum!(psum, col, nbins, wmax, zero(P))
        mvals[j] = Float64(_boxcar_scan(psum, widths, nbins, invsigma))
    end
    return mvals
end

"""
    boxcar_metrics!(bb, profs, n, psum, widths, nbins, invsigma, exactcut)

Peak boxcar matched-filter S/N of every profile column `1:n`, into `bb.mvals`.

With `exactcut > -∞` this is the two-phase production path: one *batched*
`Float32` pass scores all `n` trials ([`_boxcar_gate!`](@ref)), then only the
trials that reach `exactcut` are re-scored in `Float64` by
[`_profile_boxcar`](@ref).  With `exactcut = -∞` every trial is scored in
`Float64` directly — the reference/metricstats/normalize paths, which need every
raw metric at full precision.

Identical results to a per-column `_profile_boxcar` loop for every trial that can
become a candidate; see the section comment above for why the `Float32` gate
cannot move that set.
"""
function boxcar_metrics!(bb::BoxcarBatch, profs::AbstractMatrix{P}, n::Int,
                         psum::Vector{P}, widths::Vector{Int}, nbins::Int,
                         invsigma::P, exactcut::Float64) where {P<:AbstractFloat}
    mvals = bb.mvals
    if exactcut == -Inf || invsigma <= 0
        @inbounds for j in 1:n
            mvals[j] = Float64(_profile_boxcar(profs, j, psum, widths, nbins, invsigma))
        end
        return mvals
    end
    _boxcar_gate!(bb, profs, n, psum, widths, nbins, invsigma)
    @inbounds for j in 1:n
        mvals[j] < exactcut && continue       # cannot reach threshold: keep the Float32 value
        mvals[j] = Float64(_profile_boxcar(profs, j, psum, widths, nbins, invsigma))
    end
    return mvals
end

"""
    _profile_boxcar(profs, j, psum, widths, nbins, invsigma) -> Float64

Peak boxcar matched-filter S/N of profile column `j` (see the section comment).
`psum` (length `≥ nbins + widths[end] + 1`) holds the phase-tiled prefix sum.
`invsigma = 1/σ` is the block's robust per-bin noise scale
([`_block_sigma`](@ref)) — shared across the block so its (negligible) estimation
noise does not leak into the per-trial statistic, which is then exactly `N(0,1)`
per (phase, width) under white noise.

The statistic is
`max_{w,p} (Σ_{i=p}^{p+w-1} P_i − δ·Σᵢ Pᵢ) · invsigma / √(w(1−δ))` with
`δ = w/nbins`: a ratio of two linear-in-amplitude quantities, hence invariant to
the profile's overall scale — so the unnormalised hot-loop `brfft` and the
normalised reference `irfft` yield the identical value, and neither `ngoodbins`
nor the `scale` factor is needed.  It is likewise invariant to any constant
subtracted from `P` beforehand, which is why the prefix sum here can run against
a zero baseline (DC is held at zero, so the profile mean is already ~0 and the
`δ·S_tot` term is a rounding-level correction) while [`snr_metrics`](@ref)
removes the mean first for conditioning on arbitrary input.
"""
@inline function _profile_boxcar(profs::AbstractMatrix{T}, j::Integer,
                                 psum::Vector{T}, widths::Vector{Int},
                                 nbins::Int, invsigma::T) where {T<:AbstractFloat}
    invsigma > 0 || return zero(T)                # degenerate (flat block): no detection
    col = @view profs[:, j]
    _boxcar_psum!(psum, col, nbins, widths[end], zero(T))
    return _boxcar_scan(psum, widths, nbins, invsigma)
end

"""
    snr_metrics(profs; boxcar_fsp=1.5, boxcar_maxfrac=0.3, sigma_center=:zero) -> Vector{Float64}

Peak boxcar matched-filter detection metric for every profile (column) of the
`(nbins × L)` real profile matrix `profs`.  Public port of `snr_metric` from the
Python `coherent_search` (note the profiles are columns here, rows in Python),
and the function `crossval/crossval_accuracy.jl` pins to that oracle.

This is the *reference* implementation: readable, allocating, and — by default —
pooling `σ̂` over every bin of every profile.  The production search instead
subsamples to `_BOXCAR_SIGMA_SAMPLES` bins (see [`_block_sigma`](@ref)), a
deliberate approximation the hot loop can afford and the oracle comparison
cannot; `sigma_samples` exists so [`block_metrics`](@ref) can ask for that same
estimator when it is standing in as the optimised path's equivalence partner.

`sigma_center` is the one place this port *deliberately* differs from Python.
Both default to `:zero` here and in the search — folding `σ̂` about the exact
zero the DC-held-at-zero construction guarantees, rather than about a sample
median estimating it — which is faster and statistically better, but is not what
`snr_metric` does upstream.  Pass `:median` for the Python-identical estimator;
`crossval/crossval_accuracy.jl` does exactly that, so its 1e-9 pin still covers
everything else in the metric.

The metric is scale-free — a ratio of two linear-in-amplitude quantities — so it
does not care whether `profs` came from a normalised `irfft` or the unnormalised
`brfft` the hot loop uses.
"""
function snr_metrics(profs::AbstractMatrix{<:Real};
                     boxcar_fsp::Real=1.5, boxcar_maxfrac::Real=0.3,
                     sigma_samples::Integer=typemax(Int),
                     sigma_center::Symbol=:zero,
                     widths::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    nbins, L = size(profs)
    P = profs isa Matrix{Float64} ? profs : convert(Matrix{Float64}, profs)
    # `widths` overrides the derived bank so this function can stand in for a
    # *pruned* ladder fold (see `ladder_boxcar_widths`); left `nothing` it builds
    # the full geometric bank, which is what the Python oracle is pinned to.
    widths = widths === nothing ?
        boxcar_widths(nbins; fsp=boxcar_fsp, maxfrac=boxcar_maxfrac) : collect(Int, widths)
    psum = Vector{Float64}(undef, nbins + widths[end] + 1)
    sigbuf = Vector{Float64}(undef, min(nbins * L, sigma_samples))
    sigma = _block_sigma(P, nbins, L, sigbuf; center=sigma_center)   # one robust σ for the set
    invsigma = sigma > 0 ? 1.0 / sigma : 0.0
    # Remove each profile's mean before the prefix sum.  The statistic is exactly
    # invariant to it (see `_profile_boxcar`); this is conditioning only, so that
    # profiles carrying a large constant offset — which the search's own never do,
    # but a caller's may — do not lose digits in the prefix sum.
    mean_j = [sum(view(P, :, j)) / nbins for j in 1:L]
    return [begin
                _boxcar_psum!(psum, view(P, :, j), nbins, widths[end], mean_j[j])
                invsigma > 0 ? _boxcar_scan(psum, widths, nbins, invsigma) : 0.0
            end for j in 1:L]
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
    reference_profiles(ft, rfund, params; kernel=:fft) -> Matrix{Float64}

Build the `(2*nharms, L)` real coherent-fold pulse profiles (one column per
trial fundamental in `rfund`) via the simple, allocating reference path, kept
separate from the detection metric so each can be cross-validated on its own.

`kernel` selects the interpolation:

  * `:fft` (default) — one [`finterp_fft`](@ref) per harmonic onto a uniform
    `numbetween` grid, then linear interpolation onto the trial frequencies.
    This is what the Python original does, so it is the form the oracle
    cross-validation pins, and it carries that method's ~1e-2 linear-interpolation
    error in the amplitudes.
  * `:direct` — [`fourier_interp`](@ref), Eqn. 30 evaluated point by point with
    no grid and no linear interpolation.  Slower (it recomputes the `m` weights
    per point) and *exact*, which is what makes it the reference the production
    hot loop can be pinned against: the production path tabulates the same
    weights, so the two differ only by tabulation rounding (~1e-10), not by
    method.

Both share every other step, so switching `kernel` isolates the interpolator.
"""
function reference_profiles(ft::FFTFile, rfund::AbstractVector{<:Real},
                            params::SearchParams; kernel::Symbol=:fft)
    kernel in (:fft, :direct) ||
        throw(ArgumentError("kernel must be :fft or :direct, got :$kernel"))
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

        if kernel === :fft
            amps = finterp_fft(lobin, numbins, nb, ft.amps, m)
            @inbounds for j in 1:L
                ftprofs[h + 1, j] = uniform_linear_interp(rfund[j] * h, lobin, nb, amps)
            end
        else
            @inbounds for j in 1:L
                ftprofs[h + 1, j] = fourier_interp(rfund[j] * h, ft.amps, m)
            end
        end
    end

    return coherent_profiles(ftprofs, 2nh)
end

"""
    block_metrics(ft, rfund, params; kernel=:fft) -> Vector{Float64}

Compute the coherent-fold detection S/N (see [`snr_metrics`](@ref)) for every
trial fundamental Fourier frequency in `rfund`.  Self-contained reference
implementation built on [`reference_profiles`](@ref), whose `kernel` it forwards:
the default `:fft` is the computation the Python oracle reproduces in
cross-validation, and `:direct` is the exact-kernel form the optimised path is
pinned against (see [`chunk_metrics`](@ref)).
"""
function block_metrics(ft::FFTFile, rfund::AbstractVector{<:Real},
                       params::SearchParams; kernel::Symbol=:fft,
                       widths::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    nh = params.nharms
    nbins = 2nh
    L = length(rfund)
    profs = reference_profiles(ft, rfund, params; kernel=kernel)
    # Default to the bank the production *base* (k = 1) pass would use, so the
    # equivalence gate stays exact when `params` carries a decimation ladder.
    widths = widths === nothing ? ladder_boxcar_widths(nbins, 1, params) : widths
    # `sigma_samples` matched to the production path: this function's job is to be
    # the thing `chunk_metrics` must equal to machine precision, and σ̂ is part of
    # the metric.  `snr_metrics` on its own defaults to the exact pooled MAD, which
    # is what the Python oracle computes; the two agree whenever
    # `nbins*L ≤ _BOXCAR_SIGMA_SAMPLES`, which the tests pin.
    return snr_metrics(profs; boxcar_fsp=params.boxcar_fsp,
                       boxcar_maxfrac=params.boxcar_maxfrac,
                       sigma_samples=_BOXCAR_SIGMA_SAMPLES, widths=widths)
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

# FFTW planning rigor used by every plan constructor below.  A mutable ref (not a
# hardcoded flag) so callers can raise or lower it around a planning pass; see
# `prime_wisdom` in `wisdom.jl`.
#
# **`PATIENT` by default, because it is measurably better and self-amortising.**
# Measured 2026-08-22 (Xeon Silver 4114, nharms=60, decimations 1…6, Nprof=2048),
# interleaved over two rounds against the same wisdom file primed each way:
# `decim-brfft` 4.69 → 4.24 s and 4.83 → 4.40 s, i.e. **~9% off the decimated
# transforms**, ~2–3% end to end.  The base `brfft` does not move (2.30 → 2.28):
# the win is entirely in the *strided* decimated plans, which is where FFTW has a
# real choice to make.
#
# The cost is paid once.  Cold planning of all six transforms is 0.006 s at
# `ESTIMATE`, 0.30 s at `MEASURE`, **1.30 s at `PATIENT`** — and 0.0155 s once the
# result is in the wisdom file, which `search` imports and exports by default.  So
# `PATIENT` costs ~1 s on the first run of a given (host, nharms, decimations,
# blocksize, precision) and nothing after, against ~0.5 s saved on every 20 s run.
#
# **`EXHAUSTIVE` is not worth it and was measured, not assumed**: 14.96 s of
# planning for an FFT share of 34.4% against `PATIENT`'s 33.9%, i.e. no better
# than drift.  `PATIENT` is the knee.
#
# `search` drops back to `MEASURE` when `wisdom=false`, since without a wisdom
# file to cache into, `PATIENT` would re-pay its 1.3 s on *every* run.
const _PLAN_RIGOR = Ref{UInt32}(FFTW.PATIENT)
plan_rigor() = _PLAN_RIGOR[]

"""
    with_plan_rigor(f, rigor)

Run `f()` with the FFTW planning rigor temporarily set to `rigor`, restoring it
afterwards.  Planning is single-threaded and happens before the parallel region,
so a process-wide ref is safe here; do not call this from inside a search.
"""
function with_plan_rigor(f, rigor::Integer)
    old = _PLAN_RIGOR[]
    _PLAN_RIGOR[] = UInt32(rigor)
    try
        return f()
    finally
        _PLAN_RIGOR[] = old
    end
end

"""
    DecimBuf

Per-decimation-factor `k` scratch for the harmonic-decimation multi-frequency
search: the real profile array `dprofs` (`(2Hₖ, Nprof)`, `Hₖ = ⌊nharms/k⌋`), a
per-profile median buffer, and the batched complex→real inverse plan.  Built once
per `Workspace`; never shared.

**There is no decimated amplitude stack.**  The `k`-decimated stack is rows
`1, k+1, 2k+1, …, Hₖk+1` of the workspace's `ftprofs` — a uniform stride-`k`
slice — and `src` is exactly that view, which `brfftplan` transforms in place.
The DC row it starts on is `ftprofs`' own DC row, which the search never writes
and `fill_chunk_profiles!` zeroes, so it is the zero a compact stack would have
put there.

This replaces a strided row-by-row copy into a compact `(Hₖ+1, Nprof)` buffer
that the transform then re-read.  Letting FFTW take the stride is **1.36x
(`Float64`) / 1.60x (`Float32`) faster than copy-then-transform** across
`k = 2…6` at `nharms = 60, Nprof = 2048` — the copy was reading exactly the same
elements the transform reads, so it was pure duplicated traffic — and it removes
`Σₖ (Hₖ+1)·Nprof` complex words per workspace, which is most of the per-thread
footprint.
"""
struct DecimBuf{B,P<:AbstractFloat,V<:AbstractMatrix{<:Complex}}
    k::Int
    Hk::Int
    src::V                         # view(ftprofs, 1:k:(Hk*k+1), :) — the decimated stack
    dprofs::Matrix{P}             # (2*Hk, Nprof)
    brfftplan::B                   # plan_brfft(src, 2*Hk, 1)
    bcwidths::Vector{Int}          # boxcar width bank for 2*Hk-bin profiles
    bcpsum::Vector{P}              # prefix-sum scratch (2*Hk + wmax + 1)
    bcsig::Vector{P}               # per-block σ subsample scratch
    bcbatch::BoxcarBatch{_BC_TILE}    # cross-profile SIMD gate scratch + metrics
end

# `ftprofs` is the workspace array this buffer decimates; the plan is built
# against a view of it and must only ever be executed on that same view (FFTW
# plans encode strides and alignment), which is why the view is stored.
function DecimBuf(k::Integer, nharms::Integer, ftprofs::Matrix{Complex{P}},
                  params::SearchParams) where {P<:AbstractFloat}
    Hk = fld(nharms, k)
    Nprof = size(ftprofs, 2)
    src = @view ftprofs[1:k:(Hk * k + 1), :]
    dprofs   = Matrix{P}(undef, 2Hk, Nprof)
    brfftplan = plan_brfft(src, 2Hk, 1; flags=plan_rigor())
    bcwidths = ladder_boxcar_widths(2Hk, k, params)
    bcpsum   = Vector{P}(undef, 2Hk + bcwidths[end] + 1)
    bcsig    = Vector{P}(undef, min(2Hk * Nprof, _BOXCAR_SIGMA_SAMPLES))
    bcbatch  = BoxcarBatch(2Hk, bcwidths[end], Nprof)
    return DecimBuf(Int(k), Hk, src, dprofs, brfftplan, bcwidths, bcpsum,
                    bcsig, bcbatch)
end

"""
    Workspace

Everything one task needs to process a chunk with zero allocation in the hot
loop: the stacked-harmonic amplitude array `ftprofs`, the real profile array
`profs`, the prebuilt batched complex→real inverse plan, the interpolator's
de-interleaved bin-window planes, and a [`DecimBuf`](@ref) for each
harmonic-decimation factor `k > 1`.  One `Workspace` per task; never shared.
"""
# Parameterised on the *concrete* inverse-plan (`B`) and decimation-buffer (`D`)
# types so field access stays type-stable: an untyped `Vector{DecimBuf}` would
# make `db.brfftplan` `::Any`, turning every `mul!` in the hot loop into a
# dynamic dispatch (and boxing its result).
struct Workspace{B, D<:DecimBuf, P<:AbstractFloat}
    ftprofs::Matrix{Complex{P}}   # (nharms+1, Nprof)
    profs::Matrix{P}              # (2*nharms, Nprof)
    brfftplan::B                   # plan_brfft(ftprofs, 2*nharms, 1)
    bcwidths::Vector{Int}          # boxcar width bank for the base 2*nharms-bin profiles
    bcpsum::Vector{P}              # prefix-sum scratch (2*nharms + wmax + 1)
    bcsig::Vector{P}               # per-block σ subsample scratch
    bcbatch::BoxcarBatch{_BC_TILE}    # cross-profile SIMD gate scratch + metrics
    decims::Vector{D}              # one per decimation factor k > 1
    # The chunk's bin window, de-interleaved into real/imaginary planes.  Kept at
    # `Float32`, the storage type of the `.fft` file: widening on load is exact,
    # so the accumulated sum is bit-identical to holding `Float64` planes, while
    # halving the traffic through the inner loop (~1.1x measured).
    re::Vector{Float32}
    im::Vector{Float32}
end

Workspace(params::SearchParams, Nprof::Integer) =
    Workspace(proftype(params), params, Nprof)

# As with `DecimBuf`: `P` arrives as a type argument so the body (and the
# `decims` vector it builds) infers concretely.
function Workspace(::Type{P}, params::SearchParams, Nprof::Integer) where {P<:AbstractFloat}
    nh = params.nharms
    ftprofs = zeros(Complex{P}, nh + 1, Nprof)
    profs   = Matrix{P}(undef, 2nh, Nprof)
    brfftplan = plan_brfft(ftprofs, 2nh, 1; flags=plan_rigor())
    bcwidths = ladder_boxcar_widths(2nh, 1, params)   # the base pass is k = 1
    bcpsum   = Vector{P}(undef, 2nh + bcwidths[end] + 1)
    bcsig    = Vector{P}(undef, min(2nh * Nprof, _BOXCAR_SIGMA_SAMPLES))
    bcbatch  = BoxcarBatch(2nh, bcwidths[end], Nprof)
    # A DecimBuf per k > 1 (k = 1 is the base ftprofs/profs above).  `map` (not a
    # `DecimBuf[...]` comprehension) keeps the element type the concrete
    # `DecimBuf{B}` even when empty, so `Workspace`'s `D<:DecimBuf` stays concrete.
    decims = map(k -> DecimBuf(k, nh, ftprofs, params), filter(>(1), params.decimations))
    # Every plan above (base and decimated) is built with `FFTW.MEASURE`, which
    # writes into the arrays it plans on — and the decimated plans plan on views
    # of `ftprofs`.  So zero it *after* the last plan, not after the first.
    fill!(ftprofs, 0)
    nw = direct_window_size(params, Nprof)
    re = Vector{Float32}(undef, nw)
    im = Vector{Float32}(undef, nw)
    return Workspace(ftprofs, profs, brfftplan, bcwidths, bcpsum,
                     bcsig, bcbatch, decims, re, im)
end

"""
    fill_chunk_profiles!(ws, dplans, ft, params, rstart, lodr, n; t0=0)

Fill `ws.ftprofs` (zeroed first) and inverse-FFT it into `ws.profs` for a chunk
of `n` trial fundamentals starting at fundamental Fourier frequency `rstart`,
stepping by `lodr` bins.  After this call `ws.profs[:, 1:n]` holds the real
coherent-fold profiles.

The harmonic rows come from `dplans` (built by [`build_direct_plans`](@ref)
against the search's global `r_lo`) and `t0` is the chunk's *global* trial index,
which is what the exact phase bookkeeping keys off; `rstart` is unused for
interpolation and only carried for the callers that report it.
"""
function fill_chunk_profiles!(ws::Workspace, dplans::AbstractVector, ft::FFTFile,
                              params::SearchParams, rstart::Real, lodr::Real, n::Integer;
                              t0::Integer=0)
    @phase 6 fill!(ws.ftprofs, 0)
    @phase 1 for dp in dplans
        fill_harmonic_row_direct!(ws, dp, ft, params, t0, n)
    end
    # One batched complex→real transform for all Nprof profiles at once.  This
    # is an unnormalised `brfft` (= Nbins × a true irfft).  The boxcar metric is
    # scale-free — a ratio of two linear-in-amplitude quantities — so the missing
    # 1/Nbins cancels and the profiles are left unnormalised.
    @phase 2 mul!(ws.profs, ws.brfftplan, ws.ftprofs)
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
count.  They were written for the retired on-pulse metrics, whose noise floor
scaled ~`√nbins = √(2·Hk)`, so the low-`k` (more-bin) decimations sat at a
systematically higher floor and dominated the candidate list at a fixed
`threshold` — the defect the boxcar matched filter was written to fix, since its
per-trial statistic is `N(0,1)` at every width and so has a `k`-independent
floor.  They remain the way to *check* that on real data (red noise and RFI are
not white, so the analytic flatness is a claim about the noise model, not about a
given observation) and to choose a `--threshold` from measured distributions.
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
                     norm::Union{Nothing,MetricNorm}=nothing, exactcut::Real=-Inf)
    k = db.k
    Hk = db.Hk
    nbins = 2Hk
    # No gather: `db.src` already *is* the decimated stack (a stride-`k` view of
    # `ws.ftprofs`), so the transform reads the base amplitudes where they lie.
    @phase 7 @phase _decim_brfft_slot(k) mul!(db.dprofs, db.brfftplan, db.src)

    rmean = rstart + (n - 1) * lodr / 2
    nyq = ft.N / 2
    # Valid decimated trials are the prefix j = 1..nvalid (k·rf increases with j).
    nvalid = 0
    @inbounds while nvalid < n && k * (rstart + nvalid * lodr) < nyq
        nvalid += 1
    end
    # One robust per-bin σ for this (block, k), from the valid profiles only (past-
    # Nyquist columns are partly zero-padded and would deflate it).
    P = eltype(db.dprofs)
    @phase 8 begin
        sig = _block_sigma(db.dprofs, nbins, nvalid, db.bcsig)
        invsigma = sig > 0 ? one(P) / sig : zero(P)
        # Only the valid prefix is scored; past-Nyquist columns are skipped below.
        boxcar_metrics!(db.bcbatch, db.dprofs, nvalid, db.bcpsum,
                        db.bcwidths, nbins, invsigma, Float64(exactcut))
    end
    mbuf = stats === nothing ? nothing : Float64[]     # gather metrics if requested
    @phase 9 @inbounds for j in 1:n
        r_dec = k * (rstart + (j - 1) * lodr)
        r_dec < nyq || continue                       # fundamental past Nyquist
        mval = db.bcbatch.mvals[j]
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
        # `ngoodbins` is a recorded diagnostic, not an input to the metric, so it
        # is computed here rather than every chunk.
        ngood = chunk_ngoodbins(ft, Hk, k * rmean)
        push!(stats, _block_stats(block, k, Hk, nbins, ngood, flo, fhi, mbuf))
    end
    return
end

"""
    chunk_metrics(ft, params, rstart, n; lodr) -> Vector{Float64}

Single-threaded convenience that runs the optimised path over one chunk of `n`
trial fundamentals starting at `rstart` and returns the S/N metric for each.

This is the bridge the test suite uses to pin the optimised path to the
oracle-validated [`block_metrics`](@ref) reference.  Both evaluate the exact
Eqn.-30 kernel — `block_metrics` point by point through
[`reference_profiles`](@ref), this through the tabulated hot-loop path — so on
identical grids they agree to ~1e-10, limited by the tabulation rather than by
either being an approximation of the other.

`weights` selects the interpolation weight type and defaults to `Float32`, as the
search does.  The machine-precision equivalence pins pass `Float64` so that they
keep measuring *tabulation* rounding rather than the deliberate `Float32` trade
(see [`build_direct_plans`](@ref)); at `Float32` the same comparison lands at
~2e-7, which a separate testset pins.
"""
function chunk_metrics(ft::FFTFile, params::SearchParams, rstart::Real, n::Integer;
                       lodr::Real = params.hidr / params.nharms,
                       weights::Type{<:AbstractFloat} = Float32)
    nh = params.nharms
    nbins = 2nh
    ws = Workspace(params, n)
    # This chunk *is* the whole "search", so its global trial index starts at 0
    # and the direct plans are built against `rstart` as `r_lo`.
    dplans = build_direct_plans(weights, params, rstart)
    fill_chunk_profiles!(ws, dplans, ft, params, rstart, lodr, n; t0=0)
    P = eltype(ws.profs)
    sigma = _block_sigma(ws.profs, nbins, n, ws.bcsig)
    invsigma = sigma > 0 ? one(P) / sigma : zero(P)
    return [Float64(_profile_boxcar(ws.profs, j, ws.bcpsum, ws.bcwidths,
                                    nbins, invsigma)) for j in 1:n]
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
function _search_region!(ft::FFTFile, params::SearchParams,
                         workspaces::Vector{<:Workspace}, nbins::Integer,
                         r_lo::Real, r_hi::Real, lodr::Real, total::Integer,
                         Nprof::Integer, nchunks::Integer, nt::Integer;
                         threshold::Real, norm::Union{Nothing,MetricNorm},
                         metricstats::Union{Nothing,MetricStats}, progress::Symbol,
                         dplans::AbstractVector)
    collect_stats = metricstats !== nothing
    # Fast path: keep the batched `Float32` gate's value for trials that land more
    # than `boxcar_gatemargin` below `threshold`, and re-score the rest in `Float64`
    # (see `boxcar_metrics!`).  Forced off (all `Float64`) when collecting stats or
    # normalising, which need every raw metric at full precision.
    exactcut = (!collect_stats && norm === nothing) ?
        threshold - params.boxcar_gatemargin : -Inf
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
            P = eltype(ws.profs)
            c = t
            while c <= nchunks
                i0 = (c - 1) * Nprof
                n = min(Nprof, total - i0)
                rstart = r_lo + i0 * lodr
                fill_chunk_profiles!(ws, dplans, ft, params, rstart, lodr, n; t0=i0)
                rmean = rstart + (n - 1) * lodr / 2
                # one robust σ per block
                @phase 3 begin
                    sig = _block_sigma(ws.profs, nbins, n, ws.bcsig)
                    invsigma = sig > 0 ? one(P) / sig : zero(P)
                end
                # All n trials at once: batched `Float32` gate, then a `Float64`
                # rescore only for those that reach `exactcut`.
                @phase 4 boxcar_metrics!(ws.bcbatch, ws.profs, n, ws.bcpsum,
                                         ws.bcwidths, nbins, invsigma, exactcut)
                # Whole (narrow) block → one window, keyed by its centre freq.
                basehist = collect_stats ?
                    hists[1][_window_index(wedges[1], rmean / ft.T)] : nothing
                @phase 5 for j in 1:n
                    metric = ws.bcbatch.mvals[j]
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
                    ngood = chunk_ngoodbins(ft, params.nharms, rmean)   # diagnostic only
                    push!(stats, _block_stats(c, 1, params.nharms, nbins, ngood, flo, fhi, mbuf[1:n]))
                end
                # Harmonic-decimation multi-frequency passes (k > 1), re-using the
                # base harmonic amplitudes already in ws.ftprofs (see decimation_design.md).
                for db in ws.decims
                    whist = collect_stats ?
                        hists[db.k][_window_index(wedges[db.k], db.k * rmean / ft.T)] : nothing
                    decim_pass!(out, ws, db, ft, params, rstart, lodr, n;
                                threshold=threshold, block=c, stats=stats, hist=whist, norm=norm,
                                exactcut=exactcut)
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
    SearchCache()

Reusable per-`SearchParams` setup — the harmonic plans and the per-task
[`Workspace`](@ref)s — so a run over *many* `.fft` files pays for them once
instead of once per file.  Pass the same cache to every [`search`](@ref) call;
it is repopulated automatically whenever `params` or `blocksize` changes.

Only setup that is independent of the file is cached.  The direct interpolator's
phase tables depend on the starting Fourier bin (`lofreq * ft.T`), which varies
with the observation length, so those are rebuilt per file — they are cheap once
warm.  Reuse is keyed on the `params` *object identity*, not on `==`: build one
`SearchParams` and hand the same one to every call.

Not thread-safe: like the `Workspace`s it holds, a cache belongs to the serial
setup phase.  FFTW planning is not thread-safe either, which is the whole reason
workspaces are built outside the parallel region.
"""
mutable struct SearchCache
    params::Union{Nothing,SearchParams}
    Nprof::Int
    # A concretely-typed `Vector{Workspace{B,D,P}}` held behind an `Any` field:
    # `B`/`D`/`P` are not known until `params` is, and this is a once-per-file
    # setup value.  Extracting it costs one dynamic dispatch at the
    # `_search_region!` call, which is a function barrier — everything inside is
    # specialised on the concrete element type, so the hot loop is untouched.
    workspaces::Any
end
SearchCache() = SearchCache(nothing, 0, nothing)

"""
    _plans!(cache, params, Nprof, nt) -> workspaces

Fetch at least `nt` workspaces from `cache`, building (or extending) them as
needed.  With `cache === nothing`, builds fresh ones.
"""
function _plans!(cache::Union{Nothing,SearchCache}, params::SearchParams,
                 Nprof::Integer, nt::Integer)
    if cache !== nothing && cache.params === params && cache.Nprof == Nprof &&
       cache.workspaces !== nothing
        ws = cache.workspaces
        # A later file may need more tasks than an earlier one (more chunks);
        # top up rather than rebuild.  Extra workspaces are simply not indexed.
        while length(ws) < nt
            push!(ws, Workspace(params, Nprof))
        end
        return ws
    end
    # Planning is not thread-safe: build all workspaces serially, here.  The
    # vector is typed off the first element rather than built by comprehension:
    # `params.precision` is a runtime `Symbol`, so the `Workspace` constructor's
    # return type is a two-way union, and only a concretely-typed vector makes
    # `workspaces[t]` concrete inside `_search_region!` (the function barrier).
    w1 = Workspace(params, Nprof)
    ws = Vector{typeof(w1)}(undef, nt)
    ws[1] = w1
    for i in 2:nt
        ws[i] = Workspace(params, Nprof)
    end
    if cache !== nothing
        cache.params = params
        cache.Nprof = Int(Nprof)
        cache.workspaces = ws
    end
    return ws
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

Searching several files with the same `params`?  Pass one [`SearchCache`](@ref)
as `cache` to every call: the harmonic plans and per-task workspaces are then
built once for the whole run instead of once per file.  Results are unchanged.
"""
function search(ft::FFTFile, params::SearchParams=SearchParams();
                lofreq::Real=0.1, hifreq::Real=100.0, lobin::Integer=100,
                blocksize::Integer=2048, threshold::Real=params.threshold,
                remove::Bool=true, dr_tol::Real=1.0,
                harm_remove::Bool=true, numharm::Integer=16, harm_tol::Real=1.0,
                progress::Symbol=:none,
                metricstats::Union{Nothing,MetricStats}=nothing,
                normalize::Bool=false, verbose::Bool=false,
                wisdom::Bool=true, wisdom_file::Union{Nothing,AbstractString}=nothing,
                cache::Union{Nothing,SearchCache}=nothing)
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

    # Load saved FFTW wisdom so the (single-threaded, up-front) planning below
    # collapses to a lookup; persist it afterwards so the first run teaches every
    # subsequent one.  Off with `wisdom=false`.
    wpath = wisdom ? (wisdom_file === nothing ? wisdom_path() : String(wisdom_file)) : ""
    wisdom && import_wisdom!(wpath)

    # The direct interpolator's phase tables key off the *global* trial index, so
    # they are built once here against `r_lo` and shared read-only by every task.
    dplans = build_direct_plans(params, r_lo)
    nt = max(1, min(nthreads(), nchunks))
    # `PATIENT` is worth ~9% of the decimated transforms and costs ~1 s of
    # planning, which the wisdom file amortises to nothing.  With no wisdom file
    # there is nothing to amortise into, so fall back to `MEASURE` (see
    # `_PLAN_RIGOR`).  Only the *first* workspace pays: FFTW's wisdom is
    # process-global, so the other `nt-1` are lookups.
    workspaces = with_plan_rigor(wisdom ? FFTW.PATIENT : FFTW.MEASURE) do
        _plans!(cache, params, Nprof, nt)
    end

    wisdom && export_wisdom!(wpath)

    if verbose
        @printf(stderr, "Search: %d trials in %d chunks of %d over r = %.1f … %.1f bins (%.4f … %.4f Hz), %d thread(s)\n",
                total, nchunks, Nprof, r_lo, r_hi, r_lo / ft.T, r_hi / ft.T, nt)
        let nofb = count(dp -> dp.P == 0, dplans)
            if nofb > 0
                @printf(stderr, "  NOTE: %d harmonic(s) fell back to per-trial coefficients (hidr=%g is not a simple rational)\n",
                        nofb, params.hidr)
            else
                @printf(stderr, "  phase-cycle lengths P: min %d, max %d (of q = %d)\n",
                        minimum(dp.P for dp in dplans), maximum(dp.P for dp in dplans),
                        dplans[1].q)
            end
        end
        # The interpolation weight type is a deliberate speed/accuracy trade and
        # is disclosed rather than left implicit.  Quote it against the two errors
        # that actually dominate, so the number is read in proportion.
        let WT = eltype(dplans[1].W)
            @printf(stderr, "  interpolation: m=%d, %s weights (~%s relative); truncation loss ~%.2f%%, hidr grid ~6.5%%\n",
                    params.m, WT, WT === Float32 ? "2e-7" : "7e-11", 100 * 0.203 / params.m)
        end
        println(stderr, "─"^96)
    end

    norm = nothing
    if normalize
        # Pass 1/2: measure the per-(k, frequency window) noise, build the model.
        # Reuse the user's metricstats sink if given (same measurement); else an
        # internal one.  threshold=Inf skips candidate bookkeeping in this pass.
        normstats = metricstats === nothing ? MetricStats() : metricstats
        @info "Normalising: measuring per-(k,frequency) noise (pass 1/2)"
        _search_region!(ft, params, workspaces, nbins, r_lo, r_hi, lodr,
                        total, Nprof, nchunks, nt;
                        threshold=Inf, norm=nothing, metricstats=normstats, progress=progress,
                        dplans=dplans)
        norm = build_metricnorm(normstats)
        @info "Built in-situ normalisation model; searching (pass 2/2)" windows=normstats.nwin
    end

    # Detection pass.  When normalising, stats were collected in pass 1, so pass 2
    # does not re-collect (metricstats=nothing here).
    cands = _search_region!(ft, params, workspaces, nbins, r_lo, r_hi, lodr,
                            total, Nprof, nchunks, nt;
                            threshold=threshold, norm=norm,
                            metricstats=(normalize ? nothing : metricstats), progress=progress,
                            dplans=dplans)

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
