# Direct O(m) Fourier interpolation.
#
# The FFT-correlation interpolator (`finterp_fft`, still in `fourierinterp.jl` as
# the reference kernel and as what the Python oracle is pinned to) evaluates the
# sinc/phase kernel on a *uniform fine grid* of `numbetween` points per Fourier
# bin and then linearly interpolates that grid at the trial frequencies.  It
# exists because the Python original could only go fast by vectorising, and an
# FFT is the only way to vectorise a correlation in numpy.  It costs two
# transforms of ~`numbetween*(numbins+m)` points to produce `Nprof` values, and
# the final linear interpolation is an *approximation* — at `numbetween=16` it is
# wrong by up to ~5% in amplitude at high harmonics.  It was the search's
# interpolator until 2026-08-08 and was deleted from the search on 2026-08-16.
#
# This module evaluates Eqn. 30 directly at exactly the frequencies wanted.  Two
# observations make that cheaper than the FFT here, not more expensive:
#
# 1. **The kernel's coefficients are a real vector times one complex scalar.**
#    With `offsets = dr - j` and `j` an integer,
#
#        sin(pi (dr - j)) = (-1)^j sin(pi dr)      cispi(dr - j) = (-1)^j cispi(dr)
#
#    so the `(-1)^j` cancels between `sinc` and the phase and
#
#        coeff_j = sinc(dr-j) cispi(dr-j) = A(dr) / (dr - j),  A = sin(pi dr) cispi(dr) / pi
#
#    Since `fourier_interp` forms `dot(coeffs, bins) = sum conj(coeff_j) bin_j`
#    and `1/(dr-j)` is *real*, one interpolated point is
#
#        amp(r) = conj(A(dr)) * sum_j  bin_j / (dr - j)
#
#    i.e. `m` real-times-complex multiply-adds (4 flops each), not `m` complex
#    ones (8 flops).  It is also perfectly SIMD-friendly and reads only `m`
#    consecutive bins, so it runs out of L1 rather than streaming 16-32k-point
#    arrays the way the transforms do.
#
# 2. **Only `q = 2*nharms` distinct `dr` values occur in an entire search.**
#    Trial `t` (a *global*, 0-based trial index) sits at `r_t = r_lo + t*lodr`
#    with `lodr = hidr/nharms`, so harmonic `h` is evaluated at
#    `h*r_t = h*r_lo + t*h*pnum/q` for `lodr = pnum/q`.  The fractional part
#    therefore takes at most `q` values — `frac(h*r_lo) + i/q` — for *every*
#    chunk and every trial.  So the `m` reciprocals `1/(dr-j)` and the scalar
#    `A(dr)` are tabulated once per harmonic at plan time and indexed by an
#    integer residue that advances by a fixed step per trial.  The inner loop
#    contains no transcendentals and no divisions.
#
# Both the integer bin index and the phase index come out of exact integer
# arithmetic on that residue, so there is no drift over a long search and no
# ambiguity when a trial lands on (or a hair below) an exact Fourier bin — a
# corner the float-accumulated `floor(r + 1e-15)` handles less cleanly.
#
# The result matches `fourier_interp` — the oracle-pinned exact kernel — to
# ~1e-15 where `dr` is exactly representable and ~1e-10 otherwise, versus the
# ~1e-2 the FFT path's linear interpolation carries.

"""
    finterp_direct(r0, n, step, ft, m) -> Vector{ComplexF64}
    finterp_direct!(out, r0, n, step, ft, m)

Exact Fourier interpolation at the `n` frequencies `r0 .+ (0:n-1).*step`, the
direct counterpart of [`finterp_fft`](@ref).

This is the general, self-contained form: it computes the `m` real weights per
point, so it needs no phase table and places no constraint on `step`.  It is
what [`fourier_interp`](@ref) does at each point, but with the coefficient
vector factored into `A(dr)` times `1/(dr-j)` (see the module comment), which is
both cheaper and free of transcendentals in the inner sum.

The search's hot loop uses [`fill_harmonic_row_direct!`](@ref) instead, which
additionally tabulates the weights across the search's finitely many `dr` values.
"""
function finterp_direct(r0::Real, n::Integer, step::Real, ft::AbstractVector, m::Integer)
    out = Vector{ComplexF64}(undef, n)
    return finterp_direct!(out, r0, n, step, ft, m)
end

function finterp_direct!(out::AbstractVector{ComplexF64}, r0::Real, n::Integer,
                         step::Real, ft::AbstractVector, m::Integer)
    iseven(m) || throw(ArgumentError("m must be even"))
    length(out) >= n || throw(ArgumentError("out must hold at least n points"))
    m2 = m ÷ 2
    @inbounds for k in 1:n
        r = r0 + (k - 1) * step
        rint = floor(Int, r + 1e-15) + 1        # 0-based bin index, as in Python
        dr = mod(r, 1.0)
        base = rint - m2
        if dr == 0.0
            out[k] = ComplexF64(ft[base + m2])   # kernel collapses to a delta
            continue
        end
        sre = 0.0
        sim = 0.0
        @simd for i in 1:m
            w = 1.0 / (dr - (i - m2))
            a = ft[base + i]
            sre = muladd(w, Float64(real(a)), sre)
            sim = muladd(w, Float64(imag(a)), sim)
        end
        out[k] = conj(sinpi(dr) * cispi(dr) / pi) * complex(sre, sim)
    end
    return out
end

"""
    DIRECT_QMAX

Largest trial-grid denominator `q` for which the phase table is built.  `q` is
`2*nharms` for the default `hidr = 1/2`; a `hidr` that is not a simple rational
produces a huge `q`, in which case [`fill_harmonic_row_direct!`](@ref) falls back
to computing the `m` reciprocals per trial (same result, ~2x slower).
"""
const DIRECT_QMAX = 1 << 16

"""
    DIRECT_GROUP_V

Trials per group in the trials-axis kernel (see [`fill_harmonic_row_direct!`](@ref)).
A compile-time constant, not a plan field, because the kernel's accumulators are an
`NTuple{V}` — with a runtime `V` the `ntuple`s do not unroll and the whole win is
lost (the same trap `_BC_BATCH` documents for the boxcar gate).

16 measured fastest over a sweep of `V = 8/16/32/64`, which gave 121/99/107/157 µs
for the sampled harmonics: below 16 there are too few lanes to hide the FMA
latency, above it the extended window `m+Δ` grows (Δ ≈ V·h/q) and both the wasted
FMAs and the table footprint (0.65/1.5/3.5/9.3 MB over 60 harmonics) run away.
"""
const DIRECT_GROUP_V = 16

"""
    DirectPlan

Per-harmonic recipe for the direct interpolator, built once at plan time and
shared read-only across all threads.

The trial grid is the exact rational `r_t = r_lo + t*pnum/q`, so harmonic `h`
advances by `h*pnum/q` bins per trial: an integer part `base_adv` plus a
residue step `s`.  Carrying `(residue, integer)` alongside each other makes both
the fractional phase and the Fourier bin index exact.

`row[i+1]` maps residue `i` to its row in `W`/`A`/`carry`, or `0` for residues
the cycle never visits (only `P = q ÷ gcd(h*pnum, q)` of the `q` do).  `W[:, p]`
holds the `m` real weights `1/(dr - j)`, `A[p]` the complex scalar
`conj(sin(pi dr) cispi(dr) / pi)`, and `carry[p]` whether `frac(h*r_lo) + i/q`
wrapped past 1 (which bumps the integer bin index by one).

`P == 0` marks the no-table fallback described in [`DIRECT_QMAX`](@ref).

The table's element type `WT` is the interpolation working precision, taken from
`SearchParams.precision` (see [`proftype`](@ref)).  Every entry is *computed* in
`Float64` and rounded once on storage, so `WT = Float32` costs one rounding of a
tabulated constant — not an accumulated error — while halving the bytes the inner
sum streams and doubling its SIMD width.  See [`fill_harmonic_row_direct!`](@ref)
for what that is worth and what it costs in accuracy.
"""
struct DirectPlan{WT<:AbstractFloat}
    h::Int
    m::Int
    q::Int                    # trial-grid denominator
    s::Int                    # per-trial residue advance, (h*pnum) mod q
    base_adv::Int             # per-trial integer-bin advance, (h*pnum) ÷ q
    pnum::Int                 # trial-grid numerator
    rfloor0::Int              # floor(h * r_lo)
    dr0::Float64              # h*r_lo - floor(h*r_lo)
    P::Int                    # phase-cycle length (0 ⇒ per-trial fallback)
    row::Vector{Int32}        # (q,) residue -> row index, 0 if unvisited
    W::Matrix{WT}             # (m, P) real weights 1/(dr - j)
    A::Vector{Complex{WT}}    # (P,) conj(A(dr))
    carry::Vector{Bool}       # (P,)
    # --- trials-axis group tables (see `fill_harmonic_row_direct!`) ----------
    # A group is `DIRECT_GROUP_V` consecutive trials anchored to the *global*
    # trial index.  Its weight block `gW` is the `(V, m+Δ)` matrix `Wx` with
    # `Wx[k, j] = W[j - δₖ, pₖ]` (zero outside), `δₖ = bₖ - b₀`; it depends only
    # on the residue the group starts at, of which there are `ngrp = q ÷
    # gcd(V·s, q)`.  Empty when `P == 0` (no-table fallback).
    grow::Vector{Int32}       # (q,) residue -> group index, 0 if not a group start
    goff::Vector{Int32}       # (ngrp+1,) offsets into gW
    gnj::Vector{Int32}        # (ngrp,) m+Δ, the extended window length
    gW::Vector{WT}            # flat, group g is (V, gnj[g]) column-major at goff[g]
    gA::Matrix{Complex{WT}}   # (V, ngrp) per-trial conj(A(dr))
end

"""
    trial_grid_rational(lodr) -> (pnum, q) or nothing

Express the trial step `lodr = hidr/nharms` as an exact small rational `pnum/q`.
Returns `nothing` when no such rational reproduces `lodr` to within a rounding
error, which is the signal to use the per-trial fallback.

The default `hidr = 1/2` gives `lodr = 1/(2*nharms)` exactly, so this is the
ordinary case; snapping the grid to the rational shifts trial frequencies by
<1 ulp, far below any physical scale, and buys exact phase bookkeeping.
"""
function trial_grid_rational(lodr::Real)
    lodr > 0 || return nothing
    rat = rationalize(Int, float(lodr); tol = 1e-13)
    q = denominator(rat)
    p = numerator(rat)
    (q <= DIRECT_QMAX && p >= 1) || return nothing
    abs(p / q - lodr) <= 8 * eps(float(lodr)) || return nothing
    return (p, q)
end

# Tabulate the trials-axis group blocks for one harmonic.  Group `g` starts at
# global trial `(g-1)*V`, hence at residue `(g-1)*V*s mod q`; there are
# `ngrp = q ÷ gcd(V*s, q)` distinct such residues, and `grow` maps residue → group.
#
# Within a group the `k`-th trial reads the `m` bins starting at `bₖ`, and
# `δₖ = bₖ - b₀ ∈ [0, Δ]` because `b` is non-decreasing with 0/1 steps (plus
# `base_adv`).  Writing every trial's window as an offset into the *group's*
# window turns the group into one `(V, m+Δ)` matvec against a single contiguous
# slice of the bin planes — which is the whole point: no gather, and no per-trial
# horizontal reduce.
function build_group_table(::Type{WT}, m::Int, q::Int, s::Int, base_adv::Int,
                           row::Vector{Int32}, W::Matrix{WT},
                           A::Vector{Complex{WT}}, carry::Vector{Bool}) where {WT}
    V = DIRECT_GROUP_V
    CT = Complex{WT}
    gstep = mod(V * s, q)
    ngrp = gstep == 0 ? 1 : q ÷ gcd(gstep, q)
    grow = zeros(Int32, q)
    gnj = Vector{Int32}(undef, ngrp)
    goff = Vector{Int32}(undef, ngrp + 1)
    gA = Matrix{CT}(undef, V, ngrp)
    blocks = Vector{Vector{WT}}(undef, ngrp)
    res0 = 0
    total = 0
    ps = Vector{Int}(undef, V)
    δs = Vector{Int}(undef, V)
    for g in 1:ngrp
        grow[res0 + 1] = Int32(g)
        res = res0; B = 0; B0 = 0
        for k in 1:V
            p = Int(row[res + 1])
            Bk = B + (carry[p] ? 1 : 0)
            k == 1 && (B0 = Bk)
            ps[k] = p; δs[k] = Bk - B0
            gA[k, g] = A[p]
            res += s; B += base_adv
            if res >= q; res -= q; B += 1; end
        end
        nj = m + δs[V]
        gnj[g] = Int32(nj)
        blk = zeros(WT, V * nj)
        for k in 1:V, i in 1:m
            blk[(δs[k] + i - 1) * V + k] = W[i, ps[k]]
        end
        blocks[g] = blk
        goff[g] = Int32(total)
        total += V * nj
        res0 = mod(res0 + gstep, q)
    end
    goff[ngrp + 1] = Int32(total)
    gW = Vector{WT}(undef, total)
    for g in 1:ngrp
        copyto!(gW, Int(goff[g]) + 1, blocks[g], 1, length(blocks[g]))
    end
    return grow, goff, gnj, gW, gA
end

"""
    build_direct_plans(params, r_lo) -> Vector{DirectPlan}
    build_direct_plans(WT, params, r_lo) -> Vector{DirectPlan{WT}}

Build the per-harmonic phase tables for a search whose *global* trial 0 sits at
fundamental Fourier frequency `r_lo`.  Cheap (a few thousand divisions per
harmonic) and done once, before the parallel region.
"""
# The interpolation working precision is deliberately **not** tied to
# `params.precision`, and the search asks for `Float64`.
#
# **The standing reason for that is now obsolete and the question is reopened.**
# A fully-`Float32` inner sum was measured on 2026-08-16 at **1.64x slower** than
# `Float64` (Xeon Silver 4114, AVX-512, m=16, nharms=60, Nprof=2048), and the
# diagnosis was the *per-trial horizontal reduce*: at `m = 16` a `Float32` sum is
# exactly one 16-lane vector, so one FMA with no ILP followed by a four-stage
# cross-lane reduce, against `Float64`'s two 8-lane accumulators and a three-stage
# reduce.  The trials-axis kernel (2026-08-22) has **no cross-lane reduce at all**
# — each vector lane is a different trial — so that mechanism cannot apply, and
# `Float32` would now both double the lanes and halve a weight table that grew
# from 395 KB to ~1.45 MB.  `build_direct_plans(WT, …)` still takes the type;
# re-measure before quoting the 1.64x at anyone.
build_direct_plans(params::SearchParams, r_lo::Real) =
    build_direct_plans(Float64, params, r_lo)

function build_direct_plans(::Type{WT}, params::SearchParams, r_lo::Real) where {WT<:AbstractFloat}
    lodr = params.hidr / params.nharms
    rq = trial_grid_rational(lodr)
    m = params.m
    iseven(m) || throw(ArgumentError("m must be even"))
    m2 = m ÷ 2
    CT = Complex{WT}
    plans = DirectPlan{WT}[]
    for h in 1:params.nharms
        hr = h * float(r_lo)
        rfloor0 = floor(Int, hr)
        dr0 = hr - rfloor0
        if rq === nothing
            push!(plans, DirectPlan{WT}(h, m, 1, 0, 0, 1, rfloor0, dr0, 0,
                                        Int32[], Matrix{WT}(undef, m, 0),
                                        CT[], Bool[],
                                        Int32[], Int32[], Int32[], WT[],
                                        Matrix{CT}(undef, DIRECT_GROUP_V, 0)))
            continue
        end
        pnum, q = rq
        step = h * pnum
        s = mod(step, q)
        base_adv = fld(step, q)
        P = q ÷ gcd(step, q)
        row = zeros(Int32, q)
        W = Matrix{WT}(undef, m, P)
        A = Vector{CT}(undef, P)
        carry = Vector{Bool}(undef, P)
        res = 0
        # Every entry is computed in Float64 and rounded once into `WT`: the table
        # is small (m×P) and built once per file, so there is nothing to gain from
        # evaluating the transcendentals at reduced precision, and rounding the
        # exact value is the most accurate `WT` table available.
        for p in 1:P
            row[res + 1] = Int32(p)
            u = dr0 + res / q
            carry[p] = u >= 1.0
            dr = carry[p] ? u - 1.0 : u
            if dr == 0.0
                # r is exactly a Fourier bin: the kernel collapses to a delta on
                # that bin (A -> 0 while 1/(dr-j) -> Inf at j=0; the product is 1).
                A[p] = one(CT)
                @inbounds for i in 1:m
                    W[i, p] = (i == m2) ? one(WT) : zero(WT)
                end
            else
                A[p] = CT(conj(sinpi(dr) * cispi(dr) / pi))
                @inbounds for i in 1:m
                    W[i, p] = WT(1.0 / (dr - (i - m2)))
                end
            end
            res += s
            res >= q && (res -= q)
        end
        grow, goff, gnj, gW, gA = build_group_table(WT, m, q, s, base_adv, row, W, A, carry)
        push!(plans, DirectPlan{WT}(h, m, q, s, base_adv, pnum, rfloor0, dr0,
                                    P, row, W, A, carry, grow, goff, gnj, gW, gA))
    end
    return plans
end

"""
    direct_chunk_state(dp, t0) -> (res, qint)

Residue and accumulated integer-bin count of harmonic `dp.h` at *global* trial
`t0`, i.e. the state the per-trial recurrence starts a chunk from.  `floor(h*r_t)
= rfloor0 + qint + carry`, so `t0` enters only through exact integer arithmetic.
"""
@inline function direct_chunk_state(dp::DirectPlan, t0::Integer)
    T = widemul(Int(t0), dp.h * dp.pnum)      # can exceed Int64 for very long searches
    q = dp.q
    return (Int(mod(T, q)), Int(fld(T, q)))
end

"""
    fill_harmonic_row_direct!(ws, dp, ft, params, t0, n)

Fill row `dp.h+1`, columns `1:n`, of `ws.ftprofs` with the exactly-interpolated
complex amplitude of harmonic `dp.h` at the `n` trial fundamentals starting at
*global* trial index `t0`.  Leaves the row at zero if the harmonic runs off the
end of the available amplitudes or past Nyquist (matching
[`fill_harmonic_row!`](@ref)).

The `m` bins each trial reads are de-interleaved once per chunk into the
workspace's real/imaginary plane buffers, which turns the inner sum into two
independent real dot products.

**The loop vectorises across *trials*, not across `m`.**  Summing over `m` per
trial ends in a horizontal reduce, and with `m = 16` that reduce — not the
arithmetic — was what the kernel spent its time on.  Instead, `V =
DIRECT_GROUP_V` consecutive trials are handled together.  Substituting
`j = δₖ + i` (with `δₖ = bₖ - b₀`, the `k`-th trial's bin offset within the
group's window) rewrites the group as a matrix-vector product

    sreₖ = Σⱼ Wx[k, j] · re[b₀ + j],     Wx[k, j] = W[j - δₖ, pₖ]

against one contiguous `m+Δ` slice of the planes.  `Wx` depends only on the
residue the group *starts* at, so it is tabulated at plan time ([`DirectPlan`](@ref)).
The payoff is that `re[b₀+j]` is a broadcast scalar and `Wx[:, j]` a contiguous
column, so there is **no gather and no horizontal reduce** — each lane of the
accumulator is a different trial and stays live across the whole group.

The cost is `(m+Δ)/m` wasted FMAs, since `Wx` is zero outside each row's `m`
nonzeros.  `Δ ≈ V·h/q`, so it grows with harmonic and with `V`; at the defaults
that is 1.06x at `h = 1` and 1.5x at `h = 59`, and the kernel still wins at
*every* harmonic — 1.7x at the worst of them.  Measured in situ: **36% off the
interpolation phase at both `-t 1` and `-t 4`.**

Note that this obsoletes the reason a `Float32` weight table was rejected (see
[`build_direct_plans`](@ref)): that verdict rested on the cross-lane reduce,
which no longer exists here.

The sum accumulates in the plan's weight type `WT`.  At `WT = Float64` the
`Float32` planes widen exactly, so the result is bit-identical to `Float64`
planes.  At `WT = Float32` the whole sum is single precision: twice the SIMD
lanes and half the weight-table traffic — and the table is now `(V, m+Δ)` per
group rather than `(m, P)`, i.e. ~1.45 MB over 60 harmonics against the old
395 KB, so halving it is worth more than it used to be.  The accuracy cost is a
relative error of ~1e-6 against the ~1e-10 the `Float64` sum achieves.  That is far below the ~1.3% signal-power
loss the `m = 16` kernel already accepts and the ~6.5% the `hidr` grid costs at
the top harmonic, so it is invisible in a detection — but it is *not* invisible
to the equivalence pins, which is why `Float64` remains the default and the pins
run there.
"""
function fill_harmonic_row_direct!(ws::Workspace, dp::DirectPlan{WT}, ft::FFTFile,
                                   params::SearchParams, t0::Integer, n::Integer) where {WT}
    n >= 1 || return
    h = dp.h
    m = dp.m
    m2 = m ÷ 2
    Nhalf = ft.N ÷ 2
    namps = length(ft.amps)

    if dp.P == 0
        return _fill_row_direct_slow!(ws, dp, ft, params, t0, n)
    end

    # --- range guard, on the TRUE trial range -------------------------------
    # Julia bin range of trial t is  binstart : binstart+m-1  with
    # binstart = floor(h*r_t) + 2 - m2   (see `nearby_fourier_bin_range`).
    # This is deliberately *not* widened to the group range below: the point at
    # which a harmonic gives up (off the end of the amplitudes, or past Nyquist)
    # must stay a property of the trials actually being searched, not of how the
    # group grid happens to straddle them.
    res, qint = direct_chunk_state(dp, t0)
    p0 = dp.row[res + 1]
    lo_trial = dp.rfloor0 + qint + dp.carry[p0] + 2 - m2
    resl, qintl = direct_chunk_state(dp, t0 + n - 1)
    pl = dp.row[resl + 1]
    hi_trial = dp.rfloor0 + qintl + dp.carry[pl] + 1 + m2
    (lo_trial >= 1 && hi_trial <= namps && (dp.rfloor0 + qintl) < Nhalf) || return

    # --- group grid, anchored to the GLOBAL trial index ----------------------
    # Groups of `V` trials start at global trials 0, V, 2V, …, so which group a
    # trial belongs to — and hence the exact arithmetic it gets — does not depend
    # on where chunk boundaries fall.  A chunk's first and last groups may hang
    # off either end; they are computed in full and masked on store, so the plane
    # buffers are widened to the *group* range and zero-filled outside the file.
    V = DIRECT_GROUP_V
    T_first = fld(t0, V) * V
    T_last = fld(t0 + n - 1, V) * V + V - 1
    res_e, qint_e = direct_chunk_state(dp, T_first)
    lo = dp.rfloor0 + qint_e + dp.carry[dp.row[res_e + 1]] + 2 - m2
    resx, qintx = direct_chunk_state(dp, T_last)
    hi = dp.rfloor0 + qintx + dp.carry[dp.row[resx + 1]] + 1 + m2

    # De-interleave that window into real/imaginary planes.
    nw = hi - lo + 1
    re = ws.re
    im = ws.im
    # A short window buffer would silently drop a whole harmonic, so make it loud:
    # it can only mean the workspace was built for a smaller chunk than `n`.
    length(re) >= nw || throw(ArgumentError(
        "workspace plane buffers hold $(length(re)) bins but harmonic $(h) spans $nw " *
        "for a chunk of $n — the Workspace was built for a smaller Nprof"))
    amps = ft.amps
    # Split rather than branch per element, so the copy still vectorises: the
    # head/tail are the (at most V-trial) overhang of the end groups past the
    # file, and only ever run at the very ends of the band.
    ihead = clamp(1 - lo, 0, nw)                    # entries with lo+i-1 < 1
    itail = clamp(hi - namps, 0, nw - ihead)        # entries with lo+i-1 > namps
    @inbounds @simd for i in 1:ihead
        re[i] = 0.0f0
        im[i] = 0.0f0
    end
    @inbounds @simd for i in (ihead + 1):(nw - itail)
        a = amps[lo + i - 1]
        re[i] = real(a)
        im[i] = imag(a)
    end
    @inbounds @simd for i in (nw - itail + 1):nw
        re[i] = 0.0f0
        im[i] = 0.0f0
    end

    gW = dp.gW; gA = dp.gA; grow = dp.grow; goff = dp.goff; gnj = dp.gnj
    row = dp.row
    carry = dp.carry
    q = dp.q
    s = dp.s
    base_adv = dp.base_adv
    ftprofs = ws.ftprofs
    hrow = h + 1
    res = res_e
    qint = qint_e
    T = T_first
    vv = Val(V)
    @inbounds while T <= T_last
        g = Int(grow[res + 1])
        b0 = dp.rfloor0 + qint + carry[row[res + 1]] + 2 - m2 - lo
        o = Int(goff[g])
        nj = Int(gnj[g])
        j0 = T - t0                            # local column of lane 1, minus one
        if j0 >= 0 && j0 + V <= n
            _group_store!(ftprofs, hrow, j0, gW, o, nj, gA, g, re, im, b0, vv)
        else
            _group_store_masked!(ftprofs, hrow, j0, n, gW, o, nj, gA, g, re, im, b0, vv)
        end
        # advance V trials at once (exactly what V applications of the per-trial
        # recurrence do: every wrap past q carries one bin).
        tot = res + V * s
        qint += V * base_adv + fld(tot, q)
        res = mod(tot, q)
        T += V
    end
    return
end

# One group: the (V, nj) weight block times the nj-long slice of the bin planes.
# The accumulators are an `NTuple{V}` so they stay in vector registers — a scratch
# `Vector` would put store-to-load forwarding in the FMA chain — and the two
# `ntuple` helpers are top-level so their closures capture only their arguments.
# Capturing an assigned-in-loop local instead boxes them: measured 2000x slower.
@inline _grp_w(gW::Vector{T}, o::Int, ::Val{V}) where {T,V} =
    ntuple(k -> @inbounds(gW[o + k]), Val(V))
@inline _grp_fma(a::NTuple{V,T}, w::NTuple{V,T}, x::T) where {V,T} =
    ntuple(k -> muladd(w[k], x, a[k]), Val(V))

@inline function _group_lanes(gW::Vector{WT}, o::Int, nj::Int, re, im, b0::Int,
                              ::Val{V}) where {WT,V}
    ar = ntuple(_ -> zero(WT), Val(V))
    ai = ntuple(_ -> zero(WT), Val(V))
    @inbounds for j in 1:nj
        rv = WT(re[b0 + j])
        iv = WT(im[b0 + j])
        w = _grp_w(gW, o + (j - 1) * V, Val(V))
        ar = _grp_fma(ar, w, rv)
        ai = _grp_fma(ai, w, iv)
    end
    return ar, ai
end

@inline function _group_store!(ftprofs, hrow::Int, j0::Int, gW::Vector{WT}, o::Int,
                               nj::Int, gA, g::Int, re, im, b0::Int,
                               ::Val{V}) where {WT,V}
    ar, ai = _group_lanes(gW, o, nj, re, im, b0, Val(V))
    @inbounds for k in 1:V
        ftprofs[hrow, j0 + k] = gA[k, g] * complex(ar[k], ai[k])
    end
end

# Partial end group: computed in full, stored only where it lands inside the
# chunk.  The out-of-chunk lanes read the zero-padded ends of the planes, so they
# are harmless — and computing them anyway is exactly what makes a trial's
# arithmetic independent of where the chunk boundaries fall.
@inline function _group_store_masked!(ftprofs, hrow::Int, j0::Int, n::Int,
                                      gW::Vector{WT}, o::Int, nj::Int, gA, g::Int,
                                      re, im, b0::Int, ::Val{V}) where {WT,V}
    ar, ai = _group_lanes(gW, o, nj, re, im, b0, Val(V))
    @inbounds for k in 1:V
        j = j0 + k
        (1 <= j <= n) && (ftprofs[hrow, j] = gA[k, g] * complex(ar[k], ai[k]))
    end
end

# Fallback for a trial step that is not a small rational (see `DIRECT_QMAX`):
# same exact kernel, but `dr` and the `m` reciprocals are computed per trial.
function _fill_row_direct_slow!(ws::Workspace, dp::DirectPlan, ft::FFTFile,
                                params::SearchParams, t0::Integer, n::Integer)
    h = dp.h
    m = dp.m
    m2 = m ÷ 2
    lodr = params.hidr / params.nharms
    hr_lo = dp.rfloor0 + dp.dr0         # h * r_lo, reconstructed from the plan
    dh = h * lodr                        # this harmonic's step per trial
    Nhalf = ft.N ÷ 2
    namps = length(ft.amps)
    rfirst = hr_lo + t0 * dh
    rlast = hr_lo + (t0 + n - 1) * dh
    lo = floor(Int, rfirst + 1e-15) + 2 - m2
    hi = floor(Int, rlast + 1e-15) + 1 + m2
    (lo >= 1 && hi <= namps && floor(Int, rlast) < Nhalf) || return
    amps = ft.amps
    ftprofs = ws.ftprofs
    hrow = h + 1
    @inbounds for k in 1:n
        r = hr_lo + (t0 + k - 1) * dh
        rint = floor(Int, r + 1e-15) + 1
        dr = mod(r, 1.0)
        base = rint - m2
        if dr == 0.0
            ftprofs[hrow, k] = ComplexF64(amps[base + m2])
            continue
        end
        sre = 0.0
        sim = 0.0
        @simd for i in 1:m
            w = 1.0 / (dr - (i - m2))
            a = amps[base + i]
            sre = muladd(w, Float64(real(a)), sre)
            sim = muladd(w, Float64(imag(a)), sim)
        end
        ftprofs[hrow, k] = conj(sinpi(dr) * cispi(dr) / pi) * complex(sre, sim)
    end
    return
end

"""
    direct_window_size(params, Nprof) -> Int

Widest bin window any harmonic's chunk can span, i.e. how long the workspace's
de-interleaved real/imaginary plane buffers must be.  The top harmonic steps by
`hidr` bins per trial, so the span is `(Nprof-1)*hidr` bins plus the kernel's
`m` bins, plus slack for the two end roundings — and plus up to `DIRECT_GROUP_V`
trials of overhang at *each* end, because the trials-axis kernel computes the two
partial end groups in full (see [`fill_harmonic_row_direct!`](@ref)).
"""
direct_window_size(params::SearchParams, Nprof::Integer) =
    ceil(Int, (Nprof - 1 + 2 * DIRECT_GROUP_V) * params.hidr) + params.m + 4
