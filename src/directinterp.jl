# Direct O(m) Fourier interpolation.
#
# The FFT-correlation interpolator (`finterp_fft`, `interp_tile!`) evaluates the
# sinc/phase kernel on a *uniform fine grid* of `numbetween` points per Fourier
# bin and then linearly interpolates that grid at the trial frequencies.  It
# exists because the Python original could only go fast by vectorising, and an
# FFT is the only way to vectorise a correlation in numpy.  It costs two
# transforms of ~`numbetween*(numbins+m)` points to produce `Nprof` values, and
# the final linear interpolation is an *approximation* — at the production
# `numbetween=16` it is wrong by up to ~5% in amplitude at high harmonics.
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

"""
    build_direct_plans(params, r_lo) -> Vector{DirectPlan}
    build_direct_plans(WT, params, r_lo) -> Vector{DirectPlan{WT}}

Build the per-harmonic phase tables for a search whose *global* trial 0 sits at
fundamental Fourier frequency `r_lo`.  Cheap (a few thousand divisions per
harmonic) and done once, before the parallel region.
"""
# The interpolation working precision is deliberately **not** tied to
# `params.precision`.  A fully-`Float32` inner sum was measured at **1.64x
# slower** than `Float64` on the production configuration (Xeon Silver 4114,
# AVX-512, m=16, nharms=60, Nprof=2048): at `m = 16` the sum is exactly one
# 16-lane `Float32` vector, so there is a single FMA with no instruction-level
# parallelism followed by a *four*-stage cross-lane reduce, against `Float64`'s
# two independent 8-lane accumulators and a three-stage reduce.  The per-trial
# horizontal reduction — not the width of the data — is what this loop is paying
# for, and narrowing makes it worse.  `build_direct_plans(WT, …)` keeps the
# experiment one argument away (and a machine with cheap cross-lane reduces, or a
# GPU port, may well flip it), but the search always asks for `Float64`.
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
                                        CT[], Bool[]))
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
        push!(plans, DirectPlan{WT}(h, m, q, s, base_adv, pnum, rfloor0, dr0,
                                    P, row, W, A, carry))
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
workspace's real/imaginary plane buffers, which turns the inner loop into two
independent real dot products — the form the vectoriser handles best.

The sum accumulates in the plan's weight type `WT`.  At `WT = Float64` the
`Float32` planes widen exactly, so the result is bit-identical to `Float64`
planes.  At `WT = Float32` the whole `m`-sum is single precision: twice the SIMD
lanes and half the weight-table traffic, for a relative error of ~1e-6 against
the ~1e-10 the `Float64` sum achieves.  That is far below the ~1.3% signal-power
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

    res, qint = direct_chunk_state(dp, t0)
    # Julia bin range of trial t is  binstart : binstart+m-1  with
    # binstart = floor(h*r_t) + 2 - m2   (see `nearby_fourier_bin_range`).
    p0 = dp.row[res + 1]
    lo = dp.rfloor0 + qint + dp.carry[p0] + 2 - m2
    # State after the final trial, for the range check and the copy window.
    resl, qintl = direct_chunk_state(dp, t0 + n - 1)
    pl = dp.row[resl + 1]
    hi = dp.rfloor0 + qintl + dp.carry[pl] + 1 + m2
    (lo >= 1 && hi <= namps && (dp.rfloor0 + qintl) < Nhalf) || return

    # De-interleave the chunk's bin window into real/imaginary planes.
    nw = hi - lo + 1
    re = ws.re
    im = ws.im
    # A short window buffer would silently drop a whole harmonic, so make it loud:
    # it can only mean the workspace was built for a smaller chunk than `n`.
    length(re) >= nw || throw(ArgumentError(
        "workspace plane buffers hold $(length(re)) bins but harmonic $(h) spans $nw " *
        "for a chunk of $n — the Workspace was built for a smaller Nprof"))
    amps = ft.amps
    @inbounds @simd for i in 1:nw
        a = amps[lo + i - 1]
        re[i] = real(a)
        im[i] = imag(a)
    end

    W = dp.W
    A = dp.A
    row = dp.row
    carry = dp.carry
    q = dp.q
    s = dp.s
    base_adv = dp.base_adv
    ftprofs = ws.ftprofs
    hrow = h + 1
    @inbounds for k in 1:n
        p = row[res + 1]
        # Offset of this trial's first bin within the de-interleaved window.
        b = dp.rfloor0 + qint + carry[p] + 2 - m2 - lo
        sre = zero(WT)
        sim = zero(WT)
        # `Float32 -> WT` is exact for both `WT`, so at `WT = Float64` this
        # accumulates identically to `Float64` planes while reading half the bytes,
        # and at `WT = Float32` it is a no-op conversion.
        @simd for i in 1:m
            w = W[i, p]
            sre = muladd(w, WT(re[b + i]), sre)
            sim = muladd(w, WT(im[b + i]), sim)
        end
        ftprofs[hrow, k] = A[p] * complex(sre, sim)
        res += s
        qint += base_adv
        if res >= q
            res -= q
            qint += 1
        end
    end
    return
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
`m` bins, plus slack for the two end roundings.
"""
direct_window_size(params::SearchParams, Nprof::Integer) =
    ceil(Int, (Nprof - 1) * params.hidr) + params.m + 4
