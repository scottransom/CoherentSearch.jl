"""
    CoherentSearchCUDAExt

CUDA backend for CoherentSearch.jl.  Loaded automatically when `CUDA` is present
alongside `CoherentSearch`; registers itself in `CoherentSearch._GPU_BACKEND`
only if `CUDA.functional()`.

**This module may only add methods on types it owns.**  Redefining a method the
base module already defined is method overwriting, which Julia rejects during
precompilation.  Hence `CUDABackend` and dispatch on it.  See `gpu_design.md` §3.4.
"""
module CoherentSearchCUDAExt

using CoherentSearch, CUDA
using CoherentSearch: SearchBackend, SearchParams, FFTFile, DirectPlan, Candidate,
                      build_direct_plans, DIRECT_GROUP_V, ladder_boxcar_widths,
                      _analytic_sigma, _render_progress
using LinearAlgebra: mul!
using CUDA.CUFFT: plan_brfft
# `import`, not `using`: extending a function from another module requires the
# name be brought in for extension, not merely for reference.
import CoherentSearch: chunk_ftprofs, chunk_profiles, chunk_boxcar, _region!

"""
    CUDABackend

The CUDA compute backend.  Registered as `CoherentSearch.gpu_backend()` at load
time when a functional device is present.
"""
struct CUDABackend <: SearchBackend end

__init__() = CUDA.functional() && (CoherentSearch._GPU_BACKEND[] = CUDABackend())

# ---------------------------------------------------------------------------
# Device-side interpolation plan
#
# The CPU's `DirectPlan` is one object per harmonic.  Sixty small device arrays
# per field would mean sixty kernel launches per chunk; at the large `Nprof` the
# GPU wants (`gpu_design.md` §3.2) launch overhead is already the most likely way
# this design fails, so the per-harmonic tables are **concatenated once** into
# flat device vectors with per-harmonic base offsets, and one launch covers the
# whole (harmonic × trial-group) grid.
#
# Everything here is built once per `(params, r_lo)` and is read-only on the
# device, exactly as the CPU tables are read-only across threads.  At the
# defaults that is ~1.45 MB of `gW` over 60 harmonics.
# ---------------------------------------------------------------------------
struct GPUInterpPlan{WT}
    nharms::Int
    m2::Int                     # m ÷ 2
    q::Int                      # trial-grid denominator
    pnum::Int                   # trial-grid numerator
    # per-harmonic bases into the flattened tables (all length nharms)
    wbase::CuVector{Int32}      # into gW
    abase::CuVector{Int32}      # into gA
    gbase::CuVector{Int32}      # into goff/gnj/gcarry
    rfloor0::CuVector{Int64}    # floor(h * r_lo)
    sadv::CuVector{Int32}       # V*s per harmonic: residue advance per GROUP
    badv::CuVector{Int32}       # V*base_adv per harmonic: integer-bin advance per GROUP
    # flattened tables
    grow::CuVector{Int32}       # (q, nharms): residue -> group index (0 = not a start)
    goff::CuVector{Int32}       # per group, 0-based offset into that harmonic's gW
    gnj::CuVector{Int32}        # per group, the extended window length m+Δ
    gcarry::CuVector{Int32}     # per group, `carry` at the group's START residue
    gW::CuVector{WT}
    gA::CuVector{Complex{WT}}
    plans::Vector{DirectPlan{WT}}   # host copies, for the per-chunk range guard
end

function GPUInterpPlan(plans::Vector{DirectPlan{WT}}) where {WT}
    nharms = length(plans)
    nharms >= 1 || throw(ArgumentError("no harmonic plans"))
    any(p -> p.P == 0, plans) && throw(ArgumentError(
        "the GPU interpolator needs the tabulated path, but this trial step is " *
        "not a small rational (DirectPlan.P == 0).  Use --hidr 0.5 (the default) " *
        "or run on the CPU."))
    q = plans[1].q
    all(p -> p.q == q, plans) || throw(ArgumentError("harmonics disagree on q"))

    wbase = Int32[]; abase = Int32[]; gbase = Int32[]
    growf = Int32[]; gofff = Int32[]; gnjf = Int32[]; gcarryf = Int32[]
    gWf = WT[]; gAf = Complex{WT}[]
    V = DIRECT_GROUP_V
    for dp in plans
        push!(wbase, Int32(length(gWf)))
        push!(abase, Int32(length(gAf)))
        push!(gbase, Int32(length(gofff)))
        append!(growf, dp.grow)
        ngrp = length(dp.gnj)
        append!(gofff, @view dp.goff[1:ngrp])       # goff has ngrp+1 entries; the last is the total
        append!(gnjf, dp.gnj)
        # `carry` at each group's start residue.  `grow` is the residue -> group
        # map, so inverting it gives each group's start residue exactly, without
        # re-deriving the recurrence.
        cg = zeros(Int32, ngrp)
        for r in 0:(q - 1)
            g = dp.grow[r + 1]
            g == 0 && continue
            cg[g] = dp.carry[dp.row[r + 1]] ? Int32(1) : Int32(0)
        end
        append!(gcarryf, cg)
        append!(gWf, dp.gW)
        append!(gAf, vec(dp.gA))
    end
    return GPUInterpPlan{WT}(nharms, plans[1].m ÷ 2, q, plans[1].pnum,
                             CuArray(wbase), CuArray(abase), CuArray(gbase),
                             CuArray(Int64[p.rfloor0 for p in plans]),
                             CuArray(Int32[DIRECT_GROUP_V * p.s for p in plans]),
                             CuArray(Int32[DIRECT_GROUP_V * p.base_adv for p in plans]),
                             CuArray(growf), CuArray(gofff), CuArray(gnjf),
                             CuArray(gcarryf), CuArray(gWf), CuArray(gAf), plans)
end

# ---------------------------------------------------------------------------
# The interpolation kernel.
#
# **One warp per (harmonic, trial-group), lane = trial.**  This is not a
# rearrangement of the CPU kernel — it *is* the CPU kernel.  `DIRECT_GROUP_V` is
# 32, and the group weight block `gW` is already laid out `(V, m+Δ)` column-major
# so that lane k is trial k and `gW[:, j]` is contiguous, with the amplitude
# `amps[b0+j]` a broadcast scalar.  That is exactly a CUDA warp: coalesced weight
# loads, a broadcast operand, no gather and no cross-lane reduction.  The
# trials-axis reformulation done for AVX2 transfers verbatim.
#
# Groups are anchored to the GLOBAL trial index, so which group a trial belongs
# to — and hence the exact arithmetic it gets — does not depend on where chunk
# boundaries fall.  That is what makes the GPU free to use a completely different
# chunk size from the CPU without moving a single result, and it is pinned.
#
# `ftp` is `(Nprof, nharms+1)` — TRANSPOSED relative to the CPU's
# `(nharms+1, Nprof)`.  Consecutive lanes are consecutive trials, so this layout
# makes the store one coalesced 256-byte write per warp; the CPU layout would
# make it a 32-way scatter with a 488-byte stride.  CLAUDE.md records the
# transposed layout as 2-3x SLOWER for FFTW, which is a CPU result and must not
# be carried across (`gpu_design.md` §3.3).
# ---------------------------------------------------------------------------
function _interp_kernel!(ftp, amps, namps::Int32, t0::Int64, n::Int32,
                         tfirst::Int64, ngroups::Int32, ::Val{V},
                         q::Int32, res0, qint0, sadv, badv,
                         wbase, abase, gbase, rfloor0, m2::Int32,
                         grow, goff, gnj, gcarry, gW, gA, active,
                         ::Val{TRANSPOSED}) where {V,TRANSPOSED}
    lane = threadIdx().x                                   # 1..V, the trial within the group
    gi = (blockIdx().x - Int32(1)) * blockDim().y + threadIdx().y   # 1..ngroups
    h = blockIdx().y                                       # 1..nharms
    gi <= ngroups || return nothing
    @inbounds active[h] || return nothing                  # harmonic gives up this chunk

    @inbounds begin
        # --- exact integer phase bookkeeping, in 32 bits ---------------------
        # `direct_chunk_state` is `mod/fld(t0*h*pnum, q)`, which on a GPU is a
        # 64-bit division -- microcoded and very slow, once per group.  The whole
        # of it is avoided: the host supplies (res0, qint0) at the chunk's first
        # group (computed there in Int128, exactly as the CPU does), and the
        # advance over `gi` groups is linear, so what remains is one 32-bit
        # division of a quantity bounded by ngroups*V*s < 2^25.  Identical
        # arithmetic, different-sized registers.
        gm1 = gi - Int32(1)
        tot = res0[h] + gm1 * sadv[h]
        res = tot % q
        qint = qint0[h] + Int64(gm1) * Int64(badv[h]) + Int64(tot ÷ q)

        g = grow[(h - Int32(1)) * q + res + Int32(1)]       # group index within harmonic h
        gb = gbase[h] + g
        o = wbase[h] + goff[gb]
        nj = gnj[gb]
        b0 = rfloor0[h] + qint + Int64(gcarry[gb]) + Int64(1 - m2)   # amps[b0 + j]

        WT = eltype(gW)
        ar = zero(WT)
        ai = zero(WT)
        # --- the inner sum ----------------------------------------------------
        # Running 32-bit indices rather than `o + (j-1)*V + lane`: recomputing
        # that is a 64-bit integer multiply-add every iteration, against two
        # FMAs of actual work.  `wi` walks the weight column (stride V, so the
        # warp's 32 lanes read 128 contiguous bytes -- one coalesced load) and
        # `ax` walks the amplitudes (the same address for all 32 lanes -- a
        # broadcast served from L1).
        wi = o + lane
        ax = b0 + Int64(1)
        if ax >= 1 && ax + Int64(nj) - 1 <= namps
            # Fast path: the whole window is inside the file, which is every
            # group but the handful at the very ends of the band.
            for _ in Int32(1):nj
                a = amps[ax]
                w = gW[wi]
                ar = fma(w, real(a), ar)
                ai = fma(w, imag(a), ai)
                wi += V
                ax += Int64(1)
            end
        else
            # End-of-band overhang: the CPU zero-fills its plane buffers here,
            # and this is the same zero by another route.
            for _ in Int32(1):nj
                a = (ax >= 1 && ax <= namps) ? amps[ax] : zero(eltype(amps))
                w = gW[wi]
                ar = fma(w, real(a), ar)
                ai = fma(w, imag(a), ai)
                wi += V
                ax += Int64(1)
            end
        end

        # Partial end groups are computed in full and masked on store -- which is
        # precisely what makes a trial's arithmetic chunk-boundary independent.
        jcol = (tfirst - t0) + Int64(gm1) * Int64(V) + Int64(lane)
        if 1 <= jcol <= n
            A = gA[abase[h] + (Int32(g) - Int32(1)) * V + lane]
            v = A * Complex(ar, ai)
            # TRANSPOSED=true  -> (Nprof, nharms+1): consecutive lanes are
            #   consecutive addresses, one coalesced 256 B write per warp.
            # TRANSPOSED=false -> (nharms+1, Nprof), the CPU layout: consecutive
            #   lanes are (nharms+1) apart, a 32-way scatter.  Slower here, but it
            #   is the layout cuFFT transforms 1.75-1.88x faster (dim 1 vs dim 2),
            #   which is the larger phase.  See gpu_design.md.
            if TRANSPOSED
                ftp[jcol, h + 1] = v
            else
                ftp[h + 1, jcol] = v
            end
        end
    end
    return nothing
end

# Per-(harmonic, chunk) range guard, evaluated on the host exactly as
# `fill_harmonic_row_direct!` does.  Keeping it here rather than in the kernel is
# deliberate: where a harmonic gives up must stay a property of the TRUE trial
# range, not of the group grid, and it also feeds `filled` for the analytic σ.
function _active_harmonics(gp::GPUInterpPlan, ft::FFTFile, t0::Integer, n::Integer)
    namps = length(ft.amps)
    Nhalf = ft.N ÷ 2
    act = fill(false, gp.nharms)
    n >= 1 || return act
    for dp in gp.plans
        m2 = dp.m ÷ 2
        res, qint = CoherentSearch.direct_chunk_state(dp, t0)
        lo_trial = dp.rfloor0 + qint + dp.carry[dp.row[res + 1]] + 2 - m2
        resl, qintl = CoherentSearch.direct_chunk_state(dp, t0 + n - 1)
        hi_trial = dp.rfloor0 + qintl + dp.carry[dp.row[resl + 1]] + 1 + m2
        act[dp.h] = lo_trial >= 1 && hi_trial <= namps && (dp.rfloor0 + qintl) < Nhalf
    end
    return act
end

"""
    gpu_fill_ftprofs!(ftp, gp, amps, namps, N, t0, n; groups_per_block=8)

Fill the device stack `ftp` (`(Nprof, nharms+1)`, zeroed by the caller) for `n`
trials starting at global trial `t0`.  Returns the host `filled` flags.
"""
function gpu_fill_ftprofs!(ftp::CuMatrix, gp::GPUInterpPlan, amps::CuVector,
                           ft::FFTFile, t0::Integer, n::Integer;
                           groups_per_block::Int = 8, transposed::Bool = true)
    V = DIRECT_GROUP_V
    act = _active_harmonics(gp, ft, t0, n)
    any(act) || return act
    tfirst = fld(t0, V) * V
    tlast = fld(t0 + n - 1, V) * V + V - 1
    ngroups = (tlast - tfirst + 1) ÷ V
    # Guard the one place Int64 could bite (see the kernel comment).
    # ngroups*V*s must stay inside Int32 for the kernel's 32-bit recurrence.
    ngroups * V * maximum(p.s for p in gp.plans) < typemax(Int32) ÷ 2 || error(
        "chunk too long for the GPU interpolator's 32-bit phase recurrence " *
        "(ngroups=$ngroups); use a smaller Nprof")
    d_act = CuArray(act)
    # (res, qint) at the chunk's first group, per harmonic -- computed HERE, in
    # the CPU's exact Int128 arithmetic, so the kernel never needs a 64-bit
    # division.  60 elements per chunk; the copy is noise.
    res0 = Vector{Int32}(undef, gp.nharms)
    qint0 = Vector{Int64}(undef, gp.nharms)
    for dp in gp.plans
        r, qi = CoherentSearch.direct_chunk_state(dp, tfirst)
        res0[dp.h] = Int32(r); qint0[dp.h] = Int64(qi)
    end
    threads = (V, groups_per_block)
    blocks = (cld(ngroups, groups_per_block), gp.nharms)
    @cuda threads = threads blocks = blocks _interp_kernel!(
        ftp, amps, Int32(length(ft.amps)), Int64(t0), Int32(n),
        Int64(tfirst), Int32(ngroups), Val(V), Int32(gp.q),
        CuArray(res0), CuArray(qint0), gp.sadv, gp.badv,
        gp.wbase, gp.abase, gp.gbase, gp.rfloor0, Int32(gp.m2),
        gp.grow, gp.goff, gp.gnj, gp.gcarry, gp.gW, gp.gA, d_act,
        transposed ? Val(true) : Val(false))
    return act
end

# --- the §5 stage-1 equivalence entry point -------------------------------
function chunk_ftprofs(::CUDABackend, ft::FFTFile, params::SearchParams,
                       rstart::Real, n::Integer; t0::Integer = 0,
                       weights::Type{<:AbstractFloat} = Float32)
    plans = build_direct_plans(weights, params, rstart)
    gp = GPUInterpPlan(plans)
    d_amps = CuArray(ft.amps)
    ftp = CUDA.zeros(Complex{weights}, n, params.nharms + 1)
    filled = gpu_fill_ftprofs!(ftp, gp, d_amps, ft, t0, n)
    # Returned in the CPU's `(nharms+1, Nprof)` layout so the comparison is
    # against an identical object.  The transpose is in this testing path only —
    # the production pipeline keeps the device layout and hands it to cuFFT.
    host = Array(ftp)
    return permutedims(host), filled
end

# ---------------------------------------------------------------------------
# Stage 2: the inverse transform.
#
# Two cuFFT facts, both measured (2026-08-25) rather than assumed, and the second
# overturns a CPU design decision:
#
# 1. **`brfft` along dimension 2 works on the transposed `(Nprof, nharms+1)`
#    layout**, and agrees with a CPU `irfft` to 1.7e-7.  So the layout chosen for
#    the interpolation kernel's coalesced stores is also the layout cuFFT wants,
#    and the profile output `(Nprof, nbins)` is trial-major -- which is exactly
#    what the boxcar gate below needs.  The CPU's tile transpose has no analogue
#    here; the whole `_bc_transpose!` problem simply does not arise.
#
# 2. **cuFFT cannot transform a STRIDED VIEW** ("Illegal conversion of a
#    DeviceMemory to a Ptr"), so the CPU's decimation trick -- letting FFTW take
#    the stride and reading rows `1, k+1, 2k+1, ...` of `ftprofs` in place, worth
#    1.36-1.60x on the CPU (2026-08-16) -- **does not port.**  The decimated stack
#    has to be materialised.
#
# The answer to (2) is NOT the gather that the CPU deleted.  It is to have the
# interpolation kernel write each harmonic into its rung buffers as well as into
# `ftprofs`, since it already holds the value in registers: harmonic `h` belongs
# to rung `k` when `k` divides `h`, landing at position `h/k + 1`.  CLAUDE.md
# records that fusing the stack writes into the interpolator was measured and
# REJECTED on the CPU, because there `ftprofs` is `(nharms+1, Nprof)` and a row
# write is strided by 976 B.  **In the transposed device layout every one of
# those stores is coalesced**, so the CPU's structural objection does not carry --
# one more verdict that does not travel between the two.
# ---------------------------------------------------------------------------

"""
    GPUChunk{WT}

Device-side scratch for one chunk: the harmonic stack, one dense decimated stack
per rung `k > 1`, the real profiles per rung, and their cuFFT plans.  Built once
per `(params, Nprof)` and reused across chunks and files, exactly as the CPU's
`SearchCache` reuses `Workspace`s.
"""
struct GPUChunk{WT,P}
    Nprof::Int
    ks::Vector{Int}                       # decimation factors, ks[1] == 1
    Hk::Vector{Int}                       # harmonics summed per rung
    ftprofs::CuMatrix{Complex{WT}}        # (Nprof, nharms+1), what the interpolator writes
    cpulayout::Vector{CuMatrix{Complex{WT}}} # per rung: (Hk+1, Nprof), DENSE, what cuFFT reads
    profs::Vector{CuMatrix{WT}}           # per rung: (2Hk, Nprof)
    plans::Vector{P}
    widths::Vector{CuVector{Int32}}       # per rung, the ladder-pruned boxcar bank
    hwidths::Vector{Vector{Int}}          # host copies
end

function GPUChunk(::Type{WT}, params::SearchParams, Nprof::Integer) where {WT}
    nh = params.nharms
    ks = sort!(unique(params.decimations))
    ks[1] == 1 || throw(ArgumentError("decimations must include 1"))
    Hk = [fld(nh, k) for k in ks]
    ftprofs = CUDA.zeros(Complex{WT}, Nprof, nh + 1)
    cpulayout = CuMatrix{Complex{WT}}[]
    profs = CuMatrix{WT}[]
    plans = []
    hwidths = Vector{Int}[]
    widths = CuVector{Int32}[]
    for (i, k) in enumerate(ks)
        src = CUDA.zeros(Complex{WT}, Hk[i] + 1, Nprof)
        push!(cpulayout, src)
        dst = CUDA.zeros(WT, 2Hk[i], Nprof)
        push!(profs, dst)
        push!(plans, plan_brfft(src, 2Hk[i], 1))     # dim 1: 1.75-1.88x faster
        w = ladder_boxcar_widths(2Hk[i], k, params)
        push!(hwidths, w)
        push!(widths, CuArray(Int32.(w)))
    end
    pl = [p for p in plans]
    return GPUChunk{WT,eltype(pl)}(Int(Nprof), ks, Hk, ftprofs, cpulayout, profs,
                                   pl, widths, hwidths)
end

# ---------------------------------------------------------------------------
# Stage 2: the boxcar gate.
#
# One thread per trial, the profile's wrapped prefix sum staged in shared memory.
# `profs` is `(Nprof, nbins)`, so `profs[trial, i]` for consecutive trials is
# contiguous: the load is coalesced with no transpose, and the shared array is
# indexed `[(i-1)*B + tid]` so the per-width scan is bank-conflict free too.
#
# This is `_boxcar_psum!` + `_boxcar_scan` with the zero baseline, in the same
# operation order per profile.  DC is held at zero so the profile mean is already
# ~0 and the `delta*S_tot` term is the rounding-level correction the CPU applies.
# ---------------------------------------------------------------------------
function _boxcar_kernel!(out, profs, n::Int32, nbins::Int32, wmax::Int32,
                         widths, nw::Int32, invsigma::Float32, ::Val{B}) where {B}
    tid = threadIdx().x
    j = (blockIdx().x - Int32(1)) * Int32(B) + tid          # trial
    ps = CuDynamicSharedArray(Float32, (Int(nbins) + Int(wmax) + 1) * B)
    live = j <= n
    @inbounds begin
        ps[tid] = 0.0f0
        acc = 0.0f0
        for i in Int32(1):(nbins + wmax)
            idx = i > nbins ? i - nbins : i                 # wrap: a boxcar may straddle phase 0
            # `profs` is (nbins, Nprof), so profs[idx, j] for the block's B
            # consecutive trials is a contiguous B-element run -- coalesced with no
            # transpose.  This is the GPU form of the CPU's `_bc_transpose!`, and
            # it costs nothing because the prefix sum needed shared memory anyway.
            acc += live ? profs[idx, j] : 0.0f0
            ps[i * B + tid] = acc
        end
        live || return nothing
        stot = ps[nbins * B + tid] - ps[tid]
        best = -Inf32
        for wi in Int32(1):nw
            w = widths[wi]
            duty = Float32(w) / Float32(nbins)
            invsw = invsigma / sqrt(Float32(w) * (1.0f0 - duty))
            m = ps[w * B + tid] - ps[tid]                   # finite seed, as on the CPU
            for p in Int32(2):nbins
                o = (p - Int32(1)) * B + tid
                d = ps[o + w * B] - ps[o]
                m = ifelse(d > m, d, m)
            end
            c = (m - duty * stot) * invsw
            best = ifelse(c > best, c, best)
        end
        out[j] = best
    end
    return nothing
end

# Variant 2: no prefix sum and no shared memory.  Per width, slide the window
# directly, keeping the running sum in a register:
#
#     S = sum(prof[1..w]);  for p in 2:nbins:  S += prof[p+w-1] - prof[p-1]
#
# It re-reads the profile ~2*nwidths times instead of ~once, which looks worse and
# may not be: the block's working set is B*nbins*4 bytes (15 KB at B=32,
# nbins=120), so the re-reads are L1 hits, and in exchange the kernel uses **no
# shared memory at all** -- which is what caps occupancy in variant 1, where
# (nbins+wmax+1)*B floats allows only ~192 threads per SM.
#
# Same arithmetic and same operation order per profile as `_boxcar_scan`.
function _boxcar_slide_kernel!(out, profs, n::Int32, nbins::Int32,
                               widths, nw::Int32, invsigma::Float32)
    j = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j <= n || return nothing
    @inbounds begin
        stot = 0.0f0
        for i in Int32(1):nbins
            stot += profs[i, j]
        end
        best = -Inf32
        for wi in Int32(1):nw
            w = widths[wi]
            duty = Float32(w) / Float32(nbins)
            invsw = invsigma / sqrt(Float32(w) * (1.0f0 - duty))
            S = 0.0f0
            for i in Int32(1):w
                S += profs[i, j]
            end
            m = S                                   # finite seed, as on the CPU
            for p in Int32(2):nbins
                a = p + w - Int32(1)
                a > nbins && (a -= nbins)           # wrap: a boxcar may straddle phase 0
                S += profs[a, j] - profs[p - Int32(1), j]
                m = ifelse(S > m, S, m)
            end
            c = (m - duty * stot) * invsw
            best = ifelse(c > best, c, best)
        end
        out[j] = best
    end
    return nothing
end

# Variant 3: hoist the subtracted prefix term out of the width loop.
#
# The scan is `max_p (psum[p+w] - psum[p])` per width, i.e. TWO shared loads per
# (phase, width).  But `psum[p]` does not depend on `w` -- so putting phase
# outside and width inside loads it once and serves every width, cutting shared
# traffic from `2*nw` to `nw+1` loads per phase (10 -> 6 at the k=1 bank).
#
# The per-width running maxima must be an `NTuple`, not an array: indexed by a
# runtime `wi` they spill to local memory and the whole point is lost.  Same trap
# `DIRECT_GROUP_V` and `_BC_BATCH` document -- and the reason `NW` is a `Val`.
@inline _bc_upd(m::NTuple{NW,Float32}, ps, o::Int32, base::Float32, ws,
                ::Val{NW}) where {NW} =
    ntuple(i -> (@inbounds d = ps[o + ws[i]] - base; ifelse(d > m[i], d, m[i])), Val(NW))

function _boxcar_fused_kernel!(out, profs, n::Int32, nbins::Int32, wmax::Int32,
                               widths, invsigma::Float32, ::Val{B},
                               ::Val{NW}) where {B,NW}
    tid = threadIdx().x
    j = (blockIdx().x - Int32(1)) * Int32(B) + tid
    ps = CuDynamicSharedArray(Float32, (Int(nbins) + Int(wmax) + 1) * B)
    live = j <= n
    @inbounds begin
        ps[tid] = 0.0f0
        acc = 0.0f0
        for i in Int32(1):(nbins + wmax)
            idx = i > nbins ? i - nbins : i
            acc += live ? profs[idx, j] : 0.0f0
            ps[i * B + tid] = acc
        end
        live || return nothing
        # width offsets in shared-array units, hoisted out of the phase loop
        ws = ntuple(i -> widths[i] * Int32(B), Val(NW))
        stot = ps[nbins * B + tid] - ps[tid]
        m = ntuple(i -> ps[ws[i] + tid] - ps[tid], Val(NW))   # finite seed, as on the CPU
        for p in Int32(2):nbins
            o = (p - Int32(1)) * Int32(B) + tid
            m = _bc_upd(m, ps, o, ps[o], ws, Val(NW))
        end
        best = -Inf32
        for i in Int32(1):NW
            w = widths[i]
            duty = Float32(w) / Float32(nbins)
            invsw = invsigma / sqrt(Float32(w) * (1.0f0 - duty))
            c = (m[i] - duty * stot) * invsw
            best = ifelse(c > best, c, best)
        end
        out[j] = best
    end
    return nothing
end

const _GPU_BC_B = 32          # trials per block; see bench/gpu_boxcar_bench.jl
const _GPU_BC_VARIANT = 3     # 1 = per-width scan, 2 = register sliding, 3 = width-fused

function gpu_boxcar!(out, profs::CuMatrix, n::Integer,
                     nbins::Integer, widths::CuVector{Int32}, wmax::Integer,
                     invsigma::Real; B::Integer = _GPU_BC_B,
                     variant::Integer = _GPU_BC_VARIANT)
    if variant == 2
        @cuda threads = B blocks = cld(n, B) _boxcar_slide_kernel!(
            out, profs, Int32(n), Int32(nbins), widths,
            Int32(length(widths)), Float32(invsigma))
        return out
    end
    shmem = (Int(nbins) + Int(wmax) + 1) * Int(B) * sizeof(Float32)
    if variant == 3
        nw = length(widths)
        nw <= 8 || return gpu_boxcar!(out, profs, n, nbins, widths, wmax, invsigma;
                                      B = B, variant = 1)
        _bcf_launch!(out, profs, n, nbins, wmax, widths, invsigma, shmem,
                     Val(Int(B)), Val(nw))
        return out
    end
    B == 32 ? _bc_launch!(out, profs, n, nbins, wmax, widths, invsigma, shmem, Val(32)) :
    B == 64 ? _bc_launch!(out, profs, n, nbins, wmax, widths, invsigma, shmem, Val(64)) :
    B == 128 ? _bc_launch!(out, profs, n, nbins, wmax, widths, invsigma, shmem, Val(128)) :
    B == 256 ? _bc_launch!(out, profs, n, nbins, wmax, widths, invsigma, shmem, Val(256)) :
    throw(ArgumentError("unsupported boxcar block size $B"))
    return out
end

# `B` must reach the kernel as a `Val`: the shared-array offsets are computed
# from it in the inner loop, and with a runtime `Int` they are not folded.  Same
# trap `_BC_BATCH` and `DIRECT_GROUP_V` document on the CPU side.
@inline function _bcf_launch!(out, profs, n, nbins, wmax, widths, invsigma, shmem,
                              ::Val{B}, ::Val{NW}) where {B,NW}
    @cuda threads = B blocks = cld(n, B) shmem = shmem _boxcar_fused_kernel!(
        out, profs, Int32(n), Int32(nbins), Int32(wmax), widths,
        Float32(invsigma), Val(B), Val(NW))
end

@inline function _bc_launch!(out, profs, n, nbins, wmax, widths, invsigma, shmem,
                             ::Val{B}) where {B}
    @cuda threads = B blocks = cld(n, B) shmem = shmem _boxcar_kernel!(
        out, profs, Int32(n), Int32(nbins), Int32(wmax), widths,
        Int32(length(widths)), Float32(invsigma), Val(B))
end

# --- stage-2 equivalence entry points --------------------------------------

function chunk_profiles(::CUDABackend, ft::FFTFile, params::SearchParams,
                        rstart::Real, n::Integer; t0::Integer = 0,
                        weights::Type{<:AbstractFloat} = Float32, k::Integer = 1)
    gc = GPUChunk(weights, params, n)
    plans = build_direct_plans(weights, params, rstart)
    gp = GPUInterpPlan(plans)
    d_amps = CuArray(ft.amps)
    gpu_fill_ftprofs!(gc.ftprofs, gp, d_amps, ft, t0, n)
    i = findfirst(==(k), gc.ks)
    i === nothing && throw(ArgumentError("k=$k not in decimations $(gc.ks)"))
    fill_stacks!(gc, n)
    mul!(gc.profs[i], gc.plans[i], gc.cpulayout[i])
    return Array(gc.profs[i])[:, 1:n]               # already (nbins, n)
end

function chunk_boxcar(::CUDABackend, ft::FFTFile, params::SearchParams,
                      rstart::Real, n::Integer; t0::Integer = 0,
                      weights::Type{<:AbstractFloat} = Float32, k::Integer = 1,
                      invsigma::Real = 1.0)
    gc = GPUChunk(weights, params, n)
    plans = build_direct_plans(weights, params, rstart)
    gp = GPUInterpPlan(plans)
    d_amps = CuArray(ft.amps)
    gpu_fill_ftprofs!(gc.ftprofs, gp, d_amps, ft, t0, n)
    i = findfirst(==(k), gc.ks)
    i === nothing && throw(ArgumentError("k=$k not in decimations $(gc.ks)"))
    fill_stacks!(gc, n)
    mul!(gc.profs[i], gc.plans[i], gc.cpulayout[i])
    out = CUDA.zeros(Float32, n)
    gpu_boxcar!(out, gc.profs[i], n, 2gc.Hk[i], gc.widths[i],
                gc.hwidths[i][end], invsigma)
    return Float64.(Array(out))
end

# ---------------------------------------------------------------------------
# Transpose-and-decimate: `(Nprof, nharms+1)` -> one dense `(Hk+1, Nprof)` stack
# per rung.
#
# **Why this kernel exists, measured 2026-08-25.** The interpolator wants the
# transposed layout (its store is then one coalesced 256 B write per warp; the
# CPU layout makes it a 32-way scatter and costs **2.54-2.65x** on the interp
# kernel). cuFFT wants the CPU layout (**dim 1 is 1.75-1.88x faster than dim 2**
# at every rung and every `Nprof`). Neither pure layout wins: transposed
# throughout pays 0.23 s on the transform, CPU-layout throughout pays 0.21 s on
# interpolation. So the two phases keep the layout each wants and a dedicated
# transpose sits between them, where a shared-memory tile makes both the read and
# the write coalesced.
#
# **§3.3 of `gpu_design.md` asserted the transposed layout was "almost certainly
# right" for cuFFT. It was verified for CORRECTNESS and never timed against the
# alternative, and it is wrong.** Recorded as the mistake it was: this file's
# standing lesson about benchmarking against the shipped kernel rather than the
# obvious one has a twin -- verifying that something *works* is not evidence that
# it is *fast*.
#
# It also replaces the strided `copyto!` this started as, which measured 0.40 s
# on the reference workload -- a gather doing exactly what the CPU deleted in
# 2026-08-16 for the same reason.
# ---------------------------------------------------------------------------
const _GPU_TR_T = 32          # trials per tile

function _transpose_stack_kernel!(dst, src, n::Int32, nrow::Int32, k::Int32,
                                  ::Val{T}) where {T}
    tile = CuDynamicSharedArray(ComplexF32, (T + 1) * Int(nrow))
    t0 = (blockIdx().x - Int32(1)) * Int32(T)
    tx = threadIdx().x
    ty = threadIdx().y
    @inbounds begin
        # The block is sized `max(T, nrow)` in x because the two phases use x for
        # DIFFERENT axes: the trial when reading, the row when writing.  Each phase
        # must therefore guard against the other's bound -- getting this wrong is an
        # out-of-bounds shared write, not a wrong answer.
        if tx <= Int32(T)
            # read: consecutive threadIdx().x are consecutive TRIALS -> coalesced
            r = ty
            while r <= nrow
                t = t0 + tx
                v = t <= n ? src[t, (r - Int32(1)) * k + Int32(1)] : zero(eltype(src))
                tile[(r - Int32(1)) * (T + 1) + tx] = v
                r += blockDim().y
            end
        end
        sync_threads()
        # write: consecutive threadIdx().x are consecutive ROWS -> coalesced
        c = ty
        while c <= Int32(T)
            t = t0 + c
            if t <= n && tx <= nrow
                dst[tx, t] = tile[(tx - Int32(1)) * (T + 1) + c]
            end
            c += blockDim().y
        end
    end
    return nothing
end

"""
    fill_stacks!(gc, n)

Fill every rung's dense stack (and rung 1's, which is the plain transpose) from
the transposed `ftprofs`, for `n` trials.
"""
function fill_stacks!(gc::GPUChunk, n::Integer)
    T = _GPU_TR_T
    for i in eachindex(gc.ks)
        k = gc.ks[i]
        dst = gc.cpulayout[i]
        nrow = gc.Hk[i] + 1
        shmem = (T + 1) * nrow * sizeof(ComplexF32)
        @cuda threads = (max(T, nrow), 8) blocks = cld(n, T) shmem = shmem _transpose_stack_kernel!(
            dst, gc.ftprofs, Int32(n), Int32(nrow), Int32(k), Val(T))
    end
    return gc
end

# Kept for the record and for the bench's baseline column: the strided gather
# this replaced.  Do not use it in the pipeline.
function _fill_stack_gather!(gc::GPUChunk, i::Integer)
    k = gc.ks[i]
    copyto!(gc.cpulayout[i], view(gc.ftprofs, :, 1:k:(gc.Hk[i] * k + 1)))
end

# ---------------------------------------------------------------------------
# The chunk loop.  Mirrors `_search_region!`, and everything outside it -- the
# trial grid, the plans, the analytic-sigma sanity check, duplicate and harmonic
# collapsing, the candidate file -- is backend-independent and unchanged.
#
# **Candidate extraction is a plain download, not a device-side compaction.**
# One `Float32` per (trial, rung) is 6 floats per trial: ~200 MB over the whole
# reference workload, or ~0.02 s of PCIe. A device compaction with atomics would
# save nothing measurable and would add a capacity guard and a failure mode.
#
# **There is no `Float64` exact rescan**, and it is not needed. The CPU runs a
# `Float32` gate and re-scores everything within `boxcar_gatemargin` of the
# threshold in `Float64`; here the whole metric is `Float32` throughout and
# agrees with the CPU's `Float64` value to ~2e-7 (§4.2) -- four orders of
# magnitude inside the 0.01 gate margin, so it cannot move which trials become
# candidates, only the seventh digit of their reported S/N.
# ---------------------------------------------------------------------------
function _region!(::CUDABackend, ft::FFTFile, params::SearchParams,
                  workspaces::Vector, nbins::Integer,
                  r_lo::Real, r_hi::Real, lodr::Real, total::Integer,
                  Nprof::Integer, nchunks::Integer, nt::Integer;
                  threshold::Real, norm, metricstats, progress::Symbol,
                  dplans::AbstractVector)
    params.sigma === :analytic || error(
        "the CUDA backend requires sigma = :analytic; the measured scale is a " *
        "per-chunk MAD with no device implementation yet (gpu_design.md stage 4)")
    norm === nothing || error("the CUDA backend does not support --normalize yet")
    metricstats === nothing || error("the CUDA backend does not support --metricstats yet")

    WT = eltype(dplans[1].W)
    gc = GPUChunk(WT, params, Nprof)
    gp = GPUInterpPlan(dplans)
    d_amps = CuArray(ft.amps)
    # One column per rung, so the whole chunk's metrics come back in ONE D2H
    # copy.  Six separate `copyto!`s were six synchronisation points per chunk --
    # each blocking until its boxcar kernel retired, which serialised the queue
    # and cost more than every kernel in it.
    nk = length(gc.ks)
    out = CUDA.zeros(Float32, Nprof, nk)
    hostm = Matrix{Float32}(undef, Nprof, nk)
    nvalids = Vector{Int}(undef, nk)
    cands = Candidate[]
    nyq = ft.N / 2

    for c in 1:nchunks
        i0 = (c - 1) * Nprof
        n = min(Nprof, total - i0)
        rstart = r_lo + i0 * lodr
        # Zeroed every chunk: the interpolator writes only the harmonics that
        # carry data, and a harmonic that gives up must leave a ZERO row -- which
        # is also what the DC row (never written) has to be.
        fill!(gc.ftprofs, 0)
        filled = gpu_fill_ftprofs!(gc.ftprofs, gp, d_amps, ft, i0, n)
        fill_stacks!(gc, n)
        for i in eachindex(gc.ks)
            mul!(gc.profs[i], gc.plans[i], gc.cpulayout[i])
        end
        for i in eachindex(gc.ks)
            k = gc.ks[i]
            Hk = gc.Hk[i]
            # Valid decimated trials are the prefix j = 1..nvalid; past-Nyquist
            # columns are partly zero-padded and are skipped, exactly as
            # `decim_pass!` does.
            nvalid = n
            if k > 1
                nvalid = 0
                while nvalid < n && k * (rstart + nvalid * lodr) < nyq
                    nvalid += 1
                end
            end
            nvalids[i] = nvalid
            nvalid == 0 && continue
            sig = _analytic_sigma(filled, k, Hk)
            invsig = sig > 0 ? 1.0 / sig : 0.0
            gpu_boxcar!(view(out, :, i), gc.profs[i], nvalid, 2Hk, gc.widths[i],
                        gc.hwidths[i][end], invsig)
        end
        # One copy for the whole chunk, after every rung has been queued.  The
        # offset form, not `copyto!(view(...), view(...))`: a SubArray of a
        # CuArray falls back to scalar indexing, which CUDA.jl refuses outright.
        copyto!(hostm, 1, out, 1, Nprof * nk)
        thr = Float32(threshold)
        @inbounds for i in 1:nk
            k = gc.ks[i]
            Hk = gc.Hk[i]
            for j in 1:nvalids[i]
                hostm[j, i] > thr || continue          # Float32 compare: no conversion
                rf = k * (rstart + (j - 1) * lodr)
                push!(cands, Candidate(rf / ft.T, Float64(hostm[j, i]), rf, Hk))
            end
        end
        _render_progress(progress, c, nchunks)
    end
    if progress !== :none
        _render_progress(progress, nchunks, nchunks)
        println(stderr)
    end
    return cands
end

end # module
