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
using CoherentSearch: SearchBackend, SearchParams, FFTFile, DirectPlan,
                      build_direct_plans, DIRECT_GROUP_V
# `import`, not `using`: extending a function from another module requires the
# name be brought in for extension, not merely for reference.
import CoherentSearch: chunk_ftprofs

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
                         grow, goff, gnj, gcarry, gW, gA, active) where {V}
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
            ftp[jcol, h + 1] = A * Complex(ar, ai)
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
                           groups_per_block::Int = 8)
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
        gp.grow, gp.goff, gp.gnj, gp.gcarry, gp.gW, gp.gA, d_act)
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

end # module
