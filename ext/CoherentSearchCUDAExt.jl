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
                      _GPU_PHASE_NS, _GPU_TIMING, _GPU_SUBBATCH,
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

# ---------------------------------------------------------------------------
# Zero only what the interpolator will NOT overwrite.
#
# `fill!(ftprofs, 0)` wrote all `nharms+1` columns every chunk, and measured
# **4.0% of device time on an RTX A4000 and 2.3% on an A100** at 388 and
# 1833 GB/s respectively -- i.e. it ran at the card's full DRAM write bandwidth,
# so the only way to make it cheaper is to move fewer bytes (gpu_design.md
# §4.15).  Almost all of it was redundant:
#
#  - `_interp_kernel!` writes EVERY trial `1:n` of every ACTIVE harmonic's
#    column (the store is masked to `1 <= jcol <= n`, and `harmonic_fits` is now
#    uniform over a chunk thanks to §4.14's forced splits), so those columns need
#    no pre-zeroing.
#  - Column 1 is the DC row.  Nothing in this file ever writes it, so it is zero
#    from `CUDA.zeros` in the `GPUChunk` constructor and stays that way for the
#    life of the cached workspace.
#  - Columns beyond `n` are never read: the transpose kernel guards `t <= n` and
#    the boxcar guards `j <= nvalid <= n`.
#
# What is left is the columns of harmonics that gave up on THIS chunk and may
# hold the previous chunk's values -- in the standard 0.1-33.3 Hz band, none.
# ---------------------------------------------------------------------------
function _zero_inactive!(ftp::CuMatrix, act::Vector{Bool}, n::Integer)
    z = zero(eltype(ftp))
    @inbounds for h in eachindex(act)
        act[h] || fill!(view(ftp, 1:Int(n), h + 1), z)
    end
    return ftp
end

"""
    gpu_fill_ftprofs!(ftp, gp, amps, namps, N, t0, n; groups_per_block=8)

Fill the device stack `ftp` (`(Nprof, nharms+1)`, zeroed by the caller) for `n`
trials starting at global trial `t0`.  Returns the host `filled` flags.
"""
function gpu_fill_ftprofs!(ftp::CuMatrix, gp::GPUInterpPlan, amps::CuVector,
                           ft::FFTFile, t0::Integer, n::Integer;
                           groups_per_block::Int = 8, transposed::Bool = true,
                           act::Union{Nothing,Vector{Bool}} = nothing)
    V = DIRECT_GROUP_V
    # `act` is a pure function of `(gp, ft, t0, n)`, and `_region!` needs it
    # BEFORE this call in order to zero only the columns this one will not
    # write, so it computes it once and passes it in.  Recomputed here when a
    # caller (the stage-1 entry points, the benches) does not.
    act = act === nothing ? _active_harmonics(gp, ft, t0, n) : act
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

# ---------------------------------------------------------------------------
# Per-rung transform SUB-BATCHING -- the one place the six rungs are allowed to
# disagree about the chunk size.
#
# The rungs read the same `ftprofs`, but they want very different batch widths.
# A rung-`k` column is `(Hk+1)` complex in and `2Hk` real out, so the deep fold
# carries 968 B per trial and the shallowest 168 B: the deep fold fills L2 with
# few columns and the shallow ones need many before there is enough parallelism
# to fill the SMs.  `gpu_design.md` §0.3 measured the per-rung optima spread
# across the whole `Nprof` sweep on a 40 MB-L2 card, worth 1.40x on the stage.
#
# Nothing forces one batch, though: each rung is a batched transform over
# COLUMNS of a column-major array, so a contiguous column range is contiguous
# memory and can be transformed on its own.  That **decouples the two competing
# pressures on `--blocksize`** -- the interpolator, the transpose and the boxcar
# all want the largest chunk available (§4.8 measured 1.24x and 1.42x from 16384
# to 262144 on two cards, with the transform subtracted out), while the
# transforms want L2-sized batches.
#
# **The policy is per-device and derived, not hardcoded**, because §0.4 measured
# sub-batching at 1.40x on the 40 MB Ada and exactly 1.00x on the 4 MB RTX 2080
# Super: with no residency to arrange there is nothing to gain, and splitting the
# batch would only add launches.  So a rung is sub-batched only when the split
# actually achieves residency -- when `_SUB_MIN_COLS` columns of it fit in the
# target -- which turns itself off on a small-L2 card without a device list.
#
# **The formula reproduces §0.3's measured per-rung optima**, which is why it is
# a fraction of L2 rather than a table.  At `_SUB_L2_FRACTION = 0.5` on the Ada
# it derives 20661 / 40983 / 60975 / 80645 / 100000 / 119047 columns for
# `k = 1…6` against the measured 16384 / 32768 / 65536 / 65536 / 131072 / 131072
# -- monotone in the same direction and within ~1.3x at every rung, from a
# formula fitted to none of them.
#
# `_SUB_MIN_COLS` is the parallelism floor, and it is §0.3's other reading: at
# `Nprof = 16384` the `k = 4,5,6` rungs sat at 85-93% of the DRAM copy despite
# working sets of only 2.6-3.9 MB, which cannot be a cache effect -- 16384
# batches of a 20-bin transform is simply too little work for 48 SMs.
# ---------------------------------------------------------------------------

const _SUB_L2_FRACTION = 0.5      # of device L2; see above
const _SUB_MIN_COLS    = 16384    # parallelism floor, and the residency gate

"""Bytes of L2 to aim a single rung's transform working set at."""
_sub_target_bytes() =
    floor(Int, _SUB_L2_FRACTION *
          CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_L2_CACHE_SIZE))

"""
    _sub_cols(WT, Hk, Nprof, target; policy = true) -> Int

Columns per transform sub-batch for a rung summing `Hk` harmonics, or `Nprof`
(i.e. do not sub-batch) when the split would not achieve L2 residency anyway.

`policy = true` is the `:auto` path and applies both guards -- the residency
gate and the `_SUB_MIN_COLS` parallelism floor.  An explicit byte target
(`gpu_subbatch!(n)`) sets `policy = false` and means exactly what it says, with
no guards: that is what lets a bench sweep past the knee in both directions and
a small-L2 card exercise the split in a test, neither of which the shipped
default should ever do on its own.
"""
function _sub_cols(::Type{WT}, Hk::Integer, Nprof::Integer, target::Integer;
                   policy::Bool = true) where {WT}
    bytes = sizeof(Complex{WT}) * (Hk + 1) + sizeof(WT) * 2Hk
    target <= 0 && return Int(Nprof)
    if policy
        # The gate: if even the parallelism floor does not fit, there is no
        # residency to arrange and sub-batching can only cost launches.
        bytes * _SUB_MIN_COLS > target && return Int(Nprof)
        return min(Int(Nprof), max(_SUB_MIN_COLS, fld(target, bytes)))
    end
    return clamp(fld(target, bytes), 1, Int(Nprof))
end

"""
    GPUChunk{WT}

Device-side scratch for one chunk: the harmonic stack, one dense decimated stack
per rung `k > 1`, the real profiles per rung, and their cuFFT plans.  Built once
per `(params, Nprof)` and reused across chunks and files, exactly as the CPU's
`SearchCache` reuses `Workspace`s.

Each rung's transform is run in column sub-batches of `sub[i]` (`nblocks[i]`
of them, the last of `tail[i]` columns) -- see the note above.  On a small-L2
card `sub[i] == Nprof` and `nblocks[i] == 1`, which is exactly the single
un-split `mul!` this had before.
"""
struct GPUChunk{WT,P}
    Nprof::Int
    ks::Vector{Int}                       # decimation factors, ks[1] == 1
    Hk::Vector{Int}                       # harmonics summed per rung
    ftprofs::CuMatrix{Complex{WT}}        # (Nprof, nharms+1), what the interpolator writes
    cpulayout::Vector{CuMatrix{Complex{WT}}} # per rung: (Hk+1, Nprof), DENSE, what cuFFT reads
    profs::Vector{CuMatrix{WT}}           # per rung: (2Hk, Nprof)
    plans::Vector{P}                      # per rung, for a full `sub[i]`-column batch
    tailplans::Vector{P}                  # per rung, for the last (short) batch
    sub::Vector{Int}                      # columns per transform sub-batch
    tail::Vector{Int}                     # columns in the last sub-batch
    nblocks::Vector{Int}                  # 1 == not sub-batched
    widths::Vector{CuVector{Int32}}       # per rung, the ladder-pruned boxcar bank
    hwidths::Vector{Vector{Int}}          # host copies
end

function GPUChunk(::Type{WT}, params::SearchParams, Nprof::Integer;
                  subbatch = :auto) where {WT}
    nh = params.nharms
    ks = sort!(unique(params.decimations))
    ks[1] == 1 || throw(ArgumentError("decimations must include 1"))
    Hk = [fld(nh, k) for k in ks]
    # `:auto` derives the target from this device's L2; `:off` disables
    # sub-batching outright; an Integer is a target in bytes, which is what the
    # bench sweeps and what lets a small-L2 card exercise the split in a test.
    auto = subbatch === :auto
    target = auto ? _sub_target_bytes() : subbatch === :off ? 0 : Int(subbatch)
    ftprofs = CUDA.zeros(Complex{WT}, Nprof, nh + 1)
    cpulayout = CuMatrix{Complex{WT}}[]
    profs = CuMatrix{WT}[]
    plans = []; tailplans = []
    subs = Int[]; tails = Int[]; nblocks = Int[]
    hwidths = Vector{Int}[]
    widths = CuVector{Int32}[]
    for (i, k) in enumerate(ks)
        src = CUDA.zeros(Complex{WT}, Hk[i] + 1, Nprof)
        push!(cpulayout, src)
        dst = CUDA.zeros(WT, 2Hk[i], Nprof)
        push!(profs, dst)
        # Balance the blocks rather than leaving a ragged remainder: `nb` blocks
        # of `sub`, the last one absorbing what is left.  `tail` is in
        # `[1, sub]`, so at most two plan sizes per rung.
        sub0 = _sub_cols(WT, Hk[i], Nprof, target; policy = auto)
        nb = max(1, cld(Int(Nprof), sub0))
        sub = cld(Int(Nprof), nb)
        tail = Int(Nprof) - (nb - 1) * sub
        push!(subs, sub); push!(tails, tail); push!(nblocks, nb)
        # dim 1: 1.75-1.88x faster than the transposed layout (see above).
        p = plan_brfft(nb == 1 ? src : view(src, :, 1:sub), 2Hk[i], 1)
        push!(plans, p)
        push!(tailplans, (nb == 1 || tail == sub) ? p :
                         plan_brfft(view(src, :, 1:tail), 2Hk[i], 1))
        w = ladder_boxcar_widths(2Hk[i], k, params)
        push!(hwidths, w)
        push!(widths, CuArray(Int32.(w)))
    end
    pl = [p for p in plans]; tpl = eltype(pl)[p for p in tailplans]
    return GPUChunk{WT,eltype(pl)}(Int(Nprof), ks, Hk, ftprofs, cpulayout, profs,
                                   pl, tpl, subs, tails, nblocks, widths, hwidths)
end

"""
    transform!(gc, i)

The rung-`i` inverse transform, in `gc.nblocks[i]` column sub-batches.  Each
`view` is a contiguous column range of a column-major array, so it is dense
memory and cuFFT accepts it -- unlike the STRIDED view of the CPU's decimation
trick, which is what `Illegal conversion of a DeviceMemory to a Ptr` refers to
above.  Every column's transform is independent of the others, so splitting the
batch is a scheduling change and not a numerical one.
"""
@inline function transform!(gc::GPUChunk, i::Integer)
    nb = gc.nblocks[i]
    if nb == 1
        mul!(gc.profs[i], gc.plans[i], gc.cpulayout[i])
        return nothing
    end
    sub = gc.sub[i]; src = gc.cpulayout[i]; dst = gc.profs[i]
    @inbounds for b in 1:nb
        j0 = (b - 1) * sub + 1
        j1 = b == nb ? gc.Nprof : b * sub
        p = b == nb ? gc.tailplans[i] : gc.plans[i]
        mul!(view(dst, :, j0:j1), p, view(src, :, j0:j1))
    end
    return nothing
end

"""Human-readable sub-batch policy, for the bench scripts."""
function subbatch_report(gc::GPUChunk)
    io = IOBuffer()
    any(>(1), gc.nblocks) || return "transform sub-batching: off (no rung achieves L2 residency)"
    println(io, "transform sub-batching (Nprof = $(gc.Nprof)):")
    for i in eachindex(gc.ks)
        bytes = (sizeof(eltype(gc.cpulayout[i])) * (gc.Hk[i] + 1) +
                 sizeof(eltype(gc.profs[i])) * 2gc.Hk[i])
        println(io, "  k=$(gc.ks[i]) H=$(gc.Hk[i]): $(gc.nblocks[i]) x $(gc.sub[i])" *
                    (gc.tail[i] == gc.sub[i] ? "" : " (+ tail $(gc.tail[i]))") *
                    "  working set $(round(bytes * gc.sub[i] / 2^20, digits = 1)) MB")
    end
    return String(take!(io))
end

# ---------------------------------------------------------------------------
# Stage 2: the boxcar gate.
#
# One thread per trial, the profile's wrapped prefix sum staged in shared memory.
# This is `_boxcar_psum!` + `_boxcar_scan` with the zero baseline, in the same
# operation order per profile.  DC is held at zero so the profile mean is already
# ~0 and the `delta*S_tot` term is the rounding-level correction the CPU applies.
#
# **THE STAGING LOAD IS NOT COALESCED, A COMMENT HERE USED TO CLAIM IT WAS, AND
# COALESCING IT IS A LOSS.  Do not re-guess this one (gpu_design.md §4.15).**
#
# The comment removed from here read: "`profs` is `(Nprof, nbins)`, so
# `profs[trial, i]` for consecutive trials is contiguous".  It WAS -- until §4.2
# measured cuFFT transforming dimension 1 1.75-1.88x faster than dimension 2 and
# the layout flipped to `(nbins, Nprof)`.  In column-major that makes
# `profs[i, j]`, with `j` the trial and consecutive threads holding consecutive
# `j`, a stride-`nbins` gather: 32 separate sectors per warp per phase bin, of
# which 4 bytes each are wanted.  The rationale was carried forward and never
# re-derived -- the same failure as the stale `GPUChunk` comment about "the
# answer to (2)", in the other direction.
#
# **So the comment was wrong and the code is right.**  `_bc_stage!(..., Val(true))`
# reads the block's `B` columns as `B` contiguous runs and transposes them into
# shared, cutting L1 transactions ~8x.  Measured on a GTX 1080 over all six
# rungs, variant 3 at `B = 32`, bit-exact to the shipped path:
#
#     per-thread gather, stride B       0.1297 s   <- ships
#     per-thread gather, stride B+1     0.1363 s   padding alone, 1.05x
#     coalesced,         stride B+1     0.2003 s   **1.55x SLOWER**
#
# The gather's re-requests were already L1 hits -- a warp's 32 columns are 15 kB,
# well inside L1 -- so the transaction count was never what the phase was paying.
# What coalescing ADDS is the loop shape it forces: `nbins` is not a multiple of
# `B` at any rung (120/60/40/30/24/20 against 32), so one flat 126-iteration
# stream with good ILP becomes a 32-trip outer loop over columns around a 3-4
# trip inner one, plus a `sync_threads()`.  That costs 1.47x; the odd shared
# stride the transpose needs costs the other 1.05x.
#
# **This is positive evidence for §4.12's reading of the boxcar**, not just a
# dead end: an 8x cut in memory transactions changing nothing confirms the phase
# is issue-bound rather than memory-bound, which is why it tracks `SMs x clock`
# and not bandwidth.  Optimising it means removing WORK, not improving access.
#
# The coalesced path stays in the build behind `Val{COAL}`, defaulting OFF.  It
# costs nothing (Julia specialises it only if the bench asks), and it lets
# `bench/gpu_boxcar_bench.jl` re-test the verdict on a card with a larger L1 in
# ONE process -- rather than across a checkout, where a precompile difference
# could masquerade as the effect.
#
# Both staging paths are **bit-exact** to each other, deliberately: the wrap
# region is materialised in shared (slots `nbins+1 .. nbins+wmax` copy slots
# `1 .. wmax`) and the prefix reads each slot before overwriting it, so the
# running accumulation is the identical sequence of adds in the identical order.
# Folding the wrap arithmetically instead -- `ps[i] = stot + ps[i-nbins]`, which
# would also shrink shared by `wmax/(nbins+wmax)` -- re-associates the sum and is
# NOT bit-exact.  That is a separate experiment, not something to smuggle in here.
# ---------------------------------------------------------------------------

"""
    _bc_stage!(ps, profs, j0, n, nbins, wmax, tid, Val(B), Val(COAL)) -> Bool

Stage this block's `B` profile columns into shared as the wrapped raw sequence,
returning whether this thread's own column is live.  Callers must treat a
`true` from `COAL` as "a `sync_threads()` has happened".
"""
@inline function _bc_stage!(ps, profs, j0::Int32, n::Int32, nbins::Int32,
                            wmax::Int32, tid::Int32, ::Val{B},
                            ::Val{COAL}) where {B,COAL}
    S = Int32(COAL ? B + 1 : B)
    live = (j0 + tid) <= n
    @inbounds if COAL
        # Coalesced: `b` outer walks columns (contiguous in `profs`), `i` inner is
        # strided by `B` so the warp reads 32 consecutive floats per step.  No
        # integer division -- computing `(i, b)` from a flat index would cost one
        # `divrem` by a runtime `nbins` per element, which is not free on a GPU.
        for b in Int32(0):(Int32(B) - Int32(1))
            col = j0 + b + Int32(1)
            goff = (col - Int32(1)) * nbins
            hot = col <= n
            i = tid
            while i <= nbins
                ps[i * S + b + Int32(1)] = hot ? profs[goff + i] : 0.0f0
                i += Int32(B)
            end
        end
        sync_threads()
        # The wrap, copied within this thread's own column: shared-to-shared,
        # stride 1 across threads, and it is what keeps the prefix bit-exact.
        for i in Int32(1):wmax
            ps[(nbins + i) * S + tid] = ps[i * S + tid]
        end
    else
        for i in Int32(1):(nbins + wmax)
            idx = i > nbins ? i - nbins : i
            ps[i * S + tid] = live ? profs[idx, j0 + tid] : 0.0f0
        end
    end
    return live
end

"""Shared-memory floats one boxcar block needs, for the launch's `shmem`."""
@inline _bc_shmem(nbins, wmax, B, COAL) =
    (Int(nbins) + Int(wmax) + 1) * (COAL ? Int(B) + 1 : Int(B))

function _boxcar_kernel!(out, profs, n::Int32, nbins::Int32, wmax::Int32,
                         widths, nw::Int32, invsigma::Float32, ::Val{B},
                         ::Val{COAL}) where {B,COAL}
    tid = threadIdx().x
    j0 = (blockIdx().x - Int32(1)) * Int32(B)
    j = j0 + tid                                            # trial
    S = Int32(COAL ? B + 1 : B)
    ps = CuDynamicSharedArray(Float32, _bc_shmem(nbins, wmax, B, COAL))
    live = _bc_stage!(ps, profs, j0, n, nbins, wmax, tid, Val(B), Val(COAL))
    live || return nothing                                  # AFTER the stage's sync
    @inbounds begin
        # In-place prefix: each slot is read before it is overwritten, so this is
        # the same running accumulation, in the same order, as the pre-staging
        # version -- hence bit-exact.
        ps[tid] = 0.0f0
        acc = 0.0f0
        for i in Int32(1):(nbins + wmax)
            acc += ps[i * S + tid]
            ps[i * S + tid] = acc
        end
        stot = ps[nbins * S + tid] - ps[tid]
        best = -Inf32
        for wi in Int32(1):nw
            w = widths[wi]
            duty = Float32(w) / Float32(nbins)
            invsw = invsigma / sqrt(Float32(w) * (1.0f0 - duty))
            m = ps[w * S + tid] - ps[tid]                   # finite seed, as on the CPU
            for p in Int32(2):nbins
                o = (p - Int32(1)) * S + tid
                d = ps[o + w * S] - ps[o]
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
                               ::Val{NW}, ::Val{COAL}) where {B,NW,COAL}
    tid = threadIdx().x
    j0 = (blockIdx().x - Int32(1)) * Int32(B)
    j = j0 + tid
    S = Int32(COAL ? B + 1 : B)
    ps = CuDynamicSharedArray(Float32, _bc_shmem(nbins, wmax, B, COAL))
    live = _bc_stage!(ps, profs, j0, n, nbins, wmax, tid, Val(B), Val(COAL))
    live || return nothing                                  # AFTER the stage's sync
    @inbounds begin
        ps[tid] = 0.0f0
        acc = 0.0f0
        for i in Int32(1):(nbins + wmax)
            acc += ps[i * S + tid]
            ps[i * S + tid] = acc
        end
        # width offsets in shared-array units, hoisted out of the phase loop
        ws = ntuple(i -> widths[i] * S, Val(NW))
        stot = ps[nbins * S + tid] - ps[tid]
        m = ntuple(i -> ps[ws[i] + tid] - ps[tid], Val(NW))   # finite seed, as on the CPU
        for p in Int32(2):nbins
            o = (p - Int32(1)) * S + tid
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
const _GPU_BC_COALESCED = false  # measured 1.55x SLOWER on a GTX 1080; see above

function gpu_boxcar!(out, profs::CuMatrix, n::Integer,
                     nbins::Integer, widths::CuVector{Int32}, wmax::Integer,
                     invsigma::Real; B::Integer = _GPU_BC_B,
                     variant::Integer = _GPU_BC_VARIANT,
                     coalesced::Bool = _GPU_BC_COALESCED)
    if variant == 2
        @cuda threads = B blocks = cld(n, B) _boxcar_slide_kernel!(
            out, profs, Int32(n), Int32(nbins), widths,
            Int32(length(widths)), Float32(invsigma))
        return out
    end
    shmem = _bc_shmem(nbins, wmax, B, coalesced) * sizeof(Float32)
    C = coalesced ? Val(true) : Val(false)
    if variant == 3
        nw = length(widths)
        nw <= 8 || return gpu_boxcar!(out, profs, n, nbins, widths, wmax, invsigma;
                                      B = B, variant = 1, coalesced = coalesced)
        _bcf_launch!(out, profs, n, nbins, wmax, widths, invsigma, shmem,
                     Val(Int(B)), Val(nw), C)
        return out
    end
    B == 32 ? _bc_launch!(out, profs, n, nbins, wmax, widths, invsigma, shmem, Val(32), C) :
    B == 64 ? _bc_launch!(out, profs, n, nbins, wmax, widths, invsigma, shmem, Val(64), C) :
    B == 128 ? _bc_launch!(out, profs, n, nbins, wmax, widths, invsigma, shmem, Val(128), C) :
    B == 256 ? _bc_launch!(out, profs, n, nbins, wmax, widths, invsigma, shmem, Val(256), C) :
    throw(ArgumentError("unsupported boxcar block size $B"))
    return out
end

# `B` must reach the kernel as a `Val`: the shared-array offsets are computed
# from it in the inner loop, and with a runtime `Int` they are not folded.  Same
# trap `_BC_BATCH` and `DIRECT_GROUP_V` document on the CPU side.
@inline function _bcf_launch!(out, profs, n, nbins, wmax, widths, invsigma, shmem,
                              ::Val{B}, ::Val{NW}, ::Val{COAL}) where {B,NW,COAL}
    @cuda threads = B blocks = cld(n, B) shmem = shmem _boxcar_fused_kernel!(
        out, profs, Int32(n), Int32(nbins), Int32(wmax), widths,
        Float32(invsigma), Val(B), Val(NW), Val(COAL))
end

@inline function _bc_launch!(out, profs, n, nbins, wmax, widths, invsigma, shmem,
                             ::Val{B}, ::Val{COAL}) where {B,COAL}
    @cuda threads = B blocks = cld(n, B) shmem = shmem _boxcar_kernel!(
        out, profs, Int32(n), Int32(nbins), Int32(wmax), widths,
        Int32(length(widths)), Float32(invsigma), Val(B), Val(COAL))
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
    transform!(gc, i)
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
    transform!(gc, i)
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
# **FUSED ACROSS RUNGS (gpu_design.md §4.15).**  The per-rung kernel below reads
# `ftprofs` once PER RUNG, so a six-rung ladder reads `sum(Hk+1) = 153` of its 61
# columns -- 2.5x over -- and writes 153 rows: **2448 B per trial.**  Measured, it
# ran at **389 GB/s on an RTX A4000 (102% of that card's device copy) and
# 1459 GB/s on an A100 (87%)**, i.e. the kernel is DRAM-saturated and the only
# lever is moving fewer bytes.  Reading each column ONCE into a shared tile and
# writing every rung's rows from it moves `61 + 153 = 1712 B` per trial, **1.43x
# less**, and it also collapses six kernel launches per chunk into one -- which
# matters most at a small `--blocksize`, where there are tens of thousands of
# chunks (the Ada's optimum is 8192, i.e. 12881 chunks on the reference file).
#
# The tile is `(T+1) x (nharms+1)`, 16.1 kB at `T = 32, nharms = 60` -- the SAME
# size the per-rung kernel already used for rung 1, so peak shared use is
# unchanged and occupancy stays thread-limited rather than shared-limited.
#
# **The `+1` row padding is load-bearing and only half works here.**  A 64-bit
# shared access is conflict-free when the element stride is odd; the per-rung
# write reads with stride `T+1 = 33` and is clean.  The fused write reads rung
# `k`'s row `rr` from tile column `(rr-1)*k + 1`, i.e. stride `33k`, which is odd
# only for odd `k` -- so `k = 2, 4, 6` take a 2-, 4- and 2-way conflict on their
# 31, 16 and 11 rows.  That is ~1.6x the shared transactions on the write phase,
# spent on a kernel with ~7x of shared-bandwidth headroom over the DRAM traffic
# it is actually limited by.  Measured rather than argued -- see §4.15.
const _GPU_TR_T = 32          # trials per tile

# Unrolled over the rungs: a runtime index into the `dsts` tuple would spill it
# to local memory.  Same trap `_bc_upd`'s `NTuple` and `DIRECT_GROUP_V` document.
@inline function _tr_write_one!(dst, k::Int32, nr::Int32, tile, t0::Int32,
                                n::Int32, tx::Int32, ty::Int32, ::Val{T}) where {T}
    @inbounds begin
        stride = k * (Int32(T) + Int32(1))
        c = ty
        while c <= Int32(T)
            t = t0 + c
            # write: consecutive threadIdx().x are consecutive ROWS -> coalesced
            if t <= n && tx <= nr
                dst[tx, t] = tile[(tx - Int32(1)) * stride + c]
            end
            c += blockDim().y
        end
    end
    return nothing
end

# Both methods must carry the SAME argument types beyond the tuples: with the
# trailing arguments left untyped on the base case, the two are AMBIGUOUS at the
# empty tuple (one more specific in args 1-3, the other in args 5-9) and the
# kernel fails to compile with a `jl_f_throw_methoderror` in the IR.
@inline _tr_write!(::Tuple{}, ::Tuple{}, ::Tuple{}, tile,
                   t0::Int32, n::Int32, tx::Int32, ty::Int32,
                   ::Val{T}) where {T} = nothing
@inline function _tr_write!(dsts::Tuple, ks::Tuple, nrows::Tuple, tile,
                            t0::Int32, n::Int32, tx::Int32, ty::Int32,
                            ::Val{T}) where {T}
    _tr_write_one!(dsts[1], ks[1], nrows[1], tile, t0, n, tx, ty, Val(T))
    _tr_write!(Base.tail(dsts), Base.tail(ks), Base.tail(nrows),
               tile, t0, n, tx, ty, Val(T))
end

function _transpose_fused_kernel!(dsts::Tuple, src, n::Int32, nharm1::Int32,
                                  ks::Tuple, nrows::Tuple, ::Val{T}) where {T}
    tile = CuDynamicSharedArray(eltype(src), (T + 1) * Int(nharm1))
    t0 = (blockIdx().x - Int32(1)) * Int32(T)
    tx = threadIdx().x
    ty = threadIdx().y
    @inbounds begin
        # The block is sized `max(T, nharms+1)` in x because the two phases use x
        # for DIFFERENT axes -- the trial when reading, the row when writing -- so
        # each must guard against the other's bound.  Getting this wrong is an
        # out-of-bounds SHARED write, not a wrong answer.
        if tx <= Int32(T)
            # read: consecutive threadIdx().x are consecutive TRIALS -> coalesced,
            # and each column of `ftprofs` is read exactly ONCE for all six rungs.
            r = ty
            while r <= nharm1
                t = t0 + tx
                v = t <= n ? src[t, r] : zero(eltype(src))
                tile[(r - Int32(1)) * (T + 1) + tx] = v
                r += blockDim().y
            end
        end
        sync_threads()
        _tr_write!(dsts, ks, nrows, tile, t0, n, tx, ty, Val(T))
    end
    return nothing
end

# Kept for `bench/gpu_transpose_bench.jl`'s baseline column and because it is the
# reference the fused kernel is pinned against.  Not used in the pipeline.
function _transpose_stack_kernel!(dst, src, n::Int32, nrow::Int32, k::Int32,
                                  ::Val{T}) where {T}
    tile = CuDynamicSharedArray(eltype(src), (T + 1) * Int(nrow))
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
    fill_stacks!(gc, n; fused = true)

Fill every rung's dense stack (and rung 1's, which is the plain transpose) from
the transposed `ftprofs`, for `n` trials.

`fused = false` selects the per-rung kernel this replaced, which
`bench/gpu_transpose_bench.jl` times as the baseline and `test_gpu.jl` pins the
fused one against **bit-exactly** -- both kernels copy the same values with no
arithmetic, so anything short of equality is an indexing bug.
"""
fill_stacks!(gc::GPUChunk, n::Integer; fused::Bool = true) =
    fused ? _fill_stacks_fused!(gc, n, Val(length(gc.ks))) : _fill_stacks_perrung!(gc, n)

# `Val(length(gc.ks))` costs one dynamic dispatch per chunk (~100 ns) and buys a
# compile-time-unrolled rung loop; against it, the fusion removes five of the six
# kernel launches per chunk (~6 us each), so it pays for itself many times over
# even at the smallest blocksize anyone runs.
@noinline function _fill_stacks_fused!(gc::GPUChunk, n::Integer,
                                       ::Val{NK}) where {NK}
    T = _GPU_TR_T
    nh1 = size(gc.ftprofs, 2)
    dsts  = ntuple(i -> gc.cpulayout[i], Val(NK))
    ks    = ntuple(i -> Int32(gc.ks[i]), Val(NK))
    nrows = ntuple(i -> Int32(gc.Hk[i] + 1), Val(NK))
    shmem = (T + 1) * nh1 * sizeof(eltype(gc.ftprofs))
    @cuda threads = (max(T, nh1), 8) blocks = cld(n, T) shmem = shmem _transpose_fused_kernel!(
        dsts, gc.ftprofs, Int32(n), Int32(nh1), ks, nrows, Val(T))
    return gc
end

function _fill_stacks_perrung!(gc::GPUChunk, n::Integer)
    T = _GPU_TR_T
    for i in eachindex(gc.ks)
        k = gc.ks[i]
        dst = gc.cpulayout[i]
        nrow = gc.Hk[i] + 1
        shmem = (T + 1) * nrow * sizeof(eltype(gc.ftprofs))
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

"""
    _check_device_memory(ft, params, Nprof, WT)

Refuse to start a search that will not fit, and say what to change.

**This exists because the failure mode is other people's problem, not ours.** A
GPU that drives a display shares its memory with the desktop: on `fitzroy` a
1.29 GB `.fft` plus a `Nprof = 131072` workspace pushed the card to 86% and
produced a system low-memory warning while the search ran perfectly well. An
out-of-memory *error* is recoverable and obvious; quietly starving the compositor
is neither. So this checks first, and the message names the two knobs that
actually help.
"""
function _check_device_memory(ft::FFTFile, params::SearchParams, Nprof::Integer,
                              ::Type{WT}) where {WT}
    nh = params.nharms
    ks = sort!(unique(params.decimations))
    amps = length(ft.amps) * sizeof(ComplexF32)
    stack = sum((fld(nh, k) + 1) * Nprof * 2 * sizeof(WT) for k in ks)
    prof = sum(2 * fld(nh, k) * Nprof * sizeof(WT) for k in ks)
    # cuFFT allocates scratch of its own for the batched C2R plans, and this used
    # to be missing entirely -- so the gate passed configurations that then died
    # with a bare OutOfGPUMemoryError, which is the exact failure it exists to
    # replace with a useful message.  Measured on a GTX 1080 (PM0063, six rungs,
    # pool `used` after a search minus the accounted total): 262 B/trial at
    # `Nprof = 32768`, 983 at 131072, 1164 at 524288 -- rising, then flattening
    # near ~1.2 kB as cuFFT switches strategy with batch size.  Charging the
    # plateau over-counts at small `Nprof`, but by 0.03 GiB at 32768, which is
    # nothing; under-counting at large `Nprof` was worth **1.39x** and is not.
    # Approximate and from one card: it is a budget, not a model.
    fftws = 1200 * Nprof
    need = amps + (nh + 1) * Nprof * 2 * sizeof(WT) + stack + prof +
           2 * Nprof * length(ks) * sizeof(Float32) +   # `out` is double-buffered
           fftws
    # `CUDA.memory_info()` reports what the DRIVER has free, and CUDA.jl's pool
    # holds on to freed blocks rather than returning them -- deliberately, since
    # §4.6 moved `CUDA.reclaim()` out to `release_backend!` so that a many-file
    # run does not pay for it per file.  So from the second search in a process
    # onwards this free figure EXCLUDES the previous file's amplitudes and chunk
    # workspace, while `need` above still counts them: the gate charges for the
    # same bytes twice and refuses configurations that fit.
    #
    # Not hypothetical.  On a 3.67 GiB RTX A400 it refused `--blocksize 131072`
    # with "needs about 1.64 GiB but only 1.77 GiB is free" -- a message that
    # reads as self-contradictory until you find the 0.90 margin -- when the
    # actual new demand was ~0.35 GiB.  Two rows of that card's sweep were lost
    # to it and its reported optimum is a ceiling, not an optimum
    # (gpu_design.md §4.12).
    #
    # The fix reclaims ONLY on the path that was about to fail, so the normal
    # path still never pays the ~0.36 s.  Reclaiming and re-reading is exact,
    # where adding the pool's reserve to the free figure would not be: the
    # reserve counts bytes that are still in use as well as bytes that are not.
    free, tot = CUDA.memory_info()
    if need > 0.90 * free
        CUDA.reclaim()
        free, tot = CUDA.memory_info()
    end
    gb(x) = x / 2^30
    if need > 0.90 * free
        error("""
            This search needs about $(round(gb(need), digits=2)) GiB on the GPU but only $(round(gb(free), digits=2)) GiB of $(round(gb(tot), digits=2)) GiB is free (after reclaiming the CUDA.jl pool).

              amplitudes      $(round(gb(amps), digits=2)) GiB  (the whole .fft)
              chunk workspace $(round(gb(need - amps - fftws), digits=2)) GiB  (--blocksize $Nprof)
              cuFFT scratch   $(round(gb(fftws), digits=2)) GiB  (estimated; also scales with --blocksize)

            Lower --blocksize (the workspace scales with it; the amplitudes do not),
            or use a GPU that is not also driving a display.""")
    elseif need > 0.60 * free
        # A triple-quoted string with line continuations is not a valid @warn
        # message argument; keep it on one line.
        @warn "GPU search will use most of the free device memory; if this card also drives a display, expect the desktop to notice" needed_GiB = round(gb(need), digits = 2) free_GiB = round(gb(free), digits = 2) blocksize = Nprof
    end
    return nothing
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
# Time phase `i` if timing is on.  `CUDA.synchronize()` is what makes the number
# meaningful and also what makes it not free -- see `gpu_timing!`.
# ---------------------------------------------------------------------------
# Throughput cache: `GPUChunk`, `GPUInterpPlan` and the metric download buffers
# reused ACROSS FILES.
#
# All three are pure functions of `(WT, params, Nprof)` — and `GPUInterpPlan`
# additionally of `r_lo`, through the `DirectPlan`s it is built from — and of
# nothing in the file's contents.  That is exactly the property that makes the
# CPU's `SearchCache` safe across a heterogeneous glob, and it is safe here for
# the same reason.  A dedispersion sweep gives every DM identical `N`, `dt` and
# hence `T` and `r_lo`, so the whole 220-file run reuses one workspace.
#
# **Why this is worth caching and the amplitudes are not.** Measured per file on
# a GTX 1080: `CUDA.reclaim()` 0.362 s, `GPUChunk` build+free 0.073-0.133 s,
# `GPUInterpPlan` 0.004 s — against an amplitude upload of only 0.042 s for a
# 188 MB file.  So the genuinely per-file part is ~8% of the per-file cost and
# the cacheable part is ~92%.  End to end that was 1.73 s marginal per file
# against a ~1.14 s search (gpu_design.md §4.6).
#
# **`reclaim()` is deliberately NOT here any more.** It returns memory to the
# *driver* rather than to CUDA.jl's pool, which is what lets a desktop have its
# memory back — but it only needs doing once per invocation, not once per file.
# `release_backend!` does it, and `CoherentSearch.main` calls that after the file
# loop.  Peak memory is unchanged either way: the workspace was live during every
# search before, and the amplitudes are still freed per file, so nothing
# accumulates.  What changes is only that the valleys BETWEEN files are no longer
# handed back.
# ---------------------------------------------------------------------------

"""
    _ChunkIO(Nprof, nk)

The host side of the download/scan overlap: two page-locked staging matrices,
their per-buffer valid-trial counts and chunk start bins, a copy stream, and two
pairs of events (`ready` = this chunk's boxcars have retired, `done` = its D2H
has landed).  Cached across files by `_cached_chunk`; see `_region!` for what
makes the double buffering safe.
"""
struct _ChunkIO
    hostm::NTuple{2,Matrix{Float32}}
    nvalids::NTuple{2,Vector{Int}}
    rstarts::Vector{Float64}
    stream::CuStream
    ready::NTuple{2,CuEvent}
    done::NTuple{2,CuEvent}
end

function _ChunkIO(Nprof::Int, nk::Int)
    _ChunkIO(ntuple(_ -> CUDA.pin(Matrix{Float32}(undef, Nprof, nk)), 2),
             ntuple(_ -> Vector{Int}(undef, nk), 2),
             zeros(Float64, 2),
             CuStream(),
             ntuple(_ -> CuEvent(CUDA.EVENT_DISABLE_TIMING), 2),
             ntuple(_ -> CuEvent(CUDA.EVENT_DISABLE_TIMING), 2))
end

const _CACHE_CHUNK = Ref{Any}(nothing)
const _CACHE_GP    = Ref{Any}(nothing)
const _CACHE_OUT   = Ref{Any}(nothing)
const _CACHE_IO    = Ref{Any}(nothing)   # pinned host buffers, copy stream, events
const _CACHE_KEY   = Ref{Any}(nothing)   # (WT, params, Nprof, subbatch)
const _CACHE_GPKEY = Ref{Any}(nothing)   # (WT, params, r_lo)

"""Free the cached device workspace (not the driver-level reclaim)."""
function _free_cache!()
    gc = _CACHE_CHUNK[]
    if gc !== nothing
        CUDA.unsafe_free!(gc.ftprofs)
        for a in gc.cpulayout; CUDA.unsafe_free!(a); end
        for a in gc.profs; CUDA.unsafe_free!(a); end
        for w in gc.widths; CUDA.unsafe_free!(w); end
    end
    _CACHE_OUT[] !== nothing && CUDA.unsafe_free!(_CACHE_OUT[])
    # The pinned host buffers unpin themselves through the finalizer `CUDA.pin`
    # attaches, so dropping the reference is all that is needed here.
    _CACHE_IO[] = nothing
    _CACHE_CHUNK[] = nothing; _CACHE_OUT[] = nothing; _CACHE_KEY[] = nothing
    _CACHE_GP[] = nothing; _CACHE_GPKEY[] = nothing
    return nothing
end

"""
    _cached_chunk(WT, params, Nprof) -> (GPUChunk, out, io)

The chunk workspace, metric buffers and download plumbing for this
`(WT, params, Nprof)`, rebuilt only when the key changes.  `params` is compared
by `===`, as `_plans!` does on the CPU: the CLI hands the same `SearchParams`
object to every file.

**`io` is cached for the same reason the device workspace is** (§4.6): its host
matrices are PAGE-LOCKED, and `CUDA.pin` is a driver call, not a `malloc`.
Building it per file would put two registrations, two unregistrations and
`2 * Nprof * nk * 4` bytes of host allocation on every one of a 220-file
throughput run -- exactly the per-file cost §4.6 removed from everything else.
"""
function _cached_chunk(::Type{WT}, params::SearchParams, Nprof::Integer) where {WT}
    key = (WT, params, Int(Nprof), _GPU_SUBBATCH[])
    k = _CACHE_KEY[]
    if k !== nothing && k[1] === key[1] && k[2] === key[2] &&
       k[3] == key[3] && k[4] === key[4]
        return _CACHE_CHUNK[]::GPUChunk{WT}, _CACHE_OUT[]::CuArray{Float32,3},
               _CACHE_IO[]::_ChunkIO
    end
    _free_cache!()                       # a changed key means the old one is dead
    gc = GPUChunk(WT, params, Nprof; subbatch = _GPU_SUBBATCH[])
    # Two metric buffers, not one: the download/scan overlap in `_region!`
    # alternates between them.  It costs 24 B per trial of the ~2912 B the whole
    # chunk workspace uses (§4.11), i.e. under 1%.
    out = CUDA.zeros(Float32, Nprof, length(gc.ks), 2)
    io = _ChunkIO(Int(Nprof), length(gc.ks))
    _CACHE_CHUNK[] = gc; _CACHE_OUT[] = out; _CACHE_IO[] = io; _CACHE_KEY[] = key
    return gc, out, io
end

"""The interpolation plan for this `(WT, params, r_lo)`, rebuilt only on change."""
function _cached_gp(::Type{WT}, params::SearchParams, r_lo::Real,
                    dplans::AbstractVector) where {WT}
    key = (WT, params, Float64(r_lo))
    k = _CACHE_GPKEY[]
    if k !== nothing && k[1] === key[1] && k[2] === key[2] && k[3] == key[3]
        return _CACHE_GP[]::GPUInterpPlan{WT}
    end
    gp = GPUInterpPlan(dplans)
    _CACHE_GP[] = gp; _CACHE_GPKEY[] = key
    return gp
end

"""
    release_backend!(::CUDABackend)

Free the cross-file workspace and return its memory to the driver.  Call after a
batch of searches; `CoherentSearch.main` does so after its file loop.
"""
function CoherentSearch.release_backend!(::CUDABackend)
    _free_cache!()
    CUDA.reclaim()
    return nothing
end

@inline function _gpt(f, i::Int)
    if _GPU_TIMING[]
        CUDA.synchronize(); t = time_ns()
        r = f()
        CUDA.synchronize(); @inbounds _GPU_PHASE_NS[i] += time_ns() - t
        return r
    end
    return f()
end

function _region!(::CUDABackend, ft::FFTFile, params::SearchParams,
                  workspaces::Vector, nbins::Integer,
                  r_lo::Real, r_hi::Real, lodr::Real, total::Integer,
                  Nprof::Integer, nchunks::Integer, nt::Integer,
                  cstarts::AbstractVector{Int};
                  threshold::Real, norm, metricstats, progress::Symbol,
                  dplans::AbstractVector)
    params.sigma === :analytic || error(
        "the CUDA backend requires sigma = :analytic; the measured scale is a " *
        "per-chunk MAD with no device implementation yet (gpu_design.md stage 4)")
    norm === nothing || error("the CUDA backend does not support --normalize yet")
    metricstats === nothing || error("the CUDA backend does not support --metricstats yet")

    WT = eltype(dplans[1].W)
    _check_device_memory(ft, params, Nprof, WT)
    gc, out, io = _cached_chunk(WT, params, Nprof)
    gp = _cached_gp(WT, params, r_lo, dplans)
    d_amps = CuArray(ft.amps)          # the one genuinely per-file allocation
    # One column per rung, so the whole chunk's metrics come back in ONE D2H
    # copy.  Six separate `copyto!`s were six synchronisation points per chunk --
    # each blocking until its boxcar kernel retired, which serialised the queue
    # and cost more than every kernel in it.
    nk = length(gc.ks)
    # --- Download/scan overlap -------------------------------------------
    # Chunk `c`'s D2H copy and host-side candidate scan run WHILE chunk `c+1`'s
    # kernels are on the device.  Before this, both were on the critical path:
    # `download` + `scan` measured **31.5% of wall clock on an A100 and 31.4%
    # on an RTX A4000** (against 4.6% on an RTX A400), because the device got
    # ~8x faster over six cards and the host did not (gpu_design.md §4.12).
    # It is the largest single item left on any modern card.
    #
    # Three things make it safe, and all three are needed:
    #
    #  - **Double buffering.** `out[:, :, b]` and `hostm[b]` alternate, so
    #    chunk `c+2` is the first to reuse chunk `c`'s buffers -- and the host
    #    has already waited for chunk `c`'s copy (while scanning it, at
    #    iteration `c+1`) before iteration `c+2` queues anything into them.
    #    Host-side ordering is what makes the write-after-read safe; there is no
    #    need for the compute stream to wait on an event.
    #  - **A separate copy stream plus an event.** `copystream` waits on
    #    `ready[b]`, recorded on the compute stream after the last boxcar, so
    #    the transfer cannot start early; running it off the compute stream is
    #    what lets PCIe and the SMs work at the same time.
    #  - **Pinned host memory.** `unsafe_copyto!(...; async = true)` only really
    #    returns early from page-locked memory, and pageable D2H measured
    #    10-11 GB/s on two hosts -- about half of what the link can do.
    #
    # Candidate ORDER is unchanged: chunk `c-1` is scanned before chunk `c` is,
    # and within a chunk the rung/trial loops are untouched.  So the output is
    # byte-identical, which is what `test_gpu.jl` and §5's batch-invariance pin
    # check.
    # Built once per `(WT, params, Nprof)` and reused across files, like the
    # device workspace -- pinning host memory is a driver call (see `_ChunkIO`).
    hostm, nvalids, rstarts = io.hostm, io.nvalids, io.rstarts
    copystream, ready, done = io.stream, io.ready, io.done
    pending = 0            # buffer holding a chunk not yet scanned; 0 = none
    cands = Candidate[]
    nyq = ft.N / 2
    thr = Float32(threshold)

    # Scan one landed host buffer.  Split out only so that the overlapped and
    # the timing-on schedules can share one implementation -- if these ever
    # diverge, the phase table stops describing the code that ships.
    @inline function scan_buffer!(b::Int)
        hm = hostm[b]; nv = nvalids[b]; rs = rstarts[b]
        @inbounds for i in 1:nk
            k = gc.ks[i]
            Hk = gc.Hk[i]
            for j in 1:nv[i]
                hm[j, i] > thr || continue             # Float32 compare: no conversion
                rf = k * (rs + (j - 1) * lodr)
                push!(cands, Candidate(rf / ft.T, Float64(hm[j, i]), rf, Hk))
            end
        end
        return nothing
    end

    for c in 1:nchunks
        b = 1 + (c - 1) % 2
        # Chunk bounds come from `cstarts`, not from `c * Nprof`: forced splits
        # at harmonic-availability changes keep `filled` correct for every trial
        # in the chunk (see `chunk_starts`).  Splits only shorten, so `n <= Nprof`
        # and every device buffer sized on `Nprof` is unaffected.
        i0 = cstarts[c]
        n = (c < nchunks ? cstarts[c + 1] : Int(total)) - i0
        rstart = r_lo + i0 * lodr
        # Only the columns the interpolator will NOT write need zeroing -- the
        # harmonics that give up on this chunk.  See `_zero_inactive!`; the DC
        # column is zero from construction and no harmonic that is active here
        # can leave a stale value behind.  `act` is computed once and handed to
        # the fill so it is not derived twice.
        act = _active_harmonics(gp, ft, i0, n)
        _gpt(() -> _zero_inactive!(gc.ftprofs, act, n), 1)
        filled = _gpt(() -> gpu_fill_ftprofs!(gc.ftprofs, gp, d_amps, ft, i0, n;
                                              act = act), 2)
        _gpt(() -> fill_stacks!(gc, n), 3)
        _gpt(4) do
            for i in eachindex(gc.ks)
                transform!(gc, i)
            end
        end
        _gpt(5) do
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
            nvalids[b][i] = nvalid
            nvalid == 0 && continue
            sig = _analytic_sigma(filled, k, Hk)
            invsig = sig > 0 ? 1.0 / sig : 0.0
            gpu_boxcar!(view(out, :, i, b), gc.profs[i], nvalid, 2Hk, gc.widths[i],
                        gc.hwidths[i][end], invsig)
        end
        end
        rstarts[b] = rstart
        # One copy for the whole chunk, after every rung has been queued, issued
        # on `copystream` behind an event so it cannot overtake the boxcars.
        # The pointer form, not `copyto!(view(...), view(...))`: a SubArray of a
        # CuArray falls back to scalar indexing, which CUDA.jl refuses outright.
        CUDA.record(ready[b])
        CUDA.wait(ready[b], copystream)
        GC.@preserve hostm out begin
            CUDA.unsafe_copyto!(pointer(hostm[b]),
                                pointer(out, 1 + (b - 1) * Nprof * nk),
                                Nprof * nk; stream = copystream, async = true)
        end
        CUDA.record(done[b], copystream)

        if _GPU_TIMING[]
            # Timing already serialises the queue (see `gpu_timing!`), so make
            # the schedule serial too: the phase table then measures download
            # and scan un-overlapped, which is what makes it comparable with
            # every card report taken before this change.  The clean total, run
            # with timing OFF, is where the overlap shows up.
            _gpt(() -> CUDA.synchronize(done[b]), 6)
            tscan = time_ns()
            scan_buffer!(b)
            @inbounds _GPU_PHASE_NS[7] += time_ns() - tscan
        else
            # Scan the PREVIOUS chunk while this one's kernels run.  The wait is
            # on that chunk's own copy event, so it does not block on anything
            # queued for chunk `c`.
            if pending != 0
                CUDA.synchronize(done[pending])
                scan_buffer!(pending)
            end
            pending = b
        end
        _render_progress(progress, c, nchunks)
    end
    # Drain the last chunk.
    if pending != 0
        CUDA.synchronize(done[pending])
        scan_buffer!(pending)
        pending = 0
    end
    if progress !== :none
        _render_progress(progress, nchunks, nchunks)
        println(stderr)
    end
    # **The amplitudes are the ONE genuinely per-file allocation, and they are
    # still freed here.**  Without this each call uploads the whole `.fft` again
    # and CUDA.jl's pool holds every call's copy until the next GC: three
    # searches of NGC6624 (1.29 GB of amplitudes each) reached 5.87 GB of an
    # 7.92 GB card.  In throughput mode -- hundreds of files, the deployment
    # target -- that is an out-of-memory failure, not an inefficiency.
    CUDA.unsafe_free!(d_amps)
    # The chunk workspace, the interpolation plan and `out` are NOT freed here:
    # they are keyed on `(WT, params, Nprof)` / `(WT, params, r_lo)` and reused by
    # the next file.  `release_backend!` frees them and calls `CUDA.reclaim()`
    # once, after the whole batch.  Peak memory is unchanged -- all of this was
    # live during the search anyway -- so what a display GPU loses is only the
    # valley between files, in exchange for ~0.5 s per file.
    return cands
end

end # module
