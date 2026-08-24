#!/usr/bin/env julia
#
# Standalone AVX-512 downclocking / scatter probe.   NO dependencies, no data
# files, no CoherentSearch install needed -- base Julia + InteractiveUtils only.
# Runs in well under a minute.
#
#     julia bench/avx512_probe.jl
#     julia --cpu-target=skylake bench/avx512_probe.jl     # AVX2-only control
#
# WHY.  CoherentSearch's boxcar gate transposes a tile of profiles.  With
# `--precision f32` the source array is already `Float32`, and LLVM vectorises
# the *wrong* loop of that nest: it reads down a column (contiguous) and writes
# the tile with a 512-bit SCATTER (`vscatterqps`).  On a Xeon Silver 4114
# (Skylake-SP) that is a *heavy* AVX-512 op, so the core drops to turbo licence
# level 2 -- measured at 57.5% of all cycles and an effective 2.06 GHz against
# 2.77 GHz for the `Float64` path, which never leaves level 1.  The whole search
# paid for it, including phases with no 512-bit code in them at all.
#
# The question this script answers for a NEW host: does that host emit the same
# scatter, and if so does it cost anything?  AMD Zen 4/5 has AVX-512 with no
# Skylake-style licensing, so it may show the scatter with no clock penalty --
# which would make the right fix per-host rather than global.
#
# Please send back the WHOLE output, plus `lscpu | head -20`.

using InteractiveUtils, Printf

# ---- the two kernels, copied verbatim in shape from src/search.jl ------------

# v0: what ships today.  `BJ` is a compile-time constant, so the `b` loop fully
# unrolls and LLVM is free to vectorise `i` instead -- which is where the
# scatter comes from.
@inline function tr_scatter!(tile::Vector{T}, profs::AbstractMatrix{<:AbstractFloat},
                             j0::Int, nbins::Int, ::Val{B}, ::Val{BJ}) where {T,B,BJ}
    @inbounds for b0 in 0:BJ:(B - 1), i in 1:nbins
        o = (i - 1) * B + b0
        @simd for b in 1:BJ
            tile[o + b] = T(profs[i, j0 + b0 + b])
        end
    end
end

# v6: gather BJ values into an NTuple and store it as ONE aggregate.  LLVM
# cannot turn a single 32-byte store back into a scatter.
@inline function tr_aggregate!(tile::Vector{T}, profs::AbstractMatrix{<:AbstractFloat},
                               j0::Int, nbins::Int, ::Val{B}, ::Val{BJ}) where {T,B,BJ}
    GC.@preserve tile begin
        p = pointer(tile)
        @inbounds for b0 in 0:BJ:(B - 1), i in 1:nbins
            v = ntuple(b -> T(@inbounds profs[i, j0 + b0 + b]), Val(BJ))
            unsafe_store!(Ptr{NTuple{BJ,T}}(p + (((i - 1) * B + b0) * sizeof(T))), v)
        end
    end
end

# ---- codegen inspection -----------------------------------------------------

function codegen(f, tt)
    io = IOBuffer()
    code_native(io, f, tt; syntax = :intel, dump_module = false)
    s = String(take!(io))
    (zmm     = count(!isnothing, eachmatch(r"\bzmm\d+", s)),
     ymm     = count(!isnothing, eachmatch(r"\bymm\d+", s)),
     scatter = count(!isnothing, eachmatch(r"vscatter", s)),
     gather  = count(!isnothing, eachmatch(r"vgather", s)))
end

# ---- timing -----------------------------------------------------------------

const B     = 128        # _BC_BATCH
const BJ    = 8          # _BC_TR_BJ
const NBINS = 120        # profile length at decimation k=1

function bench(f, tile, profs, n)
    ncol = size(profs, 2)
    f(tile, profs, 0, NBINS, Val(B), Val(BJ))              # warm
    t = time_ns()
    for r in 1:n
        f(tile, profs, ((r - 1) * B) % (ncol - B), NBINS, Val(B), Val(BJ))
    end
    (time_ns() - t) / n / 1000                              # µs per call
end

# A fixed scalar workload with no vector code in it at all.  Timed on its own,
# and then ALTERNATED with each transpose kernel.  If the machine downclocks for
# AVX-512, this scalar loop gets slower next to the scatter kernel and not next
# to the aggregate one -- which measures the licence effect with no `perf`.
@noinline function scalar_work(x::Float64, n::Int)
    a = x
    for _ in 1:n
        a = muladd(a, 1.0000001, 1.0e-9)   # serial dependency: latency-bound
    end
    return a
end

function alternating(f, tile, profs, reps, scal)
    ncol = size(profs, 2)
    f(tile, profs, 0, NBINS, Val(B), Val(BJ)); scalar_work(1.0, scal)
    tsc = 0.0
    for r in 1:reps
        f(tile, profs, ((r - 1) * B) % (ncol - B), NBINS, Val(B), Val(BJ))
        t0 = time_ns()
        scalar_work(1.0, scal)
        tsc += (time_ns() - t0)
    end
    tsc / reps / 1000                                       # µs per scalar block
end

# ---- report -----------------------------------------------------------------

target = Base.JLOptions().cpu_target == C_NULL ? "native" :
         unsafe_string(Base.JLOptions().cpu_target)
println("="^74)
println("CoherentSearch AVX-512 scatter / downclocking probe")
println("julia ", VERSION, "   cpu-target = ", target)
try
    println(strip(read(ignorestatus(pipeline(`lscpu`, `grep -m1 "Model name"`)), String)))
    feats = read(ignorestatus(pipeline(`lscpu`, `grep -oE "avx512[a-z]+"`)), String)
    fs = sort(unique(split(strip(feats))))
    println("avx512 features: ", isempty(fs) ? "(NONE)" : join(fs, " "))
catch
    println("(lscpu unavailable)")
end
println("="^74)

println("\n--- 1. code generation (does the shipped kernel emit a 512-bit scatter?)")
@printf("%-40s %6s %6s %8s %8s\n", "kernel", "zmm", "ymm", "scatter", "gather")
for (nm, f) in (("tr_scatter!   shipped", tr_scatter!),
                ("tr_aggregate! candidate fix", tr_aggregate!))
    for P in (Float32, Float64)
        r = codegen(f, Tuple{Vector{Float32},Matrix{P},Int,Int,Val{B},Val{BJ}})
        @printf("%-40s %6d %6d %8d %8d   %s\n", "$nm, $(P) profs",
                r.zmm, r.ymm, r.scatter, r.gather,
                r.scatter > 0 ? "<<< SCATTER" : (r.zmm > 0 ? "(512-bit)" : ""))
    end
end

println("\n--- 2. kernel throughput in isolation (µs per call, lower is better)")
@printf("%-16s %14s %14s\n", "profs type", "shipped", "aggregate")
results = Dict{DataType,Tuple{Float64,Float64}}()
for P in (Float32, Float64)
    profs = rand(P, NBINS, 4096)
    tile  = Vector{Float32}(undef, B * NBINS)
    fill!(tile, 0f0); tr_scatter!(tile, profs, 0, NBINS, Val(B), Val(BJ))
    ref = copy(tile)
    fill!(tile, 0f0); tr_aggregate!(tile, profs, 0, NBINS, Val(B), Val(BJ))
    tile == ref || println("  *** MISMATCH between the two kernels for $P ***")
    a = bench(tr_scatter!,   tile, profs, 3000)
    b = bench(tr_aggregate!, tile, profs, 3000)
    results[P] = (a, b)
    @printf("%-16s %14.2f %14.2f     %.2fx\n", string(P), a, b, a / b)
end

println("\n--- 3. THE KEY TEST: does the scatter slow down UNRELATED scalar code?")
println("    A fixed scalar loop (no vector instructions), timed alone and then")
println("    alternated with each transpose kernel.  A licence-based downclock")
println("    shows up as the scalar block getting slower next to the scatter.")
const SCAL = 200_000
begin
    profs32 = rand(Float32, NBINS, 4096)
    tile    = Vector{Float32}(undef, B * NBINS)
    scalar_work(1.0, SCAL)
    base = minimum((t = time_ns(); scalar_work(1.0, SCAL); (time_ns() - t) / 1000)
                   for _ in 1:20)
    alt_s = alternating(tr_scatter!,   tile, profs32, 2000, SCAL)
    alt_a = alternating(tr_aggregate!, tile, profs32, 2000, SCAL)
    @printf("    scalar block alone                     %9.1f µs   (1.000x)\n", base)
    @printf("    scalar block next to SHIPPED   (f32)   %9.1f µs   (%.3fx)\n", alt_s, alt_s / base)
    @printf("    scalar block next to AGGREGATE (f32)   %9.1f µs   (%.3fx)\n", alt_a, alt_a / base)
    println()
    if alt_s / base > 1.10 && alt_s / alt_a > 1.08
        @printf("    => DOWNCLOCKING PRESENT: the scatter costs unrelated code %.0f%%.\n",
                100 * (alt_s / alt_a - 1))
    elseif codegen(tr_scatter!, Tuple{Vector{Float32},Matrix{Float32},Int,Int,Val{B},Val{BJ}}).scatter > 0
        println("    => scatter is emitted but does NOT slow neighbouring scalar code")
        println("       on this host (no Skylake-style licensing).")
    else
        println("    => no scatter emitted on this host; nothing to see.")
    end
end

println("\n--- 4. perf turbo-licence counters (Intel only; skipped if unavailable)")
try
    ev = "core_power.lvl0_turbo_license,core_power.lvl1_turbo_license,core_power.lvl2_turbo_license,cycles,ref-cycles"
    out = read(pipeline(`perf stat -e $ev -x, -- julia -e "1+1"`, stderr = devnull), String)
    println("    perf works here; re-run the real search under:")
    println("      perf stat -e $ev -- julia --project=. -t 1 bin/coherent_search.jl \\")
    println("          --threshold 6 --lofreq 0.1 --hifreq 33.3333 --precision f32 FILE.fft")
catch
    println("    perf not available / not permitted -- section 3 is the substitute.")
end
println("\n", "="^74)
