# Split `decim-brfft` — the largest phase on fitzroy — into the two things it
# could be, on the real `Workspace`/`DecimBuf` buffers holding a genuine chunk.
#
#     julia --project=bench -t 1 bench/decim_brfft_bench.jl [FILE.fft] [freq_Hz] [maxdecim]
#
# The in-situ phase timers bill `decim-brfft` per `k` (slots 10..), but not *why*
# each costs what it does.  Two candidate causes, and they call for opposite
# fixes:
#
#   (a) **the stride.**  `db.src` is `view(ftprofs, 1:k:(Hk*k+1), :)`, so the
#       transform reads every `k`-th element of each column.  Column extent is
#       `(nharms+1)*sizeof(Complex{P})` = 976 B (`:f64`), ~15 cache lines, and a
#       stride-`k` read still touches nearly all of them to use `1/k` of the
#       data.  Measured against the same transform on a *contiguous* copy of the
#       identical stack (the gather done OUTSIDE the timed region), the
#       difference is what the stride costs — i.e. the headroom a layout change
#       could recover, NOT a proposal to reinstate the gather, which was measured
#       slower than this in 51ccdf6 precisely because it paid the copy.
#
#   (b) **small-transform efficiency.**  Output lengths are 2Hk = 60,40,30,24,20
#       against the base pass's 120.  Per output bin, a short batched c2r is
#       simply worse.  Normalising each `k` by its own output-bin count separates
#       this from (a).
#
# Reports both, in BOTH precisions, because `decim-brfft`'s response to `:f32`
# changes sign with thread count (+10.9% at `-t 1`, −20.2% at `-t 16`).
#
# **This is an isolated kernel benchmark and re-reads its input hot.**  In the
# search, `decim-brfft` reads `ftprofs` right after the interpolator wrote it and
# while five other passes compete for L3 — three optimisation calls in this
# project have been decided by an isolated micro-benchmark and come out backwards
# in situ.  Use this to understand the *shape* of the cost; use
# `bench/precision_ab.jl`'s phase table to score any change.

using CoherentSearch
using BenchmarkTools
using LinearAlgebra: mul!
using FFTW: plan_brfft
using Printf
const CS = CoherentSearch

const FILE   = length(ARGS) >= 1 ? ARGS[1] : "PM0063_034C1_DM445.0_red.fft"
const FREQ   = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 10.0
const MAXDEC = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 6
const NHARMS = 60
const NPROF  = 2048

us(b) = minimum(b).time / 1000

function run_precision(ft, prec::Symbol)
    P = prec === :f32 ? Float32 : Float64
    params = SearchParams(nharms=NHARMS, threshold=6.0, precision=prec,
                          decimations=decimation_set(NHARMS, MAXDEC))
    rstart = FREQ * ft.T
    ws = CS.Workspace(params, NPROF)
    dplans = CS.build_direct_plans(params, rstart)
    # One genuine chunk into ws.ftprofs (and hence every db.src view).
    fill!(ws.ftprofs, 0)
    for dp in dplans
        CS.fill_harmonic_row_direct!(ws, dp, ft, params, 0, NPROF)
    end

    csz = sizeof(Complex{P})
    @printf("\n=== precision %s  (Complex{%s} = %d B, ftprofs = %.2f MB) ===\n",
            prec, P, csz, (NHARMS + 1) * NPROF * csz / 2^20)
    @printf("  %-3s %5s %7s %9s %9s %7s %10s %10s %9s\n",
            "k", "nbins", "outbins", "strided", "contig", "stride", "ns/outbin", "ns/outbin",
            "in-touch")
    @printf("  %-3s %5s %7s %9s %9s %7s %10s %10s %9s\n",
            "", "", "(M)", "us", "us", "cost", "strided", "contig", "MB")

    # k = 1 (the base pass) as the reference point: contiguous by construction.
    rows = Tuple{Int,Int,Float64,Float64,Float64,Float64}[]
    t1 = us(@benchmark mul!($(ws.profs), $(ws.brfftplan), $(ws.ftprofs)) evals=1 samples=200)
    ob1 = 2 * NHARMS * NPROF
    @printf("  %-3d %5d %7.2f %9.1f %9s %7s %10.3f %10s %9.2f\n",
            1, 2NHARMS, ob1 / 1e6, t1, "-", "-", 1000 * t1 / ob1, "-",
            (NHARMS + 1) * NPROF * csz / 2^20)
    push!(rows, (1, 2NHARMS, t1, t1, ob1, (NHARMS + 1) * NPROF * csz / 2^20))

    for db in ws.decims
        k, Hk = db.k, db.Hk
        nb = 2Hk
        outbins = nb * NPROF
        # (a) production: stride-k view of ftprofs.
        ts = us(@benchmark mul!($(db.dprofs), $(db.brfftplan), $(db.src)) evals=1 samples=200)
        # (b) the identical stack, contiguous.  The copy is OUTSIDE the timing.
        srcC = Matrix{Complex{P}}(undef, Hk + 1, NPROF)
        copyto!(srcC, db.src)
        dstC = Matrix{P}(undef, nb, NPROF)
        planC = CS.with_plan_rigor(CS.plan_rigor()) do
            plan_brfft(srcC, nb, 1; flags=CS.plan_rigor())
        end
        copyto!(srcC, db.src)          # MEASURE overwrote it
        tc = us(@benchmark mul!($dstC, $planC, $srcC) evals=1 samples=200)
        # Input actually touched: the strided read pulls whole cache lines out of
        # the full column extent, so it is ~the whole ftprofs; contiguous is just
        # the stack.
        in_strided = (NHARMS + 1) * NPROF * csz / 2^20
        in_contig  = (Hk + 1) * NPROF * csz / 2^20
        @printf("  %-3d %5d %7.2f %9.1f %9.1f %6.2fx %10.3f %10.3f %9.2f\n",
                k, nb, outbins / 1e6, ts, tc, ts / tc,
                1000 * ts / outbins, 1000 * tc / outbins, in_strided)
        push!(rows, (k, nb, ts, tc, outbins, in_contig))
    end

    tot_s = sum(r[3] for r in rows if r[1] > 1)
    tot_c = sum(r[4] for r in rows if r[1] > 1)
    @printf("  %-3s %5s %7s %9.1f %9.1f %6.2fx\n", "sum", "", "", tot_s, tot_c, tot_s / tot_c)
    @printf("  base(k=1) ns/outbin = %.3f;  decimated range = %.3f .. %.3f (strided)\n",
            1000 * rows[1][3] / rows[1][5],
            minimum(1000 * r[3] / r[5] for r in rows if r[1] > 1),
            maximum(1000 * r[3] / r[5] for r in rows if r[1] > 1))
    return (; tot_s, tot_c, rows)
end

function main()
    ft = FFTFile(FILE)
    @printf("decim-brfft bench — %s  f=%.4g Hz  nharms=%d  Nprof=%d  maxdecim=%d\n",
            FILE, FREQ, NHARMS, NPROF, MAXDEC)
    println("  'stride cost' = strided / contiguous, same data, copy excluded.")
    r64 = run_precision(ft, :f64)
    r32 = run_precision(ft, :f32)
    println()
    @printf("  decimated total: f64 %.1f us, f32 %.1f us  (f32 = %.3fx of f64, strided)\n",
            r64.tot_s, r32.tot_s, r32.tot_s / r64.tot_s)
    @printf("  contiguous:      f64 %.1f us, f32 %.1f us  (f32 = %.3fx of f64)\n",
            r64.tot_c, r32.tot_c, r32.tot_c / r64.tot_c)
end

main()

# ---------------------------------------------------------------------------
# Can anything beat the stride?
#
# The characterisation above says the strided read is the whole excess, and that
# a contiguous transform of the same stack is 1.47x (`:f64`) / 1.22x (`:f32`)
# cheaper.  That is the BUDGET: any scheme that makes the input contiguous has to
# produce it for less than the difference, or it loses — which is exactly what
# happened to the per-`k` gather removed in 51ccdf6.
#
# Three ways to produce it, timed against that budget:
#   naive    — `copyto!(srcC, db.src)` per k, i.e. the thing 51ccdf6 deleted.
#   blocked  — same, but with the profile axis blocked, the loop shape that was
#              worth 3.5x on `_bc_transpose!` on this machine (see `_BC_TR_BJ`).
#   fused    — ONE pass over each ftprofs column writing all five stacks at once.
#              Untried.  Each column is ~15 cache lines and every decimation
#              wants a subset of them, so read once and every line is fully used,
#              instead of five passes each pulling all 15 to use 1/k.

function gather_naive!(dsts, srcs)
    @inbounds for t in eachindex(dsts)
        copyto!(dsts[t], srcs[t])
    end
end

function gather_blocked!(dsts, ftprofs, ks, BJ::Int)
    n = size(ftprofs, 2)
    @inbounds for t in eachindex(dsts)
        d = dsts[t]; k = ks[t]; H1 = size(d, 1)
        for j0 in 1:BJ:n
            jhi = min(j0 + BJ - 1, n)
            for i in 1:H1
                r = (i - 1) * k + 1
                @simd for j in j0:jhi
                    d[i, j] = ftprofs[r, j]
                end
            end
        end
    end
end

# One pass over the columns; every decimated stack written from the same lines.
function gather_fused!(dsts, ftprofs, ks)
    n = size(ftprofs, 2)
    @inbounds for j in 1:n
        for t in eachindex(dsts)
            d = dsts[t]; k = ks[t]
            @simd for i in 1:size(d, 1)
                d[i, j] = ftprofs[(i - 1) * k + 1, j]
            end
        end
    end
end

function gather_study(ft, prec::Symbol)
    P = prec === :f32 ? Float32 : Float64
    params = SearchParams(nharms=NHARMS, threshold=6.0, precision=prec,
                          decimations=decimation_set(NHARMS, MAXDEC))
    ws = CS.Workspace(params, NPROF)
    dplans = CS.build_direct_plans(params, FREQ * ft.T)
    fill!(ws.ftprofs, 0)
    for dp in dplans
        CS.fill_harmonic_row_direct!(ws, dp, ft, params, 0, NPROF)
    end
    ks   = [db.k for db in ws.decims]
    srcs = [db.src for db in ws.decims]
    dsts = [Matrix{Complex{P}}(undef, db.Hk + 1, NPROF) for db in ws.decims]
    ftp  = ws.ftprofs

    t_naive = us(@benchmark gather_naive!($dsts, $srcs) evals=1 samples=200)
    t_fused = us(@benchmark gather_fused!($dsts, $ftp, $ks) evals=1 samples=200)
    blocked = [(BJ, us(@benchmark gather_blocked!($dsts, $ftp, $ks, $BJ) evals=1 samples=200))
               for BJ in (4, 8, 16, 32, 64, 128)]
    # correctness: every scheme must reproduce the view
    gather_fused!(dsts, ftp, ks)
    ok = all(dsts[t] == Array(srcs[t]) for t in eachindex(dsts))
    return (; t_naive, t_fused, blocked, ok)
end

function gather_main()
    ft = FFTFile(FILE)
    println("\n" * "="^78)
    println("Can anything beat the stride?  (gather cost vs the strided-minus-contiguous budget)")
    for prec in (:f64, :f32)
        r = gather_study(ft, prec)
        @printf("\n  %s   (gathers verified against the strided view: %s)\n",
                prec, r.ok ? "exact" : "MISMATCH!")
        @printf("    naive copyto! per k : %8.1f us\n", r.t_naive)
        @printf("    fused single pass   : %8.1f us\n", r.t_fused)
        for (BJ, t) in r.blocked
            @printf("    blocked BJ=%-3d      : %8.1f us\n", BJ, t)
        end
    end
end

gather_main()

# ---------------------------------------------------------------------------
# Fusing the stack writes into `fill_harmonic_row_direct!`
#
# The idea: the interpolator already holds harmonic `h`'s value in registers when
# it stores `ftprofs[h+1, j]`, and `h` belongs to decimated stack `k` whenever
# `h % k == 0` — so the stacks could be filled with ZERO extra reads, turning the
# 269 us fused gather into just its write traffic.
#
# The catch, and why this needs measuring rather than assuming: `_group_store!`
# writes a *row* of a column-major matrix, so its stores are already strided by
# `(nharms+1)*sizeof(Complex{P})` = 976 B (`:f64`).  The stacks are the same
# shape, so the fused stores are strided too — `(Hk+1)*16` = 496 B at `k=2` — and
# every one of them touches its own cache line.  87 extra harmonic-rows over the
# five stacks against the interpolator's own 60 is 1.45x more strided stores.
#
# `stackwrite_cost!` is the idea's BEST case: pure scatter stores of a register
# value, in the interpolator's loop order, no reads at all.  If that alone
# exceeds the strided-minus-contiguous budget, the fusion cannot win however it
# is written, and no src change is needed to know it.
function stackwrite_cost!(dsts, ks, nharms::Int, Nprof::Int, val)
    @inbounds for h in 1:nharms
        for t in eachindex(dsts)
            k = ks[t]
            h % k == 0 || continue
            d = dsts[t]
            i = h ÷ k + 1
            @simd for j in 1:Nprof
                d[i, j] = val
            end
        end
    end
end

# The same stores in the opposite nesting — column-major, all stacks per column —
# which is the fused *gather*'s order, for reference.
function stackwrite_colmajor!(dsts, ks, nharms::Int, Nprof::Int, val)
    @inbounds for j in 1:Nprof
        for t in eachindex(dsts)
            d = dsts[t]
            @simd for i in 1:size(d, 1)
                d[i, j] = val
            end
        end
    end
end

function fusion_study(ft, prec::Symbol, budget::Float64)
    P = prec === :f32 ? Float32 : Float64
    params = SearchParams(nharms=NHARMS, threshold=6.0, precision=prec,
                          decimations=decimation_set(NHARMS, MAXDEC))
    ws = CS.Workspace(params, NPROF)
    ks   = [db.k for db in ws.decims]
    dsts = [Matrix{Complex{P}}(undef, db.Hk + 1, NPROF) for db in ws.decims]
    val  = one(Complex{P})
    nrows = sum(count(h -> h % k == 0, 1:NHARMS) for k in ks)
    t_interp_order = us(@benchmark stackwrite_cost!($dsts, $ks, $NHARMS, $NPROF, $val) evals=1 samples=200)
    t_col_order    = us(@benchmark stackwrite_colmajor!($dsts, $ks, $NHARMS, $NPROF, $val) evals=1 samples=200)
    @printf("\n  %s  (%d extra strided harmonic-rows against the interpolator's own %d)\n",
            prec, nrows, NHARMS)
    @printf("    stack writes, INTERPOLATOR order (harmonic-major) : %8.1f us   %s budget %.1f us\n",
            t_interp_order, t_interp_order < budget ? "UNDER" : "OVER ", budget)
    @printf("    stack writes, column-major order (for reference)  : %8.1f us\n", t_col_order)
end

function fusion_main()
    ft = FFTFile(FILE)
    println("\n" * "="^78)
    println("Fusing the stack writes into the interpolator: pure-write cost (best case)")
    println("  Budget = strided minus contiguous, i.e. what the whole scheme has to beat.")
    fusion_study(ft, :f64, 312.2)
    fusion_study(ft, :f32, 127.5)
end

fusion_main()
