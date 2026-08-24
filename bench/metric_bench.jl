# Split the boxcar *metric* phases into their components, on the real
# `Workspace`/`DecimBuf` buffers holding a genuine chunk.
#
#     julia --project=bench -t 1 bench/metric_bench.jl [FILE.fft] [freq_Hz] [maxdecim]
#
# The in-situ phase timers bill `gate+metric` (k=1) and `decim-metric` (k=2…6),
# but `decim-metric` also contains `_block_sigma` and neither says how much of
# the gate is the transpose, the prefix sum, or the width×phase scan.  This
# measures each on the buffers the search uses, per decimation factor, so a
# proposed kernel change can be scored against the part it actually touches.

using CoherentSearch
using BenchmarkTools
using LinearAlgebra: mul!
using Printf
const CS = CoherentSearch

const FILE   = length(ARGS) >= 1 ? ARGS[1] : "PM0063_034C1_DM445.0_red.fft"
const FREQ   = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 10.0
const MAXDEC = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 6
const PREC   = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :f64
const NHARMS = 60
const NPROF  = 2048
const THRESH = 6.0

ft = FFTFile(FILE)
rstart = FREQ * ft.T
params = SearchParams(nharms=NHARMS, threshold=THRESH, precision=PREC,
                      decimations=decimation_set(NHARMS, MAXDEC))
ws = CS.Workspace(params, NPROF)
dplans = CS.build_direct_plans(params, rstart)
exactcut = THRESH - params.boxcar_gatemargin

# One genuine chunk into ws.ftprofs / ws.profs (and hence every db.dprofs).
function fillchunk!()
    fill!(ws.ftprofs, 0)
    for dp in dplans
        CS.fill_harmonic_row_direct!(ws, dp, ft, params, 0, NPROF)
    end
    mul!(ws.profs, ws.brfftplan, ws.ftprofs)
    for db in ws.decims
        mul!(db.dprofs, db.brfftplan, db.src)
    end
end

us(b) = minimum(b).time / 1000

# Transpose-only and scan-only halves of `_boxcar_gate!`, at the production tile.
function transpose_only!(bb, profs, n, nbins)
    B = CS._BC_BATCH; vb = Val(B); j0 = 0
    while j0 + B <= n
        CS._bc_transpose!(bb.tile, profs, j0, nbins, vb)
        j0 += B
    end
end
function scan_only!(bb, n, widths, nbins, invsig)
    B = CS._BC_BATCH; vb = Val(B); j0 = 0
    while j0 + B <= n
        CS._bc_scan_batch!(bb, widths, nbins, invsig, vb)
        j0 += B
    end
end

# Rows: one per fold depth, k=1 first.
struct Fold{P<:AbstractFloat}
    k::Int; nbins::Int; profs::Matrix{P}; widths::Vector{Int}
    bb::Any; psum::Vector{P}; bcsig::Vector{P}
end

function main()
    fillchunk!()
    folds = [Fold(1, 2NHARMS, ws.profs, ws.bcwidths, ws.bcbatch,
                  ws.bcpsum, ws.bcsig)]
    for db in ws.decims
        push!(folds, Fold(db.k, 2db.Hk, db.dprofs, db.bcwidths, db.bcbatch,
                          db.bcpsum, db.bcsig))
    end

    @printf("metric bench — %s  f=%.4g Hz  nharms=%d  Nprof=%d  maxdecim=%d  precision=%s  exactcut=%.2f\n\n",
            FILE, FREQ, NHARMS, NPROF, MAXDEC, PREC, exactcut)
    @printf("  %-3s %6s %-26s %8s %8s %8s %8s %8s %7s\n",
            "k", "nbins", "widths", "sigma", "transp", "scan", "gate", "metric", "rescan%")
    tot = zeros(5)
    for f in folds
        sig = CS._block_sigma(f.profs, f.nbins, NPROF, f.bcsig)
        P = eltype(f.profs)
        invs = one(P) / P(sig)          # the search's own type, not always Float64
        invt = CS._BC_TILE(invs)
        t_sig = us(@benchmark CS._block_sigma($(f.profs), $(f.nbins), $NPROF, $(f.bcsig)) evals=1 samples=100)
        t_tr  = us(@benchmark transpose_only!($(f.bb), $(f.profs), $NPROF, $(f.nbins)) evals=1 samples=100)
        t_sc  = us(@benchmark scan_only!($(f.bb), $NPROF, $(f.widths), $(f.nbins), $invt) evals=1 samples=100)
        t_g   = us(@benchmark CS._boxcar_gate!($(f.bb), $(f.profs), $NPROF, $(f.psum),
                                               $(f.widths), $(f.nbins), $invs) evals=1 samples=100)
        t_m   = us(@benchmark CS.boxcar_metrics!($(f.bb), $(f.profs), $NPROF,
                                                 $(f.psum), $(f.widths), $(f.nbins), $invs,
                                                 $exactcut) evals=1 samples=100)
        CS._boxcar_gate!(f.bb, f.profs, NPROF, f.psum, f.widths, f.nbins, invs)
        nres = count(>=(exactcut), @view f.bb.mvals[1:NPROF])
        @printf("  %-3d %6d %-26s %8.1f %8.1f %8.1f %8.1f %8.1f %6.2f%%\n",
                f.k, f.nbins, string(f.widths), t_sig, t_tr, t_sc, t_g, t_m,
                100 * nres / NPROF)
        tot .+= (t_sig, t_tr, t_sc, t_g, t_m)
    end
    @printf("  %-3s %6s %-26s %8.1f %8.1f %8.1f %8.1f %8.1f\n",
            "all", "", "", tot[1], tot[2], tot[3], tot[4], tot[5])
    @printf("\n  per-chunk metric total (sigma + metric) = %.0f µs\n", tot[1] + tot[5])
end

main()
