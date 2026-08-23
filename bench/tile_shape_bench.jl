# Loop-shape sweep for the boxcar gate's per-tile transpose.
#
#     julia --project=bench -t 1 bench/tile_shape_bench.jl
#
# `_bc_transpose!` is one nest — `for i in 1:nbins, b in 1:B` — with a
# contiguous 512 B write per phase and a stride-`nbins` READ gather across the
# B profiles.  (The 2026-08-22 brief had this backwards and blamed write-scatter;
# the source comment has it right.)  This sweeps blockings of BOTH axes, plus the
# transposed nesting (contiguous read, scattered write), because the two hosts
# disagree about which is fastest by a factor of 2.6 and the output is
# byte-identical for every shape — so the only question is which nest to ship.

using BenchmarkTools, Printf
using CoherentSearch
const CS = CoherentSearch
const B  = CS._BC_BATCH
const T  = CS._BC_TILE

# `b` innermost (contiguous write, gathered read), blocked BI phases x BJ profiles.
@inline function tr_bi_bj!(tile::Vector{T}, profs::AbstractMatrix, j0::Int,
                           nbins::Int, ::Val{B}, ::Val{BI}, ::Val{BJ}) where {T,B,BI,BJ}
    @inbounds for i0 in 0:BI:(nbins - 1)
        imax = min(i0 + BI, nbins)
        for b0 in 0:BJ:(B - 1)
            for i in (i0 + 1):imax
                o = (i - 1) * B + b0
                @simd for b in 1:BJ
                    tile[o + b] = T(profs[i, j0 + b0 + b])
                end
            end
        end
    end
end

# `i` innermost (contiguous read, scattered write), blocked the same way.
@inline function tr_bj_bi!(tile::Vector{T}, profs::AbstractMatrix, j0::Int,
                           nbins::Int, ::Val{B}, ::Val{BI}, ::Val{BJ}) where {T,B,BI,BJ}
    @inbounds for b0 in 0:BJ:(B - 1)
        for i0 in 0:BI:(nbins - 1)
            imax = min(i0 + BI, nbins)
            for b in 1:BJ
                jb = j0 + b0 + b
                for i in (i0 + 1):imax
                    tile[(i - 1) * B + b0 + b] = T(profs[i, jb])
                end
            end
        end
    end
end

const NPROF = 2048
const NBINS = (120, 60, 40, 30, 24, 20)
const SHAPES = ((:bibj, 2, 4), (:bibj, 2, 8), (:bibj, 4, 4), (:bibj, 4, 8),
                (:bibj, 4, 16), (:bibj, 8, 8), (:bibj, 120, 4), (:bibj, 120, 8),
                (:bibj, 120, 16), (:bibj, 120, 32), (:bibj, 120, 64))

function sweep(::Type{P}) where {P}
    @printf("\n%s profiles -> %s tile, Nprof=%d, B=%d — µs per chunk, all folds summed\n",
            P, T, NPROF, B)
    ref = Dict{Int,Vector{T}}()
    tot_prod = 0.0
    times = zeros(length(SHAPES))
    for nbins in NBINS
        profs = rand(P, nbins, NPROF)
        tile  = Vector{T}(undef, B * nbins)
        CS._bc_transpose!(tile, profs, 0, nbins, Val(B)); ref[nbins] = copy(tile)
        tot_prod += (@belapsed CS._bc_transpose!($tile, $profs, 0, $nbins, Val($B))) * (NPROF ÷ B)
        for (s, (kind, bi, bj)) in enumerate(SHAPES)
            f = kind === :bibj ? tr_bi_bj! : tr_bj_bi!
            fill!(tile, 0); f(tile, profs, 0, nbins, Val(B), Val(bi), Val(bj))
            @assert tile == ref[nbins] "shape $kind $bi x $bj is wrong at nbins=$nbins"
            times[s] += (@belapsed $f($tile, $profs, 0, $nbins, Val($B), Val($bi), Val($bj))) * (NPROF ÷ B)
        end
    end
    @printf("  %-22s %8.1f  %5s\n", "shipped (nbins x B)", 1e6tot_prod, "1.00x")
    for (s, (kind, bi, bj)) in enumerate(SHAPES)
        @printf("  %-22s %8.1f  %5.2fx\n",
                "$(kind === :bibj ? "b-inner" : "i-inner") $(bi)x$(bj)",
                1e6times[s], tot_prod / times[s])
    end
end

sweep(Float64)
sweep(Float32)
