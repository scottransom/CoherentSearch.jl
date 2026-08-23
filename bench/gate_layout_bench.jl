# The boxcar gate under two profile layouts, end to end (transpose + prefix sum
# + width scan), per chunk, per fold depth.
#
#     julia --project=bench -t 1 bench/gate_layout_bench.jl
#
# TODAY:  `profs` is (nbins, Nprof).  `_bc_transpose!` gathers a B-profile tile
#         into (B, nbins) `Float32` — a stride-`nbins` READ gather, contiguous
#         512 B writes — and `_bc_scan_batch!` reads the tile back.
#
# PROFST: one whole-chunk transpose into `profsT` (Nprof, nbins), same eltype as
#         `profs`, done by FFTW's guru rank-0 r2r plan (the PRESTO trick, see
#         ~/src/presto/tests/test_transpose.c) or by an 8x8 blocked loop.  A tile
#         is then just `profsT[j0+1:j0+B, :]`, contiguous in the batch axis, so
#         there is NO per-tile transpose and NO tile buffer: the prefix sum reads
#         `profsT` directly and narrows to `Float32` on the way in, which is a
#         pass it was making anyway.
#
# The point of measuring the whole gate rather than the transpose alone is that
# PROFST moves MORE bytes in the scan (it reads `profs`-width data instead of a
# narrowed tile) and fewer in the transpose, so the transpose column on its own
# does not decide it.

using BenchmarkTools, FFTW, Printf
using CoherentSearch
const CS = CoherentSearch
const B  = CS._BC_BATCH
const T  = CS._BC_TILE

const L64 = FFTW.libfftw3
const L32 = FFTW.libfftw3f
for (P, lib, pf, xf) in ((Float64, :L64, :fftw_plan_guru64_r2r,  :fftw_execute_r2r),
                         (Float32, :L32, :fftwf_plan_guru64_r2r, :fftwf_execute_r2r))
    @eval begin
        function guru_transpose_plan(in::AbstractArray{$P}, out::AbstractArray{$P},
                                     n1, is1, os1, n2, is2, os2, flags::UInt32)
            hm = reshape(Int[n1, is1, os1, n2, is2, os2], 3, 2)
            p = ccall(($(QuoteNode(pf)), $lib), Ptr{Cvoid},
                      (Int32, Ptr{Int}, Int32, Ptr{Int}, Ptr{$P}, Ptr{$P}, Ptr{Int32}, UInt32),
                      0, C_NULL, 2, hm, in, out, C_NULL, flags)
            p == C_NULL && error("FFTW refused the guru r2r transpose plan")
            p
        end
        guru_exec!(p::Ptr{Cvoid}, in::AbstractArray{$P}, out::AbstractArray{$P}) =
            ccall(($(QuoteNode(xf)), $lib), Cvoid,
                  (Ptr{Cvoid}, Ptr{$P}, Ptr{$P}), p, in, out)
        guru_exec_ptr!(p::Ptr{Cvoid}, in::Ptr{$P}, out::Ptr{$P}, ::Type{$P}) =
            ccall(($(QuoteNode(xf)), $lib), Cvoid,
                  (Ptr{Cvoid}, Ptr{$P}, Ptr{$P}), p, in, out)
    end
end
plan_profsT(profs::Matrix{P}, profsT::Matrix{P}, flags) where {P} =
    guru_transpose_plan(profs, profsT, size(profs,1), 1, size(profs,2),
                        size(profs,2), size(profs,1), 1, flags)

# Per-TILE variant: transpose only the B columns of one tile, (nbins, B) -> (B, nbins).
# Columns j0+1..j0+B of a column-major `profs` are contiguous, so this is the same
# plan executed at a pointer offset — and it keeps the whole move inside L2, at the
# cost of a `Nprof x nbins` buffer it does not need (the tile buffer suffices).
plan_tileT(profs::Matrix{P}, tileP::Vector{P}, nbins, ::Val{B}, flags) where {P,B} =
    guru_transpose_plan(profs, tileP, nbins, 1, B, B, nbins, 1, flags)

function tile_transpose_all!(p::Ptr{Cvoid}, tileP::Vector{P}, profs::Matrix{P},
                             nbins::Int, n::Int, bb, widths, invsig, ::Val{B}) where {P,B}
    j0 = 0
    GC.@preserve profs tileP while j0 + B <= n
        src = pointer(profs, j0 * nbins + 1)
        guru_exec_ptr!(p, src, pointer(tileP), P)
        _bc_scan_profsT!(bb, tileP, 0, B, widths, nbins, invsig, Val(B))
        j0 += B
    end
end

# 8x8 blocked transpose into profsT, as a plan-free fallback.
function blocked_profsT!(profsT::Matrix{P}, profs::Matrix{P}) where {P}
    nbins, n = size(profs)
    @inbounds for i0 in 0:8:(nbins - 1), j0 in 0:8:(n - 1)
        imax = min(i0 + 8, nbins); jmax = min(j0 + 8, n)
        for i in (i0+1):imax, j in (j0+1):jmax
            profsT[j, i] = profs[i, j]
        end
    end
    profsT
end

# The shipped prefix+width scan, but reading a (ld, nbins) profile-major array
# at column offset `j0` instead of a materialised tile.  Same recurrence, same
# order; only the source of `tile[t + b]` changes, and the Float32 narrowing
# moves into a pass the kernel was already making.
@inline function _bc_scan_profsT!(bb, profsT::AbstractArray{P}, j0::Int, ld::Int,
                                  widths::Vector{Int}, nbins::Int, invsigma::T,
                                  ::Val{B}) where {P,T,B}
    psT = bb.psT; res = bb.res; mbuf = bb.mbuf
    wmax = widths[end]
    @inbounds for b in 1:B
        psT[b] = zero(T)
    end
    @inbounds for i in 1:(nbins + wmax)
        idx = i > nbins ? i - nbins : i
        o = (i - 1) * B
        t = (idx - 1) * ld + j0
        @simd for b in 1:B
            psT[o + B + b] = psT[o + b] + T(profsT[t + b])
        end
    end
    @inbounds for b in 1:B
        res[b] = T(-Inf)
    end
    @inbounds for w in widths
        invsw = invsigma / sqrt(T(w))
        wo = w * B
        @simd for b in 1:B
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
            c = mbuf[b] * invsw
            res[b] = ifelse(c > res[b], c, res[b])
        end
    end
end

function gate_today!(bb, profs, n, widths, nbins, invsig)
    j0 = 0
    while j0 + B <= n
        CS._bc_transpose!(bb.tile, profs, j0, nbins, Val(B))
        CS._bc_scan_batch!(bb, widths, nbins, invsig, Val(B))
        j0 += B
    end
end
function gate_profsT!(bb, profsT, n, widths, nbins, invsig)
    ld = size(profsT, 1); j0 = 0
    while j0 + B <= n
        _bc_scan_profsT!(bb, profsT, j0, ld, widths, nbins, invsig, Val(B))
        j0 += B
    end
end

const NPROF = 2048
const KS    = CS.decimation_set(60, 6)

function run(::Type{P}) where {P}
    params = SearchParams(nharms=60, decimations=KS)
    @printf("\n%s profiles, Nprof=%d, B=%d — per chunk\n", P, NPROF, B)
    @printf("  %-3s %6s | %8s %8s %8s | %8s %8s %8s | %8s %6s | %8s %6s\n",
            "k", "nbins", "transp", "scan", "today", "fftwT", "blkT", "scanT",
            "chunkT", "gain", "tileT", "gain")
    t_today = 0.0; t_fftw = 0.0; t_blk = 0.0; t_tile = 0.0
    for k in KS
        nbins  = 2 * fld(60, k)
        profs  = randn(P, nbins, NPROF)
        profsT = Matrix{P}(undef, NPROF, nbins)
        widths = CS.ladder_boxcar_widths(nbins, k, params)
        bb     = CS.BoxcarBatch(nbins, widths[end], NPROF)
        invsig = T(1.0)
        tileP = Vector{P}(undef, B * nbins)
        pl  = plan_profsT(profs, profsT, FFTW.PATIENT)
        ptl = plan_tileT(profs, tileP, nbins, Val(B), FFTW.PATIENT)
        profs .= randn(P, nbins, NPROF)          # PATIENT clobbers its arrays
        guru_exec!(pl, profs, profsT)
        @assert profsT == permutedims(profs)
        @assert blocked_profsT!(similar(profsT), profs) == profsT

        gate_today!(bb, profs, NPROF, widths, nbins, invsig); r1 = copy(bb.res)
        gate_profsT!(bb, profsT, NPROF, widths, nbins, invsig); r2 = copy(bb.res)
        @assert r1 == r2 "profsT gate disagrees with the shipped one"

        ttr = @belapsed CS._bc_transpose!($bb.tile, $profs, 0, $nbins, Val($B))
        ttr *= NPROF ÷ B
        tsc = @belapsed CS._bc_scan_batch!($bb, $widths, $nbins, $invsig, Val($B))
        tsc *= NPROF ÷ B
        tft = @belapsed guru_exec!($pl, $profs, $profsT)
        tbl = @belapsed blocked_profsT!($profsT, $profs)
        tsT = @belapsed gate_profsT!($bb, $profsT, $NPROF, $widths, $nbins, $invsig)
        tile_transpose_all!(ptl, tileP, profs, nbins, NPROF, bb, widths, invsig, Val(B))
        @assert bb.res == r2 "per-tile FFTW gate disagrees"
        ttile = @belapsed tile_transpose_all!($ptl, $tileP, $profs, $nbins, $NPROF, $bb, $widths, $invsig, Val($B))
        t_tile += ttile
        today = ttr + tsc
        t_today += today; t_fftw += tft + tsT; t_blk += tbl + tsT
        @printf("  %-3d %6d | %6.1fus %6.1fus %6.1fus | %6.1fus %6.1fus %6.1fus | %6.1fus %5.2fx | %6.1fus %5.2fx\n",
                k, nbins, 1e6ttr, 1e6tsc, 1e6today, 1e6tft, 1e6tbl, 1e6tsT,
                1e6*(tft + tsT), today/(tft + tsT), 1e6ttile, today/ttile)
    end
    @printf("  %-3s %6s | %6s %6s %6.1fus | %6s %6s %6s | %6.1fus %5.2fx | %6.1fus %5.2fx\n",
            "all", "", "", "", 1e6t_today, "", "", "", 1e6t_fftw, t_today/t_fftw,
            1e6t_tile, t_today/t_tile)
end

run(Float64)
run(Float32)
