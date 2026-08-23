# Standalone floor for the boxcar gate's tile transpose.
#
#     julia --project=bench -t 1 bench/transpose_bench.jl
#
# Three questions, in order:
#   (a) what copy bandwidth does this core actually give at this footprint, so
#       the transpose's GB/s has something to be a fraction OF;
#   (b) how does the shipped per-tile `_bc_transpose!` compare with a
#       cache-line-filling blocked variant and with FFTW's guru rank-0 r2r
#       "transpose" plan (the PRESTO trick, ~/src/presto/tests/test_transpose.c:
#       howmany_rank=2, rank=0 — a pure strided copy that FFTW cache-blocks);
#   (c) the same in `Float32`, since `--precision f32` removes the narrowing and
#       halves the read side.
#
# Treat every number here as an UPPER bound on the in-situ win: this benchmark
# re-runs on hot data, while the search reads `profs` once, immediately after the
# transform that wrote it.  Decide in `bench/metric_bench.jl` / a real search.

using BenchmarkTools, FFTW, Printf
using CoherentSearch
const CS = CoherentSearch
const B  = CS._BC_BATCH

# ---------------------------------------------------------------------------
# FFTW guru rank-0 r2r transpose
# ---------------------------------------------------------------------------
const L64 = FFTW.libfftw3
const L32 = FFTW.libfftw3f
for (T, lib, pf, xf) in ((Float64, :L64, :fftw_plan_guru64_r2r,  :fftw_execute_r2r),
                         (Float32, :L32, :fftwf_plan_guru64_r2r, :fftwf_execute_r2r))
    @eval begin
        # Move A[i,j] (n1 x n2, input strides is1/is2) to B[j,i] (output strides
        # os1/os2).  rank=0 + howmany_rank=2 means "no transform, just this
        # strided copy"; FFTW plans a blocked kernel for it.
        function guru_transpose_plan(in::AbstractArray{$T}, out::AbstractArray{$T},
                                     n1, is1, os1, n2, is2, os2, flags::UInt32)
            hm = reshape(Int[n1, is1, os1, n2, is2, os2], 3, 2)
            p = ccall(($(QuoteNode(pf)), $lib), Ptr{Cvoid},
                      (Int32, Ptr{Int}, Int32, Ptr{Int}, Ptr{$T}, Ptr{$T}, Ptr{Int32}, UInt32),
                      0, C_NULL, 2, hm, in, out, C_NULL, flags)
            p == C_NULL && error("FFTW refused the guru r2r transpose plan")
            p
        end
        guru_exec!(p::Ptr{Cvoid}, in::AbstractArray{$T}, out::AbstractArray{$T}) =
            ccall(($(QuoteNode(xf)), $lib), Cvoid,
                  (Ptr{Cvoid}, Ptr{$T}, Ptr{$T}), p, in, out)
    end
end
plan_profsT(profs::Matrix{T}, profsT::Matrix{T}, flags) where {T} =
    guru_transpose_plan(profs, profsT, size(profs,1), 1, size(profs,2),
                        size(profs,2), size(profs,1), 1, flags)

# ---------------------------------------------------------------------------
# transpose variants, all over a FULL chunk of `Nprof` profiles
# ---------------------------------------------------------------------------
function prod_transpose!(tile, profs, nbins, n)      # what ships
    j0 = 0
    while j0 + B <= n
        CS._bc_transpose!(tile, profs, j0, nbins, Val(B))
        j0 += B
    end
end

# 16 Float32 = 64 B = exactly one cache line, so a block row's writes fill a line
# instead of dirtying 16 of them.  This is the specific shape the transpose brief
# proposed; `BL` is swept because the claim is about the line, not the blocking.
@inline function _bc_transpose_blocked!(tile::Vector{T}, profs::AbstractMatrix,
                                        j0::Int, nbins::Int, ::Val{B}, ::Val{BL}) where {T,B,BL}
    @inbounds for i0 in 0:BL:(nbins - 1)
        imax = min(i0 + BL, nbins)
        for b0 in 0:BL:(B - 1)
            for i in (i0 + 1):imax
                o = (i - 1) * B + b0
                @simd for b in 1:BL
                    tile[o + b] = T(profs[i, j0 + b0 + b])
                end
            end
        end
    end
end
function blocked_transpose!(tile, profs, nbins, n, ::Val{BL}) where {BL}
    j0 = 0
    while j0 + B <= n
        _bc_transpose_blocked!(tile, profs, j0, nbins, Val(B), Val(BL))
        j0 += B
    end
end

const NPROF = 2048
const NBINS = (120, 60, 40, 30, 24, 20)     # the default k = 1…6 ladder

function bandwidth_table()
    println("achievable single-core bandwidth, GB/s (copy, and the f64->f32")
    println("narrowing the shipped transpose actually performs):")
    for mb in (1, 2, 4, 8, 32)
        nb = mb * 1024 * 1024; n = nb ÷ 8
        a = rand(Float64, n); b64 = similar(a); b32 = Vector{Float32}(undef, n)
        tc = @belapsed copyto!($b64, $a)
        tn = @belapsed copyto!($b32, $a)
        @printf("  %3d MB : copy %5.1f   narrow %5.1f\n", mb, 2nb/tc/1e9, 1.5nb/tn/1e9)
    end
end

function transpose_table(::Type{P}) where {P}
    T = CS._BC_TILE
    @printf("\n%s profiles -> %s tile, Nprof=%d, B=%d, per full chunk:\n", P, T, NPROF, B)
    @printf("  %-5s %9s %9s %9s %9s | %9s %9s\n",
            "nbins", "prod", "blk8", "blk16", "blk32", "fftw", "fftw GB/s")
    tot = zeros(6)
    for (r, nbins) in enumerate(NBINS)
        profs  = rand(P, nbins, NPROF)
        tile   = Vector{T}(undef, B * nbins)
        profsT = Matrix{P}(undef, NPROF, nbins)
        pl = plan_profsT(profs, profsT, FFTW.MEASURE)
        profs .= rand(P, nbins, NPROF)          # MEASURE clobbers the arrays
        guru_exec!(pl, profs, profsT)
        @assert profsT == permutedims(profs)
        tp = @belapsed prod_transpose!($tile, $profs, $nbins, $NPROF)
        t8 = @belapsed blocked_transpose!($tile, $profs, $nbins, $NPROF, Val(8))
        t16= @belapsed blocked_transpose!($tile, $profs, $nbins, $NPROF, Val(16))
        t32= @belapsed blocked_transpose!($tile, $profs, $nbins, $NPROF, Val(32))
        tf = @belapsed guru_exec!($pl, $profs, $profsT)
        tot .+= (tp, t8, t16, t32, tf, 0.0)
        @printf("  %-5d %7.1fus %7.1fus %7.1fus %7.1fus | %7.1fus %7.1f\n",
                nbins, 1e6tp, 1e6t8, 1e6t16, 1e6t32, 1e6tf,
                2 * NPROF * nbins * sizeof(P) / tf / 1e9)
    end
    @printf("  %-5s %7.1fus %7.1fus %7.1fus %7.1fus | %7.1fus   (fftw %.2fx prod)\n",
            "all", 1e6tot[1], 1e6tot[2], 1e6tot[3], 1e6tot[4], 1e6tot[5], tot[1]/tot[5])
end

bandwidth_table()
transpose_table(Float64)
transpose_table(Float32)
