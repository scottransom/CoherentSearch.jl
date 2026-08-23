# Does the guru profile-major `brfft` (bench/guru_transpose_probe.jl) still win
# across the WHOLE decimation ladder, and in both precisions?
#
#     julia --project=bench -t 1 bench/guru_brfft_ladder.jl
#
# The probe only did k = 1 in Float64, and compared against a naive whole-array
# `copyto!(tile, transpose(...))` rather than the tiled `_bc_transpose!` that
# actually ships — which is ~2.7x faster than the naive one on the laptop, so the
# probe overstated what is there to win.  This compares like for like:
#
#   today  = dense batched brfft  +  the shipped per-tile `_bc_transpose!`
#   guru   = one c2r that writes profile-major (Nprof, nbins) — the gate's own
#            tile layout — plus, in :f64 only, a contiguous Float64->Float32 pass
#
# The decimated passes (k > 1) take a stride-k *input* view of `ftprofs`, so under
# this scheme they are strided on BOTH sides; that is the case most likely to
# disappoint and is the reason this sweeps the ladder.

using FFTW, LinearAlgebra, BenchmarkTools, Printf
using CoherentSearch
const CS = CoherentSearch
const B  = CS._BC_BATCH
const PI_ = FFTW.PRESERVE_INPUT

const L64 = FFTW.libfftw3
const L32 = FFTW.libfftw3f
for (R, C, lib, pf, xf) in ((Float64, ComplexF64, :L64, :fftw_plan_guru64_dft_c2r,  :fftw_execute_dft_c2r),
                            (Float32, ComplexF32, :L32, :fftwf_plan_guru64_dft_c2r, :fftwf_execute_dft_c2r))
    @eval begin
        # rank-1 c2r of length `n`, batched `nb` times, with every stride given
        # explicitly: input transform stride `is`, batch stride `ib`; output
        # transform stride `os`, batch stride `ob`.  Setting os = Nprof, ob = 1
        # makes the output profile-major, i.e. the boxcar gate's tile layout.
        function guru_c2r_plan(X::AbstractArray{$C}, Y::AbstractArray{$R},
                               n, is, os, nb, ib, ob, flags::UInt32)
            dims    = reshape(Int[n,  is, os], 3, 1)
            howmany = reshape(Int[nb, ib, ob], 3, 1)
            p = ccall(($(QuoteNode(pf)), $lib), Ptr{Cvoid},
                      (Int32, Ptr{Int}, Int32, Ptr{Int}, Ptr{$C}, Ptr{$R}, UInt32),
                      1, dims, 1, howmany, X, Y, flags)
            p == C_NULL && error("FFTW refused the guru c2r plan")
            p
        end
        guru_exec!(p::Ptr{Cvoid}, X::AbstractArray{$C}, Y::AbstractArray{$R}) =
            ccall(($(QuoteNode(xf)), $lib), Cvoid,
                  (Ptr{Cvoid}, Ptr{$C}, Ptr{$R}), p, X, Y)
    end
end

prod_transpose!(tile, profs, nbins, n) = begin
    j0 = 0
    while j0 + B <= n
        CS._bc_transpose!(tile, profs, j0, nbins, Val(B))
        j0 += B
    end
end

const NH    = 60
const NPROF = 2048
const KS    = CS.decimation_set(NH, 6)
const FLAGS = FFTW.PATIENT | PI_

function run(::Type{P}) where {P}
    C = Complex{P}
    T = CS._BC_TILE
    ftprofs = randn(C, NH + 1, NPROF)
    ref = copy(ftprofs)
    @printf("\n%s profiles (--precision %s), nharms=%d, Nprof=%d, PATIENT|PRESERVE_INPUT\n",
            P, P === Float32 ? "f32" : "f64", NH, NPROF)
    @printf("  %-3s %6s | %8s %8s %8s | %8s %8s %8s | %6s\n",
            "k", "nbins", "dense", "transp", "today", "guru", "narrow", "guru+", "speedup")
    tot_today = 0.0; tot_guru = 0.0
    for k in KS
        Hk = fld(NH, k); nbins = 2Hk
        src = @view ftprofs[1:k:(Hk*k + 1), :]
        dprofs = Matrix{P}(undef, nbins, NPROF)
        profsT = Matrix{P}(undef, NPROF, nbins)
        tile   = Vector{T}(undef, B * nbins)
        tile2  = Matrix{T}(undef, NPROF, nbins)
        pd = plan_brfft(src, nbins, 1; flags=FLAGS)
        # input: transform stride k, batch stride NH+1 (both in C elements)
        # output: transform stride NPROF, batch stride 1  => profile-major
        pg = guru_c2r_plan(ftprofs, profsT, nbins, k, NPROF, NPROF, NH + 1, 1, FLAGS)
        copyto!(ftprofs, ref)
        mul!(dprofs, pd, src)
        @assert ftprofs == ref  "dense plan destroyed its input"
        guru_exec!(pg, ftprofs, profsT)
        @assert ftprofs == ref  "guru plan destroyed its input"
        err = maximum(abs, dprofs .- transpose(profsT)) / maximum(abs, dprofs)
        @assert err < (P === Float32 ? 1e-5 : 1e-12) "guru output disagrees with dense: $err"

        td = @belapsed mul!($dprofs, $pd, $src)
        tt = @belapsed prod_transpose!($tile, $dprofs, $nbins, $NPROF)
        tg = @belapsed guru_exec!($pg, $ftprofs, $profsT)
        tn = P === Float32 ? 0.0 : @belapsed copyto!($tile2, $profsT)
        today = td + tt; guru = tg + tn
        tot_today += today; tot_guru += guru
        @printf("  %-3d %6d | %6.1fus %6.1fus %6.1fus | %6.1fus %6.1fus %6.1fus | %5.2fx  (rel err %.1e)\n",
                k, nbins, 1e6td, 1e6tt, 1e6today, 1e6tg, 1e6tn, 1e6guru, today/guru, err)
    end
    @printf("  %-3s %6s | %6s %6s %6.1fus | %6s %6s %6.1fus | %5.2fx\n",
            "all", "", "", "", 1e6tot_today, "", "", 1e6tot_guru, tot_today/tot_guru)
end

# run(Float64)  (done above)
run(Float64)
run(Float32)
