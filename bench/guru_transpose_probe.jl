# SUPERSEDED (2026-08-22) — kept because the guru mechanics in it are correct and
# reusable, but its CONCLUSION was wrong.  It compares the guru profile-major
# `brfft` against `copyto!(tile, transpose(Yd))`, a naive whole-array transpose
# that is ~2.7x slower than the `_bc_transpose!` the search actually runs, so the
# "1.25-1.50x" below is a baseline error.  Swept across the whole decimation
# ladder in both precisions against the shipped kernel
# (`bench/guru_brfft_ladder.jl`), the guru c2r is **0.78x (:f64) / 0.99x (:f32)**
# on the laptop.  The transpose was instead fixed by blocking its profile axis;
# see `docs/Summary_and_Future_Work.md` §3.3.
using FFTW, LinearAlgebra, BenchmarkTools, Printf
const lib = FFTW.libfftw3
function guru_c2r(X, yT, flags)
    n, Np = 2*(size(X,1)-1), size(X,2)
    dims    = reshape(Int[n, 1, Np], 3, 1)
    howmany = reshape(Int[Np, size(X,1), 1], 3, 1)
    p = ccall((:fftw_plan_guru64_dft_c2r, lib), Ptr{Cvoid},
              (Int32, Ptr{Int}, Int32, Ptr{Int}, Ptr{ComplexF64}, Ptr{Float64}, UInt32),
              1, dims, 1, howmany, X, yT, flags)
    p == C_NULL && error("FFTW refused the guru plan (flags=$flags)")
    p
end
exec!(p, X, y) = ccall((:fftw_execute_dft_c2r, lib), Cvoid,
                       (Ptr{Cvoid}, Ptr{ComplexF64}, Ptr{Float64}), p, X, y)
const PI_ = FFTW.PRESERVE_INPUT
nh, Nprof = 60, 2048; nbins = 2nh
X0 = randn(ComplexF64, nh+1, Nprof)
Yd = Matrix{Float64}(undef, nbins, Nprof); yT = Matrix{Float64}(undef, Nprof, nbins)
tile = Matrix{Float32}(undef, Nprof, nbins)
pd = plan_brfft(copy(X0), nbins, 1; flags=FFTW.PATIENT)
pg = guru_c2r(copy(X0), yT, FFTW.PATIENT | PI_)
Xc = copy(X0); exec!(pg, Xc, yT)
@printf("guru+PRESERVE_INPUT: input preserved? %s\n", Xc == X0 ? "YES" : "NO")
X = copy(X0); mul!(Yd, pd, X)
@printf("correct? max|dense - transpose(guru)| = %.3e\n", maximum(abs, Yd .- transpose(yT)))
Xa=copy(X0); Xb=copy(X0)
td  = @belapsed mul!($Yd, $pd, $Xa)
tg  = @belapsed exec!($pg, $Xb, $yT)
ttr = @belapsed copyto!($tile, transpose($Yd))
tnar= @belapsed copyto!($tile, $yT)
@printf("\n  dense brfft (PRESERVE_INPUT, as shipped)   : %7.1f us\n", 1e6td)
@printf("  guru profile-major brfft (PRESERVE_INPUT)  : %7.1f us  (%.2fx dense)\n", 1e6tg, tg/td)
@printf("  scattered transpose->Float32 (today)       : %7.1f us\n", 1e6ttr)
@printf("  contiguous narrow Float64->Float32         : %7.1f us\n", 1e6tnar)
@printf("\n  today            : %7.1f us\n", 1e6*(td+ttr))
@printf("  guru (:f64 mode) : %7.1f us   => %.2fx\n", 1e6*(tg+tnar), (td+ttr)/(tg+tnar))
@printf("  guru (:f32 mode, no narrowing needed): %7.1f us   => %.2fx\n", 1e6tg, (td+ttr)/tg)
