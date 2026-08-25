# GPU backend tests — the stage-1 equivalence gate of `gpu_design.md` §5.
#
# **Skipped unless CUDA is loadable AND functional.**  CUDA is a weak dependency,
# so an ordinary `Pkg.test()` does not have it and these are silently skipped;
# that is the intended behaviour, and it is what keeps the CPU test suite free of
# any GPU cost.  To actually run them, use an environment that has both:
#
#   julia --project=<env with CoherentSearch + CUDA> -e 'using CUDA; include("test/test_gpu.jl")'
#
# The tolerances here are deliberately NOT machine precision.  The GPU sums the
# same Eqn.-30 kernel in a different order, so ~1e-6 at `Float32` weights is the
# right pin — the same reasoning that gave the shipped `Float32` weight path its
# own ~1e-6 pins rather than loosening the 1e-8 ones.  Do not relax these to
# swallow a discrepancy; add a named pin for the deliberate difference instead.

using Test, CoherentSearch

const _GPU_OK = try
    @eval using CUDA
    CoherentSearch.has_gpu()
catch
    false
end

if !_GPU_OK
    @info "GPU tests skipped: no functional CUDA backend (CUDA is a weak dependency)"
else
@testset "gpu backend" begin
    CS = CoherentSearch
    # A small synthetic observation, in the spirit of the precompile workload:
    # these tests pin GPU-vs-CPU agreement, which needs no real pulsar.
    N = 1 << 16
    dt = 1.0e-4
    amps = ComplexF32[ComplexF32(sinpi(0.013f0 * i), cospi(0.007f0 * i))
                      for i in 1:(N ÷ 2)]
    inf = SimpleInf("synthetic.inf", "GPUTEST", 5.0e4, N, dt, 0.0)
    ft = FFTFile("synthetic.fft", amps, inf, N, N * dt, 1.0 / (N * dt),
                 true, true, real(amps[1]), imag(amps[1]))
    params = SearchParams(nharms = 20, m = 16, decimations = collect(1:4))
    lodr = params.hidr / params.nharms
    r_lo = 200.0

    @testset "interp equivalence, weights=$WT" for WT in (Float32, Float64)
        tol = WT === Float32 ? 1e-5 : 1e-6
        for (n, t0) in ((256, 0), (2048, 0), (517, 37), (1024, 4000))
            c, fc = chunk_ftprofs(CPUBackend(), ft, params, r_lo, n; t0 = t0, weights = WT)
            g, fg = chunk_ftprofs(CS.require_gpu(), ft, params, r_lo, n; t0 = t0, weights = WT)
            @test fc == fg                       # same harmonics gave up
            @test size(g) == size(c)
            scale = maximum(abs, c)
            @test scale > 0
            @test maximum(abs, g .- c) / scale < tol
        end
    end

    # The pin that protects the sensitivity Monte Carlo: the GPU may use a
    # completely different chunk size from the CPU, and must not thereby change a
    # single result.  Bit-exactness (not a tolerance) is the correct assertion —
    # groups are anchored to the GLOBAL trial index, and partial end groups are
    # computed in full and masked on store, precisely so this holds.
    @testset "gpu batch invariance is bit-exact" begin
        n_all = 2048
        gall, _ = chunk_ftprofs(CS.require_gpu(), ft, params, r_lo, n_all; t0 = 0)
        for nsub in (1024, 512, 100, 37)         # 100 and 37 straddle 32-trial groups
            for off in 0:nsub:(n_all - 1)
                k = min(nsub, n_all - off)
                # Plans are built against the GLOBAL r_lo; position enters ONLY
                # through t0.  Passing `r_lo + off*lodr` as well double-counts it.
                gsub, _ = chunk_ftprofs(CS.require_gpu(), ft, params, r_lo, k; t0 = off)
                @test gsub == gall[:, (off + 1):(off + k)]
            end
        end
    end

    @testset "backend registry" begin
        @test CoherentSearch.has_gpu()
        @test CoherentSearch.require_gpu() === CoherentSearch.gpu_backend()
    end
end
end
