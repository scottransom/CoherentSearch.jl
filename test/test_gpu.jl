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

    # --- stage 2: transform + boxcar gate ---------------------------------
    # Pinned per rung, because each rung is a separately-planned transform over a
    # separately-built dense stack, and a decimation bug would otherwise hide
    # behind rung 1 being right.
    @testset "profiles vs CPU, k=$k" for k in 1:4
        for n in (256, 1024)
            c = chunk_profiles(CPUBackend(), ft, params, r_lo, n; k = k)
            g = chunk_profiles(CS.require_gpu(), ft, params, r_lo, n; k = k)
            @test size(g) == size(c) == (2 * fld(params.nharms, k), n)
            @test maximum(abs, g .- c) / maximum(abs, c) < 1e-5
        end
    end

    # `invsigma` is supplied rather than estimated, so this pins the FILTER --
    # width bank, wrapped prefix sums, the delta*S_tot baseline -- without either
    # side having to agree about sigma estimation too.  The statistic is exactly
    # linear in 1/sigma, so one value covers all of them.
    @testset "boxcar metric vs CPU, k=$k" for k in 1:4
        for n in (256, 1024), invsig in (1.0, 0.37)
            c = chunk_boxcar(CPUBackend(), ft, params, r_lo, n; k = k, invsigma = invsig)
            g = chunk_boxcar(CS.require_gpu(), ft, params, r_lo, n; k = k, invsigma = invsig)
            @test length(g) == n
            @test maximum(abs.(g .- c)) / maximum(abs, c) < 1e-5
        end
    end

    # The metric is linear in 1/sigma; assert that on the GPU path itself, since
    # the production search will supply a per-chunk sigma and this is what makes
    # a single pinned value above sufficient.
    @testset "boxcar is linear in invsigma" begin
        a = chunk_boxcar(CS.require_gpu(), ft, params, r_lo, 512; invsigma = 1.0)
        b = chunk_boxcar(CS.require_gpu(), ft, params, r_lo, 512; invsigma = 2.5)
        @test maximum(abs.(b .- 2.5 .* a)) / maximum(abs, b) < 1e-6
    end

    # ------------------------------------------------------------------
    # Transform sub-batching (gpu_design.md 4.9).  Splitting a rung's batched
    # transform into column blocks is a SCHEDULING change: every column's
    # transform is independent, so the profiles must come back bit-identical,
    # not merely within the 1e-5 pin the rest of this file uses.  Anything less
    # than exact equality here means cuFFT changed algorithm with batch size,
    # which is exactly what a per-device policy must not do silently.
    #
    # The explicit byte targets bypass the residency gate on purpose -- on a
    # small-L2 card `:auto` correctly declines to split at all, so without the
    # override this testset would pass by doing nothing.  Hence the assertion
    # that a real split actually happened.
    # ------------------------------------------------------------------
    @testset "transform sub-batching is bit-exact, k=$k" for k in 1:4
        E = Base.get_extension(CoherentSearch, :CoherentSearchCUDAExt)
        for n in (1024, 999)                       # 999: a ragged final block
            CS.gpu_subbatch!(:off)
            ref = chunk_profiles(CS.require_gpu(), ft, params, r_lo, n; k = k)
            for target in (1 << 16, 1 << 14)
                CS.gpu_subbatch!(target)
                gc = E.GPUChunk(Float32, params, n; subbatch = target)
                @test any(>(1), gc.nblocks)        # a split really happened
                @test sum(gc.sub[i] * (gc.nblocks[i] - 1) + gc.tail[i]
                          for i in eachindex(gc.ks)) == n * length(gc.ks)
                @test chunk_profiles(CS.require_gpu(), ft, params, r_lo, n; k = k) == ref
            end
            CS.gpu_subbatch!(:auto)
        end
    end

    # The policy itself, with no device involved: it must decline to split on a
    # small-L2 card (measured 1.00x on the 4 MB RTX 2080 Super, so splitting
    # there could only cost launches) and must reproduce the per-rung optima
    # measured on the 40 MB Ada.
    @testset "sub-batch policy is per-device" begin
        E = Base.get_extension(CoherentSearch, :CoherentSearchCUDAExt)
        cols(l2) = [E._sub_cols(Float32, fld(60, k), 262144, l2 ÷ 2) for k in 1:6]
        @test all(==(262144), cols(2 * 2^20))      # GTX 1080: off
        @test all(==(262144), cols(4 * 2^20))      # RTX 2080 Super: off
        ada = cols(40 * 2^20)
        @test all(<(262144), ada)                  # RTX 4000 Ada: on at every rung
        @test issorted(ada)                        # deep folds want SMALLER batches
        # Within 1.35x of gpu_design.md 0.3's independently measured optima.
        for (d, m) in zip(ada, (16384, 32768, 65536, 65536, 131072, 131072))
            @test 1/1.35 < d / m < 1.35
        end
        # An explicit target bypasses the guards; :off is always one batch.
        @test E._sub_cols(Float32, 60, 262144, 1 << 20; policy = false) == 1083
        @test E._sub_cols(Float32, 60, 262144, 0) == 262144
    end

    @testset "backend registry" begin
        @test CoherentSearch.has_gpu()
        @test CoherentSearch.require_gpu() === CoherentSearch.gpu_backend()
    end
end
end
