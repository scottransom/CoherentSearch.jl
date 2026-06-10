using Test
using CoherentSearch

# The optimised chunk/plan-caching path must agree with the simple reference
# `block_metrics` (which is itself pinned to the Python oracle to ~1e-15), and
# must actually detect the 10.0123 Hz test pulsar.  These run only when the
# example data from the sibling Python repo is present.
const EXAMPLE_FFT = joinpath(@__DIR__, "..", "..", "coherent_search",
                             "examples", "harmonics_hi.fft")

@testset "harmonic_numbetween schedule" begin
    nh, hidr, minnb = 32, 0.5, 16
    # Never below the floor; finer at the low harmonics; matched to deltar_h.
    @test harmonic_numbetween(1,  nh, hidr, minnb) == 64    # = 2*nharms
    @test harmonic_numbetween(2,  nh, hidr, minnb) == 32
    @test harmonic_numbetween(4,  nh, hidr, minnb) == 16
    @test harmonic_numbetween(32, nh, hidr, minnb) == 16    # floored
    @test all(harmonic_numbetween(h, nh, hidr, minnb) >= minnb for h in 1:nh)
end

if isfile(EXAMPLE_FFT)
    ft = FFTFile(EXAMPLE_FFT)

    @testset "optimised path reproduces the reference (align=false)" begin
        # With a fixed numbetween and one chunk, the production path uses the
        # same grids as block_metrics, so it should match to ~machine precision.
        params = SearchParams(nharms=32, m=32, numbetween=16, align=false)
        lodr = params.hidr / params.nharms
        rstart = 10010.0
        n = 256
        rfund = rstart .+ (0:n-1) .* lodr

        ref = block_metrics(ft, rfund, params)
        opt = chunk_metrics(ft, params, rstart, n; lodr=lodr)

        relerr = maximum(abs.(opt .- ref)) / maximum(abs.(ref))
        @info "align=false reference agreement" relerr
        @test relerr < 1e-10
    end

    @testset "per-harmonic alignment is more accurate at low harmonics" begin
        # The whole point of per-harmonic numbetween: low harmonics, whose
        # finterp grid is coarse relative to their curvature at a fixed
        # numbetween, get a finer grid and so far more accurate amplitudes.
        # Demonstrate it directly on harmonic 1's interpolated amplitudes,
        # using nb=256 as the near-exact reference.
        rstart = 10010.0
        n = 16
        rs = rstart .+ (0:n-1) .* (0.5 / 32)          # harmonic 1 trial freqs

        function amps_at(nb)
            lobin = floor(Int, minimum(rs))
            numbins = ceil(Int, maximum(rs)) + 1 - lobin
            grid = finterp_fft(lobin, numbins, nb, ft.amps, 32)
            [CoherentSearch.uniform_linear_interp(r, lobin, nb, grid) for r in rs]
        end

        nb_aligned = harmonic_numbetween(1, 32, 0.5, 16)   # = 64
        truth = amps_at(256)
        rel(a) = maximum(abs.(a .- truth)) / maximum(abs.(truth))
        err_fixed   = rel(amps_at(16))
        err_aligned = rel(amps_at(nb_aligned))
        @info "harmonic-1 amplitude error" err_fixed err_aligned
        @test err_aligned < err_fixed / 100      # finer grid is far more accurate
    end

    @testset "detects the 10.0123 Hz pulsar" begin
        params = SearchParams(nharms=32, m=32, numbetween=16)
        cands = search(ft, params; lofreq=9.5, hifreq=10.5, threshold=8.0)
        @test !isempty(cands)
        best = cands[argmax(c.metric for c in cands)]
        @info "strongest candidate" best.freq best.metric
        @test isapprox(best.freq, 10.0123; atol=1e-2)
    end

    @testset "chunk size does not change the detection" begin
        params = SearchParams(nharms=32, m=32, numbetween=16)
        c1 = search(ft, params; lofreq=9.5, hifreq=10.5, threshold=8.0, blocksize=512)
        c2 = search(ft, params; lofreq=9.5, hifreq=10.5, threshold=8.0, blocksize=4096)
        b1 = c1[argmax(c.metric for c in c1)]
        b2 = c2[argmax(c.metric for c in c2)]
        @test isapprox(b1.freq, b2.freq; atol=1e-3)
        @test isapprox(b1.metric, b2.metric; rtol=1e-2)
    end
else
    @info "Skipping search data tests; example file not found" EXAMPLE_FFT
end
