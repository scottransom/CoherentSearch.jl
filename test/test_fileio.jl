using Test
using CoherentSearch

# Path to the example data in the sibling Python repo (used as a real-data
# fixture).  These tests are skipped if the data is not present.
const EXAMPLE_FFT = joinpath(@__DIR__, "..", "..", "coherent_search",
                             "examples", "harmonics_hi.fft")

@testset "FFTFile / SimpleInf" begin
    if isfile(EXAMPLE_FFT)
        ft = FFTFile(EXAMPLE_FFT)
        @test ft.N == 1_000_000
        @test ft.inf.dt ≈ 0.001
        @test ft.T ≈ 1000.0
        @test ft.df ≈ 1.0 / 1000.0
        @test length(ft.amps) == ft.N ÷ 2          # 500_000 complex bins
        @test eltype(ft.amps) == ComplexF32
        @test ft.dereddened == false
        # freqs covers DC .. just below Nyquist
        f = freqs(ft)
        @test length(f) == ft.N ÷ 2
        @test first(f) == 0.0
    else
        @info "Skipping FFTFile data tests; example file not found" EXAMPLE_FFT
    end
end
