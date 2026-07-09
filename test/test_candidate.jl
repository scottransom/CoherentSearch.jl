using Test
using CoherentSearch

# `candidate_profile` reconstructs a single candidate's pulse profile with a
# wide exact-interpolation kernel and a plain irfft.  It is an independent code
# path from the search's `reference_profiles` (FFT-correlation interpolation +
# linear resampling), so pinning the two together — with matched kernel width
# `m` and a fine reference grid so only the interpolation *method* differs —
# guards the indexing, FFT convention, and orientation of the reconstruction.
const EXAMPLE_FFT = joinpath(@__DIR__, "..", "..", "coherent_search",
                             "examples", "harmonics_hi.fft")

@testset "candidate_profile / rotate_to_peak" begin
    @testset "rotate_to_peak centers the maximum" begin
        for n in (8, 9, 64, 65)
            prof = collect(1.0:n)                       # max at the last bin
            rp = rotate_to_peak(prof)
            @test length(rp) == n
            @test argmax(rp) == n ÷ 2 + 1               # peak moved to the center
            @test sort(rp) == sort(prof)                # circular shift preserves values
        end
    end

    if isfile(EXAMPLE_FFT)
        @testset "pinned to reference_profiles (10.0123 Hz pulsar)" begin
            ft = FFTFile(EXAMPLE_FFT)
            nh = 32
            rbest = 10012.33125                          # fundamental Fourier freq (bins)

            # Matched kernel (m=32) + a fine reference grid (numbetween=256) so the
            # only remaining difference is exact- vs linear-interpolation error.
            p = SearchParams(nharms=nh, m=32, numbetween=256, align=false)
            ref = reference_profiles(ft, [rbest], p)[:, 1]
            cand = candidate_profile(ft, rbest, nh; m=32)

            # All 32 harmonics fit below Nyquist here, so the fold has 2*nh bins.
            @test length(cand) == 2nh
            @test eltype(cand) == Float64
            reldiff = maximum(abs, cand .- ref) / maximum(abs, ref)
            @test reldiff < 1e-3

            wide = candidate_profile(ft, rbest, nh)      # default m=64
            @test length(wide) == 2nh
            @test argmax(rotate_to_peak(wide)) == nh + 1  # center of 2nh bins

            # Nyquist capping: request far more harmonics than fit.  Only the
            # harmonics below the Nyquist bin (ft.N/2) are usable; the profile
            # must use exactly those (2*H bins) and NOT zero-pad to the requested
            # depth, so its length is well below 2*200.
            Nhalf = ft.N ÷ 2
            capped = candidate_profile(ft, rbest, 200)   # request 200 harmonics
            H = length(capped) ÷ 2
            @test iseven(length(capped))
            @test H < 200                                 # capping actually happened
            @test rbest * H < Nhalf                       # top harmonic used is sub-Nyquist
            @test rbest * (H + 2) > Nhalf                 # and the fold stopped near Nyquist
        end
    else
        @info "Skipping candidate_profile data tests; example file not found" EXAMPLE_FFT
    end
end
