using Test
using CoherentSearch
using FFTW: irfft
using Random

# The toy search (`bin/toy_coherent_search.jl`) is the paper's pseudo-code
# written out as runnable Julia.  It is only worth having if it computes the
# *same* quantities the production path does, so these tests pin it piece by
# piece against the oracle-validated reference path — the interpolation, the
# fold, the metric — and then end to end.
#
# It also pins the one thing the toy does that production does not: the
# **analytic** noise scale, `sigma = sqrt(2*nlow + 0.5*nnyq)/nbins`, derived
# from the input FFT being normalised rather than measured from the data.  That
# claim is checked against synthetic unit-variance noise, because if it is wrong
# every S/N the toy reports is wrong by the same factor.

include(joinpath(@__DIR__, "..", "bin", "toy_coherent_search.jl"))
using .ToyCoherentSearch

const TOY_EXAMPLE_FFT = joinpath(@__DIR__, "..", "..", "coherent_search",
                                 "examples", "harmonics_hi_red.fft")

# A synthetic observation whose Fourier amplitudes are exactly the normalised
# white noise the analytic sigma assumes: |A|^2 has mean 1, so Re and Im each
# have variance 1/2.  `seed` makes the whole testset deterministic.
function noise_fftfile(N::Integer; dt::Real = 1.0e-4, seed::Integer = 20260824)
    rng = MersenneTwister(seed)
    nb = N ÷ 2
    amps = ComplexF32[ComplexF32(randn(rng, Float32), randn(rng, Float32)) / sqrt(2f0)
                      for _ in 1:nb]
    inf = SimpleInf("noise.inf", "TOYNOISE", 5.0e4, Int(N), Float64(dt), 0.0)
    return FFTFile("noise.fft", amps, inf, Int(N), N * dt, 1.0 / (N * dt),
                   true, true, real(amps[1]), imag(amps[1]))
end


@testset "fourier_interpolate: wrapper agrees with the kernel, and gates its range" begin
    ft = noise_fftfile(1 << 14)
    m = 16
    for r in (150.0, 150.5, 1234.25, 3000.75)
        @test fourier_interpolate(ft, r, m) == fourier_interp(r, ft.amps, m)
    end
    # Past Nyquist, and off the low end (the m/2-bin window would underflow):
    # exactly zero rather than an error, which is what lets a search loop just
    # keep going.
    @test fourier_interpolate(ft, ft.N / 2, m) == 0
    @test fourier_interpolate(ft, ft.N / 2 + 10, m) == 0
    @test fourier_interpolate(ft, 1.0, m) == 0
    @test fourier_interpolate(ft, -1.0, m) == 0
    @test_throws ArgumentError fourier_interpolate(ft, 200.0, 15)
    # Availability is monotone in r, which `harmonic_amplitudes` relies on to
    # stop at the first missing harmonic rather than testing every one.
    rs = range(ft.N / 2 - 50; step = 1.0, length = 100)
    avail = [!iszero(fourier_interpolate(ft, r, m)) for r in rs]
    @test issorted(avail; rev = true)
end

@testset "toy fold == reference_profiles(:direct), every decimation" begin
    ft = FFTFile(TOY_EXAMPLE_FFT)
    nharms, m = 60, 16
    params = SearchParams(nharms = nharms, m = m)
    # Every harmonic must clear Nyquist for the fold to be full: r*nharms < N/2,
    # i.e. r < 8333 here.  (The fixture's own 10.0123 Hz pulsar sits at r = 10012
    # and so truncates at harmonic 49 — exercised by the end-to-end testset.)
    for r in (3000.5, 7777.25, 8000.125)
        amps, navail = ToyCoherentSearch.harmonic_amplitudes(ft, r, nharms, m)
        @test navail == nharms
        # k = 1: the toy's fold is the reference path's, to machine precision.
        # Both evaluate Eqn. 30 exactly and both `irfft`, so any difference is
        # summation order, not method.
        prof, nfilled = ToyCoherentSearch.fold_profile(amps, 1, nharms)
        ref = reference_profiles(ft, [r], params; kernel = :direct)[:, 1]
        @test nfilled == nharms
        @test length(prof) == 2nharms
        @test maximum(abs, prof .- ref) <= 1e-12 * maximum(abs, ref)

        # k > 1: the decimated fold must equal the native H_k-harmonic fold of
        # k*r.  This is the whole claim of `decimation_design.md` §1 — that
        # every k-th base harmonic *is* the harmonic stack of the multiple —
        # and it is checked here against an independent computation of it.
        for k in 2:6
            Hk = nharms ÷ k
            dprof, dfilled = ToyCoherentSearch.fold_profile(amps, k, nharms)
            dref = reference_profiles(ft, [k * r], SearchParams(nharms = Hk, m = m);
                                      kernel = :direct)[:, 1]
            @test dfilled == Hk
            @test length(dprof) == 2Hk
            # Looser than the k=1 pin, and necessarily so: the toy asks for
            # harmonic j of the multiple at `r*(j*k)` while the reference asks
            # at `(r*k)*j`, which differ by an ulp of a ~5e4-bin frequency.  The
            # interpolation phase turns that into ~1e-11 relative, so this is
            # floating-point associativity, not a difference in method.
            @test maximum(abs, dprof .- dref) <= 1e-9 * maximum(abs, dref)
        end
    end
end

@testset "toy fold truncates at Nyquist rather than zero-padding blindly" begin
    ft = FFTFile(TOY_EXAMPLE_FFT)
    nharms, m = 60, 16
    # The fixture's pulsar: r = 10012.3 bins with Nyquist at 500000, so
    # harmonics above floor(500000/10012.3) = 49 have no data.
    r = 10.0123456789123 * ft.T
    amps, navail = ToyCoherentSearch.harmonic_amplitudes(ft, r, nharms, m)
    @test navail == floor(Int, (ft.N / 2) / r)
    @test all(!iszero, amps[1:navail])
    @test all(iszero, amps[navail+1:end])
    # A decimated stack inherits the truncation: it holds harmonics k, 2k, ...,
    # so it is filled up to floor(navail/k).
    for k in 1:6
        _, nfilled = ToyCoherentSearch.fold_profile(amps, k, nharms)
        @test nfilled == min(nharms ÷ k, navail ÷ k)
    end
    # ...and the analytic noise scale follows the filling, not the stack length.
    # Getting this wrong would overestimate sigma by sqrt(60/49) = 11% here and
    # far more at the top of the band, silently suppressing fast candidates.
    prof, nfilled = ToyCoherentSearch.fold_profile(amps, 1, nharms)
    @test ToyCoherentSearch.analytic_sigma(length(prof), nfilled) <
          ToyCoherentSearch.analytic_sigma(length(prof), nharms)
end

@testset "toy boxcar_snr == snr_metrics (the production metric)" begin
    rng = MersenneTwister(4114)
    for nbins in (20, 24, 30, 40, 60, 120)
        for trial in 1:5
            prof = randn(rng, nbins)
            prof .-= sum(prof) / nbins           # DC held at zero, as the fold does
            # `snr_metrics` on a single column pools sigma over exactly that
            # column and centres it at zero, i.e. it *is* `mad_sigma`.  So the
            # two must agree to rounding, not approximately.
            sig = ToyCoherentSearch.mad_sigma(prof)
            got = ToyCoherentSearch.boxcar_snr(prof, sig)
            want = snr_metrics(reshape(prof, :, 1); sigma_center = :zero)[1]
            @test isapprox(got, want; rtol = 1e-12)
        end
    end
    # The statistic is exactly 1/sigma, and invariant to any constant baseline
    # (the profile mean is removed inside `snr_metrics`, and the `delta*S_tot`
    # term removes it here).
    prof = randn(rng, 120)
    sig = ToyCoherentSearch.mad_sigma(prof)
    @test ToyCoherentSearch.boxcar_snr(prof, 2sig) ≈ 0.5 * ToyCoherentSearch.boxcar_snr(prof, sig)
    @test ToyCoherentSearch.boxcar_snr(prof .+ 37.0, sig) ≈ ToyCoherentSearch.boxcar_snr(prof, sig)
    @test ToyCoherentSearch.boxcar_snr(prof, 0.0) == -Inf
    # A boxcar planted at a known width and phase must be found there.
    planted = zeros(120); planted[41:49] .= 10.0    # width 9 is in the bank
    @test ToyCoherentSearch.boxcar_snr(planted, 1.0) ≈
          maximum(w -> begin
                      d = w / 120
                      (sum(planted[41:min(40 + w, 120)]) - d * sum(planted)) /
                          sqrt(w * (1 - d))
                  end, boxcar_widths(120))
end

@testset "analytic_sigma: the closed form, and its arithmetic" begin
    # Full stack: sqrt(2*(H-1) + 0.5)/nbins, i.e. 1/sqrt(nbins) times a
    # sqrt(1 - 3/(4H)) correction that is 0.6% at H=60 and 3.8% at H=10.
    for H in (10, 12, 15, 20, 30, 60)
        n = 2H
        s = ToyCoherentSearch.analytic_sigma(n, H)
        @test s ≈ sqrt(2 * (H - 1) + 0.5) / n
        @test s / (1 / sqrt(n)) ≈ sqrt(1 - 3 / (4H)) rtol = 1e-12
    end
    @test ToyCoherentSearch.analytic_sigma(120, 60) / (1 / sqrt(120)) ≈ 0.99374 atol = 1e-5
    @test ToyCoherentSearch.analytic_sigma(20, 10)  / (1 / sqrt(20))  ≈ 0.96177 atol = 1e-5
    # A partly-filled stack (harmonics past Nyquist) carries proportionally less
    # noise, and none of it is the profile's Nyquist bin.
    @test ToyCoherentSearch.analytic_sigma(120, 30) ≈ sqrt(2 * 30) / 120
    @test ToyCoherentSearch.analytic_sigma(120, 0) == 0.0
end

@testset "analytic_sigma matches the true fold noise on normalised white noise" begin
    # THE test for the analytic scale: fold real (synthetic) normalised noise
    # many times and compare the empirical per-bin sd against the prediction.
    # If this drifts, every S/N the toy reports drifts with it.
    ft = noise_fftfile(1 << 20)
    nharms = 60

    # The trial step matters here, for two reasons that both cost a debugging
    # round when they were got wrong:
    #   * it must be MUCH coarser than the search grid (0.5/nharms bins), or
    #     adjacent trials read almost the same Fourier bins and the sample has
    #     far fewer independent realisations than it has bins — which showed up
    #     as a spurious 2% "bias" at the shallow folds;
    #   * it must be IRRATIONAL, or every trial lands on an integer bin, the
    #     interpolation weights collapse to (1, 0, 0, ...) and no interpolation
    #     is exercised at all (that arrangement does confirm the closed form
    #     exactly, at every k, but it says nothing about the kernel).
    # r*nharms < N/2 = 524288 keeps every stack full.
    rs = range(5000.0; step = sqrt(2), length = 1500)

    # Empirical per-bin sd of the fold at decimation `k`, kernel width `m`.
    function fold_sd(m::Integer, k::Integer)
        acc, cnt = 0.0, 0
        for r in rs
            amps, navail = ToyCoherentSearch.harmonic_amplitudes(ft, r, nharms, m)
            navail == nharms || continue
            prof, _ = ToyCoherentSearch.fold_profile(amps, k, nharms)
            acc += sum(abs2, prof); cnt += length(prof)
        end
        return sqrt(acc / cnt)
    end

    for m in (16, 32, 64), k in (1, 3, 6)
        nbins = 2 * (nharms ÷ k)
        predicted = ToyCoherentSearch.analytic_sigma(nbins, nharms ÷ k)
        @test isapprox(fold_sd(m, k), predicted; rtol = 0.015)
    end

    # The residual is the interpolation truncation and nothing else, so it has a
    # predicted SIGN and SIZE: the m-bin kernel keeps S_m = sum sinc^2 ~ 1 -
    # 0.203/m of the noise power along with the signal, making the prediction
    # high by ~0.203/(2m) — 0.64% at m=16, 0.32% at m=32, 0.16% at m=64.
    # Measured 2026-08-24: 1.0086 / 1.0053 / 1.0035 at k=1.
    bias(m) = ToyCoherentSearch.analytic_sigma(2nharms, nharms) / fold_sd(m, 1)
    b16, b32, b64 = bias(16), bias(32), bias(64)
    @test b16 > b32 > b64 > 1.0                  # always high, and less so with wider m
    for (m, b) in ((16, b16), (32, b32), (64, b64))
        @test isapprox(b - 1, 0.203 / (2m); atol = 0.003)
    end
end

@testset "toy_search: end to end against the production search" begin
    ft = FFTFile(TOY_EXAMPLE_FFT)
    nharms, m, lo, hi = 60, 16, 9.98, 10.05
    toy = ToyCoherentSearch.toy_search(ft; nharms = nharms, m = m, lofreq = lo,
                                       hifreq = hi, maxdecim = 6, threshold = 8.0,
                                       progress = false)
    @test !isempty(toy)
    toy = remove_harmonics(remove_duplicates(toy; dr_tol = 1.0); numharm = 16)
    best = argmax(c -> c.metric, toy)
    @test isapprox(best.freq, 10.0123456789123; atol = 2e-4)

    # The production search over the same band must find the same signal at the
    # same fold depth.  The S/N values are NOT expected to be identical: the toy
    # divides by the analytic noise scale and production by a measured one, and
    # the difference between those is exactly what this comparison is for.  A
    # few percent is the bias budget (0.6% interpolation truncation + ~1%
    # sampling error in production's sigma-hat); 10% would mean a real bug.
    params = SearchParams(nharms = nharms, m = m, precision = :f64,
                          decimations = decimation_set(nharms, 6))
    prod = search(ft, params; lofreq = lo, hifreq = hi, threshold = 8.0,
                  progress = :none, wisdom = false)
    @test !isempty(prod)
    pbest = argmax(c -> c.metric, prod)
    @test isapprox(pbest.freq, best.freq; atol = 1e-6)
    @test pbest.nharm == best.nharm
    @test isapprox(best.metric, pbest.metric; rtol = 0.10)

    # Feeding the toy production's *measured* noise scale instead removes the
    # analytic assumption from the comparison, and the agreement tightens.
    toy_mad = ToyCoherentSearch.toy_search(ft; nharms = nharms, m = m, lofreq = lo,
                                           hifreq = hi, maxdecim = 6, threshold = 8.0,
                                           sigma = :mad, progress = false)
    mbest = argmax(c -> c.metric, remove_duplicates(toy_mad; dr_tol = 1.0))
    # Within a few trial steps, not exactly: a per-profile MAD carries ~7% (120
    # bins) to ~17% (20 bins) sampling error, so WHICH trial of the cluster
    # scores highest is close to arbitrary.  Measured, it lands 2 steps
    # (dr = 0.5/nharms bins) off the analytic arm's peak.  That is the cost of
    # the per-profile estimate, and the reason production pools sigma over a
    # whole chunk and the toy computes it instead.
    @test abs(mbest.freq - best.freq) < 20 * (0.5 / nharms) / ft.T

    @test_throws ArgumentError ToyCoherentSearch.profile_sigma(:nope, [1.0], 2, 1)
end
