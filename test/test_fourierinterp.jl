using Test
using FFTW
using CoherentSearch

@testset "finterp_coeffs (golden values from PRESTO gen_r_response)" begin
    # Python: pp.gen_r_response(0.1, 1, 12)[2:]
    coeffs = finterp_coeffs(0.1, 10)
    expected = ComplexF64[
        0.02281681 + 0.00741363im, 0.03017707 + 0.00980513im,
        0.04454711 + 0.01447423im, 0.08504448 + 0.02763263im,
        0.9354893  + 0.3039589im, -0.10394325 - 0.03377321im,
        -0.04923628 - 0.01599784im, -0.03225825 - 0.01048134im,
        -0.0239869  - 0.00779382im, -0.01909162 - 0.00620324im,
    ]
    @test coeffs ≈ expected rtol = 1e-5 atol = 1e-6

    # dr = 0.0 must be a unit impulse at the centre bin
    @test finterp_coeffs(0.0, 6) ≈ ComplexF64[0, 0, 1, 0, 0, 0] atol = 1e-12

    @test_throws ArgumentError finterp_coeffs(0.1, 5)   # m must be even
    @test_throws ArgumentError finterp_coeffs(1.0, 6)   # dr out of range
end

@testset "nearby_fourier_bins (0-based→1-based index translation)" begin
    ft = ComplexF64.(0:9)
    # These mirror tests/test_fourierinterp.py exactly.
    @test collect(nearby_fourier_bins(4.5, ft, 4)) == ComplexF64[3, 4, 5, 6]
    @test collect(nearby_fourier_bins(2.2, ft, 6)) == ComplexF64[0, 1, 2, 3, 4, 5]
    @test collect(nearby_fourier_bins(3.0, ft, 4)) == ComplexF64[2, 3, 4, 5]   # exact integer r
    # Range helper directly: window is m bins wide and centred on the bin above r.
    @test length(nearby_fourier_bin_range(100.7, 32)) == 32
    @test nearby_fourier_bin_range(100.7, 32) == nearby_fourier_bin_range(100.2, 32)
end

@testset "fourier_interp (analytic cosine)" begin
    for (r, idx) in ((12400.55, 12), (12400.0, 1))   # idx = 1-based slot in finterp_multi grid
        N = 32768
        phs = π / 4
        signal = cos.(2π * r .* (0:N-1) ./ N .+ phs)
        ft = rfft(signal)
        m = 60
        iv = fourier_interp(r, ft, m)
        expected = N / 2 / sqrt(2) * (1 + 1im)
        @test iv ≈ expected rtol = 1e-2 atol = 1e-3

        rs = floor(r) .+ range(0.0, 1.0; length=21)[1:end-1]
        iv2 = finterp_multi(rs, ft, m)
        @test iv2[idx] ≈ iv atol = 1e-9
    end
end

@testset "finterp_multi == finterp_fft" begin
    N = 32768
    r = 12400.0
    signal = cos.(2π * r .* (0:N-1) ./ N .+ π / 4)
    ft = rfft(signal)
    rs = floor(r) .+ range(0.0, 1.0; length=21)[1:end-1]
    m = 16
    v1 = finterp_multi(rs, ft, m)
    v2 = finterp_fft(12400, 1, length(rs), ft, m)
    @test v1 ≈ v2 rtol = 1e-5 atol = 1e-7
end

@testset "next_pow_of_2" begin
    @test next_pow_of_2(1) == 1
    @test next_pow_of_2(5) == 8
    @test next_pow_of_2(16) == 16
    @test next_pow_of_2(17) == 32
    @test_throws ArgumentError next_pow_of_2(0)
end

@testset "is_smooth / next_smooth" begin
    @test is_smooth(1) && is_smooth(2) && is_smooth(2 * 3 * 5 * 7)
    @test !is_smooth(11) && !is_smooth(2 * 11) && !is_smooth(13)
    @test next_smooth(1000) == 1000        # 2^3 * 5^3 is already smooth
    @test next_smooth(1001) == 1008        # 1001 = 7*11*13 -> 2^4 * 3^2 * 7
    @test next_smooth(1024) == 1024        # powers of two are smooth
    @test next_smooth(16928) == 17010      # the top harmonic's need in the default plan
    @test is_smooth(next_smooth(12345))
    @test next_smooth(12345) >= 12345
    @test_throws ArgumentError next_smooth(0)
    # Smooth padding is a *lot* tighter than power-of-two padding.
    needs = [2618, 4140, 6240, 8464, 12016, 16928]
    @test all(next_smooth(n) / n < 1.02 for n in needs)
    @test maximum(next_pow_of_2(n) / n for n in needs) > 1.9
end

@testset "finterp_fft is independent of the padded length" begin
    # The FFT-correlation is a *circular* correlation, so any fftlen >= the
    # points it needs gives the same answer.  This is what licenses smooth
    # sizing over the Python original's power-of-two rule.
    N = 32768
    r = 12400.0
    signal = cos.(2π * r .* (0:N-1) ./ N .+ π / 4)
    ft = rfft(signal)
    m = 16
    nb = 8
    numbins = 4
    need = (numbins + m) * nb
    v_p2 = finterp_fft(12400, numbins, nb, ft, m; fftlen=next_pow_of_2(need))
    v_sm = finterp_fft(12400, numbins, nb, ft, m; fftlen=next_smooth(need))
    @test next_smooth(need) != next_pow_of_2(need)     # genuinely different lengths
    @test maximum(abs.(v_p2 .- v_sm)) / maximum(abs.(v_p2)) < 1e-13
    @test_throws ArgumentError finterp_fft(12400, numbins, nb, ft, m; fftlen=need - 1)
end

@testset "finterp_direct == fourier_interp (exact kernel, factored)" begin
    N = 32768
    r = 12400.0
    signal = cos.(2π * r .* (0:N-1) ./ N .+ π / 4)
    ft = rfft(signal)
    for m in (8, 16, 32)
        r0 = 12395.137
        step = 0.25
        n = 41
        got = finterp_direct(r0, n, step, ft, m)
        want = [fourier_interp(r0 + (k - 1) * step, ft, m) for k in 1:n]
        @test maximum(abs.(got .- want)) / maximum(abs.(want)) < 1e-13
    end
    # A trial landing exactly on a Fourier bin: the kernel degenerates to a
    # delta there (A -> 0 while 1/(dr-j) -> Inf), so it must be special-cased.
    exact = finterp_direct(12400.0, 1, 1.0, ft, 16)
    @test isfinite(real(exact[1])) && isfinite(imag(exact[1]))
    @test exact[1] ≈ fourier_interp(12400.0, ft, 16) rtol = 1e-12
    @test_throws ArgumentError finterp_direct(12400.0, 4, 0.25, ft, 15)
end

@testset "irfft convention matches numpy (DC/Nyquist handling)" begin
    # The coherent fold relies on irfft of (nharms+1) complex amplitudes whose
    # Nyquist term generally has a nonzero imaginary part.  numpy's irfft and
    # FFTW's c2r both *ignore* the imaginary parts of the DC and Nyquist bins;
    # this guards that assumption so the search matches the Python oracle.
    nh = 8
    X = ComplexF64.(randn(nh + 1), randn(nh + 1))
    prof = irfft(X, 2nh)
    # Reference: build the full Hermitian spectrum, zeroing DC/Nyquist imag.
    Xc = copy(X)
    Xc[1] = real(Xc[1])
    Xc[end] = real(Xc[end])
    full = vcat(Xc, conj.(reverse(Xc[2:end-1])))
    ref = real.(ifft(full))
    @test prof ≈ ref atol = 1e-10
end
