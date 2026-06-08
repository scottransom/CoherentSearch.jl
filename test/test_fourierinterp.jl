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
