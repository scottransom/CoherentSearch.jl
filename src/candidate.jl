# Per-candidate pulse-profile reconstruction.
#
# This is the "brute-force, high-accuracy" path called for in
# `Summary_and_Future_Work.md`: for the handful of surviving candidates we do
# not need the throughput-tuned approximations of the search hot loop.  Each
# harmonic is interpolated once at the candidate's exact Fourier frequency with
# a wide `fourier_interp` kernel, and a single inverse real FFT folds the
# harmonic stack into the pulse profile.  No plotting dependency lives here so
# the reconstruction is unit-tested alongside the rest of the search.

using FFTW: irfft

"""
    candidate_profile(ft, r, nharm; m=64) -> Vector{Float64}

Reconstruct the coherent-fold pulse profile of a candidate whose fundamental
Fourier frequency is `r` (bins), summing as many of the first `nharm` harmonics
as physically fit below the Nyquist frequency.

For each harmonic `h`, the complex amplitude at `r*h` is obtained by a single
high-accuracy [`fourier_interp`](@ref) with an `m`-bin kernel (default `m=64`,
far wider than the search hot loop uses — cheap here since it runs on only a few
candidates).  Harmonics are summed contiguously from `h=1` upward and the sum
**stops** at the first harmonic whose interpolation window would cross the
Nyquist frequency or run off the end of the available amplitudes: the remainder
is *not* zero-padded.  The `H` amplitudes actually available (plus a zero DC
term) are inverse-real-FFT'd into a profile of `2*H` bins.

`nharm` is the number of harmonics *requested* (the search's `--nharms`),
independent of the harmonic-decimation factor `k` that found the candidate — so
every profile is folded at the full requested harmonic depth, much more closely
matching a true time-domain fold of the time series at `r`.  When the requested
depth exceeds Nyquist (fast candidates), only the available `H < nharm`
harmonics are used.  The returned profile is **not** rotated (see
[`rotate_to_peak`](@ref)).
"""
function candidate_profile(ft::FFTFile, r::Real, nharm::Integer; m::Integer=64)
    nharm >= 1 || throw(ArgumentError("nharm must be >= 1"))
    iseven(m) || throw(ArgumentError("m must be even"))

    m2 = m ÷ 2
    Nhalf = ft.N ÷ 2
    namps = length(ft.amps)

    amps = ComplexF64[]
    for h in 1:nharm
        rh = r * h
        # The kernel reads bins floor(rh)-m2+1 .. floor(rh)+m2 (1-based).  Stop
        # (rather than zero-pad) at the first harmonic that would cross Nyquist
        # or run past the available amplitudes; harmonics only get worse with h.
        rint = floor(Int, rh + 1e-15) + 1               # 0-based bin index (as in Python)
        (rint - m2 >= 0 && rint + m2 <= namps && rh < Nhalf) || break
        push!(amps, fourier_interp(rh, ft.amps, m))
    end

    H = length(amps)
    H >= 1 || return zeros(Float64, 2)                   # nothing usable (degenerate)
    stack = zeros(ComplexF64, H + 1)                     # index 1 is DC, left at 0
    @views stack[2:end] .= amps
    return irfft(stack, 2H)
end

"""
    rotate_to_peak(prof) -> Vector{Float64}

Circularly shift a pulse profile so its maximum bin lands at the center
(phase 0.5).  Pulse phase is arbitrary, so this just makes a page of profiles
easy to compare by eye.
"""
function rotate_to_peak(prof::AbstractVector{<:Real})
    n = length(prof)
    n == 0 && return Float64.(prof)
    shift = (n ÷ 2 + 1) - argmax(prof)      # bring argmax to the center bin
    return Float64.(circshift(prof, shift))
end
