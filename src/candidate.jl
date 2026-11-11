# Per-candidate pulse-profile reconstruction.
#
# This is the "brute-force, high-accuracy" path called for in
# `docs/Summary_and_Future_Work.md`: for the handful of surviving candidates we do
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
    amps = candidate_harmonics(ft, r, nharm; m=m)
    H = length(amps)
    H >= 1 || return zeros(Float64, 2)                   # nothing usable (degenerate)
    stack = zeros(ComplexF64, H + 1)                     # index 1 is DC, left at 0
    @views stack[2:end] .= amps
    return irfft(stack, 2H)
end

"""
    candidate_harmonics(ft, r, nharm; m=64) -> Vector{ComplexF64}

The exactly interpolated complex amplitudes of harmonics `1, 2, …` of the
Fourier frequency `r` (bins), each from an `m`-bin [`fourier_interp`](@ref),
stopping (not zero-padding) at the first harmonic whose kernel would cross
Nyquist or run past the available amplitudes, or at `nharm`.  Element `h` is
harmonic `h`.  The stack [`candidate_profile`](@ref) inverse-transforms, and the
one [`remove_periodic`](@ref) inspects.
"""
function candidate_harmonics(ft::FFTFile, r::Real, nharm::Integer; m::Integer=64)
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
    return amps
end

# The comb filter's defaults (PRESTO's `CS_PERIODIC_*`, include/coherentsearch.h).
const PERIODIC_NLOW = 4
const PERIODIC_MINHARM = 7
const PERIODIC_RATIO = 8.0

"""
    remove_periodic(ft, cands; nlow=4, minharm=7, ratio=8.0) -> Vector{Candidate}

Drop candidates whose refolded profile is dominated by one high harmonic: the
strongest of its harmonics (from [`candidate_harmonics`](@ref) at the
candidate's own `r` and `nharm`) is number `minharm` or above, and has more than
`ratio` times the power of the strongest of harmonics `1:nlow` — so of each of
them, which means one low harmonic zapped by RFI excision cannot trigger it.

Such a profile is a comb of identical peaks.  It is almost always a strong
sinusoid — a bright pulsar's fundamental or one of its harmonics, or RFI —
landing on a high harmonic of a slow trial, and it lights up many slow trials
at once, which harmonic collapse does not catch because the trials are not
harmonically related to each other in the `numharm` sense.  Candidates with
`nharm < minharm` cannot qualify and are kept.  Keeps the input order.

The C port's `cs_remove_periodic()` (PRESTO, `src/cs_search.c`), same rule and
same defaults.
"""
function remove_periodic(ft::FFTFile, cands::AbstractVector{Candidate};
                         nlow::Integer=PERIODIC_NLOW, minharm::Integer=PERIODIC_MINHARM,
                         ratio::Real=PERIODIC_RATIO)
    keep = trues(length(cands))
    for (i, c) in enumerate(cands)
        (c.nharm >= minharm && c.r > 0) || continue
        amps = candidate_harmonics(ft, c.r, c.nharm)
        lowmax, topmax, toph = 0.0, 0.0, 0
        for (h, a) in enumerate(amps)
            pw = abs2(a)
            h <= nlow && pw > lowmax && (lowmax = pw)
            pw > topmax && ((topmax, toph) = (pw, h))
        end
        keep[i] = !(toph >= minharm && topmax > ratio * lowmax)
    end
    return cands[keep]
end

"""
    read_cohout(path, T) -> Vector{Candidate}

Read a candidate file written by [`main`](@ref) (or by PRESTO's
`coherentsearch`, whose `.cohout` files are the same) back into `Candidate`s,
recovering the fundamental Fourier frequency as `r = freq * T`.  Comment (`#`)
and blank lines are skipped.  Columns are whitespace-delimited: rank, S/N,
frequency (Hz), period (ms), harmonic count, then optionally the duty cycle in
percent (`-` if unmeasured) and a known-pulsar label, which is ignored.
"""
function read_cohout(path::AbstractString, T::Real)
    cands = Candidate[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        cols = split(s)
        length(cols) >= 5 || continue
        metric = parse(Float64, cols[2])
        freq = parse(Float64, cols[3])
        nharm = parse(Int, cols[5])
        ducy = NaN
        if length(cols) >= 6
            d = tryparse(Float64, cols[6])
            d === nothing || (ducy = d / 100)
        end
        push!(cands, Candidate(freq, metric, freq * T, nharm, ducy))
    end
    return cands
end

"""
    measure_ducy(ft, cands, params; m=64) -> Vector{Candidate}

Fill in each candidate's `ducy` — the duty cycle of the best-matching boxcar,
`width / profile bins`, as riptide's `rseek` reports it — and return the updated
candidates.  Input candidates are not modified (`Candidate` is immutable).

Each candidate is refolded at its own `r` with its own `nharm` harmonics, which
reproduces the geometry of the profile the search actually scored: a candidate
found at decimation `k` carries `r = k·r_fund` and `nharm = ⌊nharms/k⌋`, and the
decimated fold is exactly the `⌊nharms/k⌋`-harmonic fold of `k·r_fund`.  So the
`2·nharm`-bin profile here is the same fold, differing only in using the wide
`m`-bin exact kernel instead of the search's grid-snapped interpolation — i.e. a
more accurate estimate of the same thing.  Cheap: this runs on the reported
candidates only, never in the hot loop.

The width bank (`boxcar_fsp`, `boxcar_maxfrac` from `params`) matches the
search's.  Candidates whose profile cannot be reconstructed keep `ducy = NaN`.
"""
function measure_ducy(ft::FFTFile, cands::AbstractVector{Candidate},
                      params::SearchParams; m::Integer=64)
    out = Vector{Candidate}(undef, length(cands))
    for (i, c) in enumerate(cands)
        ducy = NaN
        if c.nharm >= 1 && c.r > 0
            prof = candidate_profile(ft, c.r, c.nharm; m=m)
            # `candidate_profile` truncates at Nyquist, so the profile can be
            # shorter than 2*nharm; the duty cycle is relative to what it returns.
            if length(prof) >= 2
                _, ducy = boxcar_best_width(prof; fsp=params.boxcar_fsp,
                                            maxfrac=params.boxcar_maxfrac)
            end
        end
        out[i] = Candidate(c.freq, c.metric, c.r, c.nharm, ducy)
    end
    return out
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
