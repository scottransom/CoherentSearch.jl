#!/usr/bin/env julia
#
# toy_coherent_search.jl — the coherent harmonic-summing search, written out in
# full, with nothing in it but the algorithm.
#
#     julia --project=. bin/toy_coherent_search.jl FILE.fft [options]
#
# WHAT THIS IS FOR.  `src/search.jl` is the production search: chunk-parallel,
# with tabulated interpolation weights, a batched inverse FFT, a Float32 SIMD
# gate over 128 profiles at a time, a pruned width bank and a subsampled noise
# estimate.  Every one of those is a speed optimisation, and together they make
# the algorithm hard to read.  This file is the same search with all of them
# removed: brute-force per-point Fourier interpolation, one `irfft` per fold,
# the boxcar matched filter evaluated from its definition, and plain
# single-threaded nested loops.  It is the code the paper's pseudo-code figure
# describes, line for line.
#
# Expect it to run roughly 100-200x slower than the production search, and treat
# that as a range rather than a number: it depends on the machine, on the band,
# and on how much of the profile the boxcar bank covers.  Two runs on the same
# host and band gave 190.8x and 177.1x (2026-08-24, i7-10510U, `-t 1`, PM0063
# over 0.1-0.4 Hz: ~243 us per trial fundamental against ~1.27).  It is not meant
# for real work.  What it *is* meant for: reading, and being the independent
# implementation the production path can be checked against (`test/test_toy.jl`,
# `bench/toy_vs_production.jl`).
#
# THE ALGORITHM, as the pseudo-code figure states it.  `F` is the (normalised,
# barycentred) Fourier amplitude array of the observation, `T` its duration.
#
#    1  for each trial fundamental r in [r_lo, r_hi] step dr do        # `trial_grid`
#    2      for h = 1 .. H do                                          # `harmonic_amplitudes`
#    3          A[h] <- interpolate(F, h*r, m)                         #   Eqn. 30
#    4      for each decimation k do                                   # `fold_profile`
#    5          H_k  <- floor(H / k)
#    6          S    <- [0, A[k], A[2k], ..., A[H_k * k]]              #   every k-th harmonic
#    7          P    <- inverse_real_FFT(S)                            #   2*H_k phase bins
#    8          sigma <- noise_scale(P)                                # `profile_sigma`
#    9          best <- -inf
#   10          for each boxcar width w do                             # `boxcar_snr`
#   11              for each phase p do
#   12                  S_w  <- sum of the w bins of P starting at p (wrapping)
#   13                  snr  <- (S_w - (w/n)*S_tot) / (sigma*sqrt(w*(1-w/n)))
#   14                  best <- max(best, snr)
#   15          if best > threshold then
#   16              record candidate at frequency k*r/T with H_k harmonics
#   17  collapse near-duplicate candidates, then harmonically-related ones
#   18  report the strongest `ncands`
#
# Lines 1-16 are this file.  Line 17-18 reuse the production code unchanged
# (`remove_duplicates`, `remove_harmonics`, `measure_ducy`, the candidate
# table), because they are candidate bookkeeping rather than search, and having
# two versions of them would only invite them to drift apart.
#
# References:
#   - Fourier interpolation: Eqn. 30 of Ransom, Eikenberry & Middleditch (2002),
#     https://arxiv.org/pdf/astro-ph/0204349
#   - The boxcar matched filter: Morello et al. (2020), MNRAS 497, 4654, §5.4;
#     numerically riptide's `snr1` (`cpp/snr.hpp`).
#   - Harmonic decimation: `decimation_design.md`.

module ToyCoherentSearch

using CoherentSearch
using CoherentSearch: write_candidates       # not exported; the candidate table
using FFTW: irfft
using Printf: @printf

export toy_search, harmonic_amplitudes, fold_profile, profile_sigma,
       analytic_sigma, mad_sigma, boxcar_snr, trial_grid

# ---------------------------------------------------------------------------
# Line 1: the trial grid
# ---------------------------------------------------------------------------

"""
    trial_grid(ft, lofreq, hifreq, nharms, hidr) -> StepRangeLen

The trial fundamental Fourier frequencies (in bins), `r = f*T`, stepping by
`dr = hidr / nharms`.

The step is set by the *highest* harmonic: harmonic `nharms` of the fundamental
advances by `nharms * dr = hidr` bins per trial, and `hidr = 0.5` — half a
Fourier bin of drift across the observation — is the anti-aliasing constraint
for the whole coherent sum.  Every harmonic below the top is then sampled more
finely than it needs, which is the price of summing them coherently.

A decimation-`k` fold inherits this for free: its fundamental `k*r` steps by
`k*dr`, but its own top harmonic is `H_k = floor(nharms/k)`, so that harmonic
advances by `H_k * k * dr <= nharms * dr = hidr` — never coarser than the base
grid (`decimation_design.md` §2a).
"""
function trial_grid(ft::FFTFile, lofreq::Real, hifreq::Real,
                    nharms::Integer, hidr::Real)
    dr = hidr / nharms
    r_lo = lofreq * ft.T
    r_hi = hifreq * ft.T
    n = max(0, floor(Int, (r_hi - r_lo) / dr) + 1)
    return range(r_lo; step = dr, length = n)
end

# ---------------------------------------------------------------------------
# Lines 2-3: the harmonic amplitudes
# ---------------------------------------------------------------------------

"""
    harmonic_amplitudes(ft, r, nharms, m) -> (amps, navail)

Complex Fourier amplitudes of harmonics `1 .. nharms` of the trial fundamental
`r` (bins), each interpolated from the observation's Fourier transform by
[`fourier_interpolate`](@ref) — Eqn. 30 of Ransom, Eikenberry & Middleditch
(2002), evaluated exactly at `h*r` from the `m` Fourier bins nearest to it.

`navail` is the number of *leading* harmonics that actually carry data.
Availability is monotone in `h`: harmonic `h` sits at `h*r`, so once one
harmonic passes the Nyquist frequency every higher one has too, and
`fourier_interpolate` returns exactly zero for each.  Those zeros are the right
answer — a harmonic with no data contributes nothing to the coherent sum — but
the *count* matters, because it sets how much noise the fold carries (see
[`analytic_sigma`](@ref)).

This is the brute-force interpolation: `m` complex multiply-adds per harmonic
per trial, with the `m` weights recomputed every time.  The production search
computes the identical quantity by tabulating those weights once per harmonic
(`src/directinterp.jl`) — the same arithmetic, several times faster, and the
single biggest difference between this file and `src/search.jl`.
"""
function harmonic_amplitudes(ft::FFTFile, r::Real, nharms::Integer, m::Integer)
    amps = zeros(ComplexF64, nharms)
    navail = 0
    for h in 1:nharms
        a = fourier_interpolate(ft, h * r, m)
        iszero(a) && break              # past Nyquist (or off the end): so is every h above
        amps[h] = a
        navail = h
    end
    return amps, navail
end

# ---------------------------------------------------------------------------
# Lines 4-7: the coherent fold, and harmonic decimation
# ---------------------------------------------------------------------------

"""
    fold_profile(amps, k, nharms) -> (profile, nfilled)

The coherent pulse profile at `k` times the trial fundamental, folded from the
harmonic amplitudes `amps` of that fundamental.

**The base fold (`k = 1`) is the search.**  Stack the harmonic amplitudes into a
half-complex spectrum with the DC term held at zero, inverse-real-FFT it, and
what comes out is the pulse profile the observation would give if folded at
period `T/r` — with `2*nharms` phase bins, because `nharms` harmonics plus DC
inverse-transform to `2*nharms` real points.  Holding DC at zero is what makes
the profile's mean exactly zero, which the detection metric relies on.

**Decimation (`k > 1`) is nearly free, and it is why the search reaches fast
pulsars.**  A signal at fundamental `k*r` has its harmonics at `k*r, 2k*r, ...`
— Fourier frequencies at which we have *already* interpolated amplitudes, as
base harmonics `k, 2k, ...`.  So taking every `k`-th element of `amps` gives the
harmonic stack of the multiple: `H_k = floor(nharms/k)` harmonics, folded into
`2*H_k` bins, at the cost of one short inverse FFT and no interpolation at all.
Shallower folds suit faster pulsars, which tend to have wider duty cycles and so
need fewer harmonics — the same trade riptide's FFA makes by downsampling
(`decimation_design.md`).

`nfilled` is how many of the `H_k` stacked harmonics carry data; the rest are
past Nyquist and stay zero.
"""
function fold_profile(amps::Vector{ComplexF64}, k::Integer, nharms::Integer)
    Hk = nharms ÷ k
    stack = zeros(ComplexF64, Hk + 1)       # stack[1] is DC, deliberately left at 0
    nfilled = 0
    for j in 1:Hk
        stack[j + 1] = amps[j * k]          # base harmonic j*k IS harmonic j of k*r
        iszero(stack[j + 1]) || (nfilled = j)
    end
    return irfft(stack, 2Hk), nfilled
end

# ---------------------------------------------------------------------------
# Line 8: the noise scale
# ---------------------------------------------------------------------------

"""
    analytic_sigma(nbins, nfilled) -> Float64

The per-bin noise standard deviation of a folded profile, computed rather than
measured.

The input `.fft` must be normalised (Fourier powers with mean 1) for the search
to mean anything at all, and that assumption already fixes the noise in the
fold: mean power 1 means the real and imaginary part of each amplitude have
variance 1/2 each.  `irfft` of a stack holding `H` harmonics plus a zero DC
gives, at every phase bin `j`,

    P_j = (1/n) * [ 2*sum_{h<H} (a_h cos(2*pi*h*j/n) - b_h sin(2*pi*h*j/n))
                    + a_H * (-1)^j ],        n = 2H

so with `var(a_h) = var(b_h) = 1/2` each harmonic below the top contributes
`4 * (1/2) = 2` to `n^2 * var(P_j)`, and the top harmonic — which lands on the
profile's own Nyquist bin, where `irfft` keeps only the real part — contributes
`1/2`.  Hence

    sigma = sqrt(2*nlow + 0.5*nnyq) / nbins

which is `1/sqrt(nbins)` to within a factor `sqrt(1 - 3/(4H))` when the stack is
full.  That correction is not decoration: it is 0.6% at `H = 60` but **3.8% at
`H = 10`**, the depth of a `k = 6` decimated fold, so dropping it would bias the
shallow folds against the deep ones — exactly the cross-decimation bias the
boxcar metric was chosen to avoid.

`nfilled` is the number of harmonics that actually carry data
([`fold_profile`](@ref)).  Near the top of the searched band most of a stack is
past Nyquist and zero, so those bins carry no noise either; using the full `H`
there would overestimate `sigma` badly and silently suppress fast candidates.

**Two known biases, both small, neither corrected here.**  (i) The `m`-bin
interpolation kernel recovers `S_m = sum_j sinc^2(dr - j) ~ 1 - 0.203/m` of the
noise power along with the signal, so this runs ~0.6% high at `m = 16` — a
constant factor, not scatter.  (ii) It assumes the normalisation *holds*, which
a measured estimate would not: red-noise residue and RFI move the true local
noise level and this cannot see it.  [`mad_sigma`](@ref) is the measured
alternative, and `bench/toy_vs_production.jl` reports the two side by side.
"""
function analytic_sigma(nbins::Integer, nfilled::Integer)
    Hk = nbins ÷ 2
    nfilled >= 1 || return 0.0
    nnyq = (nfilled == Hk) ? 1 : 0          # is the profile's own Nyquist bin filled?
    nlow = nfilled - nnyq                   # harmonics below it, which carry 4x the variance
    return sqrt(2 * nlow + 0.5 * nnyq) / nbins
end

"""
    mad_sigma(prof) -> Float64

The per-bin noise scale *measured* from the profile itself: `1.4826 * median(|P|)`,
the median absolute deviation about the structural zero, scaled to a Gaussian
standard deviation.

Folding about zero rather than about the sample median is exact here, not an
approximation: the DC term is held at zero in [`fold_profile`](@ref), so the
profile's mean is identically zero and the only thing left to estimate is scale.

This is what the production search does — with one difference that matters.  It
pools the estimate over a whole chunk of ~2048 profiles at once, where this sees
a single profile of 20-120 bins, so this estimate carries a `~0.76/sqrt(nbins)`
sampling error: 7% at 120 bins, 17% at 20.  Reported S/N is exactly `1/sigma`,
so that error lands directly on every candidate.  Pooling is what production
buys with its chunk structure; the analytic scale above avoids the question.
"""
function mad_sigma(prof::Vector{Float64})
    a = sort!(abs.(prof))
    n = length(a)
    med = isodd(n) ? a[n ÷ 2 + 1] : 0.5 * (a[n ÷ 2] + a[n ÷ 2 + 1])
    return 1.4826 * med
end

"""
    profile_sigma(mode, prof, nbins, nfilled) -> Float64

Dispatch between [`analytic_sigma`](@ref) (`:analytic`, the default) and
[`mad_sigma`](@ref) (`:mad`).  Kept as one call so the two can be A/B'd without
touching the search loop.
"""
function profile_sigma(mode::Symbol, prof::Vector{Float64}, nbins::Integer, nfilled::Integer)
    mode === :analytic && return analytic_sigma(nbins, nfilled)
    mode === :mad && return mad_sigma(prof)
    throw(ArgumentError("sigma mode must be :analytic or :mad, got :$mode"))
end

# ---------------------------------------------------------------------------
# Lines 9-14: the detection metric
# ---------------------------------------------------------------------------

"""
    boxcar_snr(prof, sigma) -> Float64

The peak boxcar matched-filter S/N of the pulse profile `prof`, given the
per-bin noise scale `sigma`.  This is the detection statistic: one number per
folded profile, and the only thing the search thresholds on.

The filter is a top-hat of width `w` bins made **zero-mean and unit-L2** —
height `h = sqrt((n-w)/(n*w))` on the `w` on-pulse bins and `-b` with
`b = (w/(n-w))*h` on the rest.  Correlating that template against the profile
and dividing by `sigma` gives, after substituting `b = delta*(h+b)` with
`delta = w/n`,

    snr(w, p) = (S_w - delta*S_tot) / (sigma * sqrt(w * (1 - delta)))

where `S_w` is the sum of the `w` profile bins starting at phase `p` (wrapping
past the end, since pulse phase is circular) and `S_tot` is the profile total.
The reported statistic is the maximum over every width and every phase.

**Why this template and not a plain on-pulse sum.**  Because the widths are a
bank fixed in advance and the template is normalised, every `(phase, width)`
trial is exactly `N(0,1)` under white noise.  The pure-noise distribution of the
peak is therefore analytic, with a trials factor that barely depends on the
profile length — so a single `--threshold` means the same false-alarm rate at
every width *and* every decimation, which is what makes candidates from folds of
20 and 120 bins comparable at all.  It is also numerically riptide's `snr1`, so
the two codes' S/N columns are the same quantity.

The width bank is [`boxcar_widths`](@ref): `w = 1, 2, 3, 4, 6, 9, 13, ...`
(`w_{k+1} = max(floor(1.5*w_k), w_k+1)`), stopping at 30% duty cycle.  The
geometric spacing is dense enough that the worst-case mismatch loss between
adjacent widths is small, and 30% is past the widest duty cycle worth a
matched filter.

**This is the definition, evaluated literally**, at `O(nbins * sum(w))` per
profile — the toy's dominant cost by a wide margin.  Production computes the
identical maximum from a phase-wrapped prefix sum (`S_w = psum[p+w] - psum[p]`,
so each width costs one pass rather than `w`), batched across 128 profiles at
once so the phase scan vectorises.  That is a rearrangement, not a different
statistic.
"""
function boxcar_snr(prof::Vector{Float64}, sigma::Float64)
    nbins = length(prof)
    sigma > 0 || return -Inf                 # degenerate fold: nothing to detect
    stot = sum(prof)
    best = -Inf
    for w in boxcar_widths(nbins)
        delta = w / nbins
        norm = sigma * sqrt(w * (1 - delta))
        for p in 1:nbins
            sw = 0.0
            for i in 0:(w - 1)
                sw += prof[mod1(p + i, nbins)]      # wraps: pulse phase is circular
            end
            snr = (sw - delta * stot) / norm
            snr > best && (best = snr)
        end
    end
    return best
end

# ---------------------------------------------------------------------------
# Lines 1-16: the search
# ---------------------------------------------------------------------------

"""
    toy_search(ft; kwargs...) -> Vector{Candidate}

Run the whole search over `[lofreq, hifreq]` Hz and return every trial that
scored above `threshold`, before any candidate collapsing.

Keywords: `nharms`, `m`, `hidr`, `lofreq`, `hifreq`, `maxdecim`, `threshold`,
`sigma` (`:analytic` or `:mad`), `progress`.

This is lines 1-16 of the figure and nothing else: three nested loops (trial,
decimation, then width and phase inside [`boxcar_snr`](@ref)), single-threaded,
allocating freely.  `hifreq` bounds the *fundamental*; decimation carries the
searched band up to `maxdecim * hifreq`.
"""
function toy_search(ft::FFTFile;
                    nharms::Integer = 60, m::Integer = 16, hidr::Real = 0.5,
                    lofreq::Real = 0.1, hifreq::Real = 125.0,
                    maxdecim::Integer = 6, threshold::Real = 8.0,
                    sigma::Symbol = :analytic, progress::Bool = true)
    rs = trial_grid(ft, lofreq, hifreq, nharms, hidr)
    ks = decimation_set(nharms, maxdecim)      # 1:maxdecim, keeping H_k >= 2
    nyquist = ft.N / 2
    cands = Candidate[]

    for (i, r) in enumerate(rs)                                        # line 1
        amps, navail = harmonic_amplitudes(ft, r, nharms, m)           # lines 2-3
        navail >= 1 || continue                # not one harmonic below Nyquist

        for k in ks                                                    # line 4
            k * r < nyquist || continue        # the decimated fundamental itself
            prof, nfilled = fold_profile(amps, k, nharms)              # lines 5-7
            nfilled >= 1 || continue
            nbins = length(prof)

            sig = profile_sigma(sigma, prof, nbins, nfilled)           # line 8
            snr = boxcar_snr(prof, sig)                                # lines 9-14

            if snr > threshold                                         # line 15
                # Reported at k*r: the searched frequency is the decimated
                # fundamental, and `nharm` records which fold found it
                # (k = nharms / nharm).
                push!(cands, Candidate(k * r / ft.T, snr, k * r, nharms ÷ k))
            end                                                        # line 16
        end
        progress && _tick(i, length(rs))
    end
    return cands
end

# Progress meter: not part of the algorithm, and the only reason `toy_search`
# knows what a terminal is.  A full-band toy run takes hours.
function _tick(i::Integer, n::Integer)
    step = max(1, n ÷ 200)
    (i % step == 0 || i == n) || return
    @printf(stderr, "\r  Searching: %3d%%  (%d/%d trials)", round(Int, 100i / n), i, n)
    i == n && println(stderr)
    flush(stderr)
    return
end

# ---------------------------------------------------------------------------
# Lines 17-18, and the command line
# ---------------------------------------------------------------------------

const USAGE = """
Usage: julia --project=. bin/toy_coherent_search.jl FILE.fft [options]

  A deliberately simple, single-threaded reference implementation of the
  coherent harmonic-summing search.  Candidates are written to stdout.
  Roughly 100-200x slower than bin/coherent_search.jl, depending on the
  machine and the band -- use a narrow one.

Options (defaults match bin/coherent_search.jl):
  --threshold X   S/N cutoff for reporting candidates          [8.0]
  --nharms N      harmonics coherently summed in the base fold [60]
  --m N           Fourier bins in the interpolation kernel     [16]
  --ncands N      most candidates to report                    [100]
  --lofreq X      lowest fundamental frequency, Hz             [0.1]
  --hifreq X      highest fundamental frequency, Hz            [125.0]
  --hidr X        Fourier-bin step at the highest harmonic     [0.5]
  --drtol X       bin tolerance for collapsing duplicates      [1.0]
  --maxdecim N    highest harmonic-decimation factor k         [6]
  --sigma MODE    noise scale: analytic | mad                  [analytic]
  --noprogress    do not print the progress meter
"""

function parse_cmdline(argv)
    o = Dict{String,Any}("threshold" => 8.0, "nharms" => 60, "m" => 16,
                         "ncands" => 100, "lofreq" => 0.1, "hifreq" => 125.0,
                         "hidr" => 0.5, "drtol" => 1.0, "maxdecim" => 6,
                         "sigma" => "analytic", "progress" => true, "fftfile" => "")
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a in ("-h", "--help")
            print(USAGE); exit(0)
        elseif a == "--noprogress"
            o["progress"] = false
        elseif startswith(a, "--")
            key = a[3:end]
            haskey(o, key) || error("unknown option $a\n\n$USAGE")
            i += 1
            i <= length(argv) || error("$a needs a value")
            o[key] = o[key] isa Float64 ? parse(Float64, argv[i]) :
                     o[key] isa Int     ? parse(Int, argv[i]) : argv[i]
        else
            isempty(o["fftfile"]) || error("only one FFT file at a time\n\n$USAGE")
            o["fftfile"] = a
        end
        i += 1
    end
    isempty(o["fftfile"]) && error("no FFT file given\n\n$USAGE")
    o["m"] > 0 && iseven(o["m"]) || error("--m must be a positive even number")
    o["sigma"] in ("analytic", "mad") || error("--sigma must be analytic or mad")
    return o
end

function main(argv)
    o = parse_cmdline(argv)
    ft = FFTFile(o["fftfile"])
    @printf(stderr, "# %s: T = %.2f s, N = %d, %d harmonics, m = %d, k = 1..%d, sigma = %s\n",
            basename(ft.path), ft.T, ft.N, o["nharms"], o["m"], o["maxdecim"], o["sigma"])

    elapsed = @elapsed cands = toy_search(ft;
        nharms = o["nharms"], m = o["m"], hidr = o["hidr"],
        lofreq = o["lofreq"], hifreq = o["hifreq"], maxdecim = o["maxdecim"],
        threshold = o["threshold"], sigma = Symbol(o["sigma"]),
        progress = o["progress"])
    @printf(stderr, "# searched in %.2f s; %d trials above threshold\n", elapsed, length(cands))

    # Line 17: collapse the run of adjacent trials one signal lights up, then the
    # f/2, 2f, 3f/2, ... family it lights up across decimations.  Both are the
    # production functions, used unmodified.
    cands = remove_duplicates(cands; dr_tol = o["drtol"])
    cands = remove_harmonics(cands; numharm = 16)

    # Line 18: strongest first, truncated, with each survivor's best-fitting
    # boxcar duty cycle filled in by refolding it (the search loop above throws
    # away *which* width won, since only reported candidates need it).
    sort!(cands; by = c -> c.metric, rev = true)
    length(cands) > o["ncands"] && (cands = cands[1:o["ncands"]])
    isempty(cands) || (cands = measure_ducy(ft, cands,
                                            SearchParams(nharms = o["nharms"], m = o["m"])))
    write_candidates(cands, "", o["threshold"])       # "" => stdout
    return nothing
end

end # module ToyCoherentSearch

if abspath(PROGRAM_FILE) == @__FILE__
    ToyCoherentSearch.main(ARGS)
end
