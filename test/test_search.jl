using Test
using CoherentSearch
using Random

# The optimised chunk/plan-caching path must agree with the simple reference
# `block_metrics` (which is itself pinned to the Python oracle to ~1e-15), and
# must actually detect the 10.0123 Hz test pulsar.  These run only when the
# example data from the sibling Python repo is present.
const EXAMPLE_FFT = joinpath(@__DIR__, "..", "..", "coherent_search",
                             "examples", "harmonics_hi.fft")

# Tolerance for the optimised-vs-reference equivalence pins.  The profile stage's
# precision is a runtime choice (`SearchParams.precision`), and these pins run at
# the default `:f64`, where they are machine-precision pins.  `PIN_PRECISION` is
# the knob a `:f32` sweep flips; the bound is *derived* from it rather than
# relaxed by hand, and a real regression is orders of magnitude larger than
# either bound.
const PIN_PRECISION = :f64
const PIN_TOL = PIN_PRECISION === :f32 ? 1e-6 : 1e-8
const PIN_PROFT = PIN_PRECISION === :f32 ? Float32 : Float64


@testset "boxcar_widths: geometric bank capped at maxfrac*nbins" begin
    w = boxcar_widths(64; fsp=1.5, maxfrac=0.3)
    @test w == [1, 2, 3, 4, 6, 9, 13, 19]        # riptide's wₖ₊₁=max(⌊1.5wₖ⌋,wₖ+1)
    @test w[1] == 1 && issorted(w) && allunique(w)
    @test w[end] <= floor(Int, 0.3 * 64)
    @test all(w[i+1] == max(floor(Int, 1.5 * w[i]), w[i] + 1) for i in 1:length(w)-1)
    @test boxcar_widths(4) == [1]                # tiny profile keeps only width-1
    @test boxcar_widths(64; maxfrac=0.5)[end] <= 32
end

@testset "ladder_boxcar_widths: prunes the redundant (k,W) corners" begin
    nharms = 60
    # No ladder => untouched.  This is what keeps the equivalence gate (which
    # runs decimations=[1]) byte-identical, so it is a pin, not a nicety.
    solo = SearchParams(nharms=nharms)
    @test solo.decimations == [1]
    @test ladder_boxcar_widths(2nharms, 1, solo) == boxcar_widths(2nharms)

    p = SearchParams(nharms=nharms, decimations=decimation_set(nharms, 6))
    @test p.decimations == 1:6
    banks = Dict(k => ladder_boxcar_widths(2 * fld(nharms, k), k, p) for k in p.decimations)
    # Deep fold keeps the narrow end (nothing deeper covers width 1) and drops
    # the wide end (the shallow folds cover those duties without carrying 50
    # harmonics of noise); shallow fold is the mirror image.
    @test banks[1] == [1, 2, 3, 4, 6]
    @test banks[6] == [2, 3, 4, 6]
    for k in 2:6
        @test 1 ∉ banks[k]                          # covered by W=2 one rung deeper
    end
    for k in 1:5
        @test maximum(banks[k]) <= 6                # wide duties belong to k=6
    end
    # Only the shallowest fold reaches boxcar_maxfrac, and it must, or the
    # widest duty cycles would go unsearched by the ladder as a whole.
    @test maximum(banks[6]) == maximum(boxcar_widths(2 * fld(nharms, 6)))
    # Every fold keeps a usable bank, and the pruning is a strict subset.
    for k in p.decimations
        full = boxcar_widths(2 * fld(nharms, k))
        @test !isempty(banks[k]) && banks[k] ⊆ full
    end
    @test sum(2 * fld(nharms, k) * length(banks[k]) for k in p.decimations) <
          sum(2 * fld(nharms, k) * length(boxcar_widths(2 * fld(nharms, k)))
              for k in p.decimations)

    # A sparse ladder leaves a duty-cycle hole between the cap of one rung and
    # the floor of the next, so it must not be pruned at all.
    sparse = SearchParams(nharms=nharms, decimations=[1, 6])
    @test ladder_boxcar_widths(2nharms, 1, sparse) == boxcar_widths(2nharms)
    @test ladder_boxcar_widths(2 * fld(nharms, 6), 6, sparse) ==
          boxcar_widths(2 * fld(nharms, 6))
    # ...but a ladder that only skips within the safe ratio still prunes.
    @test ladder_boxcar_widths(2nharms, 1, SearchParams(nharms=nharms,
                                                        decimations=[1, 2, 4])) !=
          boxcar_widths(2nharms)
end

@testset "boxcar metric: scale-invariant, robust, detects a pulse" begin
    nbins = 64

    # A ratio of two linear-in-amplitude quantities: invariant to overall scale
    # (this is why the unnormalised brfft hot path and the normalised reference
    # irfft yield the identical value, with no `scale`/`ngoodbins` correction).
    ramp = collect(1.0:nbins)
    m1 = snr_metrics(reshape(ramp, nbins, 1))[1]
    m2 = snr_metrics(reshape(ramp .* 7.0, nbins, 1))[1]
    @test m1 ≈ m2 rtol=1e-12
    @test m1 > 0

    # A flat profile has zero MAD -> guarded to 0.0, not NaN/Inf.
    @test snr_metrics(reshape(fill(3.0, nbins), nbins, 1))[1] == 0.0

    # A narrow pulse on Gaussian noise scores far above the noise-only profile,
    # and the pure-noise peak-over-trials sits at a few sigma (analytic EVD).
    noise = randn(MersenneTwister(1234), nbins)
    snr_noise = snr_metrics(reshape(copy(noise), nbins, 1))[1]
    sig = copy(noise); sig[30] += 20.0
    snr_sig = snr_metrics(reshape(sig, nbins, 1))[1]
    @test snr_sig > snr_noise + 10
    @test 0 < snr_noise < 8
    @test snr_sig > 15

    # σ̂ subsampling: the default is the exact pooled MAD (what the Python oracle
    # computes), and it must agree with the production estimator whenever the block
    # is small enough that no subsampling happens.  This is the relationship
    # `block_metrics` relies on to be both oracle-faithful and an exact
    # equivalence partner for `chunk_metrics`.
    small = randn(MersenneTwister(7), nbins, 4)         # nbins*L well under the cap
    @test snr_metrics(small) == snr_metrics(small; sigma_samples=CoherentSearch._BOXCAR_SIGMA_SAMPLES)
end

# riptide's `cpp/snr.hpp:snr1`, written out longhand: correlate against the
# width-`w` boxcar made zero-mean and unit-L2, maximise over phase and width.
# This is the definition our `_boxcar_scan` and the Python oracle's `snr_metric`
# both implement in prefix-sum form.
function _snr1_longhand(prof, widths, stdnoise)
    n = length(prof)
    stot = sum(prof)
    best = -Inf
    for w in widths
        h = sqrt((n - w) / (n * w))                 # boxcar height  = +h
        b = w / (n - w) * h                         # boxcar baseline = -b
        dmax = maximum(sum(prof[mod1(p + i, n)] for i in 0:(w - 1)) for p in 1:n)
        best = max(best, ((h + b) * dmax - b * stot) / stdnoise)
    end
    return best
end

@testset "the metric IS riptide's snr1 (zero-mean, unit-L2 boxcar template)" begin
    CS = CoherentSearch

    # The two properties that define the template, and that make the statistic
    # unit-variance per (phase, width).  If either fails, the normalisation below
    # is not a matched filter and no threshold has a calculable false-alarm rate.
    for nbins in (20, 30, 64, 120), w in boxcar_widths(nbins)
        h = sqrt((nbins - w) / (nbins * w))
        b = w / (nbins - w) * h
        t = fill(-b, nbins); t[1:w] .= h
        @test abs(sum(t)) < 1e-13                   # zero mean
        @test isapprox(sum(abs2, t), 1.0; rtol=1e-13)   # unit L2
    end

    # `snr_metrics` against that definition, on profiles built the way the search
    # builds them (DC held at zero) plus one carrying a real pulse.  This is the
    # pin that says our prefix-sum form and riptide's template form agree; it was
    # measured against the riptide binary itself at 1.4e-7, which is riptide's own
    # Float32 accumulation, so the tolerance here is machine precision instead.
    rng = MersenneTwister(20260824)
    for nbins in (20, 30, 120)
        P = randn(rng, nbins, 32)
        P .-= sum(P, dims=1) ./ nbins
        P[3:5, 7] .+= 6.0
        widths = boxcar_widths(nbins)
        sigma = CS._block_sigma(copy(P), nbins, size(P, 2), Vector{Float64}(undef, length(P)))
        got = snr_metrics(P)
        want = [_snr1_longhand(view(P, :, j), widths, sigma) for j in 1:size(P, 2)]
        @test maximum(abs.(got .- want)) < 1e-12 * maximum(abs.(want))
    end

    # Unit variance per (phase, width), which is the whole point of the change.
    # Checked on the longhand template at a *fixed* phase (the peak over phase is
    # an extreme value, not N(0,1)); the pin above ties our code to it.  The old
    # `median baseline / σ√w` metric gave 0.951 here at w = 28, which this
    # tolerance would catch.
    nbins = 120
    N = 20_000
    rng2 = MersenneTwister(31415)
    X = randn(rng2, nbins, N)
    X .-= sum(X, dims=1) ./ nbins                   # DC held at zero
    for w in (1, 4, 13, 28)
        h = sqrt((nbins - w) / (nbins * w))
        b = w / (nbins - w) * h
        z = [(h + b) * sum(view(X, 1:w, j)) - b * sum(view(X, :, j)) for j in 1:N]
        sd = sqrt(sum(abs2, z .- sum(z) / N) / (N - 1))
        @test isapprox(sd, 1.0; atol=0.03)
    end

    # Invariant to any constant baseline already removed from the profile — which
    # is what lets the hot loop prefix-sum against 0 while `snr_metrics` removes
    # the mean for conditioning, and still get the same number.
    prof = randn(MersenneTwister(5), 60)
    widths = boxcar_widths(60)
    psum = Vector{Float64}(undef, 60 + widths[end] + 1)
    vals = map((0.0, sum(prof) / 60, 37.0, -1.0e3)) do base
        CS._boxcar_psum!(psum, prof, 60, widths[end], base)
        CS._boxcar_scan(psum, widths, 60, 1.0)
    end
    @test all(isapprox(v, vals[1]; rtol=1e-10) for v in vals)
end

@testset "remove_duplicates collapses clusters" begin
    # Two tight clusters (near-identical r) plus one isolated candidate; each
    # cluster should collapse to its single strongest member.
    mk(r, s) = Candidate(r / 1000.0, s, r, 32)   # T=1000 so freq=r/1000
    cands = [mk(10000.0, 8.5), mk(10000.02, 12.0), mk(10000.05, 9.0),  # cluster A
             mk(20000.0, 7.0),                                          # isolated
             mk(30000.1, 15.0), mk(30000.2, 11.0)]                      # cluster B
    kept = remove_duplicates(cands; dr_tol=1.0)
    @test length(kept) == 3
    @test issorted(kept; by=c -> c.freq)
    metrics = sort([c.metric for c in kept])
    @test metrics ≈ [7.0, 12.0, 15.0]

    # A larger tolerance merges everything within range; empty input is empty.
    @test length(remove_duplicates(cands; dr_tol=1e9)) == 1
    @test isempty(remove_duplicates(Candidate[]))
end

@testset "remove_harmonics collapses harmonic families" begin
    mk(r, s) = Candidate(r / 1000.0, s, r, 16)
    # A family around r0=10000 (r0/2, r0, 3r0/2, 2r0): keep only the strongest.
    fam = [mk(5000.0, 9.0), mk(10000.0, 20.0), mk(15000.0, 8.5), mk(20000.0, 12.0)]
    kept = remove_harmonics(fam; numharm=16, tol=1.0)
    @test length(kept) == 1
    @test kept[1].r == 10000.0 && kept[1].metric == 20.0

    # An unrelated candidate (13337/10000 = 1.3337, no small n/m) survives.
    mixed = vcat(fam, [mk(13337.0, 11.0)])
    kept2 = remove_harmonics(mixed; numharm=16, tol=1.0)
    @test length(kept2) == 2
    @test issorted(kept2; by=c -> c.freq)
    @test Set(round(Int, c.r) for c in kept2) == Set([10000, 13337])

    # numharm bounds the ratios tested: with numharm=1 only exact (1/1, i.e.
    # near-identical) matches collapse, so the whole spread-out family survives.
    @test length(remove_harmonics(fam; numharm=1, tol=1.0)) == 4
    @test isempty(remove_harmonics(Candidate[]))
end

if isfile(EXAMPLE_FFT)
    ft = FFTFile(EXAMPLE_FFT)

    @testset "optimised path reproduces the exact-kernel reference" begin
        # The end-to-end equivalence gate: `chunk_metrics` (chunking, cached
        # tables, batched brfft, gated boxcar) against `block_metrics` (one
        # allocating pass, per-point interpolation, exact median).
        #
        # `kernel=:direct` makes the reference evaluate Eqn. 30 point by point, so
        # both sides compute the *same* quantity and the residual is tabulation
        # rounding rather than method.  This replaces an older pin that ran both
        # sides through the FFT-correlation interpolator: that agreed at 7e-16,
        # but only because both sides shared an interpolator carrying a ~1e-2
        # error, so it could not have caught the interpolator being wrong.  This
        # one is looser in the number and stronger in what it asserts.
        params = SearchParams(nharms=32, m=32)
        lodr = params.hidr / params.nharms
        rstart = 10010.0
        n = 256
        rfund = rstart .+ (0:n-1) .* lodr

        # `weights=Float64` is what keeps this a machine-precision statement.
        # The search itself defaults to `Float32` interpolation weights, a
        # deliberate ~2e-7 speed trade (see `build_direct_plans`); measuring the
        # optimised path at `Float32` here would fold that constant into a pin
        # whose whole job is to catch tabulation bugs at 1e-8.  The `Float32`
        # path has its own pin below.
        ref = block_metrics(ft, rfund, params; kernel=:direct)
        opt = chunk_metrics(ft, params, rstart, n; lodr=lodr, weights=Float64)
        relerr = maximum(abs.(opt .- ref)) / maximum(abs.(ref))
        @info "exact-kernel reference agreement" relerr
        @test relerr < 1e-8

        # The shipped default, against the same reference.  Loose by the pin
        # above's standards and tight by the search's: two orders of magnitude
        # under the ~1.3% of signal power the m-truncation discards anyway.
        optf32 = chunk_metrics(ft, params, rstart, n; lodr=lodr, weights=Float32)
        relf32 = maximum(abs.(optf32 .- ref)) / maximum(abs.(ref))
        @info "Float32-weight agreement (the shipped default)" relf32
        @test relf32 < 1e-6

        # And the FFT-correlation reference — the Python-oracle form — is much
        # further away, which is the accuracy the production interpolator buys.
        reffft = block_metrics(ft, rfund, params; kernel=:fft)
        relfft = maximum(abs.(opt .- reffft)) / maximum(abs.(reffft))
        @info "fft-kernel reference agreement (the approximation)" relfft
        @test relfft > 1e3 * relerr
    end

    @testset "direct interpolation is the exact kernel" begin
        # The strong pin for the default path.  `fourier_interp` is Eqn. 30
        # evaluated point by point and is itself pinned to the Python oracle at
        # ~3e-16, so agreeing with it *is* agreeing with the oracle — a stronger
        # statement than the FFT path's equivalence, which inherits that path's
        # linear-interpolation error.
        nharms = 60
        params = SearchParams(nharms=nharms, m=32)
        lodr = params.hidr / nharms
        # Low enough that harmonic 60 stays inside the (small) bundled test FFT,
        # so the per-point reference can be evaluated for every harmonic.
        rstart = 5000.0
        n = 256
        # Both weight types, against the same exact reference.  `Float64` is the
        # machine-precision statement about the tabulation; `Float32` is the
        # shipped default and its own documented trade, so it gets its own
        # tolerance rather than relaxing the first one.
        for (WT, tol) in ((Float64, 1e-7), (Float32, 1e-6))
            ws = CoherentSearch.Workspace(params, n)
            dplans = build_direct_plans(WT, params, rstart)
            CoherentSearch.fill_chunk_profiles!(ws, dplans, ft, params, rstart, lodr, n; t0=0)
            worst = 0.0
            for h in (1, 2, 7, 15, 30, 59, 60), j in (1, 2, 97, n)
                r = h * (rstart + (j - 1) * lodr)
                v = fourier_interp(r, ft.amps, params.m)
                worst = max(worst, abs(ws.ftprofs[h + 1, j] - v) / abs(v))
            end
            @info "direct interpolation vs exact fourier_interp" WT worst
            @test worst < tol     # relative, and largest where |amp| ~ 0
        end

    end

    @testset "direct interpolation is invariant to chunking" begin
        # The phase/bin bookkeeping keys off a *global* trial index, so a chunk
        # starting at global trial t0 must reproduce what one long chunk gives at
        # the same trials.  This is the property that a naive float-accumulated
        # `rstart` would drift on.
        nharms = 32
        params = SearchParams(nharms=nharms, m=32)
        lodr = params.hidr / nharms
        r_lo = 10010.0
        dplans = build_direct_plans(params, r_lo)
        nbig = 384
        wb = CoherentSearch.Workspace(params, nbig)
        CoherentSearch.fill_chunk_profiles!(wb, dplans, ft, params, r_lo, lodr, nbig; t0=0)
        # Chunk sizes deliberately spanning both cases: multiples of
        # `DIRECT_GROUP_V` (so the trials-axis groups tile each chunk exactly) and
        # NON-multiples (so both end groups hang off the chunk and are computed in
        # full, then masked on store).  The second case is the one the group
        # kernel is built to survive: if groups were anchored to the chunk rather
        # than to the global trial index, or if partial groups fell back to a
        # different kernel, these would differ in the last bits and the pin below
        # would have to be loosened to ~1e-16.
        V = CoherentSearch.DIRECT_GROUP_V
        worst = 0.0
        misaligned = 0
        for nsm in (128, 100, 37, 51)
            nsm % V == 0 || (misaligned += 1)
            ws2 = CoherentSearch.Workspace(params, nsm)
            t0 = 0
            while t0 + nsm <= nbig
                CoherentSearch.fill_chunk_profiles!(ws2, dplans, ft, params,
                                                    r_lo + t0 * lodr, lodr, nsm; t0=t0)
                for h in (1, 5, 32), j in 1:nsm
                    a = ws2.ftprofs[h + 1, j]
                    b = wb.ftprofs[h + 1, t0 + j]
                    worst = max(worst, abs(a - b) / abs(b))
                end
                t0 += nsm
            end
        end
        @info "direct interpolation chunk invariance" worst V misaligned
        @test misaligned == 3     # the sizes above really do straddle groups
        @test worst == 0.0        # identical arithmetic, not merely close
    end

    @testset "boxcar metric detects the 10.0123 Hz pulsar" begin
        params = SearchParams(nharms=32, m=32, numbetween=16)
        cands = search(ft, params; lofreq=9.5, hifreq=10.5, threshold=8.0)
        @test !isempty(cands)
        best = cands[argmax(c.metric for c in cands)]
        @info "boxcar strongest candidate" best.freq best.metric
        @test isapprox(best.freq, 10.0123; atol=1e-2)
    end

    @testset "boxcar fast gate does not change the candidate list" begin
        # The production :boxcar path scores ~99% of trials with only the
        # batched `Float32` kernel, and re-scores in `Float64` only those that
        # land within `boxcar_gatemargin` of `threshold`.
        # `boxcar_gatemargin = Inf` forces `exactcut = -Inf`, i.e. the `Float64`
        # scalar path for every trial — so the two runs must produce
        # byte-identical candidates.  This is the pin for both the gate itself
        # and the cross-profile SIMD batching behind it.
        gated = SearchParams(nharms=60, m=32,
                             decimations=decimation_set(60, 4))
        exact = SearchParams(nharms=60, m=32,
                             decimations=decimation_set(60, 4),
                             boxcar_gatemargin=Inf)
        kw = (lofreq=9.5, hifreq=10.5, threshold=6.0, blocksize=512)
        cg = search(ft, gated; kw...)
        ce = search(ft, exact; kw...)
        @info "boxcar gate vs exact" ngated=length(cg) nexact=length(ce)
        @test length(cg) == length(ce)
        @test all(a.freq == b.freq && a.nharm == b.nharm for (a, b) in zip(cg, ce))
        # Candidates come from the `Float64` path in both runs, so even the
        # metric is bit-for-bit equal — the gate only decides *what* to re-score.
        @test all(a.metric == b.metric for (a, b) in zip(cg, ce))
    end

    @testset "block sigma: :zero centring matches the classic MAD" begin
        # `_block_sigma` folds about the exact zero the DC-held-at-zero
        # construction guarantees (`center=:zero`, production) rather than about a
        # sample median (`:median`, what Python computes and what
        # `crossval/crossval_accuracy.jl` pins to).  The two must agree to well
        # inside σ̂'s own ~1% sampling error.  The premise is checked directly too
        # — if the pooled median ever stopped sitting at zero, `:zero` would be
        # silently biased and every reported S/N with it.
        CS = CoherentSearch
        params = SearchParams(nharms=60, m=32)
        Nprof = 512
        lodr = params.hidr / params.nharms
        ws = CS.Workspace(params, Nprof)
        for rstart in (10010.0, 20010.0, 40010.0)
            dplans = CS.build_direct_plans(params, rstart)
            CS.fill_chunk_profiles!(ws, dplans, ft, params, rstart, lodr, Nprof; t0=0)
            # `fill_chunk_profiles!` does the base fold only; the decimated
            # stacks are transformed in `decim_pass!`, so do that here.
            for db in ws.decims
                db.dprofs .= db.brfftplan * db.src   # `*` avoids a LinearAlgebra test dep
            end
            folds = vcat([(2params.nharms, ws.profs, ws.bcsig)],
                         [(2db.Hk, db.dprofs, db.bcsig) for db in ws.decims])
            for (nb, P, buf) in folds
                s0 = CS._block_sigma(P, nb, Nprof, buf)                    # :zero
                sm = CS._block_sigma(P, nb, Nprof, buf; center=:median)    # classic MAD
                @test s0 > 0 && isapprox(s0, sm; rtol=0.02)
                # The location `:zero` assumes, measured on the same profiles.
                allbins = vec(copy(P))
                @test abs(CS._median!(allbins, length(allbins))) < 0.1 * sm
            end
        end
    end

    @testset "batched boxcar gate matches the scalar gate" begin
        # Directly: the cross-profile SIMD gate against the per-column scalar
        # one it replaces, on real profiles.  Both now compute the *same*
        # statistic, so the only difference is the `Float32` tile; what matters
        # is that it stays far under `boxcar_gatemargin` (0.01), the slack the
        # `Float64` re-score reserves.
        CS = CoherentSearch
        params = SearchParams(nharms=60, m=32)
        nbins = 2params.nharms
        Nprof = 500          # deliberately not a multiple of _BC_BATCH (tail path)
        lodr = params.hidr / params.nharms
        rstart = 10010.0
        ws = CS.Workspace(params, Nprof)
        dplans = CS.build_direct_plans(params, rstart)
        CS.fill_chunk_profiles!(ws, dplans, ft, params, rstart, lodr, Nprof; t0=0)
        sigma = CS._block_sigma(ws.profs, nbins, Nprof, ws.bcsig)
        invsigma = one(PIN_PROFT) / sigma

        CS._boxcar_gate!(ws.bcbatch, ws.profs, Nprof, ws.bcpsum, ws.bcwidths,
                         nbins, PIN_PROFT(invsigma))
        got = copy(ws.bcbatch.mvals[1:Nprof])
        want = [Float64(CS._profile_boxcar(ws.profs, j, ws.bcpsum, ws.bcwidths,
                                           nbins, invsigma)) for j in 1:Nprof]
        err = maximum(abs.(got .- want))
        @info "batched vs scalar boxcar gate" maxabs=err gatemargin=params.boxcar_gatemargin
        @test err < 1e-3 * params.boxcar_gatemargin
        # The `< _BC_BATCH` tail columns take the scalar kernel, so they are exact.
        ntail = Nprof - (Nprof ÷ CS._BC_BATCH) * CS._BC_BATCH
        @test all(got[j] == want[j] for j in (Nprof - ntail + 1):Nprof)
    end

    @testset "the tile transpose is bit-identical at every block width" begin
        # `_bc_transpose!` blocks the *profile* axis by `_BC_TR_BJ` because the
        # number of concurrent strided read streams is what the two development
        # hosts disagree about by 3.5x (see the comment on the kernel).  `BJ` is
        # a loop nest, not a method: every value must produce the same bytes, or
        # the gate's bound — and with it the candidate list — moves with a
        # performance knob.  This is also what would catch `_BC_BATCH` being set
        # to something `_BC_TR_BJ` does not divide.
        CS = CoherentSearch
        params = SearchParams(nharms=60, m=32)
        nbins = 2params.nharms
        Nprof = 2CS._BC_BATCH
        ws = CS.Workspace(params, Nprof)
        dplans = CS.build_direct_plans(params, 10010.0)
        CS.fill_chunk_profiles!(ws, dplans, ft, params, 10010.0,
                                params.hidr / params.nharms, Nprof; t0=0)
        B = CS._BC_BATCH
        @test B % CS._BC_TR_BJ == 0
        ref = Vector{CS._BC_TILE}(undef, B * nbins)
        CS._bc_transpose!(ref, ws.profs, 0, nbins, Val(B), Val(B))   # unblocked
        got = similar(ref)
        for bj in (1, 2, 4, 8, 16, 32, 64, B)
            for j0 in (0, B)
                CS._bc_transpose!(ref, ws.profs, j0, nbins, Val(B), Val(B))
                fill!(got, NaN32)
                CS._bc_transpose!(got, ws.profs, j0, nbins, Val(B), Val(bj))
                @test got == ref
            end
        end
        # And the shipped default is one of them.
        CS._bc_transpose!(ref, ws.profs, 0, nbins, Val(B), Val(B))
        fill!(got, NaN32)
        CS._bc_transpose!(got, ws.profs, 0, nbins, Val(B))
        @test got == ref
    end

    @testset "the same-type tile transpose matches the widening one" begin
        # `_bc_transpose!` has a second method for the case `--precision f32`
        # produces: `profs` already the tile's eltype.  It exists because the
        # generic nest lets LLVM vectorise the `i` loop and write the tile with a
        # 512-bit SCATTER, which on Skylake-SP drops the core to turbo licence
        # level 2 and taxes the whole search (see the kernel's comment).  It is a
        # second implementation of the same copy, so it must produce the same
        # bytes -- widening to `Float64` and back is exact, so the generic nest
        # on a widened copy is an exact reference.
        CS = CoherentSearch
        params = SearchParams(nharms=60, m=32)
        nbins = 2params.nharms
        Nprof = 2CS._BC_BATCH
        ws = CS.Workspace(params, Nprof)
        dplans = CS.build_direct_plans(params, 10010.0)
        CS.fill_chunk_profiles!(ws, dplans, ft, params, 10010.0,
                                params.hidr / params.nharms, Nprof; t0=0)
        B = CS._BC_BATCH
        T = CS._BC_TILE
        p32 = T.(ws.profs)                      # what a `:f32` Workspace holds
        p64 = Float64.(p32)                     # exact widening

        # The dispatch itself is the fragile part: the two methods were
        # AMBIGUOUS on first writing (`T` unconstrained is not a subset of
        # `<:AbstractFloat`), so the fast path silently never ran.  Pin it.
        @test which(CS._bc_transpose!,
                    Tuple{Vector{T},Matrix{T},Int,Int,Val{B},Val{CS._BC_TR_BJ}}) !==
              which(CS._bc_transpose!,
                    Tuple{Vector{T},Matrix{Float64},Int,Int,Val{B},Val{CS._BC_TR_BJ}})

        ref = Vector{T}(undef, B * nbins)
        got = similar(ref)
        for bj in (1, 2, 4, 8, 16, 32, 64, B)
            for j0 in (0, B)
                CS._bc_transpose!(ref, p64, j0, nbins, Val(B), Val(bj))
                fill!(got, NaN32)
                CS._bc_transpose!(got, p32, j0, nbins, Val(B), Val(bj))
                @test got == ref
            end
        end
        # It writes exactly `B * nbins` elements and not one more.
        big = fill(T(NaN), B * nbins + 8)
        CS._bc_transpose!(big, p32, 0, nbins, Val(B), Val(CS._BC_TR_BJ))
        @test all(isnan, @view big[(B * nbins + 1):end])
        @test_throws BoundsError CS._bc_transpose!(Vector{T}(undef, B * nbins - 1),
                                                   p32, 0, nbins, Val(B),
                                                   Val(CS._BC_TR_BJ))
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

        nb_fine = 64                                  # what the retired
                                                      # per-harmonic schedule chose for h=1
        truth = amps_at(256)
        rel(a) = maximum(abs.(a .- truth)) / maximum(abs.(truth))
        err_fixed   = rel(amps_at(16))
        err_aligned = rel(amps_at(nb_fine))
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

        # De-duplication collapses the cluster of above-threshold trials around
        # the pulsar: far fewer candidates out.
        raw = search(ft, params; lofreq=9.5, hifreq=10.5, threshold=8.0,
                     remove=false, harm_remove=false)
        @test length(cands) < length(raw)
        # Dedup is a pure function of the list: it must return the exact strongest
        # candidate untouched (checked on the same list to avoid FFTW-plan jitter
        # between two separate search calls).
        rbest = raw[argmax(c.metric for c in raw)]
        dbest = remove_duplicates(raw)[argmax(c.metric for c in remove_duplicates(raw))]
        @test dbest.freq == rbest.freq && dbest.metric == rbest.metric
    end

    @testset "chunk size does not change the detection" begin
        # This used to run on `:non`, a pure per-profile metric that is *exactly*
        # chunk-invariant.  The boxcar metric normalises by a per-*block* σ̂, so it
        # is only approximately so — by design, and the approximation is what the
        # test now has to state precisely rather than absorb into a tolerance:
        #
        #   * which trial wins is *exactly* chunk-invariant (asserted as equality,
        #     which is the stronger claim the old `atol=1e-3` was hiding), and
        #   * the metric drifts with block size because σ̂ is measured over the
        #     block.  Measured 26.03 → 27.23 (4.6%) across blocksize 512 → 4096,
        #     monotonically; 10% is the bound with room for the estimator's noise.
        params = SearchParams(nharms=32, m=32, numbetween=16)
        bests = map((512, 1024, 2048, 4096)) do bs
            c = search(ft, params; lofreq=9.5, hifreq=10.5, threshold=8.0, blocksize=bs)
            c[argmax(x.metric for x in c)]
        end
        @test allequal(b.freq for b in bests)
        lo, hi = extrema(b.metric for b in bests)
        @info "chunk-size metric drift (per-block σ̂)" lo hi rel=(hi - lo) / lo
        @test (hi - lo) / lo < 0.10
    end

    @testset "decimation pass k reproduces the native Hk-harmonic fold" begin
        # The strong equivalence: gathering every k-th of the base nharms=60
        # harmonics and folding must equal a *native* Hk=⌊60/k⌋-harmonic search
        # at the multiplied frequencies k*rf.  Both sides use the exact kernel
        # (`kernel=:direct`), so the residual is tabulation rounding, not method.
        for k in (2, 3, 4)
            nharms = 60
            Hk = fld(nharms, k)
            params = SearchParams(nharms=nharms, m=32, decimations=[1, k])
            lodr = params.hidr / nharms
            rstart = 5000.0
            n = 64
            rfund = rstart .+ (0:n-1) .* lodr

            # Native reduced-harmonic fold at the multiplied frequencies.  The
            # boxcar bank is passed explicitly: the ladder prunes the decimated
            # fold's bank (see `ladder_boxcar_widths`), and this testset is about
            # the harmonic gather and transform, not the width selection, so both
            # sides must scan the same widths for the comparison to mean anything.
            pnat = SearchParams(nharms=Hk, m=32)
            ref = block_metrics(ft, k .* rfund, pnat; kernel=:direct,
                                widths=ladder_boxcar_widths(2Hk, k, params))

            # Decimated fold via the production path.
            ws = CoherentSearch.Workspace(params, n)
            # Float64 weights: this pin is about the harmonic gather and the
            # transform, so it must not absorb the Float32 default's ~2e-7.
            dpl = build_direct_plans(Float64, params, rstart)
            CoherentSearch.fill_chunk_profiles!(ws, dpl, ft, params, rstart, lodr, n; t0=0)
            db = only(ws.decims)                      # decimations=[1,k] -> just k
            @test db.k == k && db.Hk == Hk
            out = Candidate[]
            CoherentSearch.decim_pass!(out, ws, db, ft, params, rstart, lodr, n; threshold=-Inf)
            @test length(out) == n
            got = [c.metric for c in out]             # emitted in ascending-r (j) order
            relerr = maximum(abs.(got .- ref)) / maximum(abs.(ref))
            @info "decimation k native-fold agreement" k relerr
            @test relerr < PIN_TOL
            @test all(c.nharm == Hk for c in out)
        end
    end

    @testset "decimation source view is the gather it replaced" begin
        # `DecimBuf` no longer copies every k-th harmonic row into a compact
        # stack; it hands FFTW a stride-k view of `ftprofs` instead.  That is only
        # correct if the view holds *exactly* what the copy would have: DC in row
        # 1, then base harmonic j*k in row j+1.  Checked against an explicit
        # gather, since a silently mis-strided view would still transform fine and
        # simply fold the wrong harmonics.
        params = SearchParams(nharms=60, decimations=[1, 2, 3, 4, 5, 6])
        n = 128
        lodr = params.hidr / params.nharms
        rstart = 5000.0
        ws = CoherentSearch.Workspace(params, n)
        dplans = CoherentSearch.build_direct_plans(params, rstart)
        CoherentSearch.fill_chunk_profiles!(ws, dplans, ft, params, rstart, lodr, n; t0=0)
        for db in ws.decims
            want = zeros(eltype(ws.ftprofs), db.Hk + 1, n)   # DC row stays zero
            for j in 1:db.Hk
                want[j + 1, :] .= ws.ftprofs[j * db.k + 1, 1:n]
            end
            @test size(db.src) == (db.Hk + 1, size(ws.ftprofs, 2))
            @test db.src[:, 1:n] == want
            @test all(iszero, db.src[1, :])                  # DC never written
        end
    end

    @testset "precision=:f32 tracks the :f64 profile stage" begin
        # The narrowed profile stage is a *runtime* choice, so both widths run in
        # one build and can be compared directly.  It must find the same
        # candidates; the metrics differ only by the single rounding per harmonic
        # amplitude, so they agree to ~1e-6 relative — orders of magnitude below
        # the ~1.3% signal-power loss the m=16 kernel already accepts.
        f = 10.0123456789123
        base = (nharms=32, threshold=6.0)
        c64 = search(ft, SearchParams(; base..., precision=:f64);
                     lofreq=f - 0.05, hifreq=f + 0.05, progress=:none)
        c32 = search(ft, SearchParams(; base..., precision=:f32);
                     lofreq=f - 0.05, hifreq=f + 0.05, progress=:none)
        @test !isempty(c64)
        @test length(c32) == length(c64)
        @test [c.freq for c in c32] == [c.freq for c in c64]
        @test [c.nharm for c in c32] == [c.nharm for c in c64]
        relerr = maximum(abs(a.metric - b.metric) / abs(b.metric)
                         for (a, b) in zip(c32, c64))
        @info "f32 vs f64 profile stage" relerr ncands=length(c64)
        @test relerr < 1e-5
        @test CoherentSearch.proftype(SearchParams(precision=:f32)) === Float32
        @test CoherentSearch.proftype(SearchParams(precision=:f64)) === Float64
        @test_throws ArgumentError CoherentSearch.proftype(SearchParams(precision=:f16))
    end

    @testset "detects the 10.0123 Hz pulsar via decimation" begin
        f = 10.0123
        nharms = 60
        for k in (2, 3)
            base_f = f / k                            # fundamental band that only k hits
            params = SearchParams(nharms=nharms, decimations=decimation_set(nharms, k))
            # harm_remove=false to isolate decimation: otherwise the whole f/k, 2f/k,
            # ... family (which decimation lights up) collapses to its single
            # strongest member, which need not be the direct-f (k-pass) detection.
            cands = search(ft, params; lofreq=base_f - 0.5, hifreq=base_f + 0.5,
                           threshold=8.0, harm_remove=false)
            match = filter(c -> isapprox(c.freq, f; atol=1e-2), cands)
            @info "decimation detection" k n_match=length(match)
            @test !isempty(match)
            @test any(c.nharm == fld(nharms, k) for c in match)   # found via the k pass

            # With decimation off, the same (sub-harmonic) band finds no signal.
            off = search(ft, SearchParams(nharms=nharms);
                         lofreq=base_f - 0.5, hifreq=base_f + 0.5, threshold=8.0)
            @test isempty(filter(c -> isapprox(c.freq, f; atol=1e-2), off))
        end
    end

    @testset "harmonic removal collapses the subharmonic family" begin
        # In a band near f/3, decimation lights up the pulsar as a three-member
        # harmonic family: f/3 (60 harm), 2f/3 (30 harm), f (20 harm), at Fourier
        # frequency ratios 1:2:3.  Harmonic removal must collapse it to a single
        # survivor (the strongest), independent of which physical frequency wins.
        nharms = 60
        params = SearchParams(nharms=nharms, decimations=decimation_set(nharms, 3))
        kw = (lofreq=10.0123 / 3 - 0.4, hifreq=10.0123 / 3 + 0.4, threshold=8.0)
        raw = search(ft, params; kw..., harm_remove=false)
        one = search(ft, params; kw..., harm_remove=true)
        @test length(one) < length(raw)

        top3 = sort(raw; by=c -> c.metric, rev=true)[1:3]
        strongest = top3[1]
        rel(a, b) = CoherentSearch._harmonically_related(a, b; numharm=16, tol=1.0)
        @info "harmonic family" freqs=[round(c.freq, digits=4) for c in top3]
        # The three strongest detections are one harmonic family (all related to the top).
        @test all(rel(strongest.r, c.r) for c in top3)
        # Removal keeps exactly one member of that family -- the strongest -- and drops the rest.
        @test count(c -> rel(strongest.r, c.r), one) == 1
        surv = one[argmax(c.metric for c in one)]
        @test surv.freq == strongest.freq && surv.metric == strongest.metric
    end

    @testset "metricstats: read-only diagnostic, histograms + per-k aggregation" begin
        nharms = 60
        params = SearchParams(nharms=nharms, decimations=decimation_set(nharms, 3))
        kw = (lofreq=8.0, hifreq=12.0, threshold=8.0, blocksize=1024)
        # Collecting stats must not change the candidate results.  (This example
        # file is signal-dominated, with metric values far above the default
        # histogram range, so widen it here to keep the quantiles in-range.)
        ref = search(ft, params; kw...)
        ms = MetricStats(hist_hi=30000.0, hist_nb=6000)
        with = search(ft, params; kw..., metricstats=ms)
        @test length(with) == length(ref)
        @test all(a.freq == b.freq && a.metric == b.metric for (a, b) in zip(with, ref))

        # Per-block table
        @test !isempty(ms.blocks)
        @test Set(s.k for s in ms.blocks) == Set([1, 2, 3])
        @test all(s.nbins == 2 * s.Hk for s in ms.blocks)
        @test all(s.Hk == fld(nharms, s.k) for s in ms.blocks)
        @test all(s.min <= s.median <= s.max && s.min <= s.mean <= s.max for s in ms.blocks)

        # Per-k global histograms: one per decimation, exact counts + moments.
        @test [h.k for h in ms.hists] == [1, 2, 3]
        for h in ms.hists
            @test sum(h.counts) + h.under + h.over == h.total
            # histogram total == sum of that k's per-block trial counts
            @test h.total == sum(s.n for s in ms.blocks if s.k == h.k)
            @test h.vmin <= hist_quantile(h, 0.5) <= h.vmax
            @test hist_quantile(h, 0.1) <= hist_quantile(h, 0.9)   # monotone
        end

        # Windowed histograms: nwin per k, and each k's windows must sum back to
        # its band-wide histogram (counts, total, exact moments) -- i.e. the
        # per-k `hists` is exactly the merge of the per-window `whists`.
        @test Set(h.k for h in ms.whists) == Set([1, 2, 3])
        for h in ms.whists
            @test count(w -> w.k == h.k, ms.whists) == ms.nwin
            @test h.flo < h.fhi                                    # window has positive width
        end
        for g in ms.hists
            wk = [h for h in ms.whists if h.k == g.k]
            @test sum(h.total for h in wk) == g.total
            @test sum(h.sum for h in wk) ≈ g.sum
            @test mapreduce(h -> h.counts, +, wk) == g.counts
            # windows tile the k band contiguously, low -> high
            sort!(wk; by = h -> h.win)
            @test all(wk[i].fhi ≈ wk[i+1].flo for i in 1:length(wk)-1)
        end

        # Summary: exact mean matches the histogram accumulator; nbins ordering
        # shows the sqrt(nbins) noise-floor growth (k=1 mean > k=2 > k=3); and the
        # FAP thresholds are monotone in k the same way.
        summ = metricstats_summary(ms; faps=(0.1, 0.01, 1e-3))
        @test [r.k for r in summ] == [1, 2, 3]
        for (r, h) in zip(summ, ms.hists)
            @test r.ntrials == h.total
            @test r.mean ≈ h.sum / h.total
            @test r.max == h.vmax
            @test length(r.fap) == 3
        end
        @test summ[1].mean > summ[2].mean > summ[3].mean          # more bins -> higher floor
        @test summ[1].fap[1] > summ[3].fap[1]                     # same FAP -> higher threshold at k=1

        # Windowed summary rows: only nonempty windows, tagged with their k/win.
        wrows = metricstats_windows(ms; faps=(0.1, 0.01))
        @test !isempty(wrows)
        @test all(r.ntrials > 0 for r in wrows)
        @test Set(r.k for r in wrows) ⊆ Set([1, 2, 3])
    end

    @testset "normalize: two-pass adaptive threshold" begin
        nharms = 60
        params = SearchParams(nharms=nharms, decimations=decimation_set(nharms, 3))
        # This example file is signal-dominated (huge metric values), so widen the
        # histogram range via the sink so the noise loc/scale are well resolved.
        ms = MetricStats(hist_hi=30000.0, hist_nb=6000, nwin=8)
        kw = (lofreq=9.5, hifreq=10.5, blocksize=1024)
        cands = search(ft, params; kw..., threshold=5.0, metricstats=ms, normalize=true)
        @test !isempty(ms.hists)                       # pass 1 measured the noise
        @test any(isapprox(c.freq, 10.0123; atol=1e-2) for c in cands)   # still detects

        # build_metricnorm: per-k edges/loc/scale, all scales strictly positive,
        # loc equals the histogram median, and normalization is monotone in M.
        norm = build_metricnorm(ms)
        @test Set(keys(norm.loc)) == Set([1, 2, 3])
        for g in ms.hists
            k = g.k
            @test length(norm.loc[k]) == ms.nwin
            @test all(>(0), norm.scale[k])
            @test all(g.vmin <= l <= g.vmax for l in norm.loc[k])   # loc within the data
        end
        f = 10.0123
        @test CoherentSearch._normalize(norm, 1, f, 200.0) >
              CoherentSearch._normalize(norm, 1, f, 100.0)

        # Normalizing must not perturb the *measurement* pass: the pulsar is found.
        @test !isempty(cands)
    end
else
    @info "Skipping search data tests; example file not found" EXAMPLE_FFT
end

@testset "boxcar_best_width recovers the width behind the metric" begin
    # A clean top-hat of width w (from the bank) must be matched by width w.
    nbins = 64
    widths = boxcar_widths(nbins)              # 1,2,3,4,6,9,13,19 at fsp=1.5,maxfrac=0.3
    for w in widths
        prof = zeros(nbins)
        prof[10:(10 + w - 1)] .= 1.0
        best, ducy = boxcar_best_width(prof)
        @test best == w
        @test ducy ≈ w / nbins
    end

    # A pulse that wraps the phase boundary is found at the same width: the
    # prefix sums are tiled by wmax precisely so a boxcar can wrap.
    prof = zeros(nbins); prof[63:64] .= 1.0; prof[1:2] .= 1.0
    @test boxcar_best_width(prof)[1] == 4

    # The width is independent of scale and of an additive baseline (the
    # zero-mean template kills any constant, and sigma is a common factor across
    # widths) -- which is what licenses measuring it on an isolated profile, with
    # no block statistics.
    prof = zeros(nbins); prof[20:25] .= 1.0
    w0, _ = boxcar_best_width(prof)
    @test boxcar_best_width(1e6 .* prof)[1] == w0
    @test boxcar_best_width(prof .+ 37.0)[1] == w0

    # Pin the prefix-sum machinery to a naive, obviously-correct scan: sum every
    # circular window of every bank width and score it with riptide's zero-mean,
    # unit-L2 boxcar template written out longhand.  The wrapped-window indexing
    # is the part that could plausibly be wrong.
    function naive_best(prof, nb)
        stot = sum(prof)
        best, bw = -Inf, 0
        for w in boxcar_widths(nb)
            h = sqrt((nb - w) / (nb * w))          # riptide cpp/snr.hpp:snr1
            b = w / (nb - w) * h
            for p in 1:nb
                s = sum(prof[mod1(p + i, nb)] for i in 0:(w - 1))
                c = (h + b) * s - b * stot
                c > best && (best = c; bw = w)
            end
        end
        return bw
    end

    rng = MersenneTwister(20260811)
    for w in (1, 3, 9, 13)
        for trial in 1:5
            prof = 0.05 .* randn(rng, nbins)
            prof[30:(30 + w - 1)] .+= 3.0
            @test boxcar_best_width(prof)[1] == naive_best(prof, nbins)
        end
    end
    # Pure noise too, where the winning width is arbitrary and the two
    # implementations must still agree on it (including tie-breaking).
    for trial in 1:20
        prof = randn(rng, nbins)
        @test boxcar_best_width(prof)[1] == naive_best(prof, nbins)
    end
end

@testset "Candidate carries ducy, defaulting to NaN" begin
    c = Candidate(1.0, 10.0, 1000.0, 32)
    @test isnan(c.ducy)                       # 4-arg form keeps the hot loop unchanged
    c2 = Candidate(1.0, 10.0, 1000.0, 32, 0.125)
    @test c2.ducy == 0.125
    # De-duplication and harmonic collapse must not disturb it.
    kept = remove_duplicates([c2]; dr_tol=1.0)
    @test kept[1].ducy == 0.125
end
