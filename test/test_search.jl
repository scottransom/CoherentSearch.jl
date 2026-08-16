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


@testset "harmonic_numbetween schedule" begin
    nh, hidr, minnb = 32, 0.5, 16
    # Never below the floor; finer at the low harmonics; matched to deltar_h.
    @test harmonic_numbetween(1,  nh, hidr, minnb) == 64    # = 2*nharms
    @test harmonic_numbetween(2,  nh, hidr, minnb) == 32
    @test harmonic_numbetween(4,  nh, hidr, minnb) == 16
    @test harmonic_numbetween(32, nh, hidr, minnb) == 16    # floored
    @test all(harmonic_numbetween(h, nh, hidr, minnb) >= minnb for h in 1:nh)
end

@testset "boxcar_widths: geometric bank capped at maxfrac*nbins" begin
    w = boxcar_widths(64; fsp=1.5, maxfrac=0.3)
    @test w == [1, 2, 3, 4, 6, 9, 13, 19]        # riptide's wₖ₊₁=max(⌊1.5wₖ⌋,wₖ+1)
    @test w[1] == 1 && issorted(w) && allunique(w)
    @test w[end] <= floor(Int, 0.3 * 64)
    @test all(w[i+1] == max(floor(Int, 1.5 * w[i]), w[i] + 1) for i in 1:length(w)-1)
    @test boxcar_widths(4) == [1]                # tiny profile keeps only width-1
    @test boxcar_widths(64; maxfrac=0.5)[end] <= 32
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

    @testset "optimised path reproduces the reference (align=false)" begin
        # With a fixed numbetween and one chunk, the production path uses the same
        # grids as `block_metrics`, so it should match to ~machine precision.  The
        # metric is scale-free, so the unnormalised brfft (chunk_metrics) and the
        # normalised irfft (block_metrics) agree despite the missing 1/nbins.
        # interp=:fft + fftsizing=:pow2 is what `block_metrics` itself does, so this
        # is the strict equivalence gate; the default :direct path is a *different*
        # (exact) interpolator and is pinned separately below.
        params = SearchParams(nharms=32, m=32, numbetween=16, align=false,
                              interp=:fft, fftsizing=:pow2)
        lodr = params.hidr / params.nharms
        rstart = 10010.0
        n = 256
        rfund = rstart .+ (0:n-1) .* lodr

        ref = block_metrics(ft, rfund, params)
        opt = chunk_metrics(ft, params, rstart, n; lodr=lodr)
        relerr = maximum(abs.(opt .- ref)) / maximum(abs.(ref))
        @info "boxcar align=false reference agreement" relerr
        @test relerr < PIN_TOL
    end

    @testset "direct interpolation is the exact kernel" begin
        # The strong pin for the default path.  `fourier_interp` is Eqn. 30
        # evaluated point by point and is itself pinned to the Python oracle at
        # ~3e-16, so agreeing with it *is* agreeing with the oracle — a stronger
        # statement than the FFT path's equivalence, which inherits that path's
        # linear-interpolation error.
        nharms = 60
        params = SearchParams(nharms=nharms, m=32, interp=:direct)
        lodr = params.hidr / nharms
        # Low enough that harmonic 60 stays inside the (small) bundled test FFT,
        # so the per-point reference can be evaluated for every harmonic.
        rstart = 5000.0
        n = 256
        hplans = build_harmonic_plans(params, n)
        ws = CoherentSearch.Workspace(params, hplans, n)
        dplans = build_direct_plans(params, rstart)
        CoherentSearch.fill_chunk_profiles!(ws, hplans, ft, params, rstart, lodr, n;
                                            dplans=dplans, t0=0)
        worst = 0.0
        for h in (1, 2, 7, 15, 30, 59, 60), j in (1, 2, 97, n)
            r = h * (rstart + (j - 1) * lodr)
            v = fourier_interp(r, ft.amps, params.m)
            worst = max(worst, abs(ws.ftprofs[h + 1, j] - v) / abs(v))
        end
        @info "direct interpolation vs exact fourier_interp" worst
        @test worst < 1e-7        # relative, and largest where |amp| ~ 0

        # ... and the FFT path is *much* further from exact, which is the
        # accuracy the direct path buys (not a defect of this test).
        pf = SearchParams(nharms=nharms, m=32, interp=:fft)
        hf = build_harmonic_plans(pf, n)
        wf = CoherentSearch.Workspace(pf, hf, n)
        CoherentSearch.fill_chunk_profiles!(wf, hf, ft, pf, rstart, lodr, n)
        wfft = maximum(abs(wf.ftprofs[h + 1, j] -
                           fourier_interp(h * (rstart + (j - 1) * lodr), ft.amps, pf.m)) /
                       abs(fourier_interp(h * (rstart + (j - 1) * lodr), ft.amps, pf.m))
                       for h in (8, 30, 60), j in (1, 97, n))
        @info "fft+linear interpolation vs exact fourier_interp" wfft
        # How large the linear-interpolation error is depends on the data and the
        # frequency (it reaches ~5e-2 on a long real observation, ~6e-4 here), so
        # the portable statement is the *ratio*: the FFT path's error is a
        # data-dependent approximation, the direct path's is rounding.
        @test wfft > 1e-5
        @test worst < wfft / 1e4  # direct is orders of magnitude closer to exact
    end

    @testset "direct interpolation is invariant to chunking" begin
        # The phase/bin bookkeeping keys off a *global* trial index, so a chunk
        # starting at global trial t0 must reproduce what one long chunk gives at
        # the same trials.  This is the property that a naive float-accumulated
        # `rstart` would drift on.
        nharms = 32
        params = SearchParams(nharms=nharms, m=32, interp=:direct)
        lodr = params.hidr / nharms
        r_lo = 10010.0
        dplans = build_direct_plans(params, r_lo)
        nbig = 384
        hb = build_harmonic_plans(params, nbig)
        wb = CoherentSearch.Workspace(params, hb, nbig)
        CoherentSearch.fill_chunk_profiles!(wb, hb, ft, params, r_lo, lodr, nbig;
                                            dplans=dplans, t0=0)
        nsm = 128
        hs = build_harmonic_plans(params, nsm)
        ws2 = CoherentSearch.Workspace(params, hs, nsm)
        worst = 0.0
        for c in 0:2
            t0 = c * nsm
            CoherentSearch.fill_chunk_profiles!(ws2, hs, ft, params, r_lo + t0 * lodr,
                                                lodr, nsm; dplans=dplans, t0=t0)
            for h in (1, 5, 32), j in 1:nsm
                a = ws2.ftprofs[h + 1, j]
                b = wb.ftprofs[h + 1, t0 + j]
                worst = max(worst, abs(a - b) / abs(b))
            end
        end
        @info "direct interpolation chunk invariance" worst
        @test worst == 0.0        # identical arithmetic, not merely close
    end

    @testset "the two interpolators agree on the detection metric" begin
        # They are different algorithms (one approximate), so they cannot be
        # bit-identical — but the metric they produce must track closely, and the
        # candidate they find must be the same one.
        params_d = SearchParams(nharms=32, m=32, interp=:direct)
        params_f = SearchParams(nharms=32, m=32, interp=:fft)
        lodr = params_d.hidr / params_d.nharms
        rstart = 10010.0
        n = 256
        md = chunk_metrics(ft, params_d, rstart, n; lodr=lodr)
        mf = chunk_metrics(ft, params_f, rstart, n; lodr=lodr)
        rel = maximum(abs.(md .- mf)) / maximum(abs.(mf))
        @info "direct vs fft metric agreement" rel
        @test rel < 0.05
        @test argmax(md) == argmax(mf)
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
        # The production :boxcar path scores ~99% of trials with only a cheap
        # zero-baseline *lower bound* (batched across profiles, in Float32), and
        # pays for the exact median baseline only when that bound comes within
        # `boxcar_medmargin` of `threshold`.  `boxcar_medmargin = Inf` forces
        # `medcut = -Inf`, i.e. the exact scalar path for every trial — so the
        # two runs must produce byte-identical candidates.  This is the pin for
        # both the gate itself and the cross-profile SIMD batching behind it.
        gated = SearchParams(nharms=60, m=32,
                             decimations=decimation_set(60, 4))
        exact = SearchParams(nharms=60, m=32,
                             decimations=decimation_set(60, 4),
                             boxcar_medmargin=Inf)
        kw = (lofreq=9.5, hifreq=10.5, threshold=6.0, blocksize=512)
        cg = search(ft, gated; kw...)
        ce = search(ft, exact; kw...)
        @info "boxcar gate vs exact" ngated=length(cg) nexact=length(ce)
        @test length(cg) == length(ce)
        @test all(a.freq == b.freq && a.nharm == b.nharm for (a, b) in zip(cg, ce))
        # Candidates come from the exact path in both runs, so even the metric
        # is bit-for-bit equal — the Float32 gate only decides *what* to score.
        @test all(a.metric == b.metric for (a, b) in zip(cg, ce))
    end

    @testset "batched boxcar gate matches the scalar gate" begin
        # Directly: the cross-profile SIMD gate against the per-column scalar
        # one it replaces, on real profiles.  Float32 tiles make this close, not
        # exact; the bound that matters is that it stays far under
        # `boxcar_medmargin` (2.0), which is the slack the rescue reserves.
        CS = CoherentSearch
        params = SearchParams(nharms=60, m=32)
        nbins = 2params.nharms
        Nprof = 500          # deliberately not a multiple of _BC_BATCH (tail path)
        lodr = params.hidr / params.nharms
        rstart = 10010.0
        hplans = CS.build_harmonic_plans(params, Nprof)
        ws = CS.Workspace(params, hplans, Nprof)
        dplans = CS.build_direct_plans(params, rstart)
        CS.fill_chunk_profiles!(ws, hplans, ft, params, rstart, lodr, Nprof;
                                dplans=dplans, t0=0)
        sigma = CS._block_sigma(ws.profs, nbins, Nprof, ws.bcsig)
        invsigma = one(PIN_PROFT) / sigma

        CS._boxcar_gate!(ws.bcbatch, ws.profs, Nprof, ws.bcpsum, ws.bcwidths,
                         nbins, PIN_PROFT(invsigma))
        got = copy(ws.bcbatch.mvals[1:Nprof])
        want = [Float64(CS._profile_boxcar(ws.profs, j, ws.medbuf, ws.bcpsum, ws.bcwidths,
                                           nbins, invsigma, ws.medpairs, Inf)) for j in 1:Nprof]
        err = maximum(abs.(got .- want))
        @info "batched vs scalar boxcar gate" maxabs=err medmargin=params.boxcar_medmargin
        @test err < 1e-3 * params.boxcar_medmargin
        # The `< _BC_BATCH` tail columns take the scalar kernel, so they are exact.
        ntail = Nprof - (Nprof ÷ CS._BC_BATCH) * CS._BC_BATCH
        @test all(got[j] == want[j] for j in (Nprof - ntail + 1):Nprof)
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
        # at the multiplied frequencies k*rf.  Pinned with align=false so both
        # use identical fixed-numbetween interpolation grids -> machine precision.
        for k in (2, 3, 4)
            nharms = 60
            Hk = fld(nharms, k)
            params = SearchParams(nharms=nharms, m=32, numbetween=16, align=false,
                                  decimations=[1, k],
                                  interp=:fft, fftsizing=:pow2)
            lodr = params.hidr / nharms
            rstart = 5000.0
            n = 64
            rfund = rstart .+ (0:n-1) .* lodr

            # Native reduced-harmonic fold at the multiplied frequencies.
            pnat = SearchParams(nharms=Hk, m=32, numbetween=16, align=false,
                                interp=:fft, fftsizing=:pow2)
            ref = block_metrics(ft, k .* rfund, pnat)

            # Decimated fold via the production path.
            hplans = build_harmonic_plans(params, n)
            ws = CoherentSearch.Workspace(params, hplans, n)
            CoherentSearch.fill_chunk_profiles!(ws, hplans, ft, params, rstart, lodr, n)
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
        hplans = build_harmonic_plans(params, n)
        ws = CoherentSearch.Workspace(params, hplans, n)
        dplans = CoherentSearch.build_direct_plans(params, rstart)
        CoherentSearch.fill_chunk_profiles!(ws, hplans, ft, params, rstart, lodr, n;
                                            dplans=dplans, t0=0)
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

    # The width is independent of scale and of an additive baseline (the median
    # is subtracted, and sigma is a common factor across widths) -- which is what
    # licenses measuring it on an isolated profile, with no block statistics.
    prof = zeros(nbins); prof[20:25] .= 1.0
    w0, _ = boxcar_best_width(prof)
    @test boxcar_best_width(1e6 .* prof)[1] == w0
    @test boxcar_best_width(prof .+ 37.0)[1] == w0

    # Pin the prefix-sum machinery to a naive, obviously-correct scan: subtract
    # the median, sum every circular window of every bank width, divide by sqrt(w).
    # The wrapped-window indexing is the part that could plausibly be wrong.
    function naive_best(prof, nb)
        s = sort(prof)
        med = isodd(nb) ? s[(nb + 1) ÷ 2] : 0.5 * (s[nb ÷ 2] + s[nb ÷ 2 + 1])
        z = prof .- med
        best, bw = -Inf, 0
        for w in boxcar_widths(nb)
            for p in 1:nb
                s = sum(z[mod1(p + i, nb)] for i in 0:(w - 1)) / sqrt(w)
                s > best && (best = s; bw = w)
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
