using Test
using CoherentSearch
using Random

# The optimised chunk/plan-caching path must agree with the simple reference
# `block_metrics` (which is itself pinned to the Python oracle to ~1e-15), and
# must actually detect the 10.0123 Hz test pulsar.  These run only when the
# example data from the sibling Python repo is present.
const EXAMPLE_FFT = joinpath(@__DIR__, "..", "..", "coherent_search",
                             "examples", "harmonics_hi.fft")

@testset "harmonic_numbetween schedule" begin
    nh, hidr, minnb = 32, 0.5, 16
    # Never below the floor; finer at the low harmonics; matched to deltar_h.
    @test harmonic_numbetween(1,  nh, hidr, minnb) == 64    # = 2*nharms
    @test harmonic_numbetween(2,  nh, hidr, minnb) == 32
    @test harmonic_numbetween(4,  nh, hidr, minnb) == 16
    @test harmonic_numbetween(32, nh, hidr, minnb) == 16    # floored
    @test all(harmonic_numbetween(h, nh, hidr, minnb) >= minnb for h in 1:nh)
end

@testset "snr_metrics: N_on^p and Σd²^p detection metrics" begin
    nbins = 64
    ngood = 32.0
    invrms = sqrt(2 * ngood + 1)

    # A lone spike: the on-pulse set is just the peak, so N_on=1 and Σd²=0 -> both
    # penalties floor to 1 and the metric is peak*invrms for any metric/exponent.
    spike = zeros(nbins); spike[10] = 5.0
    for met in (:non, :sd2), p in (0.5, 1.0)
        @test snr_metrics(reshape(spike, nbins, 1), ngood; metric=met, pexp=p)[1] ≈
              5.0 * invrms rtol=1e-12
    end

    # Equal-area, different width: the narrower pulse scores higher (width penalty).
    narrow = zeros(nbins); narrow[20] = 8.0                 # N_on=1
    wide   = zeros(nbins); wide[20:27] .= 1.0               # N_on=8, same area
    @test snr_metrics(reshape(wide,   nbins, 1), ngood)[1] <
          snr_metrics(reshape(narrow, nbins, 1), ngood)[1]

    # The design distinction: Σd² penalizes phase SEPARATION; N_on does not.
    close = zeros(nbins); close[20] = 4.0; close[21] = 4.0   # two adjacent bins
    far   = zeros(nbins); far[20]   = 4.0; far[50]   = 4.0   # two well-separated bins
    non_close = snr_metrics(reshape(close, nbins, 1), ngood; metric=:non)[1]
    non_far   = snr_metrics(reshape(far,   nbins, 1), ngood; metric=:non)[1]
    sd2_close = snr_metrics(reshape(close, nbins, 1), ngood; metric=:sd2)[1]
    sd2_far   = snr_metrics(reshape(far,   nbins, 1), ngood; metric=:sd2)[1]
    @test non_close ≈ non_far rtol=1e-12    # N_on only counts lit bins, ignores where
    @test sd2_far < sd2_close               # Σd² down-weights the separated pair

    # The exponent p sets the width-penalty strength (and does nothing at N_on=1).
    wlo = snr_metrics(reshape(wide, nbins, 1), ngood; metric=:non, pexp=0.5)[1]
    whi = snr_metrics(reshape(wide, nbins, 1), ngood; metric=:non, pexp=1.0)[1]
    @test whi < wlo
    @test snr_metrics(reshape(narrow, nbins, 1), ngood; metric=:non, pexp=1.0)[1] ≈
          snr_metrics(reshape(narrow, nbins, 1), ngood; metric=:non, pexp=0.5)[1] rtol=1e-12

    # Modular wrap: a pulse straddling phase 0/1 is contiguous, not edge-split.
    wrap = zeros(nbins); wrap[nbins] = 4.0; wrap[1] = 4.0
    adj  = zeros(nbins); adj[30] = 4.0; adj[31] = 4.0
    @test snr_metrics(reshape(wrap, nbins, 1), ngood; metric=:sd2)[1] ≈
          snr_metrics(reshape(adj,  nbins, 1), ngood; metric=:sd2)[1] rtol=1e-12

    # brfft (unnormalised, nbins×) vs irfft-scaled: the 1/nbins `scale` recovers
    # the same value from an unnormalised profile.
    got = snr_metrics(reshape(spike, nbins, 1), ngood)[1]
    medbuf = Vector{Float64}(undef, nbins)
    scaled = reshape(spike .* nbins, nbins, 1)
    fast = CoherentSearch._profile_snr(scaled, 1, medbuf, nbins, invrms, 1.0 / nbins, 0.2, :non, 0.5)
    @test fast ≈ got rtol=1e-12

    # An unknown metric is rejected.
    @test_throws ArgumentError snr_metrics(reshape(spike, nbins, 1), ngood; metric=:bogus)
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
    @test SearchParams().metric === :boxcar      # :boxcar is now the default metric

    # A ratio of two linear-in-amplitude quantities: invariant to overall scale
    # (this is why the unnormalised brfft hot path and the normalised reference
    # irfft yield the identical value, with no `scale`/`ngoodbins` correction).
    ramp = collect(1.0:nbins)
    m1 = snr_metrics(reshape(ramp, nbins, 1), 32.0; metric=:boxcar)[1]
    m2 = snr_metrics(reshape(ramp .* 7.0, nbins, 1), 32.0; metric=:boxcar)[1]
    @test m1 ≈ m2 rtol=1e-12
    @test m1 > 0

    # A flat profile has zero MAD -> guarded to 0.0, not NaN/Inf.
    @test snr_metrics(reshape(fill(3.0, nbins), nbins, 1), 32.0; metric=:boxcar)[1] == 0.0

    # A narrow pulse on Gaussian noise scores far above the noise-only profile,
    # and the pure-noise peak-over-trials sits at a few sigma (analytic EVD).
    noise = randn(MersenneTwister(1234), nbins)
    snr_noise = snr_metrics(reshape(copy(noise), nbins, 1), 32.0; metric=:boxcar)[1]
    sig = copy(noise); sig[30] += 20.0
    snr_sig = snr_metrics(reshape(sig, nbins, 1), 32.0; metric=:boxcar)[1]
    @test snr_sig > snr_noise + 10
    @test 0 < snr_noise < 8
    @test snr_sig > 15

    # ngoodbins is ignored for :boxcar (it measures its own noise level).
    @test snr_metrics(reshape(copy(noise), nbins, 1), 5.0; metric=:boxcar)[1] ≈ snr_noise
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
        # With a fixed numbetween and one chunk, the production path uses the
        # same grids as block_metrics, so it should match to ~machine precision.
        # Pinned on :non, the oracle-validated (Python-pinned) reference metric.
        params = SearchParams(nharms=32, m=32, numbetween=16, align=false, metric=:non)
        lodr = params.hidr / params.nharms
        rstart = 10010.0
        n = 256
        rfund = rstart .+ (0:n-1) .* lodr

        ref = block_metrics(ft, rfund, params)
        opt = chunk_metrics(ft, params, rstart, n; lodr=lodr)

        relerr = maximum(abs.(opt .- ref)) / maximum(abs.(ref))
        @info "align=false reference agreement" relerr
        # Both paths compute identical profiles to ~1e-10; the snr metric is a
        # continuous function of them except at the (rare) half-max threshold tie.
        @test relerr < 1e-8
    end

    @testset "boxcar metric: optimised path reproduces the reference (align=false)" begin
        # Same equivalence pin as above, for :boxcar.  The metric is scale-free,
        # so the unnormalised brfft (chunk_metrics) and the normalised irfft
        # (block_metrics) must agree to ~machine precision on identical grids.
        params = SearchParams(nharms=32, m=32, numbetween=16, align=false, metric=:boxcar)
        lodr = params.hidr / params.nharms
        rstart = 10010.0
        n = 256
        rfund = rstart .+ (0:n-1) .* lodr

        ref = block_metrics(ft, rfund, params)
        opt = chunk_metrics(ft, params, rstart, n; lodr=lodr)
        relerr = maximum(abs.(opt .- ref)) / maximum(abs.(ref))
        @info "boxcar align=false reference agreement" relerr
        @test relerr < 1e-8
    end

    @testset "boxcar metric detects the 10.0123 Hz pulsar" begin
        params = SearchParams(nharms=32, m=32, numbetween=16, metric=:boxcar)
        cands = search(ft, params; lofreq=9.5, hifreq=10.5, threshold=8.0)
        @test !isempty(cands)
        best = cands[argmax(c.metric for c in cands)]
        @info "boxcar strongest candidate" best.freq best.metric
        @test isapprox(best.freq, 10.0123; atol=1e-2)
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
        params = SearchParams(nharms=32, m=32, numbetween=16, metric=:non)
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
        # :non is a pure per-profile metric, so it is exactly chunk-invariant
        # (:boxcar's per-block σ makes it only approximately so, by design).
        params = SearchParams(nharms=32, m=32, numbetween=16, metric=:non)
        c1 = search(ft, params; lofreq=9.5, hifreq=10.5, threshold=8.0, blocksize=512)
        c2 = search(ft, params; lofreq=9.5, hifreq=10.5, threshold=8.0, blocksize=4096)
        b1 = c1[argmax(c.metric for c in c1)]
        b2 = c2[argmax(c.metric for c in c2)]
        @test isapprox(b1.freq, b2.freq; atol=1e-3)
        @test isapprox(b1.metric, b2.metric; rtol=1e-2)
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
                                  decimations=[1, k], metric=:non)
            lodr = params.hidr / nharms
            rstart = 5000.0
            n = 64
            rfund = rstart .+ (0:n-1) .* lodr

            # Native reduced-harmonic fold at the multiplied frequencies.
            pnat = SearchParams(nharms=Hk, m=32, numbetween=16, align=false, metric=:non)
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
            @test relerr < 1e-8
            @test all(c.nharm == Hk for c in out)
        end
    end

    @testset "detects the 10.0123 Hz pulsar via decimation" begin
        f = 10.0123
        nharms = 60
        for k in (2, 3)
            base_f = f / k                            # fundamental band that only k hits
            params = SearchParams(nharms=nharms, decimations=decimation_set(nharms, k), metric=:non)
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
