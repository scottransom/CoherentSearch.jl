using Test
using CoherentSearch

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

if isfile(EXAMPLE_FFT)
    ft = FFTFile(EXAMPLE_FFT)

    @testset "optimised path reproduces the reference (align=false)" begin
        # With a fixed numbetween and one chunk, the production path uses the
        # same grids as block_metrics, so it should match to ~machine precision.
        params = SearchParams(nharms=32, m=32, numbetween=16, align=false)
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
        raw = search(ft, params; lofreq=9.5, hifreq=10.5, threshold=8.0, remove=false)
        @test length(cands) < length(raw)
        # Dedup is a pure function of the list: it must return the exact strongest
        # candidate untouched (checked on the same list to avoid FFTW-plan jitter
        # between two separate search calls).
        rbest = raw[argmax(c.metric for c in raw)]
        dbest = remove_duplicates(raw)[argmax(c.metric for c in remove_duplicates(raw))]
        @test dbest.freq == rbest.freq && dbest.metric == rbest.metric
    end

    @testset "chunk size does not change the detection" begin
        params = SearchParams(nharms=32, m=32, numbetween=16)
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
                                  decimations=[1, k])
            lodr = params.hidr / nharms
            rstart = 5000.0
            n = 64
            rfund = rstart .+ (0:n-1) .* lodr

            # Native reduced-harmonic fold at the multiplied frequencies.
            pnat = SearchParams(nharms=Hk, m=32, numbetween=16, align=false)
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
            params = SearchParams(nharms=nharms, decimations=decimation_set(nharms, k))
            cands = search(ft, params; lofreq=base_f - 0.5, hifreq=base_f + 0.5, threshold=8.0)
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
else
    @info "Skipping search data tests; example file not found" EXAMPLE_FFT
end
