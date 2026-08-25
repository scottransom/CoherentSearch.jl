# Compare `bin/toy_coherent_search.jl` against the production `search`.
#
#     julia --project=bench -t 1 bench/toy_vs_production.jl [FILE.fft] \
#         [--lofreq 0.1] [--hifreq 0.4] [--nharms 60] [--maxdecim 6] [--m 16]
#         [--threshold 6.0] [--reps 2] [--nband 4] [--sigma-only] [--no-sigma]
#
# Three things, in one process so both arms run under identical conditions:
#
#   1. TIMING.  Wall clock and per-trial cost for each, at the same parameters
#      over the same band.  The ratio measures what the whole optimisation
#      programme in `Summary_and_Future_Work.md` bought, against an
#      implementation of the same algorithm with none of it.  Run with `-t 1`:
#      the toy is single-threaded, so a threaded production arm measures the
#      thread count rather than the code.
#
#   2. CANDIDATES.  Both lists, cross-matched by Fourier frequency.  These
#      should agree on what is there and roughly on how strongly; they are NOT
#      expected to agree exactly, because the toy divides by an analytic noise
#      scale and production by a measured one (see 3), and because the toy
#      scans the full geometric width bank where production prunes it across
#      the decimation ladder (`ladder_boxcar_widths`).
#
#   3. SIGMA.  The open question this harness exists to answer: production
#      measures sigma-hat per chunk (`_block_sigma`), at ~26% of all metric work
#      and with ~1% sampling error that lands directly on every reported S/N
#      (reported S/N is exactly 1/sigma-hat).  The toy computes it instead, from
#      the input FFT being normalised.  This reports the two side by side, per
#      fold depth, on the real data — which is the measurement that decides
#      whether production can drop the estimator.
#
# Note the toy is ~190x slower, so pick a NARROW band: the defaults search
# 0.1-0.4 Hz — 75k trial fundamentals, which is ~18 s for the toy and ~0.1 s for
# production.
#
# Measured 2026-08-24, i7-10510U, `-t 1`, PM0063 0.1-0.4 Hz, `--nband 4`:
#   * timing: toy 18.34 s (243 us/trial) vs production 0.096 s (1.27 us/trial),
#     i.e. **190.8x**.
#   * candidates: the 0.2603 Hz candidate at 7.37 (toy) vs 7.32 (production),
#     a ratio of 1.0059 — the interpolation-truncation bias, and nothing else.
#   * sigma: analytic/exact has median 1.00436 and spans 0.9918-1.0217 (3.02%),
#     against production's OWN subsampling error of 0.9806-1.0335 (5.39%).
#     **The closed form is closer to the exact sigma-hat than the estimator the
#     search currently ships.**  The deep folds are the tight ones (k=1 spans
#     1.0071-1.0095, around the predicted 1.0064); the shallow ones scatter more
#     because they average over fewer harmonics and fewer bins.

using CoherentSearch
using CoherentSearch: Workspace, build_direct_plans, fill_chunk_profiles!,
                      _block_sigma, DecimBuf
using LinearAlgebra: mul!
using Printf
using Base.Threads: nthreads

include(joinpath(@__DIR__, "..", "bin", "toy_coherent_search.jl"))
using .ToyCoherentSearch

function parseargs(argv)
    o = (file = "PM0063_034C1_DM445.0_red.fft", lofreq = 0.1, hifreq = 0.4,
         nharms = 60, maxdecim = 6, m = 16, threshold = 6.0, reps = 2,
         sigma_only = false, do_sigma = true, nband = 4)
    d = Dict(pairs(o))
    i = 1
    while i <= length(argv)
        a = argv[i]
        if     a == "--lofreq";     d[:lofreq]    = parse(Float64, argv[i += 1])
        elseif a == "--hifreq";     d[:hifreq]    = parse(Float64, argv[i += 1])
        elseif a == "--nharms";     d[:nharms]    = parse(Int, argv[i += 1])
        elseif a == "--maxdecim";   d[:maxdecim]  = parse(Int, argv[i += 1])
        elseif a == "--m";          d[:m]         = parse(Int, argv[i += 1])
        elseif a == "--threshold";  d[:threshold] = parse(Float64, argv[i += 1])
        elseif a == "--reps";       d[:reps]      = parse(Int, argv[i += 1])
        elseif a == "--nband";      d[:nband]     = parse(Int, argv[i += 1])
        elseif a == "--sigma-only"; d[:sigma_only] = true
        elseif a == "--no-sigma";   d[:do_sigma]   = false
        elseif !startswith(a, "--"); d[:file] = a
        else error("unknown flag $a")
        end
        i += 1
    end
    return NamedTuple(d)
end

const OPT = parseargs(ARGS)

# ---------------------------------------------------------------------------
# 3. The sigma comparison — the reason this file exists
# ---------------------------------------------------------------------------

"""
    sigma_report(ft, params, r_lo, nprof) -> Vector

For one production chunk of real data, print the noise scale three ways at every
fold depth: the EXACT pooled MAD over every bin of the chunk, the SUBSAMPLED one
production actually uses (`_BOXCAR_SIGMA_SAMPLES = 8192` bins), and the ANALYTIC
closed form.

Quoting all three separates two questions that are easy to conflate.
`analytic/exact` is the real quantity of interest: does the closed form describe
this data?  `sub/exact` is the estimator's own sampling error (~1% rms by
construction), which bounds how precisely any single chunk can answer it.

Everything is in the production hot loop's units.  Production folds with an
UNNORMALISED `brfft`, which is `nbins` times `irfft`, so the analytic scale there
is `nbins * analytic_sigma(nbins, nfilled) = sqrt(2*nlow + 0.5*nnyq)`, about
`sqrt(nbins)`.  The null hypothesis is NOT 1.000: the m-bin interpolation kernel
keeps only `S_m ~ 1 - 0.203/m` of the noise power, so the analytic value should
sit ~0.203/(2m) high (0.64% at m = 16).  Anything beyond that is the data saying
the normalisation does not hold as well as the closed form assumes — which is
the real risk in dropping the measurement, and cannot be settled on synthetic
noise.
"""
function sigma_report(ft::FFTFile, params::SearchParams, r_lo::Real, nprof::Integer;
                      header::Bool = true)
    nharms = params.nharms
    lodr = params.hidr / nharms
    ws = Workspace(params, nprof)
    dplans = build_direct_plans(params, r_lo)
    fill_chunk_profiles!(ws, dplans, ft, params, r_lo, lodr, nprof; t0 = 0)

    if header
        println("\nNoise scale: exact / production-subsampled / analytic, per fold depth")
        @printf("  %9s %-3s %6s %7s %10s %10s %10s %9s %9s\n",
                "f_lo (Hz)", "k", "nbins", "nfilled", "exact", "subsampled",
                "analytic", "ana/exact", "sub/exact")
    end

    # How many harmonics of the LAST trial in the chunk still carry data; the
    # analytic scale depends on it (past-Nyquist rows are zero and noiseless).
    rmax = r_lo + (nprof - 1) * lodr
    navail = 0
    for h in 1:nharms
        iszero(fourier_interpolate(ft, h * rmax, params.m)) && break
        navail = h
    end

    rows = NamedTuple[]
    for (k, prof) in vcat([(1, ws.profs)], [(db.k, _decim_profiles!(ws, db)) for db in ws.decims])
        Hk = nharms ÷ k
        nbins = 2Hk
        nfilled = min(Hk, navail ÷ k)
        nfilled >= 1 || continue
        # A buffer long enough for every bin makes `_block_sigma` take its exact
        # branch; the 8192-sample one is what the search actually runs.
        exact = Float64(_block_sigma(prof, nbins, nprof, similar(prof, nbins * nprof)))
        sub   = Float64(_block_sigma(prof, nbins, nprof, similar(prof, 8192)))
        ana   = nbins * ToyCoherentSearch.analytic_sigma(nbins, nfilled)
        push!(rows, (; f = r_lo / ft.T, k, nbins, nfilled, exact, sub, ana,
                       ratio = ana / exact, subratio = sub / exact))
        @printf("  %9.4f %-3d %6d %7d %10.5f %10.5f %10.5f %9.5f %9.5f\n",
                r_lo / ft.T, k, nbins, nfilled, exact, sub, ana, ana / exact, sub / exact)
    end
    return rows
end

"""
    sigma_scan(ft, params, lofreq, hifreq, nband, nprof)

Run [`sigma_report`](@ref) at `nband` log-spaced points across the band, so a
smooth frequency TREND (red-noise residue, which is a real property of the data)
can be told apart from chunk-to-chunk SCATTER (the estimator's own error).  This
is the distinction that decides whether the analytic scale is usable in
production: a constant offset is calibratable, a frequency-dependent one is not.
"""
function sigma_scan(ft::FFTFile, params::SearchParams, lofreq::Real, hifreq::Real,
                    nband::Integer, nprof::Integer)
    edges = exp.(range(log(lofreq), log(hifreq); length = nband + 1))
    rows = NamedTuple[]
    for i in 1:nband
        append!(rows, sigma_report(ft, params, edges[i] * ft.T, nprof; header = (i == 1)))
    end
    println("\n  Summary of analytic/exact:")
    @printf("  %-3s %6s %9s %9s %9s   %s\n", "k", "nbins", "min", "median", "max",
            "by frequency window (low -> high)")
    for k in sort(unique(r.k for r in rows))
        rk = [r for r in rows if r.k == k]
        v = sort([r.ratio for r in rk])
        med = v[(length(v) + 1) ÷ 2]
        @printf("  %-3d %6d %9.5f %9.5f %9.5f   ", k, rk[1].nbins, first(v), med, last(v))
        for r in rk
            @printf("%7.4f", r.ratio)
        end
        println()
    end
    allr = [r.ratio for r in rows]
    allsub = [r.subratio for r in rows]
    println()
    @printf("  analytic/exact over all (k, window): min %.5f, median %.5f, max %.5f, spread %.2f%%\n",
            minimum(allr), sort(allr)[(length(allr) + 1) ÷ 2], maximum(allr),
            100 * (maximum(allr) / minimum(allr) - 1))
    @printf("  production's own subsampling error (sub/exact):  min %.5f, max %.5f, spread %.2f%%\n",
            minimum(allsub), maximum(allsub), 100 * (maximum(allsub) / minimum(allsub) - 1))
    @printf("  predicted analytic/exact from interpolation truncation alone: %.5f (m = %d)\n",
            1 / sqrt(1 - 0.203 / params.m), params.m)
    return rows
end

# Run one decimation's transform on the chunk already in `ws`, returning its
# profile matrix (the same thing `decim_pass!` scores).
function _decim_profiles!(ws::Workspace, db::DecimBuf)
    mul!(db.dprofs, db.brfftplan, db.src)
    return db.dprofs
end

# ---------------------------------------------------------------------------
# 1 & 2. Timing and candidates
# ---------------------------------------------------------------------------

function candidate_table(label, cands)
    @printf("\n%s: %d candidate(s)\n", label, length(cands))
    isempty(cands) && return
    @printf("  %-8s %18s %10s %7s\n", "S/N", "freq (Hz)", "r (bins)", "#harm")
    for c in first(cands, 12)
        @printf("  %-8.2f %18.10f %10.2f %7d\n", c.metric, c.freq, c.r, c.nharm)
    end
    length(cands) > 12 && println("  ... and $(length(cands) - 12) more")
end

# Match two candidate lists by Fourier frequency (bins), within `tol`.
function crossmatch(a, b; tol = 1.0)
    println("\nCross-matched candidates (by Fourier frequency, within $tol bins)")
    @printf("  %18s %10s %10s %8s %7s %7s\n",
            "freq (Hz)", "toy S/N", "prod S/N", "toy/prod", "toy #h", "prod #h")
    used = falses(length(b))
    for c in a
        j = findfirst(i -> !used[i] && abs(b[i].r - c.r) <= tol, eachindex(b))
        if j === nothing
            @printf("  %18.10f %10.2f %10s %8s %7d %7s\n", c.freq, c.metric, "-", "-", c.nharm, "-")
        else
            used[j] = true
            @printf("  %18.10f %10.2f %10.2f %8.4f %7d %7d\n",
                    c.freq, c.metric, b[j].metric, c.metric / b[j].metric, c.nharm, b[j].nharm)
        end
    end
    for (i, hit) in enumerate(used)
        hit || @printf("  %18.10f %10s %10.2f %8s %7s %7d\n",
                       b[i].freq, "-", b[i].metric, "-", "-", b[i].nharm)
    end
end

function main()
    ft = FFTFile(OPT.file)
    params = SearchParams(nharms = OPT.nharms, m = OPT.m, threshold = OPT.threshold,
                          decimations = decimation_set(OPT.nharms, OPT.maxdecim),
                          precision = :f64)   # :f64 so the comparison is about
                                              # the algorithm, not the profile width
    ntrials = length(ToyCoherentSearch.trial_grid(ft, OPT.lofreq, OPT.hifreq,
                                                  OPT.nharms, params.hidr))
    @printf("%s: T = %.1f s, N = %d\n", basename(ft.path), ft.T, ft.N)
    @printf("band %.4f-%.4f Hz, nharms %d, m %d, k = 1..%d, threshold %.2f\n",
            OPT.lofreq, OPT.hifreq, OPT.nharms, OPT.m, OPT.maxdecim, OPT.threshold)
    @printf("%d trial fundamentals x %d fold depths = %d folds; %d thread(s)\n",
            ntrials, length(params.decimations), ntrials * length(params.decimations),
            nthreads())
    nthreads() == 1 || println("NOTE: run with -t 1 — the toy is single-threaded, " *
                               "so a threaded production arm measures the thread count.")

    if OPT.do_sigma
        sigma_scan(ft, params, OPT.lofreq, OPT.hifreq, OPT.nband,
                   min(2048, max(64, ntrials)))
    end
    OPT.sigma_only && return

    # Warm-up: compile both arms before anything is timed.
    ToyCoherentSearch.toy_search(ft; nharms = OPT.nharms, m = OPT.m,
                                 lofreq = OPT.lofreq, hifreq = min(OPT.hifreq, OPT.lofreq + 1e-3),
                                 maxdecim = OPT.maxdecim, threshold = 1e9, progress = false)
    search(ft, params; lofreq = OPT.lofreq, hifreq = min(OPT.hifreq, OPT.lofreq + 1e-3),
           threshold = 1e9, progress = :none, wisdom = false)

    ttoy, tprod = Float64[], Float64[]
    local toycands, prodcands
    for rep in 1:OPT.reps
        # Interleaved, so any machine drift hits both arms equally.
        t = @elapsed toycands = ToyCoherentSearch.toy_search(ft;
                nharms = OPT.nharms, m = OPT.m, lofreq = OPT.lofreq, hifreq = OPT.hifreq,
                maxdecim = OPT.maxdecim, threshold = OPT.threshold, progress = false)
        push!(ttoy, t)
        t = @elapsed prodcands = search(ft, params; lofreq = OPT.lofreq, hifreq = OPT.hifreq,
                threshold = OPT.threshold, progress = :none, wisdom = false,
                remove = false, harm_remove = false)
        push!(tprod, t)
        @printf("  rep %d: toy %8.3f s   production %8.3f s\n", rep, ttoy[end], tprod[end])
    end

    mt, mp = minimum(ttoy), minimum(tprod)
    println("\nTiming (best of $(OPT.reps), single-threaded)")
    @printf("  toy         %8.3f s   %8.2f us/trial\n", mt, 1e6 * mt / ntrials)
    @printf("  production  %8.3f s   %8.2f us/trial\n", mp, 1e6 * mp / ntrials)
    @printf("  production is %.1fx faster\n", mt / mp)

    # Both lists get the same post-processing, so the comparison is of the
    # search and not of the candidate bookkeeping.
    post(c) = remove_harmonics(remove_duplicates(c; dr_tol = 1.0); numharm = 16)
    toycands, prodcands = post(toycands), post(prodcands)
    sort!(toycands; by = c -> c.metric, rev = true)
    sort!(prodcands; by = c -> c.metric, rev = true)
    candidate_table("toy (analytic sigma, full width bank)", toycands)
    candidate_table("production (measured sigma, pruned ladder bank)", prodcands)
    crossmatch(toycands, prodcands)
    return nothing
end

main()
