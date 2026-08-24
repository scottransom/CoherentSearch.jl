# A/B the profile-stage precision (`SearchParams.precision`) *in one process*.
#
#     julia --project=bench -t 1 bench/precision_ab.jl [FILE.fft] [--lofreq 0.1]
#         [--hifreq 33.3333] [--reps 3] [--nharms 60] [--maxdecim 6]
#         [--threshold 6.3] [--m 16] [--arms f64,f32]
#
# Both precisions are compiled into one build (`Workspace{…,P}`), so the arms
# alternate under identical cache/thermal conditions with no branch switching and
# no Julia precompile cost inside the measured interval — the failure mode
# recorded in the `ab-benchmark-worktrees` note, which cost one confidently wrong
# result.  Only the *warm in-process* `search` is timed; start-up is excluded.
#
# Reports median wall clock per arm plus the in-situ phase split (`phase_times`),
# which is the thing to read: three optimisation calls in this project have been
# decided by an isolated micro-benchmark and come out backwards in the real
# search, so the phase table — measured where the work actually runs — is the
# authority, not a kernel timed on its own.

using CoherentSearch
using Printf
using Base.Threads: nthreads

function parseargs(argv)
    file = "PM0063_034C1_DM445.0_red.fft"
    lofreq, hifreq, reps = 0.1, 33.3333, 3
    nharms, maxdecim, threshold, m = 60, 6, 6.3, 16
    arms = [:f64, :f32]
    i = 1
    while i <= length(argv)
        a = argv[i]
        if     a == "--lofreq";    lofreq = parse(Float64, argv[i += 1])
        elseif a == "--hifreq";    hifreq = parse(Float64, argv[i += 1])
        elseif a == "--reps";      reps = parse(Int, argv[i += 1])
        elseif a == "--nharms";    nharms = parse(Int, argv[i += 1])
        elseif a == "--m";         m = parse(Int, argv[i += 1])
        elseif a == "--maxdecim";  maxdecim = parse(Int, argv[i += 1])
        elseif a == "--threshold"; threshold = parse(Float64, argv[i += 1])
        elseif a == "--arms";      arms = Symbol.(split(argv[i += 1], ","))
        elseif !startswith(a, "--"); file = a
        else error("unknown flag $a")
        end
        i += 1
    end
    return (; file, lofreq, hifreq, reps, nharms, maxdecim, threshold, m, arms)
end

const OPT = parseargs(ARGS)

mkparams(prec) = SearchParams(nharms=OPT.nharms, threshold=OPT.threshold, m=OPT.m,
                              decimations=decimation_set(OPT.nharms, OPT.maxdecim),
                              precision=prec)

# One timed repetition: wall seconds, CPU seconds, candidates, and the phase
# accumulators for *this* run only.
function timed(ft, params, hifreq)
    phase_reset!()
    c0 = Sys.cpu_info()
    t = @elapsed cands = search(ft, params; lofreq=OPT.lofreq, hifreq=hifreq,
                                blocksize=2048, threshold=params.threshold,
                                progress=:none)
    c1 = Sys.cpu_info()
    cpu = sum((c1[i].cpu_times!user + c1[i].cpu_times!nice + c1[i].cpu_times!sys) -
              (c0[i].cpu_times!user + c0[i].cpu_times!nice + c0[i].cpu_times!sys)
              for i in eachindex(c1)) / 1e9
    return (; wall=t, cpu, cands, phases=phase_times())
end

med(v) = sort(v)[cld(length(v), 2)]

function main()
    ft = FFTFile(OPT.file)
    @printf("precision A/B — %s  T=%.1f s   threads=%d\n", OPT.file, ft.T, nthreads())
    @printf("  band %g–%g Hz, nharms=%d, m=%d, maxdecim=%d, threshold=%g, reps=%d\n\n",
            OPT.lofreq, OPT.hifreq, OPT.nharms, OPT.m, OPT.maxdecim, OPT.threshold, OPT.reps)

    allparams = Dict(a => mkparams(a) for a in OPT.arms)
    # Compile + page in every arm on a sliver of the same band (discarded).
    for a in OPT.arms
        timed(ft, allparams[a], OPT.lofreq * 1.002)
    end

    walls  = Dict(a => Float64[] for a in OPT.arms)
    cpus   = Dict(a => Float64[] for a in OPT.arms)
    ncands = Dict(a => Int[] for a in OPT.arms)
    # Phase accumulators are summed over reps and divided at the end.
    phases = Dict(a => Dict{String,Float64}() for a in OPT.arms)
    for r in 1:OPT.reps
        for a in OPT.arms                      # alternate every rep, not every block
            res = timed(ft, allparams[a], OPT.hifreq)
            push!(walls[a], res.wall); push!(cpus[a], res.cpu)
            push!(ncands[a], length(res.cands))
            for (k, v) in res.phases
                phases[a][k] = get(phases[a], k, 0.0) + v
            end
            @printf("  rep %d  %-4s  wall %7.2f s   cpu %7.2f s   ncands %d\n",
                    r, a, res.wall, res.cpu, length(res.cands))
        end
    end

    println("\n=== medians ===")
    ref = med(walls[first(OPT.arms)])
    for a in OPT.arms
        @printf("  %-4s  wall %7.2f s   cpu %7.2f s   ncands %d   %.3fx\n",
                a, med(walls[a]), med(cpus[a]), ncands[a][1], ref / med(walls[a]))
    end

    println("\n=== phase split (CPU seconds per rep, in situ) ===")
    keys_all = sort(collect(PHASE_NAMES);
                    by = k -> -maximum(get(phases[a], k, 0.0) for a in OPT.arms))
    @printf("  %-18s", "phase")
    for a in OPT.arms; @printf("%10s", a); end
    length(OPT.arms) == 2 && @printf("%10s", "Δ")
    println()
    # Slots 10.. (`decim-brfft-k*`) are nested *inside* `decim-brfft`, so summing
    # every row double-counts the decimated transforms.  Exclude them from the
    # total; the accounted sum should land just under the wall clock at `-t 1`.
    nested(k) = startswith(k, "decim-brfft-k")
    tot = Dict(a => 0.0 for a in OPT.arms)
    for k in keys_all
        @printf("  %-18s", k)
        for a in OPT.arms
            v = get(phases[a], k, 0.0) / OPT.reps
            nested(k) || (tot[a] += v)
            @printf("%10.2f", v)
        end
        if length(OPT.arms) == 2
            a, b = OPT.arms
            va = get(phases[a], k, 0.0); vb = get(phases[b], k, 0.0)
            @printf("%9.1f%%", va > 0 ? 100 * (vb - va) / va : 0.0)
        end
        println()
    end
    @printf("  %-18s", "TOTAL (top-level)")
    for a in OPT.arms; @printf("%10.2f", tot[a]); end
    println()
end

main()
