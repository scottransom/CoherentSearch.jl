# Profile a warm coherent search and emit a StatProfilerHTML flame graph.
#
# Run SINGLE-THREADED for clean attribution:
#     julia --project=bench -t 1 bench/profile_search.jl FILE.fft [hifreq]
#
# The search is compiled on a tiny warm-up band first so the profile reflects
# steady-state hot-loop cost, not JIT.  Output: bench/statprof/index.html.
#
# Defaults mirror the standard heavy configuration (`--metric sd2 --maxdecim 6`
# ⇒ nharms=60, six decimation factors), over a sub-band so a single-threaded
# run finishes in tens of seconds while still exercising every code path
# (all harmonics, all decimations, the metric, and post-processing).

using CoherentSearch
using Profile
using StatProfilerHTML
using Printf
using Base.Threads: nthreads

const FILE   = length(ARGS) >= 1 ? ARGS[1] : "PM0063_034C1_DM445.0_red.fft"
const HIFREQ = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 5.0

function build_params()
    nharms = 60
    SearchParams(
        nharms      = nharms,
        threshold   = 6.0,
        metric      = :sd2,
        pexp        = 0.5,
        decimations = decimation_set(nharms, 6),
    )
end

function run_search(ft, params; lofreq, hifreq)
    search(ft, params; lofreq=lofreq, hifreq=hifreq,
           blocksize=2048, threshold=params.threshold,
           progress=:none)
end

function main()
    ft = FFTFile(FILE)
    params = build_params()
    @info "profile_search" file=FILE T=ft.T nharms=params.nharms decims=params.decimations threads=nthreads()

    # Warm-up: compile the whole pipeline on a tiny band (results discarded).
    @info "warming up (compiling)…"
    run_search(ft, params; lofreq=0.1, hifreq=0.2)

    # Timed, un-profiled baseline for this band.
    @info "timing warm search" hifreq=HIFREQ
    t = @elapsed cands = run_search(ft, params; lofreq=0.1, hifreq=HIFREQ)
    @info "warm search done" seconds=round(t; digits=2) ncands=length(cands)

    # Profiled run.
    Profile.clear()
    Profile.init(; n=10^8, delay=0.0005)   # deep buffer, 0.5 ms sampling
    @profile run_search(ft, params; lofreq=0.1, hifreq=HIFREQ)

    # Flat text summary first (robust even if the HTML writer errors).
    println("\n=== Profile.print (flat, sorted by count) ===")
    Profile.print(; format=:flat, sortedby=:count, mincount=100)

    # Aggregate self-time (leaf frames) into coarse buckets so the split is
    # authoritative rather than eyeballed from the flat list.
    aggregate_buckets()

    # StatProfilerHTML's writer is broken on Julia 1.12 (Core.MethodInstance
    # field change); the flame-graph HTML is optional — the bucket table above
    # is the authoritative split.  Guard it so the run still exits 0.
    try
        cd(@__DIR__) do
            statprofilehtml()
        end
        @info "wrote flame graph" path=joinpath(@__DIR__, "statprof", "index.html")
    catch err
        @warn "StatProfilerHTML flame graph unavailable on this Julia" err
    end
end

# Classify a leaf stack frame into a coarse cost bucket, by function name so it
# is robust to line-number shifts in search.jl.
function classify(sf)
    fn   = string(sf.func)
    file = string(sf.file)
    (occursin("fft", lowercase(file)) || occursin("FFTW", file)) && return "FFTW"
    occursin("sort", file)                             && return "median-sort"
    (fn in ("_median!", "_select!", "_swap!"))         && return "median-select"
    (fn == "pow_body" || fn == "^")                    && return "pow(w^pexp)"
    fn == "_profile_snr"                               && return "profile_snr-body"
    (fn == "interp_tile!" || fn == "fill_harmonic_row!") && return "interp_tile/fill_row"
    fn == "uniform_linear_interp"                      && return "uniform_linear_interp"
    occursin("complex.jl", file)                       && return "complex-arith"
    occursin("broadcast.jl", file)                     && return "broadcast(spec*coeffs)"
    return "other"
end

function aggregate_buckets()
    data, lidict = Profile.retrieve()
    counts = Dict{String,Int}()
    total = 0
    # Walk the sample buffer: 0 separates samples; the FIRST nonzero after a 0
    # (reading backwards) is the leaf. Simpler: count the leaf of each sample.
    i = 1
    n = length(data)
    while i <= n
        # find end of this sample (a 0 delimiter, possibly a metadata block)
        j = i
        while j <= n && data[j] != 0
            j += 1
        end
        if j > i
            leaf = data[i]                      # first entry = leaf frame ip
            frames = get(lidict, leaf, nothing)
            if frames !== nothing && !isempty(frames)
                b = classify(frames[1])   # innermost inlined frame = the real leaf
                counts[b] = get(counts, b, 0) + 1
                total += 1
            end
        end
        # skip the 0 delimiter plus the trailing metadata zeros
        i = j
        while i <= n && data[i] == 0
            i += 1
        end
    end
    println("\n=== self-time by bucket (leaf frame, ", total, " samples) ===")
    for (b, c) in sort(collect(counts); by=x->-x[2])
        @printf("  %-26s %6d  %5.1f%%\n", b, c, 100c/max(total,1))
    end
end

main()
