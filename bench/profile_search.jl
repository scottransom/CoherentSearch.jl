# Profile a warm coherent search and emit a StatProfilerHTML flame graph.
#
# Run SINGLE-THREADED for clean attribution:
#     julia --project=bench -t 1 bench/profile_search.jl FILE.fft [hifreq]
#
# The search is compiled on a tiny warm-up band first so the profile reflects
# steady-state hot-loop cost, not JIT.  Output: bench/statprof/index.html.
#
# Defaults mirror the standard heavy configuration (`--metric boxcar --maxdecim 6`
# ⇒ nharms=60, six decimation factors), over a clean mid-band sub-band so a
# single-threaded run finishes in tens of seconds while still exercising every
# code path (all harmonics, all decimations, the boxcar metric + per-block σ,
# and post-processing) without the low-frequency red-noise candidate flood.

using CoherentSearch
using Profile
using StatProfilerHTML
using Printf
using Base.Threads: nthreads

const FILE   = length(ARGS) >= 1 ? ARGS[1] : "PM0063_034C1_DM445.0_red.fft"
const LOFREQ = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 5.0
const HIFREQ = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 30.0

function build_params()
    nharms = 60
    SearchParams(
        nharms      = nharms,
        threshold   = 6.0,
        metric      = :boxcar,
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

    # Warm-up: compile the whole pipeline on a tiny mid-band (results discarded).
    @info "warming up (compiling)…"
    run_search(ft, params; lofreq=LOFREQ, hifreq=LOFREQ + 0.2)

    # Timed, un-profiled baseline for this band.
    @info "timing warm search" lofreq=LOFREQ hifreq=HIFREQ
    t = @elapsed cands = run_search(ft, params; lofreq=LOFREQ, hifreq=HIFREQ)
    @info "warm search done" seconds=round(t; digits=2) ncands=length(cands)

    # Profiled run.
    Profile.clear()
    Profile.init(; n=10^8, delay=0.0005)   # deep buffer, 0.5 ms sampling
    @profile run_search(ft, params; lofreq=LOFREQ, hifreq=HIFREQ)

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

# Classify a stack frame into a coarse cost bucket, by function name so it is
# robust to line-number shifts in search.jl.  Base arithmetic/indexing frames
# deliberately return "other" so the aggregator walks *past* them to the nearest
# enclosing search.jl/FFTW frame (the boxcar width-scan and the interp
# `spec.*coeffs` multiply/gather both inline down to Base getindex/setindex/`*`,
# which a leaf-only classifier would dump into an uninformative 56% "other").
function classify(sf)
    fn   = string(sf.func)
    file = string(sf.file)
    (occursin("fft", lowercase(file)) || occursin("FFTW", file)) && return "FFTW"
    occursin("sort", file)                             && return "median-sort"
    (fn in ("_median!", "_select!", "_swap!"))         && return "median-select"
    fn == "_profile_boxcar"                            && return "boxcar-metric"
    fn == "_block_sigma"                               && return "block-sigma"
    fn == "boxcar_widths"                              && return "boxcar-setup"
    (fn == "pow_body" || fn == "^")                    && return "pow(w^pexp)"
    fn == "_profile_snr"                               && return "profile_snr-body"
    (fn == "interp_tile!" || fn == "fill_harmonic_row!") && return "interp (fft-correlate)"
    fn == "decim_pass!"                                && return "decim (gather+brfft)"
    fn == "fill_chunk_profiles!"                       && return "fill-chunk (other)"
    fn == "uniform_linear_interp"                      && return "uniform_linear_interp"
    return "other"
end

# Attribute a sample to the *nearest-leaf* frame that maps to a named bucket, so
# Base leaves (getindex/setindex/`*`) are charged to the search.jl function that
# runs them rather than to "other".  The sample buffer is leaf-first; metadata
# ints don't resolve through `lidict`, so a `get(...) === nothing` miss is simply
# skipped — no dependence on the exact Julia-version metadata-block layout.
function aggregate_buckets()
    data, lidict = Profile.retrieve()
    counts = Dict{String,Int}()
    total = 0
    i = 1
    n = length(data)
    while i <= n
        j = i
        while j <= n && data[j] != 0
            j += 1
        end
        if j > i
            bucket = "other"
            @inbounds for t in i:(j - 1)          # leaf-first: first named frame wins
                frames = get(lidict, data[t], nothing)
                frames === nothing && continue
                hit = false
                for f in frames                   # innermost inlined frame first
                    b = classify(f)
                    if b != "other"
                        bucket = b; hit = true; break
                    end
                end
                hit && break
            end
            counts[bucket] = get(counts, bucket, 0) + 1
            total += 1
        end
        i = j
        while i <= n && data[i] == 0
            i += 1
        end
    end
    println("\n=== self-time by bucket (nearest named frame, ", total, " samples) ===")
    for (b, c) in sort(collect(counts); by=x->-x[2])
        @printf("  %-26s %6d  %5.1f%%\n", b, c, 100c/max(total,1))
    end
end

main()
