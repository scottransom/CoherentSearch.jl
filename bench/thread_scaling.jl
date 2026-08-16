# Thread-scaling benchmark: wall clock, CPU-seconds and speedup vs thread count,
# against both the ideal line and an Amdahl fit.  Writes a CSV and a two-panel
# plot.
#
#     julia --project=bench bench/thread_scaling.jl FILE.fft
#     julia --project=bench bench/thread_scaling.jl FILE.fft --threads 1,2,4,8,20 --reps 3
#
# The driver re-invokes itself as a `--worker` at each thread count (nthreads is
# fixed at process start), and each worker times the *warm* in-process `search`
# call only.  Start-up is a fixed serial cost that would otherwise contaminate
# the Amdahl fit — see the start-up section of CLAUDE.md for what it costs.
#
# CPU-seconds are reported alongside wall clock deliberately.  For *identical*
# work they should be flat in the thread count; the amount by which they inflate
# is memory-stall and clock-throttle time, and reading it is how you tell a code
# problem from a machine limit (this repo has been bitten by that before).
#
# NOTE ON WHAT THIS MEASURES.  Thread scaling is the right axis only if a search
# is actually deployed multi-threaded.  Production pulsar searches are commonly
# run one single-threaded process per DM, in which case the figure that governs
# throughput is the `-t 1` CPU-seconds column, not the speedup curve.  See §3 of
# Summary_and_Future_Work.md.

using CoherentSearch
using Base.Threads: nthreads
using Printf: @printf, @sprintf
using Logging: ConsoleLogger, Warn, global_logger

# `search` narrates its candidate post-processing at Info level, once per rep per
# thread count.  Nothing here reads it, and it buries the RESULT lines.
global_logger(ConsoleLogger(stderr, Warn))

const REPO = dirname(@__DIR__)

# Process CPU-seconds (user + sys) from /proc, so a run's CPU cost is visible
# without wrapping the process in /usr/bin/time.  NaN off Linux.
function cpu_seconds()
    try
        f = read("/proc/self/stat", String)
        # utime/stime are fields 14/15, but comm (field 2) may contain spaces --
        # split after the closing paren so the field numbering is trustworthy.
        rest = split(f[findlast(==(')'), f)+2:end])
        return (parse(Float64, rest[12]) + parse(Float64, rest[13])) /
               parse(Float64, strip(read(`getconf CLK_TCK`, String)))
    catch
        return NaN
    end
end

# --- argument parsing -------------------------------------------------------

function parse_args(argv)
    file = "PM0063_034C1_DM445.0_red.fft"
    threads = [1, 2, 4, 8, 16, Sys.CPU_THREADS]
    reps, lofreq, hifreq = 3, 0.1, 33.3333
    nharms, maxdecim, threshold = 60, 6, 6.3
    worker, out = false, joinpath(REPO, "bench", "thread_scaling")
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--worker";        worker = true
        elseif a == "--threads";   threads = parse.(Int, split(argv[i+=1], ","))
        elseif a == "--reps";      reps = parse(Int, argv[i+=1])
        elseif a == "--lofreq";    lofreq = parse(Float64, argv[i+=1])
        elseif a == "--hifreq";    hifreq = parse(Float64, argv[i+=1])
        elseif a == "--nharms";    nharms = parse(Int, argv[i+=1])
        elseif a == "--maxdecim";  maxdecim = parse(Int, argv[i+=1])
        elseif a == "--threshold"; threshold = parse(Float64, argv[i+=1])
        elseif a == "--out";       out = argv[i+=1]
        elseif startswith(a, "--"); error("unknown option $a")
        else                       file = a
        end
        i += 1
    end
    threads = sort(unique(threads))
    return (; file, threads, reps, lofreq, hifreq, nharms, maxdecim, threshold, worker, out)
end

const OPT = parse_args(ARGS)

# --- worker: time the warm search at this process's nthreads() --------------

function run_worker()
    ft = FFTFile(OPT.file)
    params = SearchParams(nharms=OPT.nharms, threshold=OPT.threshold,
                          decimations=decimation_set(OPT.nharms, OPT.maxdecim))
    go(hi) = search(ft, params; lofreq=OPT.lofreq, hifreq=hi, blocksize=2048,
                    threshold=params.threshold, progress=:none)
    go(OPT.lofreq * 1.002)                   # compile / plan / page in (discarded)
    for r in 1:OPT.reps
        c0 = cpu_seconds()
        t = @elapsed cands = go(OPT.hifreq)
        # RESULT lines are what the driver parses; anything else here is noise.
        println("RESULT ", nthreads(), " ", r, " ", t, " ", cpu_seconds() - c0,
                " ", length(cands))
    end
end

OPT.worker && (run_worker(); exit(0))

# --- driver -----------------------------------------------------------------

"""
    amdahl_fit(n, S) -> (s, smax)

Least-squares serial fraction `s` for `S(n) = 1/(s + (1-s)/n)`, and the implied
ceiling `1/s`.  Linear in `s` after inverting: with `u = 1/n`,
`1/S = u + s(1-u)`, so this is a fit through the origin — no optimiser needed.
"""
function amdahl_fit(n::Vector{Int}, S::Vector{Float64})
    u = 1 ./ n
    x = 1 .- u
    y = 1 ./ S .- u
    s = sum(x .* y) / sum(x .^ 2)
    s = clamp(s, 1e-9, 1.0)
    return s, 1 / s
end

med(v) = sort(v)[cld(length(v), 2)]

println("thread_scaling: ", OPT.file)
println("  band ", OPT.lofreq, "-", OPT.hifreq, " Hz, nharms=", OPT.nharms,
        ", maxdecim=", OPT.maxdecim, ", threshold=", OPT.threshold,
        ", reps=", OPT.reps)
println("  host has ", Sys.CPU_THREADS, " CPU threads\n")

walls, cpus, ncands = Float64[], Float64[], Int[]
for nt in OPT.threads
    cmd = `$(Base.julia_cmd()[1]) --project=$(joinpath(REPO,"bench")) -t $nt
           $(@__FILE__) $(OPT.file) --worker --reps $(OPT.reps)
           --lofreq $(OPT.lofreq) --hifreq $(OPT.hifreq) --nharms $(OPT.nharms)
           --maxdecim $(OPT.maxdecim) --threshold $(OPT.threshold)`
    w, c, nc = Float64[], Float64[], Int[]
    for ln in eachline(open(cmd))
        startswith(ln, "RESULT ") || continue
        f = split(ln)
        push!(w, parse(Float64, f[4])); push!(c, parse(Float64, f[5]))
        push!(nc, parse(Int, f[6]))
    end
    isempty(w) && error("worker at -t $nt produced no RESULT lines")
    push!(walls, med(w)); push!(cpus, med(c)); push!(ncands, nc[1])
    @printf("  -t %-3d  wall %7.2f s   cpu %7.1f s   cpu/wall %5.2f   ncands %d\n",
            nt, med(w), med(c), med(c) / med(w), nc[1])
end

# Candidate count must not depend on thread count -- chunk->thread assignment is
# round-robin over whole chunks, so a difference here means a real parallel bug.
allequal(ncands) || @warn "candidate count varies with thread count!" ncands

t1 = walls[1]
speedup = t1 ./ walls
eff = speedup ./ OPT.threads
s, smax = amdahl_fit(OPT.threads, speedup)

@printf("\n  Amdahl fit: serial fraction s = %.4f  ->  ceiling %.1fx\n", s, smax)
@printf("  best measured: %.2fx at -t %d (%.0f%% efficiency)\n",
        maximum(speedup), OPT.threads[argmax(speedup)],
        100 * eff[argmax(speedup)])
@printf("  CPU-seconds inflate %.0f%% from -t %d to -t %d",
        100 * (cpus[end] / cpus[1] - 1), OPT.threads[1], OPT.threads[end])
println(" (identical work: pure stall/throttle)")

csv = OPT.out * ".csv"
open(csv, "w") do io
    println(io, "threads,wall_s,cpu_s,speedup,efficiency,ncands")
    for i in eachindex(OPT.threads)
        @printf(io, "%d,%.4f,%.3f,%.4f,%.4f,%d\n", OPT.threads[i], walls[i],
                cpus[i], speedup[i], eff[i], ncands[i])
    end
end
println("  wrote ", csv)

# --- plot -------------------------------------------------------------------

using CairoMakie

fig = Figure(size = (1000, 430))
nt = OPT.threads
xt = (nt, string.(nt))

ax1 = Axis(fig[1, 1], xscale = log2, yscale = log2, xticks = xt, yticks = xt,
           xlabel = "threads", ylabel = "speedup vs -t 1",
           title = "Thread scaling")
lines!(ax1, nt, float.(nt), color = :gray, linestyle = :dash, label = "ideal")
nn = range(extrema(nt)...; length = 200)
lines!(ax1, nn, 1 ./ (s .+ (1 - s) ./ nn), color = :orangered,
       label = @sprintf("Amdahl s=%.3f (max %.1fx)", s, smax))
scatterlines!(ax1, nt, speedup, color = :dodgerblue, label = "measured")
axislegend(ax1, position = :lt)

ax2 = Axis(fig[1, 2], xscale = log2, xticks = xt, xlabel = "threads",
           ylabel = "CPU-seconds", title = "CPU cost of identical work")
hlines!(ax2, [cpus[1]], color = :gray, linestyle = :dash, label = "-t 1 baseline")
scatterlines!(ax2, nt, cpus, color = :seagreen, label = "measured")
ylims!(ax2, 0, 1.1 * maximum(cpus))
axislegend(ax2, position = :lt)

png = OPT.out * ".png"
save(png, fig)
println("  wrote ", png)
