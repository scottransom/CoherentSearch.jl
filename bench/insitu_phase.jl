# One-arm in-situ measurement: warm wall clock plus the `phase_times` split, as
# machine-parseable lines.  Built for sweeping a `const` in `src/` that cannot be
# varied in-process (`_BC_BATCH`, `_BC_TR_BJ`, `DIRECT_GROUP_V`), where each
# value costs a re-precompile and so a separate invocation.
#
#     julia --project=bench -t 1 bench/insitu_phase.jl FILE.fft [--reps 3]
#         [--precision f32] [--lofreq 0.1] [--hifreq 33.3333] [--nharms 60]
#         [--maxdecim 6] [--threshold 6.3] [--m 16] [--tag LABEL]
#
# Read the phase SHARES, not the seconds.  Whole-machine drift between
# invocations runs ~6% on fitzroy while within-run scatter is ~1%, so ranking
# absolute times across separately-launched arms ranks noise — that mistake is
# recorded in CLAUDE.md's `_BC_BATCH` entry.  Shares are drift-robust because the
# phases a knob does not touch drift with the ones it does.
#
# Emits `RESULT <tag> wall=<s> ncands=<n>` and `PHASE <tag> <name> <s> <share>`.

using CoherentSearch, Printf

file = "PM0063_034C1_DM445.0_red.fft"
lofreq, hifreq, reps = 0.1, 33.3333, 3
nharms, maxdecim, threshold, m = 60, 6, 6.3, 16
prec, tag = :f32, "-"
i = 1
while i <= length(ARGS)
    a = ARGS[i]
    if     a == "--reps";      global reps = parse(Int, ARGS[i += 1])
    elseif a == "--lofreq";    global lofreq = parse(Float64, ARGS[i += 1])
    elseif a == "--hifreq";    global hifreq = parse(Float64, ARGS[i += 1])
    elseif a == "--nharms";    global nharms = parse(Int, ARGS[i += 1])
    elseif a == "--maxdecim";  global maxdecim = parse(Int, ARGS[i += 1])
    elseif a == "--threshold"; global threshold = parse(Float64, ARGS[i += 1])
    elseif a == "--m";         global m = parse(Int, ARGS[i += 1])
    elseif a == "--precision"; global prec = Symbol(ARGS[i += 1])
    elseif a == "--tag";       global tag = ARGS[i += 1]
    elseif !startswith(a, "--"); global file = a
    else error("unknown flag $a")
    end
    global i += 1
end

params = SearchParams(nharms=nharms, threshold=threshold, m=m, precision=prec,
                      decimations=decimation_set(nharms, maxdecim))
ft = FFTFile(file)

run1() = search(ft, params; lofreq=lofreq, hifreq=hifreq, blocksize=2048,
                threshold=threshold, progress=:none)

run1()                                            # warm: JIT + FFTW plans
walls = Float64[]
phases = Dict{String,Vector{Float64}}()
ncands = 0
for r in 1:reps
    phase_reset!()
    t = @elapsed cands = run1()
    push!(walls, t)
    global ncands = length(cands)
    for (nm, sec) in phase_times()
        push!(get!(phases, nm, Float64[]), sec)
    end
end

med(v) = sort(v)[cld(length(v), 2)]
wall = med(walls)
# Top-level phases only: the `decim-brfft-k*` rows are a breakdown OF
# `decim-brfft` and would be double-counted in a share.
tops = [nm for nm in keys(phases) if !occursin(r"^decim-brfft-k", nm)]
total = sum(med(phases[nm]) for nm in tops)
@printf("RESULT %s wall=%.3f ncands=%d accounted=%.3f\n", tag, wall, ncands, total)
for nm in sort(tops; by = nm -> -med(phases[nm]))
    s = med(phases[nm])
    @printf("PHASE %s %-16s %8.3f %7.3f%%\n", tag, nm, s, 100 * s / total)
end
