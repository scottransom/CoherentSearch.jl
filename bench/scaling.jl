# Warm thread-scaling timer (no profiling): compile on a tiny band, then time the
# real search at the current nthreads().  Run at several thread counts:
#     for t in 1 2 4; do julia --project=bench -t $t bench/scaling.jl FILE.fft 30; done
# Reports wall seconds and candidate count (which must be identical across -t).

using CoherentSearch
using Base.Threads: nthreads

const FILE   = length(ARGS) >= 1 ? ARGS[1] : "PM0063_034C1_DM445.0_red.fft"
const LOFREQ = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 5.0
const HIFREQ = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 30.0

ft = FFTFile(FILE)
params = SearchParams(nharms=60, threshold=6.0, metric=:boxcar,
                      decimations=decimation_set(60, 6))
run(hi) = search(ft, params; lofreq=LOFREQ, hifreq=hi, blocksize=2048,
                 threshold=params.threshold, progress=:none)

run(LOFREQ + 0.2)                          # warm-up / compile (discarded)
t = @elapsed cands = run(HIFREQ)
println("threads=", nthreads(), "  lofreq=", LOFREQ, "  hifreq=", HIFREQ,
        "  seconds=", round(t; digits=2), "  ncands=", length(cands))
