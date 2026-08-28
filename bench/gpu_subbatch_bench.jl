# Per-rung transform SUB-BATCHING, measured end to end (`docs/gpu_design.md` §4.9).
#
#   julia --project=<env with CoherentSearch + CUDA> bench/gpu_subbatch_bench.jl FILE.fft [--band lo hi]
#
# **What this is testing.** The six rungs want very different transform batch
# widths -- §0.3 measured their optima spread across the whole `Nprof` sweep on a
# 40 MB-L2 card -- while the interpolator, the transpose and the boxcar all want
# the largest chunk available.  With one batch per rung those pressures fight
# over `--blocksize`, and on the Ada the transform wins: it drags the chunk down
# to 16384, where §4.8 measured every other phase paying 1.42x.  Sub-batching
# lets each side have what it wants.
#
# **The pre-registered prediction this is here to score** (`docs/gpu_design.md` §4.8,
# finding (c)), from decomposing the blocksize sweep rather than from a model of
# the kernels:
#
#   RTX 4000 SFF Ada  : 40.7 -> ~32.7 ns/trial, i.e. **1.25x**, at blocksize 262144
#   RTX 2080 Super    : **1.00x** -- the control.  It already runs at 262144 and
#                       its 4 MB L2 has no residency to arrange, so `:auto`
#                       declines to split at all and the row should be a no-op.
#   GTX 1080          : 1.00x, same reason (2 MB L2).
#
# **That 1.25x is a model, not a result.**  It assumes the standalone probe's
# transform-sweep SHAPE carries into the search, and §4.8 established that the
# probe's *magnitude* does not (the Ada's isolated 3.9x transform win measured
# 3.07x in the pipeline, because cuFFT there shares L2 with everything else).
# So treat it as an upper bound.  If the Ada comes in near 1.25x the mechanism is
# confirmed; near 1.00x, the in-pipeline L2 is too contended for residency to be
# arrangeable and the whole item should be dropped rather than tuned.
#
# The sweep over explicit byte targets is the useful diagnostic either way: it
# bypasses the residency gate, so it shows where the real in-search knee is
# rather than where `_SUB_L2_FRACTION` guesses it is.  If the best target is far
# from 0.5 x L2, that constant is what to change.

using CoherentSearch, CUDA, Printf
const CS = CoherentSearch
const E = Base.get_extension(CoherentSearch, :CoherentSearchCUDAExt)

args = copy(ARGS)
lo, hi = 0.1, 33.3333
if (i = findfirst(==("--band"), args)) !== nothing
    lo = parse(Float64, args[i+1]); hi = parse(Float64, args[i+2])
    deleteat!(args, i:i+2)
end
isempty(args) && error("usage: gpu_subbatch_bench.jl FILE.fft [--band lo hi]")

CUDA.functional() || error("no functional CUDA device")
dev = CUDA.device()
l2 = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
ft = FFTFile(args[1])
params = SearchParams(nharms = 60, m = 16, decimations = collect(1:6))
lodr = params.hidr / params.nharms
total = max(0, floor(Int, (hi * ft.T - lo * ft.T) / lodr) + 1)
B = CS.require_gpu()
go(bs) = search(ft, params; lofreq = lo, hifreq = hi, blocksize = bs,
                threshold = 6.0, progress = :none, wisdom = false, backend = B)

println("="^78)
@printf("device : %s   L2 = %.1f MB   (0.5 x L2 = %d bytes, the :auto target)\n",
        CUDA.name(dev), l2 / 2^20, l2 ÷ 2)
@printf("host   : %s   file : %s   %d trials\n", gethostname(), basename(args[1]), total)
println("="^78)

# What :auto actually decides on this card, at the blocksize sub-batching is for.
CS.gpu_subbatch!(:auto)
gc = E.GPUChunk(Float32, params, 262144; subbatch = :auto)
println("\n:auto policy at blocksize 262144:")
println(E.subbatch_report(gc))
CUDA.unsafe_free!(gc.ftprofs)
for a in gc.cpulayout; CUDA.unsafe_free!(a); end
for a in gc.profs; CUDA.unsafe_free!(a); end
CUDA.reclaim()

time_at(bs) = (go(bs); minimum(begin s = time_ns(); go(bs); (time_ns()-s)/1e9 end for _ in 1:2))

# ---------------------------------------------------------------------------
# 1. The headline A/B: does sub-batching let the big blocksize win?
# ---------------------------------------------------------------------------
println("\nheadline A/B (:off is the pre-sub-batching behaviour):")
println("  blocksize   policy      wall (s)   ns/trial   cands")
base = NaN
for bs in (16384, 262144), pol in (:off, :auto)
    CS.gpu_subbatch!(pol)
    local t, nc
    try
        nc = length(go(bs)); t = time_at(bs)
    catch e
        @printf("  %9d   %-9s  SKIPPED: %s\n", bs, pol,
                first(split(sprint(showerror, e), "\n"))); continue
    end
    bs == 16384 && pol === :off && (global base = t)
    @printf("  %9d   %-9s %9.3f  %9.1f   %5d%s\n", bs, pol, t, t*1e9/total, nc,
            isnan(base) ? "" : @sprintf("   %.3fx", base/t))
end
@printf("  (ratios are against blocksize 16384 :off = %.3f s)\n", base)

# ---------------------------------------------------------------------------
# 2. Where the real in-search knee is.  Explicit targets bypass the residency
#    gate, so this sweeps past it in both directions.
# ---------------------------------------------------------------------------
println("\ntarget sweep at blocksize 262144 (explicit targets bypass the gate):")
println("  target        as x L2    wall (s)   ns/trial   k=1 cols")
for frac in (0.125, 0.25, 0.5, 1.0, 2.0)
    target = floor(Int, frac * l2)
    CS.gpu_subbatch!(target)
    local t
    try
        t = time_at(262144)
    catch e
        @printf("  %9d   %8.3f   SKIPPED: %s\n", target, frac,
                first(split(sprint(showerror, e), "\n"))); continue
    end
    @printf("  %9d   %8.3f  %9.3f  %9.1f   %8d\n", target, frac, t, t*1e9/total,
            E._sub_cols(Float32, 60, 262144, target; policy = false))
end
CS.gpu_subbatch!(:auto)
println("\nIf the best target is far from 0.5, change _SUB_L2_FRACTION in the extension.")
println("If :auto declined to split above, this card is a control and 1.00x is the")
println("expected answer -- record it as a confirmation, not a failure.")
