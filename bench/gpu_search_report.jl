# One-command GPU search report, for classifying a new card (`gpu_design.md`).
#
#   julia --project=<env with CoherentSearch + CUDA> bench/gpu_search_report.jl FILE.fft [--cpu] [--band lo hi]
#
# Prints device identity, a `--blocksize` sweep with per-phase GPU timings, the
# candidates found, and a pasteable summary block.  Everything a new card needs,
# in one output.
#
# `--cpu` adds a CPU arm for comparison.  It is OPTIONAL and off by default
# because a CPU run on an unfamiliar host is not comparable to fitzroy's 20-core
# Xeon anyway (three hosts have already differed by 2.4x per core), and it can
# double the runtime.  When you do use it, quote the host.
#
# Two things to know about the numbers:
#
#  - **Per-phase timing SERIALISES the GPU queue** (a wall-clock timer around a
#    CUDA launch measures the launch, not the work, so each phase needs a
#    synchronise around it).  So the sweep reports both: a clean total with timing
#    OFF, and the phase breakdown from a separate pass with it on.  Read the
#    shares from the second and the total from the first.
#  - **The best `--blocksize` is per-device and the known cards want opposite
#    ends** (GTX 1080, RTX 2080 Super and A100: 262144; RTX A4000: 131072;
#    RTX 4000 Ada: 8192, because its 40 MB L2 holds the whole pipeline at that
#    size).  Note the A100 has the SAME 40 MB and wants the other end, because
#    108 SMs need a big chunk to fill -- so this sweeps rather than assuming.
#    The sweep starts at 2048 because the Ada's in-search
#    optimum turned out to sit BELOW the standalone probe's knee -- when cuFFT
#    shares that L2 with the rest of the pipeline the effective knee moves down,
#    so a sweep that bottoms out at 16384 can miss it (`gpu_design.md` §4.8).
#  - **`scan` is HOST work and `download` is PCIe; only the five `:device` phases
#    describe the card.**  The `scan` share moved 1.75x between two hosts purely
#    because one had a faster CPU, so the breakdown below reports device-only
#    shares alongside the raw ones.  Classify a card on the device column.

using CoherentSearch, CUDA, Printf
const CS = CoherentSearch

args = copy(ARGS)
docpu = "--cpu" in args; filter!(!=("--cpu"), args)
lo, hi = 0.1, 33.3333
if (i = findfirst(==("--band"), args)) !== nothing
    lo = parse(Float64, args[i+1]); hi = parse(Float64, args[i+2])
    deleteat!(args, i:i+2)
end
isempty(args) && error("usage: gpu_search_report.jl FILE.fft [--cpu] [--band lo hi]")
fftfile = args[1]

CUDA.functional() || error("no functional CUDA device")
dev = CUDA.device()
cap = CUDA.capability(dev)
sms = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
clk = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_CLOCK_RATE) / 1e6
l2 = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
free, tot = CUDA.memory_info()

ft = FFTFile(fftfile)
params = SearchParams(nharms = 60, m = 16, decimations = collect(1:6))
lodr = params.hidr / params.nharms
total = max(0, floor(Int, (hi * ft.T - lo * ft.T) / lodr) + 1)

println("="^78)
@printf("device : %s  sm_%s  SMs=%d  clock=%.3f GHz  L2=%.1f MB  mem %.2f/%.2f GiB free\n",
        CUDA.name(dev), string(cap), sms, clk, l2 / 2^20, free / 2^30, tot / 2^30)
# The load average is in the header because it is the one property of the host
# that moves these numbers and was never recorded.  `usnea`'s A4000 report was
# taken at a load average of ~100 on 104 cores and reads ~1.2x slow, with its
# host-side `scan` share inflated well past anything the card explains
# (gpu_design.md §4.13).  Every card classified so far has been on someone
# else's machine.
const _LOADAVG = try
    strip(first(split(read("/proc/loadavg", String))))
catch
    "?"
end
@printf("host   : %s   julia %s   %d thread(s)   %d cores   load %s\n",
        gethostname(), VERSION, Threads.nthreads(), Sys.CPU_THREADS, _LOADAVG)
if _LOADAVG != "?" && parse(Float64, _LOADAVG) > 0.5 * Sys.CPU_THREADS
    @warn "This host is busy; the timings below will be slow and the phase table " *
          "may be meaningless (`gpu_timing!` brackets phases with a synchronise, " *
          "which measures scheduler latency on a saturated machine).  Re-run when " *
          "it is quiet, and quote the load with any number you report." load=_LOADAVG cores=Sys.CPU_THREADS
end
@printf("file   : %s   N=%d  T=%.1f s  amps=%.2f GiB\n",
        basename(fftfile), ft.N, ft.T, length(ft.amps) * 8 / 2^30)
@printf("search : %.4f-%.4f Hz  nharms=%d maxdecim=%d  ->  %d trial fundamentals\n",
        lo, hi, params.nharms, maximum(params.decimations), total)
println("="^78)

B = CS.require_gpu()
go(bk, bs) = search(ft, params; lofreq = lo, hifreq = hi, blocksize = bs,
                    threshold = 6.0, progress = :none, wisdom = false, backend = bk)

# candidate sanity, once
CS.gpu_timing!(false)
cands = go(B, 65536)
@printf("\ncandidates above S/N 6: %d\n", length(cands))
for c in first(cands, min(5, length(cands)))
    @printf("   %12.7f Hz   S/N %7.3f   nharm %3d\n", c.freq, c.metric, c.nharm)
end

println("\nblocksize sweep (timing OFF -- these are the honest totals):")
println("  blocksize    wall (s)    ns/trial   cands")
results = Tuple{Int,Float64}[]
# Up to 1048576 because the A100's sweep was still improving at 262144 (1.04x
# from the row below it) and so never bracketed its optimum -- the same mistake
# §4.11 caught at the OTHER end, where the Ada's real optimum turned out to sit
# below the old floor of 16384.  Rows that do not fit are skipped by
# `_check_device_memory` and cost nothing: at ~2912 B per trial, 1048576 needs
# 3.05 GiB of workspace on top of the amplitudes, so a small card simply drops
# the top rows.
for bs in (2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576)
    # Three searches per row: one warm-up plus two timed.  The candidate count
    # comes from the warm-up rather than a fourth call -- on a big file each
    # search is seconds, and a redundant one per row is minutes over the sweep.
    local t, nc
    try
        nc = length(go(B, bs))                        # warm, and gives the count
        t = minimum(begin s = time_ns(); go(B, bs); (time_ns() - s) / 1e9 end for _ in 1:2)
    catch e
        @printf("  %9d    SKIPPED: %s\n", bs, first(split(sprint(showerror, e), "\n")))
        continue
    end
    push!(results, (bs, t))
    @printf("  %9d   %9.3f   %9.1f   %5d\n", bs, t, t * 1e9 / total, nc)
end
best = isempty(results) ? (65536, NaN) : results[argmin(last.(results))]
@printf("  best: blocksize %d at %.3f s\n", best[1], best[2])

# ---------------------------------------------------------------------------
# The recommendation.  This is the whole point of the script for a user (as
# opposed to for `gpu_design.md`): `--blocksize` under `--gpu` defaults to
# `GPU_DEFAULT_BLOCKSIZE`, which is one constant chosen for its WORST case
# (within 1.14x of the optimum on all six cards measured), not a per-device rule.
# This script finds the remaining few percent.
#
# The optimum spans 8192 to 262144 -- a factor of 32 -- and is NOT predictable
# from the hardware.  It is a tug of war between L2 (a big cache wants a small
# chunk, so the whole pipeline stays resident) and SM count (a lot of SMs want a
# big one, because a small chunk cannot fill them).  Which wins is not something
# a spec sheet settles: the RTX 4000 Ada (40 MB L2, 48 SMs) wants 8192 while the
# A100 (the same 40 MB, 108 SMs) wants 262144 and is nearly 2x slower at 8192.
# A rule fitted to the first three cards predicted 6868 for the A100 and was
# wrong by 32x -- see gpu_design.md §4.11-§4.12 for why this is measured and not
# derived.
# ---------------------------------------------------------------------------
if !isempty(results)
    d = Dict(results)
    println("\n" * "-"^78)
    @printf("RECOMMENDATION for this card:  --blocksize %d\n", best[1])
    gdef = CS.GPU_DEFAULT_BLOCKSIZE
    if haskey(d, gdef)
        pen = d[gdef] / best[2]
        @printf("  Not passing --blocksize gets the GPU default of %d, which on this\n", gdef)
        @printf("  card costs %.2fx (%.3f s against %.3f s).\n", pen, d[gdef], best[2])
        pen < 1.02 && println("  (So on this card the default is already as good as tuning it.)")
    end
    if haskey(d, 2048)
        @printf("  For reference, the CPU's default of 2048 would cost %.2fx here.\n",
                d[2048] / best[2])
    end
    # How sharp is the optimum?  A user who guesses one row away should know
    # whether that costs 1% or 20%.
    near = sort([(abs(log2(bs / best[1])), bs, t) for (bs, t) in results])
    length(near) > 1 && @printf("  Neighbouring rows: %s\n",
        join((@sprintf("%d -> %.2fx", bs, t / best[2]) for (_, bs, t) in near[2:min(3, end)]), ", "))
    println("-"^78)
end

println("\nper-phase breakdown at blocksize $(best[1]) (timing ON -- read the SHARES;")
println("the synchronisation this needs inflates the total, so take that from above):")
CS.gpu_timing!(true); CS.gpu_phase_reset!()
go(B, best[1])
CS.gpu_timing!(false)
pt = CS.gpu_phase_times()
kind = CS.GPU_PHASE_KIND
acc = sum(last.(pt))
devsecs = sum(s for (i, (_, s)) in enumerate(pt) if kind[i] === :device)
println("                                  share of  share of")
println("  phase       where      time (s)     total    device")
for (i, (name, secs)) in enumerate(pt)
    devshare = kind[i] === :device ? @sprintf("%7.1f%%", 100 * secs / devsecs) : "       -"
    @printf("  %-10s %-9s %8.4f  %7.1f%%  %s\n",
            name, string(kind[i]), secs, 100 * secs / acc, devshare)
end
@printf("  %-10s %-9s %8.4f  %7.1f%%\n", "= device", "", devsecs, 100 * devsecs / acc)
@printf("  %-10s %-9s %8.4f  %7.1f%%\n", "= host+pcie", "", acc - devsecs, 100 * (acc - devsecs) / acc)
@printf("  %-10s %-9s %8.4f   (instrumented; clean total was %.3f s)\n", "TOTAL", "", acc, best[2])
println("  NOTE: `scan` is this HOST's CPU and `download` is PCIe -- neither is a")
println("        property of the card.  Compare cards on the device column.")

if docpu
    println("\nCPU arm (this host's CPU, NOT fitzroy's Xeon -- quote the host):")
    go(CPUBackend(), 2048)
    tc = minimum(begin s = time_ns(); go(CPUBackend(), 2048); (time_ns() - s) / 1e9 end for _ in 1:2)
    @printf("  CPU -t %-2d  %8.3f s   ->  GPU is %.2fx\n", Threads.nthreads(), tc, tc / best[2])
end

println("\n" * "="^78)
println("PASTE THIS BLOCK BACK:")
@printf("  %s | sm_%s | %d SMs | %.2f GHz | L2 %.0f MB | %.1f GiB\n",
        CUDA.name(dev), string(cap), sms, clk, l2 / 2^20, tot / 2^30)
@printf("  %s | %d trials | best blocksize %d | %.3f s | %.1f ns/trial | %d cands\n",
        basename(fftfile), total, best[1], best[2], best[2] * 1e9 / total, length(cands))
@printf("  phases (of total):  %s\n",
        join((@sprintf("%s %.1f%%", n, 100 * s / acc) for (n, s) in pt), "  "))
@printf("  phases (of device): %s   [device %.1f%% of instrumented]\n",
        join((@sprintf("%s %.1f%%", n, 100 * s / devsecs)
              for (i, (n, s)) in enumerate(pt) if kind[i] === :device), "  "),
        100 * devsecs / acc)
println("="^78)
