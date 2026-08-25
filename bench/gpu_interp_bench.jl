# Stage-1 gate for the GPU track (`gpu_design.md` §4): the interpolation kernel,
# GPU against CPU, on the production plan tables and a real `.fft`.
#
#   julia --project=<env-with-CoherentSearch+CUDA> bench/gpu_interp_bench.jl [FILE.fft]
#
# Reports ns per (harmonic, trial) so the two are comparable across chunk sizes,
# and the GPU at several `Nprof` because §3.2 expects the GPU chunk to be ~10^5
# trials rather than the CPU's 2048.
#
# The CPU arm is single-threaded on purpose HERE -- it is the per-core cost the
# kernel is being compared against -- but the headline ratio that matters is
# against `-t 20`, and the script prints both.  §0.1 of `gpu_design.md` records
# the one time that distinction was got wrong.
#
# TRAP: a wall-clock timer around a CUDA launch measures the LAUNCH, not the
# work.  Every GPU timing here is wrapped in `CUDA.@sync`.

using CoherentSearch, CUDA, Printf
const CS = CoherentSearch
const Ext = Base.get_extension(CoherentSearch, :CoherentSearchCUDAExt)

fftfile = length(ARGS) >= 1 ? ARGS[1] :
          "/data1/git/CoherentSearch.jl/PM0063_034C1_DM445.0_red.fft"
ft = FFTFile(fftfile)
params = SearchParams(nharms = 60, m = 16, decimations = collect(1:6))
r_lo = 0.1 * ft.T
WT = Float32

bestof(f, n = 5) = minimum(begin
    t0 = time_ns(); f(); (time_ns() - t0) / 1e9
end for _ in 1:n)

println("file    : ", basename(fftfile))
println("params  : nharms=", params.nharms, " m=", params.m, " weights=", WT)
println("threads : ", Threads.nthreads(), " (Julia)")

# --- CPU: the fill loop, exactly as `fill_chunk_profiles!` runs it -----------
function cpu_fill_time(n)
    ws = CS.Workspace(params, n)
    dplans = build_direct_plans(WT, params, r_lo)
    run() = begin
        fill!(ws.ftprofs, 0)
        for dp in dplans
            CS.fill_harmonic_row_direct!(ws, dp, ft, params, 0, n)
        end
    end
    run()                                   # warm up (JIT out of the measurement)
    return bestof(run)
end

# --- GPU: the same fill, on device ------------------------------------------
function gpu_fill_time(n)
    plans = build_direct_plans(WT, params, r_lo)
    gp = Ext.GPUInterpPlan(plans)
    d_amps = CuArray(ft.amps)
    ftp = CUDA.zeros(Complex{WT}, n, params.nharms + 1)
    run() = CUDA.@sync begin
        fill!(ftp, 0)
        Ext.gpu_fill_ftprofs!(ftp, gp, d_amps, ft, 0, n)
    end
    run()
    t = bestof(run)
    CUDA.unsafe_free!(ftp); CUDA.unsafe_free!(d_amps)
    return t
end

println("\n              Nprof        ms    ns per (harmonic,trial)")
const CPU_NS = Ref(0.0)
for n in (2048, 65536)
    t = cpu_fill_time(n)
    ns = t * 1e9 / (n * params.nharms)
    n == 2048 && (CPU_NS[] = ns)
    @printf("  CPU  -t 1  %7d  %8.3f    %8.4f\n", n, t * 1e3, ns)
end
for n in (2048, 65536, 262144)
    t = gpu_fill_time(n)
    ns = t * 1e9 / (n * params.nharms)
    @printf("  GPU        %7d  %8.3f    %8.4f    %6.1fx one core\n",
            n, t * 1e3, ns, CPU_NS[] / ns)
end

nt = Threads.nthreads()
println()
println("Against one CPU core the ratio above is the honest per-core number.")
@printf("Against a 20-core socket, divide by 20: that is the comparison that\n")
@printf("matters (gpu_design.md 0.1).  This run had %d Julia thread%s.\n",
        nt, nt == 1 ? "" : "s")
