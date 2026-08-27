# The transpose-and-decimate stage, per-rung against fused (`gpu_design.md` §4.15).
#
#   julia --project=<env with CoherentSearch + CUDA> bench/gpu_transpose_bench.jl [FILE.fft] [Nprof...]
#
# **Why this stage has a bench of its own.**  On the two cards re-measured on a
# quiet host it runs at 87-102% of the device copy bandwidth -- it is DRAM
# saturated, so no amount of kernel tuning helps and the only lever is moving
# fewer bytes.  The per-rung kernel reads `ftprofs` once per rung (153 of its 61
# columns, 2.5x over) and writes 153 rows: 2448 B per trial.  The fused kernel
# reads each column ONCE into a shared tile and writes every rung from it: 1712 B
# per trial, 1.43x less.  So the number to watch is not the speedup on its own
# but the **%copy column** -- if the fused kernel is still at ~100% of copy then
# the win is exactly the traffic ratio and there is nothing further to get here;
# if it has dropped, the shared-memory bank conflicts on the even-`k` rungs (see
# the kernel comment) are costing something and `_GPU_TR_T` is worth sweeping.
#
# TRAP: every timing is inside `CUDA.@sync`.  A bare wall-clock timer around a
# CUDA call measures the launch, not the work.

using CoherentSearch, CUDA, Printf
const CS = CoherentSearch
const Ext = Base.get_extension(CoherentSearch, :CoherentSearchCUDAExt)

const REF_FFT = "/data1/git/CoherentSearch.jl/PM0063_034C1_DM445.0_red.fft"

function load_or_synth(path)
    if path !== nothing && isfile(path)
        return FFTFile(path), basename(path)
    end
    N, dt = 8388608, 0.00025
    amps = [ComplexF32(randn(Float32) * 0.7071f0, randn(Float32) * 0.7071f0) for _ in 1:(N ÷ 2)]
    inf = SimpleInf("synthetic.inf", "GPUBENCH", 5.0e4, N, dt, 0.0)
    return FFTFile("synthetic.fft", amps, inf, N, N * dt, 1.0 / (N * dt), true, true,
                   real(amps[1]), imag(amps[1])), "synthetic (PM0063 geometry)"
end

bestof(f, n = 5) = minimum(begin
    CUDA.@sync f(); t = time_ns(); CUDA.@sync f(); (time_ns() - t) / 1e9
end for _ in 1:n)

# The device copy figure the probe prints, re-measured here so the %copy column
# never has to be carried between scripts (`gbs` counts read AND write, which is
# what makes it comparable with the traffic model below).
function copy_gbs()
    n = 1 << 26
    a = CUDA.rand(Float32, n); b = CUDA.zeros(Float32, n)
    t = bestof(() -> copyto!(b, a))
    g = 2 * 4n / t / 1e9
    CUDA.unsafe_free!(a); CUDA.unsafe_free!(b)
    return g
end

args = filter(a -> !occursin(r"^\d+$", a), ARGS)
nprofs = [parse(Int, a) for a in ARGS if occursin(r"^\d+$", a)]
isempty(nprofs) && (nprofs = [65536, 131072, 262144])
ft, ftname = load_or_synth(isempty(args) ? (isfile(REF_FFT) ? REF_FFT : nothing) : args[1])
params = SearchParams(nharms = 60, m = 16, decimations = collect(1:6))
WT = Float32
r_lo = 0.1 * ft.T

nh1 = params.nharms + 1
rows = sum(fld(params.nharms, k) + 1 for k in params.decimations)
B_perrung = (rows + rows) * 2 * sizeof(WT)        # read 153 cols + write 153 rows
B_fused   = (nh1 + rows) * 2 * sizeof(WT)         # read 61 cols  + write 153 rows

println("device : ", CUDA.name(CUDA.device()))
println("file   : ", ftname)
println("params : nharms=", params.nharms, " decimations=", params.decimations,
        " weights=", WT, "  tile T=", Ext._GPU_TR_T)
gbs = copy_gbs()
@printf("copy   : %.0f GB/s (read+write, the bar for the %%copy column)\n", gbs)
@printf("traffic: per-rung %d B/trial, fused %d B/trial -> %.2fx less\n\n",
        B_perrung, B_fused, B_perrung / B_fused)

println("   Nprof   per-rung ns/trial  %copy |   fused ns/trial  %copy |  speedup")

d_amps = CuArray(ft.amps)
plans = build_direct_plans(WT, params, r_lo)
gp = Ext.GPUInterpPlan(plans)

for Nprof in nprofs
    gc = Ext.GPUChunk(WT, params, Nprof)
    Ext.gpu_fill_ftprofs!(gc.ftprofs, gp, d_amps, ft, 0, Nprof)

    tp = bestof(() -> Ext.fill_stacks!(gc, Nprof; fused = false))
    ref = [Array(x) for x in gc.cpulayout]
    for x in gc.cpulayout; fill!(x, Complex{WT}(-999, 999)); end
    tf = bestof(() -> Ext.fill_stacks!(gc, Nprof; fused = true))
    ok = all(Array(gc.cpulayout[i]) == ref[i] for i in eachindex(gc.ks))
    ok || error("fused transpose disagrees with per-rung -- indexing bug")

    nsp = tp / Nprof * 1e9
    nsf = tf / Nprof * 1e9
    @printf("%8d   %9.4f  %11.0f%% | %9.4f  %11.0f%% |   %.3fx\n",
            Nprof, nsp, 100 * (B_perrung / nsp) / gbs,
            nsf, 100 * (B_fused / nsf) / gbs, tp / tf)

    for x in gc.cpulayout; CUDA.unsafe_free!(x); end
    for x in gc.profs;     CUDA.unsafe_free!(x); end
    CUDA.unsafe_free!(gc.ftprofs)
end

println("""
\nThe values are verified equal to the per-rung kernel's on every row (bit-exact,
not within a tolerance -- both kernels only copy).  Read %copy, not the speedup:
at ~100% the stage is DRAM-bound and the speedup IS the traffic ratio.""")
