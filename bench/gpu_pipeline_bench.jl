# Stage-2 phase timings for the GPU pipeline (`docs/gpu_design.md` §4).
#
#   julia --project=<env with CoherentSearch + CUDA> bench/gpu_pipeline_bench.jl [FILE.fft] [Nprof...]
#
# Times the three device phases separately -- interpolation, the six inverse
# transforms, and the six boxcar gates -- and scales each to the reference
# workload (PM0063 at the riptide bench config, 8,363,442 trial fundamentals), so
# every column is directly comparable with the CPU's ~1.01 s at `-t 20` on
# fitzroy and with the projections in `docs/gpu_design.md` §0.4/§0.46.
#
# It also times the decimated-stack copy separately, because that copy is a
# placeholder: cuFFT cannot transform a strided view, so the CPU's in-place
# stride trick does not port, and the intended fix is to fold those stores into
# the interpolation kernel (where they cost no extra reads).  Its line here is
# the budget that fix has to beat.
#
# TRAP: every timing is inside `CUDA.@sync`.  A bare wall-clock timer around a
# CUDA call measures the launch, not the work.

using CoherentSearch, CUDA, Printf, LinearAlgebra
const CS = CoherentSearch
const Ext = Base.get_extension(CoherentSearch, :CoherentSearchCUDAExt)
const BENCH_TRIALS = 8_363_442

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

args = filter(a -> !occursin(r"^\d+$", a), ARGS)
nprofs = [parse(Int, a) for a in ARGS if occursin(r"^\d+$", a)]
isempty(nprofs) && (nprofs = [16384, 32768, 65536, 131072])
ft, ftname = load_or_synth(isempty(args) ? (isfile(REF_FFT) ? REF_FFT : nothing) : args[1])
params = SearchParams(nharms = 60, m = 16, decimations = collect(1:6))
WT = Float32
r_lo = 0.1 * ft.T

println("device : ", CUDA.name(CUDA.device()))
println("file   : ", ftname)
println("params : nharms=", params.nharms, " maxdecim=", maximum(params.decimations),
        " weights=", WT)
println("\nper-phase, scaled to the reference workload (", BENCH_TRIALS, " trials):")
println("   Nprof     interp  transpose   transform    boxcar  |    total   vs CPU -t 20")

d_amps = CuArray(ft.amps)
plans = build_direct_plans(WT, params, r_lo)
gp = Ext.GPUInterpPlan(plans)

for Nprof in nprofs
    gc = Ext.GPUChunk(WT, params, Nprof)
    # The kernels index a per-width normaliser the host builds (`_boxcar_shape!`);
    # with nothing lost to Nyquist that is the fully-filled table.  Without this the
    # device array is still zeros and the timings would be measured on garbage.
    for i in eachindex(gc.ks)
        Ext.refresh_invsws!(gc, i, gc.ks[i], gc.Hk[i], fill(true, params.nharms), 1.0)
    end
    out = CUDA.zeros(Float32, Nprof)
    reps = BENCH_TRIALS / Nprof                     # chunks in the reference workload

    t_interp = bestof(() -> Ext.gpu_fill_ftprofs!(gc.ftprofs, gp, d_amps, ft, 0, Nprof))
    t_stack = bestof(() -> Ext.fill_stacks!(gc, Nprof))
    t_gather = bestof(() -> for i in eachindex(gc.ks); Ext._fill_stack_gather!(gc, i); end)
    t_xform = bestof(function ()
        for i in eachindex(gc.ks)
            mul!(gc.profs[i], gc.plans[i], gc.cpulayout[i])
        end
    end)
    t_box = bestof(function ()
        for i in eachindex(gc.ks)
            Ext.gpu_boxcar!(out, gc.profs[i], Nprof, 2gc.Hk[i], gc.widths[i],
                            gc.invsws[i])
        end
    end)
    tot = (t_interp + t_stack + t_xform + t_box) * reps
    @printf("  %7d   %7.4f   %7.4f   %9.4f  %8.4f  | %8.4f   %6.2fx   (gather would be %.3f)\n",
            Nprof, t_interp * reps, t_stack * reps, t_xform * reps, t_box * reps,
            tot, 1.01 / tot, t_gather * reps)
    for a in (gc.profs..., gc.cpulayout...); CUDA.unsafe_free!(a); end
    CUDA.unsafe_free!(gc.ftprofs); CUDA.unsafe_free!(out)
end
println("\n`transpose` turns the interpolator's (Nprof, nharms+1) into the six dense")
println("(Hk+1, Nprof) stacks cuFFT transforms 1.75-1.88x faster; the parenthesised")
println("figure is the strided copyto! gather it replaced.")
println("CPU reference is fitzroy's 20-core Xeon at 1.01 s -- NOT this host's CPU.")
