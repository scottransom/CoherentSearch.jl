# Boxcar-gate tuning for the GPU backend (`docs/gpu_design.md` §4.2).
#
#   julia --project=<env with CoherentSearch + CUDA> bench/gpu_boxcar_bench.jl
#
# Two kernels over the block size, on GENUINE chunk profiles (not `randn` of the
# same shape) sitting in the production device buffers:
#
#   variant 1  shared-memory wrapped prefix sum, then a max-scan per width.
#              Uses (nbins+wmax+1)*B floats of shared, which is what caps occupancy.
#   variant 2  no shared memory: per width, slide the window keeping the running
#              sum in a register.  Re-reads the profile ~2*nwidths times, betting
#              those are L1 hits and that occupancy is worth more.
#   variant 3  as 1, with the subtracted prefix term hoisted out of the width
#              loop.  This is what ships.
#
# Variants 1 and 3 are additionally run with and without the COALESCED staging
# load (`docs/gpu_design.md` 4.15).  Both are in one build precisely so the A/B is a
# kwarg rather than a checkout -- this file's standing lesson about precompile
# differences masquerading as effects.  The coalesced arm reads the block's B
# columns as B contiguous runs and transposes into shared (row stride B+1, hence
# the slightly larger shared figure); the other is the per-thread stride-nbins
# gather that shipped until then, and the two are bit-exact to each other.
#
# Timings are summed over all six rungs and scaled to the reference workload, so
# the numbers are directly comparable with bench/gpu_pipeline_bench.jl's boxcar
# column.

using CoherentSearch, CUDA, Printf, LinearAlgebra
const CS = CoherentSearch
const Ext = Base.get_extension(CoherentSearch, :CoherentSearchCUDAExt)
const BENCH_TRIALS = 8_363_442

const REF = "/data1/git/CoherentSearch.jl/PM0063_034C1_DM445.0_red.fft"
ft = FFTFile(isfile(REF) ? REF : ARGS[1])
params = SearchParams(nharms = 60, m = 16, decimations = collect(1:6))
Nprof = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 65536
WT = Float32

bestof(f, n = 5) = minimum(begin
    CUDA.@sync f(); t = time_ns(); CUDA.@sync f(); (time_ns() - t) / 1e9
end for _ in 1:n)

gc = Ext.GPUChunk(WT, params, Nprof)
# The kernels index a per-width normaliser the host builds (`_boxcar_shape!`);
# with nothing lost to Nyquist that is the fully-filled table.  Without this the
# device array is still zeros and the timings would be measured on garbage.
for i in eachindex(gc.ks)
    Ext.refresh_invsws!(gc, i, gc.ks[i], gc.Hk[i], fill(true, params.nharms), 1.0)
end
gp = Ext.GPUInterpPlan(build_direct_plans(WT, params, 0.1 * ft.T))
d = CuArray(ft.amps)
Ext.gpu_fill_ftprofs!(gc.ftprofs, gp, d, ft, 0, Nprof)
Ext.fill_stacks!(gc, Nprof)
for i in eachindex(gc.ks)
    mul!(gc.profs[i], gc.plans[i], gc.cpulayout[i])
end
out = CUDA.zeros(Float32, Nprof)
reps = BENCH_TRIALS / Nprof

println("device : ", CUDA.name(CUDA.device()))
println("Nprof  : ", Nprof, "   rungs: ", gc.ks)
println("widths : ", [length(w) for w in gc.hwidths], "   wmax: ", [w[end] for w in gc.hwidths])
println("\nboxcar over all six rungs, scaled to the reference workload:")
println("  variant  coal    B     stage (s)   shared/block")

allrungs(B, v, c) = () -> for i in eachindex(gc.ks)
    Ext.gpu_boxcar!(out, gc.profs[i], Nprof, 2gc.Hk[i], gc.widths[i],
                    gc.invsws[i]; B = B, variant = v, coalesced = c)
end
shbytes(B, v, c) = v == 2 ? 0 :
    Ext._bc_shmem(2 * gc.Hk[1], gc.hwidths[1][end], B, c) * 4

# correctness: every configuration must agree with the shipped one
const REFVALS = Ref{Union{Nothing,Vector{Float32}}}(nothing)
for (v, Bs, coals) in ((1, (32, 64), (false, true)),
                       (2, (32, 64), (false,)),
                       (3, (32, 64), (false, true)))
    for B in Bs, c in coals
        shbytes(B, v, c) > 48000 && continue
        f = allrungs(B, v, c)
        f(); CUDA.synchronize()
        got = Array(out)
        REFVALS[] === nothing && (REFVALS[] = got)
        r = REFVALS[]
        # Variants 1 and 3 and both staging paths are BIT-EXACT to each other, so
        # anything but 0.0 here is a bug, not a tolerance.  (Variant 2 sums in a
        # different order and is the one row allowed to differ.)
        agree = maximum(abs.(got .- r)) / maximum(abs, r)
        t = bestof(f)
        @printf("     %d     %-5s %3d    %8.4f      %6d B   (agrees to %.1e)\n",
                v, c, B, t * reps, shbytes(B, v, c), agree)
    end
end
