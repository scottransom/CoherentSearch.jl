# Micro-benchmarks for the three hot-loop buckets identified in the workload
# model, plus a whole-chunk timing.  Uses the real Workspace/plans so the
# measured cost is representative of the production path.
#
#     julia --project=bench -t 1 bench/microbench.jl [FILE.fft]
#
# All benchmarks are single-threaded and run on one warm chunk in mid-band.

using CoherentSearch
using BenchmarkTools
const CS = CoherentSearch

const FILE = length(ARGS) >= 1 ? ARGS[1] : "PM0063_034C1_DM445.0_red.fft"

ft = FFTFile(FILE)
nharms = 60
params = SearchParams(nharms=nharms, threshold=6.0,
                      decimations=decimation_set(nharms, 6))
Nprof = 2048
lodr  = params.hidr / params.nharms
# A mid-band starting fundamental (~10 Hz) so all harmonics stay below Nyquist.
rstart = 10.0 * ft.T

ws     = CS.Workspace(params, Nprof)
# The interpolator needs the per-harmonic phase tables and a *global* trial
# index; this bench treats `rstart` as global trial 0.
dplans = CS.build_direct_plans(params, rstart)

println("File: ", FILE, "   T=", round(ft.T; digits=3), " s")
println("nharms=", nharms, "  Nprof=", Nprof, "  rstart=", round(rstart; digits=1), " bins")
println("="^70)

# --- Bucket 1: whole-chunk fill (all harmonic interps + batched brfft) --------
b_chunk = @benchmark CS.fill_chunk_profiles!($ws, $dplans, $ft, $params, $rstart, $lodr, $Nprof; t0=0)
println("fill_chunk_profiles!  (60 harmonic interps + batched brfft, one chunk):")
show(stdout, MIME"text/plain"(), b_chunk); println("\n", "-"^70)

# Prime ws.profs for the metric benchmarks below.
CS.fill_chunk_profiles!(ws, dplans, ft, params, rstart, lodr, Nprof; t0=0)

# --- Bucket 3: the boxcar metric over a full chunk of profiles ----------------
# Two parts: (a) one per-block robust σ (_block_sigma: two MADs over a strided
# subsample), amortised across the whole block; (b) _profile_boxcar per profile
# (per-profile median + prefix sum + width×phase matched-filter scan).
let profs = ws.profs, nbins = 2nharms, medbuf = ws.medbuf,
    widths = ws.bcwidths, psum = ws.bcpsum, sigbuf = ws.bcsig
    println("boxcar widths (nbins=$nbins): ", widths, "   (", length(widths), " widths)")

    b_sig = @benchmark CS._block_sigma($profs, $nbins, $Nprof, $sigbuf)
    println("_block_sigma  (once per block, nbins=$nbins, Nprof=$Nprof):  ",
            BenchmarkTools.prettytime(minimum(b_sig).time),
            "  => ", round(minimum(b_sig).time/Nprof; digits=2), " ns/profile amortised")

    sigma = CS._block_sigma(profs, nbins, Nprof, sigbuf)
    invsigma = sigma > 0 ? 1.0 / sigma : 0.0
    function boxcar_all(profs, medbuf, psum, widths, nbins, invsigma, n)
        s = 0.0
        @inbounds for j in 1:n
            s += CS._profile_boxcar(profs, j, medbuf, psum, widths, nbins, invsigma)
        end
        s
    end
    b = @benchmark $boxcar_all($profs, $medbuf, $psum, $widths, $nbins, $invsigma, $Nprof)
    println("_profile_boxcar x$Nprof (median + prefix-sum + $(length(widths))-width scan, nbins=$nbins):  ",
            BenchmarkTools.prettytime(minimum(b).time),
            "  => ", round(minimum(b).time/Nprof; digits=1), " ns/call")
end
println("="^70)
