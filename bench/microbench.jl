# Micro-benchmarks for the three hot-loop buckets identified in the workload
# model, plus a whole-chunk timing.  Uses the real Workspace/plans so the
# measured cost (including any dynamic dispatch through the FFTScratch Dict)
# is representative of the production path.
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
params = SearchParams(nharms=nharms, threshold=6.0, metric=:boxcar,
                      decimations=decimation_set(nharms, 6))
Nprof = 2048
lodr  = params.hidr / params.nharms
# A mid-band starting fundamental (~10 Hz) so all harmonics stay below Nyquist.
rstart = 10.0 * ft.T

hplans = CS.build_harmonic_plans(params, Nprof)
ws     = CS.Workspace(params, hplans, Nprof)

println("File: ", FILE, "   T=", round(ft.T; digits=3), " s")
println("nharms=", nharms, "  Nprof=", Nprof, "  rstart=", round(rstart; digits=1), " bins")
println("distinct harmonic fftlens: ", sort(unique(hp.fftlen for hp in hplans)))
println("harmonic nb schedule (h=>nb): ", [(hp.h, hp.nb) for hp in hplans[[1,2,4,8,16,32,60]]])
println("="^70)

# --- Bucket 1: whole-chunk fill (all harmonic interps + batched brfft) --------
b_chunk = @benchmark CS.fill_chunk_profiles!($ws, $hplans, $ft, $params, $rstart, $lodr, $Nprof)
println("fill_chunk_profiles!  (60 harmonic interps + batched brfft, one chunk):")
show(stdout, MIME"text/plain"(), b_chunk); println("\n", "-"^70)

# Prime ws.profs for the metric benchmarks below.
CS.fill_chunk_profiles!(ws, hplans, ft, params, rstart, lodr, Nprof)

# --- Bucket 1a: a single harmonic row, low vs high harmonic -------------------
for h in (1, 60)
    hp = hplans[h]
    r0 = hp.h * rstart
    b = @benchmark CS.fill_harmonic_row!($ws, $hp, $ft, $params, $r0, $Nprof)
    println("fill_harmonic_row!  h=$h  (nb=$(hp.nb) fftlen=$(hp.fftlen)):  ",
            BenchmarkTools.prettytime(minimum(b).time),
            "  (", BenchmarkTools.prettymemory(minimum(b).memory), ")")
end
println("-"^70)

# --- Bucket 1b: interp_tile! in isolation, low vs high harmonic ---------------
for h in (1, 60)
    hp = hplans[h]
    sc = ws.scratch[hp.fftlen]
    lobin = floor(Int, hp.h * rstart)
    rlast = hp.h * rstart + (Nprof - 1) * (params.hidr * hp.h / params.nharms)
    numbins = floor(Int, rlast) - lobin + 2
    b = @benchmark CS.interp_tile!($sc, $(hp.coeffs), $lobin, $numbins, $(hp.nb), $(ft.amps), $(params.m))
    println("interp_tile!  h=$h  fftlen=$(hp.fftlen) numbins=$numbins:  ",
            BenchmarkTools.prettytime(minimum(b).time),
            "  (", BenchmarkTools.prettymemory(minimum(b).memory), ")")
end
println("-"^70)

# --- Bucket 2: uniform_linear_interp, the ~1.5e9-call inner interp ------------
let hp = hplans[1]
    sc = ws.scratch[hp.fftlen]
    lobin = floor(Int, hp.h * rstart)
    rlast = hp.h * rstart + (Nprof - 1) * (params.hidr * hp.h / params.nharms)
    numbins = floor(Int, rlast) - lobin + 2
    amps = CS.interp_tile!(sc, hp.coeffs, lobin, numbins, hp.nb, ft.amps, params.m)
    dh = params.hidr * hp.h / params.nharms
    function interp_row(amps, lobin, nb, r0, dh, n)
        s = zero(ComplexF64)
        @inbounds for k in 1:n
            s += CS.uniform_linear_interp(r0 + (k-1)*dh, lobin, nb, amps)
        end
        s
    end
    b = @benchmark $interp_row($amps, $lobin, $(hp.nb), $(hp.h*rstart), $dh, $Nprof)
    println("uniform_linear_interp x$Nprof (one harmonic row):  ",
            BenchmarkTools.prettytime(minimum(b).time),
            "  => ", round(minimum(b).time/Nprof; digits=2), " ns/call")
end
println("-"^70)

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
