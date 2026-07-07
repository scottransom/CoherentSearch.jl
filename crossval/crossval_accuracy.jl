#!/usr/bin/env julia
#
# Accuracy cross-validation: CoherentSearch.jl vs the Python `coherent_search`
# package, used as an independent oracle.
#
# It runs the Python oracle (oracle_accuracy.py) to dump reference arrays, then
# recomputes the same quantities in Julia and compares them.  This is the main
# guard against the 0-based↔1-based indexing and FFT-convention pitfalls.
#
# Usage:
#   julia --project=crossval crossval/crossval_accuracy.jl [FILE.fft]
#
# Config via environment:
#   COHERENT_PYTHON  python executable that can `import coherent_search`
#   COHERENT_FFT     default .fft file to use

using CoherentSearch
using JSON
using Printf

const DEFAULT_PY = "/home/sransom/python_venvs/pixiPSR/.pixi/envs/default/bin/python"
const DEFAULT_FFT = joinpath(@__DIR__, "..", "..", "coherent_search",
                             "examples", "harmonics_hi.fft")

read_cf64(path) = reinterpret(ComplexF64, read(path))
read_f64(path) = reinterpret(Float64, read(path))

function main()
    py = get(ENV, "COHERENT_PYTHON", DEFAULT_PY)
    fftfile = length(ARGS) >= 1 ? ARGS[1] : get(ENV, "COHERENT_FFT", DEFAULT_FFT)

    isfile(fftfile) || error("FFT file not found: $fftfile")
    isfile(py) || @warn "Python executable not found; relying on PATH" py

    outdir = mktempdir()
    oracle_py = joinpath(@__DIR__, "oracle_accuracy.py")
    @info "Generating Python oracle" py fftfile outdir
    run(`$py $oracle_py $fftfile $outdir`)

    meta = JSON.parsefile(joinpath(outdir, "oracle.json"))
    ft = FFTFile(fftfile)

    failures = 0

    # ---- 1) finterp_FFT kernel ----
    k = meta["kernel"]
    kernel_ref = read_cf64(joinpath(outdir, "kernel_ref.bin"))
    kernel_jl = finterp_fft(Int(k["lobin"]), Int(k["numbins"]),
                            Int(meta["numbetween"]), ft.amps, Int(meta["m"]))
    kerr = maximum(abs.(kernel_jl .- kernel_ref))
    krel = kerr / maximum(abs.(kernel_ref))
    @printf("[kernel]  finterp_FFT  max|Δ| = %.3e   rel = %.3e   (n=%d)\n",
            kerr, krel, length(kernel_ref))
    krel < 1e-9 || (failures += 1; @warn "kernel accuracy out of tolerance")

    # ---- 2) end-to-end coherent-fold PROFILES (indexing + FFT conventions) ----
    # This is the continuous check: the reconstructed profiles must match numpy's
    # bin for bin.  (The detection metric has a discontinuous half-max threshold,
    # so it is validated separately below on identical inputs.)
    params = SearchParams(nharms=Int(meta["nharms"]), m=Int(meta["m"]),
                          numbetween=Int(meta["numbetween"]))
    nbins = Int(meta["nbins"])
    L = Int(meta["e2e"]["L"])
    rfund = collect(read_f64(joinpath(outdir, "rfund.bin")))
    # Python profs are (L, nbins) C-order; reshape to (nbins, L) so column j is
    # profile j (matching reference_profiles' column-per-trial layout).
    profs_ref = reshape(collect(read_f64(joinpath(outdir, "profs_ref.bin"))), nbins, L)
    profs_jl = reference_profiles(ft, rfund, params)
    perr = maximum(abs.(profs_jl .- profs_ref))
    prel = perr / maximum(abs.(profs_ref))
    @printf("[e2e]     profiles     max|Δ| = %.3e   rel = %.3e   (n=%d)\n",
            perr, prel, length(profs_ref))
    prel < 1e-9 || (failures += 1; @warn "end-to-end profile accuracy out of tolerance")

    # ---- 3) detection metric port (snr_metric) on identical profiles ----
    # Feed Julia's snr the *Python* profiles so any difference is purely the
    # metric implementation, not the FFT.  Both width penalties must agree to
    # machine precision.
    ngood = Float64(meta["ngoodbins"]); xsig = Float64(meta["xsignal"]); pex = Float64(meta["pexp"])
    for m in (:non, :sd2)
        metric_ref = read_f64(joinpath(outdir, "metric_$(m)_ref.bin"))
        metric_jl = snr_metrics(profs_ref, ngood; xsignal=xsig, metric=m, pexp=pex)
        merr = maximum(abs.(metric_jl .- metric_ref))
        mrel = merr / maximum(abs.(metric_ref))
        @printf("[metric]  snr(:%-3s)    max|Δ| = %.3e   rel = %.3e   (n=%d)\n",
                m, merr, mrel, length(metric_ref))
        mrel < 1e-9 || (failures += 1; @warn "snr metric port out of tolerance" metric=m)
    end

    println()
    if failures == 0
        println("✓ CROSS-VALIDATION PASSED — Julia matches the Python oracle.")
    else
        println("✗ CROSS-VALIDATION FAILED — $failures check(s) out of tolerance.")
        exit(1)
    end
end

main()
