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

    # ---- 2) end-to-end coherent-fold metric ----
    params = SearchParams(nharms=Int(meta["nharms"]), m=Int(meta["m"]),
                          numbetween=Int(meta["numbetween"]))
    rfund = collect(read_f64(joinpath(outdir, "rfund.bin")))
    metric_ref = read_f64(joinpath(outdir, "metric_ref.bin"))
    metric_jl = block_metrics(ft, rfund, params)
    merr = maximum(abs.(metric_jl .- metric_ref))
    mrel = merr / maximum(abs.(metric_ref))
    @printf("[e2e]     block_metric max|Δ| = %.3e   rel = %.3e   (n=%d)\n",
            merr, mrel, length(metric_ref))
    mrel < 1e-9 || (failures += 1; @warn "end-to-end accuracy out of tolerance")

    println()
    if failures == 0
        println("✓ CROSS-VALIDATION PASSED — Julia matches the Python oracle.")
    else
        println("✗ CROSS-VALIDATION FAILED — $failures check(s) out of tolerance.")
        exit(1)
    end
end

main()
