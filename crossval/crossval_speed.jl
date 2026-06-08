#!/usr/bin/env julia
#
# Speed cross-validation: time the same Fourier-interpolation kernel in
# CoherentSearch.jl and in the Python `coherent_search` package, and report the
# speedup.  Also times the full multi-threaded search as a headline number.
#
# Usage:
#   julia --project=crossval -t auto crossval/crossval_speed.jl [FILE.fft]
#
# Config via environment:
#   COHERENT_PYTHON  python executable that can `import coherent_search`
#   COHERENT_FFT     default .fft file to use

using CoherentSearch
using BenchmarkTools
using JSON
using Printf
using Base.Threads: nthreads

const DEFAULT_PY = "/home/sransom/python_venvs/pixiPSR/.pixi/envs/default/bin/python"
const DEFAULT_FFT = joinpath(@__DIR__, "..", "..", "coherent_search",
                             "examples", "harmonics_hi.fft")

function main()
    py = get(ENV, "COHERENT_PYTHON", DEFAULT_PY)
    fftfile = length(ARGS) >= 1 ? ARGS[1] : get(ENV, "COHERENT_FFT", DEFAULT_FFT)
    isfile(fftfile) || error("FFT file not found: $fftfile")

    ft = FFTFile(fftfile)
    m, numbetween = 32, 16
    lobin, numbins = 10000, 1024

    # Match the Python benchmark: reuse a precomputed coefficient kernel.
    fftlen = next_pow_of_2((numbins + m) * numbetween)
    coeffs = finterp_fft_coeffs(numbetween, m, fftlen)

    jl_kernel = @belapsed finterp_fft($lobin, $numbins, $numbetween, $ft.amps, $m;
                                      coeffs=$coeffs)

    # Python side
    @info "Timing Python oracle" py
    bench_py = joinpath(@__DIR__, "bench_python.py")
    py_json = read(`$py $bench_py $fftfile`, String)
    py_best = JSON.parse(py_json)["finterp_FFT"]["best_seconds"]

    println()
    println("── finterp_FFT kernel (lobin=$lobin, numbins=$numbins, numbetween=$numbetween, m=$m) ──")
    @printf("  Python : %10.3f ms\n", py_best * 1e3)
    @printf("  Julia  : %10.3f ms\n", jl_kernel * 1e3)
    @printf("  speedup: %10.1f×  (Julia vs Python)\n", py_best / jl_kernel)

    # Headline: full threaded search over a band around the test pulsar.
    params = SearchParams()
    t0 = time()
    cands = search(ft, params; lofreq=9.0, hifreq=11.0, threshold=8.0)
    dt = time() - t0
    println()
    @printf("── full search 9–11 Hz on %d thread(s): %.3f s, %d candidates ──\n",
            nthreads(), dt, length(cands))
end

main()
