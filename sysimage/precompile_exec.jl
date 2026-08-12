# Execution trace for the PackageCompiler sysimage build (see build_sysimage.jl).
#
# Everything this script touches gets its native code baked into the sysimage.
# So: run a real end-to-end CLI search, over both metric kernels and both
# interpolators, and — unlike the in-package PrecompileTools workload, which
# deliberately avoids it — plot the results, so CairoMakie's first-call
# compilation is captured too.  That plot pass is the single biggest win here:
# `using CairoMakie` alone costs ~9 s from a stock sysimage.

using CoherentSearch

# A small synthetic observation with an injected sinusoid, so the search finds
# something and the plotting path actually runs.
dir = mktempdir()
N = 1 << 15
dt = 1.0e-4
T = N * dt
nfreq = N ÷ 2
amps = ComplexF32[ComplexF32(0.5f0, 0.5f0) for _ in 1:nfreq]
# A bright fundamental plus a few harmonics, near bin 210 (~0.64 Hz).
for h in 1:8
    b = 210 * h + 1
    b <= nfreq && (amps[b] = ComplexF32(60.0f0 / h, 0.0f0))
end

fftpath = joinpath(dir, "synthetic.fft")
write(fftpath, amps)
write(joinpath(dir, "synthetic.inf"),
      " Object being observed                      =  SYNTH\n" *
      " Epoch of observation (MJD)                 =  50000.0\n" *
      " Number of bins in the time series          =  $N\n" *
      " Width of each time series bin (sec)        =  $dt\n" *
      " Dispersion measure (cm-3 pc)               =  0.0\n")

# Frequency window around the injected signal, kept narrow so the build is quick.
band = ["--lofreq", string(200 / T), "--hifreq", string(240 / T)]
common = [fftpath, "--noprogress", "--nharms", "8", "--blocksize", "64",
          "--nowisdom", "--ncands", "2", band...]

# The searching paths: both metrics, both interpolators, and decimation.
for extra in (["--metric", "boxcar"],
              ["--metric", "sd2"],
              ["--interp", "fft", "--fftsizing", "pow2"],
              ["--maxdecim", "2", "--nharms", "12"])
    CoherentSearch.main(vcat(common, extra, ["--noplot", "--threshold", "0.0",
                                             "-o", joinpath(dir, "s.cohout")]))
end

# The multi-file path (two inputs ⇒ per-file .cohout naming + the SearchCache
# reuse branch), then the plotting pass with CairoMakie.
cp(fftpath, joinpath(dir, "synthetic2.fft"))
cp(joinpath(dir, "synthetic.inf"), joinpath(dir, "synthetic2.inf"))
CoherentSearch.main([fftpath, joinpath(dir, "synthetic2.fft"),
                     "--noprogress", "--nharms", "8", "--blocksize", "64",
                     "--nowisdom", "--ncands", "2", "--threshold", "0.0",
                     "--outdir", dir, band...])
