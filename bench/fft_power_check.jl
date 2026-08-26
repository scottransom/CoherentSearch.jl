# Is a `.fft` actually normalised to mean power 1, and where is it not?
#
#   julia --project=. bench/fft_power_check.jl FILE.fft [more.fft ...]
#
# `--sigma analytic` (the default) assumes the input has Fourier powers with
# mean 1 -- that is the whole basis of the closed-form noise scale.  When
# `_sigma_sanity_check` warns, this says whether the assumption fails globally or
# only in part of the band, and in which direction.
#
# Read the `mean P` column:  1.0 is correct; BELOW 1 means the spectrum is
# over-suppressed there, so the analytic sigma is too LARGE and reported S/N is
# too low (conservative);  ABOVE 1 means power is leaking through -- residual red
# noise, an RFI comb -- so the analytic sigma is too SMALL and S/N is INFLATED.
#
# The median column is the robust one: `mean P` is pulled up by real signals and
# birdies, and for pure exponential noise median/mean = ln 2 = 0.693, so a
# median/mean ratio far from that is itself a sign of non-noise content.

using CoherentSearch, Printf, Statistics

isempty(ARGS) && error("usage: fft_power_check.jl FILE.fft [more.fft ...]")

for path in ARGS
    ft = FFTFile(path)
    nb = length(ft.amps)
    @printf("\n%s\n  N=%d  T=%.1f s  %d Fourier bins  (bin 1 = %.6f Hz)\n",
            basename(path), ft.N, ft.T, nb, 1 / ft.T)
    println("     bin range          freq range (Hz)      mean P   median P  med/mean   analytic sigma is")
    # Log-spaced bands: the low end is where rednoise does its work and where
    # the sanity check looks, so it needs finer coverage than the top.
    edges = unique(round.(Int, exp10.(range(log10(2), log10(nb), length = 13))))
    for i in 1:(length(edges) - 1)
        lo, hi = edges[i], edges[i + 1] - 1
        hi < lo && continue
        # Power = |amp|^2.  Element 1 packs DC/Nyquist and is skipped by `edges`
        # starting at 2.
        p = [abs2(ft.amps[j]) for j in lo:hi]
        m, md = mean(p), median(p)
        verdict = m < 0.9  ? @sprintf("TOO LARGE by %.1f%% (S/N low)",  100*(1/sqrt(m) - 1)) :
                  m > 1.1  ? @sprintf("TOO SMALL by %.1f%% (S/N HIGH)", 100*(1 - 1/sqrt(m))) : "ok"
        @printf("  %8d-%-8d  %8.4f-%-9.4f  %8.4f  %8.4f  %8.4f   %s\n",
                lo, hi, lo / ft.T, hi / ft.T, m, md, md / m, verdict)
    end
    allp = [abs2(ft.amps[j]) for j in 2:nb]
    @printf("  WHOLE FILE (bin 2+):%29.4f  %8.4f  %8.4f\n",
            mean(allp), median(allp), median(allp) / mean(allp))
end
