#!/usr/bin/env julia
#
# Command-line front-end for CoherentSearch.jl.
# Mirrors the Python `coherent_search` console script.
#
# Usage:
#   julia --project=. -t auto bin/coherent_search.jl FILE.fft [options]
#
# Use `-t auto` (or JULIA_NUM_THREADS) to enable multi-threaded searching.

using CoherentSearch
using ArgParse
using Printf
using Base.Threads: nthreads

function parse_cmdline(argv)
    s = ArgParseSettings(
        description = "Search a PRESTO-style FFT file for pulsations using coherent harmonic folding.",
        epilog = """
        The input FFT file MUST be normalized (Fourier powers with mean ~ 1); the
        width-sensitive S/N metric assumes unit-variance noise, so an un-normalized
        FFT produces meaningless (hugely inflated) S/N values.  Normalize an
        un-normalized FFT with PRESTO's `rednoise` routine, which also removes red
        noise.  The FFT should also be barycentered and have known RFI zapped.  The
        detection metric sums the on-pulse flux and divides by a width penalty
        (--metric): 'non' = N_on^p (duty cycle; p=1/2 is a calibrated equivalent-σ,
        larger p suppresses broad/RFI-like signals) or 'sd2' = Σd²^p (phase spread).
        Near-identical candidates are collapsed by default (--noremove disables it),
        as are harmonically-related ones -- the f/2, 2f, 3f/2, ... family of a real
        signal (--noharmremove disables it, --numharm sets the max harmonic).
        With --maxdecim>1, harmonic decimation also folds each fundamental at 2..k
        times its frequency almost for free (re-using the interpolated harmonics),
        extending the search to faster pulsars with fewer summed harmonics; the
        reported harmonic count identifies which decimation found each candidate.
        A progress meter prints to stderr (--progressbar for a bar, --noprogress off).
        If no output filename is given, results are written to stdout.  Pass
        `-t auto` to Julia for multi-threaded runs.
        """,
    )
    @add_arg_table! s begin
        "fftfile"
            help = "PRESTO FFT file to be searched."
            required = true
        "--threshold", "-t"
            help = "S/N cutoff for picking candidates"
            arg_type = Float64
            default = 8.0
        "--outputfilenm", "-o"
            help = "Output filename to record candidates (default: stdout)"
            arg_type = String
            default = ""
        "--nharms", "-n"
            help = "Number of harmonics to sum (default: 32, or 60 when --maxdecim>1)"
            arg_type = Int
            default = -1
        "--ncands"
            help = "Maximum number of candidates to return"
            arg_type = Int
            default = 100
        "--lobin"
            help = "Lowest frequency bin to search"
            arg_type = Int
            default = 100
        "--lofreq"
            help = "Lowest frequency (in Hz) to search"
            arg_type = Float64
            default = 0.1
        "--hifreq"
            help = "Highest frequency (in Hz) to search"
            arg_type = Float64
            default = 100.0
        "--hidr"
            help = "Fourier bin resolution at highest harmonic"
            arg_type = Float64
            default = 0.5
        "--numbetween"
            help = "Minimum points to interpolate between Fourier bins"
            arg_type = Int
            default = 16
        "--xsignal"
            help = "Peak fraction bounding the on-pulse signal sum in the S/N metric"
            arg_type = Float64
            default = 0.2
        "--metric"
            help = "Width penalty: 'non' (N_on^p, duty cycle) or 'sd2' (Σd²^p, phase spread)"
            arg_type = String
            default = "non"
            range_tester = x -> x in ("non", "sd2")
        "--pexp"
            help = "Width-penalty exponent p (1/2 = calibrated matched filter for 'non')"
            arg_type = Float64
            default = 0.5
        "--blocksize"
            help = "Trial fundamentals per parallel chunk (Nprof)"
            arg_type = Int
            default = 2048
        "--maxdecim"
            help = "Max harmonic-decimation factor k: also search 2..k times each fundamental (default: 1 = off)"
            arg_type = Int
            default = 1
        "--drtol"
            help = "Fourier-bin tolerance for collapsing near-identical candidates"
            arg_type = Float64
            default = 1.0
        "--numharm"
            help = "Max harmonic number when removing harmonically-related candidates"
            arg_type = Int
            default = 16
        "--noalign"
            help = "Use a fixed numbetween for all harmonics (disable per-harmonic tuning)"
            action = :store_true
        "--noremove"
            help = "Do not collapse near-identical (duplicate) candidates"
            action = :store_true
        "--noharmremove"
            help = "Do not collapse harmonically-related candidates (f/2, 2f, 3f/2, ...)"
            action = :store_true
        "--progressbar"
            help = "Show a progress bar instead of the default text percentage meter"
            action = :store_true
        "--noprogress"
            help = "Do not print a progress meter"
            action = :store_true
        "--metricstats"
            help = "Report per-block, per-decimation metric stats (min/median/mean/std/max) to help set --threshold; writes a full per-block table to <stem>_metricstats.txt"
            action = :store_true
        "--noplot"
            help = "Do not plot the candidate pulse profiles (plotting is on by default)"
            action = :store_true
        "--plotstem"
            help = "Output path stem for profile-plot PNGs (default: derived from -o or the FFT name)"
            arg_type = String
            default = ""
        "--plotcols"
            help = "Profile-plot grid columns per page"
            arg_type = Int
            default = 3
        "--plotrows"
            help = "Profile-plot grid rows per page"
            arg_type = Int
            default = 5
    end
    return parse_args(argv, s)
end

function main(argv)
    a = parse_cmdline(argv)

    ft = FFTFile(a["fftfile"])
    maxdecim = a["maxdecim"]
    # Resolve nharms: explicit if given, else 60 when decimating (composite) or 32.
    nharms = a["nharms"] >= 1 ? a["nharms"] : (maxdecim > 1 ? 60 : 32)
    decimations = decimation_set(nharms, maxdecim)
    params = SearchParams(
        nharms = nharms,
        numbetween = a["numbetween"],
        threshold = a["threshold"],
        align = !a["noalign"],
        xsignal = a["xsignal"],
        metric = Symbol(a["metric"]),
        pexp = a["pexp"],
        decimations = decimations,
    )

    @info "Searching" file=a["fftfile"] T=ft.T nharms=params.nharms decimations=decimations threads=nthreads()

    progress = a["noprogress"] ? :none : (a["progressbar"] ? :bar : :text)
    mstats = a["metricstats"] ? BlockMetricStats[] : nothing
    cands = search(ft, params;
                   lofreq = a["lofreq"], hifreq = a["hifreq"], lobin = a["lobin"],
                   blocksize = a["blocksize"], threshold = a["threshold"],
                   remove = !a["noremove"], dr_tol = a["drtol"],
                   harm_remove = !a["noharmremove"], numharm = a["numharm"],
                   progress = progress, metricstats = mstats)

    if mstats !== nothing
        base = isempty(a["outputfilenm"]) ?
               first(splitext(basename(a["fftfile"]))) : a["outputfilenm"]
        report_metricstats(mstats, string(base, "_metricstats.txt"), params, a["threshold"])
    end

    # Report the strongest `ncands` candidates, sorted best metric first.
    sort!(cands; by = c -> c.metric, rev = true)
    if length(cands) > a["ncands"]
        cands = cands[1:a["ncands"]]
    end

    # Fixed-width columns; %.12g keeps at least 12 significant figures for the
    # frequency and period at any magnitude (fast pulsars have very short periods,
    # where a fixed number of decimal places would lose precision).
    header = ["#       'S/N'      Frequency (Hz)        Period (ms)    #Harm"]
    lines = [@sprintf("%-4d  %8.2f  %18.12f  %18.12f   %3d",
                      i, c.metric, c.freq, 1000.0 / c.freq, c.nharm) for (i, c) in enumerate(cands)]
    outlines = vcat(header, lines)
    if isempty(a["outputfilenm"])
        foreach(println, outlines)
        println(stderr, "# $(length(cands)) candidates above threshold $(a["threshold"])")
    else
        open(a["outputfilenm"], "w") do io
            foreach(l -> println(io, l), outlines)
        end
        @info "Wrote candidates" n=length(cands) file=a["outputfilenm"]
    end

    # Plot the candidate pulse profiles (on by default; --noplot disables).  The
    # plotting backend (CairoMakie) is loaded lazily here so ordinary searches,
    # tests, and the cross-validation never pay for it.
    if !a["noplot"] && !isempty(cands)
        stem = plot_stem(a["plotstem"], a["outputfilenm"], a["fftfile"])
        include(joinpath(@__DIR__, "plotting.jl"))
        # `include` defines CandidatePlots in a newer world age than this running
        # function.  Resolve the binding *and* call it inside an `invokelatest`
        # closure so both happen in the latest world (Julia 1.12 world semantics).
        files = Base.invokelatest() do
            CandidatePlots.plot_candidates(ft, cands, params;
                                           outstem = stem,
                                           ncols = a["plotcols"], nrows = a["plotrows"])
        end
        @info "Wrote candidate profile plots" pages=length(files) stem=stem
    end
end

"""
    plot_stem(plotstem, outputfilenm, fftfile) -> String

Resolve the PNG output stem: an explicit `--plotstem` wins; otherwise derive it
from the candidate output filename, or (for stdout runs) from the FFT filename.
"""
function plot_stem(plotstem, outputfilenm, fftfile)
    isempty(plotstem) || return plotstem
    # Use the candidate filename verbatim; do NOT run splitext on it, since a
    # trailing token like `sd2_0.5` looks like an extension but is part of the
    # name.  Only the fallback fftfile name has a real extension (.fft) to strip.
    isempty(outputfilenm) || return string(outputfilenm, "_profiles")
    return string(first(splitext(basename(fftfile))), "_profiles")
end

"""
    report_metricstats(stats, filename, params, threshold)

Print a per-decimation summary of the metric distribution to `stderr` (to help
choose `--threshold`) and write the full per-block table to `filename`.  The
summary makes the key point visible: with `--metric non --pexp 0.5` the metric's
noise floor grows ~`√nbins = √(2·Hk)`, so each decimation `k` sits at a
different mean/max and a single `--threshold` is *not* comparable across them.
"""
function report_metricstats(stats::Vector{BlockMetricStats}, filename::String,
                            params, threshold)
    if isempty(stats)
        @warn "No metric statistics collected (no blocks searched)"
        return
    end
    summ = metricstats_summary(stats)

    # --- per-decimation summary to stderr ---
    println(stderr)
    println(stderr, "Metric statistics by decimation  (metric=$(params.metric), pexp=$(params.pexp), threshold=$(threshold))")
    println(stderr, "  Pure-noise floor scales ~sqrt(nbins); a fixed threshold is NOT comparable across k.")
    @printf(stderr, "  %-3s %5s %6s %8s %11s %8s %8s %8s %8s %10s %8s\n",
            "k", "Hk", "nbins", "nblocks", "ntrials",
            "min", "median", "mean", "std", "blkmax", "max")
    for r in summ
        @printf(stderr, "  %-3d %5d %6d %8d %11d %8.3f %8.3f %8.3f %8.3f %10.3f %8.3f\n",
                r.k, r.Hk, r.nbins, r.nblocks, r.ntrials,
                r.min, r.median, r.mean, r.std, r.blockmax_mean, r.max)
    end
    println(stderr, "  (blkmax = mean of per-block maxima; max = global max — the worst noise excursion seen)")
    println(stderr)

    # --- full per-block table to file ---
    open(filename, "w") do io
        println(io, "# Per-block, per-decimation metric statistics")
        println(io, "# metric=$(params.metric) pexp=$(params.pexp) xsignal=$(params.xsignal) nharms=$(params.nharms) threshold=$(threshold)")
        println(io, "# nbins = 2*Hk; the pure-noise metric floor scales ~sqrt(nbins).")
        @printf(io, "#%-7s %3s %4s %5s %10s %14s %14s %8s %9s %9s %9s %9s %9s\n",
                "block", "k", "Hk", "nbins", "ngoodbins", "f_lo(Hz)", "f_hi(Hz)",
                "n", "min", "median", "mean", "std", "max")
        for s in stats
            @printf(io, "%-8d %3d %4d %5d %10.3f %14.8f %14.8f %8d %9.3f %9.3f %9.3f %9.3f %9.3f\n",
                    s.block, s.k, s.Hk, s.nbins, s.ngoodbins, s.flo, s.fhi,
                    s.n, s.min, s.median, s.mean, s.std, s.max)
        end
    end
    @info "Wrote per-block metric statistics" file=filename nrows=length(stats)
    return
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
