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
    cands = search(ft, params;
                   lofreq = a["lofreq"], hifreq = a["hifreq"], lobin = a["lobin"],
                   blocksize = a["blocksize"], threshold = a["threshold"],
                   remove = !a["noremove"], dr_tol = a["drtol"],
                   harm_remove = !a["noharmremove"], numharm = a["numharm"],
                   progress = progress)

    # Report the strongest `ncands` candidates, sorted best metric first.
    sort!(cands; by = c -> c.metric, rev = true)
    if length(cands) > a["ncands"]
        cands = cands[1:a["ncands"]]
    end

    # Fixed-width columns; %.12g keeps at least 12 significant figures for the
    # frequency and period at any magnitude (fast pulsars have very short periods,
    # where a fixed number of decimal places would lose precision).
    lines = [@sprintf("Candidate:  f = %-18.12g Hz   P = %-18.12g s   S/N = %8.2f   harmonics = %3d",
                      c.freq, 1.0 / c.freq, c.metric, c.nharm) for c in cands]
    if isempty(a["outputfilenm"])
        foreach(println, lines)
        println(stderr, "# $(length(cands)) candidates above threshold $(a["threshold"])")
    else
        open(a["outputfilenm"], "w") do io
            foreach(l -> println(io, l), lines)
        end
        @info "Wrote candidates" n=length(cands) file=a["outputfilenm"]
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
