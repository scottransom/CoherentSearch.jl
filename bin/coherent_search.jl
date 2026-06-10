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
        In general, the FFT file should probably be barycentered, have known RFI
        zapped, and have rednoise removed.  The threshold is a single-trial
        peak/|trough| profile metric.  If no output filename is given, results
        are written to stdout.  Pass `-t auto` to Julia for multi-threaded runs.
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
            help = "Number of harmonics to sum (a power of two)"
            arg_type = Int
            default = 32
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
        "--blocksize"
            help = "Trial fundamentals per parallel chunk (Nprof)"
            arg_type = Int
            default = 2048
        "--noalign"
            help = "Use a fixed numbetween for all harmonics (disable per-harmonic tuning)"
            action = :store_true
        "--noremove"
            help = "Do not filter duplicate or harmonically-related candidates"
            action = :store_true
    end
    return parse_args(argv, s)
end

function main(argv)
    a = parse_cmdline(argv)

    ft = FFTFile(a["fftfile"])
    params = SearchParams(
        nharms = a["nharms"],
        numbetween = a["numbetween"],
        threshold = a["threshold"],
        align = !a["noalign"],
    )

    @info "Searching" file=a["fftfile"] T=ft.T nharms=params.nharms threads=nthreads()

    cands = search(ft, params;
                   lofreq = a["lofreq"], hifreq = a["hifreq"], lobin = a["lobin"],
                   blocksize = a["blocksize"], threshold = a["threshold"])

    # Keep the strongest `ncands` candidates, but report them in frequency order.
    sort!(cands; by = c -> c.metric, rev = true)
    if length(cands) > a["ncands"]
        cands = cands[1:a["ncands"]]
    end
    sort!(cands; by = c -> c.freq)

    lines = [@sprintf("Candidate at %.6f Hz with S/N %.2f", c.freq, c.metric) for c in cands]
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
