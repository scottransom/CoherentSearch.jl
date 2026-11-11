#!/usr/bin/env julia
#
# One-time FFTW plan-wisdom priming for CoherentSearch.jl.
#
# Builds every FFTW plan a search uses for a given configuration at FFTW.PATIENT
# rigor (thorough, slow — do it once) and saves the wisdom to the per-host cache.
# Later `bin/coherent_search.jl` runs (which plan at MEASURE) reuse these plans
# directly: planning collapses from seconds to milliseconds, with better plans.
#
# Usage:
#   julia --project=. bin/prime_wisdom.jl [--maxdecim K] [--nharms N]
#         [--blocksize B] [--rigor patient|measure] [--wisdomfile PATH]
#
# With neither --nharms nor --maxdecim it primes BOTH configurations the search
# CLI runs by default -- the default search (nharms 60, maxdecim 6) and the
# --deep pass (nharms 120, maxdecim 12) -- as PRESTO's `makewisdom` does.  Give
# either option to prime one configuration instead (the other defaults to the
# default search's value).  Run once per host, and again after changing
# --blocksize; other configs are still learned incrementally by ordinary searches.

using CoherentSearch
using ArgParse
using FFTW

function parse_cmdline(argv)
    s = ArgParseSettings(description =
        "Prime the FFTW plan-wisdom cache (one-time PATIENT planning) for faster search start-up.")
    @add_arg_table! s begin
        "--maxdecim"
            help = "Max harmonic-decimation factor (0 = prime both the default search's 6 and --deep's 12, unless --nharms is given)"
            arg_type = Int
            default = 0
        "--nharms"
            help = "Number of harmonics (0 = prime both the default search's 60 and --deep's 120, unless --maxdecim is given)"
            arg_type = Int
            default = 0
        "--blocksize"
            help = "Trial fundamentals per chunk (must match the search's --blocksize)"
            arg_type = Int
            default = 2048
        "--rigor"
            help = "FFTW planning rigor: 'patient' (thorough) or 'measure'"
            arg_type = String
            default = "patient"
        "--wisdomfile"
            help = "Wisdom cache path (default: per-host file under the Julia depot, or \$COHERENT_WISDOM)"
            arg_type = String
            default = ""
    end
    return parse_args(argv, s)
end

function main(argv)
    a = parse_cmdline(argv)
    nh, md = a["nharms"], a["maxdecim"]
    configs = if nh <= 0 && md <= 0
        [production_params(), production_params(deep = true)]
    else
        md = md >= 1 ? md : 6
        [production_params(nharms = nh >= 1 ? nh : 60, maxdecim = md)]
    end
    rigor = lowercase(a["rigor"]) == "measure" ? FFTW.MEASURE : FFTW.PATIENT
    path = isempty(a["wisdomfile"]) ? wisdom_path() : a["wisdomfile"]

    for params in configs
        @info "Priming FFTW wisdom" nharms=params.nharms decimations=params.decimations blocksize=a["blocksize"] rigor=a["rigor"] path
        t = @elapsed prime_wisdom(params; blocksize = a["blocksize"], rigor = rigor, path = path)
        @info "Wisdom written" path bytes=filesize(path) seconds=round(t; digits=1)
    end
end

main(ARGS)
