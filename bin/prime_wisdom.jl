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
# Run once per host, and again after changing --nharms/--maxdecim/--blocksize
# (those change the transform lengths); other configs are still learned
# incrementally by ordinary searches.

using CoherentSearch
using ArgParse
using FFTW

function parse_cmdline(argv)
    s = ArgParseSettings(description =
        "Prime the FFTW plan-wisdom cache (one-time PATIENT planning) for faster search start-up.")
    @add_arg_table! s begin
        "--maxdecim"
            help = "Max harmonic-decimation factor (matches the intended search)"
            arg_type = Int
            default = 6
        "--nharms"
            help = "Number of harmonics (0 = auto: 60 when decimating, else 32)"
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
    maxdecim = a["maxdecim"]
    nharms = a["nharms"] >= 1 ? a["nharms"] : (maxdecim > 1 ? 60 : 32)
    params = SearchParams(nharms = nharms, decimations = decimation_set(nharms, maxdecim))
    rigor = lowercase(a["rigor"]) == "measure" ? FFTW.MEASURE : FFTW.PATIENT
    path = isempty(a["wisdomfile"]) ? wisdom_path() : a["wisdomfile"]

    @info "Priming FFTW wisdom" nharms decimations=params.decimations blocksize=a["blocksize"] rigor=a["rigor"] path
    t = @elapsed prime_wisdom(params; blocksize = a["blocksize"], rigor = rigor, path = path)
    @info "Wisdom written" path bytes=filesize(path) seconds=round(t; digits=1)
end

main(ARGS)
