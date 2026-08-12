#!/usr/bin/env julia
#
# Command-line front-end for CoherentSearch.jl.
#
# Usage:
#   julia --project=. -t auto bin/coherent_search.jl FILE.fft [FILE2.fft ...] [options]
#
# Use `-t auto` (or JULIA_NUM_THREADS) to enable multi-threaded searching, and
# pass several .fft files at once to amortise Julia's start-up compilation.
#
# This is deliberately a bare shim: the real driver is `CoherentSearch.main` in
# `src/cli.jl`, so that it lands in the package's precompile cache instead of
# being re-inferred on every run.

using CoherentSearch

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    try
        CoherentSearch.main(ARGS)
    catch err
        # Usage mistakes are the user's, not a bug: report them as a one-line
        # message rather than a Julia stack trace.  Anything else re-throws.
        err isa ArgumentError || rethrow()
        println(stderr, "coherent_search: ", err.msg)
        exit(1)
    end
end
