#!/usr/bin/env julia
#
# Standalone candidate-profile plotter for CoherentSearch.jl.
#
# Re-creates the pulse-profile plots from a candidate file previously written by
# `coherent_search.jl -o`, without re-running the search.  It re-reads the FFT
# file, recovers each candidate's fundamental Fourier frequency (r = f * T), and
# reconstructs its profile with the same high-accuracy path used inline.
#
# Usage:
#   julia --project=. bin/plot_candidates.jl FILE.fft CANDS.txt [options]

using CoherentSearch
using ArgParse

function parse_cmdline(argv)
    s = ArgParseSettings(
        description = "Plot pulse profiles from a saved CoherentSearch candidate file.",
    )
    @add_arg_table! s begin
        "fftfile"
            help = "PRESTO FFT file that was searched."
            required = true
        "candfile"
            help = "Candidate file written by `coherent_search.jl -o`."
            required = true
        "--plotstem"
            help = "Output path stem for PNGs (default: derived from the candidate filename)"
            arg_type = String
            default = ""
        "--plotcols"
            help = "Grid columns per page"
            arg_type = Int
            default = 3
        "--plotrows"
            help = "Grid rows per page"
            arg_type = Int
            default = 5
        "--nharms"
            help = "Base --nharms of the search: profiles are folded at this depth (up to Nyquist) and it sets the per-panel decimation k = nharms/#Harm.  Default: fold at each candidate's detection harmonic count."
            arg_type = Int
            default = -1
        "--ncands"
            help = "Plot only the first this-many candidates from the file"
            arg_type = Int
            default = typemax(Int)
    end
    return parse_args(argv, s)
end

"""
    read_candidates(path, T) -> Vector{Candidate}

Parse a candidate file (the `# 'S/N' Frequency Period #Harm` table) back into
`Candidate`s, recovering the fundamental Fourier frequency as `r = freq * T`.
Comment (`#`) and blank lines are skipped; columns are whitespace-delimited:
index, S/N, frequency (Hz), period (ms), harmonic count.
"""
function read_candidates(path::AbstractString, T::Real)
    cands = Candidate[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        cols = split(s)
        length(cols) >= 5 || continue
        metric = parse(Float64, cols[2])
        freq = parse(Float64, cols[3])
        nharm = parse(Int, cols[5])
        push!(cands, Candidate(freq, metric, freq * T, nharm))
    end
    return cands
end

function main(argv)
    a = parse_cmdline(argv)
    ft = FFTFile(a["fftfile"])
    cands = read_candidates(a["candfile"], ft.T)
    if length(cands) > a["ncands"]
        cands = cands[1:a["ncands"]]
    end
    if isempty(cands)
        @warn "No candidates found in file" file=a["candfile"]
        return
    end

    # Use the candidate filename verbatim (no splitext: a trailing token like
    # `sd2_0.5` is part of the name, not an extension).
    stem = isempty(a["plotstem"]) ? string(a["candfile"], "_profiles") : a["plotstem"]
    base_nharms = a["nharms"] >= 1 ? a["nharms"] : nothing

    include(joinpath(@__DIR__, "plotting.jl"))
    # `include` defines CandidatePlots in a newer world age than this running
    # function.  Resolve the binding *and* call it inside an `invokelatest`
    # closure so both happen in the latest world (Julia 1.12 world semantics).
    files = Base.invokelatest() do
        CandidatePlots.plot_candidates(ft, cands;
                                       outstem = stem, base_nharms = base_nharms,
                                       ncols = a["plotcols"], nrows = a["plotrows"])
    end
    @info "Wrote candidate profile plots" pages=length(files) stem=stem n=length(cands)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
