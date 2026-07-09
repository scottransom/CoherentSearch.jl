# Candidate pulse-profile plotting for CoherentSearch.
#
# Kept out of the core `CoherentSearch` module so that `using CoherentSearch`,
# `Pkg.test()`, and the cross-validation never load a plotting backend.  This
# file is `include`d by the two command-line entry points that actually draw:
# `bin/coherent_search.jl` (plots by default after a search) and the standalone
# `bin/plot_candidates.jl` (re-plots from a saved candidate file).  Loading it
# pulls in CairoMakie, which must be available in the active project.

module CandidatePlots

using CairoMakie
using Printf
using Dates: now
using CoherentSearch: FFTFile, Candidate, SearchParams, candidate_profile, rotate_to_peak

export plot_candidates

# US-Letter portrait at 100 px/inch (8.5 x 11").  `px_per_unit` at save time
# scales this up to the actual PNG resolution (2 -> 1700 x 2200).
const PAGE_SIZE = (850, 1100)

"""
    _page_header(ft, params, threshold, page, npages) -> String

The self-describing banner drawn across the top of each page: which file, its
key metadata, and the search configuration that produced the candidates.
"""
function _page_header(ft::FFTFile, params, threshold, page, npages)
    obj = something(ft.inf.object, "?")
    dm = ft.inf.DM === nothing ? "?" : @sprintf("%.2f", ft.inf.DM)
    cfg = params === nothing ? "" :
          @sprintf("   nharms=%d   metric=%s   pexp=%g", params.nharms, params.metric, params.pexp)
    thr = threshold === nothing ? "" : @sprintf("   threshold=%.1f", threshold)
    return string(
        @sprintf("%s      Candidate pulse profiles", basename(ft.path)),
        "\n",
        @sprintf("object=%s   DM=%s   T=%.1f s   N=%d", obj, dm, ft.T, ft.N),
        cfg, thr,
        @sprintf("\n%s        Page %d of %d", now(), page, npages),
    )
end

"""
    _panel_title(i, c, base_nharms) -> String

The two-line caption over one candidate's profile, carrying every field of the
text-output line (index, S/N, frequency, period, harmonic count) plus the
decimation factor `k = base_nharms ÷ nharm` when the base harmonic count is
known.
"""
function _panel_title(i::Integer, c::Candidate, base_nharms)
    kstr = (base_nharms === nothing || c.nharm == 0) ? "" :
           @sprintf(" (k=%d)", base_nharms ÷ c.nharm)
    return string(
        @sprintf("#%d    S/N %.2f    #Harm %d%s", i, c.metric, c.nharm, kstr),
        "\n",
        @sprintf("f = %.10f Hz    P = %.6f ms", c.freq, 1000.0 / c.freq),
    )
end

"""
    plot_candidates(ft, cands, params=nothing; outstem, kwargs...) -> Vector{String}

Reconstruct and plot the pulse profiles of `cands` (already sorted best-first)
into a grid of `ncols x nrows` panels per US-Letter portrait page, writing one
PNG per page named `<outstem>_NN.png` (zero-padded so pages sort).  Each profile
is rotated so its peak sits at phase 0.5 and captioned with the full text-line
information.  `params`/`threshold` (optional) fill in the page header and the
per-panel decimation factor.  Returns the list of files written.
"""
function plot_candidates(ft::FFTFile, cands::AbstractVector{Candidate}, params=nothing;
                         outstem::AbstractString,
                         ncols::Integer=3, nrows::Integer=5,
                         m::Integer=64,
                         threshold=params === nothing ? nothing : params.threshold,
                         base_nharms=params === nothing ? nothing : params.nharms,
                         px_per_unit::Real=2)
    isempty(cands) && return String[]
    per_page = ncols * nrows
    npages = cld(length(cands), per_page)
    outdir = dirname(outstem)
    isempty(outdir) || isdir(outdir) || mkpath(outdir)

    written = String[]
    for page in 1:npages
        lo = (page - 1) * per_page + 1
        hi = min(page * per_page, length(cands))

        fig = Figure(size=PAGE_SIZE)
        Label(fig[0, 1:ncols], _page_header(ft, params, threshold, page, npages);
              fontsize=11, font=:bold, halign=:left, justification=:left, tellwidth=false)

        # Always lay out the full ncols x nrows grid, even on a partly filled
        # last page: empty cells get a hidden placeholder axis so the populated
        # panels keep exactly the same size and aspect as on the full pages.
        nfilled = hi - lo + 1
        last_row = cld(nfilled, ncols)             # bottom-most row that has panels
        for slot in 1:per_page
            row = (slot - 1) ÷ ncols + 1
            col = (slot - 1) % ncols + 1
            idx = lo + slot - 1

            if idx > hi
                ax = Axis(fig[row, col])           # reserve the cell, draw nothing
                hidedecorations!(ax)
                hidespines!(ax)
                continue
            end

            c = cands[idx]
            # Fold at the full requested harmonic depth (--nharms), not the
            # decimation-reduced count that found the candidate, so the profile
            # approximates a true time-domain fold.  Fall back to the detection
            # count only when the base depth is unknown (a bare re-plot).
            nfold = base_nharms === nothing ? c.nharm : base_nharms
            prof = rotate_to_peak(candidate_profile(ft, c.r, nfold; m=m))
            phase = ((0:length(prof)-1) ./ length(prof))

            ax = Axis(fig[row, col];
                      title=_panel_title(idx, c, base_nharms), titlesize=9,
                      titlefont=:regular, titlegap=2,
                      xlabel=(row == last_row ? "Pulse phase" : ""), xlabelsize=9,
                      xticks=[0.0, 0.5, 1.0], xticklabelsize=8,
                      yticksvisible=false, yticklabelsvisible=false)
            xlims!(ax, 0, 1)
            lines!(ax, phase, prof; color=:navy, linewidth=1.2)
        end

        fname = @sprintf("%s_%02d.png", outstem, page)
        save(fname, fig; px_per_unit=px_per_unit)
        push!(written, fname)
    end
    return written
end

end # module CandidatePlots
