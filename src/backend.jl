# Backend-dispatched entry points.  Split from `backendtypes.jl` because these
# annotate their arguments with `SearchParams`/`FFTFile`, and a method
# signature's types are resolved when the method is DEFINED -- so they must come
# after `search.jl`, while the TYPES must come before it (`search` takes a
# `backend` keyword).
#
# The extension adds methods here on its own `CUDABackend`; it must never
# redefine one this module owns.  See `gpu_design.md` §3.4.

"""
    chunk_ftprofs(backend, ft, params, rstart, n; t0=0, weights=Float32)
        -> (ftprofs, filled)

The interpolated harmonic amplitude stack for one chunk of `n` trial
fundamentals starting at global trial index `t0`, as an `(nharms+1, Nprof)`
matrix — the state of `Workspace.ftprofs` after `fill_chunk_profiles!` and
*before* the inverse transform, plus the per-harmonic `filled` flags that
[`_analytic_sigma`](@ref) reads.

This is the stage-1 GPU equivalence gate (`gpu_design.md` §5): the CUDA backend
must reproduce the CPU backend's result to ~1e-6 at `weights = Float32`.  It is a
comparison and testing entry point, not the production path — it allocates a
whole `Workspace` per call.

Note the CPU path cannot simply read `ws.ftprofs` after `fill_chunk_profiles!`:
FFTW's complex→real transforms destroy their input, so the stack must be captured
before `mul!`.  That is why this reimplements the fill loop rather than calling it.
"""
function chunk_ftprofs end

function chunk_ftprofs(::CPUBackend, ft::FFTFile, params::SearchParams,
                       rstart::Real, n::Integer; t0::Integer = 0,
                       weights::Type{<:AbstractFloat} = Float32)
    ws = Workspace(params, n)
    dplans = build_direct_plans(weights, params, rstart)
    fill!(ws.ftprofs, 0)
    filled = fill(false, params.nharms)
    for dp in dplans
        filled[dp.h] = fill_harmonic_row_direct!(ws, dp, ft, params, t0, n)
    end
    return copy(ws.ftprofs), filled
end

"""
    gpu_chunk_ftprofs(ft, params, rstart, n; kwargs...)

[`chunk_ftprofs`](@ref) on the registered GPU backend.
"""
gpu_chunk_ftprofs(args...; kwargs...) = chunk_ftprofs(require_gpu(), args...; kwargs...)


"""
    chunk_profiles(backend, ft, params, rstart, n; t0=0, weights=Float32, k=1)

The real coherent-fold profiles for one chunk at decimation `k`, as an
`(nbins, n)` matrix with `nbins = 2*fld(nharms, k)`.  Stage-2 equivalence gate.
"""
function chunk_profiles end

"""
    chunk_boxcar(backend, ft, params, rstart, n; t0=0, weights=Float32, k=1, invsigma=1)

Peak boxcar matched-filter S/N of every profile in the chunk at decimation `k`,
with the noise scale supplied rather than estimated.  Passing `invsigma`
explicitly is what makes this a clean test of the *filter* arithmetic: the
statistic is exactly linear in `1/sigma`, so pinning it at `invsigma = 1` covers
the width bank, the wrapped prefix sums and the `delta*S_tot` baseline without
either side having to agree about sigma estimation as well.
"""
function chunk_boxcar end

function chunk_profiles(::CPUBackend, ft::FFTFile, params::SearchParams,
                        rstart::Real, n::Integer; t0::Integer = 0,
                        weights::Type{<:AbstractFloat} = Float32, k::Integer = 1)
    ws = Workspace(params, n)
    dplans = build_direct_plans(weights, params, rstart)
    fill_chunk_profiles!(ws, dplans, ft, params, rstart,
                         params.hidr / params.nharms, n; t0 = t0)
    k == 1 && return copy(ws.profs[:, 1:n])
    db = ws.decims[findfirst(d -> d.k == k, ws.decims)]
    mul!(db.dprofs, db.brfftplan, db.src)
    return copy(db.dprofs[:, 1:n])
end

function chunk_boxcar(::CPUBackend, ft::FFTFile, params::SearchParams,
                      rstart::Real, n::Integer; t0::Integer = 0,
                      weights::Type{<:AbstractFloat} = Float32, k::Integer = 1,
                      invsigma::Real = 1.0)
    profs = chunk_profiles(CPUBackend(), ft, params, rstart, n;
                           t0 = t0, weights = weights, k = k)
    nbins = size(profs, 1)
    widths = ladder_boxcar_widths(nbins, k, params)
    psum = Vector{Float64}(undef, nbins + widths[end] + 1)
    P64 = Matrix{Float64}(profs)
    return [_profile_boxcar(P64, j, psum, widths, nbins, Float64(invsigma))
            for j in 1:n]
end

gpu_chunk_profiles(args...; kwargs...) = chunk_profiles(require_gpu(), args...; kwargs...)
gpu_chunk_boxcar(args...; kwargs...) = chunk_boxcar(require_gpu(), args...; kwargs...)


"""
    _region!(backend, ft, params, workspaces, nbins, r_lo, r_hi, lodr,
             total, Nprof, nchunks, nt, cstarts; kwargs...) -> Vector{Candidate}

Run the chunk loop over `[r_lo, r_hi]` on `backend` and return the
above-`threshold` candidates.  The CPU method is the threaded `_search_region!`
that every pin is written against; the CUDA extension adds a method on its own
backend type.  Everything outside this call -- the trial grid, the plans, the
analytic-sigma sanity check, duplicate and harmonic collapsing, and all output --
is backend-independent and runs unchanged.
"""
function _region! end

_region!(::CPUBackend, args...; kwargs...) = _search_region!(args...; kwargs...)
