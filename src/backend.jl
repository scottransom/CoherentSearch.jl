# ---------------------------------------------------------------------------
# Compute-backend selection.
#
# The GPU implementation lives in a package *extension*
# (`ext/CoherentSearchCUDAExt.jl`, weak dependency on CUDA), so a CPU-only user
# never downloads, precompiles or loads CUDA — verified, not assumed: with CUDA
# declared under `[weakdeps]` it does not appear in the user's Manifest at all,
# and `using CoherentSearch` is unchanged.  See `gpu_design.md` §3.4.
#
# **An extension may only add methods on NEW types.**  It must never redefine a
# method this module already owns — that is method overwriting, which Julia
# rejects outright during precompilation.  Hence the shape here: an abstract
# `SearchBackend` plus a `Ref` that the extension *populates* in its `__init__`,
# and every backend-specific entry point dispatches on its first argument.  A
# consequence worth stating: every existing CPU method keeps its current
# definition, so the three correctness pins are untouched by construction rather
# than by testing.
# ---------------------------------------------------------------------------

"""
    SearchBackend

Abstract supertype for compute backends.  [`CPUBackend`](@ref) is always
available; `CUDABackend` is defined by the CUDA extension and registers itself
when `CUDA` is loaded and functional.
"""
abstract type SearchBackend end

"""
    CPUBackend

The threaded CPU implementation — the default, and the one every oracle and
equivalence pin is written against.
"""
struct CPUBackend <: SearchBackend end

# Populated by `CoherentSearchCUDAExt.__init__` when CUDA is functional.  `Any`
# rather than `Union{Nothing,SearchBackend}` because the concrete type does not
# exist until the extension loads; it is read once per search, never in a hot loop.
const _GPU_BACKEND = Ref{Any}(nothing)

"""
    gpu_backend() -> backend or nothing

The registered GPU backend, or `nothing` if no GPU backend is loaded.
"""
gpu_backend() = _GPU_BACKEND[]

"""
    has_gpu() -> Bool

Whether a functional GPU backend is loaded.  `false` until `CUDA` is loaded *and*
`CUDA.functional()` — so it stays `false` on a machine with the package but no
driver, which is the case that should degrade politely rather than crash.
"""
has_gpu() = _GPU_BACKEND[] !== nothing

"""
    require_gpu() -> backend

The registered GPU backend, or a `MethodError`-free explanation of how to get one.
"""
function require_gpu()
    b = _GPU_BACKEND[]
    b === nothing && error("""
        No GPU backend is loaded.  GPU support lives in a package extension, so
        it activates only once CUDA.jl is present and functional:

            julia> using Pkg; Pkg.add("CUDA")
            julia> using CUDA, CoherentSearch

        From the CLI, `--gpu` loads CUDA for you.  If CUDA is already loaded and
        this still fails, `CUDA.functional()` is false — usually a missing or
        mismatched NVIDIA driver; `CUDA.versioninfo()` says which.""")
    return b
end


# ---------------------------------------------------------------------------
# Backend-dispatched entry points.
#
# Declared here with no method for the GPU: the extension adds one, on its own
# `CUDABackend` type.  The `gpu_*` wrappers exist so a caller gets `require_gpu`'s
# explanation rather than a `MethodError` when no GPU backend is loaded.
# ---------------------------------------------------------------------------

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
