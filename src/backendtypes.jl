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

