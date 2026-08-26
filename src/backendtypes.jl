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

"""
    release_backend!(backend)

Release any resources a backend holds between searches, and return device memory
to the driver.  A no-op on the CPU.

**Why this exists.** The GPU path caches its chunk workspace and interpolation
plan across files — they are a pure function of `(params, Nprof, r_lo)` and not
of the file's contents, exactly as the CPU's `SearchCache` is — so a multi-file
run does not rebuild them per file.  Something then has to hand the memory back
at the end, and that is this.

Call it after a batch of searches, not between them; `CoherentSearch.main` does.
Calling it more often than necessary is safe but costs ~0.36 s each time
(measured on a GTX 1080), which is the whole reason the caching exists.
"""
release_backend!(::SearchBackend) = nothing

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



# ---------------------------------------------------------------------------
# GPU phase timers.
#
# The CPU's `phase_times` costs one `time_ns` pair per phase per chunk (~0.03% of
# runtime) and is always on.  The GPU equivalent cannot be: a wall-clock timer
# around a CUDA launch measures the launch, not the work, so timing a phase means
# `CUDA.synchronize()` around it -- which SERIALISES the queue and slightly
# changes what it is measuring.  So this is opt-in, and a run with it on is
# a diagnostic, not a benchmark: read the SHARES, and take the total from a run
# with timing off.
# ---------------------------------------------------------------------------

"""GPU pipeline phases, in the order `_region!` runs them."""
const GPU_PHASE_NAMES = ("zero", "interp", "transpose", "transform",
                         "boxcar", "download", "scan")

# ---------------------------------------------------------------------------
# NOT every phase above is the GPU's.  `scan` is the host loop over the
# downloaded metrics in `_region!` -- pure CPU work -- and `download` is the
# PCIe transfer.  Only the first five are device kernels.
#
# This distinction is not cosmetic.  Measured 2026-08-25 on the same file and
# the same harness, `scan` ran 5.20 ns/trial on `spare2` against 2.97 on
# `hypatia` -- **1.75x**, against those hosts' own CPU search arms at 1.71x.  So
# it tracks the host CPU and says nothing whatever about the card, yet it was
# 12.3% of one card's phase table and 7.3% of the other's.  Anyone classifying a
# GPU from an unlabelled table is reading a property of the machine it was
# plugged into.  See `gpu_design.md` §4.8, finding (a).
# ---------------------------------------------------------------------------

"""
Where each [`GPU_PHASE_NAMES`](@ref) phase actually runs: `:device` (a CUDA
kernel), `:transfer` (PCIe), or `:host` (CPU work in `_region!`).  Only
`:device` phases are a property of the card.
"""
const GPU_PHASE_KIND = (:device, :device, :device, :device,
                        :device, :transfer, :host)

const _GPU_PHASE_NS = zeros(Int64, length(GPU_PHASE_NAMES))
const _GPU_TIMING = Ref(false)

# ---------------------------------------------------------------------------
# Transform sub-batching policy, read by the CUDA extension when it builds a
# `GPUChunk`.  `:auto` derives the target from the device's L2 (the shipped
# behaviour, and a no-op on a small-L2 card); `:off` forces one un-split batch
# per rung; an `Integer` is an explicit L2 target in bytes, which is what
# `bench/gpu_subbatch_bench.jl` sweeps and what lets a small-L2 card exercise
# the split in a test.  A `Ref` rather than a `SearchParams` field: it is a
# device tuning knob with no meaning on the CPU path.
# ---------------------------------------------------------------------------
const _GPU_SUBBATCH = Ref{Any}(:auto)

"""
    gpu_subbatch!(x)

Set the GPU transform sub-batching policy: `:auto` (default, derive from device
L2), `:off`, or an `Integer` L2 target in bytes.  See `gpu_design.md` §4.9.
"""
function gpu_subbatch!(x)
    x === :auto || x === :off || x isa Integer ||
        throw(ArgumentError("gpu_subbatch! expects :auto, :off or an Integer, got $x"))
    _GPU_SUBBATCH[] = x
end
gpu_subbatch() = _GPU_SUBBATCH[]

"""
    gpu_timing!(on::Bool)

Turn GPU phase timing on or off.  Off by default; see the note above on why it
is not free.
"""
gpu_timing!(on::Bool) = (_GPU_TIMING[] = on)
gpu_timing() = _GPU_TIMING[]

"""    gpu_phase_reset!()  — zero the GPU phase accumulators."""
gpu_phase_reset!() = (fill!(_GPU_PHASE_NS, 0); nothing)

"""
    gpu_phase_times() -> Vector{Pair{String,Float64}}

Accumulated seconds per GPU phase since the last [`gpu_phase_reset!`](@ref).
Empty numbers unless [`gpu_timing!`](@ref)`(true)` was set.
"""
gpu_phase_times() = [GPU_PHASE_NAMES[i] => _GPU_PHASE_NS[i] / 1e9
                     for i in eachindex(GPU_PHASE_NAMES)]
