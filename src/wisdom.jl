# ---------------------------------------------------------------------------
# FFTW plan-wisdom persistence
#
# A short search spends a disproportionate share of wall-time *before* the hot
# loop, planning every distinct-length FFTW transform with `FFTW.MEASURE` — and
# `MEASURE` re-times the transforms on *every* process start (one 120-point
# `brfft` alone is ~140 ms cold vs ~0.1 ms once wisdom exists).  FFTW's plan cache
# is serialisable, so we import any saved wisdom before building plans and export
# it afterwards; the second run onward collapses planning to a wisdom lookup.
#
# Wisdom is CPU- and FFTW-version-specific, so the cache is keyed per host and an
# incompatible file simply fails to import (we then re-measure and overwrite).
# `prime_wisdom` optionally does a one-time `FFTW.PATIENT` planning pass, whose
# (better) plans a later `MEASURE` run reuses directly.
# ---------------------------------------------------------------------------

"""
    wisdom_path() -> String

Location of the FFTW wisdom cache: `\$COHERENT_WISDOM` if set, else a per-host file
under the Julia depot (`<depot>/coherent_search/fftw_wisdom_<host>.dat`).
"""
wisdom_path()::String = get(ENV, "COHERENT_WISDOM") do
    joinpath(first(DEPOT_PATH), "coherent_search", "fftw_wisdom_$(gethostname()).dat")
end

"""
    import_wisdom!(path = wisdom_path()) -> Bool

Load saved FFTW wisdom so subsequent `MEASURE`/`PATIENT` planning is a lookup.
Returns `false` (and leaves planning to re-measure) if the file is absent or
incompatible; never throws.
"""
function import_wisdom!(path::AbstractString = wisdom_path())
    isfile(path) || return false
    try
        FFTW.import_wisdom(path)
        return true
    catch err
        @warn "FFTW wisdom import failed; plans will be re-measured" path exception=err
        return false
    end
end

"""
    export_wisdom!(path = wisdom_path()) -> Bool

Persist the accumulated FFTW wisdom (atomically, via a temp file + rename, so
concurrent searches can't corrupt it).  Returns `false` on I/O error; never throws.
"""
function export_wisdom!(path::AbstractString = wisdom_path())
    try
        mkpath(dirname(path))
        tmp = string(path, '.', getpid(), ".tmp")
        FFTW.export_wisdom(tmp)
        mv(tmp, path; force=true)
        return true
    catch err
        @warn "FFTW wisdom export failed" path exception=err
        return false
    end
end

"""
    prime_wisdom(params=production_params(); blocksize=2048, rigor=FFTW.PATIENT,
                 precisions=(:f64, :f32), path=wisdom_path()) -> String

One-time, more-thorough planning pass: build every FFTW plan the search uses for
`params`/`blocksize` at `rigor` (default `FFTW.PATIENT`) and save the wisdom to
`path`, augmenting whatever is already there.  Returns the wisdom path.

Since 2026-08-22 `search` already plans at `PATIENT` whenever `wisdom=true`, so
this is no longer needed to *get* patient plans — it exists to pay their ~1 s
planning cost ahead of time, and to cover configurations (notably the other
`precision`) that the run you are about to do will not itself build.

**`params` defaults to the CLI's production configuration, not `SearchParams()`.**
That distinction was a live bug until 2026-08-22: `SearchParams()` is
`nharms = 32, decimations = [1]`, a *single* 64-bin transform, while the CLI
defaults to `nharms = 60, --maxdecim 6`, i.e. six transforms of
120/60/40/30/24/20 bins — five of them on strided views, which are exactly the
ones `PATIENT` helps.  So `prime_wisdom()` as previously documented ("run once per
host") primed one transform the search never executes and none of the six it does.

Wisdom is keyed by transform length, stride and precision, so priming covers a
given `nharms`/`decimations`/`blocksize`/`precision`; other configurations are
still learned incrementally by `search`'s own import/export.  Run once per host
after an environment change.
"""
function prime_wisdom(params::SearchParams = production_params();
                      blocksize::Integer = 2048, rigor::Integer = FFTW.PATIENT,
                      precisions = (:f64, :f32),
                      path::AbstractString = wisdom_path())
    import_wisdom!(path)                         # augment, don't discard, existing wisdom
    Nprof = max(1, Int(blocksize))
    with_plan_rigor(rigor) do
        for prec in precisions
            # `Workspace` constructs every FFTW plan the search uses: the base
            # `brfft` and one strided `brfft` per decimation factor.
            Workspace(_with_precision(params, prec), Nprof)
        end
    end
    export_wisdom!(path)
    return path
end

"""
    production_params(; nharms=60, maxdecim=6) -> SearchParams

The configuration `bin/coherent_search.jl` searches with by default, which is
*not* `SearchParams()`.  Used as `prime_wisdom`'s default so that priming covers
the plans a default run actually builds.
"""
production_params(; nharms::Integer = 60, maxdecim::Integer = 6) =
    SearchParams(nharms = nharms, decimations = decimation_set(nharms, maxdecim))

# `Base.@kwdef` gives no copy-with-override constructor.  Rebuild by reflection so
# this keeps working when `SearchParams` gains a field.
_with_precision(p::SearchParams, prec::Symbol) =
    SearchParams(; (f => (f === :precision ? prec : getfield(p, f))
                    for f in fieldnames(SearchParams))...)
