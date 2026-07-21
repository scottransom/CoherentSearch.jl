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
    prime_wisdom(params=SearchParams(); blocksize=2048, rigor=FFTW.PATIENT,
                 path=wisdom_path()) -> String

One-time, more-thorough planning pass: build every FFTW plan the search uses for
`params`/`blocksize` at `rigor` (default `FFTW.PATIENT`) and save the wisdom to
`path`, augmenting whatever is already there.  A later ordinary `search` (which
plans at `FFTW.MEASURE`) reuses these plans directly — instant planning *and* the
better `PATIENT` transforms.  Returns the wisdom path.

Wisdom is keyed by transform length, so priming covers a given `nharms`/
`decimations`/`blocksize`; other configurations are still learned incrementally by
`search`'s own import/export.  Run once per host after an environment change.
"""
function prime_wisdom(params::SearchParams = SearchParams();
                      blocksize::Integer = 2048, rigor::Integer = FFTW.PATIENT,
                      path::AbstractString = wisdom_path())
    import_wisdom!(path)                         # augment, don't discard, existing wisdom
    old = _PLAN_RIGOR[]
    _PLAN_RIGOR[] = UInt32(rigor)
    try
        Nprof = max(1, Int(blocksize))
        hplans = build_harmonic_plans(params, Nprof)
        Workspace(params, hplans, Nprof)         # constructs every FFTW plan at `rigor`
    finally
        _PLAN_RIGOR[] = old
    end
    export_wisdom!(path)
    return path
end
