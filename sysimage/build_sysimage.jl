# Build a custom Julia sysimage with CoherentSearch (and CairoMakie) baked in.
#
#   julia --project=sysimage sysimage/build_sysimage.jl
#
# Then run searches against it:
#
#   julia --sysimage sysimage/coherent_search.so --project=. -t auto \
#         bin/coherent_search.jl FILE.fft [FILE2.fft ...] [options]
#
# WHAT IT IS ACTUALLY WORTH (measured 2026-08-11, `-t 1`, 32 MB .fft,
# --hifreq 20 --nharms 32; warm page cache):
#
#   --noplot :  2.3 s with the sysimage vs  2.4 s without  — NOTHING.
#   plotting :  7.4 s with the sysimage vs 18.7 s without  — 2.5x, ~11 s saved.
#
# So the sysimage buys exactly one thing: it removes CairoMakie's ~9 s load and
# first-call compilation.  The in-package PrecompileTools workload (see
# `src/CoherentSearch.jl`) already covers the search itself, and Julia's own boot
# is only ~0.2 s, so there is nothing else left for a sysimage to take.
#
# Costs, all real: ~28 min to build, 1.14 GB on disk, and the FIRST run after a
# build or a reboot pays ~23 s reading the image into page cache (it was 23.2 s
# cold vs 2.3 s warm).  An occasional single search is therefore *slower* with
# the sysimage than without it.
#
# WHEN TO USE IT: repeated, plotting-enabled production runs on a machine that
# stays warm.  If you run with --noplot, skip it entirely.  During development,
# do NOT use it — the sysimage freezes `src/` at build time, so edits are
# silently ignored until you rebuild.  (`Pkg.test()` and a plain
# `julia --project=.` always see the live source.)
#
# The sysimage is CPU- and Julia-version-specific; rebuild after a Julia upgrade
# or when moving to a different machine.

using PackageCompiler

const HERE = @__DIR__
const OUT = joinpath(HERE, "coherent_search.so")

@info "Building sysimage (expect several minutes)" out=OUT

create_sysimage(
    [:CoherentSearch, :CairoMakie];
    sysimage_path = OUT,
    precompile_execution_file = joinpath(HERE, "precompile_exec.jl"),
    # Keep the stock stdlib images so the result still behaves like a normal
    # Julia for everything unrelated to this package.
    incremental = true,
)

@info "Done" out=OUT size_MB=round(filesize(OUT) / 2^20, digits=1)
println("""

Run searches with it:

  julia --sysimage $OUT --project=. -t auto \\
        bin/coherent_search.jl FILE.fft [FILE2.fft ...] [options]

Rebuild it after any change to src/ — a sysimage freezes the code it was built
from, and a stale one will silently run the old search.
""")
