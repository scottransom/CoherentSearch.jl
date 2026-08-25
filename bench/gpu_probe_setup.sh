#!/usr/bin/env bash
# Bootstrap a throwaway Julia + CUDA.jl environment on a new GPU host and run
# bench/gpu_probe.jl.  Needs NO root, NO system CUDA toolkit, and nothing from
# this repo except gpu_probe.jl itself -- scp the two files and run this.
#
#   ./gpu_probe_setup.sh                        # install, then run gpu_probe.jl
#   ./gpu_probe_setup.sh bench/gpu_interp_bench.jl   # ...or any other bench/gpu* script
#   PREFIX=/scratch/$USER/gpuprobe ./gpu_probe_setup.sh
#   JULIA_DEPOT_PATH=/fast/local/depot ./gpu_probe_setup.sh
#
# If it is run from inside a CoherentSearch.jl checkout it ALSO `Pkg.develop`s the
# package into the environment, so the scripts that need it (gpu_interp_bench.jl,
# test/test_gpu.jl) work too.  CUDA is added to that separate environment, never
# to the repo's own Project.toml -- CUDA is a weak dependency there on purpose,
# and adding it as a hard one would undo the whole point (gpu_design.md 3.4).
#
# Only an NVIDIA *driver* is required: CUDA.jl downloads its own toolkit as
# artifacts, so `nvcc` and a module-loaded CUDA are irrelevant (and a
# module-loaded CUDA will NOT be used).
#
# Precompilation is the slow part: CUDA.jl and its stack are ~4-6 min the first
# time (216 s of it measured on fitzroy), and more over NFS.  It is ONE-TIME per
# environment, so:
#   - do NOT delete PREFIX between runs; re-running reuses everything;
#   - if several GPU hosts share a home directory, give them the same PREFIX and
#     JULIA_DEPOT_PATH.  Julia keys its precompile caches by CPU as well as by
#     package version, so different host CPUs store separate copies side by side
#     in the one depot rather than fighting over it -- sharing is safe, it just
#     will not save the second host's compile unless the CPUs match.  Setting
#     JULIA_CPU_TARGET to a common baseline would make them share, at the cost of
#     the CPU-side benchmark arm being compiled for that baseline -- not worth it
#     when the GPU column is what you are after.
#
# Disk: the CUDA artifacts are ~2.2 GB and land in the Julia depot.  On a cluster
# whose $HOME is small or on slow NFS, set JULIA_DEPOT_PATH to local scratch --
# this is the single most common way this goes wrong.
#
# Internet: needed for the install only.  If compute nodes are air-gapped, run
# this once on a login node with the same filesystem, then run the probe itself
# on the GPU node with the same PREFIX and JULIA_DEPOT_PATH.
set -euo pipefail

PREFIX="${PREFIX:-$HOME/.gpuprobe}"
BENCHDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# First positional argument selects the script to run; default the probe.
SCRIPT="${1:-$BENCHDIR/gpu_probe.jl}"
[ -f "$SCRIPT" ] || SCRIPT="$BENCHDIR/$(basename "$SCRIPT")"
PROBE="${PROBE:-$SCRIPT}"
# A checkout looks like <repo>/bench/this-script, with <repo>/Project.toml naming
# the package.
REPO="$(dirname "$BENCHDIR")"
if grep -q '^name = "CoherentSearch"' "$REPO/Project.toml" 2>/dev/null; then
    HAVE_REPO=1
else
    HAVE_REPO=0
    REPO=""
fi
JULIA_VERSION="${JULIA_VERSION:-1.12.7}"

[ -f "$PROBE" ] || { echo "error: probe script not found at $PROBE" >&2; exit 1; }

echo "== host: $(hostname)  prefix: $PREFIX"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
else
    echo "warning: no nvidia-smi on PATH.  If this is a login node, that is expected;" >&2
    echo "         install here, then run the probe on the GPU node." >&2
fi

mkdir -p "$PREFIX"

# --- Julia ----------------------------------------------------------------
# Prefer whatever is already on PATH; otherwise fetch a private tarball.  We do
# NOT use juliaup here: it wants to own a shell profile, which is rude on a
# shared machine and unnecessary for a one-shot probe.
if command -v julia >/dev/null 2>&1; then
    JULIA="$(command -v julia)"
    echo "== using existing julia: $JULIA ($($JULIA --version))"
else
    ARCH="$(uname -m)"
    case "$ARCH" in
        x86_64)  JARCH=x64;   JDIR=x86_64 ;;
        aarch64) JARCH=aarch64; JDIR=aarch64 ;;
        *) echo "unsupported arch $ARCH" >&2; exit 1 ;;
    esac
    MINOR="${JULIA_VERSION%.*}"
    TARBALL="julia-${JULIA_VERSION}-linux-${JARCH}.tar.gz"
    URL="https://julialang-s3.julialang.org/bin/linux/${JDIR}/${MINOR}/${TARBALL}"
    if [ ! -x "$PREFIX/julia-${JULIA_VERSION}/bin/julia" ]; then
        echo "== downloading Julia ${JULIA_VERSION}"
        curl -fL "$URL" -o "$PREFIX/$TARBALL"
        tar -xzf "$PREFIX/$TARBALL" -C "$PREFIX"
        rm -f "$PREFIX/$TARBALL"
    fi
    JULIA="$PREFIX/julia-${JULIA_VERSION}/bin/julia"
    echo "== installed julia: $JULIA"
fi

# --- environment ----------------------------------------------------------
ENVDIR="$PREFIX/env"
mkdir -p "$ENVDIR"
echo "== depot: ${JULIA_DEPOT_PATH:-$HOME/.julia}  (CUDA artifacts are ~2.2 GB)"
if [ "$HAVE_REPO" = 1 ]; then
    echo "== repo detected at $REPO; adding CUDA + CoherentSearch (dev) to $ENVDIR"
    echo "== (ONE-TIME: ~4-6 min of precompilation, more on NFS.  It is not hung.)"
    "$JULIA" --project="$ENVDIR" -e "using Pkg; Pkg.add(\"CUDA\"); Pkg.develop(path=\"$REPO\"); Pkg.precompile()"
else
    echo "== no repo checkout alongside; adding CUDA only (gpu_probe.jl needs nothing else)"
    echo "== (ONE-TIME: ~4-6 min of precompilation, more on NFS.  It is not hung.)"
    "$JULIA" --project="$ENVDIR" -e 'using Pkg; Pkg.add("CUDA"); Pkg.precompile()'
fi

# --- run ------------------------------------------------------------------
echo
echo "== running the probe"
"$JULIA" --project="$ENVDIR" "$PROBE"
echo
echo "Re-run later without reinstalling (any bench/gpu* script, or the GPU tests):"
echo "  $JULIA --project=$ENVDIR $BENCHDIR/gpu_probe.jl"
if [ "$HAVE_REPO" = 1 ]; then
echo "  $JULIA --project=$ENVDIR $BENCHDIR/gpu_interp_bench.jl [FILE.fft]"
echo "  $JULIA --project=$ENVDIR -e 'using CUDA; include(\"$REPO/test/test_gpu.jl\")'"
fi
