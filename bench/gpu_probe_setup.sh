#!/usr/bin/env bash
# Bootstrap a throwaway Julia + CUDA.jl environment on a new GPU host and run
# bench/gpu_probe.jl.  Needs NO root, NO system CUDA toolkit, and nothing from
# this repo except gpu_probe.jl itself -- scp the two files and run this.
#
#   ./gpu_probe_setup.sh                  # installs under ~/.gpuprobe, runs the probe
#   PREFIX=/scratch/$USER/gpuprobe ./gpu_probe_setup.sh
#   JULIA_DEPOT_PATH=/fast/local/depot ./gpu_probe_setup.sh
#
# Only an NVIDIA *driver* is required: CUDA.jl downloads its own toolkit as
# artifacts, so `nvcc` and a module-loaded CUDA are irrelevant (and a
# module-loaded CUDA will NOT be used).
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
PROBE="${PROBE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_probe.jl}"
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
echo "== adding CUDA.jl (first run precompiles for several minutes)"
"$JULIA" --project="$ENVDIR" -e 'using Pkg; Pkg.add("CUDA"); Pkg.precompile()'

# --- run ------------------------------------------------------------------
echo
echo "== running the probe"
"$JULIA" --project="$ENVDIR" "$PROBE"
echo
echo "Re-run later without reinstalling:"
echo "  $JULIA --project=$ENVDIR $PROBE"
