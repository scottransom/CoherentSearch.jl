#!/bin/bash
# Run 2 of the detection-efficiency Monte Carlo, on fitzroy.
#
# Design and the run-1 post-mortem it answers: ../docs/monte_carlo.md.
# 15 workers of the 40 logical CPUs, leaving the desktop usable -- fitzroy is
# Scott's workstation, not a batch node.
#
# Estimated ~137 s per realisation per worker, so ~9,400 realisations per day
# and ~56,000 injections per day.  Stop it whenever: every line is a
# self-contained realisation, the output of several runs combines with `cat`,
# and re-running skips indices already present.
set -euo pipefail

PIXI=/data1/environments/pixiPSR/.pixi/envs/default/bin
REPO=/data1/git/CoherentSearch.jl
OUT=${1:-/data1/mc/run2}
TPA=${TPA:-/data1/mc/table_1.csv}
NWORK=${NWORK:-15}

mkdir -p "$OUT"

# **Warm the Julia cache before spawning anything.**  15 workers hitting a cold
# precompile cache at once is 15 simultaneous precompilations of the same
# package, which is slow, noisy, and has raced badly enough elsewhere to leave a
# corrupt .ji behind.  One serial run first costs ~30 s and removes the whole
# class of problem.
julia --project="$REPO" -e 'using CoherentSearch' >/dev/null

cd "$REPO"
exec $PIXI/python mc/mc_simulate.py \
    --outdir "$OUT" --nreal 400000 --workers "$NWORK" \
    --deep-every 5 --deep-coh-every 5 --noise-every 10 \
    --presto-bin $PIXI --rseek $PIXI/rseek --tpa "$TPA"
