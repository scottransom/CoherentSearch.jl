#!/bin/bash
# Re-run ONLY the two accelsearch arms over realisations already in a run.
#
# Why this exists: run 2 scored `accelsearch` and `accelsearch_red` as zero
# candidates on every realisation for its first 1.5 days, because
# `from presto import sifting` failed -- the job was started without the pixi
# environment activated, so `$CONDA_PREFIX/lib64` (where `libpresto.so` lives)
# was on no loader path -- and `parse_accel` swallowed the ImportError (see
# `check_presto_python` in mc_simulate.py).  Everything else in
# those realisations is good, so the cheap repair is to regenerate the noise --
# which depends only on the realisation index and `--master-seed`, so it is
# bit-for-bit the original -- and re-run the ~12 s of accelsearch work rather
# than the ~137 s of the whole realisation.
#
# It writes `mcpatch_accel_<host>_NNN.jsonl` beside the originals and never
# touches them; `mc_analyze.load` merges a patch row into its parent by index,
# overwriting the bad arm.  Re-running is safe: indices already patched are
# skipped.
#
# **Every population argument below must match the run being patched.**  The
# driver checks: it compares the regenerated injection frequencies against the
# ones stored in the record and aborts on any mismatch, because a mismatch means
# the noise differs too.
set -euo pipefail

PIXI=/data1/environments/pixiPSR/.pixi/envs/default/bin
REPO=/data1/git/CoherentSearch.jl
OUT=${1:-/data1/mc/run2}
TPA=${TPA:-/data1/mc/table_1.csv}
NWORK=${NWORK:-15}

export LD_LIBRARY_PATH="$PIXI/../lib64:/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
$PIXI/python -c 'from presto import sifting' || {
    echo "presto.sifting will not import -- that is the bug this repairs" >&2
    exit 1
}

cd "$REPO"
exec $PIXI/python mc/mc_simulate.py \
    --outdir "$OUT" --workers "$NWORK" --arms accel \
    --indices-from "$OUT" \
    --noise-every 10 \
    --presto-bin $PIXI --rseek $PIXI/rseek --tpa "$TPA"
