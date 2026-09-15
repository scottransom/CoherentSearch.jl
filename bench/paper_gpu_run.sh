#!/usr/bin/env bash
#
# paper_gpu_run.sh -- everything one GPU host needs to contribute to the paper's
# cross-card table, in one command, into one file.
#
#   bench/paper_gpu_run.sh /path/to/NGC6624_16L_DM87.40_red.fft [julia-project]
#
# `julia-project` is the environment that has BOTH CoherentSearch and CUDA (see
# the README's "Installing CUDA.jl").  It defaults to $CS_GPU_PROJECT, then to
# the repo itself.  Set SKIP_CPU=1 to drop step 4 (the slowest step, ~5-15 min).
#
# Output goes to ./paper_<host>_<card>_<date>.txt and to the terminal.  Bring
# that one file back; it carries its own provenance.
#
# WHAT IT RUNS, and why each step is there:
#
#   0. Provenance.  Host, git revision, Julia, driver, card, load average.  A
#      timing without these has bitten this project repeatedly.
#   1. The test suite, then test_gpu.jl on its own in the CUDA project.  An
#      ordinary Pkg.test() NEVER runs the GPU tests, even on a GPU host:
#      test/Project.toml has no CUDA, so test_gpu.jl skips itself.
#   2. *** The CPU-vs-GPU candidate diff. ***  The one step that is not about
#      speed.  The boxcar renormalisation of 2026-08-28 (commit 1b8fed9) changed
#      the per-width table the GPU kernels take from the host, and NO COMMIT HAS
#      TOUCHED ext/ SINCE -- so that path has never run on a device.  If either
#      diff is non-empty, every GPU number and every GPU candidate below is
#      suspect and nothing else in this file should be used.  Read it first.
#      Two bands, because they test different things: a low one where every
#      harmonic is below Nyquist, and a high one chosen from THIS file's length
#      so that the top harmonics run past it -- the `nfilled` path, which is the
#      half of the renormalisation that is new and untested.
#   3. The search report in --paper mode: pinned wide band (0.1-133.3333 Hz),
#      blocksize sweep, per-phase breakdown, and the fixed-cost fit.
#   4. This host's own CPU on the same band, at 1 thread and at all threads.
#
# Steps 1-3 run single-threaded (-t 1): every card in docs/gpu_design.md was
# measured that way, `scan` is host-CPU work, and letting it vary with the
# host's core count makes the device column uncomparable across hosts.

set -u
FFT=${1:?usage: paper_gpu_run.sh FILE.fft [julia-project]}
PROJ=${2:-${CS_GPU_PROJECT:-.}}
SKIP_CPU=${SKIP_CPU:-0}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO" || exit 1
[ -r "$FFT" ] || { echo "cannot read $FFT" >&2; exit 1; }

JL="julia --project=$PROJ"
CARD=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 |
       tr -c 'A-Za-z0-9' '_' | sed 's/__*/_/g; s/^_//; s/_$//')
OUT="$REPO/paper_$(hostname -s)_${CARD:-nogpu}_$(date +%Y%m%d).txt"

# The band above which harmonic `nharms` of the fundamental runs past this
# file's Nyquist frequency, i.e. where the truncated-stack renormalisation
# starts to matter.  Derived from the file rather than hardcoded, so this works
# on any input.
read -r NYQ_LO <<<"$($JL -e '
    using CoherentSearch
    ft = FFTFile(ARGS[1])
    println(round((ft.N / 2) / (60 * ft.T); digits = 4))' "$FFT" 2>/dev/null)"
NYQ_LO=${NYQ_LO:-100.0}

{
echo "############################## 0. provenance ##############################"
date -u +'utc      : %Y-%m-%dT%H:%M:%SZ'
echo    "host     : $(hostname -f)"
echo    "uptime   :$(uptime)"
echo    "cpu      : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')  ($(nproc) threads)"
echo    "repo     : $REPO"
echo    "git      : $(git rev-parse --short HEAD)  $(git log -1 --format=%s)"
echo    "dirty    : $(git status --porcelain | wc -l) modified path(s)"
echo    "julia    : $($JL --version)"
echo    "project  : $PROJ"
echo    "fft      : $FFT  ($(stat -c %s "$FFT" 2>/dev/null) bytes)"
echo    "nyq knee : harmonic 60 crosses Nyquist above $NYQ_LO Hz"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv 2>/dev/null

echo
echo "############################## 1. test suite ##############################"
# Keep the whole log, then print what matters: every failure/error header with
# a few lines of context (tail alone showed only Pkg's stack trace on gina4),
# and the summary table.
$JL -e 'using Pkg; Pkg.test("CoherentSearch")' > "${OUT%.txt}_tests.log" 2>&1
grep -n -A6 -E 'Test Failed at|Error During Test at|Got exception outside of a @test' \
    "${OUT%.txt}_tests.log" | head -120
grep -E -A30 '^Test Summary:' "${OUT%.txt}_tests.log" | grep -vE '^\s*$' | head -40
tail -3 "${OUT%.txt}_tests.log"
echo "  (full log: ${OUT%.txt}_tests.log)"
echo
echo "  -- test_gpu.jl in the CUDA project (Pkg.test cannot run it):"
$JL -e 'using CUDA, Test; include("test/test_gpu.jl")' 2>&1 |
    grep -E -A6 'Test Failed at|Error During Test at|Test Summary:|^gpu backend|skipped' | head -60

echo
echo "################### 2. CPU vs GPU candidates (CORRECTNESS) ################"
echo "# A non-empty diff invalidates every number below.  See the header."
TMP=$(mktemp -d)
NYQ_HI=$(awk -v x="$NYQ_LO" 'BEGIN{printf "%.4f", x*1.25}')
run_pair() {   # $1=label $2=lofreq $3=hifreq
  echo "  -- $1 band: $2 - $3 Hz"
  for ARM in cpu gpu; do
    [ "$ARM" = gpu ] && FLAG=--gpu || FLAG=
    # -t auto on the CPU arm only: the search is chunk-invariant, so the
    # candidate file does not depend on the thread count, and this is the slow
    # side of the pair.
    $JL -t auto bin/coherent_search.jl $FLAG --threshold 6.0 \
        --lofreq "$2" --hifreq "$3" -o "$TMP/$1.$ARM.txt" "$FFT" \
        > "$TMP/$1.$ARM.log" 2>&1 || echo "     ($ARM run FAILED; see log)"
  done
  if [ ! -s "$TMP/$1.cpu.txt" ] || [ ! -s "$TMP/$1.gpu.txt" ]; then
    echo "     RESULT: NOT COMPARED -- one arm produced no candidate file."
    echo "     cpu log tail:"; tail -5 "$TMP/$1.cpu.log" 2>/dev/null | sed 's/^/       /'
    echo "     gpu log tail:"; tail -5 "$TMP/$1.gpu.log" 2>/dev/null | sed 's/^/       /'
    return
  fi
  echo "     cpu $(grep -vc '^#' "$TMP/$1.cpu.txt") cands, gpu $(grep -vc '^#' "$TMP/$1.gpu.txt") cands"
  if diff -q "$TMP/$1.cpu.txt" "$TMP/$1.gpu.txt" >/dev/null 2>&1; then
    echo "     RESULT: BYTE-IDENTICAL"
  else
    echo "     RESULT: *** FILES DIFFER *** -- first 30 lines of diff:"
    diff "$TMP/$1.cpu.txt" "$TMP/$1.gpu.txt" | head -30
    echo "     (S/N differing in the last digit is the known ~2e-7 FP32 tolerance;"
    echo "      a changed frequency, nharm, or candidate COUNT is a real disagreement.)"
  fi
}
run_pair below_nyquist 0.1 5.0
run_pair past_nyquist "$NYQ_LO" "$NYQ_HI"

echo
echo "############################ 3. search report #############################"
$JL -t 1 bench/gpu_search_report.jl "$FFT" --paper 2>&1

echo
echo "########################## 4. this host's own CPU #########################"
if [ "$SKIP_CPU" = 1 ]; then
  echo "# skipped (SKIP_CPU=1)"
else
echo "# Same file and same band as step 3, so the GPU speedup can be quoted"
echo "# against THIS host rather than against fitzroy's Xeon."
$JL -t auto -e '
    using CoherentSearch, Printf
    const CS = CoherentSearch
    ft = FFTFile(ARGS[1])
    par = SearchParams(nharms = 60, m = 16, decimations = collect(1:6))
    lo, hi = 0.1, 133.3333
    ntr = floor(Int, (hi - lo) * ft.T / (par.hidr / par.nharms)) + 1
    go() = search(ft, par; lofreq = lo, hifreq = hi, blocksize = 2048,
                  threshold = 6.0, progress = :none, wisdom = false)
    go()                                  # warm: plans, codegen, page-in
    t = minimum(begin s = time_ns(); nc = length(go()); (time_ns() - s) / 1e9 end for _ in 1:2)
    @printf("  CPU -t %-3d  %8.3f s   %6.1f ns/trial   (%d trials)\n",
            Threads.nthreads(), t, t * 1e9 / ntr, ntr)' "$FFT" 2>&1

if [ "$(nproc)" -gt 1 ]; then
$JL -t 1 -e '
    using CoherentSearch, Printf
    ft = FFTFile(ARGS[1])
    par = SearchParams(nharms = 60, m = 16, decimations = collect(1:6))
    lo, hi = 0.1, 133.3333
    ntr = floor(Int, (hi - lo) * ft.T / (par.hidr / par.nharms)) + 1
    go() = search(ft, par; lofreq = lo, hifreq = hi, blocksize = 2048,
                  threshold = 6.0, progress = :none, wisdom = false)
    go()
    t = minimum(begin s = time_ns(); go(); (time_ns() - s) / 1e9 end for _ in 1:2)
    @printf("  CPU -t 1    %8.3f s   %6.1f ns/trial\n", t, t * 1e9 / ntr)' "$FFT" 2>&1
fi
fi

rm -rf "$TMP"
echo
echo "############################### done ######################################"
} 2>&1 | tee "$OUT"

echo
echo "Wrote $OUT"
