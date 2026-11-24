#!/bin/bash
# run_replay.sh: prices the exported items on this machine: canary, the replay programs, the canary again, the fit of the unit prices.
# Usage: run_replay.sh [--dry-run] --out DIR EXPORT_DIR [EXPORT_DIR ...]
#   EXPORT_DIR   the export directories of one or more runs (<run>/export, with index.tsv)
#   --out DIR    the output directory: replay_l4a/l4b/l4c/l4d/l4e.tsv, canary_before.json, canary_after.json, unit_costs_cl.json (the unit prices that estimate.py reads)
# Every program runs single-threaded; the canary runs before and after (the speed of the machine in between: drift above 10% means the prices are to be repeated).
# Environment:
#   CL_WRAP   a command prefix for every program (e.g. "flock LOCK experiments/cudss/capped.sh 7G --"), default empty
#   CL_PIN    a CPU pin prefix (Linux: "taskset -c 3"), default empty (macOS has no taskset: close other large jobs instead)
#   CL_RUN_L4B=0 skips the CCD replay (C1-C3 are then not priced)
set -u
DRY=0; OUT=""
while [ $# -gt 0 ]; do
  case "$1" in --dry-run) DRY=1; shift ;; --out) OUT="$2"; shift 2 ;; *) break ;; esac
done
[ -n "$OUT" ] && [ $# -ge 1 ] || { echo "usage: $0 [--dry-run] --out DIR EXPORT_DIR..."; exit 2; }
HERE="$(cd "$(dirname "$0")" && pwd)"
WRAP="${CL_WRAP:-}"; PIN="${CL_PIN:-}"
EXPS=(); for d in "$@"; do EXPS+=(--export "$d"); done
ENVV=(MKL_CBWR=COMPATIBLE MKL_NUM_THREADS=1 OMP_NUM_THREADS=1)
run() { # name, then the command
  local name="$1"; shift
  if [ $DRY = 1 ]; then echo "[$name] $WRAP $PIN env ${ENVV[*]} $*"; return 0; fi
  echo "$(date '+%F %T') start $name"
  # shellcheck disable=SC2086
  $WRAP $PIN env "${ENVV[@]}" "$@"
  local rc=$?
  echo "$(date '+%F %T') done $name rc=$rc"
  return $rc
}
[ $DRY = 1 ] || mkdir -p "$OUT"
run canary_before "$HERE/canary" --out "$OUT/canary_before.json" || exit 1
run l4a "$HERE/cl_replay_solver" l4a "${EXPS[@]}" --out "$OUT/replay_l4a.tsv" --reps 7 || exit 1
run l4c "$HERE/cl_replay_solver" l4c "${EXPS[@]}" --out "$OUT/replay_l4c.tsv" --reps 21 || exit 1
run l4e "$HERE/cl_replay_solver" l4e --out "$OUT/replay_l4e.tsv" --reps 21 || exit 1
if [ "${CL_RUN_L4B:-1}" = 1 ]; then
  rm -f "$OUT/replay_l4b.tsv"
  run l4b "$HERE/cl_replay_ipc" l4b "${EXPS[@]}" --out "$OUT/replay_l4b.tsv" --reps 15 || echo "$(date '+%F %T') l4b failed (soft: C1-C3 unpriced)"
else
  rm -f "$OUT/replay_l4b.tsv" # a stale file would be fitted
  echo "l4b skipped (CL_RUN_L4B=0): C1-C3 unpriced"
fi
if [ "${CL_RUN_L4B:-1}" = 1 ]; then
  rm -f "$OUT/replay_l4t.tsv"
  run l4t "$HERE/cl_replay_ipc" l4t "${EXPS[@]}" --out "$OUT/replay_l4t.tsv" || echo "$(date '+%F %T') l4t failed (soft: the Tight Inclusion part of C2 is then priced by the per-query fit)"
fi
run l4d "$HERE/cl_replay_ipc" l4d "${EXPS[@]}" --out "$OUT/replay_l4d.tsv" --reps 15 || exit 1
run canary_after "$HERE/canary" --out "$OUT/canary_after.json" || exit 1
if [ $DRY = 1 ]; then echo "[fit] python3 -I $HERE/fit_replay.py --replay $OUT --out $OUT/unit_costs_cl.json"; exit 0; fi
python3 -I "$HERE/fit_replay.py" --replay "$OUT" --out "$OUT/unit_costs_cl.json"
