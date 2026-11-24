#!/bin/bash
# run_with_hooks.sh: PolyFEM_bin of the export build with the hooks on. The arguments are those of PolyFEM_bin; the counts and the export go to <-o directory>/cl and <-o directory>/cl/export.
# It can stand in for the binary wherever a path to a binary is taken (e.g. nonsmooth_identity.py --new).
# Environment: EXPORT_WORK (default $HOME/nonsmooth_export); CL_NO_EXPORT=1: counts and category timers only, no export files (the hook share is small, the run is closer to a clean one).
WORK="${EXPORT_WORK:-$HOME/nonsmooth_export}"
out=""; prev=""
for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done
[ -n "$out" ] || { echo "run_with_hooks.sh: no -o DIR in the arguments" >&2; exit 2; }
export POLYFEM_CL_COUNTS="$out/cl"
[ -n "${CL_NO_EXPORT:-}" ] || export POLYFEM_CL_EXPORT="$out/cl/export"
exec "$WORK/build/PolyFEM_bin" "$@"
