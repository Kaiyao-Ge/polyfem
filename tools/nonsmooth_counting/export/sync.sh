#!/bin/bash
# sync.sh: rebuild the export trees from the production code.
#   1. copies the working trees of the production polyfem and polysolve (without .git, build directories and data) to $EXPORT_WORK/pf_orig and $EXPORT_WORK/ps_orig
#   2. copies pf_orig -> pf and ps_orig -> ps (the trees the export build compiles)
#   3. applies the hook patches in patches/ to pf and ps with `patch -p1` (the copies are not git repositories; nothing here touches a git repository)
# Usage: sync.sh [--no-fetch]      --no-fetch: keep pf_orig and ps_orig as they are and only re-copy and re-patch (used after editing the patches)
# Environment:
#   EXPORT_WORK  work folder for the copies and the build, outside the repository (default $HOME/nonsmooth_export)
#   PF_SRC       production polyfem tree (default: the repository this script is in)
#   PS_SRC       production polysolve tree (default $HOME/polysolve-pr)
# Needs rsync and patch. Exit 1 if a hunk of a patch does not apply (the rejected hunks are in pf/*.rej, ps/*.rej).
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${EXPORT_WORK:-$HOME/nonsmooth_export}"
PF_SRC="${PF_SRC:-$(cd "$HERE/../../.." && pwd)}"
PS_SRC="${PS_SRC:-$HOME/polysolve-pr}"
case "$WORK/" in "$PF_SRC"/*) echo "sync.sh: EXPORT_WORK ($WORK) must be outside the polyfem tree ($PF_SRC)"; exit 2 ;; esac
mkdir -p "$WORK"
EXCL=(--exclude=/.git --exclude='/build*' --exclude=/data --exclude=/diff-data --exclude=/polyspline-data --exclude=/pref-data --exclude='*:Zone.Identifier' --exclude='__pycache__')
if [ "${1:-}" != "--no-fetch" ]; then
  for pair in "$PF_SRC:pf_orig" "$PS_SRC:ps_orig"; do
    src="${pair%%:*}"; dst="${pair##*:}"
    [ -d "$src/src" ] || { echo "sync.sh: $src is not a source tree"; exit 2; }
    echo "copy $src -> $WORK/$dst"
    rsync -a --delete "${EXCL[@]}" "$src/" "$WORK/$dst/"
  done
fi
for t in pf ps; do
  echo "copy ${t}_orig -> $t"
  rsync -a --delete "$WORK/${t}_orig/" "$WORK/$t/"
done
rc=0
for pair in "pf:polyfem_export.patch" "ps:polysolve_export.patch"; do
  t="${pair%%:*}"; p="$HERE/patches/${pair##*:}"
  if [ -s "$p" ]; then
    echo "patch $t with $p"
    patch -p1 -d "$WORK/$t" --no-backup-if-mismatch < "$p" || rc=1
  else
    echo "no patch $p (the $t tree is the production code)"
  fi
done
[ $rc = 0 ] && echo "sync done ($WORK)" || { echo "sync.sh: a patch did not apply cleanly"; exit 1; }
