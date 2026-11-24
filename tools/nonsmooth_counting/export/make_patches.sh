#!/bin/bash
# make_patches.sh: writes patches/polyfem_export.patch and patches/polysolve_export.patch (diff -ruN of the unmodified copies pf_orig, ps_orig against the hooked trees pf, ps in $EXPORT_WORK).
# Run it after editing $EXPORT_WORK/pf or $EXPORT_WORK/ps. sync.sh applies the patches to fresh copies.
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${EXPORT_WORK:-$HOME/nonsmooth_export}"
mkdir -p "$HERE/patches"
cd "$WORK" || exit 2
command diff -ruN pf_orig pf > "$HERE/patches/polyfem_export.patch"
command diff -ruN ps_orig ps > "$HERE/patches/polysolve_export.patch"
wc -l "$HERE"/patches/*.patch
