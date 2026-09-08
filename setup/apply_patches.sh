#!/usr/bin/env bash
# Apply the PlannerForge local patches to the planner submodules.
# Idempotent: already patched submodules are detected and skipped.
# See setup/patches/UPSTREAM.md for upstream URLs and base commits.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="$ROOT/setup/patches"

FRENETIX_BASE="fc6e7708ba384f991c2a0a07b7300ab52276c70f"
RBFN_BASE="223c11ddcb3436a28666d86e7b7f4a23833d35d6"

fail=0

apply_patch() {
    local subdir="$1" patch="$2" base="$3"
    local dir="$ROOT/$subdir"
    local name
    name="$(basename "$patch")"

    if [ ! -e "$dir/.git" ]; then
        echo "ERROR: submodule '$subdir' is not initialized." >&2
        echo "       Run: git submodule update --init $subdir" >&2
        fail=1
        return
    fi

    local head
    head="$(git -C "$dir" rev-parse HEAD)"
    if [ "$head" != "$base" ]; then
        echo "WARNING: $subdir is at $head, expected base $base." >&2
        echo "         The patch is only guaranteed to apply at the base commit." >&2
    fi

    if git -C "$dir" apply --reverse --check -p1 "$patch" >/dev/null 2>&1; then
        echo "SKIP: $subdir already patched ($name)."
        return
    fi

    if ! git -C "$dir" apply --check -p1 --whitespace=nowarn "$patch"; then
        echo "ERROR: $name does not apply cleanly in $subdir (dirty checkout?)." >&2
        echo "       Reset with: git -C $subdir checkout -- . && git -C $subdir clean -fd" >&2
        fail=1
        return
    fi

    git -C "$dir" apply -p1 --whitespace=nowarn "$patch"
    echo "OK: applied $name to $subdir."
}

apply_patch "Frenetix-Motion-Planner" "$PATCH_DIR/frenetix.patch" "$FRENETIX_BASE"
apply_patch "RBFN-Motion-Primitives"  "$PATCH_DIR/rbfn.patch"     "$RBFN_BASE"

if [ "$fail" -ne 0 ]; then
    echo "One or more patches failed. See messages above." >&2
    exit 1
fi
echo "All planner patches processed."
