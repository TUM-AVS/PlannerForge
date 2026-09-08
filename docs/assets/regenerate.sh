#!/usr/bin/env bash
# Regenerate the README figures from the camera-ready paper sources.
#
# The paper is the single source of truth for every figure in the README: run
# this after the paper's figures change so the two never drift. PDFs are
# rasterised to a fixed WIDTH in pixels, not a fixed DPI — the paper's figure
# pages have very different page sizes, so a fixed DPI turns one of them into a
# 10 MB image. Figures already stored as PNG in the paper are copied
# byte-for-byte.
#
# Usage:  bash docs/assets/regenerate.sh [path/to/PlannerForge-EMNLP/latex/figures]
set -euo pipefail

FIGS="${1:-../PlannerForge-EMNLP/latex/figures}"
OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WIDTH=2400   # output width in pixels; GitHub renders READMEs at ~900 px

if [ ! -d "$FIGS" ]; then
    echo "ERROR: paper figure directory not found: $FIGS" >&2
    echo "       Pass it as the first argument." >&2
    exit 1
fi

# paper source            -> README asset
rasterise() {
    echo "  $1  ->  $2.png  (${WIDTH} px wide)"
    pdftocairo -png -scale-to-x "$WIDTH" -scale-to-y -1 -singlefile \
        "$FIGS/$1" "$OUT/$2"
}
copy() {
    echo "  $1  ->  $2.png  (copied)"
    cp "$FIGS/$1" "$OUT/$2.png"
}

rasterise 00_concept.pdf                concept
rasterise Framework.pdf                 framework
rasterise fig_all_tasks_all_models.pdf  results_tasks
copy      fig_cross_planner_frenetix.png cross_planner_frenetix
copy      fig_cross_planner_rbfn.png     cross_planner_rbfn

# Record which revision of the paper these figures came from, so drift between
# the README and the paper is detectable rather than silent.
PAPER_REPO="$(cd "$FIGS/../.." 2>/dev/null && pwd)"
{
    echo "PlannerForge README figures"
    echo "regenerated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "source:      $FIGS"
    if git -C "$PAPER_REPO" rev-parse --git-dir >/dev/null 2>&1; then
        echo "paper commit: $(git -C "$PAPER_REPO" rev-parse --short HEAD) \
$(git -C "$PAPER_REPO" log -1 --format=%s)"
    fi
    echo "width:       ${WIDTH} px"
} > "$OUT/SOURCE.txt"

echo "Done. Figures regenerated from: $FIGS"
cat "$OUT/SOURCE.txt"
