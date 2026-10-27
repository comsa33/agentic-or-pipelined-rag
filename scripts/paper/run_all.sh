#!/bin/bash
# Rebuild the paper's tables from the released runs, in dependency order.
#
# Needs data/results-campaign and data/raw (see README, Data and analysis).
# The human-labelled rows for judge_grid.py are the supplementary file
# judge_vs_human_rows.csv; pass its path as the first argument.
# Parts that read the uncontrolled trees (data/results/paper,
# data/results-openweight/final, data/results-topk) or the Sonnet smoke tree
# run only when those trees are present. Everything is written to scripts/paper/out/. No API call.
#
# Usage:
#   scripts/paper/run_all.sh <judge_vs_human_rows.csv>
set -euo pipefail
cd "$(dirname "$0")/../.."

HEDGE="${1:?usage: $0 <judge_vs_human_rows.csv>}"
run() { echo "== $*"; uv run python "scripts/paper/$1" "${@:2}"; }

run controlled_grid_analysis.py
run controlled_grid_em_rouge.py
run si_tables.py
run budget_and_external.py
run budget_matched_sonnet.py
run judge_grid.py --hedge "$HEDGE"
run judge_common_questions.py
run judge_stability.py
run judge_identical_delta.py
run grid_interaction_mde.py
run sensitivity_declined_removed.py
run sensitivity_declined_scored_wrong.py
run sensitivity_declined_scored_wrong.py --mode asis
run recovered_rows.py
run refusal_categories.py
run compare_manifest_summary.py

if [ -d data/results/paper ] && [ -d data/results-openweight/final ]; then
  run uncontrolled_refusal_markers.py
else
  echo "== uncontrolled_refusal_markers.py skipped: uncontrolled-grid trees not present"
fi
if [ -d data/results/paper ]; then
  run uncontrolled_instances.py
else
  echo "== uncontrolled_instances.py skipped: data/results/paper not present"
fi
if [ -d data/results/paper ] && [ -d data/results-openweight/final ] && [ -d data/results-topk ]; then
  run uncontrolled_judge.py
else
  echo "== uncontrolled_judge.py skipped: uncontrolled-grid or sweep trees not present"
fi
run external_smoke_refusals.py || true
