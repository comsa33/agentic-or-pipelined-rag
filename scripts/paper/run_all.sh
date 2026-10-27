#!/bin/bash
# Rebuild the paper's tables from the released runs, in dependency order.
#
# Needs data/results-campaign and data/raw (see README, Data and analysis).
# The human-labelled rows for judge_gpt41_grid.py are the supplementary file
# judge41_vs_human_hedge167_rows.csv; pass its path as the first argument.
# Parts that read the uncontrolled trees (data/results/paper,
# data/results-openweight/final) or the Sonnet smoke tree run only when those
# trees are present. Everything is written to scripts/paper/out/. No API call.
#
# Usage:
#   scripts/paper/run_all.sh <judge41_vs_human_hedge167_rows.csv>
set -euo pipefail
cd "$(dirname "$0")/../.."

HEDGE="${1:?usage: $0 <judge41_vs_human_hedge167_rows.csv>}"
run() { echo "== $*"; uv run python "scripts/paper/$1" "${@:2}"; }

run controlled_grid_analysis.py
run controlled_grid_em_rouge.py
run si_tables.py
run f2_f3.py
run f2_sonnet.py
run judge_gpt41_grid.py --hedge "$HEDGE"
run mplus_gpt41.py
run stability_gpt41.py
run j4_identical_delta.py
run interaction_equivalence.py
run sens_a_table.py
run sens_f.py
run sens_f.py --mode asis
run recovered_rows.py
run refusal_categories.py
run compare_manifest_summary.py

if [ -d data/results/paper ] && [ -d data/results-openweight/final ]; then
  run refusal_rates_v31.py
else
  echo "== refusal_rates_v31.py skipped: uncontrolled-grid trees not present"
fi
run f3_smoke_refusals.py || true
