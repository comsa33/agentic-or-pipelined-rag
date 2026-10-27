#!/usr/bin/env python3
"""Recount the uncontrolled grid's refusal-marker rates with the 31-pattern REFUSAL_PATTERNS.

Reads the March (Gemini, gpt-5-mini) and July (gpt-oss-120b) runs of the
uncontrolled grid, data/results/paper and data/results-openweight/final, with
the frozen 31-pattern set and writes scripts/paper/out/uncontrolled_refusal_markers.csv.
Reading only. The script needs both trees.

--legacy names an earlier 21-pattern refusal_rates.csv. When it is given, the
script first checks that the 21 legacy patterns reproduce that file and then
reports the cells that changed. Without it, only the new file is written.

    uv run python scripts/paper/refusal_rates_v31.py
"""

import argparse
import csv
import re
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
REPO = Path(__file__).resolve().parents[2]
ap.add_argument("--code-repo", type=Path, default=REPO)
ap.add_argument(
    "--out",
    type=Path,
    default=REPO / "scripts" / "paper" / "out" / "uncontrolled_refusal_markers.csv",
)
ap.add_argument(
    "--legacy", type=Path, default=None, help="earlier 21-pattern refusal_rates.csv to compare with"
)
args = ap.parse_args()
code = args.code_repo.resolve()
sys.path.insert(0, str(code))

from experiments.analysis.refusal_analysis import (  # noqa: E402
    LEGACY_REFUSAL_PATTERNS,
    REFUSAL_PATTERNS,
    RefusalAnalyzer,
    _prediction_text,
)
from experiments.analysis.runs import load_runs  # noqa: E402

records = []
for tree, model in (
    ("data/results/paper", None),
    ("data/results-openweight/final", "gpt-oss-120b"),
):
    if not (code / tree).is_dir():
        sys.exit(f"{tree} not present: this table needs the uncontrolled-grid trees")
    records += load_runs(code / tree, prefix="rq1", model=model)
new_rows = RefusalAnalyzer(records).compute()
legacy_re = re.compile("|".join(LEGACY_REFUSAL_PATTERNS), re.IGNORECASE)
legacy = {}
for rec in records:
    legacy[(rec["model"], rec["dataset"], rec["pipeline"])] = sum(
        bool(legacy_re.search(_prediction_text(r.get("prediction")))) for r in rec["rows"]
    )

out_path = args.out
out_path.parent.mkdir(parents=True, exist_ok=True)
fields = [
    "model",
    "dataset",
    "pipeline",
    "total",
    "refusals",
    "refusal_rate",
    "f1_all",
    "f1_non_refused",
    "f1_refused",
]
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(new_rows)

if args.legacy is None:
    print(f"wrote {out_path.name}: {len(new_rows)} cells, {len(REFUSAL_PATTERNS)} patterns")
    sys.exit(0)

old = {
    (r["model"], r["dataset"], r["pipeline"]): r
    for r in csv.DictReader(open(args.legacy, encoding="utf-8"))
}
# the legacy recount must reproduce the 2026-09-07 file, or the comparison means nothing
mismatch = [k for k, n in legacy.items() if int(old[k]["refusals"]) != n]
assert not mismatch, f"legacy recount differs from refusal_rates.csv in {mismatch[:5]}"
assert set(old) == set(legacy)

changed = []
for r in new_rows:
    k = (r["model"], r["dataset"], r["pipeline"])
    if r["refusals"] != legacy[k]:
        changed.append((k, legacy[k], r["refusals"], r["total"]))
rows_changed = sum(c[2] - c[1] for c in changed)
print(
    f"legacy recount reproduces {len(old)} cells; cells changed {len(changed)}/{len(new_rows)}, rows {rows_changed}"
)
for (m, d, p), a, b, t in sorted(changed):
    print(f"  {m} {d} {p}: {a} -> {b} of {t}")
