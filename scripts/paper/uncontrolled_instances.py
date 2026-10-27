#!/usr/bin/env python3
"""EM, F1 and ROUGE-L of the agent variants on the uncontrolled grid (table tab:instances).

Reads the March 2026 Gemini runs of data/results/paper (the rq3 evaluation-tool
runs, the rq4 structure-tool runs and the ablation runs) and the 27 March
manual-prompt reruns in data/results. Each metric takes the maximum over the
reference variants (`all_references`), as `runs.best_f1` does for F1. The
run-time summaries score against the single `reference` instead, and that
differs only on MuSiQue, whose questions carry answer aliases. Rows with an
error are left out of the means, as in the run summaries; the n column counts
the questions asked.

Before writing, the single-reference recount of every variant is checked
against its run summary, so the rows read are the rows the summary scored.

Writes rq3_evaluation_ablation.csv, rq4_structure_ablation.csv and
ablation_full.csv to scripts/paper/out/uncontrolled_instances/. Reading only.
The script needs data/results/paper and the 27 March reruns.

    uv run python scripts/paper/uncontrolled_instances.py
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
REPO = Path(__file__).resolve().parents[2]
ap.add_argument("--code-repo", type=Path, default=REPO)
ap.add_argument(
    "--out-dir", type=Path, default=REPO / "scripts" / "paper" / "out" / "uncontrolled_instances"
)
args = ap.parse_args()
code = args.code_repo.resolve()
sys.path.insert(0, str(code))

from agentic_rag.evaluation.metrics import exact_match, rouge_l, token_f1  # noqa: E402
from experiments.common import _prediction_text  # noqa: E402

PAPER = code / "data" / "results" / "paper"
RERUNS = code / "data" / "results"
RUN_DIR = re.compile(
    r"^(\d{8}_\d{6})_(rq3|rq4|ablation)_(hotpotqa|2wikimultihopqa|musique|financebench)"
    r"_n(200|150)_gemini-3\.1-flash-lite$"
)
METRICS = (("exact_match", exact_match), ("f1", token_f1), ("rouge_l", rouge_l))


def runs(root: Path, experiment: str, day: str) -> list[tuple[str, Path]]:
    """(dataset, run directory) in run-time order."""
    found = []
    for d in sorted(root.iterdir()):
        m = RUN_DIR.match(d.name)
        if m and m.group(2) == experiment and m.group(1).startswith(day):
            found.append((m.group(3), d))
    return found


def score(path: Path) -> dict:
    """Best-reference means over the valid rows; checks the single-reference recount."""
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    valid = [r for r in rows if "error" not in r]
    summary = json.loads(path.with_name(path.stem + "_summary.json").read_text(encoding="utf-8"))
    recorded = summary["metrics"]
    if recorded["n"] != len(valid):
        sys.exit(f"{path}: summary n {recorded['n']} != {len(valid)} valid rows")
    out = {"n": summary["total_items"]}  # questions asked, error rows included
    for name, fn in METRICS:
        single, best = 0.0, 0.0
        for r in valid:
            pred = _prediction_text(r.get("prediction"))
            answers = [
                a for a in (r.get("all_references") or [r.get("reference", "")]) if a is not None
            ]
            single += fn(pred, r.get("reference", ""))
            best += max(fn(pred, a) for a in answers)
        if abs(single / len(valid) - recorded[name]) > 1e-9:
            sys.exit(f"{path}: single-reference {name} does not match the run summary")
        out[name] = best / len(valid)
    return out


def table(experiment: str, day: str, root: Path, source: str | None, rename=None) -> list[dict]:
    rows = []
    for dataset, d in runs(root, experiment, day):
        for f in sorted(d.glob(f"{experiment}_*.jsonl")):
            if "_judged" in f.name:
                continue
            variant = f.stem[len(experiment) + 1 :]
            if rename:
                variant = rename(variant)
            row = {"dataset": dataset, "variant": variant, **score(f)}
            for name, _ in METRICS:
                row[name] = round(row[name], 4)
            if source is not None:
                row["source"] = source
            rows.append(row)
    return rows


for tree in (PAPER, RERUNS):
    if not tree.is_dir():
        sys.exit(f"{tree.relative_to(code)} not present: this table needs the March 2026 runs")

if len(runs(RERUNS, "ablation", "20260327")) != 4:
    sys.exit("the four 27 March 2026 manual-prompt reruns are not in data/results")

tables = {
    "rq3_evaluation_ablation.csv": table("rq3", "202603", PAPER, None),
    "rq4_structure_ablation.csv": table("rq4", "202603", PAPER, None),
    "ablation_full.csv": table("ablation", "20260325", PAPER, "paper_dir")
    + table("ablation", "20260327", RERUNS, "2026-03-27_rerun", lambda v: f"{v}_bugfixed"),
}
args.out_dir.mkdir(parents=True, exist_ok=True)
for name, rows in tables.items():
    with (args.out_dir / name).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out_dir / name} ({len(rows)} rows)")
