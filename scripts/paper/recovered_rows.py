#!/usr/bin/env python3
"""Where the Sonnet grid rows with a resend-recovered refusal end up.

Companion to refusal_categories.py (same 12 runs, prior attempts included).
A row "has a recovery" when `refusals_recovered` (row or `prior_attempt_usage`)
is above 0. Each such row falls in exactly one class:

  final_in_row      the row itself also carries a final refusal
                    (`experiments.exclusions.row_refusal_final`);
  excluded_question no final refusal on the row, but its question is in
                    exclusions.json for Sonnet (a final refusal stood on
                    another pipeline or run of that question);
  kept              neither: the row stays in the analysis.

For the kept rows it checks, from row flags only, that each is in
`runs.analysis_rows` (the load_runs filter), is not an error row, has
refusal_seen False and refusal_first_seen True, has a non-empty prediction and
an ok gpt-4.1 judge score. The prediction is tested for emptiness and run
through `is_refusal` (31 patterns) but never printed.

It also ties the count to refusal_categories.py: rows with a refusal (288)
minus rows with a final refusal (32) = rows with a refusal and no final one
(256), and those 256 are exactly the recovered rows outside final_in_row.

Writes refusal_recovered_rows.csv (per dataset x pipeline) and refusal_recovered_rows.json.
No API call.

Writes to scripts/paper/out/refusal/.

    uv run python scripts/paper/recovered_rows.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.analysis.refusal_analysis import is_refusal  # noqa: E402
from experiments.analysis.runs import analysis_rows  # noqa: E402
from experiments.exclusions import excluded_ids, load_exclusions, row_refusal_final  # noqa: E402

MODEL = "claude-sonnet-5-5"
ROOT = REPO / "data/results-campaign"
JUDGE = "gpt-4-1_39118f"
CLASSES = ("kept", "excluded_question", "final_in_row")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "scripts" / "paper" / "out" / "refusal")
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)

    exclusions = load_exclusions(ROOT / "exclusions.json")
    cell = defaultdict(Counter)
    tot = Counter()
    kept_checks = Counter()
    for run in sorted(glob.glob(str(ROOT / f"*_rq1_*_{MODEL}_k*"))):
        ds = os.path.basename(run).split("_")[3]
        excl = excluded_ids(exclusions, ds, MODEL)
        for f in sorted(glob.glob(f"{run}/rq1_*.jsonl")):
            if "_judged_" in f:
                continue
            pipe = os.path.basename(f)[len("rq1_") : -len(".jsonl")]
            with open(f, encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            in_analysis = {str(r["id"]) for _, r in analysis_rows(rows, excl)[0]}
            judged = {}
            jf = f"{f[: -len('.jsonl')]}_judged_{JUDGE}.jsonl"
            if os.path.exists(jf):
                with open(jf, encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            j = json.loads(line)
                            judged[str(j["id"])] = (
                                j.get("judge_status", "ok") == "ok"
                                and j.get("llm_judge") is not None
                            )
            for r in rows:
                usages = (r, r.get("prior_attempt_usage") or {})
                refused = sum(int(u.get("refused_calls") or 0) for u in usages)
                recovered = sum(int(u.get("refusals_recovered") or 0) for u in usages)
                final = row_refusal_final(r)
                qid = str(r["id"])
                tot["rows_refused"] += refused > 0
                tot["rows_final"] += final
                tot["rows_refused_no_final"] += refused > 0 and not final
                tot["rows_refused_no_final_no_recovery"] += (
                    refused > 0 and not final and not recovered
                )
                if not recovered:
                    continue
                cls = "final_in_row" if final else ("excluded_question" if qid in excl else "kept")
                c = cell[(ds, pipe)]
                c[cls] += 1
                c["recovered_calls"] += recovered
                tot[cls] += 1
                tot["rows_recovered"] += 1
                tot["recovered_calls"] += recovered
                if cls == "kept":
                    pred = r.get("prediction")
                    kept_checks["in_analysis_rows"] += qid in in_analysis
                    kept_checks["no_error"] += "error" not in r
                    kept_checks["refusal_seen_false"] += not r.get("refusal_seen")
                    kept_checks["refusal_first_seen_true"] += bool(r.get("refusal_first_seen"))
                    kept_checks["prediction_non_empty"] += bool(pred)
                    kept_checks["judge_ok"] += bool(judged.get(qid))
                    kept_checks["answer_text_refusal_marker"] += bool(is_refusal(pred or ""))
                    c["kept_answer_text_refusal_marker"] += bool(is_refusal(pred or ""))

    with open(out / "refusal_recovered_rows.csv", "w", newline="") as fh:
        fields = [
            "dataset",
            "pipeline",
            *CLASSES,
            "recovered_calls",
            "kept_answer_text_refusal_marker",
        ]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for (ds, pipe), c in sorted(cell.items()):
            w.writerow({"dataset": ds, "pipeline": pipe, **{k: c[k] for k in fields[2:]}})

    summary = {
        "rows_recovered": tot["rows_recovered"],
        "recovered_calls": tot["recovered_calls"],
        "by_class": {k: tot[k] for k in CLASSES},
        "kept_checks": dict(kept_checks),
        "tie_to_refusal_categories": {
            "rows_refused": tot["rows_refused"],
            "rows_final": tot["rows_final"],
            "rows_refused_no_final": tot["rows_refused_no_final"],
            "rows_refused_no_final_without_recovery": tot["rows_refused_no_final_no_recovery"],
            "kept_plus_excluded_question": tot["kept"] + tot["excluded_question"],
        },
    }
    (out / "refusal_recovered_rows.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
