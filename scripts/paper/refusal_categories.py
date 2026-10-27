#!/usr/bin/env python3
"""Refusal categories in the Sonnet controlled grid (12 runs, rq1, k1-3 x 4 datasets).

Reads the result rows of every `*_rq1_*_claude-sonnet-5-5_k*` run, the
earlier attempts at an item (`prior_attempt_usage`) included, and writes:

1. by_pipeline.csv: per (dataset, pipeline) rows, metered calls, refused
   responses split into first sends and resends, resends, recovered calls,
   final refusals, rows with a refusal and with a final refusal, refused
   responses per 1,000 metered calls and the resend recovery rate;
2. categories.csv: refused responses and final refusals per category;
3. windows.csv: per run the refused responses, the densest 10-minute window
   and the largest count on one row. Rows carry no clock time, so each run's
   timeline is rebuilt by adding row latencies in file-write order; the
   overhead between items is left out, which makes windows denser than they
   were (an upper bound);
4. row_distribution.csv: rows by refused responses and by final refusals;
5. summary.json: the totals and the cross-check against exclusions.json.

Accounting (agentic_rag.config.settings AnthropicLM, cost_tracker): a refusal
in a retried category (reasoning_extraction) is answered by up to 3 resends
of the same request. `refused_calls` counts every refused response, first
sends and resends; `refusal_retries` counts resends; `refusals_recovered`
counts calls whose resend was answered; `refusals_final_by_category` counts
calls still refused after the last resend. Hence
    refused first sends = refused_calls - (refusal_retries - refusals_recovered)
    refused first sends = refusals_recovered + final refusals   (checked)
litellm's own retries are off for these calls and a refusal is an HTTP 200,
so no refused response is a transport retry.

No API call; no prediction or question text is read out.

Writes to scripts/paper/out/refusal/.

    uv run python scripts/paper/refusal_categories.py
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
from experiments.exclusions import excluded_ids, load_exclusions, row_refusal_final  # noqa: E402

MODEL = "claude-sonnet-5-5"
ROOT = REPO / "data/results-campaign"
WINDOW_S = 600


def usages(row: dict) -> list[dict]:
    return [row, row.get("prior_attempt_usage") or {}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "scripts" / "paper" / "out" / "refusal")
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)

    fields = (
        "rows",
        "metered",
        "refused",
        "resends",
        "recovered",
        "final",
        "rows_refused",
        "rows_final",
    )
    cell = defaultdict(Counter)
    cat_refused, cat_final = Counter(), Counter()
    dist_refused, dist_final = Counter(), Counter()
    windows = []
    final_ids = defaultdict(set)  # dataset -> ids with row_refusal_final

    runs = sorted(glob.glob(str(ROOT / f"*_rq1_*_{MODEL}_k*")))
    for run in runs:
        name = os.path.basename(run)
        ds = name.split("_")[3]
        files = [f for f in glob.glob(f"{run}/rq1_*.jsonl") if "_judged_" not in f]
        files.sort(key=os.path.getmtime)
        t, events, run_refused, row_max = 0.0, [], 0, 0
        for f in files:
            pipe = os.path.basename(f)[len("rq1_") : -len(".jsonl")]
            c = cell[(ds, pipe)]
            with open(f, encoding="utf-8") as fh:
                for r in map(json.loads, fh):
                    t += r.get("latency_seconds") or 0
                    refused = final = 0
                    for u in usages(r):
                        refused += int(u.get("refused_calls") or 0)
                        final += sum((u.get("refusals_final_by_category") or {}).values())
                        c["metered"] += int(u.get("metered_calls") or 0)
                        c["resends"] += int(u.get("refusal_retries") or 0)
                        c["recovered"] += int(u.get("refusals_recovered") or 0)
                        cat_refused.update(u.get("refusal_categories") or {})
                        cat_final.update(u.get("refusals_final_by_category") or {})
                    c["rows"] += 1
                    c["refused"] += refused
                    c["final"] += final
                    c["rows_refused"] += refused > 0
                    c["rows_final"] += final > 0
                    dist_refused[refused] += 1
                    dist_final[final] += 1
                    if row_refusal_final(r):
                        final_ids[ds].add(str(r["id"]))
                    if refused:
                        events.append((t, refused))
                    run_refused += refused
                    row_max = max(row_max, refused)
        dense = max(
            (sum(n for tj, n in events if ti - WINDOW_S < tj <= ti) for ti, _ in events), default=0
        )
        windows.append(
            {
                "run": name[16:],
                "refused": run_refused,
                "max_10min": dense,
                "max_row": row_max,
                "hours_rebuilt": round(t / 3600, 1),
            }
        )

    def rates(c: Counter) -> dict:
        first = c["refused"] - (c["resends"] - c["recovered"])
        return {
            **{k: c[k] for k in fields},
            "refused_first": first,
            "refused_resend": c["resends"] - c["recovered"],
            "per_1000_calls": round(1000 * c["refused"] / c["metered"], 2) if c["metered"] else 0,
            "recovery_rate": round(c["recovered"] / first, 3) if first else "",
            "identity_ok": first == c["recovered"] + c["final"],
        }

    rows = []
    for (ds, pipe), c in sorted(cell.items()):
        rows.append({"dataset": ds, "pipeline": pipe, **rates(c)})
    for ds in sorted({d for d, _ in cell}):
        tot = sum((c for (d, _), c in cell.items() if d == ds), Counter())
        rows.append({"dataset": ds, "pipeline": "ALL", **rates(tot)})
    for pipe in sorted({p for _, p in cell}):
        tot = sum((c for (_, p), c in cell.items() if p == pipe), Counter())
        rows.append({"dataset": "ALL", "pipeline": pipe, **rates(tot)})
    grand = sum(cell.values(), Counter())
    rows.append({"dataset": "ALL", "pipeline": "ALL", **rates(grand)})

    with open(out / "by_pipeline.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "categories.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "refused", "final"])
        for k in sorted(set(cat_refused) | set(cat_final)):
            w.writerow([k, cat_refused[k], cat_final[k]])
    with open(out / "windows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(windows[0]))
        w.writeheader()
        w.writerows(windows)
    with open(out / "row_distribution.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["count", "rows_by_refused", "rows_by_final"])
        for k in sorted(set(dist_refused) | set(dist_final)):
            w.writerow([k, dist_refused[k], dist_final[k]])

    exclusions = load_exclusions(ROOT / "exclusions.json")
    excl = {}
    for ds, ids in sorted(final_ids.items()):
        listed = excluded_ids(exclusions, ds, MODEL)
        excl[ds] = {
            "questions_with_final_refusal": len(ids),
            "sonnet_exclusions": len(listed),
            "final_not_excluded": sorted(ids - listed),
            "excluded_without_final_refusal": len(listed - ids),
        }
    summary = {
        "runs": len(runs),
        "rows": grand["rows"],
        "totals": rates(grand),
        "categories_refused": dict(cat_refused),
        "categories_final": dict(cat_final),
        "max_10min_window": max(w_["max_10min"] for w_ in windows),
        "max_refused_one_row": max(dist_refused),
        "max_final_one_row": max(dist_final),
        "exclusions_check": excl,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
