#!/usr/bin/env python3
"""Refusal counts of the Sonnet IRCoT smoke run, from its result rows only.

The smoke run (2 questions, 2026-10-27) wrote one row per question to
external_ircot.jsonl. For each row this prints, separately, the row's own
usage and the usage of earlier attempts at the item (`prior_attempt_usage`,
filled when an attempt failed and the item was retried): refused responses,
resends, recovered calls, refused first sends and final refusals. The second
question ended as an error row; its refusals sit only in `prior_attempt_usage`,
which `verify` does not add up (its "refused calls" check reads the row's own
fields). The run log records no refusal lines. No API call; no prediction.
Needs the smoke tree data/results-smoke-f23-sonnet; prints a note and stops
without it.

    uv run python scripts/paper/f3_smoke_refusals.py
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = (
    REPO
    / "data/results-smoke-f23-sonnet/20261027_064644_external_2wikimultihopqa_n2_claude-sonnet-5-5_k1"
)
if not RUN.is_dir():
    sys.exit(f"skipped: {RUN.relative_to(REPO)} not present")


def counts(u: dict) -> dict:
    refused = int(u.get("refused_calls") or 0)
    resends = int(u.get("refusal_retries") or 0)
    recovered = int(u.get("refusals_recovered") or 0)
    return {
        "refused": refused,
        "resends": resends,
        "recovered": recovered,
        "first_sends": refused - (resends - recovered),
        "final": sum((u.get("refusals_final_by_category") or {}).values()),
        "categories": u.get("refusal_categories") or {},
    }


total = {"refused": 0, "first_sends": 0, "resends": 0, "recovered": 0, "final": 0}
with open(RUN / "external_ircot.jsonl", encoding="utf-8") as fh:
    for r in map(json.loads, fh):
        for source, u in (("row", r), ("prior_attempt_usage", r.get("prior_attempt_usage") or {})):
            c = counts(u)
            for k in total:
                total[k] += c[k]
            print(r["id"], "error_row" if "error" in r else "answer_row", source, c)
print("total", total)
