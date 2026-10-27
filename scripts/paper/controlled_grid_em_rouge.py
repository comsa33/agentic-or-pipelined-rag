#!/usr/bin/env python3
"""EM and ROUGE-L per cell of the controlled grid (descriptive, no test).

Cells are those of the main-text table tab:rq1_main: one model family, one
workload and one hop bin, over the 24 campaign runs (2 models x 4 workloads x
k1-3). Every configuration is reported: Naive RAG, Single-Pass, Loop
Refinement and Agentic (ReAct) over k1-3, CRAG-style over its one run (k1).
A value is the mean over the cell's questions of the per-question mean over
runs, as for F1 in tab:rq1_main. EM and ROUGE-L take the maximum over the
reference variants (`all_references`), as `runs.best_f1` does for F1; the
run-time summaries score against the single `reference` instead.

Exclusions: --exclusions grid (default) drops exclusions.json, which is what
tab:rq1_main uses (its n row is checked below); --exclusions union also drops
exclusions_budget_matched.json (the budget-matched list), for comparison only.

F1 is computed in the same pass and the Agentic and Loop levels are checked
against grid_f1_agentic_minus_loop.csv written by controlled_grid_analysis.py (run it
first). Writes scripts/paper/out/controlled_grid_em_rouge.csv. Reads only; no
API call; no prediction is printed.

    uv run python scripts/paper/controlled_grid_em_rouge.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

DATASETS = ("hotpotqa", "2wikimultihopqa", "musique", "financebench")
MODELS = ("gpt-5-mini", "claude-sonnet-5-5")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "scripts" / "paper" / "out"
AGENTIC, LOOP, CRAG = "agentic_(react)", "loop_refinement", "crag_replica"
PIPELINES = ("naive_rag", "single-pass", LOOP, AGENTIC, CRAG)
RUN_DIR = re.compile(
    r"^(\d{8}_\d{6})_rq1_(hotpotqa|2wikimultihopqa|musique|financebench)"
    r"_n(200|150)_(gpt-5-mini|claude-sonnet-5-5)_k([123])$"
)
# n row of the main-text table tab:rq1_main, cells in column order
MAIN_N = {
    "gpt-5-mini": [197, 150, 49, 112, 54, 32, 150],
    "claude-sonnet-5-5": [193, 147, 50, 111, 52, 30, 150],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--code-repo", type=Path, default=REPO)
    ap.add_argument("--grid-dir", type=Path, default=OUT / "controlled_grid")
    ap.add_argument("--exclusions", choices=("grid", "union"), default="grid")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    code = args.code_repo.resolve()
    results = code / "data" / "results-campaign"
    out = args.out or OUT / "controlled_grid_em_rouge.csv"
    sys.path.insert(0, str(code))

    from loguru import logger

    from agentic_rag.evaluation.metrics import exact_match, rouge_l
    from experiments.analysis.refusal_analysis import _prediction_text
    from experiments.analysis.runs import best_f1, load_runs
    from experiments.difficulty import hop_for_row, hop_reference

    logger.remove()

    run_dirs = sorted(p for p in results.iterdir() if p.is_dir() and RUN_DIR.match(p.name))
    if len(run_dirs) != 24:
        raise SystemExit(f"expected 24 run directories, found {len(run_dirs)}")
    stage = Path(tempfile.mkdtemp(prefix="em_rouge_stage_"))
    try:
        for p in run_dirs:
            (stage / p.name).symlink_to(p)
        merged: dict = {}
        for name in ("exclusions.json",) + (
            ("exclusions_budget_matched.json",) if args.exclusions == "union" else ()
        ):
            for ds, by_model in json.loads((results / name).read_text(encoding="utf-8")).items():
                if ds.startswith("_"):
                    continue
                for model, entries in by_model.items():
                    merged.setdefault(ds, {}).setdefault(model, []).extend(entries)
        (stage / "exclusions.json").write_text(json.dumps(merged), encoding="utf-8")
        records = load_runs(stage, prefix="rq1", require_paired_n=True)
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    refs = {ds: hop_reference(ds, code / "data" / "raw") for ds in DATASETS}
    val: dict = defaultdict(lambda: defaultdict(dict))  # (model, ds, pipe) -> k -> qid -> scores
    hop_of: dict = defaultdict(dict)
    multi_ref = 0
    for rec in records:
        model, ds, pipe, k = rec["model"], rec["dataset"], rec["pipeline"], rec["repeat_index"]
        for r in rec["rows"]:
            qid = str(r["id"])
            pred = _prediction_text(r.get("prediction"))
            answers = [
                a for a in (r.get("all_references") or [r.get("reference", "")]) if a is not None
            ]
            multi_ref += len(answers) > 1
            hop_of[(model, ds)][qid] = hop_for_row(r, refs[ds], f"{model}/{ds}/{pipe}")
            val[(model, ds, pipe)][k][qid] = {
                "em": max(exact_match(pred, a) for a in answers),
                "rouge_l": max(rouge_l(pred, a) for a in answers),
                "f1": best_f1(pred, r),
            }

    rows = []
    for model in MODELS:
        n_seen = []
        for ds in DATASETS:
            bins: dict = defaultdict(list)
            for q, h in sorted(hop_of[(model, ds)].items()):
                bins[h].append(q)
            for hop, qids in sorted(bins.items()):
                n_seen.append(len(qids))
                for pipe in PIPELINES:
                    by_k = val[(model, ds, pipe)]
                    ks = sorted(by_k)
                    if not all(q in by_k[k] for k in ks for q in qids):
                        raise SystemExit(f"missing rows: {model} {ds} {pipe}")
                    rows.append(
                        {
                            "model": model,
                            "dataset": ds,
                            "hop": hop,
                            "pipeline": pipe,
                            "n": len(qids),
                            "runs": "k" + ",".join(map(str, ks)),
                            **{
                                m: float(
                                    np.mean([np.mean([by_k[k][q][m] for k in ks]) for q in qids])
                                )
                                for m in ("em", "rouge_l", "f1")
                            },
                        }
                    )
        if args.exclusions == "grid" and n_seen != MAIN_N[model]:
            raise SystemExit(f"n per cell {n_seen} != tab:rq1_main {MAIN_N[model]} for {model}")

    # F1 check against the main-table source (Agentic and Loop levels, k1-3)
    if args.exclusions == "grid":
        src = args.grid_dir / "grid_f1_agentic_minus_loop.csv"
        with open(src, encoding="utf-8") as fh:
            ref_rows = [r for r in csv.DictReader(fh) if r["metric"] == "f1"]
        lv = {(r["model"], r["dataset"], r["hop"], r["pipeline"]): r["f1"] for r in rows}
        checked = 0
        for r in ref_rows:
            key = (r["model"], r["dataset"], int(r["hop"]))
            for pipe, col in ((AGENTIC, "mean_a"), (LOOP, "mean_b")):
                if abs(lv[(*key, pipe)] - float(r[col])) > 1e-9:
                    raise SystemExit(f"F1 mismatch {key} {pipe}: {lv[(*key, pipe)]} vs {r[col]}")
                checked += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    head = subprocess.run(
        ["git", "-C", str(code), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    print(
        f"wrote {out.name}: {len(rows)} rows, exclusions={args.exclusions}, code repo {head}, "
        f"rows with >1 reference {multi_ref}"
        + (f", F1 levels checked {checked}" if args.exclusions == "grid" else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
