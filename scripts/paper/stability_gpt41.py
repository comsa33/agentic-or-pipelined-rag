#!/usr/bin/env python3
"""Per-run stability (the f6 rule) of the judge Agentic-Loop difference under both judge tags.

Reads the 24 controlled-grid runs twice (gpt-4.1-nano and gpt-4.1 tags) and, per
(model, dataset, hop) cell, reports the three per-run differences, their mean,
sample SD (ddof=1), whether the three share a sign, and the verdict
`stable` (same sign and SD < |mean|) or `no reliable difference`. The nano rows
reproduce f6_stability.csv of controlled_grid_analysis.py. No API call.

    uv run python scripts/paper/stability_gpt41.py
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

DATASETS = ("hotpotqa", "2wikimultihopqa", "musique", "financebench")
MODELS = ("gpt-5-mini", "claude-sonnet-5-5")
AGENTIC, LOOP = "agentic_(react)", "loop_refinement"
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "scripts" / "paper" / "out"
JUDGES = {"nano": "gpt-4-1-nano_7bcf05", "gpt41": "gpt-4-1_39118f"}
RUN_DIR = re.compile(
    r"^(\d{8}_\d{6})_rq1_(hotpotqa|2wikimultihopqa|musique|financebench)"
    r"_n(200|150)_(gpt-5-mini|claude-sonnet-5-5)_k([123])$"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--code-repo", type=Path, default=REPO)
    ap.add_argument("--out", type=Path, default=OUT / "judge_gpt41_grid")
    args = ap.parse_args()
    code = args.code_repo.resolve()
    sys.path.insert(0, str(code))
    from loguru import logger

    from experiments.analysis.runs import load_runs
    from experiments.difficulty import hop_for_row, hop_reference

    logger.remove()
    results = code / "data" / "results-campaign"
    run_dirs = sorted(p for p in results.iterdir() if p.is_dir() and RUN_DIR.match(p.name))
    if len(run_dirs) != 24:
        raise SystemExit(f"expected 24 run directories, found {len(run_dirs)}")
    stage = Path(tempfile.mkdtemp(prefix="grid_stage_"))
    for p in run_dirs:
        (stage / p.name).symlink_to(p)
    (stage / "exclusions.json").symlink_to(results / "exclusions.json")
    try:
        records = {
            j: load_runs(stage, prefix="rq1", judge=tag, require_paired_n=True)
            for j, tag in JUDGES.items()
        }
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    refs = {ds: hop_reference(ds, code / "data" / "raw") for ds in DATASETS}
    score: dict = {j: defaultdict(lambda: defaultdict(dict)) for j in JUDGES}
    hop_of: dict = defaultdict(dict)
    for j, recs in records.items():
        for rec in recs:
            key = (rec["model"], rec["dataset"], rec["pipeline"])
            for r in rec["rows"]:
                qid = str(r["id"])
                hop_of[(rec["model"], rec["dataset"])][qid] = hop_for_row(
                    r, refs[rec["dataset"]], "/".join(key)
                )
                score[j][key][rec["repeat_index"]][qid] = float(r["llm_judge"])
    rows = []
    for j, tag in JUDGES.items():
        for model in MODELS:
            for ds in DATASETS:
                bins: dict = defaultdict(list)
                for q, h in sorted(hop_of[(model, ds)].items()):
                    bins[h].append(q)
                for hop in sorted(bins):
                    qids = bins[hop]
                    ks = sorted(
                        set(score[j][(model, ds, AGENTIC)]) & set(score[j][(model, ds, LOOP)])
                    )
                    deltas = []
                    for k in ks:
                        a = np.array([score[j][(model, ds, AGENTIC)][k][q] for q in qids])
                        b = np.array([score[j][(model, ds, LOOP)][k][q] for q in qids])
                        deltas.append(float((a - b).mean()))
                    d = np.array(deltas)
                    mean_d, sd = float(d.mean()), float(d.std(ddof=1))
                    same = bool(all(x > 0 for x in d) or all(x < 0 for x in d))
                    rows.append(
                        {
                            "judge": j,
                            "judge_tag": tag,
                            "metric": "judge",
                            "model": model,
                            "dataset": ds,
                            "hop": hop,
                            "n_runs": len(ks),
                            "n": len(qids),
                            "delta_per_run": " / ".join(f"{x:+.4f}" for x in d),
                            "mean_delta": mean_d,
                            "sd_delta": sd,
                            "same_sign": same,
                            "verdict": "stable"
                            if (same and sd < abs(mean_d))
                            else "no reliable difference",
                        }
                    )
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "judge_stability_by_judge.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(
            f"{r['judge']:5s} {r['model']:18s} {r['dataset']:16s} hop{r['hop']} {r['delta_per_run']}  mean={r['mean_delta']:+.4f} sd={r['sd_delta']:.4f} -> {r['verdict']}"
        )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
