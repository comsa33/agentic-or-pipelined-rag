#!/usr/bin/env python3
"""Budget-matched cell on Claude Sonnet 5.5, 2WikiMultiHopQA: Loop(budget=10) against the controlled grid, judged by gpt-4.1.

Reads the Sonnet controlled grid on 2Wiki (rq1 k1-3) and the one Sonnet
budget-matched run (budget_matched k1). Of that run only Loop(budget=10) is
complete (200 rows; its result file was written from its checkpoint, with no
model call). Agentic(budget=6) was stopped after 2 rows (both final
refusals), so Agentic(budget=6) - Loop and Agentic(budget=6) - Agentic are not
computed. IRCoT does not run on Sonnet: every step call was refused in the
smoke run.

Questions dropped: the union of the grid exclusions.json and
exclusions_f23.json for Sonnet on 2Wiki, from every pipeline and run.

Writes to scripts/paper/out/budget/:
1. budget_matched_sonnet_contrasts.csv: Agentic - Loop(budget=10) and Loop(budget=10) - Loop
   (both pre-specified) per hop bin and over the dataset, F1 and Judge, with
   the grid side as the per-question k1-3 mean (as in f2_f3.py) and, as a
   sensitivity row, the grid k1 alone;
2. budget_matched_sonnet_levels.csv: F1, Judge, metered calls and cost per question of
   Agentic, Loop and Loop(budget=10) on the paired questions;
3. budget_matched_sonnet_summary.json: n, Holm sizes, rechecked cells, cost multiples,
   and the decision-rule reading available without Agentic(budget=6).
Statistics as in f2_f3.py: paired bootstrap 10,000 (seed 42) from
hop_level._paired_bootstrap with runs.comparison_rng, Holm within (family,
metric) over the pre-specified contrasts in the hop cells (the dataset row is
descriptive), p_holm in [0.04, 0.06] redrawn 10x. Calls = metered_calls.
No API call; no prediction is printed.

    uv run python scripts/paper/f2_sonnet.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "scripts" / "paper" / "out"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f2_f3 import holm, sign_of, write_csv  # noqa: E402

MODEL = "claude-sonnet-5-5"
DS = "2wikimultihopqa"
JUDGE = "gpt-4-1_39118f"
SEED, N_BOOT = 42, 10000
AGENTIC, LOOP, LOOP10 = "agentic_(react)", "loop_refinement", "loop_(budget=10)"
SHORT = {AGENTIC: "Agentic", LOOP: "Loop", LOOP10: "Loop(budget=10)"}
PAIRS = ((AGENTIC, LOOP10), (LOOP10, LOOP))
F2_RUN = "20261027_075201_budget_matched_2wikimultihopqa_n200_claude-sonnet-5-5_k1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--code-repo", type=Path, default=REPO)
    ap.add_argument("--out", type=Path, default=OUT / "budget")
    args = ap.parse_args()

    code = args.code_repo.resolve()
    results = code / "data" / "results-campaign"
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(code))

    from loguru import logger

    from experiments.analysis.hop_level import _paired_bootstrap
    from experiments.analysis.runs import best_f1, comparison_rng, load_runs
    from experiments.difficulty import hop_for_row, hop_reference
    from experiments.exclusions import excluded_ids, load_exclusions

    logger.remove()

    grid_dirs = sorted(results.glob(f"*_rq1_{DS}_n200_{MODEL}_k[123]"))
    if len(grid_dirs) != 3 or not (results / F2_RUN).is_dir():
        raise SystemExit(f"expected 3 grid runs and {F2_RUN}, found {len(grid_dirs)}")

    # exclusions: grid file and the budget-matched file, merged for this dataset and model
    grid_ex = excluded_ids(load_exclusions(results / "exclusions.json"), DS, MODEL)
    f23_ex = excluded_ids(load_exclusions(results / "exclusions_f23.json"), DS, MODEL)
    union = sorted(grid_ex | f23_ex)
    stage = Path(tempfile.mkdtemp(prefix="f2sonnet_stage_"))
    for p in (*grid_dirs, results / F2_RUN):
        (stage / p.name).symlink_to(p)
    (stage / "exclusions.json").write_text(
        json.dumps(
            {
                DS: {
                    MODEL: [
                        {
                            "id": i,
                            "reason": "anthropic_refusal",
                            "note": "union of exclusions.json and exclusions_f23.json",
                        }
                        for i in union
                    ]
                }
            }
        )
    )
    try:
        records = [
            rec
            for prefix in ("rq1", "budget_matched")
            for rec in load_runs(stage, prefix=prefix, judge=JUDGE, require_paired_n=False)
        ]
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    if sum(rec.get("n_judge_missing", 0) for rec in records):
        raise SystemExit(f"rows without a {JUDGE} judge score")

    ref = hop_reference(DS, code / "data" / "raw")
    metrics = ("f1", "judge", "metered", "cost")
    val: dict = {m: defaultdict(lambda: defaultdict(dict)) for m in metrics}
    hop_of: dict[str, int] = {}
    for rec in records:
        pipe, k = rec["pipeline"], rec["repeat_index"]
        if pipe not in SHORT:
            continue
        for r in rec["rows"]:
            q = str(r["id"])
            hop_of[q] = hop_for_row(r, ref, f"{DS}/{pipe}")
            val["f1"][pipe][k][q] = best_f1(r.get("prediction") or "", r)
            val["judge"][pipe][k][q] = float(r["llm_judge"])
            val["metered"][pipe][k][q] = float(r.get("metered_calls") or 0)
            val["cost"][pipe][k][q] = float(r.get("cost_usd") or 0)
    if sorted(val["f1"][LOOP10]) != [1] or sorted(val["f1"][AGENTIC]) != [1, 2, 3]:
        raise SystemExit("unexpected runs per pipeline")

    def ks(pipe: str, grid_k: str) -> list[int]:
        runs = sorted(val["f1"][pipe])
        return runs if (pipe == LOOP10 or grid_k == "k1-3") else [1]

    def qids_of(*pipes: str) -> list[str]:
        return sorted(set.intersection(*[set(v) for p in pipes for v in val["f1"][p].values()]))

    def arr(metric: str, pipe: str, qids: list[str], grid_k: str) -> np.ndarray:
        runs = val[metric][pipe]
        return np.array([np.mean([runs[k][q] for k in ks(pipe, grid_k)]) for q in qids])

    def bins(qids: list[str]) -> dict:
        b: dict = defaultdict(list)
        for q in qids:
            b[hop_of[q]].append(q)
        b = dict(sorted(b.items()))
        b["all"] = list(qids)
        return b

    rows = []
    for grid_k in ("k1-3", "k1"):
        for a_p, b_p in PAIRS:
            for hop, qids in bins(qids_of(a_p, b_p)).items():
                for metric in ("f1", "judge"):
                    a, b = arr(metric, a_p, qids, grid_k), arr(metric, b_p, qids, grid_k)
                    key = ("F2-sonnet", grid_k, metric, DS, str(hop), a_p, b_p)
                    p, lo, hi, d = _paired_bootstrap(
                        a, b, comparison_rng(SEED, *key), n_boot=N_BOOT
                    )
                    sd = float((a - b).std(ddof=1))
                    rows.append(
                        {
                            "grid_runs": grid_k,
                            "dataset": DS,
                            "hop": hop,
                            "metric": metric,
                            "contrast": f"{SHORT[a_p]} - {SHORT[b_p]}",
                            "descriptive": hop == "all" or grid_k == "k1",
                            "n": len(qids),
                            "runs_a": "k" + ",".join(map(str, ks(a_p, grid_k))),
                            "runs_b": "k" + ",".join(map(str, ks(b_p, grid_k))),
                            "mean_a": float(a.mean()),
                            "mean_b": float(b.mean()),
                            "delta": d,
                            "ci_lo": lo,
                            "ci_hi": hi,
                            "p": p,
                            "cohen_d": d / sd if sd > 0 else float("nan"),
                            "sign": sign_of(lo, hi),
                            "metered_a": float(arr("metered", a_p, qids, grid_k).mean()),
                            "metered_b": float(arr("metered", b_p, qids, grid_k).mean()),
                            "cost_a": float(arr("cost", a_p, qids, grid_k).mean()),
                            "cost_b": float(arr("cost", b_p, qids, grid_k).mean()),
                            "_key": key,
                            "_pipes": (a_p, b_p),
                            "_qids": qids,
                            "_grid_k": grid_k,
                        }
                    )
    holm_m = {}
    for metric in ("f1", "judge"):
        g = [r for r in rows if r["metric"] == metric and not r["descriptive"]]
        for r, adj in zip(g, holm([x["p"] for x in g])):
            r["p_holm"] = adj
        for r in g:
            if 0.04 <= r["p_holm"] <= 0.06:
                a_p, b_p = r["_pipes"]
                r["p_recheck"] = _paired_bootstrap(
                    arr(metric, a_p, r["_qids"], r["_grid_k"]),
                    arr(metric, b_p, r["_qids"], r["_grid_k"]),
                    comparison_rng(SEED, *r["_key"]),
                    n_boot=10 * N_BOOT,
                )[0]
        for r, adj in zip(g, holm([x.get("p_recheck", x["p"]) for x in g])):
            r["p_holm_final"] = adj
            r["holm_sig"] = adj < 0.05
        holm_m[metric] = len(g)
    write_csv(out / "budget_matched_sonnet_contrasts.csv", rows)

    levels = []
    common = qids_of(AGENTIC, LOOP, LOOP10)
    for hop, qids in bins(common).items():
        for p in (AGENTIC, LOOP, LOOP10):
            levels.append(
                {
                    "dataset": DS,
                    "hop": hop,
                    "pipeline": SHORT[p],
                    "n": len(qids),
                    "runs": "k" + ",".join(map(str, ks(p, "k1-3"))),
                    **{m: float(arr(m, p, qids, "k1-3").mean()) for m in metrics},
                }
            )
    write_csv(out / "budget_matched_sonnet_levels.csv", levels)

    lv = {(r["hop"], r["pipeline"]): r for r in levels}
    loop10_m = np.array(list(val["metered"][LOOP10][1].values()))
    summary = {
        "model": MODEL,
        "dataset": DS,
        "judge": JUDGE,
        "excluded": {"grid": sorted(grid_ex), "f23": sorted(f23_ex), "union_n": len(union)},
        "n_paired_all_three": len(common),
        "holm_m": holm_m,
        "rechecked_cells": sum("p_recheck" in r for r in rows),
        "cost_multiple_loop10_over_agentic": lv[("all", "Loop(budget=10)")]["cost"]
        / lv[("all", "Agentic")]["cost"],
        "cost_multiple_loop10_over_loop": lv[("all", "Loop(budget=10)")]["cost"]
        / lv[("all", "Loop")]["cost"],
        "metered_multiple_loop10_over_agentic": lv[("all", "Loop(budget=10)")]["metered"]
        / lv[("all", "Agentic")]["metered"],
        "loop10_metered_rows": {
            "mean": float(loop10_m.mean()),
            "min": int(loop10_m.min()),
            "max": int(loop10_m.max()),
            "rows_over_10": int((loop10_m > 10).sum()),
        },
        "T2_available": {
            m: next(
                r["sign"]
                for r in rows
                if (r["grid_runs"], r["hop"], r["metric"], r["contrast"])
                == ("k1-3", "all", m, "Agentic - Loop(budget=10)")
            )
            for m in ("f1", "judge")
        },
        "not_computed": [
            "Agentic(budget=6) - Loop",
            "Agentic(budget=6) - Agentic (b6 cell aborted)",
            "Agentic - IRCoT, Loop - IRCoT (IRCoT not runnable on Sonnet)",
        ],
    }
    (out / "budget_matched_sonnet_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
