#!/usr/bin/env python3
"""Budget-matched and external (IRCoT) cells on gpt-5-mini, judged by gpt-4.1.

Reads the 30 gpt-5-mini runs (controlled grid rq1 k1-3 x 4 datasets,
budget_matched k1-3 x {2Wiki, FB}, external IRCoT k1-3 x 4 datasets) and
writes to scripts/paper/out/budget/:

1. budget_matched_contrasts.csv: the budget-matched contrasts Agentic - Loop(budget=10),
   Agentic(budget=6) - Loop, Loop(budget=10) - Loop, plus the exploratory
   Agentic(budget=6) - Agentic, per (dataset, hop bin) on F1 and Judge;
2. external_ircot_contrasts.csv: Agentic - IRCoT and Loop - IRCoT per (dataset, hop bin);
3. budget_pipeline_levels.csv: per (dataset, hop bin) and pipeline the F1, Judge,
   metered calls and cost per question on the paired questions;
4. budget_six_call_cell.csv: how many metered calls Agentic(budget=6) actually made;
5. ircot_fallback.csv: JSONAdapter re-sends per IRCoT run;
6. verdicts.json: the decision rules read off the tables.

The family column is matched_budget or external_instance (F2 and F3 in the
bootstrap keys). Statistics follow the controlled-grid scripts:
per question the mean over that pipeline's k runs, then a paired bootstrap
(10,000, seed 42) from `hop_level._paired_bootstrap` with
`runs.comparison_rng`. Grid and budget-matched runs are different runs of the
same questions, so k is not matched across them; the per-question k mean is
used for repeated cells. Holm runs within (family, metric) over the
pre-specified contrasts in the non-descriptive hop cells; dataset-level rows,
MuSiQue 4-hop, the FB judge and the exploratory contrast are descriptive.
Cells with p_holm in [0.04, 0.06] are redrawn 10x. Latency is not read.
Calls are `metered_calls`. No API call; no prediction is printed.

    uv run python scripts/paper/budget_and_external.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "scripts" / "paper" / "out"

DATASETS = ("hotpotqa", "2wikimultihopqa", "musique", "financebench")
JUDGE = "gpt-4-1_39118f"
SEED, N_BOOT = 42, 10000
AGENTIC, LOOP, NAIVE, SINGLE, CRAG = (
    "agentic_(react)",
    "loop_refinement",
    "naive_rag",
    "single-pass",
    "crag_replica",
)
AGENTIC6, LOOP10, IRCOT = "agentic_(budget=6)", "loop_(budget=10)", "ircot"
SHORT = {
    AGENTIC: "Agentic",
    LOOP: "Loop",
    NAIVE: "Naive",
    SINGLE: "Single-Pass",
    CRAG: "CRAG",
    AGENTIC6: "Agentic(budget=6)",
    LOOP10: "Loop(budget=10)",
    IRCOT: "IRCoT",
}
GRID_PIPES = (NAIVE, CRAG, SINGLE, LOOP, AGENTIC)
# (a, b, pre-specified) in the direction of the contrast a - b
F2 = (
    (AGENTIC, LOOP10, True),
    (AGENTIC6, LOOP, True),
    (LOOP10, LOOP, True),
    (AGENTIC6, AGENTIC, False),
)
F3 = ((AGENTIC, IRCOT, True), (LOOP, IRCOT, True))
# Output names of the two families. The short labels F2/F3 stay in the
# comparison_rng keys, so the bootstrap draws do not change.
FAMILY_NAME = {"F2": "matched_budget", "F3": "external_instance"}
F2_DATASETS = ("2wikimultihopqa", "financebench")
RUN_DIR = re.compile(
    r"^\d{8}_\d{6}_(rq1|budget_matched|external)_(hotpotqa|2wikimultihopqa|musique|financebench)"
    r"_n(200|150)_gpt-5-mini_k([123])$"
)
EXPECTED_DIRS = {"rq1": 12, "budget_matched": 6, "external": 12}


def holm(ps: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in input order."""
    m = len(ps)
    order = np.argsort(ps, kind="stable")
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adjusted[i] = running
    return adjusted


def sign_of(lo: float, hi: float) -> str:
    """'+'/'-' when the 95% CI excludes 0, else 'CI0' (no sign is claimed)."""
    tol = 1e-9
    return "+" if lo > tol else ("-" if hi < -tol else "CI0")


def descriptive_only(dataset: str, hop, metric: str) -> bool:
    """MuSiQue 4-hop and the FB judge are reported descriptively."""
    return (
        hop == "all"
        or (dataset == "musique" and hop == 4)
        or (dataset == "financebench" and metric == "judge")
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys and not k.startswith("_")]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


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

    logger.remove()

    run_dirs = sorted(p for p in results.iterdir() if p.is_dir() and RUN_DIR.match(p.name))
    found = {s: sum(RUN_DIR.match(p.name).group(1) == s for p in run_dirs) for s in EXPECTED_DIRS}
    if found != EXPECTED_DIRS:
        raise SystemExit(f"expected run directories {EXPECTED_DIRS}, found {found}")
    stage = Path(tempfile.mkdtemp(prefix="f2f3_stage_"))
    for p in run_dirs:
        (stage / p.name).symlink_to(p)
    (stage / "exclusions.json").symlink_to(results / "exclusions.json")
    try:
        records = [
            rec
            for prefix in EXPECTED_DIRS
            for rec in load_runs(stage, prefix=prefix, judge=JUDGE, require_paired_n=True)
        ]
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    refs = {ds: hop_reference(ds, code / "data" / "raw") for ds in DATASETS}
    # val[metric][(ds, pipe)][k][qid]
    metrics = ("f1", "judge", "metered", "cost")
    val: dict = {m: defaultdict(lambda: defaultdict(dict)) for m in metrics}
    hop_of: dict[str, dict[str, int]] = defaultdict(dict)
    raw_rows: dict[tuple, list[dict]] = defaultdict(list)
    missing = sum(rec.get("n_judge_missing", 0) for rec in records)
    if missing:
        raise SystemExit(f"{missing} rows without a {JUDGE} judge score")
    for rec in records:
        ds, pipe, k = rec["dataset"], rec["pipeline"], rec["repeat_index"]
        raw_rows[(ds, pipe, k)] = rec["rows"]
        for r in rec["rows"]:
            q = str(r["id"])
            hop_of[ds][q] = hop_for_row(r, refs[ds], f"{ds}/{pipe}")
            val["f1"][(ds, pipe)][k][q] = best_f1(r.get("prediction") or "", r)
            val["judge"][(ds, pipe)][k][q] = float(r["llm_judge"])
            val["metered"][(ds, pipe)][k][q] = float(r.get("metered_calls") or 0)
            val["cost"][(ds, pipe)][k][q] = float(r.get("cost_usd") or 0)

    def qids_of(ds: str, *pipes: str) -> list[str]:
        sets = [set(v) for p in pipes for v in val["f1"][(ds, p)].values()]
        return sorted(set.intersection(*sets))

    def arr(metric: str, ds: str, pipe: str, qids: list[str]) -> np.ndarray:
        runs = val[metric][(ds, pipe)]
        return np.array([np.mean([runs[k][q] for k in sorted(runs)]) for q in qids])

    def bins(ds: str, qids: list[str]) -> dict:
        b: dict = defaultdict(list)
        for q in qids:
            b[hop_of[ds][q]].append(q)
        b = dict(sorted(b.items()))
        if len(b) > 1:
            b["all"] = list(qids)
        return b

    def contrasts(family: str, pairs, datasets) -> list[dict]:
        rows = []
        for ds in datasets:
            for a_p, b_p, registered in pairs:
                common = qids_of(ds, a_p, b_p)
                for hop, qids in bins(ds, common).items():
                    for metric in ("f1", "judge"):
                        a, b = arr(metric, ds, a_p, qids), arr(metric, ds, b_p, qids)
                        key = (family, metric, ds, str(hop), a_p, b_p)
                        p, lo, hi, d = _paired_bootstrap(
                            a, b, comparison_rng(SEED, *key), n_boot=N_BOOT
                        )
                        diff = a - b
                        sd = float(diff.std(ddof=1))
                        rows.append(
                            {
                                "family": FAMILY_NAME[family],
                                "dataset": ds,
                                "hop": hop,
                                "metric": metric,
                                "contrast": f"{SHORT[a_p]} - {SHORT[b_p]}",
                                "registered": registered,
                                "descriptive": (not registered)
                                or descriptive_only(ds, hop, metric),
                                "n": len(qids),
                                "runs_a": f"k{min(val['f1'][(ds, a_p)])}-{max(val['f1'][(ds, a_p)])}",
                                "runs_b": f"k{min(val['f1'][(ds, b_p)])}-{max(val['f1'][(ds, b_p)])}",
                                "mean_a": float(a.mean()),
                                "mean_b": float(b.mean()),
                                "delta": d,
                                "ci_lo": lo,
                                "ci_hi": hi,
                                "p": p,
                                "cohen_d": d / sd if sd > 0 else float("nan"),
                                "sign": sign_of(lo, hi),
                                "metered_a": float(arr("metered", ds, a_p, qids).mean()),
                                "metered_b": float(arr("metered", ds, b_p, qids).mean()),
                                "cost_a": float(arr("cost", ds, a_p, qids).mean()),
                                "cost_b": float(arr("cost", ds, b_p, qids).mean()),
                                "_key": key,
                                "_pipes": (a_p, b_p),
                                "_qids": qids,
                            }
                        )
        for metric in ("f1", "judge"):
            g = [r for r in rows if r["metric"] == metric and not r["descriptive"]]
            for r, adj in zip(g, holm([x["p"] for x in g])):
                r["p_holm"] = adj
            for r in g:
                if 0.04 <= r["p_holm"] <= 0.06:
                    a_p, b_p = r["_pipes"]
                    r["p_recheck"] = _paired_bootstrap(
                        arr(metric, r["dataset"], a_p, r["_qids"]),
                        arr(metric, r["dataset"], b_p, r["_qids"]),
                        comparison_rng(SEED, *r["_key"]),
                        n_boot=10 * N_BOOT,
                    )[0]
            for r, adj in zip(g, holm([x.get("p_recheck", x["p"]) for x in g])):
                r["p_holm_final"] = adj
                r["holm_sig"] = adj < 0.05
            r_count = len(g)
            for r in rows:
                if r["metric"] == metric:
                    r["holm_m"] = r_count if not r["descriptive"] else ""
        return rows

    f2 = contrasts("F2", F2, F2_DATASETS)
    f3 = contrasts("F3", F3, DATASETS)
    write_csv(out / "budget_matched_contrasts.csv", f2)
    write_csv(out / "external_ircot_contrasts.csv", f3)

    # ---- levels per pipeline on the questions every pipeline of the table shares
    levels = []
    for table, pipes, datasets in (
        ("F2", GRID_PIPES + (LOOP10, AGENTIC6), F2_DATASETS),
        ("F3", GRID_PIPES + (IRCOT,), DATASETS),
    ):
        for ds in datasets:
            common = qids_of(ds, *pipes)
            for hop, qids in bins(ds, common).items():
                for p in pipes:
                    levels.append(
                        {
                            "table": FAMILY_NAME[table],
                            "dataset": ds,
                            "hop": hop,
                            "pipeline": SHORT[p],
                            "n": len(qids),
                            "runs": f"k{min(val['f1'][(ds, p)])}-{max(val['f1'][(ds, p)])}",
                            **{m: float(arr(m, ds, p, qids).mean()) for m in metrics},
                        }
                    )
    write_csv(out / "budget_pipeline_levels.csv", levels)

    # ---- how many calls Agentic(budget=6) made, against Loop(budget=10) and Agentic
    calls = []
    for ds in F2_DATASETS:
        for p in (AGENTIC6, LOOP10, AGENTIC, LOOP):
            for k in sorted(val["metered"][(ds, p)]):
                m = np.array(list(val["metered"][(ds, p)][k].values()))
                llm = np.array([float(r.get("llm_calls") or 0) for r in raw_rows[(ds, p, k)]])
                calls.append(
                    {
                        "dataset": ds,
                        "pipeline": SHORT[p],
                        "k": k,
                        "rows": len(m),
                        "metered_mean": float(m.mean()),
                        "metered_median": float(np.median(m)),
                        "metered_max": int(m.max()),
                        "rows_metered_over_6": int((m > 6).sum()),
                        "rows_metered_over_10": int((m > 10).sum()),
                        "llm_calls_mean": float(llm.mean()),
                        "llm_calls_max": int(llm.max()),
                    }
                )
    write_csv(out / "budget_six_call_cell.csv", calls)

    # ---- IRCoT JSONAdapter re-sends (structured_calls), per run
    fallback = []
    for ds in DATASETS:
        for k in sorted(val["metered"][(ds, IRCOT)]):
            rows = raw_rows[(ds, IRCOT, k)]
            llm = sum(int(r.get("llm_calls") or 0) for r in rows)
            met = sum(int(r.get("metered_calls") or 0) for r in rows)
            res = [
                sum(int(v or 0) for v in (r.get("structured_calls") or {}).values()) for r in rows
            ]
            cot = sum(sum(a.startswith("cot") for a in r.get("action_history") or []) for r in rows)
            outs = sum(sum(a == "output" for a in r.get("action_history") or []) for r in rows)
            if llm + sum(res) != met:
                raise SystemExit(f"{ds} k{k}: llm_calls + structured_calls != metered_calls")
            fallback.append(
                {
                    "dataset": ds,
                    "k": k,
                    "rows": len(rows),
                    "llm_calls": llm,
                    "resends": sum(res),
                    "metered_calls": met,
                    "resends_per_llm_call": sum(res) / llm,
                    "rows_with_resend": sum(x > 0 for x in res),
                    "cot_calls": cot,
                    "output_calls": outs,
                    "step_ratio_lower": max(sum(res) - outs, 0) / cot,
                    "step_ratio_upper": sum(res) / cot,
                    "cost_usd": sum(float(r.get("cost_usd") or 0) for r in rows),
                }
            )
    write_csv(out / "ircot_fallback.csv", fallback)

    # ---- decision rules T2 (budget-matched) and T4 (IRCoT), read on the dataset and 4-hop rows
    def row(rows, ds, hop, metric, contrast):
        return next(
            r
            for r in rows
            if (r["dataset"], r["hop"], r["metric"], r["contrast"]) == (ds, hop, metric, contrast)
        )

    verdicts = {"T2": [], "T4": []}
    for ds in F2_DATASETS:
        hop = "all" if len(bins(ds, qids_of(ds, AGENTIC, LOOP10))) > 1 else 1
        for metric in ("f1", "judge"):
            r1 = row(f2, ds, hop, metric, "Agentic - Loop(budget=10)")
            r2 = row(f2, ds, hop, metric, "Agentic(budget=6) - Loop")
            verdicts["T2"].append(
                {
                    "dataset": ds,
                    "hop": hop,
                    "metric": metric,
                    "agentic_minus_loop10_ci_has_0": r1["sign"] == "CI0",
                    "agentic6_minus_loop_ci_has_0": r2["sign"] == "CI0",
                    "rule_met": r1["sign"] == "CI0" or r2["sign"] == "CI0",
                }
            )
    for ds in ("2wikimultihopqa", "musique"):
        for metric in ("f1", "judge"):
            r = row(f3, ds, 4, metric, "Agentic - IRCoT")
            verdicts["T4"].append(
                {
                    "dataset": ds,
                    "hop": 4,
                    "metric": metric,
                    "delta": r["delta"],
                    "ci": [r["ci_lo"], r["ci_hi"]],
                    "descriptive": r["descriptive"],
                    "ircot_beats_agentic": r["sign"] == "-",
                }
            )
    (out / "verdicts.json").write_text(json.dumps(verdicts, indent=2) + "\n", encoding="utf-8")
    summary = {
        "runs": found,
        "judge": JUDGE,
        "f2_rows": len(f2),
        "f3_rows": len(f3),
        "holm_m": {
            f"{FAMILY_NAME[fam]}_{m}": next(
                (r["holm_m"] for r in rows if r["metric"] == m and r["holm_m"] != ""), 0
            )
            for fam, rows in (("F2", f2), ("F3", f3))
            for m in ("f1", "judge")
        },
        "rechecked_cells": sum("p_recheck" in r for r in f2 + f3),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
