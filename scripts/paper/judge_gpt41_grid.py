#!/usr/bin/env python3
"""Controlled-grid judge contrasts under gpt-4.1-nano and gpt-4.1, side by side.

Reads the 24 controlled-grid runs (2 models x 4 datasets x k1-3) twice, once
per judge tag, and reports:

1. per (model, dataset, hop) cell the paired judge contrasts Agentic-Loop,
   Agentic-Naive, Loop-Naive and Single-Pass-Naive under both judges, with
   sign changes and CI-crossing changes flagged;
2. the Sonnet 2Wiki 4-hop cell: gpt-4.1's verdicts on the human-labelled
   rows, against the human labels;
3. model-family accuracy levels (Sonnet - gpt-5-mini) per dataset and
   pipeline on the questions both families kept (M+), under both judges.

Statistics follow the controlled-grid script: per-question
mean over k, paired bootstrap (10,000, seed 42) from `hop_level._paired_bootstrap`
with `runs.comparison_rng`. The Agentic-Loop key is the one the controlled-grid
script uses, so the nano rows reproduce its Judge table. Holm runs within
(judge, contrast) over the non-descriptive hop cells. No API call.

--hedge is the 167 human-labelled rows, published with the supplementary
data as judge41_vs_human_hedge167_rows.csv. --published defaults to the
f1_agentic_minus_loop.csv of controlled_grid_analysis.py (run it first).

    uv run python scripts/paper/judge_gpt41_grid.py --hedge <judge41_vs_human_hedge167_rows.csv>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATASETS = ("hotpotqa", "2wikimultihopqa", "musique", "financebench")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "scripts" / "paper" / "out"
MODELS = ("gpt-5-mini", "claude-sonnet-5-5")
AGENTIC, LOOP, CRAG, NAIVE, SINGLE = (
    "agentic_(react)",
    "loop_refinement",
    "crag_replica",
    "naive_rag",
    "single-pass",
)
PIPELINES = (NAIVE, SINGLE, LOOP, AGENTIC, CRAG)
CONTRASTS = ((AGENTIC, LOOP), (AGENTIC, NAIVE), (LOOP, NAIVE), (SINGLE, NAIVE))
SHORT = {AGENTIC: "Agentic", LOOP: "Loop", NAIVE: "Naive", SINGLE: "Single-Pass", CRAG: "CRAG"}
JUDGES = {"nano": "gpt-4-1-nano_7bcf05", "gpt41": "gpt-4-1_39118f"}
RUN_DIR = re.compile(
    r"^(\d{8}_\d{6})_rq1_(hotpotqa|2wikimultihopqa|musique|financebench)"
    r"_n(200|150)_(gpt-5-mini|claude-sonnet-5-5)_k([123])$"
)
SEED, N_BOOT = 42, 10000


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


def sign_of(delta: float, lo: float, hi: float) -> str:
    """'+'/'-' when the 95% CI excludes 0, else 'CI0' (no sign is claimed)."""
    tol = 1e-9
    if lo > tol:
        return "+"
    if hi < -tol:
        return "-"
    return "CI0"


def descriptive_only(dataset: str, hop) -> bool:
    """MuSiQue 4-hop and the FB judge are reported descriptively."""
    return (dataset == "musique" and hop == 4) or dataset == "financebench"


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
    ap.add_argument("--out", type=Path, default=OUT / "judge_gpt41_grid")
    ap.add_argument("--hedge", type=Path, required=True, help="hedge167_rows.csv (human labels)")
    ap.add_argument(
        "--published",
        type=Path,
        default=OUT / "controlled_grid" / "f1_agentic_minus_loop.csv",
        help="f1_agentic_minus_loop.csv to reproduce",
    )
    args = ap.parse_args()

    code = args.code_repo.resolve()
    results = code / "data" / "results-campaign"
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(code))

    from loguru import logger

    from experiments.analysis.hop_level import _paired_bootstrap
    from experiments.analysis.runs import comparison_rng, load_runs
    from experiments.difficulty import hop_for_row, hop_reference

    logger.remove()

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
    # score[j][(model, ds, pipe)][k][qid]
    score: dict = {j: defaultdict(lambda: defaultdict(dict)) for j in JUDGES}
    hop_of: dict[tuple, dict[str, int]] = defaultdict(dict)
    missing = Counter()
    for j, recs in records.items():
        for rec in recs:
            missing[j] += rec.get("n_judge_missing", 0)
            key = (rec["model"], rec["dataset"], rec["pipeline"])
            for r in rec["rows"]:
                qid = str(r["id"])
                hop_of[(rec["model"], rec["dataset"])][qid] = hop_for_row(
                    r, refs[rec["dataset"]], "/".join(key)
                )
                score[j][key][rec["repeat_index"]][qid] = float(r["llm_judge"])
    if any(missing.values()):
        raise SystemExit(f"rows without a judge score: {dict(missing)}")
    for key in score["nano"]:
        for k in score["nano"][key]:
            if set(score["nano"][key][k]) != set(score["gpt41"][key][k]):
                raise SystemExit(f"id sets differ between judges at {key} k{k}")

    def ks_of(j, model, ds, *pipes) -> list[int]:
        return sorted(set.intersection(*(set(score[j][(model, ds, p)]) for p in pipes)))

    def arr(j, model, ds, pipe, ks, qids) -> np.ndarray:
        return np.array([np.mean([score[j][(model, ds, pipe)][k][q] for k in ks]) for q in qids])

    def boot(a, b, *key) -> dict:
        p, lo, hi, obs = _paired_bootstrap(a, b, comparison_rng(SEED, *key), n_boot=N_BOOT)
        return {
            "n": len(a),
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "delta": obs,
            "ci_lo": lo,
            "ci_hi": hi,
            "p": p,
            "sign": sign_of(obs, lo, hi),
        }

    def bins(model, ds) -> dict:
        out_bins: dict = defaultdict(list)
        for q, h in sorted(hop_of[(model, ds)].items()):
            out_bins[h].append(q)
        out_bins = dict(sorted(out_bins.items()))
        if len(out_bins) > 1:
            out_bins["all"] = sorted(hop_of[(model, ds)])
        return out_bins

    # ---- 1. contrasts per cell, both judges -----------------------------
    contrast_rows: list[dict] = []
    for model in MODELS:
        for ds in DATASETS:
            for hop, qids in bins(model, ds).items():
                for a_p, b_p in CONTRASTS:
                    row = {
                        "model": model,
                        "dataset": ds,
                        "hop": hop,
                        "contrast": f"{SHORT[a_p]}-{SHORT[b_p]}",
                        "_pipes": (a_p, b_p),
                        "descriptive": hop == "all" or descriptive_only(ds, hop),
                    }
                    for j in JUDGES:
                        ks = ks_of(j, model, ds, a_p, b_p)
                        s = boot(
                            arr(j, model, ds, a_p, ks, qids),
                            arr(j, model, ds, b_p, ks, qids),
                            "F1",
                            "judge",
                            model,
                            ds,
                            str(hop),
                            a_p,
                            b_p,
                        )
                        row["runs"] = f"k{ks[0]}-{ks[-1]}"
                        row.update({f"{j}_{k}": v for k, v in s.items() if k != "n"})
                        row["n"] = s["n"]
                    contrast_rows.append(row)
    for j in JUDGES:
        fam: dict = defaultdict(list)
        for r in contrast_rows:
            if not r["descriptive"]:
                fam[r["contrast"]].append(r)
        for g in fam.values():
            for r, adj in zip(g, holm([x[f"{j}_p"] for x in g])):
                r[f"{j}_p_holm"] = adj
            # A cell with p_holm in [0.04, 0.06] is redrawn 10x under its own key.
            for r in g:
                if 0.04 <= r[f"{j}_p_holm"] <= 0.06:
                    a_p, b_p = r["_pipes"]
                    ks = ks_of(j, r["model"], r["dataset"], a_p, b_p)
                    qids = bins(r["model"], r["dataset"])[r["hop"]]
                    r[f"{j}_p_recheck"] = _paired_bootstrap(
                        arr(j, r["model"], r["dataset"], a_p, ks, qids),
                        arr(j, r["model"], r["dataset"], b_p, ks, qids),
                        comparison_rng(
                            SEED, "F1", "judge", r["model"], r["dataset"], str(r["hop"]), a_p, b_p
                        ),
                        n_boot=10 * N_BOOT,
                    )[0]
            for r, adj in zip(g, holm([x.get(f"{j}_p_recheck", x[f"{j}_p"]) for x in g])):
                r[f"{j}_p_holm_final"] = adj
    for r in contrast_rows:
        sn, sg = r["nano_sign"], r["gpt41_sign"]
        r["point_sign_flip"] = (r["nano_delta"] > 0) != (r["gpt41_delta"] > 0) and r[
            "nano_delta"
        ] * r["gpt41_delta"] != 0
        r["ci_sign_change"] = "" if sn == sg else f"{sn}->{sg}"
        if not r["descriptive"]:
            hn, hg = r["nano_p_holm_final"] < 0.05, r["gpt41_p_holm_final"] < 0.05
            r["holm_change"] = "" if hn == hg else ("sig->ns" if hn else "ns->sig")
    write_csv(out / "contrasts_by_judge.csv", contrast_rows)

    # reproduce the published nano Agentic-Loop Judge rows
    pub = {
        (r["model"], r["dataset"], r["hop"]): r
        for r in csv.DictReader(open(args.published, encoding="utf-8"))
        if r["metric"] == "judge"
    }
    worst = 0.0
    for r in contrast_rows:
        if r["contrast"] == "Agentic-Loop" and r["hop"] != "all":
            p = pub[(r["model"], r["dataset"], str(r["hop"]))]
            for mine, theirs in (
                ("nano_delta", "delta"),
                ("nano_ci_lo", "ci_lo"),
                ("nano_ci_hi", "ci_hi"),
            ):
                worst = max(worst, abs(r[mine] - float(p[theirs])))
            if p["p_holm_final"]:
                worst = max(worst, abs(r["nano_p_holm_final"] - float(p["p_holm_final"])))
    if worst > 1e-9:
        raise SystemExit(
            f"nano Agentic-Loop does not reproduce the published table (max diff {worst})"
        )

    # ---- 2. Sonnet 2Wiki 4-hop cell: gpt-4.1 against the human labels ----
    hedge = list(csv.DictReader(open(args.hedge, encoding="utf-8")))
    r4 = []
    for h in hedge:
        if h["dataset"] != "2wikimultihopqa" or h["hop"] != "4":
            continue
        k = int(h["k"])
        grid = score["gpt41"][("claude-sonnet-5-5", "2wikimultihopqa", h["pipeline"])][k].get(
            h["id"]
        )
        r4.append(
            {
                "row_no": h["row_no"],
                "pipeline": h["pipeline"],
                "k": k,
                "marker": h["refusal_marker"],
                "human": h["human_1"],
                "nano": h["judge_nano"],
                "gpt41_grid": "" if grid is None else int(grid),
                "gpt41_hedge_run": h["judge_gpt41"],
            }
        )
    write_csv(out / "r4_rows.csv", r4)
    tally = Counter()
    for r in r4:
        tally[(r["nano"], r["human"], str(r["gpt41_grid"]))] += 1
    rerun_diff = sum(1 for r in r4 if str(r["gpt41_grid"]) != r["gpt41_hedge_run"])

    # ---- 3. family accuracy levels on common questions (M+) --------------
    family_rows: list[dict] = []
    for ds in DATASETS:
        common = sorted(set(hop_of[(MODELS[0], ds)]) & set(hop_of[(MODELS[1], ds)]))
        for pipe in PIPELINES:
            row = {"dataset": ds, "pipeline": SHORT[pipe], "n": len(common)}
            for j in JUDGES:
                ks = ks_of(j, MODELS[0], ds, pipe)
                if ks != ks_of(j, MODELS[1], ds, pipe):
                    raise SystemExit(f"k sets differ between models at {ds}/{pipe}")
                s = boot(
                    arr(j, MODELS[1], ds, pipe, ks, common),
                    arr(j, MODELS[0], ds, pipe, ks, common),
                    "family",
                    "judge",
                    ds,
                    pipe,
                )
                row["runs"] = f"k{ks[0]}-{ks[-1]}"
                row.update(
                    {
                        f"{j}_sonnet": s["mean_a"],
                        f"{j}_gpt5mini": s["mean_b"],
                        f"{j}_delta": s["delta"],
                        f"{j}_ci_lo": s["ci_lo"],
                        f"{j}_ci_hi": s["ci_hi"],
                        f"{j}_sign": s["sign"],
                    }
                )
            family_rows.append(row)
    write_csv(out / "family_levels_by_judge.csv", family_rows)

    summary = {
        "run_dirs": len(run_dirs),
        "judges": JUDGES,
        "nano_agentic_loop_reproduces_published": True,
        "contrast_cells": len(contrast_rows),
        "r4_rows": len(r4),
        "r4_tally_nano_human_gpt41": {"/".join(k): v for k, v in sorted(tally.items())},
        "r4_grid_vs_hedge_rerun_gpt41_differ": rerun_diff,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
