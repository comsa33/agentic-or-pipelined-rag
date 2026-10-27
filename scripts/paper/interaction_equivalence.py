#!/usr/bin/env python3
"""Family x pattern interaction and gpt-5-mini equivalence on the controlled grid (no API call).

1. Model family x pattern interaction. Per cell (workload, hop bin) and on the
   questions both families kept (M+: each dataset minus the union of the two
   families' exclusions, as in controlled_grid_analysis.py), the
   per-question difference Delta_q = Agentic - Loop (mean over k1-3 of each
   configuration) is computed for each family, and the interaction is
   mean(Delta_sonnet,q - Delta_gpt5mini,q), a paired bootstrap over questions
   (hop_level._paired_bootstrap, 10,000 draws, seed 42, runs.comparison_rng
   with its own key). Metrics: F1 (best over references) and the gpt-4.1
   judge (gpt-4-1_39118f). Holm within metric over the non-descriptive cells
   (MuSiQue 4-hop and the FinanceBench judge are descriptive);
   p_holm in [0.04, 0.06] is redrawn with 10x the draws.
   Each family's own M+ Delta is checked against grid_f1_agentic_minus_loop_common_questions.csv
   (F1) and judge_grid_common_questions.csv (judge).
2. Minimum detectable effect and equivalence for gpt-5-mini, on the main
   question set of the main table (the family's own exclusions only) and with the
   bootstrap draws of that table itself (same comparison key, so the 95%
   interval reproduces the table). MDE = (z_.975 + z_.80) x bootstrap SE.
   TOST at alpha .05 with margins +-0.03 and +-0.05: equivalent when the 90%
   bootstrap interval lies inside (-margin, +margin); p_TOST is the larger of
   the two one-sided bootstrap shares beyond the margins.

Writes grid_interaction.csv and grid_mde_gpt5mini.csv to
scripts/paper/out/robustness/. Reads the outputs of controlled_grid_analysis.py
and mplus_gpt41.py (run them first). Reads only.

    uv run python scripts/paper/interaction_equivalence.py
"""

from __future__ import annotations

import argparse
import csv
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
from controlled_grid_analysis import (  # noqa: E402
    AGENTIC,
    DATASETS,
    LOOP,
    N_BOOT,
    RUN_DIR,
    SEED,
    descriptive_only,
    holm,
    sign_of,
    write_csv,
)

JUDGE = "gpt-4-1_39118f"
GPT, SONNET = "gpt-5-mini", "claude-sonnet-5-5"
Z_975, Z_80 = 1.959963984540054, 0.8416212335729143


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--code-repo", type=Path, default=REPO)
    ap.add_argument("--out", type=Path, default=OUT / "robustness")
    args = ap.parse_args()
    code = args.code_repo.resolve()
    results = code / "data" / "results-campaign"
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(code))

    from loguru import logger

    from experiments.analysis.hop_level import _paired_bootstrap
    from experiments.analysis.refusal_analysis import _prediction_text
    from experiments.analysis.runs import best_f1, comparison_rng, load_runs
    from experiments.difficulty import hop_for_row, hop_reference

    logger.remove()

    run_dirs = sorted(p for p in results.iterdir() if p.is_dir() and RUN_DIR.match(p.name))
    if len(run_dirs) != 24:
        raise SystemExit(f"expected 24 run directories, found {len(run_dirs)}")
    stage = Path(tempfile.mkdtemp(prefix="interaction_stage_"))
    try:
        for p in run_dirs:
            (stage / p.name).symlink_to(p)
        (stage / "exclusions.json").symlink_to(results / "exclusions.json")
        records = load_runs(stage, prefix="rq1", judge=JUDGE, require_paired_n=True)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    if sum(r.get("n_judge_missing", 0) for r in records):
        raise SystemExit(f"rows without a {JUDGE} verdict")

    refs = {ds: hop_reference(ds, code / "data" / "raw") for ds in DATASETS}
    val: dict = defaultdict(
        lambda: defaultdict(dict)
    )  # (model, ds, pipe) -> k -> qid -> {f1, judge}
    hop_of: dict = {}
    for rec in records:
        model, ds, pipe, k = rec["model"], rec["dataset"], rec["pipeline"], rec["repeat_index"]
        if pipe not in (AGENTIC, LOOP):
            continue
        for r in rec["rows"]:
            q = str(r["id"])
            hop_of[(ds, q)] = hop_for_row(r, refs[ds], f"{model}/{ds}/{pipe}")
            val[(model, ds, pipe)][k][q] = {
                "f1": best_f1(_prediction_text(r.get("prediction")), r),
                "judge": float(r["llm_judge"]),
            }

    def qset(model: str, ds: str) -> set[str]:
        by_k = val[(model, ds, AGENTIC)]
        return set.intersection(*(set(v) for v in by_k.values()))

    def delta(model: str, ds: str, qids: list[str], metric: str) -> np.ndarray:
        ks = sorted(val[(model, ds, AGENTIC)])
        a = np.array([np.mean([val[(model, ds, AGENTIC)][k][q][metric] for k in ks]) for q in qids])
        b = np.array([np.mean([val[(model, ds, LOOP)][k][q][metric] for k in ks]) for q in qids])
        return a - b

    def bins(qids: set[str], ds: str) -> dict[int, list[str]]:
        out_b: dict = defaultdict(list)
        for q in sorted(qids):
            out_b[hop_of[(ds, q)]].append(q)
        return dict(sorted(out_b.items()))

    # ---- 1. interaction on M+ -------------------------------------------
    rows = []
    for ds in DATASETS:
        common = qset(GPT, ds) & qset(SONNET, ds)
        for hop, qids in bins(common, ds).items():
            for metric in ("f1", "judge"):
                ds_, dg = delta(SONNET, ds, qids, metric), delta(GPT, ds, qids, metric)
                key = ("interaction", metric, ds, str(hop))
                p, lo, hi, obs = _paired_bootstrap(
                    ds_, dg, comparison_rng(SEED, *key), n_boot=N_BOOT
                )
                rows.append(
                    {
                        "metric": metric,
                        "dataset": ds,
                        "hop": hop,
                        "n": len(qids),
                        "delta_sonnet": float(ds_.mean()),
                        "delta_gpt5mini": float(dg.mean()),
                        "interaction": obs,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "p": p,
                        "descriptive": descriptive_only(ds, hop, metric),
                        "_a": ds_,
                        "_b": dg,
                        "_key": key,
                    }
                )
    for metric in ("f1", "judge"):
        g = [r for r in rows if r["metric"] == metric and not r["descriptive"]]
        for r, adj in zip(g, holm([x["p"] for x in g])):
            r["p_holm"] = adj
        for r in g:
            if 0.04 <= r["p_holm"] <= 0.06:
                r["p_recheck"] = _paired_bootstrap(
                    r["_a"], r["_b"], comparison_rng(SEED, *r["_key"]), n_boot=10 * N_BOOT
                )[0]
        for r, adj in zip(g, holm([x.get("p_recheck", x["p"]) for x in g])):
            r["p_holm_final"] = adj
            r["holm_sig"] = adj < 0.05
            r["holm_family_size"] = len(g)
    for r in rows:
        r["sign"] = sign_of(r["interaction"], r["ci_lo"], r["ci_hi"])

    # check each family's M+ Delta against the published M+ tables
    mplus = {}
    with open(OUT / "controlled_grid" / "grid_f1_agentic_minus_loop_common_questions.csv") as fh:
        for r in csv.DictReader(fh):
            if r["metric"] == "f1":
                mplus[("f1", r["model"], r["dataset"], int(r["hop"]))] = (
                    float(r["delta"]),
                    int(r["n"]),
                )
    with open(OUT / "judge_gpt41_grid" / "judge_grid_common_questions.csv") as fh:
        for r in csv.DictReader(fh):
            mplus[("judge", r["model"], r["dataset"], int(r["hop"]))] = (
                float(r["mplus_judge_delta"]),
                int(r["mplus_n"]),
            )
    checked = 0
    for r in rows:
        for model, col in ((SONNET, "delta_sonnet"), (GPT, "delta_gpt5mini")):
            d, n = mplus[(r["metric"], model, r["dataset"], r["hop"])]
            if n != r["n"] or abs(d - r[col]) > 1e-9:
                raise SystemExit(
                    f"M+ mismatch {r['metric']} {model} {r['dataset']} {r['hop']}: {r[col]} vs {d}"
                )
            checked += 1
    write_csv(out / "grid_interaction.csv", rows)

    # ---- 2. MDE and TOST for gpt-5-mini, main set ------------------------
    published = {}
    with open(OUT / "controlled_grid" / "grid_f1_agentic_minus_loop.csv") as fh:
        for r in csv.DictReader(fh):
            if r["model"] == GPT and r["metric"] == "f1":
                published[(r["dataset"], int(r["hop"]))] = (float(r["ci_lo"]), float(r["ci_hi"]))
    eq_rows = []
    for ds in DATASETS:
        for hop, qids in bins(qset(GPT, ds), ds).items():
            for metric in ("f1", "judge"):
                d = delta(GPT, ds, qids, metric)
                key = ("F1", metric, GPT, ds, str(hop), AGENTIC, LOOP)  # the main table's key
                rng = comparison_rng(SEED, *key)
                idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
                boot = d[idx].mean(axis=1)
                lo95, hi95 = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
                if metric == "f1":
                    plo, phi = published[(ds, hop)]
                    if abs(plo - lo95) > 1e-12 or abs(phi - hi95) > 1e-12:
                        raise SystemExit(f"95% CI does not reproduce the main table for {ds} {hop}")
                se = float(boot.std(ddof=1))
                row = {
                    "metric": metric,
                    "dataset": ds,
                    "hop": hop,
                    "n": len(qids),
                    "delta": float(d.mean()),
                    "ci95_lo": lo95,
                    "ci95_hi": hi95,
                    "ci90_lo": float(np.percentile(boot, 5)),
                    "ci90_hi": float(np.percentile(boot, 95)),
                    "boot_se": se,
                    "mde_80": (Z_975 + Z_80) * se,
                    "descriptive": descriptive_only(ds, hop, metric),
                }
                for m in (0.03, 0.05):
                    p_tost = max(float(np.mean(boot <= -m)), float(np.mean(boot >= m)))
                    row[f"tost_p_{m:.2f}"] = p_tost
                    row[f"equivalent_{m:.2f}"] = row["ci90_lo"] > -m and row["ci90_hi"] < m
                eq_rows.append(row)
    write_csv(out / "grid_mde_gpt5mini.csv", eq_rows)
    print(
        json.dumps(
            {
                "interaction_rows": len(rows),
                "mplus_deltas_checked": checked,
                "holm_sig": [
                    (r["metric"], r["dataset"], r["hop"]) for r in rows if r.get("holm_sig")
                ],
                "equivalence_rows": len(eq_rows),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
