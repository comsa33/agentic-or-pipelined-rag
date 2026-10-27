#!/usr/bin/env python3
"""Judge numbers of the uncontrolled runs under gpt-4.1 next to gpt-4.1-nano.

1. FinanceBench visibility sweep (Gemini, n=150, table tab:threat_budget): per
   point, F1 and the judge score under both judges, with Agentic-Naive and
   Loop-Naive paired differences (delta, 95% CI, standard two-sided p). The
   published_p column holds the p an earlier version of the table printed for
   Agentic-Naive.
2. Judge swap (Section 5.4): per model-workload cell of the uncontrolled grid,
   whether the sign of the Agentic-Loop judge difference agrees between the
   reference judge and minimax-m3, with nano and gpt-4.1 each as reference.
   Same rule as `experiments.analysis.judge_swap`: only verdicts with
   judge_status "ok" count, rates are plain means per configuration.

Reads the stored verdicts (`*_judged_*.jsonl`) of data/results/paper,
data/results-openweight/final and the July sweep run in data/results-topk.
No API call. Writes judge41_uncontrolled_table6_1_sweep_by_judge.csv and
judge41_uncontrolled_judge_swap_by_reference.csv to
scripts/paper/out/uncontrolled_judge/.

    uv run python scripts/paper/uncontrolled_judge.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
REPO = Path(__file__).resolve().parents[2]
ap.add_argument("--code-repo", type=Path, default=REPO)
ap.add_argument(
    "--out-dir", type=Path, default=REPO / "scripts" / "paper" / "out" / "uncontrolled_judge"
)
args = ap.parse_args()
code = args.code_repo.resolve()
sys.path.insert(0, str(code))

from experiments.analysis.hop_level import _paired_bootstrap  # noqa: E402
from experiments.analysis.judge_swap import PIPELINES, _rates  # noqa: E402
from experiments.analysis.runs import best_f1, comparison_rng  # noqa: E402

PAPER = code / "data" / "results" / "paper"
OPENWEIGHT = code / "data" / "results-openweight" / "final"
SWEEP = (
    code
    / "data"
    / "results-topk"
    / "20260730_121017_topk_sensitivity_financebench_n150_gemini-3.1-flash-lite"
)
POINTS = (
    (5, "5 / 5"),
    (10, "10 / 10"),
    (20, "20 / 20"),
    (30, "30 / 30"),
    (50, "50 / 30 (reported)"),
)
GPT41 = "gpt-4-1_39118f"
NANO_SWEEP = "gpt-4-1-nano_7bcf05"
SEED, N_BOOT = 42, 10000

# p (Agentic-Naive) as an earlier version of the table printed it.
PUBLISHED_P_AN = {
    5: {"f1": "<.001", "nano": "<.001"},
    10: {"f1": "<.001", "nano": "<.001"},
    20: {"f1": ".130", "nano": ".193"},
    30: {"f1": ".002", "nano": ".034"},
    50: {"f1": ".055", "nano": ".121"},
}


def _rows(path: Path) -> dict[str, dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {str(r["id"]): r for r in (json.loads(x) for x in lines if x.strip())}


def sweep() -> list[dict]:
    """One row per (point, metric, contrast): means, delta, 95% CI, two-sided p.

    The test is `_paired_bootstrap` (10,000 resamples, two-sided), one rng per
    comparison from `runs.comparison_rng(42, "sweep", metric, point, contrast)`.
    """
    out = []
    for k, label in POINTS:
        base = {
            p: _rows(SWEEP / f"topk_sensitivity_{p}_k{k}.jsonl")
            for p in ("naive", "loop", "agentic")
        }
        ids = sorted(base["naive"])
        if any(sorted(v) != ids for v in base.values()):
            raise SystemExit(f"id sets differ at k{k}")
        scores = {
            "f1": {
                p: np.array([best_f1(base[p][q].get("prediction") or "", base[p][q]) for q in ids])
                for p in base
            }
        }
        for name, tag in (("nano", NANO_SWEEP), ("gpt41", GPT41)):
            judged = {
                p: _rows(SWEEP / f"topk_sensitivity_{p}_k{k}_judged_{tag}.jsonl") for p in base
            }
            bad = sum(
                r.get("judge_status", "ok") != "ok" for v in judged.values() for r in v.values()
            )
            if bad:
                raise SystemExit(f"{bad} non-ok verdicts at k{k} under {name}")
            if any(sorted(v) != ids for v in judged.values()):
                raise SystemExit(f"judged id sets differ at k{k} under {name}")
            scores[name] = {
                p: np.array([float(judged[p][q]["llm_judge"]) for q in ids]) for p in base
            }
        for metric, s in scores.items():
            for a_p, contrast in (("agentic", "A-N"), ("loop", "L-N")):
                rng = comparison_rng(SEED, "sweep", metric, str(k), contrast)
                p, lo, hi, d = _paired_bootstrap(s[a_p], s["naive"], rng, N_BOOT)
                out.append(
                    {
                        "point": label,
                        "metric": metric,
                        "contrast": contrast,
                        "n": len(ids),
                        "naive": float(s["naive"].mean()),
                        "loop": float(s["loop"].mean()),
                        "agentic": float(s["agentic"].mean()),
                        "delta": d,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "p": p,
                        "published_p": PUBLISHED_P_AN[k].get(metric, "")
                        if contrast == "A-N"
                        else "",
                    }
                )
    return out


def swap() -> list[dict]:
    dirs = sorted(PAPER.glob("*rq1_*")) + sorted(OPENWEIGHT.glob("*"))
    out = []
    for d in dirs:
        nano, _ = _rates(d, "_judged.jsonl")
        if not nano:
            nano, _ = _rates(d, "_judged_gpt-4-1-nano.jsonl")
        g41, bad41 = _rates(d, f"_judged_{GPT41}.jsonl")
        mm, badmm = _rates(d, "_judged_minimax-m3_*.jsonl")
        if not (nano and g41 and mm):
            raise SystemExit(f"missing judge files in {d}")
        judges = (("nano", nano), ("gpt41", g41), ("minimax", mm))
        gap = {n: r["agentic_(react)"] - r["loop_refinement"] for n, r in judges}
        # The March directories carry a timestamp; the gpt-oss ones are named by workload.
        m = re.match(r"\d{8}_\d{6}_rq1_([a-z0-9]+)_n\d+_(.+)$", d.name)
        cell = f"{m.group(1)} / {m.group(2)}" if m else f"{d.name} / gpt-oss-120b"
        out.append(
            {
                "cell": cell,
                "unusable_gpt41": bad41,
                "unusable_minimax": badmm,
                **{f"{n}_{p}": r.get(p) for n, r in judges for p in PIPELINES},
                "gap_nano": gap["nano"],
                "gap_gpt41": gap["gpt41"],
                "gap_minimax": gap["minimax"],
                "nano_vs_minimax_same": (gap["nano"] > 0) == (gap["minimax"] > 0),
                "gpt41_vs_minimax_same": (gap["gpt41"] > 0) == (gap["minimax"] > 0),
                "nano_vs_gpt41_same": (gap["nano"] > 0) == (gap["gpt41"] > 0),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


for tree in (PAPER, OPENWEIGHT, SWEEP):
    if not tree.is_dir():
        sys.exit(f"{tree.relative_to(code)} not present: these tables need the uncontrolled runs")

s, w = sweep(), swap()
args.out_dir.mkdir(parents=True, exist_ok=True)
write_csv(args.out_dir / "judge41_uncontrolled_table6_1_sweep_by_judge.csv", s)
write_csv(args.out_dir / "judge41_uncontrolled_judge_swap_by_reference.csv", w)
summary = {
    "sweep_points": len(s),
    "swap_cells": len(w),
    "sign_kept_nano_vs_minimax": sum(r["nano_vs_minimax_same"] for r in w),
    "sign_kept_gpt41_vs_minimax": sum(r["gpt41_vs_minimax_same"] for r in w),
    "sign_kept_nano_vs_gpt41": sum(r["nano_vs_gpt41_same"] for r in w),
}
print(json.dumps(summary, indent=2))
