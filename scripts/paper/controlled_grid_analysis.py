#!/usr/bin/env python3
"""Controlled-grid analysis: the Agentic (ReAct) minus Loop contrasts, refusal markers, stability, the refusal-pattern recount and cost columns.

Reads the campaign runs of gpt-5-mini and claude-sonnet-5-5 and makes no API
call. Statistics come from this repository's analysis modules
(`hop_level._paired_bootstrap`, `runs.comparison_rng`, `refusal_analysis`,
`exclusions`); this script adds the per-question k-mean, Holm, Cohen's d, the
stability verdict and the tables. Output file names keep the analysis-family
prefixes of the supplementary data (f1_, f5_, f6_).

    uv run python scripts/paper/controlled_grid_analysis.py

Writes to scripts/paper/out/controlled_grid/. Only run directories named
`<stamp>_rq1_<dataset>_n<200|150>_<model>_k<1|2|3>` are read (the n2 smoke
directories share the name pattern and are left out). `checkpoints/` is never
opened. Runs are staged as symlinks so that `runs.load_runs` sees nothing else.

The refusal-axis table adds reference rows from the uncontrolled trees
data/results/paper and data/results-openweight/final when they are present.
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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATASETS = ("hotpotqa", "2wikimultihopqa", "musique", "financebench")
MODELS = ("gpt-5-mini", "claude-sonnet-5-5")
AGENTIC, LOOP, CRAG = "agentic_(react)", "loop_refinement", "crag_replica"
PIPELINES = ("naive_rag", "single-pass", LOOP, AGENTIC, CRAG)
RUN_DIR = re.compile(
    r"^(\d{8}_\d{6})_rq1_(hotpotqa|2wikimultihopqa|musique|financebench)"
    r"_n(200|150)_(gpt-5-mini|claude-sonnet-5-5)_k([123])$"
)
JUDGE_TAG = "gpt-4-1-nano_7bcf05"
SEED, N_BOOT = 42, 10000
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "scripts" / "paper" / "out"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
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
    tol = 1e-9  # a bound this close to 0 is floating-point noise, not a sign
    if lo > tol:
        return "+"
    if hi < -tol:
        return "-"
    return "CI0"


def fmt(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{nd}f}"


def fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields and not k.startswith("_"):
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _med(values: list) -> float:
    vals = [float(v) for v in values if v is not None]
    return float(np.median(vals)) if vals else float("nan")


def descriptive_only(dataset: str, hop: int, metric: str) -> bool:
    """MuSiQue 4-hop (n=32) and the FinanceBench judge are reported descriptively."""
    return (dataset == "musique" and hop == 4) or (dataset == "financebench" and metric == "judge")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--code-repo", type=Path, default=REPO)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=OUT / "controlled_grid")
    ap.add_argument("--judge", default=JUDGE_TAG)
    args = ap.parse_args()

    code = args.code_repo.resolve()
    results = (args.results_dir or code / "data" / "results-campaign").resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(code))

    from loguru import logger

    from experiments.analysis.hop_level import _paired_bootstrap
    from experiments.analysis.refusal_analysis import (
        REFUSAL_PATTERNS,
        RefusalAnalyzer,
        _prediction_text,
        is_refusal,
    )
    from experiments.analysis.runs import best_f1, comparison_rng, load_runs
    from experiments.difficulty import hop_for_row, hop_reference
    from experiments.exclusions import (
        excluded_ids,
        load_exclusions,
        refusal_ids,
    )

    warnings: list[str] = []
    logger.remove()
    logger.add(lambda m: warnings.append(str(m).strip()), level="WARNING")

    def git(*a: str) -> str:
        return subprocess.run(
            ["git", "-C", str(code), *a], capture_output=True, text=True, check=True
        ).stdout.strip()

    head = git("rev-parse", "--short", "HEAD")
    status = git("status", "--porcelain")

    # ---- stage the runs --------------------------------------------------
    run_dirs = sorted(p for p in results.iterdir() if p.is_dir() and RUN_DIR.match(p.name))
    if len(run_dirs) != 24:
        raise SystemExit(f"expected 24 run directories, found {len(run_dirs)}")
    stage = Path(tempfile.mkdtemp(prefix="grid_stage_"))
    for p in run_dirs:
        (stage / p.name).symlink_to(p)
    exclusions_path = results / "exclusions.json"
    (stage / "exclusions.json").symlink_to(exclusions_path)

    try:
        records = load_runs(stage, prefix="rq1", judge=args.judge, require_paired_n=True)
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    # ---- raw rows (error rows included) for the pre-flight and the refusal triggers
    raw: dict[tuple, list[dict]] = {}
    for p in run_dirs:
        m = RUN_DIR.match(p.name)
        ds, model, k = m.group(2), m.group(4), int(m.group(5))
        for f in sorted(p.glob("rq1_*.jsonl")):
            if "_judged" in f.stem or f.stem.endswith("_summary"):
                continue
            raw[(model, ds, f.stem[len("rq1_") :], k)] = [
                json.loads(line)
                for line in f.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    exclusions = load_exclusions(exclusions_path)

    # ---- pre-flight ------------------------------------------------------
    pre: dict = {"code_head": head, "code_status": status or "clean", "run_dirs": len(run_dirs)}
    pre["n_judge_missing"] = sum(r.get("n_judge_missing", 0) for r in records)
    pre["warnings"] = sorted(set(warnings))

    present: dict[tuple, set] = defaultdict(set)
    for r in records:
        present[(r["model"], r["dataset"], r["pipeline"])].add(r["repeat_index"])
    pre["pipeline_runs"] = {f"{m}/{d}/{p}": sorted(ks) for (m, d, p), ks in sorted(present.items())}

    ids_by: dict[tuple, set] = {}
    for r in records:
        ids_by[(r["model"], r["dataset"], r["pipeline"], r["repeat_index"])] = {
            str(x["id"]) for x in r["rows"]
        }
    for model in MODELS:
        for ds in DATASETS:
            sets = {key: s for key, s in ids_by.items() if key[0] == model and key[1] == ds}
            if len({frozenset(s) for s in sets.values()}) != 1:
                raise SystemExit(f"id sets differ across pipelines/k in {model}/{ds}")
    pre["id_sets_equal_across_pipelines_and_k"] = True

    # exclusions: file vs recomputed from raw rows
    excl_check = []
    for model in MODELS:
        for ds in DATASETS:
            rows = [r for (m, d, _p, _k), rs in raw.items() if m == model and d == ds for r in rs]
            in_file = excluded_ids(exclusions, ds, model)
            err_ids = {
                str(r["id"]) for r in rows if "error" in r and r.get("error_class") != "truncated"
            }
            expected = err_ids | set(refusal_ids(rows, first=False))
            first_ids = set(refusal_ids(rows, first=True))
            excl_check.append(
                {
                    "model": model,
                    "dataset": ds,
                    "in_file": len(in_file),
                    "expected_from_rows": len(expected),
                    "file_not_in_rows": sorted(in_file - expected),
                    "rows_not_in_file": sorted(expected - in_file),
                    "first_attempt_refusal_ids": len(first_ids),
                    "first_only_extra": len(first_ids - in_file),
                }
            )
    pre["exclusions_check"] = excl_check

    none_pred: Counter = Counter()
    err_rows: list[dict] = []
    for (model, ds, pipe, k), rows in raw.items():
        for r in rows:
            if "error" in r:
                err_rows.append(
                    {
                        "model": model,
                        "dataset": ds,
                        "pipeline": pipe,
                        "k": k,
                        "id": r["id"],
                        "error_class": r.get("error_class", ""),
                        "excluded": str(r["id"]) in excluded_ids(exclusions, ds, model),
                    }
                )
            elif r.get("prediction") is None:
                none_pred[(model, ds, pipe, k)] += 1
    pre["none_predictions"] = {"/".join(map(str, key)): n for key, n in sorted(none_pred.items())}
    write_csv(out / "error_rows.csv", err_rows)

    # ---- per-row score table --------------------------------------------
    from scipy import stats as sps

    from agentic_rag.evaluation.metrics import _normalize_text, token_f1

    def first_sentence(text: str) -> str:
        return re.split(r"(?<=[.!?])\s+|\n+", text.strip(), maxsplit=1)[0]

    # Latency of the k=1 grid is not a cost measure (resumed and
    # concurrent runs mixed), so only k >= 2 rows enter the latency columns.
    LATENCY_MIN_K = 2

    data: dict[tuple, dict[int, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    hop_of: dict[tuple, dict[str, int]] = defaultdict(dict)
    raw_dir = code / "data" / "raw"
    refs = {ds: hop_reference(ds, raw_dir) for ds in DATASETS}
    for rec in records:
        model, ds, pipe, k = rec["model"], rec["dataset"], rec["pipeline"], rec["repeat_index"]
        for r in rec["rows"]:
            qid = str(r["id"])
            pred = _prediction_text(r.get("prediction"))
            hop = hop_for_row(r, refs[ds], f"{model}/{ds}/{pipe}")
            hop_of[(model, ds)][qid] = hop
            prior = r.get("prior_attempt_usage") or {}
            answers = [
                a for a in (r.get("all_references") or [r.get("reference", "")]) if a is not None
            ]
            norm_pred = _normalize_text(pred)
            first = first_sentence(pred)
            data[(model, ds, pipe)][k][qid] = {
                "f1": best_f1(pred, r),
                "judge": float(r["llm_judge"]),
                "refused": bool(is_refusal(pred)),
                "cost": r.get("cost_usd"),
                "metered_calls": r.get("metered_calls"),
                "llm_calls": r.get("llm_calls"),
                "latency": r.get("latency_seconds"),
                "latency_valid": k >= LATENCY_MIN_K,
                "prior_cost": prior.get("cost_usd") or 0.0,
                "prior_recorded": "prior_attempt_usage" in r,
                "truncated": bool(r.get("scored_wrong")),
                "words": len(pred.split()),
                "first_f1": max((token_f1(first, a) for a in answers), default=0.0),
                "contains": float(
                    any(_normalize_text(a) and _normalize_text(a) in norm_pred for a in answers)
                ),
            }
    n_trunc: Counter = Counter()
    for (model, ds, pipe), by_k in data.items():
        for k, rows in by_k.items():
            n_trunc[(model, ds, pipe)] += sum(v["truncated"] for v in rows.values())
    pre["truncated_kept_scored_wrong"] = {
        "/".join(key): n for key, n in sorted(n_trunc.items()) if n
    }
    pre["n_questions"] = {f"{m}/{d}": len(hop_of[(m, d)]) for m in MODELS for d in DATASETS}
    pre["n_excluded"] = {
        f"{m}/{d}": sorted(
            {r["n_excluded"] for r in records if r["model"] == m and r["dataset"] == d}
        )
        for m in MODELS
        for d in DATASETS
    }

    # sanity: the script's per-row refusal flag and F1 agree with RefusalAnalyzer on every run
    computed = RefusalAnalyzer(records).compute()
    assert len(computed) == len(records)
    for rec, row in zip(records, computed):
        vals = data[(rec["model"], rec["dataset"], rec["pipeline"])][rec["repeat_index"]].values()
        assert sum(v["refused"] for v in vals) == row["refusals"], (
            rec["model"],
            rec["dataset"],
            rec["pipeline"],
        )
        answered = [v["f1"] for v in vals if not v["refused"]]
        if answered:
            assert abs(round(float(np.mean(answered)), 4) - row["f1_non_refused"]) < 1e-4
    pre["refusal_pattern_count"] = len(REFUSAL_PATTERNS)

    def ks_of(model: str, ds: str, *pipes: str) -> list[int]:
        common = None
        for p in pipes:
            s = set(data[(model, ds, p)].keys())
            common = s if common is None else common & s
        return sorted(common or [])

    def arr(model, ds, pipe, ks, qids, metric) -> np.ndarray:
        return np.array(
            [np.mean([data[(model, ds, pipe)][k][q][metric] for k in ks]) for q in qids]
        )

    def stats(a: np.ndarray, b: np.ndarray, *key: str, wilcoxon: bool = False) -> dict:
        p, lo, hi, obs = _paired_bootstrap(a, b, comparison_rng(SEED, *key), n_boot=N_BOOT)
        diff = a - b
        sd = float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")
        s = {
            "n": len(a),
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "delta": obs,
            "ci_lo": lo,
            "ci_hi": hi,
            "d": obs / sd if sd and sd > 0 else float("nan"),
            "p": p,
            "_a": a,
            "_b": b,
            "_key": key,
        }
        if wilcoxon:
            try:
                s["wilcoxon_p"] = float(sps.wilcoxon(a, b, alternative="two-sided")[1])
            except ValueError:  # all differences zero; significance.py also returns 1.0
                s["wilcoxon_p"] = 1.0
        return s

    def apply_holm(rows: list[dict], family) -> None:
        """Holm within family(row); a boundary cell (p_holm in [0.04, 0.06]) is redrawn 10x.

        The redraw uses the same comparison key with 10 times
        the resamples; Holm is then recomputed with the redrawn p for those cells.
        """
        groups: dict = defaultdict(list)
        for r in rows:
            if "p" in r and not r["descriptive"]:
                groups[family(r)].append(r)
        for g in groups.values():
            for r, adj in zip(g, holm([x["p"] for x in g])):
                r["p_holm"] = adj
            for r in g:
                if 0.04 <= r["p_holm"] <= 0.06:
                    p2, lo2, hi2, _ = _paired_bootstrap(
                        r["_a"], r["_b"], comparison_rng(SEED, *r["_key"]), n_boot=10 * N_BOOT
                    )
                    r["p_recheck"], r["ci_recheck"] = p2, f"[{lo2:+.4f}, {hi2:+.4f}]"
                elif 0.03 <= r["p_holm"] <= 0.07:
                    # Outside the plan's band: redrawn for a footnote only; it does not
                    # change p_holm_final or the verdict.
                    r["p_recheck_note"] = _paired_bootstrap(
                        r["_a"], r["_b"], comparison_rng(SEED, *r["_key"]), n_boot=10 * N_BOOT
                    )[0]
            for r, adj in zip(g, holm([x.get("p_recheck", x["p"]) for x in g])):
                r["p_holm_final"] = adj
                r["sig_holm"] = adj < 0.05
            for r in g:
                r["holm_family_size"] = len(g)
                if "p_recheck_note" in r:
                    ps = [r["p_recheck_note"] if x is r else x.get("p_recheck", x["p"]) for x in g]
                    r["p_holm_note"] = holm(ps)[g.index(r)]
        for r in rows:
            if "delta" in r:
                r["sign"] = sign_of(r["delta"], r["ci_lo"], r["ci_hi"])

    def bins(model: str, ds: str, drop: dict) -> dict[int, list[str]]:
        gone = drop.get((model, ds), set())
        out_bins: dict[int, list[str]] = defaultdict(list)
        for q, h in sorted(hop_of[(model, ds)].items()):
            if q not in gone:
                out_bins[h].append(q)
        return dict(sorted(out_bins.items()))

    def suite(drop: dict[tuple, set]) -> dict[str, list[dict]]:
        """F1 (F1 and Judge), F6 and F5 with the questions in drop[(model, dataset)] removed."""
        f1_rows: list[dict] = []
        f6_rows: list[dict] = []
        f5_rows: list[dict] = []
        for model in MODELS:
            for ds in DATASETS:
                ks = ks_of(model, ds, AGENTIC, LOOP)
                for hop, qids in bins(model, ds, drop).items():
                    for metric in ("f1", "judge"):
                        a = arr(model, ds, AGENTIC, ks, qids, metric)
                        b = arr(model, ds, LOOP, ks, qids, metric)
                        s = stats(
                            a, b, "F1", metric, model, ds, str(hop), AGENTIC, LOOP, wilcoxon=True
                        )
                        f1_rows.append(
                            {
                                "metric": metric,
                                "model": model,
                                "dataset": ds,
                                "hop": hop,
                                "runs": f"k{ks[0]}-{ks[-1]}",
                                **s,
                                "descriptive": descriptive_only(ds, hop, metric),
                            }
                        )
                        deltas = [
                            float(
                                (
                                    arr(model, ds, AGENTIC, [k], qids, metric)
                                    - arr(model, ds, LOOP, [k], qids, metric)
                                ).mean()
                            )
                            for k in ks
                        ]
                        mean_d = float(np.mean(deltas))
                        sd = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan")
                        same_sign = all(x > 0 for x in deltas) or all(x < 0 for x in deltas)
                        f6_rows.append(
                            {
                                "metric": metric,
                                "model": model,
                                "dataset": ds,
                                "hop": hop,
                                "n_runs": len(ks),
                                "n": len(qids),
                                "delta_per_run": " / ".join(f"{x:+.4f}" for x in deltas),
                                "mean_delta": mean_d,
                                "sd_delta": sd,
                                "same_sign": same_sign,
                                "verdict": "stable"
                                if (sd < abs(mean_d) and same_sign)
                                else "no reliable difference",
                                "descriptive": descriptive_only(ds, hop, metric),
                            }
                        )
                    for other, label in ((LOOP, "agentic-loop"), (CRAG, "agentic-crag")):
                        kso = ks_of(model, ds, AGENTIC, other)
                        if not kso:
                            continue
                        runs = f"k{kso[0]}-{kso[-1]}" if len(kso) > 1 else f"k{kso[0]} single run"
                        base = {
                            "comparison": label,
                            "model": model,
                            "dataset": ds,
                            "hop": hop,
                            "runs": runs,
                        }
                        ra = arr(model, ds, AGENTIC, kso, qids, "refused")
                        rb = arr(model, ds, other, kso, qids, "refused")
                        s = stats(ra, rb, "F5", "refusal_rate", label, model, ds, str(hop))
                        f5_rows.append(
                            {
                                **base,
                                "measure": "refusal_rate",
                                **s,
                                "marg_a": float(ra.mean()),
                                "marg_b": float(rb.mean()),
                                "descriptive": descriptive_only(ds, hop, "f1"),
                            }
                        )
                        for metric in ("f1", "judge"):
                            va, vb, pool_a, pool_b, ref_a, ref_b = [], [], [], [], [], []
                            for q in qids:
                                xa = [data[(model, ds, AGENTIC)][k][q] for k in kso]
                                xb = [data[(model, ds, other)][k][q] for k in kso]
                                na = [x[metric] for x in xa if not x["refused"]]
                                nb = [x[metric] for x in xb if not x["refused"]]
                                pool_a += na
                                pool_b += nb
                                ref_a += [x[metric] for x in xa if x["refused"]]
                                ref_b += [x[metric] for x in xb if x["refused"]]
                                if na and nb:
                                    va.append(np.mean(na))
                                    vb.append(np.mean(nb))
                            row = {
                                **base,
                                "measure": f"{metric}_non_refused",
                                "descriptive": descriptive_only(ds, hop, metric),
                                "ref_a": float(np.mean(ref_a)) if ref_a else float("nan"),
                                "ref_b": float(np.mean(ref_b)) if ref_b else float("nan"),
                                "n_ref_a": len(ref_a),
                                "n_ref_b": len(ref_b),
                            }
                            if len(va) < 2:
                                row.update(
                                    {
                                        "n": len(va),
                                        "descriptive": True,
                                        "note": "fewer than 2 paired questions",
                                    }
                                )
                            else:
                                row.update(
                                    stats(
                                        np.array(va),
                                        np.array(vb),
                                        "F5",
                                        f"{metric}_non_refused",
                                        label,
                                        model,
                                        ds,
                                        str(hop),
                                    ),
                                    marg_a=float(np.mean(pool_a)) if pool_a else float("nan"),
                                    marg_b=float(np.mean(pool_b)) if pool_b else float("nan"),
                                )
                            f5_rows.append(row)
        apply_holm(f1_rows, lambda r: r["metric"])
        apply_holm(f5_rows, lambda r: "judge" if r["measure"] == "judge_non_refused" else "F5")
        for rows in (f1_rows, f5_rows):
            for r in rows:
                r.setdefault("p_holm", None)
                r.setdefault("p_holm_final", None)
                r.setdefault("sig_holm", None)
        return {"f1": f1_rows, "f6": f6_rows, "f5": f5_rows}

    main = suite({})

    # ---- sensitivity suites ------------------------------------------------
    drop_a: dict[tuple, set] = {}
    for model in MODELS:
        for ds in DATASETS:
            rows = [r for (m, d, _p, _k), rs in raw.items() if m == model and d == ds for r in rs]
            drop_a[(model, ds)] = set(refusal_ids(rows, first=True)) - excluded_ids(
                exclusions, ds, model
            )
    sens_a = suite(drop_a)

    AC786909 = "2wiki_ac786909087411ebbd67ac1f6bf848b6"
    sens_r2 = suite({("gpt-5-mini", "2wikimultihopqa"): {AC786909}})

    # M+: the questions both models keep (union of the two models' exclusions removed from both)
    drop_m = {}
    for ds in DATASETS:
        union = set().union(*(excluded_ids(exclusions, ds, m) for m in MODELS))
        for model in MODELS:
            drop_m[(model, ds)] = union - excluded_ids(exclusions, ds, model)
    mplus = suite(drop_m)

    for name, s in (
        ("main", main),
        ("declined_removed", sens_a),
        ("ac786909_removed", sens_r2),
        ("common_questions", mplus),
    ):
        suffix = "" if name == "main" else f"_{name}"
        write_csv(out / f"grid_f1_agentic_minus_loop{suffix}.csv", s["f1"])
        write_csv(out / f"grid_stability{suffix}.csv", s["f6"])
        write_csv(out / f"grid_refusal_markers{suffix}.csv", s["f5"])

    def changes(d_rows: list[dict], o_rows: list[dict], key, kind: str) -> list[dict]:
        """Cells whose verdict differs between two tables of the same shape."""
        other = {key(r): r for r in o_rows}
        out_rows = []
        for r in d_rows:
            o = other.get(key(r))
            if o is None:
                continue
            if kind == "f6":
                if r["verdict"] != o["verdict"]:
                    out_rows.append(
                        {
                            "cell": key(r),
                            "d": r["verdict"],
                            "o": o["verdict"],
                            "reason": "F6 verdict",
                            "descriptive": r["descriptive"],
                            "n": f"{r['n']}→{o['n']}",
                        }
                    )
                continue
            if "delta" not in r or "delta" not in o:
                continue
            if r["sign"] != o["sign"]:
                reason = "CI covers 0 in one table only"
            elif (r["delta"] > 0) != (o["delta"] > 0):
                reason = "direction of delta only (same sign verdict)"
            else:
                continue
            out_rows.append(
                {
                    "cell": key(r),
                    "d": f"{r['delta']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] {r['sign']}",
                    "o": f"{o['delta']:+.4f} [{o['ci_lo']:+.4f}, {o['ci_hi']:+.4f}] {o['sign']}",
                    "reason": reason,
                    "descriptive": r["descriptive"],
                    "n": f"{r['n']}→{o['n']}",
                }
            )
        return out_rows

    k_f1 = lambda r: (r["metric"], r["model"], r["dataset"], r["hop"])  # noqa: E731
    k_f5 = lambda r: (r["comparison"], r["model"], r["dataset"], r["hop"], r["measure"])  # noqa: E731
    flags = {
        "f1": [c for c in changes(main["f1"], sens_a["f1"], k_f1, "f1") if c["cell"][0] == "f1"],
        "judge": [
            c for c in changes(main["f1"], sens_a["f1"], k_f1, "f1") if c["cell"][0] == "judge"
        ],
        "f6": changes(main["f6"], sens_a["f6"], k_f1, "f6"),
        "f5": changes(main["f5"], sens_a["f5"], k_f5, "f5"),
    }
    write_csv(
        out / "sensitivity_declined_removed_changed_cells.csv",
        [
            {
                "table": t,
                "cell": "/".join(map(str, c["cell"])),
                "main": c["d"],
                "A": c["o"],
                "n": c["n"],
                "reason": c["reason"],
                "descriptive": c["descriptive"],
            }
            for t, cs in flags.items()
            for c in cs
        ],
    )

    # ---- exploratory: answer length and form ------------------------------
    expl_rows: list[dict] = []
    for model in MODELS:
        for ds in DATASETS:
            for hop, qids in bins(model, ds, {}).items():
                for pipe in (AGENTIC, LOOP):
                    vals = [
                        v
                        for k in data[(model, ds, pipe)]
                        for q, v in data[(model, ds, pipe)][k].items()
                        if q in set(qids)
                    ]
                    ref = [v for v in vals if v["refused"]]
                    expl_rows.append(
                        {
                            "model": model,
                            "dataset": ds,
                            "hop": hop,
                            "pipeline": pipe,
                            "rows": len(vals),
                            "words_median": float(np.median([v["words"] for v in vals])),
                            "f1": float(np.mean([v["f1"] for v in vals])),
                            "first_sentence_f1": float(np.mean([v["first_f1"] for v in vals])),
                            "answer_contained": float(np.mean([v["contains"] for v in vals])),
                            "judge": float(np.mean([v["judge"] for v in vals])),
                            "refusal_marker": len(ref) / len(vals),
                            "refused_rows": len(ref),
                            "refused_answer_contained": float(np.mean([v["contains"] for v in ref]))
                            if ref
                            else float("nan"),
                            "refused_judge": float(np.mean([v["judge"] for v in ref]))
                            if ref
                            else float("nan"),
                        }
                    )
    write_csv(out / "grid_answer_form.csv", expl_rows)

    # ---- cost / latency --------------------------------------------------
    cost_rows: list[dict] = []
    for model in MODELS:
        for ds in DATASETS:
            for hop, qids in bins(model, ds, {}).items():
                qset = set(qids)
                for pipe in PIPELINES:
                    by_k = data.get((model, ds, pipe))
                    if not by_k:
                        continue
                    vals = [v for k in by_k for q, v in by_k[k].items() if q in qset]
                    cost = [v["cost"] for v in vals if v["cost"] is not None]
                    lat = [
                        v["latency"]
                        for v in vals
                        if v["latency"] is not None and v["latency_valid"]
                    ]
                    prior_recorded = any(v["prior_recorded"] for v in vals)
                    cost_rows.append(
                        {
                            "model": model,
                            "dataset": ds,
                            "hop": hop,
                            "pipeline": pipe,
                            "runs": f"k{min(by_k)}-{max(by_k)}",
                            "n_rows": len(vals),
                            "median_cost_usd": float(np.median(cost)) if cost else float("nan"),
                            "median_latency_s": float(np.median(lat)) if lat else float("nan"),
                            "median_metered_calls": _med([v["metered_calls"] for v in vals]),
                            "median_llm_calls": _med([v["llm_calls"] for v in vals]),
                            "latency_rows": len(lat),
                            "rows_without_cost": len(vals) - len(cost),
                            "sum_row_cost_usd": float(sum(cost)),
                            "sum_prior_attempt_cost_usd": float(sum(v["prior_cost"] for v in vals))
                            if prior_recorded
                            else "not recorded",
                        }
                    )
    # Agentic / Loop multiples of the cell medians, repeated on every row of the cell
    by_cell = {(c["model"], c["dataset"], c["hop"], c["pipeline"]): c for c in cost_rows}
    for c in cost_rows:
        ag = by_cell.get((c["model"], c["dataset"], c["hop"], AGENTIC))
        lp = by_cell.get((c["model"], c["dataset"], c["hop"], LOOP))
        for src, name in (
            ("median_metered_calls", "metered_calls"),
            ("median_llm_calls", "llm_calls"),
            ("median_cost_usd", "cost_usd"),
            ("median_latency_s", "latency_s"),
        ):
            c[f"agentic_over_loop_{name}"] = (
                ag[src] / lp[src] if ag and lp and lp[src] == lp[src] and lp[src] else float("nan")
            )
    write_csv(out / "grid_cost_latency.csv", cost_rows)

    # ---- refusal-propensity axis: naive RAG, 2Wiki, refusal-marker rate ----
    from experiments.analysis.runs import load_runs as _load_runs

    axis_rows: list[dict] = []
    for model in MODELS:
        by_k = data[(model, "2wikimultihopqa", "naive_rag")]
        per_run = {
            k: float(np.mean([v["refused"] for v in rows.values()]))
            for k, rows in sorted(by_k.items())
        }
        axis_rows.append(
            {
                "source": "controlled grid",
                "model": model,
                "pipeline": "naive_rag@30",
                "passage_budget": 30,
                "n_questions": len(next(iter(by_k.values()))),
                "runs": f"k{min(by_k)}-{max(by_k)}",
                "rate_per_run": " / ".join(f"{x:.4f}" for x in per_run.values()),
                "refusal_rate": float(np.mean(list(per_run.values()))),
                "patterns": len(REFUSAL_PATTERNS),
            }
        )
    march: list[dict] = []
    for tree, tree_model in (
        ("data/results/paper", None),
        ("data/results-openweight/final", "gpt-oss-120b"),
    ):
        if not (code / tree).is_dir():
            print(f"skipped reference rows: {tree} not present")
            continue
        for rec in _load_runs(code / tree, prefix="rq1", model=tree_model, require_paired_n=False):
            if rec["dataset"] == "2wikimultihopqa" and rec["pipeline"] == "naive_rag":
                rate = float(
                    np.mean(
                        [is_refusal(_prediction_text(r.get("prediction"))) for r in rec["rows"]]
                    )
                )
                march.append(
                    {
                        "source": "March/July runs (reference)",
                        "model": rec["model"],
                        "pipeline": "naive_rag@50 (no M cap)",
                        "passage_budget": 50,
                        "n_questions": len(rec["rows"]),
                        "runs": "single run",
                        "rate_per_run": f"{rate:.4f}",
                        "refusal_rate": rate,
                        "patterns": len(REFUSAL_PATTERNS),
                    }
                )
    for group in (axis_rows, march):
        for rank, r in enumerate(sorted(group, key=lambda x: -x["refusal_rate"]), 1):
            r["rank_in_source"] = rank
    write_csv(out / "grid_refusal_axis.csv", axis_rows + march)

    # ---- refusal-pattern recount (refusal_pattern_recount_*) ---------------------------
    gpt_dirs = [str(p) for p in run_dirs if "_gpt-5-mini_" in p.name]
    c4_csv = out / "refusal_pattern_recount_changed_rows.csv"
    proc = subprocess.run(
        [
            sys.executable,
            "experiments/analysis/refusal_pattern_recount.py",
            *gpt_dirs,
            "--csv",
            str(c4_csv),
        ],
        cwd=code,
        capture_output=True,
        text=True,
        check=True,
    )
    c4_summary = "\n".join(
        line for line in proc.stdout.splitlines() if not line.startswith("  ")
    ).strip()
    (out / "refusal_pattern_recount_summary.txt").write_text(c4_summary + "\n", encoding="utf-8")

    (out / "preflight.json").write_text(
        json.dumps(pre, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
