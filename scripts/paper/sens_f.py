#!/usr/bin/env python3
"""Sensitivity F: provider-declined Sonnet questions kept as wrong answers.

The main table removes from every configuration and run each question that ended in
a final Anthropic refusal (exclusions.json, reason anthropic_refusal). Here
those questions stay in. A final refusal is a call refused after the third
resend; it can be an intermediate call, so the row may still carry an answer.
  --mode zero (default, sensitivity F): every row of those questions with a
    final refusal (refusals_final_by_category) or an error scores F1 0 and
    judge 0; the configuration's other runs keep their own scores.
  --mode asis: the rows keep the scores they were given (an error row, which
    has no answer, scores 0).
Everything else is as in the main table:
  - gpt-5-mini's exclusions (platform filter) stay as they are;
  - truncated rows are wrong;
  - per-question mean over k1-3;
  - paired bootstrap with the main table's comparison key (10,000, seed 42);
  - Holm within metric over the non-descriptive cells of both families;
  - p_holm in [.04, .06] redrawn 10x;
  - run stability s/u by the f6 rule;
  - gpt-4.1 judge.

Also counts, per cell, the zero-filled (configuration, run, question) entries
of Agentic and of Loop, i.e. on which side the refusals fall.

Writes sens_f_table.csv and sens_f_table.tex (layout of tab:rq1_main) to
scripts/paper/out/robustness/. Reads the outputs of controlled_grid_analysis.py and
judge_gpt41_grid.py (run them first). Reads only; no API call; no prediction
is printed.

    uv run python scripts/paper/sens_f.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "scripts" / "paper" / "out"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from controlled_grid_analysis import (  # noqa: E402
    AGENTIC,
    DATASETS,
    LOOP,
    MODELS,
    N_BOOT,
    RUN_DIR,
    SEED,
    descriptive_only,
    holm,
    sign_of,
    write_csv,
)
from sens_a_table import COLS, lvl, num  # noqa: E402

JUDGE = "gpt-4-1_39118f"
SONNET = "claude-sonnet-5-5"
LABEL = {"gpt-5-mini": "gpt-5-mini", SONNET: "Claude Sonnet 5.5"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--code-repo", type=Path, default=REPO)
    ap.add_argument("--out", type=Path, default=OUT / "robustness")
    ap.add_argument("--mode", choices=("zero", "asis"), default="zero")
    args = ap.parse_args()
    suffix = "" if args.mode == "zero" else "_asis"
    code = args.code_repo.resolve()
    results = code / "data" / "results-campaign"
    sys.path.insert(0, str(code))

    from loguru import logger

    from experiments.analysis.hop_level import _paired_bootstrap
    from experiments.analysis.refusal_analysis import _prediction_text
    from experiments.analysis.runs import best_f1, comparison_rng, load_runs
    from experiments.difficulty import hop_for_row, hop_reference

    logger.remove()

    excl = json.loads((results / "exclusions.json").read_text(encoding="utf-8"))
    refused: dict[str, set[str]] = defaultdict(set)  # dataset -> Sonnet questions kept in
    kept_excl: dict = {}
    for ds, by_model in excl.items():
        if ds.startswith("_"):
            kept_excl[ds] = by_model
            continue
        kept_excl[ds] = {}
        for model, entries in by_model.items():
            keep = [
                e for e in entries if not (model == SONNET and e["reason"] == "anthropic_refusal")
            ]
            refused[ds].update(
                str(e["id"])
                for e in entries
                if model == SONNET and e["reason"] == "anthropic_refusal"
            )
            kept_excl[ds][model] = keep

    run_dirs = sorted(p for p in results.iterdir() if p.is_dir() and RUN_DIR.match(p.name))
    if len(run_dirs) != 24:
        raise SystemExit(f"expected 24 run directories, found {len(run_dirs)}")
    stage = Path(tempfile.mkdtemp(prefix="sens_f_stage_"))
    try:
        for p in run_dirs:
            (stage / p.name).symlink_to(p)
        (stage / "exclusions.json").write_text(json.dumps(kept_excl), encoding="utf-8")
        records = load_runs(stage, prefix="rq1", judge=JUDGE, require_paired_n=False)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    if sum(r.get("n_judge_missing", 0) for r in records):
        raise SystemExit(f"rows without a {JUDGE} verdict")

    # error rows of the kept-in questions (the final refusals), from the raw files
    refusal_rows: set[tuple] = set()  # (model, ds, pipe, k, qid)
    for p in run_dirs:
        m = RUN_DIR.match(p.name)
        ds, model, k = m.group(2), m.group(4), int(m.group(5))
        if model != SONNET:
            continue
        for f in p.glob("rq1_*.jsonl"):
            if "_judged" in f.stem:
                continue
            pipe = f.stem[len("rq1_") :]
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if "error" in r and str(r["id"]) in refused[ds]:
                        refusal_rows.add((model, ds, pipe, k, str(r["id"])))

    refs = {ds: hop_reference(ds, code / "data" / "raw") for ds in DATASETS}
    val: dict = defaultdict(lambda: defaultdict(dict))
    hop_of: dict = {}
    zero_filled = Counter()
    for rec in records:
        model, ds, pipe, k = rec["model"], rec["dataset"], rec["pipeline"], rec["repeat_index"]
        for r in rec["rows"]:
            q = str(r["id"])
            hop_of[(ds, q)] = hop_for_row(r, refs[ds], f"{model}/{ds}/{pipe}")
            final = sum((r.get("refusals_final_by_category") or {}).values())
            if args.mode == "zero" and model == SONNET and q in refused[ds] and final:
                val[(model, ds, pipe)][k][q] = {"f1": 0.0, "judge": 0.0}
                zero_filled[(model, ds, pipe, q)] += 1
                continue
            val[(model, ds, pipe)][k][q] = {
                "f1": best_f1(_prediction_text(r.get("prediction")), r),
                "judge": float(r["llm_judge"]),
            }
    for model, ds, pipe, k, q in refusal_rows:
        if q in val[(model, ds, pipe)][k]:
            raise SystemExit(
                f"refusal error row and an answer row for {model} {ds} {pipe} k{k} {q}"
            )
        val[(model, ds, pipe)][k][q] = {"f1": 0.0, "judge": 0.0}
        zero_filled[(model, ds, pipe, q)] += 1

    rows = []
    for model in MODELS:
        for ds in DATASETS:
            ks = sorted(val[(model, ds, AGENTIC)])
            qids_all = set.intersection(
                *(set(val[(model, ds, p)][k]) for p in (AGENTIC, LOOP) for k in ks)
            )
            union = set().union(*(set(val[(model, ds, p)][k]) for p in (AGENTIC, LOOP) for k in ks))
            if union - qids_all:
                raise SystemExit(f"unpaired questions {model} {ds}: {sorted(union - qids_all)[:5]}")
            by_hop: dict = defaultdict(list)
            for q in sorted(qids_all):
                by_hop[hop_of[(ds, q)]].append(q)
            for hop, qids in sorted(by_hop.items()):
                for metric in ("f1", "judge"):

                    def arr(pipe, kk):
                        return np.array(
                            [
                                np.mean([val[(model, ds, pipe)][k][q][metric] for k in kk])
                                for q in qids
                            ]
                        )

                    a, b = arr(AGENTIC, ks), arr(LOOP, ks)
                    key = ("F1", metric, model, ds, str(hop), AGENTIC, LOOP)
                    p, lo, hi, obs = _paired_bootstrap(
                        a, b, comparison_rng(SEED, *key), n_boot=N_BOOT
                    )
                    per_run = [float((arr(AGENTIC, [k]) - arr(LOOP, [k])).mean()) for k in ks]
                    sd = float(np.std(per_run, ddof=1))
                    same = all(x > 0 for x in per_run) or all(x < 0 for x in per_run)
                    kept_in = [q for q in qids if q in refused[ds]] if model == SONNET else []
                    rows.append(
                        {
                            "model": model,
                            "metric": metric,
                            "dataset": ds,
                            "hop": hop,
                            "n_F": len(qids),
                            "kept_in_questions": len(kept_in),
                            "zero_agentic_runs": sum(
                                zero_filled[(model, ds, AGENTIC, q)] for q in kept_in
                            ),
                            "zero_loop_runs": sum(
                                zero_filled[(model, ds, LOOP, q)] for q in kept_in
                            ),
                            "agentic_F": float(a.mean()),
                            "loop_F": float(b.mean()),
                            "delta_F": obs,
                            "ci_lo_F": lo,
                            "ci_hi_F": hi,
                            "p_F": p,
                            "stability_F": "stable"
                            if (sd < abs(np.mean(per_run)) and same)
                            else "no reliable difference",
                            "descriptive": descriptive_only(ds, hop, metric),
                            "_a": a,
                            "_b": b,
                            "_key": key,
                        }
                    )
    for metric in ("f1", "judge"):
        g = [r for r in rows if r["metric"] == metric and not r["descriptive"]]
        for r, adj in zip(g, holm([x["p_F"] for x in g])):
            r["p_holm_F"] = adj
        for r in g:
            if 0.04 <= r["p_holm_F"] <= 0.06:
                r["p_recheck_F"] = _paired_bootstrap(
                    r["_a"], r["_b"], comparison_rng(SEED, *r["_key"]), n_boot=10 * N_BOOT
                )[0]
        for r, adj in zip(g, holm([x.get("p_recheck_F", x["p_F"]) for x in g])):
            r["p_holm_final_F"] = adj
    for r in rows:
        r["sign_F"] = sign_of(r["delta_F"], r["ci_lo_F"], r["ci_hi_F"])

    # main-table values under gpt-4.1 for the comparison column
    main = {}
    for path, judge_col in (
        (OUT / "controlled_grid" / "f1_agentic_minus_loop.csv", None),
        (OUT / "judge_gpt41_grid" / "contrasts_by_judge.csv", "gpt41"),
    ):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if judge_col is None and r["metric"] == "f1":
                    main[("f1", r["model"], r["dataset"], int(r["hop"]))] = (
                        int(r["n"]),
                        float(r["delta"]),
                        r["sign"],
                        r["p_holm_final"],
                    )
                elif judge_col and r["contrast"] == "Agentic-Loop" and r["hop"] != "all":
                    main[("judge", r["model"], r["dataset"], int(r["hop"]))] = (
                        int(r["n"]),
                        float(r["gpt41_delta"]),
                        r["gpt41_sign"],
                        r["gpt41_p_holm_final"],
                    )
    for r in rows:
        n, d, s, ph = main[(r["metric"], r["model"], r["dataset"], r["hop"])]
        r.update(
            {
                "n_main": n,
                "delta_main": d,
                "sign_main": s,
                "p_holm_final_main": "" if r["descriptive"] else ph,
            }
        )
        if r["model"] != SONNET and (n != r["n_F"] or abs(d - r["delta_F"]) > 1e-9):
            raise SystemExit(
                f"gpt-5-mini row differs from the main table: {r['metric']} {r['dataset']} {r['hop']}"
            )
        r["sign_change"] = r["sign_F"] != s
        r["holm_change"] = (not r["descriptive"]) and (
            (r["p_holm_final_F"] < 0.05) != (float(ph) < 0.05)
        )
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / f"sens_f_table{suffix}.csv", rows)

    by = {(r["model"], r["metric"], r["dataset"], str(r["hop"])): r for r in rows}
    lines = [
        "% Sensitivity F (provider-declined Sonnet questions scored 0) in the layout of tab:rq1_main.",
        "% Generated by scripts/paper/sens_f.py. Do not edit by hand.",
        "\\begin{tabular}{ll ccccccc}",
        "\\toprule",
        "& & \\textbf{HotpotQA} & \\multicolumn{2}{c}{\\textbf{2WikiMultiHopQA}} & \\multicolumn{3}{c}{\\textbf{MuSiQue}} & \\textbf{FinanceBench} \\\\",
        "\\cmidrule(lr){3-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-9}",
        "\\textbf{Model} & & 2-hop & 2-hop & 4-hop & 2-hop & 3-hop & 4-hop$\\dagger$ & 1-hop \\\\",
    ]
    for model in MODELS:
        lines += ["\\midrule", f"\\multirow{{7}}{{*}}{{{LABEL[model]}}}"]
        lines.append(
            " & $n$ & "
            + " & ".join(str(by[(model, "f1", ds, h)]["n_F"]) for ds, h in COLS)
            + " \\\\"
        )
        for metric, mlabel in (("f1", "F1"), ("judge", "Judge")):
            cells = [by[(model, metric, ds, h)] for ds, h in COLS]
            dag = [
                "$\\dagger$" if (c["descriptive"] and ds == "financebench") else ""
                for c, (ds, _) in zip(cells, COLS)
            ]
            lines.append(
                f" & {mlabel} \\agentic{{}} & "
                + " & ".join(lvl(c["agentic_F"]) + g for c, g in zip(cells, dag))
                + " \\\\"
            )
            lines.append(
                f" & {mlabel} Loop & "
                + " & ".join(lvl(c["loop_F"]) + g for c, g in zip(cells, dag))
                + " \\\\"
            )
            parts = []
            for c, g in zip(cells, dag):
                star = "^{*}" if (not c["descriptive"] and c["p_holm_final_F"] < 0.05) else ""
                stab = "s" if c["stability_F"] == "stable" else "u"
                body = (
                    f"${num(c['delta_F'])}{star}$ [${num(c['ci_lo_F'])}$, ${num(c['ci_hi_F'])}$]"
                    f"$^{{\\mathrm{{{stab}}}}}${g}"
                )
                # Bold: a tested cell whose 95% CI excludes zero (never a descriptive one).
                bold = c["sign_F"] in ("+", "-") and not c["descriptive"]
                parts.append(f"{{\\bfseries\\boldmath {body}}}" if bold else body)
            lines.append(
                f" & {mlabel} $\\Delta$ [95\\% CI]\n   & " + "\n   & ".join(parts) + " \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    (args.out / f"sens_f_table{suffix}.tex").write_text("\n".join(lines), encoding="utf-8")

    print(
        f"mode {args.mode}; kept-in questions {sum(len(v) for v in refused.values())}; zero-filled entries "
        f"{sum(zero_filled.values())} by pipeline {dict(Counter({k[2]: 0 for k in zero_filled}) + Counter(dict((p, sum(v for kk, v in zero_filled.items() if kk[2] == p)) for p in {k[2] for k in zero_filled})))}"
    )
    for r in rows:
        if r["model"] == SONNET:
            print(
                f"{r['metric']:5s} {r['dataset'][:8]:8s} {r['hop']} n {r['n_main']}->{r['n_F']} "
                f"zeros A/L {r['zero_agentic_runs']}/{r['zero_loop_runs']} "
                f"d {r['delta_main']:+.4f}->{r['delta_F']:+.4f} [{r['ci_lo_F']:+.4f},{r['ci_hi_F']:+.4f}] "
                f"sign {r['sign_main']}->{r['sign_F']} holm {r['p_holm_final_main'] or '-'}->"
                f"{'' if r['descriptive'] else round(r['p_holm_final_F'], 4)}"
                + (" CHANGE" if r["sign_change"] or r["holm_change"] else "")
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
