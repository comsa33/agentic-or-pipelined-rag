#!/usr/bin/env python3
"""Sensitivity A in the format of the main table, gpt-4.1 judge.

Sensitivity A (controlled_grid_analysis.py) adds to each model's exclusions every
question on which any call of any configuration and run was refused, so its
cells are computed only on questions no refusal touched. The _A CSVs of controlled_grid_analysis.py carry the
gpt-4.1-nano judge. This runs controlled_grid_analysis.py unchanged with
--judge gpt-4-1_39118f into a temporary directory (the same way mplus_gpt41.py
does), takes f1_agentic_minus_loop_A.csv and f6_stability_A.csv, and checks
that its F1 rows equal the _A F1 rows written by controlled_grid_analysis.py
(F1 does not depend on the judge; run that script first).

Writes sens_a_table.csv (one row per model, metric and cell) and
sens_a_table.tex (a table fragment laid out like tab:rq1_main: n, levels, and
Delta [95% CI] with bold = CI excludes 0, * = Holm p < .05, s/u = run
stability, dagger = descriptive) to scripts/paper/out/robustness/. Reads only;
no API call.

    uv run python scripts/paper/sens_a_table.py
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "scripts" / "paper" / "out"
JUDGE = "gpt-4-1_39118f"
MODELS = (("gpt-5-mini", "gpt-5-mini"), ("claude-sonnet-5-5", "Claude Sonnet 5.5"))
COLS = [
    ("hotpotqa", "2"),
    ("2wikimultihopqa", "2"),
    ("2wikimultihopqa", "4"),
    ("musique", "2"),
    ("musique", "3"),
    ("musique", "4"),
    ("financebench", "1"),
]


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def half_up(x: float) -> Decimal:
    """x to three decimals, ties away from zero (.0625 -> .063), from its shortest repr."""
    return Decimal(repr(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)


def num(x: float) -> str:
    """Signed, three decimals; a value that rounds to zero is written unsigned (.000)."""
    q = half_up(x)
    s = f"{q:+.3f}" if q != 0 else ".000"
    return s.replace("0.", ".", 1)


def lvl(x: float) -> str:
    s = f"{half_up(x):.3f}"
    return s[1:] if s.startswith("0") else s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--code-repo", type=Path, default=REPO)
    ap.add_argument("--out", type=Path, default=OUT / "robustness")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="sens_a_") as tmp:
        tmp = Path(tmp)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "controlled_grid_analysis.py"),
                "--code-repo",
                str(args.code_repo.resolve()),
                "--out",
                str(tmp / "out"),
                "--judge",
                JUDGE,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        a_rows = read(tmp / "out/f1_agentic_minus_loop_A.csv")
        f6 = {
            (r["metric"], r["model"], r["dataset"], r["hop"]): r
            for r in read(tmp / "out/f6_stability_A.csv")
        }
        d_rows = read(tmp / "out/f1_agentic_minus_loop.csv")

    published_a = {
        (r["model"], r["dataset"], r["hop"]): r
        for r in read(OUT / "controlled_grid" / "f1_agentic_minus_loop_A.csv")
        if r["metric"] == "f1"
    }
    for r in a_rows:
        if r["metric"] == "f1":
            p = published_a[(r["model"], r["dataset"], r["hop"])]
            for c in ("n", "delta", "ci_lo", "ci_hi", "p_holm_final"):
                if p[c] != r[c]:
                    raise SystemExit(
                        f"F1 row of sensitivity A differs from the published one: {r['model']} {r['dataset']} {r['hop']} {c}"
                    )
    d = {(r["metric"], r["model"], r["dataset"], r["hop"]): r for r in d_rows}

    out_rows = []
    for r in a_rows:
        k = (r["metric"], r["model"], r["dataset"], r["hop"])
        desc = r["descriptive"] == "True"
        out_rows.append(
            {
                "model": r["model"],
                "metric": r["metric"],
                "dataset": r["dataset"],
                "hop": r["hop"],
                "n_A": r["n"],
                "agentic_A": r["mean_a"],
                "loop_A": r["mean_b"],
                "delta_A": r["delta"],
                "ci_lo_A": r["ci_lo"],
                "ci_hi_A": r["ci_hi"],
                "p_holm_final_A": "" if desc else r["p_holm_final"],
                "sign_A": r["sign"],
                "stability_A": f6[k]["verdict"],
                "descriptive": desc,
                "n_main": d[k]["n"],
                "delta_main": d[k]["delta"],
                "sign_main": d[k]["sign"],
                "p_holm_final_main": "" if desc else d[k]["p_holm_final"],
                "sign_change": r["sign"] != d[k]["sign"],
                "holm_change": (not desc)
                and ((float(r["p_holm_final"]) < 0.05) != (float(d[k]["p_holm_final"]) < 0.05)),
            }
        )
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "sens_a_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    by = {(r["model"], r["metric"], r["dataset"], r["hop"]): r for r in out_rows}
    lines = [
        "% Sensitivity A in the layout of tab:rq1_main. Generated by",
        "% scripts/paper/sens_a_table.py. Do not edit by hand.",
        "\\begin{tabular}{ll ccccccc}",
        "\\toprule",
        "& & \\textbf{HotpotQA} & \\multicolumn{2}{c}{\\textbf{2WikiMultiHopQA}} & \\multicolumn{3}{c}{\\textbf{MuSiQue}} & \\textbf{FinanceBench} \\\\",
        "\\cmidrule(lr){3-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-9}",
        "\\textbf{Model} & & 2-hop & 2-hop & 4-hop & 2-hop & 3-hop & 4-hop$\\dagger$ & 1-hop \\\\",
    ]
    for model, label in MODELS:
        lines += ["\\midrule", f"\\multirow{{7}}{{*}}{{{label}}}"]
        lines.append(
            " & $n$ & " + " & ".join(by[(model, "f1", ds, h)]["n_A"] for ds, h in COLS) + " \\\\"
        )
        for metric, mlabel in (("f1", "F1"), ("judge", "Judge")):
            cells = [by[(model, metric, ds, h)] for ds, h in COLS]
            dag = [
                "$\\dagger$" if (c["descriptive"] and ds == "financebench") else ""
                for c, (ds, _) in zip(cells, COLS)
            ]
            lines.append(
                f" & {mlabel} \\agentic{{}} & "
                + " & ".join(lvl(float(c["agentic_A"])) + g for c, g in zip(cells, dag))
                + " \\\\"
            )
            lines.append(
                f" & {mlabel} Loop & "
                + " & ".join(lvl(float(c["loop_A"])) + g for c, g in zip(cells, dag))
                + " \\\\"
            )
            parts = []
            for c, g in zip(cells, dag):
                star = "^{*}" if c["p_holm_final_A"] and float(c["p_holm_final_A"]) < 0.05 else ""
                stab = "s" if c["stability_A"] == "stable" else "u"
                body = (
                    f"${num(float(c['delta_A']))}{star}$ [${num(float(c['ci_lo_A']))}$, "
                    f"${num(float(c['ci_hi_A']))}$]$^{{\\mathrm{{{stab}}}}}${g}"
                )
                parts.append(
                    f"{{\\bfseries\\boldmath {body}}}" if c["sign_A"] in ("+", "-") else body
                )
            lines.append(
                f" & {mlabel} $\\Delta$ [95\\% CI]\n   & " + "\n   & ".join(parts) + " \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    (args.out / "sens_a_table.tex").write_text("\n".join(lines), encoding="utf-8")
    ch = [r for r in out_rows if r["sign_change"] or r["holm_change"]]
    print(
        f"rows {len(out_rows)}; cells whose verdict differs from the main table: "
        + "; ".join(
            f"{r['model']} {r['metric']} {r['dataset']} {r['hop']} sign {r['sign_main']}->{r['sign_A']} holm_change {r['holm_change']}"
            for r in ch
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
