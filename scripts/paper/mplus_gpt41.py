#!/usr/bin/env python3
"""Within-family Agentic - Loop judge contrasts on the questions common to both families (M+), gpt-4.1 judge.

controlled_grid_analysis.py's M+ suite has the judge rows under gpt-4.1-nano
only. This runs that script unchanged with --judge gpt-4-1_39118f into a
temporary directory, so the M+ question sets, the bootstrap (10,000, seed 42,
same comparison_rng keys) and the Holm families (judge metric, non-descriptive
hop cells) are the script's own. It then

1. checks that the script's main-suite judge rows under gpt-4.1 equal the
   gpt41_* columns of judge_grid_contrasts_by_judge.csv (Agentic - Loop), the source of
   the Judge rows of tab:rq1_main;
2. writes judge_grid_common_questions.csv: per cell the M+ contrast under gpt-4.1 next to the
   main-set contrast under gpt-4.1, and whether the sign verdict (95% CI) or
   the Holm verdict changes.

Uses only the existing judged_gpt-4-1_39118f files; no API call; no
prediction is printed.

Run judge_gpt41_grid.py first (it writes judge_grid_contrasts_by_judge.csv).

    uv run python scripts/paper/mplus_gpt41.py
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "scripts" / "paper" / "out"
JUDGE = "gpt-4-1_39118f"
KEY = ("model", "dataset", "hop")


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def holm_sig(r: dict) -> str:
    return "" if r["descriptive"] == "True" else str(float(r["p_holm_final"]) < 0.05)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--code-repo", type=Path, default=REPO)
    ap.add_argument("--out", type=Path, default=OUT / "judge_gpt41_grid")
    args = ap.parse_args()
    grid_script = Path(__file__).resolve().parent / "controlled_grid_analysis.py"

    with tempfile.TemporaryDirectory(prefix="mplus_gpt41_") as tmp:
        tmp = Path(tmp)
        subprocess.run(
            [
                sys.executable,
                str(grid_script),
                "--code-repo",
                str(args.code_repo.resolve()),
                "--out",
                str(tmp / "out"),
                "--judge",
                JUDGE,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        main_rows = [
            r for r in read(tmp / "out/grid_f1_agentic_minus_loop.csv") if r["metric"] == "judge"
        ]
        mplus_rows = [
            r
            for r in read(tmp / "out/grid_f1_agentic_minus_loop_common_questions.csv")
            if r["metric"] == "judge"
        ]

    # 1. main suite under gpt-4.1 == judge_grid_contrasts_by_judge.csv gpt41 columns (Agentic - Loop)
    ref = {
        tuple(r[k] for k in KEY): r
        for r in read(args.out / "judge_grid_contrasts_by_judge.csv")
        if r["contrast"] == "Agentic-Loop"
    }
    for r in main_rows:
        c = ref[tuple(r[k] for k in KEY)]
        for mine, theirs in (
            ("delta", "gpt41_delta"),
            ("ci_lo", "gpt41_ci_lo"),
            ("ci_hi", "gpt41_ci_hi"),
            ("sign", "gpt41_sign"),
            ("n", "n"),
        ):
            a, b = r[mine], c[theirs]
            same = a == b if mine in ("sign", "n") else abs(float(a) - float(b)) < 1e-9
            if not same:
                raise SystemExit(
                    f"main judge row differs from contrasts_by_judge: {r['model']} {r['dataset']} "
                    f"{r['hop']} {mine} {a} vs {b}"
                )
        if (
            r["descriptive"] != "True"
            and abs(float(r["p_holm_final"]) - float(c["gpt41_p_holm_final"])) > 1e-9
        ):
            raise SystemExit(f"Holm differs: {r['model']} {r['dataset']} {r['hop']}")

    # 2. M+ next to main
    main_by = {tuple(r[k] for k in KEY): r for r in main_rows}
    out_rows = []
    for r in mplus_rows:
        d = main_by[tuple(r[k] for k in KEY)]
        out_rows.append(
            {
                "model": r["model"],
                "dataset": r["dataset"],
                "hop": r["hop"],
                "runs": r["runs"],
                "descriptive": r["descriptive"],
                "mplus_n": r["n"],
                "mplus_judge_agentic": r["mean_a"],
                "mplus_judge_loop": r["mean_b"],
                "mplus_judge_delta": r["delta"],
                "mplus_judge_ci_lo": r["ci_lo"],
                "mplus_judge_ci_hi": r["ci_hi"],
                "mplus_judge_p": r["p"],
                "mplus_judge_p_holm_final": r["p_holm_final"],
                "mplus_judge_holm_sig": holm_sig(r),
                "mplus_judge_sign": r["sign"],
                "main_n": d["n"],
                "main_judge_delta": d["delta"],
                "main_judge_ci_lo": d["ci_lo"],
                "main_judge_ci_hi": d["ci_hi"],
                "main_judge_p_holm_final": d["p_holm_final"],
                "main_judge_holm_sig": holm_sig(d),
                "main_judge_sign": d["sign"],
                "sign_change": r["sign"] != d["sign"],
                "holm_change": holm_sig(r) != holm_sig(d),
                "holm_family_size": r["holm_family_size"],
            }
        )
    with open(
        args.out / "judge_grid_common_questions.csv", "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(
        f"main judge rows checked {len(main_rows)}; M+ rows {len(out_rows)}; "
        f"sign changes {sum(r['sign_change'] for r in out_rows)}; "
        f"Holm changes {sum(r['holm_change'] for r in out_rows)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
