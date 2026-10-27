"""Recount answer-level refusals under the legacy and the current pattern sets.

REFUSAL_PATTERNS gained Claude Sonnet 5.5's surface forms on 2026-10-23 and
four generic hedge forms on 2026-10-25. The gpt-5-mini campaign was
analysed with the legacy set, so this script reports what the change does to
those rows: refusal counts by dataset, pipeline and repeat index k under both
sets, and every row whose verdict differs, with the number of distinct
questions behind them. It only reads result files.

It prints the prediction of every changed row, so name the gpt-5-mini runs
explicitly: a bare n200 glob now also matches the Sonnet grid.

Error rows are skipped: their prediction is not an answer.

Usage:
    uv run python experiments/analysis/refusal_pattern_recount.py \\
        data/results-campaign/2026*_n200_gpt-5-mini_k* \\
        data/results-campaign/2026*_n150_gpt-5-mini_k*
    uv run python experiments/analysis/refusal_pattern_recount.py <run_dir> ... --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.analysis.refusal_analysis import LEGACY_REFUSAL_PATTERNS, REFUSAL_PATTERNS
from experiments.analysis.runs import DATASETS

_LEGACY_RE = re.compile("|".join(LEGACY_REFUSAL_PATTERNS), re.IGNORECASE)
_CURRENT_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)
_K_RE = re.compile(r"_k(\d+)(?:_|$)")


@dataclass(frozen=True)
class Changed:
    dataset: str
    pipeline: str
    k: str
    id: str
    match: str
    prediction: str


def _run_meta(run_dir: Path) -> tuple[str, str]:
    dataset = next((d for d in DATASETS if d in run_dir.name), "unknown")
    m = _K_RE.search(run_dir.name)
    return dataset, f"k{m.group(1)}" if m else "k?"


def _result_files(run_dir: Path) -> list[Path]:
    return sorted(
        p for p in run_dir.glob("*.jsonl") if "_judged" not in p.name and "checkpoint" not in p.name
    )


def recount(run_dirs: list[Path]) -> tuple[dict, list[Changed]]:
    """Counts by (dataset, pipeline, k) and the rows whose verdict changed.

    Counts are [rows, legacy refusals, current refusals].
    """
    counts: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    changed: list[Changed] = []
    for run_dir in run_dirs:
        dataset, k = _run_meta(run_dir)
        for path in _result_files(run_dir):
            pipeline = path.stem.split("_", 1)[1] if "_" in path.stem else path.stem
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if "error" in row:
                    continue
                prediction = str(row.get("prediction") or "")
                legacy = bool(_LEGACY_RE.search(prediction))
                current = _CURRENT_RE.search(prediction)
                c = counts[(dataset, pipeline, k)]
                c[0] += 1
                c[1] += legacy
                c[2] += bool(current)
                if legacy != bool(current):
                    changed.append(
                        Changed(
                            dataset,
                            pipeline,
                            k,
                            str(row.get("id")),
                            current.group(0) if current else "",
                            prediction,
                        )
                    )
    return dict(counts), changed


def _print(counts: dict, changed: list[Changed]) -> None:
    print(f"{'dataset':16s} {'pipeline':18s} {'k':3s} {'rows':>5s} {'legacy':>7s} {'current':>8s}")
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for (dataset, pipeline, k), (n, old, new) in sorted(counts.items()):
        print(f"{dataset:16s} {pipeline:18s} {k:3s} {n:5d} {old:7d} {new:8d}")
        t = totals[dataset]
        t[0], t[1], t[2] = t[0] + n, t[1] + old, t[2] + new
    print()
    print(
        f"{'dataset':16s} {'rows':>5s} {'legacy':>7s} {'current':>8s} {'changed':>8s} {'questions':>9s}"
    )
    for dataset, (n, old, new) in sorted(totals.items()):
        rows = [c for c in changed if c.dataset == dataset]
        questions = len({c.id for c in rows})
        print(f"{dataset:16s} {n:5d} {old:7d} {new:8d} {len(rows):8d} {questions:9d}")
    print()
    print(
        f"Rows whose verdict differs: {len(changed)} over {len({(c.dataset, c.id) for c in changed})} question(s)"
    )
    for c in changed:
        snippet = c.prediction[:120].replace("\n", " ")
        print(f"  {c.dataset} {c.pipeline} {c.k} {c.id} | {c.match!r} | {snippet}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path, help="Also write the changed rows here")
    args = parser.parse_args(argv)

    counts, changed = recount(args.run_dirs)
    _print(counts, changed)
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "pipeline", "k", "id", "match", "prediction"])
            for c in changed:
                w.writerow([c.dataset, c.pipeline, c.k, c.id, c.match, c.prediction])
    return 0


if __name__ == "__main__":
    sys.exit(main())
