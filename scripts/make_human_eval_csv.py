"""Build a blind labelling sheet for validating the LLM judge.

Cohen's kappa only means something if the human labels are produced without
sight of anything that identifies the answer's source, so the sheet carries
neither the judge's verdict nor the pipeline that produced the answer. An
earlier version shuffled the rows for exactly that reason but still printed
`pipeline` as a column, which let a labeller read off `agentic_(react)` and
told them nothing was actually blinded.

The sheet therefore carries only an opaque sample id. The mapping back to
(dataset, pipeline, id) is written to a separate key file that must not be
given to the labellers; verdicts are joined on it after the sheet returns.

Sampling is stratified across pipelines so the sheet spans the full range of
answer quality — judging only the strongest pipeline would measure agreement
on easy cases and miss where the judge actually struggles.

Usage:
    uv run python scripts/make_human_eval_csv.py --per-dataset 50 --out human_eval.csv
    # writes human_eval.csv (give to labellers) and human_eval_key.csv (do not)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

DATASETS = ("2wikimultihopqa", "hotpotqa", "musique", "financebench")
PIPELINES = (
    "naive_rag",
    "crag_replica",
    "single-pass",
    "loop_refinement",
    "agentic_(react)",
)


def load(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def build(results_root: Path, per_dataset: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    per_pipeline = per_dataset // len(PIPELINES)

    for dataset in DATASETS:
        directory = results_root / dataset
        if not directory.exists():
            continue
        for pipeline in PIPELINES:
            path = directory / f"rq1_{pipeline}.jsonl"
            if not path.exists():
                continue
            records = [r for r in load(path) if "error" not in r]
            for r in rng.sample(records, min(per_pipeline, len(records))):
                rows.append(
                    {
                        "dataset": dataset,
                        "pipeline": pipeline,
                        "id": r["id"],
                        "question": r.get("question", ""),
                        "reference_answer": r.get("reference", ""),
                        "predicted_answer": str(r.get("prediction", "")),
                    }
                )

    # Shuffle before numbering so the sample id carries no ordering information
    # either — consecutive ids would otherwise group by pipeline.
    rng.shuffle(rows)
    for n, row in enumerate(rows, 1):
        row["sample_id"] = f"S{n:04d}"
    return rows


SHEET_COLUMNS = ("sample_id", "question", "reference_answer", "predicted_answer", "human_label")
KEY_COLUMNS = ("sample_id", "dataset", "pipeline", "id")


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build blind human-eval sheet")
    parser.add_argument("--results-root", default="data/results-openweight/final")
    parser.add_argument("--per-dataset", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="human_eval.csv")
    args = parser.parse_args()

    rows = build(Path(args.results_root), args.per_dataset, args.seed)
    if not rows:
        raise SystemExit(f"No results found under {args.results_root}")

    out = Path(args.out)
    key_out = out.with_name(f"{out.stem}_key{out.suffix}")
    for row in rows:
        row["human_label"] = ""  # 1 = correct, 0 = incorrect
    write_csv(out, SHEET_COLUMNS, rows)
    write_csv(key_out, KEY_COLUMNS, rows)

    by_dataset: dict[str, int] = {}
    for r in rows:
        by_dataset[r["dataset"]] = by_dataset.get(r["dataset"], 0) + 1
    print(f"{len(rows)} rows -> {out}")
    for dataset, count in by_dataset.items():
        print(f"  {dataset}: {count}")
    print(f"\nGive the labeller: {out}  —  fill human_label with 1 (correct) / 0 (incorrect).")
    print(f"Never give the labeller: {key_out}  (sample_id -> dataset/pipeline/id)")
    print(
        "The sheet carries neither judge verdicts nor pipeline names; showing either voids the agreement."
    )


if __name__ == "__main__":
    main()
