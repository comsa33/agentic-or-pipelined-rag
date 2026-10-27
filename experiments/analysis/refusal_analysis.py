"""Answer refusal detection and refusal-conditioned F1 analysis.

Quantifies how often a pipeline declines to answer ("Cannot determine",
"The provided passages do not contain ...") and decomposes F1 into
refused and non-refused subsets. This separates *answer quality* from
*abstention behavior*, which is required to interpret the model-pipeline
interaction effect between reasoning models and agentic pipelines.

The refusal marker set is defined once here (``REFUSAL_PATTERNS``) so that
every reported refusal number in the paper traces back to a single
executable definition.

Usage:
    from experiments.analysis.refusal_analysis import RefusalAnalyzer

    analyzer = RefusalAnalyzer.from_default_trees()
    analyzer.print_refusal_table()
    analyzer.to_csv("paper/supplementary/refusal_rates.csv")
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from loguru import logger

from experiments.analysis.runs import DATASETS, best_f1, load_default_trees, load_runs

# ---------------------------------------------------------------------------
# Refusal marker definition (single source of truth)
# ---------------------------------------------------------------------------
# Surface forms observed across Gemini Flash Lite and gpt-5-mini outputs: the
# set every number reported before 2026-10-23 was computed with. Kept as its
# own tuple so a recount can compare against it (refusal_pattern_recount.py).
# Matched case-insensitively against the raw prediction string.
LEGACY_REFUSAL_PATTERNS: tuple[str, ...] = (
    r"cannot determine",
    r"can ?not be determined",
    r"cannot be determined",
    r"unable to determine",
    r"impossible to determine",
    r"cannot answer",
    r"unable to answer",
    r"insufficient information",
    r"not enough information",
    r"no information",
    r"information (?:is )?not available",
    r"not available in the",
    r"do(?:es)? not contain",
    r"do(?:es)? not provide",
    r"do(?:es)? not specify",
    r"do(?:es)? not mention",
    r"do(?:es)? not include",
    r"not (?:mentioned|specified|provided|stated|present|found) in the",
    r"lacks? (?:the )?(?:necessary |sufficient )?information",
    r"no (?:relevant )?(?:passage|document|context)",
    r"unanswerable",
)

# Claude Sonnet 5.5's surface forms, from the Sonnet smoke run of 2026-10-23.
SONNET_REFUSAL_PATTERNS: tuple[str, ...] = (
    r"do(?:es)? not say",
    "don[\u2019']?t say",
    r"do(?:es)? not state",
    "don[\u2019']?t give this",
    r"do(?:es)? not give enough information",
    r"cannot be computed from the passages",
)

# Generic hedge forms, added once on 2026-10-25 from the ten
# hedged answers of the three Sonnet smokes, before any grid row was scored.
# The set is frozen after this: no further surface forms are added.
HEDGE_REFUSAL_PATTERNS: tuple[str, ...] = (
    r"passages do not (?:\w+ly )?(?:give|establish|confirm|link|explain)",
    r"not in the passages",
    r"outside knowledge",
    r"\bgive no\b",
)

REFUSAL_PATTERNS: tuple[str, ...] = (
    LEGACY_REFUSAL_PATTERNS + SONNET_REFUSAL_PATTERNS + HEDGE_REFUSAL_PATTERNS
)

_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def _prediction_text(prediction: str | None) -> str:
    """A prediction as text, as experiments.common records it.

    Kept here because importing experiments.common pulls in litellm. Rows
    checkpointed by earlier code versions carry None where a refusal stood.
    """
    return "" if prediction is None else str(prediction)


def is_refusal(prediction: str) -> bool:
    """Return True if the prediction declines to answer."""
    return bool(_REFUSAL_RE.search(prediction or ""))


class RefusalAnalyzer:
    """Refusal rate and refusal-conditioned F1 across pipelines and models."""

    def __init__(self, records: list[dict]) -> None:
        """Initialize with flat records: model, dataset, pipeline, rows."""
        self.records = records

    @classmethod
    def from_results_dir(
        cls, results_dir: str | Path, prefix: str = "rq1", model: str | None = None
    ) -> RefusalAnalyzer:
        """Load every ``{prefix}_*.jsonl`` under ``results_dir`` (see ``runs.load_runs``)."""
        return cls(load_runs(results_dir, prefix=prefix, model=model))

    @classmethod
    def from_default_trees(cls, prefix: str = "rq1") -> RefusalAnalyzer:
        """Load the three model trees behind the paper's tables."""
        return cls(load_default_trees(prefix=prefix))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def compute(self) -> list[dict]:
        """Compute refusal rate and conditioned F1 for every run."""
        out: list[dict] = []
        for rec in self.records:
            rows = rec["rows"]
            if not rows:
                continue
            scored = [(_prediction_text(r.get("prediction")), r) for r in rows]
            refused = [(p, r) for p, r in scored if is_refusal(p)]
            answered = [(p, r) for p, r in scored if not is_refusal(p)]

            out.append(
                {
                    "model": rec["model"],
                    "dataset": rec["dataset"],
                    "pipeline": rec["pipeline"],
                    "total": len(scored),
                    "refusals": len(refused),
                    "refusal_rate": round(len(refused) / len(scored), 4),
                    "f1_all": round(float(np.mean([best_f1(p, r) for p, r in scored])), 4),
                    "f1_non_refused": round(float(np.mean([best_f1(p, r) for p, r in answered])), 4)
                    if answered
                    else None,
                    "f1_refused": round(float(np.mean([best_f1(p, r) for p, r in refused])), 4)
                    if refused
                    else None,
                }
            )
        return out

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def print_refusal_table(self) -> None:
        """Print refusal rates and conditioned F1 grouped by model and dataset."""
        rows = self.compute()
        for model in sorted({r["model"] for r in rows}):
            logger.info(f"\n=== Refusal analysis: {model} ===")
            logger.info(
                f"{'dataset':<17}{'pipeline':<20}{'refusal':>9}"
                f"{'F1(all)':>10}{'F1(non-ref)':>13}{'F1(ref)':>10}"
            )
            for dataset in DATASETS:
                for r in [x for x in rows if x["model"] == model and x["dataset"] == dataset]:
                    non_ref = (
                        f"{r['f1_non_refused']:.3f}" if r["f1_non_refused"] is not None else "-"
                    )
                    ref = f"{r['f1_refused']:.3f}" if r["f1_refused"] is not None else "-"
                    logger.info(
                        f"{r['dataset']:<17}{r['pipeline']:<20}"
                        f"{100 * r['refusal_rate']:>8.1f}%{r['f1_all']:>10.3f}{non_ref:>13}{ref:>10}"
                    )

    def to_csv(self, out_path: str | Path) -> Path:
        """Write the full refusal table to CSV for supplementary materials."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.compute()
        fields = [
            "model",
            "dataset",
            "pipeline",
            "total",
            "refusals",
            "refusal_rate",
            "f1_all",
            "f1_non_refused",
            "f1_refused",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Refusal rates written to {out_path}")
        return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Refusal rate analysis")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="One result tree; default is all three model trees (runs.DEFAULT_TREES)",
    )
    parser.add_argument("--model", default=None, help="Model label for a flat-layout tree")
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    if args.results_dir:
        analyzer = RefusalAnalyzer.from_results_dir(args.results_dir, model=args.model)
    else:
        analyzer = RefusalAnalyzer.from_default_trees()
    analyzer.print_refusal_table()
    if args.csv:
        analyzer.to_csv(args.csv)
