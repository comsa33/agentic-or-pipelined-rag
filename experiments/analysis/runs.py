"""Load per-question result rows across the per-model result trees.

Every analysis that compares pipelines across models needs the same thing:
the ``rq1_<pipeline>.jsonl`` rows of each run, tagged with the model,
dataset and pipeline they belong to. Two directory layouts exist:

* **stamped** — ``<stamp>_<prefix>_<dataset>_n<N>_<model-tag>/`` under
  ``data/results/paper`` (Gemini and gpt-5-mini, March 2026);
* **flat** — ``<dataset>/`` under ``data/results-openweight/final``
  (gpt-oss:120b, July 2026), where the model is not in the path and must
  be supplied by the caller.

``*_judged*.jsonl`` files are judge re-scorings of the same rows and are
skipped, so that a pipeline is never counted twice.

Campaign runs (``data/results-campaign``) use the stamped layout with a
``_q<hash>`` and/or ``_k<n>`` suffix; their repeat index is kept on the
record. Error rows are dropped, except that an item whose error was
classified ``truncated`` counts as a wrong answer (empty prediction), as the
analysis plan fixed before the grid ran. Questions listed in the tree's
``exclusions.json`` for the run's model are removed from every pipeline,
and the number removed is kept on the record.

Judge scores live in ``*_judged_<tag>.jsonl`` files, which carry no
``error`` key: an error row appears there as an empty prediction scored 0.
With ``judge=<tag>`` the scores are joined onto the rows chosen above by
question id, so a judged file never decides on its own which rows count.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from loguru import logger

from experiments.exclusions import excluded_ids, load_exclusions

DATASETS = ("hotpotqa", "2wikimultihopqa", "musique", "financebench")

# Model tag as it appears in a stamped directory name -> label used in tables.
STAMPED_MODEL_TAGS = {
    "gemini-3.1-flash-lite": "gemini-flash-lite",
    "gpt-5-mini": "gpt-5-mini",
    "claude-sonnet-5-5": "claude-sonnet-5-5",
}

# Campaign run-id suffixes after the model tag: named question set, repeat.
_RUN_SUFFIX = re.compile(r"(?:_q[0-9a-f]{8})?(?:_k(\d+))?$")

# The three trees behind every three-model table in the paper.
DEFAULT_TREES: tuple[tuple[str, str | None], ...] = (
    ("data/results/paper", None),
    ("data/results-openweight/final", "gpt-oss-120b"),
)


def analysis_rows(
    rows: list[dict], excluded: set[str] | None = None, source: str = ""
) -> tuple[list[tuple[int, dict]], int]:
    """The rows an analysis uses, with their file positions, and the number excluded.

    Excluded questions leave; error rows leave, except a ``truncated`` one,
    which stays as a wrong answer (empty prediction, ``scored_wrong``).
    """
    kept: list[tuple[int, dict]] = []
    n_excluded = 0
    for position, r in enumerate(rows):
        if excluded and str(r.get("id")) in excluded:
            n_excluded += 1
            continue
        if "error" not in r:
            kept.append((position, r))
        elif r.get("error_class") == "truncated":
            # The model's own outcome: scored wrong, not dropped.
            kept.append((position, {**r, "prediction": "", "scored_wrong": True}))
        elif r.get("error_class") == "platform_filtered":
            logger.warning(
                f"{source}: question {r.get('id')} was blocked by the platform filter "
                f"but is not in exclusions.json for this model"
            )
        if r.get("refusal_seen"):
            logger.warning(
                f"{source}: question {r.get('id')} had a refusal but is not in "
                f"exclusions.json for this model"
            )
    return kept, n_excluded


def exclusions_for_run_dir(run_dir: Path) -> set[str] | None:
    """Excluded ids for a stamped run directory, from its tree's exclusions.json."""
    path = run_dir.parent / "exclusions.json"
    dataset = next((d for d in DATASETS if d in run_dir.name), None)
    model = _model_from_stamped_name(run_dir.name)
    if not path.is_file() or dataset is None or model is None:
        return None
    return excluded_ids(load_exclusions(path), dataset, model)


def _read_rows(jsonl_path: Path, excluded: set[str] | None = None) -> tuple[list[dict], int]:
    """Rows for analysis, and how many were removed as excluded questions."""
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kept, n_excluded = analysis_rows(rows, excluded, str(jsonl_path))
    return [r for _, r in kept], n_excluded


def _judge_scores(jsonl_path: Path, judge: str) -> dict[str, float]:
    """Usable judge scores by question id from the pipeline's judged file."""
    judged = jsonl_path.with_name(f"{jsonl_path.stem}_judged_{judge}.jsonl")
    if not judged.is_file():
        return {}
    scores: dict[str, float] = {}
    for line in judged.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("judge_status") == "ok" and r.get("llm_judge") is not None:
            scores[str(r.get("id"))] = float(r["llm_judge"])
    return scores


def _join_judge(rows: list[dict], scores: dict[str, float]) -> int:
    """Set each row's llm_judge from `scores`; returns rows left without one."""
    missing = 0
    for r in rows:
        if r.get("scored_wrong"):
            r["llm_judge"] = 0.0
            continue
        score = scores.get(str(r.get("id")))
        r["llm_judge"] = score
        missing += score is None
    return missing


def _split_stamped_name(name: str) -> tuple[str, int | None]:
    """The run name without its campaign suffix, and its repeat index."""
    match = _RUN_SUFFIX.search(name)
    k = int(match.group(1)) if match and match.group(1) else None
    return name[: match.start()] if match else name, k


def _model_from_stamped_name(name: str) -> str | None:
    base, _ = _split_stamped_name(name)
    for tag, label in STAMPED_MODEL_TAGS.items():
        if base.endswith(tag):
            return label
    return None


def load_runs(
    results_dir: str | Path,
    prefix: str = "rq1",
    model: str | None = None,
    exclusions_path: str | Path | None = None,
    apply_exclusions: bool = True,
    judge: str | None = None,
    require_paired_n: bool = True,
) -> list[dict]:
    """Return ``{model, dataset, pipeline, repeat_index, rows, n_excluded}`` per run.

    Args:
        results_dir: root of one result tree (either layout).
        prefix: experiment prefix of the jsonl files, e.g. ``rq1``.
        model: label to use for the flat layout, where the path carries none.
        exclusions_path: exclusions file to apply; defaults to the tree's own
            ``exclusions.json`` when it has one.
        apply_exclusions: False to keep excluded questions (e.g. to compare
            with the summaries written at run time).
        judge: judge tag (e.g. ``gpt-4-1-nano_7bcf05``) whose scores to join
            onto the rows as ``llm_judge``; the record then also carries
            ``n_judge_missing``.
        require_paired_n: raise unless every pipeline of a (model, dataset,
            repeat) cell has the same n (see ``check_paired_n``).
    """
    results_dir = Path(results_dir)
    records: list[dict] = []
    if exclusions_path is None and (results_dir / "exclusions.json").is_file():
        exclusions_path = results_dir / "exclusions.json"
    exclusions = load_exclusions(exclusions_path) if apply_exclusions and exclusions_path else []

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue

        if subdir.name in DATASETS:
            # flat layout
            dataset = subdir.name
            repeat_index = None
            run_model = model
            if run_model is None:
                logger.warning(f"Skipping {subdir}: flat layout needs an explicit model label")
                continue
        elif f"_{prefix}_" in subdir.name:
            # stamped layout
            repeat_index = _split_stamped_name(subdir.name)[1]
            dataset = next((d for d in DATASETS if d in subdir.name), None)
            run_model = model or _model_from_stamped_name(subdir.name)
            if dataset is None or run_model is None:
                logger.warning(f"Skipping (dataset/model not recognized): {subdir.name}")
                continue
        else:
            continue

        excluded = excluded_ids(exclusions, dataset, run_model) if exclusions else None
        for jsonl_path in sorted(subdir.glob(f"{prefix}_*.jsonl")):
            if "_judged" in jsonl_path.stem or jsonl_path.stem.endswith("_summary"):
                continue
            rows, n_excluded = _read_rows(jsonl_path, excluded)
            if n_excluded:
                logger.info(f"{jsonl_path}: {n_excluded} excluded question(s) removed")
            record = {
                "model": run_model,
                "dataset": dataset,
                "pipeline": jsonl_path.stem[len(prefix) + 1 :],
                "repeat_index": repeat_index,
                "rows": rows,
                "n_excluded": n_excluded,
            }
            if judge:
                record["n_judge_missing"] = _join_judge(rows, _judge_scores(jsonl_path, judge))
            records.append(record)

    logger.info(f"Loaded {len(records)} pipeline runs from {results_dir}")
    if require_paired_n:
        check_paired_n(records)
    return records


def comparison_rng(seed: int, *key: str) -> np.random.Generator:
    """A random generator of its own for one comparison, derived from its key.

    One generator shared across comparisons made each result depend on how
    many comparisons ran before it, i.e. on the order pipelines were read.
    """
    import hashlib

    digest = hashlib.sha256("|".join((str(seed), *key)).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


class UnpairedCellError(ValueError):
    """Pipelines of one (model, dataset, repeat) cell were scored on different n."""


def check_paired_n(records: list[dict]) -> None:
    """Every pipeline of a (model, dataset, repeat) cell must have the same n.

    Comparisons between pipelines are paired; a cell whose pipelines lost
    different questions (an exclusion applied to one pipeline only, an
    error row dropped from one) would compare different question sets.
    """
    cells: dict[tuple, dict[str, int]] = {}
    for rec in records:
        key = (rec["model"], rec["dataset"], rec.get("repeat_index"))
        cells.setdefault(key, {})[rec["pipeline"]] = len(rec["rows"])
    unpaired = {key: ns for key, ns in cells.items() if len(set(ns.values())) > 1}
    if unpaired:
        lines = [f"{key}: {ns}" for key, ns in sorted(unpaired.items(), key=str)]
        raise UnpairedCellError("pipelines differ in n within a cell:\n" + "\n".join(lines))


def load_default_trees(prefix: str = "rq1") -> list[dict]:
    """Load the Gemini, gpt-5-mini and gpt-oss trees that back the paper's tables."""
    records: list[dict] = []
    for root, model in DEFAULT_TREES:
        if Path(root).is_dir():
            records.extend(load_runs(root, prefix=prefix, model=model))
        else:
            logger.warning(f"Result tree missing: {root}")
    return records


def best_f1(prediction: str, row: dict) -> float:
    """Token F1 against the best-matching reference answer."""
    from agentic_rag.evaluation.metrics import token_f1

    references = row.get("all_references") or [row.get("reference", "")]
    return max(token_f1(prediction, ref) for ref in references if ref is not None)
