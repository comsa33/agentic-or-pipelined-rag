"""Shared experiment utilities.

Common setup, data loading, result saving, and reporting
functions used across all RQ and ablation experiment scripts.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import dspy
import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table

from agentic_rag.config.settings import make_lm, settings
from agentic_rag.evaluation.cost_tracker import LiteLLMMeter, empty_usage
from agentic_rag.evaluation.metrics import evaluate_batch
from agentic_rag.pipeline.base import BasePipeline
from agentic_rag.retriever.hybrid import HybridRetriever
from agentic_rag.retriever.indexer import DocumentIndexer
from experiments.difficulty import question_difficulty as _extract_question_difficulty
from experiments.exclusions import refusal_final
from experiments.manifest import git_state

console = Console()

# Process-wide usage meter, installed by `setup_experiment`. Every LLM call made
# through litellm after that point is recorded here, which is what lets a
# result row carry the measured tokens and cost of its own question.
_meter: LiteLLMMeter | None = None

# How long to wait after a question for litellm's late success callbacks.
# A question whose calls have not all reported by then is marked
# usage_complete=False and its missing calls are written off (see
# LiteLLMMeter.drain), never counted into the next question.
DRAIN_TIMEOUT_SECONDS = 10.0


def get_meter() -> LiteLLMMeter | None:
    """The installed usage meter, or None before `setup_experiment` ran."""
    return _meter


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def setup_experiment(seed: int | None = None) -> None:
    """Initialize experiment environment with reproducible seed."""
    global _meter
    seed = seed or settings.experiment.seed
    random.seed(seed)
    np.random.seed(seed)

    _meter = LiteLLMMeter().install()

    # Turn the response cache off before any LM is built. At temperature=0 a
    # rerun would otherwise replay cached completions in milliseconds, which
    # leaves accuracy intact but makes measured latency meaningless.
    if settings.disable_llm_cache:
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
        logger.info("LLM response cache disabled (latency measurement mode)")

    # Configure DSPy LM — API keys are read from environment variables
    # (OPENAI_API_KEY, GEMINI_API_KEY, etc.) by litellm automatically.
    # make_lm centralizes retry/timeout settings.
    dspy.configure(lm=make_lm(settings.model.generate_model))

    logger.info(f"Experiment initialized: seed={seed}")


def load_dataset(name: str, sample_size: int | None = None) -> list[dict]:
    """Load a prepared dataset from data/raw/."""
    path = settings.raw_dir / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Run: uv run python scripts/prepare_datasets.py --dataset {name}"
        )

    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line.strip()))

    if sample_size and len(items) > sample_size:
        items = random.sample(items, sample_size)
        logger.info(f"Sampled {sample_size} from {name}")

    logger.info(f"Loaded dataset '{name}': {len(items)} items")
    return items


DATASET_INDEX_MAP = {
    "hotpotqa": "hotpotqa",
    "2wikimultihopqa": "2wikimultihopqa",
    "musique": "musique",
    "financebench": "financebench",
}


def load_retriever(
    index_dir: Path | None = None, dataset_name: str | None = None
) -> tuple[HybridRetriever, DocumentIndexer]:
    """Load pre-built retrieval indices.

    Args:
        index_dir: Explicit index directory. Takes precedence.
        dataset_name: Dataset name to look up in data/indices/{name}/.
    """
    if index_dir is None and dataset_name:
        index_name = DATASET_INDEX_MAP.get(dataset_name, dataset_name)
        index_dir = settings.index_dir / index_name
    index_dir = index_dir or settings.index_dir
    indexer = DocumentIndexer()
    retriever = indexer.load(index_dir)
    return retriever, indexer


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------
# Error classes that are kept, not retried, when a run is resumed. A refusal
# or a reply cut off at max_tokens is the model's outcome for the question,
# and models that are sent no temperature (Claude 5.x) answer differently on
# each call, so rerunning such an item until it succeeds would make whether
# it counts as wrong depend on how often the run was resumed.
FINAL_ERROR_CLASSES = frozenset({"refused", "truncated"})


# HTTP statuses of provider errors that say nothing about the question.
_TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 529})


def is_transient_error(exc: BaseException | None) -> bool:
    """Whether `exc`, or an exception it was raised from, is a transient
    provider error: rate limit, server error, timeout or lost connection."""
    import litellm

    transient = (
        litellm.RateLimitError,
        litellm.InternalServerError,
        litellm.ServiceUnavailableError,
        litellm.Timeout,
        litellm.APIConnectionError,
    )
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, transient) or getattr(exc, "status_code", None) in _TRANSIENT_STATUSES:
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def is_platform_filtered(exc: BaseException | None) -> bool:
    """Whether `exc`, or an exception it was raised from, is a provider's
    content filter blocking the request (Azure OpenAI content_filter)."""
    import litellm

    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, litellm.ContentPolicyViolationError):
            return True
        if getattr(exc, "code", None) == "content_filter":
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def classify_error(usage: dict, exc: BaseException | None = None) -> str:
    """Why an item failed, from its final attempt's exception and usage.

    other: the attempt ended in a transient provider error, whatever its
    calls returned before; retried within the run and on resume.
    platform_filtered: otherwise, the provider's content filter blocked the
    request. Retried like other (a later attempt can build a prompt the
    filter passes), so only a question blocked to the end is recorded as
    such; it is then excluded for that model (see experiments/exclusions.py).
    refused: otherwise, a refusal stood (see `refusal_final`); a refusal
    that a resend recovered does not make the item refused.
    truncated: otherwise, a call stopped at max_tokens.
    other: anything else.
    """
    if is_transient_error(exc):
        return "other"
    if is_platform_filtered(exc):
        return "platform_filtered"
    if refusal_final(usage):
        return "refused"
    if int(usage.get("truncated_calls") or 0) > 0:
        return "truncated"
    return "other"


def refusal_seen(usage: dict, prior_usage: dict | None) -> bool:
    """Whether any attempt at the item had a refusal that stood (main analysis).

    The trigger for excluding a question, on every row: a refusal can sit
    in an earlier attempt or in a row that still produced an answer. A
    refusal recovered by a resend does not trigger it; `refusal_first_seen`
    does.
    """
    return any(refusal_final(u) for u in (usage, prior_usage) if u)


def refusal_first_seen(usage: dict, prior_usage: dict | None) -> bool:
    """Whether any call at the item was refused on its first send (sensitivity analysis).

    Resends follow only a refusal, so any refused call implies a refused
    first send.
    """
    return any(int((u or {}).get("refused_calls") or 0) > 0 for u in (usage, prior_usage))


def _attempt_usage(meter, mark, leaked_before, pipeline_name, i, item_id) -> dict:
    """Usage of the calls made since `mark`, once every one has reported."""
    if meter is None:
        return empty_usage()
    # Read the meter only after every call started by this attempt has
    # reported; litellm delivers success callbacks late.
    complete = meter.drain(DRAIN_TIMEOUT_SECONDS)
    usage = meter.usage_since(mark)
    # A timed-out call may have been billed without reporting usage.
    usage["usage_complete"] = complete and not usage["timeout_calls"]
    usage["leaked_calls"] = meter.leaked_calls - leaked_before
    if not complete:
        logger.warning(
            f"[{pipeline_name}] Item {i} ({item_id}): usage incomplete, "
            f"{usage['leaked_calls']} call(s) unreported"
        )
    return usage


def _prior_attempt_usage(meter, start, end, leaked_start, leaked_end) -> dict:
    usage = meter.usage_between(start, end)
    usage["leaked_calls"] = leaked_end - leaked_start
    return usage


def run_pipeline_on_dataset(
    pipeline: BasePipeline,
    dataset: list[dict],
    pipeline_name: str = "pipeline",
    request_delay: float = 0.0,
    checkpoint_dir: Path | None = None,
    max_item_retries: int = 2,
    retry_backoff: float = 30.0,
    run_id: str | None = None,
    repeat_index: int | None = None,
) -> list[dict]:
    """Run a pipeline on a dataset and collect results.

    Every row is stamped with the run it was produced in (`run_id`, the git
    commit and the cache flag). A resumed run reuses rows written by an
    earlier process, and those rows keep their own stamp, so a result file
    can always say which of its rows were made under which conditions.

    Resumption is keyed on question id rather than position, and only items
    that completed successfully are treated as done. An item that failed —
    typically a provider rate limit — is retried on the next run instead of
    being frozen into the results as a permanent gap.

    Args:
        request_delay: Seconds to wait between items (for API rate limiting).
        checkpoint_dir: If provided, writes results after every item to
            ``checkpoint_dir/<pipeline_name>_checkpoint.jsonl``. Rerunning the
            same command resumes from that file.
        max_item_retries: Extra attempts per item before recording an error.
        retry_backoff: Base seconds to wait before retrying an item; doubles
            with each attempt so a rate-limited run backs off rather than
            burning its remaining quota.

        run_id: Identifier of this run (its result directory name). Rows
            loaded from a checkpoint keep the run_id they were made in.

    Returns list of result dicts with predictions and metadata.
    """
    results: list[dict] = []
    checkpoint_path = None
    done_ids: set[str] = set()
    provenance = row_provenance(run_id, repeat_index)

    # Resume from checkpoint if exists
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{pipeline_name}_checkpoint.jsonl"

        if checkpoint_path.exists():
            n_retry = 0
            with open(checkpoint_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    # Failed items are dropped so this run retries them,
                    # unless the failure is the model's own answer (see
                    # FINAL_ERROR_CLASSES): rows written before error_class
                    # existed carry none and are retried as before.
                    if "error" in record and record.get("error_class") not in FINAL_ERROR_CLASSES:
                        n_retry += 1
                        continue
                    results.append(record)
                    done_ids.add(str(record.get("id")))
            source_runs = sorted({str(r.get("run_id")) for r in results})
            logger.info(
                f"[{pipeline_name}] Resumed from checkpoint: "
                f"{len(done_ids)}/{len(dataset)} done, {n_retry} to retry; "
                f"reused rows come from run(s) {source_runs}"
            )

    def _save_checkpoint() -> None:
        """Write the checkpoint atomically so an interrupt cannot truncate it."""
        if checkpoint_path is None:
            return
        tmp_path = checkpoint_path.with_suffix(".jsonl.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        tmp_path.replace(checkpoint_path)

    processed = 0
    for i, item in enumerate(dataset):
        item_id = str(item.get("id", i))
        if item_id in done_ids:
            continue
        if processed > 0 and request_delay > 0:
            time.sleep(request_delay)
        question = item["question"]
        reference = item.get("answer", "")

        record: dict = {}
        meter = get_meter()
        item_mark = meter.mark() if meter is not None else 0
        item_leaked_before = meter.leaked_calls if meter is not None else 0
        prior_complete = True
        for attempt in range(max_item_retries + 1):
            mark = meter.mark() if meter is not None else 0
            leaked_before = meter.leaked_calls if meter is not None else 0
            # Calls billed by earlier, failed attempts at this item. The row's
            # own usage covers the attempt that produced it; these are kept
            # beside it so no billed call is lost to every row.
            prior_usage = (
                _prior_attempt_usage(meter, item_mark, mark, item_leaked_before, leaked_before)
                if meter is not None and attempt > 0
                else None
            )
            if prior_usage is not None:
                prior_usage["usage_complete"] = prior_complete and not prior_usage["timeout_calls"]
            start = time.perf_counter()
            try:
                result = pipeline.run(question)
                latency = time.perf_counter() - start
                usage = _attempt_usage(meter, mark, leaked_before, pipeline_name, i, item_id)
                record = {
                    "id": item_id,
                    "question": question,
                    "reference": reference,
                    "all_references": item.get("all_answers", [reference]),
                    # A refusal that stood leaves the answer None; record it
                    # as error rows do. Exclusion goes by refusal_seen.
                    "prediction": _prediction_text(result.answer),
                    "footnotes": result.footnotes,
                    "retry_count": result.retry_count,
                    "action_history": result.action_history,
                    "evaluation_scores": result.evaluation_scores,
                    "agent_type": result.agent_type,
                    "passages_used": len(result.passages_used),
                    "total_passages_retrieved": result.total_passages_retrieved,
                    "llm_calls": result.llm_calls,
                    "latency_seconds": latency,
                    "pipeline": pipeline_name,
                    # Mediation analysis fields
                    "tool_score_trace": result.tool_score_trace,
                    "question_difficulty": _extract_question_difficulty(item),
                    # Measured usage of this question alone (litellm callback).
                    # `llm_calls` above is the pipeline's own count; the two
                    # differ when litellm retried a call or a tool call went
                    # uncounted, so both are kept.
                    **usage,
                    "prior_attempt_usage": prior_usage,
                    "refusal_seen": refusal_seen(usage, prior_usage),
                    "refusal_first_seen": refusal_first_seen(usage, prior_usage),
                    **provenance,
                }
                done_ids.add(item_id)
                break
            except Exception as e:
                # Read this attempt's calls now, before the next attempt opens
                # a window that their late reports would otherwise land in.
                usage = _attempt_usage(meter, mark, leaked_before, pipeline_name, i, item_id)
                prior_complete = prior_complete and usage["usage_complete"]
                # A refused or truncated attempt is final here as on resume
                # (see FINAL_ERROR_CLASSES): another attempt would re-roll it.
                error_class = classify_error(usage, e)
                retryable = error_class not in FINAL_ERROR_CLASSES
                if attempt < max_item_retries and retryable:
                    wait = retry_backoff * (2**attempt)
                    logger.warning(
                        f"[{pipeline_name}] Item {i} ({item_id}) failed "
                        f"(attempt {attempt + 1}/{max_item_retries + 1}): {e}. "
                        f"Retrying in {wait:.0f}s"
                    )
                    time.sleep(wait)
                    continue
                logger.error(f"Error on item {i} ({item_id}): {e}")
                record = {
                    "id": item_id,
                    "question": question,
                    "reference": reference,
                    "prediction": "",
                    "error": str(e),
                    "error_class": error_class,
                    # Stamped here too, so a row's hop never depends on whether it answered.
                    "question_difficulty": _extract_question_difficulty(item),
                    "refusal_seen": refusal_seen(usage, prior_usage),
                    "refusal_first_seen": refusal_first_seen(usage, prior_usage),
                    "pipeline": pipeline_name,
                    # Error rows keep their usage: the calls were billed.
                    **usage,
                    "prior_attempt_usage": prior_usage,
                    **provenance,
                }
                break

        results.append(record)
        processed += 1
        _save_checkpoint()

        if processed % 10 == 0:
            logger.info(f"[{pipeline_name}] {len(done_ids)}/{len(dataset)} done")

    n_failed = sum(1 for r in results if "error" in r)
    n_final = sum(1 for r in results if r.get("error_class") in FINAL_ERROR_CLASSES)
    if n_failed - n_final:
        logger.warning(
            f"[{pipeline_name}] {n_failed - n_final} item(s) still failing. "
            f"Rerun the same command to retry only those."
        )
    if n_final:
        logger.warning(
            f"[{pipeline_name}] {n_final} item(s) failed as refused or truncated; "
            f"kept as final, a rerun does not retry them."
        )

    return results


def row_provenance(run_id: str | None, repeat_index: int | None = None) -> dict:
    """The stamp every result row carries: which run, code and cache made it."""
    return {
        "run_id": run_id,
        "repeat_index": repeat_index,
        "git_commit": git_state()["commit"],
        "llm_cache_disabled": settings.disable_llm_cache,
    }


# ---------------------------------------------------------------------------
# Result saving & reporting
# ---------------------------------------------------------------------------
def settings_snapshot() -> dict:
    """The settings that decide a run's numbers, as they stand right now.

    Every field here has cost this project a day at some point, because the run
    that produced a number could not be told apart from a run that would produce
    a different one:
      max_passages  - only the accumulating pipelines apply it, so baselines
                      carried 50 passages against the agentic pipeline's 30 in
                      the March 2026 runs, and nothing on disk said so.
      enabled_tools - rq1.yaml ran four tools where rq3/rq4.yaml ran all of them,
                      so their numbers for the same instance disagree.
      max_tokens    - a limit too low for a reasoning model truncates answers
                      silently; one model lost a whole dataset to it.
      model names   - previously recoverable only from the directory name.

    Callers running a sweep must take this while the variant is running.
    `save_results` is called after every variant has finished, so reading the
    global settings there records the last variant's values for all of them --
    which is how the max_passages difference stayed invisible in the first place.
    """
    return {
        "quality_threshold": settings.evaluation.quality_threshold,
        "max_retry": settings.evaluation.max_retry_count,
        "top_k": settings.retrieval.top_k,
        "hybrid_weight": settings.retrieval.hybrid_weight,
        "seed": settings.experiment.seed,
        "max_passages": settings.retrieval.max_passages,
        "max_passages_all_pipelines": settings.retrieval.max_passages_all_pipelines,
        "query_method": settings.retrieval.query_method,
        "enabled_tools": settings.agent.enabled_tools,
        "agent_max_iterations": settings.agent.max_iterations,
        "llm_call_budget": settings.experiment.llm_call_budget,
        "ircot_max_steps": settings.ircot.max_steps,
        "ircot_per_step_k": settings.ircot.per_step_k,
        "max_tokens": settings.model.max_tokens,
        "temperature": settings.model.temperature,
        "models": {
            "preprocess": settings.model.preprocess_model,
            "evaluate": settings.model.evaluate_model,
            "generate": settings.model.generate_model,
            "agent": settings.model.agent_model,
        },
    }


def _prediction_text(prediction: str | None) -> str:
    """A prediction as text for scoring.

    Rows checkpointed before the fix carry None where a refusal stood.
    """
    return "" if prediction is None else prediction


def save_results(
    results: list[dict],
    experiment_name: str,
    extra_metadata: dict | None = None,
    *,
    run_dir: Path | None = None,
    compute_llm_judge: bool = False,
    settings_used: dict | None = None,
) -> Path:
    """Save experiment results to data/results/<run_dir>/."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if run_dir is None:
        run_dir = settings.results_dir / f"{timestamp}_{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    out_path = run_dir / f"{experiment_name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    # Save summary
    valid = [r for r in results if "error" not in r]
    predictions = [_prediction_text(r.get("prediction")) for r in valid]
    references = [r.get("reference", "") for r in valid]
    questions = [r.get("question", "") for r in valid]

    summary = {
        "experiment": experiment_name,
        "timestamp": timestamp,
        "total_items": len(results),
        "errors": sum(1 for r in results if "error" in r),
        "settings": settings_used or settings_snapshot(),
    }

    if predictions and references:
        metrics = evaluate_batch(
            predictions,
            references,
            questions=questions,
            compute_bert_score=False,
            compute_llm_judge=compute_llm_judge,
        )
        summary["metrics"] = metrics

    if extra_metadata:
        summary["extra"] = extra_metadata

    summary_path = run_dir / f"{experiment_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {out_path}")
    return out_path


def print_comparison_table(
    all_results: dict[str, list[dict]],
    title: str = "Experiment Results",
    compute_llm_judge: bool = False,
) -> None:
    """Print a rich comparison table of pipeline results."""
    table = Table(title=title)
    table.add_column("Pipeline", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("EM", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Judge", justify="right")
    table.add_column("Avg Retries", justify="right")
    table.add_column("Avg Latency (s)", justify="right")
    table.add_column("Errors", justify="right")

    for name, results in all_results.items():
        valid = [r for r in results if "error" not in r]
        errors = len(results) - len(valid)

        if not valid:
            table.add_row(name, str(len(results)), "-", "-", "-", "-", "-", str(errors))
            continue

        preds = [_prediction_text(r.get("prediction")) for r in valid]
        refs = [r["reference"] for r in valid]
        questions = [r.get("question", "") for r in valid]
        metrics = evaluate_batch(
            preds,
            refs,
            questions=questions,
            compute_bert_score=False,
            compute_llm_judge=compute_llm_judge,
        )

        avg_retries = np.mean([r.get("retry_count", 0) for r in valid])
        avg_latency = np.mean([r.get("latency_seconds", 0) for r in valid])

        judge_str = f"{metrics['llm_judge']:.3f}" if "llm_judge" in metrics else "-"

        table.add_row(
            name,
            str(len(valid)),
            f"{metrics['exact_match']:.3f}",
            f"{metrics['f1']:.3f}",
            judge_str,
            f"{avg_retries:.1f}",
            f"{avg_latency:.1f}",
            str(errors),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Question difficulty extraction (hop bins)
# ---------------------------------------------------------------------------
