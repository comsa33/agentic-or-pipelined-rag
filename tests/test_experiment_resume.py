"""Checkpoint resume behavior for long, rate-limited experiment runs.

A run against a rate-limited provider will be interrupted. What matters is
that the rerun costs only the outstanding questions: completed items must
never be re-answered, and failed items must never be silently frozen into
the results as permanent gaps.
"""

from __future__ import annotations

import json

from agentic_rag.pipeline.base import PipelineResult
from experiments.common import run_pipeline_on_dataset


class FlakyPipeline:
    """Pipeline stub that fails for a chosen set of questions."""

    def __init__(self, fail_questions: set[str], fail_times: int = 999) -> None:
        self.fail_questions = fail_questions
        self.fail_times = fail_times
        self.calls: list[str] = []
        self._failures: dict[str, int] = {}

    def run(self, question: str) -> PipelineResult:
        self.calls.append(question)
        if question in self.fail_questions:
            seen = self._failures.get(question, 0)
            if seen < self.fail_times:
                self._failures[question] = seen + 1
                raise RuntimeError("rate limit exceeded")
        return PipelineResult(question=question, answer=f"answer to {question}")


def _dataset(n: int) -> list[dict]:
    return [{"id": f"q{i}", "question": f"question {i}", "answer": f"answer {i}"} for i in range(n)]


def test_completed_items_are_not_rerun(tmp_path):
    """A rerun must not spend a single call on already-answered questions."""
    dataset = _dataset(4)

    first = FlakyPipeline(fail_questions=set())
    run_pipeline_on_dataset(first, dataset, "p", checkpoint_dir=tmp_path)
    assert len(first.calls) == 4

    second = FlakyPipeline(fail_questions=set())
    results = run_pipeline_on_dataset(second, dataset, "p", checkpoint_dir=tmp_path)

    assert second.calls == [], "resumed run re-answered completed questions"
    assert len(results) == 4
    assert {r["id"] for r in results} == {"q0", "q1", "q2", "q3"}


def test_failed_items_are_retried_on_resume(tmp_path):
    """Items that failed must come back as work, not as a permanent gap."""
    dataset = _dataset(4)

    first = FlakyPipeline(fail_questions={"question 1", "question 2"})
    results = run_pipeline_on_dataset(
        first, dataset, "p", checkpoint_dir=tmp_path, max_item_retries=0
    )
    assert sum(1 for r in results if "error" in r) == 2

    # The provider recovers; the rerun should touch only the two failures.
    second = FlakyPipeline(fail_questions=set())
    results = run_pipeline_on_dataset(second, dataset, "p", checkpoint_dir=tmp_path)

    assert sorted(second.calls) == ["question 1", "question 2"]
    assert all("error" not in r for r in results)
    assert len(results) == 4
    assert {r["id"] for r in results} == {"q0", "q1", "q2", "q3"}


def test_transient_failure_recovers_within_one_run(tmp_path):
    """A single transient failure should be retried in place, not deferred."""
    dataset = _dataset(2)
    pipeline = FlakyPipeline(fail_questions={"question 0"}, fail_times=1)

    results = run_pipeline_on_dataset(
        pipeline, dataset, "p", checkpoint_dir=tmp_path, max_item_retries=2, retry_backoff=0.0
    )

    assert all("error" not in r for r in results)
    assert pipeline.calls.count("question 0") == 2


def test_checkpoint_is_written_after_every_item(tmp_path):
    """An interrupt must cost at most the item in flight."""
    dataset = _dataset(3)
    pipeline = FlakyPipeline(fail_questions={"question 2"})

    run_pipeline_on_dataset(pipeline, dataset, "p", checkpoint_dir=tmp_path, max_item_retries=0)

    checkpoint = tmp_path / "p_checkpoint.jsonl"
    records = [json.loads(x) for x in checkpoint.read_text(encoding="utf-8").splitlines() if x]
    assert len(records) == 3
    assert not list(tmp_path.glob("*.tmp")), "temporary checkpoint file was left behind"


def test_resume_reports_only_successful_items_as_done(tmp_path):
    """Partial progress must be counted by successes, not by line count."""
    dataset = _dataset(5)

    first = FlakyPipeline(fail_questions={"question 3"})
    run_pipeline_on_dataset(first, dataset, "p", checkpoint_dir=tmp_path, max_item_retries=0)

    second = FlakyPipeline(fail_questions={"question 3"})
    run_pipeline_on_dataset(second, dataset, "p", checkpoint_dir=tmp_path, max_item_retries=0)

    # Only the still-failing question is retried, and it stays marked failed.
    assert second.calls == ["question 3"]
    checkpoint = tmp_path / "p_checkpoint.jsonl"
    records = [json.loads(x) for x in checkpoint.read_text(encoding="utf-8").splitlines() if x]
    assert sum(1 for r in records if "error" in r) == 1
    assert len(records) == 5


# ---------------------------------------------------------------------------
# Provenance of resumed rows
# ---------------------------------------------------------------------------
def test_rows_are_stamped_with_their_own_run(tmp_path):
    """A resumed row keeps the run_id of the process that produced it."""
    dataset = _dataset(3)

    first = FlakyPipeline(fail_questions={"question 2"}, fail_times=999)
    run_pipeline_on_dataset(
        first, dataset, "p", checkpoint_dir=tmp_path, run_id="run_A", max_item_retries=0
    )

    second = FlakyPipeline(fail_questions=set())
    rows = run_pipeline_on_dataset(second, dataset, "p", checkpoint_dir=tmp_path, run_id="run_B")

    by_id = {r["id"]: r for r in rows}
    assert by_id["q0"]["run_id"] == "run_A", "reused row must keep its origin"
    assert by_id["q1"]["run_id"] == "run_A"
    assert by_id["q2"]["run_id"] == "run_B", "retried row is made by this run"
    assert all(len(r["git_commit"]) == 40 for r in rows)
    assert all("llm_cache_disabled" in r for r in rows)


def test_manifest_records_resumed_rows(tmp_path):
    from experiments.manifest import resumed_rows

    rows = {
        "Naive RAG": [{"id": "q0", "run_id": "run_A"}, {"id": "q1", "run_id": "run_B"}],
        "Loop": [{"id": "q0", "run_id": "run_B"}],
    }
    resumed = resumed_rows("run_B", rows)
    assert resumed["rows_reused"] == 1
    assert resumed["source_run_ids"] == ["run_A"]
    assert resumed["by_variant"] == {"Naive RAG": {"rows_reused": 1, "source_run_ids": ["run_A"]}}


# ---------------------------------------------------------------------------
# Refused or truncated items are kept on resume, not re-rolled
# ---------------------------------------------------------------------------
def _write_checkpoint(tmp_path, rows: list[dict]) -> None:
    (tmp_path / "p_checkpoint.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )


def _error_row(i: int, **extra) -> dict:
    return {"id": f"q{i}", "question": f"question {i}", "prediction": "", "error": "x", **extra}


def test_refused_and_truncated_errors_are_final_on_resume(tmp_path):
    """Retrying them would make "wrong" depend on how often the run was resumed."""
    _write_checkpoint(
        tmp_path,
        [
            _error_row(0, error_class="refused"),
            _error_row(1, error_class="truncated"),
            _error_row(2, error_class="other"),
            _error_row(3),  # written before error_class existed
        ],
    )
    pipeline = FlakyPipeline(fail_questions=set())
    results = run_pipeline_on_dataset(pipeline, _dataset(4), "p", checkpoint_dir=tmp_path)

    assert sorted(pipeline.calls) == ["question 2", "question 3"]
    by_id = {r["id"]: r for r in results}
    assert by_id["q0"]["error_class"] == "refused"
    assert by_id["q1"]["error_class"] == "truncated"
    assert "error" not in by_id["q2"] and "error" not in by_id["q3"]


def test_provider_error_is_classified_other(tmp_path):
    """An Azure content-filter 400 or an exhausted retry stays retryable."""
    pipeline = FlakyPipeline(fail_questions={"question 0"})
    results = run_pipeline_on_dataset(
        pipeline, _dataset(1), "p", checkpoint_dir=tmp_path, max_item_retries=0
    )
    assert results[0]["error_class"] == "other"


def test_error_class_order():
    from experiments.common import classify_error

    assert classify_error({"refused_calls": 1, "truncated_calls": 1}) == "refused"
    assert classify_error({"refused_calls": 0, "truncated_calls": 2}) == "truncated"
    assert classify_error({"failed_calls": 3, "failed_by_status": {"400": 3}}) == "other"
    assert classify_error({}) == "other"


def test_transient_error_is_other_even_after_a_truncated_call():
    """A rate limit or server error says nothing about the question."""
    import litellm

    from experiments.common import classify_error

    usage = {"refused_calls": 1, "truncated_calls": 1}
    for exc in (
        litellm.RateLimitError("slow down", model="m", llm_provider="anthropic"),
        litellm.InternalServerError("overloaded", model="m", llm_provider="anthropic"),
        litellm.Timeout("timed out", model="m", llm_provider="anthropic"),
        litellm.APIConnectionError("reset", model="m", llm_provider="anthropic"),
    ):
        assert classify_error(usage, exc) == "other", type(exc).__name__
    wrapped = RuntimeError("adapter failed")
    wrapped.__cause__ = litellm.RateLimitError("slow down", model="m", llm_provider="anthropic")
    assert classify_error(usage, wrapped) == "other"
    assert classify_error(usage, ValueError("could not parse")) == "refused"
    rejected = litellm.BadRequestError("tool_choice", model="m", llm_provider="anthropic")
    assert classify_error({"truncated_calls": 1}, rejected) == "truncated"


# ---------------------------------------------------------------------------
# Platform content filter: retried like other, classified on the final attempt
# ---------------------------------------------------------------------------
def _content_filter():
    import litellm

    return litellm.ContentPolicyViolationError(
        "The response was filtered due to the prompt triggering Azure OpenAI's content management policy",
        model="gpt-5-mini",
        llm_provider="azure",
    )


class FilteredPipeline:
    """Raises the Azure content filter on the first `blocked` attempts."""

    def __init__(self, blocked: int) -> None:
        self.blocked = blocked
        self.runs = 0

    def run(self, question: str) -> PipelineResult:
        self.runs += 1
        if self.runs <= self.blocked:
            raise _content_filter()
        return PipelineResult(question=question, answer="passed")


def test_filter_blocked_to_the_end_is_platform_filtered(tmp_path):
    pipeline = FilteredPipeline(blocked=99)
    results = run_pipeline_on_dataset(
        pipeline, _dataset(1), "p", checkpoint_dir=tmp_path, max_item_retries=2, retry_backoff=0
    )
    assert pipeline.runs == 3, "a filter block is retried like any other error"
    assert results[0]["error_class"] == "platform_filtered"


def test_filter_block_that_later_passes_leaves_an_answered_row(tmp_path):
    pipeline = FilteredPipeline(blocked=1)
    results = run_pipeline_on_dataset(
        pipeline, _dataset(1), "p", checkpoint_dir=tmp_path, max_item_retries=2, retry_backoff=0
    )
    assert "error" not in results[0]
    assert results[0]["prediction"] == "passed"


def test_platform_filtered_rows_are_retried_on_resume(tmp_path):
    _write_checkpoint(tmp_path, [_error_row(0, error_class="platform_filtered")])
    pipeline = FlakyPipeline(fail_questions=set())
    results = run_pipeline_on_dataset(pipeline, _dataset(1), "p", checkpoint_dir=tmp_path)
    assert pipeline.calls == ["question 0"]
    assert "error" not in results[0]


def test_filter_classification_order():
    import litellm

    from experiments.common import classify_error

    usage = {"refused_calls": 1, "truncated_calls": 1}
    assert classify_error(usage, _content_filter()) == "platform_filtered"
    wrapped = RuntimeError("adapter failed")
    wrapped.__cause__ = _content_filter()
    assert classify_error({}, wrapped) == "platform_filtered"
    coded = litellm.BadRequestError("blocked", model="m", llm_provider="azure")
    coded.code = "content_filter"
    assert classify_error({}, coded) == "platform_filtered"
    # A transient error still comes first.
    transient = RuntimeError("x")
    transient.__cause__ = litellm.RateLimitError("slow", model="m", llm_provider="azure")
    assert classify_error(usage, transient) == "other"


def test_error_rows_carry_the_same_difficulty_stamp_as_answered_rows(tmp_path):
    """A row's hop must not depend on whether the question was answered."""
    dataset = [
        {"id": "a", "question": "who?", "answer": "x", "metadata": {"n_hops": 3}},
        {"id": "b", "question": "who?", "answer": "x", "metadata": {"n_hops": 3}},
    ]
    pipeline = FlakyPipeline(fail_questions={"who?"}, fail_times=1)
    rows = run_pipeline_on_dataset(
        pipeline, dataset, "p", checkpoint_dir=tmp_path, max_item_retries=0
    )
    error, answered = rows
    assert "error" in error and "error" not in answered
    assert error["question_difficulty"] == answered["question_difficulty"]
    assert error["question_difficulty"]["hop_count"] == 3
