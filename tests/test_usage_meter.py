"""Measured per-question usage must land on the result row.

The March 2026 runs recorded no tokens and no cost per question, so the cost
column of the paper had to be re-measured afterwards. The meter attributes
litellm's own usage report to the question that caused it, including calls
whose success callback arrives after the pipeline has already returned.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

import experiments.common as common
from agentic_rag.evaluation.cost_tracker import LiteLLMMeter
from agentic_rag.pipeline.base import PipelineResult
from experiments.common import run_pipeline_on_dataset


def _response(model="gpt-5-mini-2025-08-07", prompt=100, cached=40, completion=20, reasoning=5):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        ),
    )


def _call(meter: LiteLLMMeter, call_id: str, cost: float | None = 0.001, **kw) -> None:
    kwargs = {"litellm_call_id": call_id, "model": "azure/gpt-5-mini", "response_cost": cost}
    meter.log_pre_api_call("azure/gpt-5-mini", [], kwargs)
    t0 = datetime.now()
    meter.log_success_event(kwargs, _response(**kw), t0, t0 + timedelta(milliseconds=250))


def test_window_aggregates_only_calls_after_mark():
    meter = LiteLLMMeter()
    _call(meter, "before")
    mark = meter.mark()
    _call(meter, "a", prompt=10, cached=0, completion=1, reasoning=0)
    _call(meter, "b", prompt=20, cached=5, completion=2, reasoning=1, cost=None)

    usage = meter.usage_since(mark)
    assert usage["metered_calls"] == 2
    assert usage["prompt_tokens"] == 30
    assert usage["cached_tokens"] == 5
    assert usage["completion_tokens"] == 3
    assert usage["reasoning_tokens"] == 1
    assert usage["total_tokens"] == 33
    assert usage["cost_usd"] == 0.001
    assert usage["calls_without_cost"] == 1
    assert usage["response_models"] == ["gpt-5-mini-2025-08-07"]


def test_drain_waits_for_late_success_callback():
    """litellm reports success from a thread pool after the call returned."""
    meter = LiteLLMMeter()
    kwargs = {"litellm_call_id": "late", "model": "azure/gpt-5-mini", "response_cost": 0.5}
    meter.log_pre_api_call("azure/gpt-5-mini", [], kwargs)

    def _late_report():
        time.sleep(0.2)
        t0 = datetime.now()
        meter.log_success_event(kwargs, _response(), t0, t0)

    threading.Thread(target=_late_report).start()
    assert meter.usage_since(0)["metered_calls"] == 0
    assert meter.drain(timeout=5.0)
    assert meter.usage_since(0)["metered_calls"] == 1
    assert meter.usage_since(0)["cost_usd"] == 0.5


def test_drain_gives_up_on_a_call_that_never_reports():
    meter = LiteLLMMeter()
    meter.log_pre_api_call("m", [], {"litellm_call_id": "lost"})
    assert meter.drain(timeout=0.05) is False


def test_failure_clears_pending():
    meter = LiteLLMMeter()
    kwargs = {"litellm_call_id": "boom"}
    meter.log_pre_api_call("m", [], kwargs)
    meter.log_failure_event(kwargs, None, None, None)
    assert meter.drain(timeout=0.05) is True
    assert meter.failures == 1


class _MeteredPipeline:
    """Pipeline stub whose every run triggers one metered call."""

    def __init__(self, meter: LiteLLMMeter) -> None:
        self.meter = meter
        self.n = 0

    def run(self, question: str) -> PipelineResult:
        self.n += 1
        _call(self.meter, f"c{self.n}", prompt=100 * self.n, cached=0, completion=10, reasoning=0)
        return PipelineResult(question=question, answer="x", llm_calls=1)


def test_result_rows_carry_their_own_usage(monkeypatch):
    meter = LiteLLMMeter()
    monkeypatch.setattr(common, "_meter", meter)
    dataset = [{"id": "q1", "question": "one", "answer": "a"}, {"id": "q2", "question": "two"}]

    rows = run_pipeline_on_dataset(_MeteredPipeline(meter), dataset, "p")

    assert [r["prompt_tokens"] for r in rows] == [100, 200]
    assert all(r["metered_calls"] == 1 for r in rows)
    assert all(r["cost_usd"] == 0.001 for r in rows)
    assert rows[0]["response_models"] == ["gpt-5-mini-2025-08-07"]


def test_rows_without_a_meter_still_have_the_fields(monkeypatch):
    """Analysis code must be able to rely on the columns being present."""
    monkeypatch.setattr(common, "_meter", None)
    meter = LiteLLMMeter()
    rows = run_pipeline_on_dataset(_MeteredPipeline(meter), [{"id": "q", "question": "?"}], "p")
    assert rows[0]["metered_calls"] == 0
    assert rows[0]["cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# A call that never reports in time must not be billed to the next question
# ---------------------------------------------------------------------------
def test_leaked_call_is_written_off_not_billed_to_next_window():
    meter = LiteLLMMeter()
    kwargs = {"litellm_call_id": "slow", "model": "azure/gpt-5-mini", "response_cost": 9.0}
    meter.log_pre_api_call("azure/gpt-5-mini", [], kwargs)
    assert meter.drain(timeout=0.05) is False
    assert meter.leaked_calls == 1

    # Next question starts; the slow report lands now.
    mark = meter.mark()
    t0 = datetime.now()
    meter.log_success_event(kwargs, _response(), t0, t0)
    _call(meter, "next", cost=0.001)

    usage = meter.usage_since(mark)
    assert usage["metered_calls"] == 1, "the late report must not appear in this window"
    assert usage["cost_usd"] == 0.001
    assert meter.drain(timeout=0.05) is True


class _SlowCallbackPipeline:
    """Every run starts one call whose success report never arrives in time."""

    def __init__(self, meter: LiteLLMMeter) -> None:
        self.meter = meter
        self.n = 0

    def run(self, question: str) -> PipelineResult:
        self.n += 1
        self.meter.log_pre_api_call("m", [], {"litellm_call_id": f"slow{self.n}"})
        _call(self.meter, f"fast{self.n}")
        return PipelineResult(question=question, answer="x", llm_calls=2)


def test_row_marks_incomplete_usage(monkeypatch):
    meter = LiteLLMMeter()
    monkeypatch.setattr(common, "_meter", meter)
    monkeypatch.setattr(common, "DRAIN_TIMEOUT_SECONDS", 0.05)
    dataset = [{"id": "q1", "question": "one"}, {"id": "q2", "question": "two"}]

    rows = run_pipeline_on_dataset(_SlowCallbackPipeline(meter), dataset, "p")

    assert [r["usage_complete"] for r in rows] == [False, False]
    assert [r["leaked_calls"] for r in rows] == [1, 1]
    assert [r["metered_calls"] for r in rows] == [1, 1], "only the reported call is counted"
    assert meter.leaked_calls == 2


def _finished(
    meter: LiteLLMMeter, call_id: str, finish: str, raw_usage: dict | None = None
) -> None:
    kwargs = {
        "litellm_call_id": call_id,
        "model": "anthropic/claude-sonnet-5-5",
        "response_cost": 0.01,
    }
    if raw_usage is not None:
        kwargs["original_response"] = json.dumps({"usage": raw_usage})
    resp = _response(model="claude-sonnet-5-5", reasoning=0)
    resp.choices = [SimpleNamespace(finish_reason=finish)]
    meter.log_pre_api_call("anthropic/claude-sonnet-5-5", [], kwargs)
    t0 = datetime.now()
    meter.log_success_event(kwargs, resp, t0, t0)


def test_truncated_calls_are_counted_per_row():
    """A thinking model can spend max_tokens on reasoning and return no answer."""
    meter = LiteLLMMeter()
    _finished(meter, "ok", "stop")
    _finished(meter, "cut", "length")
    usage = meter.usage_since(0)
    assert usage["finish_reasons"] == {"stop": 1, "length": 1}
    assert usage["truncated_calls"] == 1


def test_anthropic_thinking_tokens_read_from_raw_usage():
    """litellm reports reasoning_tokens=0 for Anthropic; the API reports them here."""
    meter = LiteLLMMeter()
    _finished(
        meter,
        "t",
        "stop",
        raw_usage={"output_tokens": 5981, "output_tokens_details": {"thinking_tokens": 5663}},
    )
    assert meter.usage_since(0)["reasoning_tokens"] == 5663


def test_empty_usage_has_no_truncation():
    from agentic_rag.evaluation.cost_tracker import empty_usage

    assert empty_usage()["truncated_calls"] == 0
    assert empty_usage()["finish_reasons"] == {}


# ---------------------------------------------------------------------------
# Failed, timed-out and refused calls
# ---------------------------------------------------------------------------
def _failed(meter: LiteLLMMeter, call_id: str, exc: Exception, retries: int = 0) -> None:
    """A call litellm retried `retries` times under one id, every attempt failing.

    litellm registers the id once per attempt but reports only the first
    failure of the sequence.
    """
    kwargs = {"litellm_call_id": call_id, "model": "azure/gpt-5-mini", "exception": exc}
    meter.log_pre_api_call("azure/gpt-5-mini", [], kwargs)
    t0 = datetime.now()
    meter.log_failure_event(kwargs, None, t0, t0)
    for _ in range(retries):
        meter.log_pre_api_call("azure/gpt-5-mini", [], kwargs)


def _bad_request() -> Exception:
    import litellm

    return litellm.BadRequestError("rejected", model="gpt-5-mini", llm_provider="azure")


def test_failed_call_is_counted_by_status_and_kept_out_of_usage():
    meter = LiteLLMMeter()
    _failed(meter, "bad", _bad_request())
    _call(meter, "ok", cost=0.002)
    usage = meter.usage_since(0)
    assert usage["failed_calls"] == 1
    assert usage["failed_by_status"] == {"400": 1}
    assert usage["metered_calls"] == 1
    assert usage["cost_usd"] == 0.002
    assert usage["calls_without_cost"] == 0
    assert meter.failures == 1


def test_retry_sequence_failing_to_the_end_is_failed_not_leaked():
    """1 attempt + 12 retries under one id, all rejected: the row stays complete."""
    meter = LiteLLMMeter()
    _failed(meter, "seq", _bad_request(), retries=12)
    assert meter.drain(timeout=0.05) is True
    assert meter.leaked_calls == 0
    usage = meter.usage_since(0)
    assert usage["failed_calls"] == 1
    assert usage["failed_by_status"] == {"400": 1}


def test_settled_failure_does_not_hide_a_lost_call():
    meter = LiteLLMMeter()
    _failed(meter, "seq", _bad_request(), retries=2)
    meter.log_pre_api_call("m", [], {"litellm_call_id": "lost"})
    assert meter.drain(timeout=0.05) is False
    assert meter.leaked_calls == 1


def test_retry_that_succeeds_after_a_failure_is_metered():
    meter = LiteLLMMeter()
    _failed(meter, "flaky", _bad_request(), retries=1)
    _call(meter, "flaky", cost=0.003)
    assert meter.drain(timeout=0.05) is True
    usage = meter.usage_since(0)
    assert usage["failed_calls"] == 1
    assert usage["metered_calls"] == 1
    assert usage["cost_usd"] == 0.003


def test_timeout_is_counted_apart_from_failures():
    import litellm

    meter = LiteLLMMeter()
    _failed(meter, "slow", litellm.Timeout("timed out", model="gpt-5-mini", llm_provider="azure"))
    usage = meter.usage_since(0)
    assert usage["timeout_calls"] == 1
    assert usage["failed_calls"] == 0
    assert usage["failed_by_status"] == {}


class _TimeoutPipeline:
    def __init__(self, meter: LiteLLMMeter) -> None:
        self.meter = meter

    def run(self, question: str) -> PipelineResult:
        import litellm

        _failed(self.meter, "t", litellm.Timeout("timed out", model="m", llm_provider="azure"))
        _call(self.meter, "after")
        return PipelineResult(question=question, answer="x", llm_calls=2)


def test_row_with_a_timeout_is_marked_incomplete(monkeypatch):
    """A timed-out call may have been billed without reporting usage."""
    meter = LiteLLMMeter()
    monkeypatch.setattr(common, "_meter", meter)
    rows = run_pipeline_on_dataset(_TimeoutPipeline(meter), [{"id": "q", "question": "?"}], "p")
    assert rows[0]["timeout_calls"] == 1
    assert rows[0]["usage_complete"] is False
    assert rows[0]["leaked_calls"] == 0


def test_anthropic_refusal_is_counted():
    """Anthropic returns a refusal as HTTP 200; litellm maps it to "stop"."""
    meter = LiteLLMMeter()
    kwargs = {
        "litellm_call_id": "r",
        "model": "anthropic/claude-sonnet-5-5",
        "response_cost": 0.0,
        "original_response": json.dumps({"stop_reason": "refusal", "usage": {}}),
    }
    resp = _response(model="claude-sonnet-5-5", reasoning=0)
    resp.choices = [SimpleNamespace(finish_reason="stop")]
    meter.log_pre_api_call("anthropic/claude-sonnet-5-5", [], kwargs)
    t0 = datetime.now()
    meter.log_success_event(kwargs, resp, t0, t0)
    _finished(meter, "ok", "stop")
    usage = meter.usage_since(0)
    assert usage["refused_calls"] == 1
    assert usage["finish_reasons"] == {"stop": 2}


# ---------------------------------------------------------------------------
# Usage of failed item attempts is not lost
# ---------------------------------------------------------------------------
class _FlakyPipeline:
    """Makes one billed call per run; the first `fail_runs` runs then raise."""

    def __init__(self, meter: LiteLLMMeter, fail_runs: int) -> None:
        self.meter = meter
        self.fail_runs = fail_runs
        self.n = 0

    def run(self, question: str) -> PipelineResult:
        self.n += 1
        _call(self.meter, f"c{self.n}", cost=0.01 * self.n)
        if self.n <= self.fail_runs:
            raise RuntimeError("parse failure")
        return PipelineResult(question=question, answer="x", llm_calls=1)


def test_retried_item_keeps_usage_of_earlier_attempts(monkeypatch):
    meter = LiteLLMMeter()
    monkeypatch.setattr(common, "_meter", meter)
    rows = run_pipeline_on_dataset(
        _FlakyPipeline(meter, fail_runs=2), [{"id": "q", "question": "?"}], "p", retry_backoff=0
    )
    row = rows[0]
    assert "error" not in row
    assert row["metered_calls"] == 1
    assert row["cost_usd"] == pytest.approx(0.03)
    prior = row["prior_attempt_usage"]
    assert prior["metered_calls"] == 2
    assert prior["cost_usd"] == pytest.approx(0.03)
    assert prior["usage_complete"] is True


def test_error_row_keeps_its_usage(monkeypatch):
    meter = LiteLLMMeter()
    monkeypatch.setattr(common, "_meter", meter)
    rows = run_pipeline_on_dataset(
        _FlakyPipeline(meter, fail_runs=3),
        [{"id": "q", "question": "?"}],
        "p",
        max_item_retries=2,
        retry_backoff=0,
    )
    row = rows[0]
    assert row["error"] == "parse failure"
    assert row["metered_calls"] == 1
    assert row["cost_usd"] == pytest.approx(0.03)
    assert row["prior_attempt_usage"]["metered_calls"] == 2


def test_first_attempt_success_has_no_prior_usage(monkeypatch):
    meter = LiteLLMMeter()
    monkeypatch.setattr(common, "_meter", meter)
    rows = run_pipeline_on_dataset(_MeteredPipeline(meter), [{"id": "q", "question": "?"}], "p")
    assert rows[0]["prior_attempt_usage"] is None


# ---------------------------------------------------------------------------
# Structured-output requests: DSPy adapter fallbacks
# ---------------------------------------------------------------------------
def _requested(meter: LiteLLMMeter, call_id: str, body: dict, response_format=None) -> None:
    kwargs = {
        "litellm_call_id": call_id,
        "model": "azure/gpt-5-mini",
        "response_cost": 0.001,
        "response_format": response_format,
        "additional_args": {"complete_input_dict": body},
    }
    meter.log_pre_api_call("azure/gpt-5-mini", [], kwargs)
    t0 = datetime.now()
    meter.log_success_event(kwargs, _response(), t0, t0)


def test_structured_requests_are_counted_by_mode():
    from pydantic import BaseModel

    class Outputs(BaseModel):
        answer: str

    meter = LiteLLMMeter()
    _requested(meter, "plain", {"messages": []})
    _requested(meter, "schema", {"response_format": {"type": "json_schema"}}, Outputs)
    _requested(
        meter, "object", {"response_format": {"type": "json_object"}}, {"type": "json_object"}
    )
    usage = meter.usage_since(0)
    assert usage["structured_calls"] == {"json_schema": 1, "json_object": 1}
    assert meter.calls[0].request["structured"] == ""


def test_forced_tool_rejected_by_anthropic_counts_as_failed_fallback():
    """Claude 5.x gets the schema as a forced tool, rejects it, and DSPy retries as json_object."""
    meter = LiteLLMMeter()
    body = {"tools": [{"name": "json_tool_call"}], "tool_choice": {"type": "tool"}}
    kwargs = {
        "litellm_call_id": "forced",
        "model": "anthropic/claude-sonnet-5-5",
        "response_format": {"type": "json_schema"},
        "additional_args": {"complete_input_dict": body},
        "exception": _bad_request(),
    }
    meter.log_pre_api_call("anthropic/claude-sonnet-5-5", [], kwargs)
    t0 = datetime.now()
    meter.log_failure_event(kwargs, None, t0, t0)
    _requested(meter, "object", {}, {"type": "json_object"})
    usage = meter.usage_since(0)
    assert usage["structured_calls"] == {"json_schema": 1, "json_object": 1}
    assert usage["failed_by_status"] == {"400": 1}
    assert meter.calls[0].request["sent"] == ["tool_choice", "tools"]


def test_request_summary_reads_no_headers():
    meter = LiteLLMMeter()
    _requested(meter, "c", {"max_tokens": 10, "temperature": 0.0})
    assert meter.calls[0].request == {"max_tokens": 10, "sent": ["temperature"], "structured": ""}
    assert "headers" not in json.dumps(meter.calls[0].request)


# ---------------------------------------------------------------------------
# Attempts: one unit for every provider
# ---------------------------------------------------------------------------
def test_attempts_count_litellm_retries_under_one_id():
    """Azure path: 1 + 12 attempts share an id; one failure is reported."""
    meter = LiteLLMMeter()
    _failed(meter, "seq", _bad_request(), retries=12)
    assert meter.drain(timeout=0.05) is True
    usage = meter.usage_since(0)
    assert usage["attempts"] == 13
    assert usage["failed_calls"] == 1


def test_attempts_count_retries_made_outside_litellm():
    """Anthropic path: each attempt is its own call id and failure report."""
    meter = LiteLLMMeter()
    for i in range(3):
        _failed(meter, f"a{i}", _bad_request())
    usage = meter.usage_since(0)
    assert usage["attempts"] == 3
    assert usage["failed_calls"] == 3


def test_attempts_of_a_retry_that_succeeds_are_counted_once():
    meter = LiteLLMMeter()
    _failed(meter, "flaky", _bad_request(), retries=1)  # pre, failure, pre
    kwargs = {"litellm_call_id": "flaky", "model": "azure/gpt-5-mini", "response_cost": 0.001}
    t0 = datetime.now()
    meter.log_success_event(kwargs, _response(), t0, t0)  # the retry answers
    _call(meter, "plain")
    assert meter.drain(timeout=0.05) is True
    usage = meter.usage_since(0)
    assert usage["attempts"] == 3
    assert usage["metered_calls"] == 2


# ---------------------------------------------------------------------------
# Refused or truncated attempts are not retried within the run either
# ---------------------------------------------------------------------------
class _CutOffPipeline:
    """Each run makes one call that stops at max_tokens, then fails to parse."""

    def __init__(self, meter: LiteLLMMeter, finish: str = "length") -> None:
        self.meter = meter
        self.finish = finish
        self.runs = 0

    def run(self, question: str) -> PipelineResult:
        self.runs += 1
        _finished(self.meter, f"cut{self.runs}", self.finish)
        raise RuntimeError("could not parse")


def test_truncated_attempt_is_final_without_item_retries(monkeypatch):
    meter = LiteLLMMeter()
    monkeypatch.setattr(common, "_meter", meter)
    pipeline = _CutOffPipeline(meter)
    rows = run_pipeline_on_dataset(
        pipeline, [{"id": "q", "question": "?"}], "p", max_item_retries=2, retry_backoff=0
    )
    assert pipeline.runs == 1, "a truncated attempt must not be re-rolled"
    assert rows[0]["error_class"] == "truncated"


def test_other_failures_still_use_item_retries(monkeypatch):
    meter = LiteLLMMeter()
    monkeypatch.setattr(common, "_meter", meter)
    pipeline = _CutOffPipeline(meter, finish="stop")
    rows = run_pipeline_on_dataset(
        pipeline, [{"id": "q", "question": "?"}], "p", max_item_retries=2, retry_backoff=0
    )
    assert pipeline.runs == 3
    assert rows[0]["error_class"] == "other"


class _CutOffThenRateLimited(_CutOffPipeline):
    def run(self, question: str) -> PipelineResult:
        import litellm

        self.runs += 1
        _finished(self.meter, f"cut{self.runs}", "length")
        raise litellm.RateLimitError("slow down", model="m", llm_provider="anthropic")


def test_transient_failure_after_truncation_is_retried(monkeypatch):
    meter = LiteLLMMeter()
    monkeypatch.setattr(common, "_meter", meter)
    pipeline = _CutOffThenRateLimited(meter)
    rows = run_pipeline_on_dataset(
        pipeline, [{"id": "q", "question": "?"}], "p", max_item_retries=2, retry_backoff=0
    )
    assert pipeline.runs == 3
    assert rows[0]["error_class"] == "other"


def test_refusal_seen_marks_any_row_with_a_refused_call():
    from experiments.common import refusal_seen

    assert refusal_seen({"refused_calls": 0}, {"refused_calls": 1})
    assert refusal_seen({"refused_calls": 1}, None)
    assert not refusal_seen({"refused_calls": 0}, None)


class _RefusedButAnswered:
    def __init__(self, meter: LiteLLMMeter) -> None:
        self.meter = meter

    def run(self, question: str) -> PipelineResult:
        kwargs = {
            "litellm_call_id": "r",
            "model": "anthropic/claude-sonnet-5-5",
            "response_cost": 0.0,
            "original_response": json.dumps({"stop_reason": "refusal", "usage": {}}),
        }
        resp = _response(model="claude-sonnet-5-5", reasoning=0)
        resp.choices = [SimpleNamespace(finish_reason="stop")]
        self.meter.log_pre_api_call("anthropic/claude-sonnet-5-5", [], kwargs)
        t0 = datetime.now()
        self.meter.log_success_event(kwargs, resp, t0, t0)
        return PipelineResult(question=question, answer="x", llm_calls=1)


def test_answered_row_with_a_refusal_is_flagged(monkeypatch):
    meter = LiteLLMMeter()
    monkeypatch.setattr(common, "_meter", meter)
    rows = run_pipeline_on_dataset(_RefusedButAnswered(meter), [{"id": "q", "question": "?"}], "p")
    assert "error" not in rows[0]
    assert rows[0]["refusal_seen"] is True
    # The row carries the new fields even when the raw body names no category.
    assert rows[0]["refusal_categories"] == {"null": 1}
    assert rows[0]["usage_keys_seen"] == []


def _raw_call(meter: LiteLLMMeter, call_id: str, raw: dict) -> None:
    """A successful Anthropic call whose raw response body is `raw`."""
    kwargs = {
        "litellm_call_id": call_id,
        "model": "anthropic/claude-sonnet-5-5",
        "response_cost": 0.0,
        "original_response": json.dumps(raw),
    }
    resp = _response(model="claude-sonnet-5-5", reasoning=0)
    resp.choices = [SimpleNamespace(finish_reason="stop")]
    meter.log_pre_api_call("anthropic/claude-sonnet-5-5", [], kwargs)
    t0 = datetime.now()
    meter.log_success_event(kwargs, resp, t0, t0)


_SONNET_USAGE = {
    "input_tokens": 900,
    "output_tokens": 213,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "service_tier": "standard",
}


def test_refusal_categories_are_read_from_raw_stop_details():
    """litellm drops stop_details with the stop reason; the raw body keeps them."""
    meter = LiteLLMMeter()
    refusal = {
        "type": "message",
        "content": [{"type": "text", "text": "This likely refers to"}],
        "stop_reason": "refusal",
        "usage": _SONNET_USAGE,
    }
    _raw_call(
        meter,
        "a",
        {
            **refusal,
            "stop_details": {
                "type": "refusal",
                "category": "reasoning_extraction",
                "explanation": "asked for internal reasoning",
            },
        },
    )
    _raw_call(
        meter,
        "b",
        {
            **refusal,
            "stop_details": {
                "type": "refusal",
                "category": "reasoning_extraction",
                "explanation": None,
            },
        },
    )
    _raw_call(meter, "c", {**refusal, "stop_details": {"type": "refusal", "category": None}})
    _raw_call(meter, "d", {**refusal, "stop_reason": "end_turn", "stop_details": None})

    usage = meter.usage_since(0)
    assert usage["refused_calls"] == 3
    assert usage["refusal_categories"] == {"reasoning_extraction": 2, "null": 1}
    assert usage["refusal_explained"] == 1
    assert usage["finish_reasons"] == {"stop": 4}


def test_a_refusal_without_stop_details_is_still_counted():
    meter = LiteLLMMeter()
    _raw_call(meter, "a", {"stop_reason": "refusal", "usage": {}})
    usage = meter.usage_since(0)
    assert usage["refused_calls"] == 1
    assert usage["refusal_categories"] == {"null": 1}


def test_raw_usage_keys_and_thinking_blocks_are_recorded():
    """Tells a model that did not think apart from a field that never arrived."""
    meter = LiteLLMMeter()
    _raw_call(
        meter,
        "plain",
        {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "x"}],
            "usage": _SONNET_USAGE,
        },
    )
    _raw_call(
        meter,
        "thought",
        {
            "stop_reason": "end_turn",
            "content": [
                {"type": "thinking", "thinking": "", "signature": "s"},
                {"type": "text", "text": "y"},
            ],
            "usage": {**_SONNET_USAGE, "output_tokens_details": {"thinking_tokens": 120}},
        },
    )
    usage = meter.usage_since(0)
    assert usage["usage_keys_seen"] == sorted(
        [*_SONNET_USAGE, "output_tokens_details", "output_tokens_details.thinking_tokens"]
    )
    assert usage["output_tokens_details_seen"] is True
    assert usage["thinking_blocks"] == 1
    assert usage["reasoning_tokens"] == 120
    assert usage["refusal_categories"] == {}

    only_plain = LiteLLMMeter()
    _raw_call(only_plain, "plain", {"stop_reason": "end_turn", "usage": _SONNET_USAGE})
    usage = only_plain.usage_since(0)
    assert usage["output_tokens_details_seen"] is False
    assert usage["thinking_blocks"] == 0


def test_calls_without_a_raw_body_add_no_usage_keys():
    """gpt-5-mini rows keep their shape: the new fields stay empty."""
    meter = LiteLLMMeter()
    _finished(meter, "ok", "stop")
    usage = meter.usage_since(0)
    assert usage["usage_keys_seen"] == []
    assert usage["refusal_categories"] == {}
    assert usage["thinking_blocks"] == 0
