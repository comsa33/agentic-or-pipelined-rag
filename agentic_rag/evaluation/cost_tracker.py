"""API cost and latency tracker for experiment reproducibility.

Tracks per-call and aggregate statistics for:
  - LLM API costs (input/output tokens × price)
  - Embedding API costs
  - Latency per pipeline stage
  - Total experiment cost/time

Results are auto-logged for inclusion in the paper.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from litellm.integrations.custom_logger import CustomLogger
from loguru import logger

# The refusal policy lives with the exclusion rules it feeds, which analysis
# code reads without importing litellm.
from experiments.exclusions import REFUSAL_RETRIES, RETRIED_REFUSAL_CATEGORIES

# ---------------------------------------------------------------------------
# Pricing (USD per 1M tokens, as of 2024-12)
# Update these when models/pricing change.
# ---------------------------------------------------------------------------
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
}


def _estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate USD cost for a single API call."""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        # Try prefix match
        for key, val in MODEL_PRICING.items():
            if model.startswith(key):
                pricing = val
                break
    if pricing is None:
        logger.warning(f"No pricing for model '{model}', using gpt-4o-mini rates")
        pricing = MODEL_PRICING["gpt-4o-mini"]

    cost = (
        input_tokens * pricing["input"] / 1_000_000 + output_tokens * pricing["output"] / 1_000_000
    )
    return cost


# ---------------------------------------------------------------------------
# Call record
# ---------------------------------------------------------------------------
@dataclass
class APICallRecord:
    """Record of a single API call."""

    timestamp: float
    model: str
    stage: str  # e.g., "preprocess", "evaluate", "generate", "agent", "embed"
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "model": self.model,
            "stage": self.stage,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# Cost tracker
# ---------------------------------------------------------------------------
class CostTracker:
    """Track API costs and latency across an experiment run."""

    def __init__(self):
        self.records: list[APICallRecord] = []
        self._stage_timers: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record(
        self,
        model: str,
        stage: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        **metadata,
    ) -> APICallRecord:
        """Record a single API call."""
        total = input_tokens + output_tokens
        cost = _estimate_cost(model, input_tokens, output_tokens)

        rec = APICallRecord(
            timestamp=time.time(),
            model=model,
            stage=stage,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            latency_ms=latency_ms,
            cost_usd=cost,
            metadata=metadata,
        )
        self.records.append(rec)
        return rec

    @contextmanager
    def track_stage(self, stage: str):
        """Context manager to track latency of a pipeline stage."""
        start = time.perf_counter()
        yield
        elapsed = (time.perf_counter() - start) * 1000
        self._stage_timers[stage] = self._stage_timers.get(stage, 0.0) + elapsed

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    @property
    def total_cost(self) -> float:
        """Total cost in USD."""
        return sum(r.cost_usd for r in self.records)

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return sum(r.total_tokens for r in self.records)

    @property
    def total_calls(self) -> int:
        """Total number of API calls."""
        return len(self.records)

    def summary(self) -> dict:
        """Generate a summary of costs and latency."""
        by_stage: dict[str, dict] = defaultdict(
            lambda: {"calls": 0, "tokens": 0, "cost": 0.0, "latency_ms": 0.0}
        )
        by_model: dict[str, dict] = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost": 0.0})

        for r in self.records:
            by_stage[r.stage]["calls"] += 1
            by_stage[r.stage]["tokens"] += r.total_tokens
            by_stage[r.stage]["cost"] += r.cost_usd
            by_stage[r.stage]["latency_ms"] += r.latency_ms

            by_model[r.model]["calls"] += 1
            by_model[r.model]["tokens"] += r.total_tokens
            by_model[r.model]["cost"] += r.cost_usd

        return {
            "total_cost_usd": self.total_cost,
            "total_tokens": self.total_tokens,
            "total_calls": self.total_calls,
            "by_stage": dict(by_stage),
            "by_model": dict(by_model),
            "stage_timers_ms": dict(self._stage_timers),
        }

    def print_summary(self) -> None:
        """Print a formatted cost summary."""
        s = self.summary()
        print("\n" + "=" * 60)
        print("EXPERIMENT COST SUMMARY")
        print("=" * 60)
        print(f"Total Cost:   ${s['total_cost_usd']:.4f}")
        print(f"Total Tokens: {s['total_tokens']:,}")
        print(f"Total Calls:  {s['total_calls']}")
        print()

        print("By Stage:")
        for stage, data in s["by_stage"].items():
            print(
                f"  {stage:20s} | "
                f"calls={data['calls']:3d} | "
                f"tokens={data['tokens']:8,} | "
                f"${data['cost']:.4f} | "
                f"{data['latency_ms']:.0f}ms"
            )

        print()
        print("By Model:")
        for model, data in s["by_model"].items():
            print(
                f"  {model:30s} | "
                f"calls={data['calls']:3d} | "
                f"tokens={data['tokens']:8,} | "
                f"${data['cost']:.4f}"
            )

        if s["stage_timers_ms"]:
            print()
            print("Stage Latency:")
            for stage, ms in s["stage_timers_ms"].items():
                print(f"  {stage:20s} | {ms:.0f}ms")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Save all records and summary to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.summary(),
            "records": [r.to_dict() for r in self.records],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Cost report saved to {path}")

    def load(self, path: Path) -> None:
        """Load records from a saved JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.records = [
            APICallRecord(
                timestamp=r["timestamp"],
                model=r["model"],
                stage=r["stage"],
                input_tokens=r.get("input_tokens", 0),
                output_tokens=r.get("output_tokens", 0),
                total_tokens=r.get("total_tokens", 0),
                latency_ms=r.get("latency_ms", 0.0),
                cost_usd=r.get("cost_usd", 0.0),
            )
            for r in data.get("records", [])
        ]

    def reset(self) -> None:
        """Clear all records."""
        self.records.clear()
        self._stage_timers.clear()


# ---------------------------------------------------------------------------
# Measured usage (litellm callback)
# ---------------------------------------------------------------------------
# `CostTracker` above prices calls from a hand-maintained table, which is only
# ever an estimate: it does not know about cached-prompt discounts, reasoning
# tokens, or a provider's actual price list. `LiteLLMMeter` records what
# litellm itself reports for every completed call -- token breakdown,
# `response_cost`, and the model snapshot the provider answered with -- so the
# per-question cost written to a result row is a measurement, not a guess.
# (The estimate was 2-3x off the measured gpt-5-mini cost in September 2026.)
#
# litellm delivers success callbacks from a thread pool *after* the completion
# returns, while the pre-call hook runs synchronously in the caller's thread.
# The meter therefore tracks in-flight calls by id and offers `drain()` so a
# caller can wait for every call it started before reading the numbers.


@dataclass
class MeteredCall:
    """One completed LLM call as litellm reported it."""

    call_id: str
    model_spec: str  # model string as requested, e.g. "azure/gpt-5-mini"
    response_model: str  # model the provider answered with, e.g. "gpt-5-mini-2025-08-07"
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    # Why the provider stopped, as litellm normalises it ("stop", "length",
    # "tool_calls", ...). "length" means the call hit max_tokens: with a
    # thinking model the hidden reasoning can use the whole budget and leave
    # an empty answer that still scores as a valid prediction.
    finish_reason: str = ""
    # The provider's own stop reason where litellm loses it. Anthropic
    # returns a refusal as HTTP 200 with stop_reason "refusal", which litellm
    # maps to "stop", so the call would read as an ordinary empty answer.
    stop_reason: str = ""
    # For a refusal, the category in the raw `stop_details` ("cyber", "bio",
    # "frontier_llm", "reasoning_extraction", "general_harms", ...; "null"
    # when the response names none) and whether it gave an explanation.
    # Empty for every other call.
    refusal_category: str = ""
    refusal_explained: bool = False
    # Keys of the raw response's `usage` object, with the keys of
    # `output_tokens_details` as "output_tokens_details.<key>". Records which
    # fields the provider actually sent, so a zero reasoning-token count can
    # be told apart from a field that never arrived.
    usage_keys: tuple[str, ...] = ()
    # Thinking blocks in the raw response content. With thinking display
    # "omitted" their text is empty, and litellm derives reasoning tokens by
    # counting that text, so this is the signal that the model thought.
    thinking_blocks: int = 0
    # Which resend of a refused request this call was (0: the first send);
    # see RETRIED_REFUSAL_CATEGORIES.
    refusal_retry: int = 0
    # HTTP status of a call that raised ("400", "429", ...; "unknown" when the
    # error carries none). Empty for a call that returned. A failed call has
    # no usage and is kept out of every total but the failure counts.
    failed_status: str = ""
    # What the request asked for and carried; see `_request_summary`.
    request: dict = field(default_factory=dict)
    # Requests litellm sent for this record (its pre-call events). Retries
    # that litellm makes after a reported failure reuse the call id and are
    # added to the failure's record.
    attempts: int = 0
    cost_usd: float | None = None
    latency_ms: float = 0.0
    started_at: float = 0.0
    ended_at: float = 0.0


# Request body fields recorded per call. Values are kept only for fields that
# are model parameters; nothing from headers or credentials is read.
_BODY_VALUE_FIELDS = ("max_tokens", "max_completion_tokens", "output_config", "reasoning_effort")
_BODY_PRESENCE_FIELDS = ("temperature", "thinking", "tools", "tool_choice", "response_format")


def _structured_kind(response_format, body: dict) -> str:
    """Which structured-output mode DSPy asked for, or "" for plain text.

    The codebase never selects DSPy's JSONAdapter, so a structured request
    is the adapter's fallback after ChatAdapter failed to parse a reply.
    "json_schema" is its first try (a schema, sent as response_format or,
    for Anthropic, as a forced tool), "json_object" its second.
    """
    if response_format is not None:
        if isinstance(response_format, dict):
            return str(response_format.get("type") or "response_format")
        return "json_schema"  # a pydantic model
    if body.get("tool_choice") or body.get("tools"):
        return "tools"
    return ""


def _request_summary(kwargs) -> dict:
    additional = kwargs.get("additional_args") or {}
    body = additional.get("complete_input_dict")
    body = body if isinstance(body, dict) else {}
    summary = {k: body[k] for k in _BODY_VALUE_FIELDS if k in body}
    summary["sent"] = sorted(k for k in _BODY_PRESENCE_FIELDS if body.get(k) is not None)
    summary["structured"] = _structured_kind(kwargs.get("response_format"), body)
    return summary


def _usage_field(obj, name: str) -> int:
    value = getattr(obj, name, None) if obj is not None else None
    return int(value or 0)


def _finish_reason(response_obj) -> str:
    try:
        return str(response_obj.choices[0].finish_reason or "")
    except (AttributeError, IndexError, TypeError):
        return ""


def _raw_response(kwargs) -> dict:
    raw = kwargs.get("original_response")
    if not isinstance(raw, str):
        return {}
    try:
        body = json.loads(raw)
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _anthropic_thinking_tokens(kwargs) -> int:
    """Thinking tokens from the raw Anthropic response.

    litellm 1.82 leaves `completion_tokens_details.reasoning_tokens` at 0 for
    Anthropic and does not map `usage.output_tokens_details.thinking_tokens`,
    where the API reports them. Billing is unaffected (`output_tokens` already
    includes thinking); only the reasoning-token count would read as zero.
    """
    details = (_raw_response(kwargs).get("usage") or {}).get("output_tokens_details") or {}
    return int(details.get("thinking_tokens") or 0)


def _anthropic_stop_reason(kwargs) -> str:
    return str(_raw_response(kwargs).get("stop_reason") or "")


def _refusal_details(raw: dict) -> tuple[str, bool]:
    """Category and explanation presence of a refusal's `stop_details`.

    The API fills `stop_details` only when stop_reason is "refusal"; litellm
    drops it along with the stop reason, so it is read from the raw body.
    """
    if raw.get("stop_reason") not in REFUSAL_STOP_REASONS:
        return "", False
    details = raw.get("stop_details")
    details = details if isinstance(details, dict) else {}
    category = details.get("category")
    return (str(category) if category else "null"), bool(details.get("explanation"))


def _usage_keys(raw: dict) -> tuple[str, ...]:
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        return ()
    keys = set(usage)
    details = usage.get("output_tokens_details")
    if isinstance(details, dict):
        keys.update(f"output_tokens_details.{k}" for k in details)
    return tuple(sorted(keys))


def _thinking_blocks(raw: dict) -> int:
    content = raw.get("content")
    if not isinstance(content, list):
        return 0
    return sum(
        1
        for block in content
        if isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking")
    )


# failed_status of a call that hit the client timeout. Kept apart from other
# failures: the provider may have finished, and billed, the request.
TIMEOUT_STATUS = "timeout"


def _failed_status(kwargs) -> str:
    import litellm

    exc = kwargs.get("exception")
    if isinstance(exc, litellm.Timeout):
        return TIMEOUT_STATUS
    status = getattr(exc, "status_code", None)
    return str(status) if status else "unknown"


# finish_reason values that mean the call ran into max_tokens.
TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})

# Raw stop reasons that mean the provider declined to answer.
REFUSAL_STOP_REASONS = frozenset({"refusal"})

# Per-thread state shared by the Anthropic LM and the hooks below. litellm
# runs its pre- and post-call hooks synchronously in the calling thread, so
# the LM can tag the calls it is about to make and read what the last one
# returned before deciding whether to send it again.
_call_state = threading.local()


@contextmanager
def refusal_retry(index: int):
    """Tag the calls made inside as resend `index` of a refused request (0: first send).

    Clears the last raw refusal first, so `last_refusal_category` afterwards
    describes a call made inside this block.
    """
    previous = getattr(_call_state, "retry_index", 0)
    _call_state.retry_index = index
    _call_state.refusal = None
    try:
        yield
    finally:
        _call_state.retry_index = previous


def last_refusal_category() -> str | None:
    """Category of the last call's refusal in this thread; None if it was not refused.

    "null" when the refusal named no category.
    """
    return getattr(_call_state, "refusal", None)


class RawStopWatcher(CustomLogger):
    """Records, per thread, whether the last raw response was a refusal.

    litellm maps Anthropic's stop_reason "refusal" to "stop" and keeps
    stop_details out of the response it returns; both are only in the raw
    body, which reaches the post-call hook before the completion returns.
    """

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time) -> None:
        raw = _raw_response(kwargs)
        category, _ = _refusal_details(raw)
        _call_state.refusal = category or None


def install_raw_stop_watcher() -> None:
    """Register one RawStopWatcher with litellm (idempotent)."""
    import litellm

    if not any(isinstance(cb, RawStopWatcher) for cb in litellm.callbacks or []):
        litellm.callbacks = [*(litellm.callbacks or []), RawStopWatcher()]


def empty_usage() -> dict:
    """The usage block of a result row before any call has been metered."""
    return {
        "metered_calls": 0,
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "calls_without_cost": 0,
        "response_models": [],
        # False when a call started by this question never reported in
        # time: the totals above are then a lower bound.
        "usage_complete": True,
        "leaked_calls": 0,
        "finish_reasons": {},
        "truncated_calls": 0,
        # Refused calls, every send counted: a refused request that was sent
        # again and refused again counts twice. So refused_calls > 0 means
        # some request was refused on its first send ("first refusal", the
        # exclusion trigger of the sensitivity analysis).
        "refused_calls": 0,
        # Refused calls by `stop_details.category` ("null" when none was
        # given), every send counted, and how many carried an explanation.
        "refusal_categories": {},
        "refusal_explained": 0,
        # Resends of a refused request (RETRIED_REFUSAL_CATEGORIES), the
        # resends that came back unrefused, and the refusals that stood: a
        # category that is not retried, or the last resend refused again
        # ("final refusal", the exclusion trigger of the main analysis).
        "refusal_retries": 0,
        "refusals_recovered": 0,
        "refusals_final_by_category": {},
        # Union of the raw `usage` keys over the window's calls (see
        # MeteredCall.usage_keys), whether any carried output_tokens_details,
        # and the thinking blocks the responses contained.
        "usage_keys_seen": [],
        "output_tokens_details_seen": False,
        "thinking_blocks": 0,
        # Calls that raised instead of returning, by HTTP status. When litellm
        # retries (all attempts share one call id) it reports only the first
        # failure of the sequence, so for those models this counts sequences
        # that hit an error, not attempts, and a sequence whose retry later
        # succeeded is counted here and in metered_calls. Anthropic models
        # retry outside litellm (see make_lm), so there every attempt counts.
        "failed_calls": 0,
        "failed_by_status": {},
        # Calls that hit the client timeout. Not in failed_calls: the
        # provider may have completed and billed them, so a row with any is
        # marked usage_complete=False.
        "timeout_calls": 0,
        # Calls that asked for structured output, by mode (see
        # `_structured_kind`): each one is a DSPy adapter fallback.
        "structured_calls": {},
        # Requests sent to the provider, retries included. Unlike
        # failed_calls this has one unit for every provider, whether
        # litellm or make_lm does the retrying.
        "attempts": 0,
    }


class LiteLLMMeter(CustomLogger):
    """Collect measured token counts, cost and response model per LLM call.

    Install once per process with `install()`. Aggregate a window of calls
    with `mark()` before the work and `usage_since(mark)` after it.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._pending: set[str] = set()
        # Calls whose report did not arrive before `drain` gave up. Their
        # usage belongs to no row: when the report finally lands it is
        # dropped rather than counted into whichever question is running.
        self._leaked: set[str] = set()
        self.leaked_calls: int = 0
        self.calls: list[MeteredCall] = []
        self.failures: int = 0
        # Calls whose failure was reported. A retry re-registers the same
        # call id as pending after its failure; if every attempt fails, that
        # id never reports again and is settled by `drain`, not lost.
        self._failed_ids: set[str] = set()
        # Request summary of each call in flight, by call id.
        self._requests: dict[str, dict] = {}
        # Which resend of a refused request each pending call is (0: first send).
        self._resends: dict[str, int] = {}
        # Pre-call events not yet assigned to a record, and the failure
        # record that later retries under the same id are added to.
        self._attempts: dict[str, int] = {}
        self._failed_records: dict[str, MeteredCall] = {}

    # -- litellm hooks -------------------------------------------------
    def log_pre_api_call(self, model, messages, kwargs) -> None:
        call_id = kwargs.get("litellm_call_id")
        if call_id:
            summary = _request_summary(kwargs)
            with self._lock:
                self._pending.add(call_id)
                self._requests[call_id] = summary
                self._resends[call_id] = getattr(_call_state, "retry_index", 0)
                failed = self._failed_records.get(call_id)
                if failed is not None:
                    failed.attempts += 1
                else:
                    self._attempts[call_id] = self._attempts.get(call_id, 0) + 1

    def log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        usage = getattr(response_obj, "usage", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        cost = kwargs.get("response_cost")
        started = start_time.timestamp() if hasattr(start_time, "timestamp") else 0.0
        ended = end_time.timestamp() if hasattr(end_time, "timestamp") else 0.0
        raw = _raw_response(kwargs)
        refusal_category, refusal_explained = _refusal_details(raw)
        rec = MeteredCall(
            call_id=kwargs.get("litellm_call_id") or "",
            model_spec=str(kwargs.get("model") or ""),
            response_model=str(getattr(response_obj, "model", "") or ""),
            prompt_tokens=_usage_field(usage, "prompt_tokens"),
            cached_tokens=_usage_field(prompt_details, "cached_tokens"),
            completion_tokens=_usage_field(usage, "completion_tokens"),
            reasoning_tokens=_usage_field(completion_details, "reasoning_tokens")
            or _anthropic_thinking_tokens(kwargs),
            finish_reason=_finish_reason(response_obj),
            stop_reason=_anthropic_stop_reason(kwargs),
            refusal_category=refusal_category,
            refusal_explained=refusal_explained,
            usage_keys=_usage_keys(raw),
            thinking_blocks=_thinking_blocks(raw),
            cost_usd=float(cost) if cost is not None else None,
            latency_ms=(ended - started) * 1000 if started and ended else 0.0,
            started_at=started,
            ended_at=ended,
        )
        with self._lock:
            self._failed_ids.discard(rec.call_id)
            rec.request = self._requests.pop(rec.call_id, {})
            rec.refusal_retry = self._resends.pop(rec.call_id, 0)
            rec.attempts = self._attempts.pop(rec.call_id, 0)
            self._failed_records.pop(rec.call_id, None)
            if rec.call_id in self._leaked:
                self._leaked.discard(rec.call_id)
                return
            self.calls.append(rec)
            self._pending.discard(rec.call_id)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time) -> None:
        call_id = kwargs.get("litellm_call_id") or ""
        started = start_time.timestamp() if hasattr(start_time, "timestamp") else 0.0
        ended = end_time.timestamp() if hasattr(end_time, "timestamp") else 0.0
        rec = MeteredCall(
            call_id=call_id,
            model_spec=str(kwargs.get("model") or ""),
            response_model="",
            failed_status=_failed_status(kwargs),
            latency_ms=(ended - started) * 1000 if started and ended else 0.0,
            started_at=started,
            ended_at=ended,
        )
        with self._lock:
            self.failures += 1
            self._pending.discard(call_id)
            self._failed_ids.add(call_id)
            # Kept, not popped: a litellm retry reuses the id.
            rec.request = self._requests.get(call_id, {})
            rec.attempts = self._attempts.pop(call_id, 0)
            self._failed_records[call_id] = rec
            if call_id in self._leaked:
                self._leaked.discard(call_id)
                return
            self.calls.append(rec)

    # -- lifecycle -----------------------------------------------------
    def install(self) -> LiteLLMMeter:
        """Register with litellm, replacing any meter installed earlier."""
        import litellm

        litellm.callbacks = [
            cb for cb in (litellm.callbacks or []) if not isinstance(cb, LiteLLMMeter)
        ] + [self]
        return self

    def drain(self, timeout: float = 10.0) -> bool:
        """Wait until every call that started has been reported.

        Returns False if some callback never arrived within `timeout`. The
        numbers read afterwards are then a lower bound, and the missing
        calls are written off as leaked so that their late reports cannot
        be counted into the next window.

        A call still pending after a reported failure is a retry sequence
        that failed to the end: litellm re-registers the id for each retry
        but reports only the first failure. It is already counted in
        failed_calls and carries no usage, so it is settled without being
        written off as leaked. It still waits out `timeout`, because a
        retry that succeeded may report late.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if not self._pending:
                    # Nothing is in flight, so no retry can follow.
                    self._requests.clear()
                    self._resends.clear()
                    self._attempts.clear()
                    self._failed_records.clear()
                    return True
            if time.monotonic() >= deadline:
                with self._lock:
                    for call_id in self._pending:
                        self._requests.pop(call_id, None)
                        self._resends.pop(call_id, None)
                        self._attempts.pop(call_id, None)
                        self._failed_records.pop(call_id, None)
                    lost = self._pending - self._failed_ids
                    n = len(lost)
                    self._leaked |= self._pending
                    self._failed_ids -= self._pending
                    self._pending.clear()
                    self.leaked_calls += n
                if not n:
                    return True
                logger.warning(
                    f"LiteLLMMeter: {n} call(s) never reported within {timeout}s; "
                    f"their usage is lost to every row"
                )
                return False
            time.sleep(0.01)

    # -- aggregation ---------------------------------------------------
    def mark(self) -> int:
        """Position in the call log; pass to `usage_since` to read a window."""
        with self._lock:
            return len(self.calls)

    def usage_since(self, mark: int) -> dict:
        """Aggregate every call recorded after `mark`."""
        with self._lock:
            window = self.calls[mark:]
        return self._aggregate(window)

    def usage_between(self, start: int, end: int) -> dict:
        """Aggregate the calls recorded between two marks."""
        with self._lock:
            window = self.calls[start:end]
        return self._aggregate(window)

    def usage_total(self) -> dict:
        with self._lock:
            window = list(self.calls)
        return self._aggregate(window)

    @property
    def observed_models(self) -> set[str]:
        with self._lock:
            return {c.response_model for c in self.calls if c.response_model}

    @staticmethod
    def _aggregate(window: list[MeteredCall]) -> dict:
        out = empty_usage()
        keys_seen: set[str] = set()
        for c in window:
            out["attempts"] += c.attempts
            kind = c.request.get("structured")
            if kind:
                structured = out["structured_calls"]
                structured[kind] = structured.get(kind, 0) + 1
            if c.failed_status == TIMEOUT_STATUS:
                out["timeout_calls"] += 1
                continue
            if c.failed_status:
                out["failed_calls"] += 1
                by_status = out["failed_by_status"]
                by_status[c.failed_status] = by_status.get(c.failed_status, 0) + 1
                continue
            out["metered_calls"] += 1
            out["prompt_tokens"] += c.prompt_tokens
            out["cached_tokens"] += c.cached_tokens
            out["completion_tokens"] += c.completion_tokens
            out["reasoning_tokens"] += c.reasoning_tokens
            if c.finish_reason:
                reasons = out["finish_reasons"]
                reasons[c.finish_reason] = reasons.get(c.finish_reason, 0) + 1
            if c.finish_reason in TRUNCATED_FINISH_REASONS:
                out["truncated_calls"] += 1
            resend = c.refusal_retry
            refused = c.stop_reason in REFUSAL_STOP_REASONS
            if resend:
                out["refusal_retries"] += 1
                out["refusals_recovered"] += int(not refused)
            if refused:
                out["refused_calls"] += 1
                categories = out["refusal_categories"]
                category = c.refusal_category or "null"
                categories[category] = categories.get(category, 0) + 1
                out["refusal_explained"] += int(c.refusal_explained)
                if category not in RETRIED_REFUSAL_CATEGORIES or resend >= REFUSAL_RETRIES:
                    final = out["refusals_final_by_category"]
                    final[category] = final.get(category, 0) + 1
            keys_seen.update(c.usage_keys)
            out["thinking_blocks"] += c.thinking_blocks
            if c.cost_usd is None:
                out["calls_without_cost"] += 1
            else:
                out["cost_usd"] += c.cost_usd
        out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
        out["usage_keys_seen"] = sorted(keys_seen)
        out["output_tokens_details_seen"] = "output_tokens_details" in keys_seen
        out["response_models"] = sorted({c.response_model for c in window if c.response_model})
        return out
