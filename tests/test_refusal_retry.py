"""A reasoning_extraction refusal is sent again; every other refusal stands.

Anthropic returns a refusal as HTTP 200 with stop_reason "refusal" and the
category in stop_details; litellm hands back an ordinary "stop" and drops
both. These tests answer the LM's requests from a fake HTTP server with raw
Anthropic bodies, so the decision is exercised the way it runs. No request
leaves the process: real transports are replaced for the whole module.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import litellm
import pytest
from litellm.llms.custom_httpx.http_handler import HTTPHandler

from agentic_rag.config.settings import _anthropic_lm_class, make_lm
from agentic_rag.evaluation.cost_tracker import LiteLLMMeter
from experiments.common import classify_error, refusal_first_seen, refusal_seen
from experiments.exclusions import REFUSAL_RETRIES, refusal_final, refusal_ids

MODEL = "anthropic/claude-sonnet-5-5"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(self, request):
        raise RuntimeError(f"network blocked in tests: {request.url}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", blocked)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # litellm copies callbacks into per-event lists and skips a logger whose
    # class is already registered, so a meter from another test would keep
    # receiving the events. Start every test with none.
    for name in (
        "callbacks",
        "input_callback",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        monkeypatch.setattr(litellm, name, [])
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.fixture
def meter():
    m = LiteLLMMeter().install()
    yield m


def _anthropic_body(stop_reason="end_turn", category=None, text="[[ ## answer ## ]]\nParis"):
    body = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_details": None,
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }
    if stop_reason == "refusal":
        body["stop_details"] = {"type": "refusal", "category": category, "explanation": None}
    return body


class Server:
    """Answers each request with the next body in `bodies`, recording requests."""

    def __init__(self, bodies: list[dict]) -> None:
        self.bodies = list(bodies)
        self.requests: list[dict] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        return httpx.Response(200, json=self.bodies.pop(0))


def _anthropic_lm(server: Server):
    client = HTTPHandler()
    client.client = httpx.Client(transport=httpx.MockTransport(server.handle))
    return _anthropic_lm_class()(
        MODEL, num_retries=0, transient_retries=0, max_tokens=1000, cache=False, client=client
    )


def _ask(lm) -> str:
    return lm(messages=[{"role": "user", "content": "Which city?"}])[0]


def _usage(meter: LiteLLMMeter) -> dict:
    assert meter.drain(5.0)
    return meter.usage_since(0)


def test_reasoning_extraction_refusal_is_resent_until_answered(meter):
    server = Server(
        [
            _anthropic_body("refusal", "reasoning_extraction", text="This likely"),
            _anthropic_body("refusal", "reasoning_extraction", text="This likely"),
            _anthropic_body(),
        ]
    )
    assert "Paris" in _ask(_anthropic_lm(server))
    assert len(server.requests) == 3
    assert server.requests[1] == server.requests[0] == server.requests[2], "resend changed"

    usage = _usage(meter)
    assert usage["metered_calls"] == 3, "every send is metered"
    assert usage["refused_calls"] == 2
    assert usage["refusal_categories"] == {"reasoning_extraction": 2}
    assert usage["refusal_retries"] == 2
    assert usage["refusals_recovered"] == 1
    assert usage["refusals_final_by_category"] == {}
    assert not refusal_final(usage)
    assert not refusal_seen(usage, None), "a recovered refusal does not exclude (main)"
    assert refusal_first_seen(usage, None), "but it does in the sensitivity analysis"
    assert classify_error(usage) == "other"


def test_reasoning_extraction_refused_on_every_resend_stands(meter):
    refused = _anthropic_body("refusal", "reasoning_extraction", text="This likely")
    server = Server([refused] * (REFUSAL_RETRIES + 1))
    _ask(_anthropic_lm(server))
    assert len(server.requests) == REFUSAL_RETRIES + 1 == 4

    usage = _usage(meter)
    assert usage["metered_calls"] == 4
    assert usage["refused_calls"] == 4
    assert usage["refusal_retries"] == 3
    assert usage["refusals_recovered"] == 0
    assert usage["refusals_final_by_category"] == {"reasoning_extraction": 1}
    assert refusal_seen(usage, None)
    assert classify_error(usage) == "refused"


@pytest.mark.parametrize("category", ["cyber", "general_harms", None])
def test_other_refusals_are_not_resent(meter, category):
    server = Server([_anthropic_body("refusal", category, text="I can't")])
    _ask(_anthropic_lm(server))
    assert len(server.requests) == 1

    usage = _usage(meter)
    key = category or "null"
    assert usage["refusal_retries"] == 0
    assert usage["refusals_final_by_category"] == {key: 1}
    assert refusal_seen(usage, None) and refusal_first_seen(usage, None)
    assert classify_error(usage) == "refused"


def test_a_resend_refused_in_another_category_stands_in_that_category(meter):
    server = Server(
        [
            _anthropic_body("refusal", "reasoning_extraction", text="This likely"),
            _anthropic_body("refusal", "cyber", text="I can't"),
        ]
    )
    _ask(_anthropic_lm(server))
    assert len(server.requests) == 2
    usage = _usage(meter)
    assert usage["refusals_final_by_category"] == {"cyber": 1}
    assert usage["refusals_recovered"] == 0


def test_openai_and_azure_paths_are_never_resent(meter):
    """Only the Anthropic LM resends; gpt-5-mini's path is a plain dspy.LM."""
    assert type(make_lm("azure/gpt-5-mini")).__name__ == "LM"
    assert type(make_lm("openai/gpt-4.1-nano")).__name__ == "LM"

    sent = []
    # An OpenAI-shaped body that also carries Anthropic's refusal fields: even
    # if a raw body looked like a reasoning_extraction refusal, nothing resends.
    body = {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4.1-nano",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Paris"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "stop_reason": "refusal",
        "stop_details": {"type": "refusal", "category": "reasoning_extraction"},
    }

    def handle(request):
        sent.append(request)
        return httpx.Response(200, json=body)

    litellm.client_session = httpx.Client(transport=httpx.MockTransport(handle))
    try:
        lm = make_lm("openai/gpt-4.1-nano", cache=False)
        lm(messages=[{"role": "user", "content": "Which city?"}])
    finally:
        litellm.client_session = None
        litellm.in_memory_llm_clients_cache.flush_cache()
    assert len(sent) == 1
    assert _usage(meter)["refusal_retries"] == 0


# ---------------------------------------------------------------------------
# Exclusion triggers on rows
# ---------------------------------------------------------------------------
def _row(qid, **usage):
    return {"id": qid, **usage}


def test_refusal_ids_split_main_and_sensitivity_triggers():
    rows = [
        # recovered: sensitivity only
        _row(
            "recovered",
            refused_calls=1,
            refusal_categories={"reasoning_extraction": 1},
            refusals_final_by_category={},
            refusal_retries=1,
            refusals_recovered=1,
        ),
        # stood after every resend
        _row(
            "stood",
            refused_calls=4,
            refusal_categories={"reasoning_extraction": 4},
            refusals_final_by_category={"reasoning_extraction": 1},
        ),
        # another category, in an earlier attempt at the item
        _row(
            "prior",
            refused_calls=0,
            refusals_final_by_category={},
            prior_attempt_usage={
                "refused_calls": 1,
                "refusal_categories": {"cyber": 1},
                "refusals_final_by_category": {"cyber": 1},
            },
        ),
        _row("clean", refused_calls=0, refusals_final_by_category={}),
    ]
    assert refusal_ids(rows) == ["prior", "stood"]
    assert refusal_ids(rows, first=True) == ["prior", "recovered", "stood"]


def test_rows_from_before_resends_keep_their_trigger():
    """gpt-5-mini and earlier Sonnet rows carry no resend fields: any refusal stood."""
    old = [_row("a", refused_calls=1), _row("b", refused_calls=0)]
    assert refusal_ids(old) == refusal_ids(old, first=True) == ["a"]
    assert refusal_final({"refused_calls": 1})
    assert not refusal_final({"refused_calls": 0})


# ---------------------------------------------------------------------------
# Answer-level refusal patterns (Sonnet surface forms)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prediction",
    [
        "The passages do not say which film this is.",
        "They don't say who directed it.",
        "The documents do not state the year.",
        "The passages don\u2019t give this figure.",
        "The passages do not give enough information to decide.",
        "Cannot be computed from the passages.",
    ],
)
def test_sonnet_refusal_forms_are_detected(prediction):
    from experiments.analysis.refusal_analysis import is_refusal

    assert is_refusal(prediction)


def test_a_plain_answer_is_not_a_refusal():
    from experiments.analysis.refusal_analysis import is_refusal

    assert not is_refusal("Justin Spitzer")
    assert is_refusal("Cannot determine from the passages")
