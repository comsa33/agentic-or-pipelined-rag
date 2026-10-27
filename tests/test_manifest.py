"""A run must be able to prove the conditions it ran under.

The March 2026 runs could not, so partial re-runs had to be discarded. The
manifest records the observed model snapshot, the parameters litellm really
forwards, package versions and the commit, and refuses to start when the
provider answers with a snapshot other than the declared one.
"""

from __future__ import annotations

import json

import dspy
import pytest

import experiments.manifest as manifest_mod
from agentic_rag.config.settings import ModelSettings, settings
from agentic_rag.evaluation.cost_tracker import LiteLLMMeter
from experiments.manifest import (
    MANIFEST_FILENAME,
    ManifestMismatchError,
    RunManifest,
    build_manifest,
    git_state,
    sent_params,
)

SNAPSHOT = "gpt-5-mini-2025-08-07"


@pytest.fixture(autouse=True)
def _anthropic_key(monkeypatch, tmp_path):
    """A throwaway key in slot 1; the real environment is restored afterwards."""
    keys = tmp_path / ".env.anthropic"
    keys.write_text("ANTHROPIC_API_KEY_1=sk-test-not-a-key\n")
    monkeypatch.setattr(settings.model, "anthropic_keys_file", keys)
    monkeypatch.setattr(settings.model, "anthropic_key_slot", 1)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def _pin(monkeypatch, value: str) -> None:
    monkeypatch.setattr(settings.model, "expected_response_models", value)


def _build(**overrides) -> RunManifest:
    kwargs = dict(
        run_id="20260907_000000_rq1_2wikimultihopqa_n2_gpt-5-mini",
        experiment="RQ1",
        dataset="2wikimultihopqa",
        n=2,
        variants=["Naive RAG", "Agentic (ReAct)"],
        config_path="configs/experiment/rq1.yaml",
        sample_size=2,
    )
    kwargs.update(overrides)
    return build_manifest(**kwargs)


# ---------------------------------------------------------------------------
# Declared snapshot
# ---------------------------------------------------------------------------
def test_expected_snapshot_is_parsed_per_model_string():
    ms = ModelSettings(
        LLM_EXPECTED_RESPONSE_MODELS=f"azure/gpt-5-mini={SNAPSHOT}, openai/gpt-5-mini={SNAPSHOT}"
    )
    assert ms.expected_response_model("azure/gpt-5-mini") == SNAPSHOT
    assert ms.expected_response_model("openai/gpt-5-mini") == SNAPSHOT
    assert ms.expected_response_model("gemini/gemini-3.1-flash-lite-preview") is None


def test_unpinned_when_nothing_declared():
    ms = ModelSettings(LLM_EXPECTED_RESPONSE_MODELS="")
    assert ms.expected_response_model("azure/gpt-5-mini") is None


# ---------------------------------------------------------------------------
# Parameters as sent
# ---------------------------------------------------------------------------
def test_gpt5_mini_drops_temperature_but_sends_reasoning_effort():
    """What the paper must state: temperature=0 never reached gpt-5-mini."""
    lm = dspy.LM("azure/gpt-5-mini", temperature=0.0, max_tokens=16000, reasoning_effort="low")
    sent = sent_params(lm.model, lm.kwargs)
    assert "temperature" not in sent
    assert sent["reasoning_effort"] == "low"
    assert sent["max_completion_tokens"] == 16000


def test_sonnet_sends_effort_in_output_config_without_temperature():
    """Claude 5.x rejects temperature and litellm's reasoning_effort translation."""
    from agentic_rag.config.settings import ANTHROPIC_MIN_MAX_TOKENS, make_lm

    lm = make_lm("anthropic/claude-sonnet-5-5")
    # dspy.LM fills in temperature=None; litellm leaves a None out of the request.
    assert lm.kwargs.get("temperature") is None
    assert "reasoning_effort" not in lm.kwargs
    assert lm.kwargs["output_config"] == {"effort": settings.model.reasoning_effort}
    assert lm.kwargs["max_tokens"] >= ANTHROPIC_MIN_MAX_TOKENS
    sent = sent_params(lm.model, lm.kwargs)
    assert "temperature" not in sent
    assert sent["output_config"] == {"effort": settings.model.reasoning_effort}


def test_sonnet_lm_kwargs_at_campaign_settings(monkeypatch):
    """The exact kwargs Sonnet 5.5 runs with; each of these once produced a 400."""
    from agentic_rag.config.settings import make_lm

    monkeypatch.setattr(settings.model, "reasoning_effort", "low")
    monkeypatch.setattr(settings.model, "temperature", 0.0)
    monkeypatch.setattr(settings.model, "max_tokens", 4096)

    lm = make_lm("anthropic/claude-sonnet-5-5")
    assert lm.kwargs.get("temperature") is None
    assert "reasoning_effort" not in lm.kwargs
    assert "thinking" not in lm.kwargs
    assert lm.kwargs["max_tokens"] == 32000
    assert lm.kwargs["output_config"] == {"effort": "low"}


def test_sonnet_default_effort_sends_no_output_config(monkeypatch):
    from agentic_rag.config.settings import make_lm

    monkeypatch.setattr(settings.model, "reasoning_effort", "default")

    lm = make_lm("anthropic/claude-sonnet-5-5")
    assert "output_config" not in lm.kwargs
    assert "reasoning_effort" not in lm.kwargs
    assert "output_config" not in sent_params(lm.model, lm.kwargs)


def test_gpt5_mini_parameters_unchanged_by_anthropic_routing():
    from agentic_rag.config.settings import make_lm

    lm = make_lm("azure/gpt-5-mini")
    assert "output_config" not in lm.kwargs
    assert lm.kwargs["reasoning_effort"] == settings.model.reasoning_effort
    # dspy stores a reasoning model's budget under max_completion_tokens.
    assert lm.kwargs["max_completion_tokens"] == 16000


def test_gemini_keeps_temperature():
    lm = dspy.LM("gemini/gemini-3.1-flash-lite-preview", temperature=0.0, max_tokens=4096)
    sent = sent_params(lm.model, lm.kwargs)
    assert sent["temperature"] == 0.0
    assert sent["max_output_tokens"] == 4096


def test_credentials_never_reach_the_manifest(monkeypatch):
    monkeypatch.setattr(settings.model, "generate_model", "ollama_cloud/gpt-oss:120b")
    rec = manifest_mod.model_slot_record("generate")
    dumped = rec.model_dump_json()
    assert "api_key" not in dumped
    assert "api_base" not in dumped
    assert rec.resolved == "openai/gpt-oss:120b"
    # Ollama's OpenAI-compatible route does not carry reasoning_effort.
    assert "reasoning_effort" not in rec.sent_params


# ---------------------------------------------------------------------------
# Build / persist
# ---------------------------------------------------------------------------
def test_manifest_records_environment_and_settings():
    m = _build()
    assert m.git["commit"] and len(m.git["commit"]) == 40
    assert m.packages["dspy"] and m.packages["litellm"]
    assert m.process_defaults["retrieval"]["max_passages"] == settings.retrieval.max_passages
    assert m.process_defaults["agent"]["enabled_tools"] == settings.agent.enabled_tools
    assert m.seed == settings.experiment.seed
    assert [r.slot for r in m.models] == ["preprocess", "evaluate", "generate", "agent"]
    assert m.preflight["status"] == "pending"


def test_manifest_no_longer_writes_defaults_as_run_settings():
    """Version 1 stored the process defaults under names that read as the run's
    settings (agent.max_iterations 5 where every agentic variant ran 10)."""
    m = _build()
    assert m.manifest_version == 2
    assert m.model_defaults is m.retrieval is m.evaluation is m.agent is None
    assert set(m.process_defaults) == {"model", "retrieval", "evaluation", "agent"}


def test_version1_manifest_still_loads(tmp_path):
    """Old run directories are kept as recorded and must stay readable."""
    data = json.loads(_build().model_dump_json())
    for key in ("process_defaults", "settings_by_variant"):
        data.pop(key)
    data.update(
        manifest_version=1,
        model_defaults={"temperature": 0.0},
        retrieval={"top_k": 50},
        evaluation={"quality_threshold": 40},
        agent={"enabled_tools": None, "max_iterations": 5},
    )
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(data))
    loaded = RunManifest.load(tmp_path)
    assert loaded.manifest_version == 1
    assert loaded.agent == {"enabled_tools": None, "max_iterations": 5}
    assert loaded.settings_by_variant == {}


def test_manifest_roundtrip(tmp_path):
    m = _build()
    path = m.save(tmp_path)
    assert path.name == MANIFEST_FILENAME
    loaded = RunManifest.load(tmp_path)
    assert loaded == m
    assert json.loads(path.read_text())["run_id"] == m.run_id


def test_git_state_reports_this_repository():
    state = git_state()
    assert state["commit"]
    assert isinstance(state["dirty"], bool)
    assert all(not f.startswith(" ") for f in state["modified_files"])


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def _fake_observer(answer: str | None):
    def _observe(spec, run_id, meter):
        return answer

    return _observe


def test_preflight_aborts_on_snapshot_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(settings.model, "generate_model", "azure/gpt-5-mini")
    _pin(monkeypatch, f"azure/gpt-5-mini={SNAPSHOT}")
    monkeypatch.setattr(manifest_mod, "_observe_response_model", _fake_observer("gpt-5-mini-2099"))
    m = _build()

    with pytest.raises(ManifestMismatchError, match="gpt-5-mini-2099"):
        m.run_preflight(tmp_path, meter=object())

    # The evidence is on disk even though the run never started.
    on_disk = RunManifest.load(tmp_path)
    assert on_disk.preflight["status"] == "mismatch"
    assert any("generate" in s for s in on_disk.preflight["mismatches"])


def test_preflight_passes_on_declared_snapshot(monkeypatch, tmp_path):
    for slot in ("preprocess", "evaluate", "generate", "agent"):
        monkeypatch.setattr(settings.model, f"{slot}_model", "azure/gpt-5-mini")
    _pin(monkeypatch, f"azure/gpt-5-mini={SNAPSHOT}")
    calls: list[str] = []

    def _observe(spec, run_id, meter):
        calls.append(spec)
        return SNAPSHOT

    monkeypatch.setattr(manifest_mod, "_observe_response_model", _observe)
    m = _build()
    m.run_preflight(tmp_path, meter=object())

    assert m.preflight["status"] == "ok"
    assert m.preflight["unpinned"] == []
    assert all(r.observed_response_model == SNAPSHOT for r in m.models)
    assert calls == ["azure/gpt-5-mini"], "one probe per distinct model, not per slot"


def test_preflight_records_but_does_not_abort_when_unpinned(monkeypatch, tmp_path):
    _pin(monkeypatch, "")
    monkeypatch.setattr(manifest_mod, "_observe_response_model", _fake_observer("whatever"))
    m = _build()
    m.run_preflight(tmp_path, meter=object())
    assert m.preflight["status"] == "ok"
    assert m.preflight["unpinned"] == sorted({r.spec for r in m.models})


# ---------------------------------------------------------------------------
# Retries: Anthropic models do not retry a rejected request
# ---------------------------------------------------------------------------
def test_400_retry_off_only_for_anthropic(monkeypatch):
    from agentic_rag.config.settings import make_lm

    monkeypatch.setattr(settings.model, "num_retries", 12)
    gpt = make_lm("azure/gpt-5-mini")
    assert type(gpt) is dspy.LM
    assert gpt.num_retries == 12, "gpt-5-mini keeps litellm's own retries"

    sonnet = make_lm("anthropic/claude-sonnet-5-5")
    assert isinstance(sonnet, dspy.LM)
    assert sonnet.num_retries == 0, "litellm must not retry for Anthropic"
    assert sonnet.transient_retries == 12


def _count_forward(monkeypatch, exc: Exception) -> list[int]:
    calls: list[int] = []

    def fake_forward(self, prompt=None, messages=None, **kwargs):
        calls.append(1)
        raise exc

    monkeypatch.setattr(dspy.LM, "forward", fake_forward)
    return calls


def test_anthropic_bad_request_is_not_retried(monkeypatch):
    import litellm

    from agentic_rag.config.settings import make_lm

    calls = _count_forward(
        monkeypatch,
        litellm.BadRequestError("tool_choice", model="claude-sonnet-5-5", llm_provider="anthropic"),
    )
    lm = make_lm("anthropic/claude-sonnet-5-5", num_retries=12)
    with pytest.raises(litellm.BadRequestError):
        lm.forward(prompt="hi")
    assert len(calls) == 1


def test_anthropic_transient_error_is_retried(monkeypatch):
    import litellm

    from agentic_rag.config.settings import make_lm

    calls = _count_forward(
        monkeypatch,
        litellm.InternalServerError(
            "overloaded", model="claude-sonnet-5-5", llm_provider="anthropic"
        ),
    )
    # One retry: litellm's schedule sends the first retry without waiting.
    lm = make_lm("anthropic/claude-sonnet-5-5", num_retries=1)
    with pytest.raises(litellm.InternalServerError):
        lm.forward(prompt="hi")
    assert len(calls) == 2


def test_preflight_records_anthropic_created_at(monkeypatch, tmp_path):
    monkeypatch.setattr(manifest_mod, "_observe_response_model", lambda spec, run_id, meter: spec)
    seen: list[str] = []

    def fake_created_at(spec):
        seen.append(spec)
        return "2026-09-28T00:00:00Z"

    monkeypatch.setattr(manifest_mod, "_anthropic_model_created_at", fake_created_at)
    m = _build()
    m.models = [
        manifest_mod.ModelSlotRecord(slot=s, spec="anthropic/claude-sonnet-5-5", resolved="x")
        for s in ("preprocess", "generate")
    ] + [manifest_mod.ModelSlotRecord(slot="judge", spec="azure/gpt-5-mini", resolved="y")]
    m.run_preflight(tmp_path, meter=LiteLLMMeter())
    assert [r.provider_model_created_at for r in m.models] == [
        "2026-09-28T00:00:00Z",
        "2026-09-28T00:00:00Z",
        None,
    ]
    assert seen == ["anthropic/claude-sonnet-5-5"], "one lookup per model"


def test_created_at_lookup_never_raises(monkeypatch):
    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", boom)
    assert manifest_mod._anthropic_model_created_at("anthropic/claude-sonnet-5-5") is None


# ---------------------------------------------------------------------------
# Anthropic: prices, key slot, observed request
# ---------------------------------------------------------------------------
def test_sonnet_price_is_registered_without_the_remote_price_list(monkeypatch):
    import litellm

    from agentic_rag.config.settings import make_lm

    monkeypatch.delitem(litellm.model_cost, "claude-sonnet-5-5", raising=False)
    monkeypatch.delitem(litellm.model_cost, "anthropic/claude-sonnet-5-5", raising=False)
    make_lm("anthropic/claude-sonnet-5-5")
    resp = litellm.ModelResponse(
        model="claude-sonnet-5-5",
        usage={"prompt_tokens": 1000, "completion_tokens": 1000, "total_tokens": 2000},
    )
    cost = litellm.completion_cost(completion_response=resp, model="anthropic/claude-sonnet-5-5")
    assert cost == pytest.approx(1000 * 2e-06 + 1000 * 1e-05)


def test_manifest_records_unit_prices_and_key_slot_not_key(monkeypatch):
    monkeypatch.setattr(settings.model, "generate_model", "anthropic/claude-sonnet-5-5")
    rec = manifest_mod.model_slot_record("generate")
    assert rec.unit_prices["input_cost_per_token"] == 2e-06
    assert rec.unit_prices["output_cost_per_token"] == 1e-05
    assert rec.anthropic_key_slot == 1
    assert "sk-test-not-a-key" not in rec.model_dump_json()


def test_anthropic_model_without_key_slot_does_not_start(monkeypatch):
    from agentic_rag.config.settings import make_lm

    monkeypatch.setattr(settings.model, "anthropic_key_slot", None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-the-environment")
    with pytest.raises(ValueError, match="ANTHROPIC_KEY_SLOT"):
        make_lm("anthropic/claude-sonnet-5-5")


def test_anthropic_key_slot_without_a_key_does_not_start(monkeypatch):
    from agentic_rag.config.settings import make_lm

    monkeypatch.setattr(settings.model, "anthropic_key_slot", 2)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY_2"):
        make_lm("anthropic/claude-sonnet-5-5")


def test_anthropic_key_slot_is_exported_for_litellm():
    import os

    from agentic_rag.config.settings import make_lm

    make_lm("anthropic/claude-sonnet-5-5")
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-test-not-a-key"


def test_key_slot_is_not_needed_for_other_providers(monkeypatch):
    from agentic_rag.config.settings import make_lm

    monkeypatch.setattr(settings.model, "anthropic_key_slot", None)
    make_lm("azure/gpt-5-mini")


def test_preflight_records_the_request_it_sent(monkeypatch, tmp_path):
    meter = LiteLLMMeter()

    def observe(spec, run_id, meter_):
        kwargs = {
            "litellm_call_id": "pf",
            "additional_args": {
                "complete_input_dict": {
                    "model": "claude-sonnet-5-5",
                    "max_tokens": 32000,
                    "output_config": {"effort": "low"},
                    "messages": [],
                },
                "headers": {"x-api-key": "sk-test-not-a-key"},
            },
        }
        meter_.log_pre_api_call(spec, [], kwargs)
        from datetime import datetime
        from types import SimpleNamespace

        resp = SimpleNamespace(model="claude-sonnet-5-5", usage=None, choices=[])
        meter_.log_success_event(kwargs, resp, datetime.now(), datetime.now())
        return "claude-sonnet-5-5"

    monkeypatch.setattr(manifest_mod, "_observe_response_model", observe)
    monkeypatch.setattr(manifest_mod, "_anthropic_model_created_at", lambda spec: None)
    m = _build()
    m.models = [
        manifest_mod.ModelSlotRecord(
            slot="generate", spec="anthropic/claude-sonnet-5-5", resolved="x"
        )
    ]
    m.run_preflight(tmp_path, meter=meter)
    assert m.models[0].observed_request == {
        "max_tokens": 32000,
        "output_config": {"effort": "low"},
        "sent": [],
        "structured": "",
    }
    assert "sk-test-not-a-key" not in (tmp_path / MANIFEST_FILENAME).read_text()


def test_sonnet_capabilities_do_not_depend_on_the_remote_list(monkeypatch):
    """DSPy's fallback path reads supports_response_schema; it must not vary by network."""
    import litellm

    from agentic_rag.config.settings import make_lm

    monkeypatch.delitem(litellm.model_cost, "claude-sonnet-5-5", raising=False)
    monkeypatch.delitem(litellm.model_cost, "anthropic/claude-sonnet-5-5", raising=False)
    make_lm("anthropic/claude-sonnet-5-5")
    assert litellm.supports_response_schema(model="anthropic/claude-sonnet-5-5")


# ---------------------------------------------------------------------------
# Campaign phase and created_at at both ends of the run
# ---------------------------------------------------------------------------
def test_campaign_phase_is_recorded_before_the_run(monkeypatch):
    monkeypatch.setattr(settings.experiment, "campaign_phase", "smoke")
    assert _build().campaign_phase == "smoke"


def test_campaign_phase_rejects_unknown_values(monkeypatch):
    from agentic_rag.config.settings import ExperimentSettings

    monkeypatch.setenv("CAMPAIGN_PHASE", "final")
    with pytest.raises(ValueError):
        ExperimentSettings()


def test_finish_run_records_created_at_again(monkeypatch, tmp_path):
    monkeypatch.setattr(manifest_mod, "_anthropic_model_created_at", lambda spec: "2026-09-28")
    m = _build()
    m.models = [
        manifest_mod.ModelSlotRecord(
            slot="generate", spec="anthropic/claude-sonnet-5-5", resolved="x"
        ),
        manifest_mod.ModelSlotRecord(slot="judge", spec="azure/gpt-5-mini", resolved="y"),
    ]
    m.finish_run(tmp_path, None)
    assert [r.provider_model_created_at_end for r in m.models] == ["2026-09-28", None]


def test_anthropic_run_without_campaign_phase_does_not_start(monkeypatch):
    monkeypatch.setattr(settings.model, "generate_model", "anthropic/claude-sonnet-5-5")
    monkeypatch.setattr(settings.experiment, "campaign_phase", None)
    with pytest.raises(ValueError, match="campaign phase"):
        _build()
    monkeypatch.setattr(settings.experiment, "campaign_phase", "smoke")
    assert _build().campaign_phase == "smoke"


def test_other_runs_start_without_campaign_phase(monkeypatch):
    monkeypatch.setattr(settings.experiment, "campaign_phase", None)
    for slot in ("preprocess", "evaluate", "generate", "agent"):
        monkeypatch.setattr(settings.model, f"{slot}_model", "azure/gpt-5-mini")
    assert _build().campaign_phase is None


def test_manifest_records_the_question_set():
    qs = {"file": "/x/hardest.json", "file_sha256": "f" * 64, "ids": ["a"], "ids_sha256": "e" * 64}
    assert _build(question_set=qs).question_set == qs
    assert _build().question_set is None
