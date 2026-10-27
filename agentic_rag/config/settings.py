"""Global experiment settings for Self-Corrective RAG.

All hyperparameters and configuration values are centralized here
for reproducibility and easy ablation study control.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indices"
RESULTS_DIR = DATA_DIR / "results"

# Model-name prefix routing a request to Ollama Cloud (see make_lm).
OLLAMA_CLOUD_PREFIX = "ollama_cloud/"

# Floor on max_tokens for reasoning models, which spend part of the budget on
# hidden reasoning. DSPy enforces this same floor for the gpt-5 family.
REASONING_MIN_MAX_TOKENS = 16000

# Anthropic models reached through litellm ("anthropic/<model>"). The Claude
# 5.x models think adaptively and cannot turn thinking off, so the same budget
# trap applies, with two differences from the OpenAI reasoning models: litellm
# translates `reasoning_effort` into a thinking budget these models reject
# (400), and they reject `temperature` outright. Effort is therefore sent the
# way the Anthropic API defines it, in `output_config`. The ceiling is set
# higher than the OpenAI floor because a hard multi-step prompt measured
# 11,433 output tokens at effort "high"; max_tokens is a ceiling, so the
# headroom costs nothing unless it is used.
ANTHROPIC_PREFIX = "anthropic/"
ANTHROPIC_MIN_MAX_TOKENS = 32000

# USD per token for Anthropic models, registered with litellm so cost does
# not depend on its remote price list: the price table bundled with litellm
# 1.82.2 has no Claude 5.5 entry, and when the remote list cannot be fetched
# every call would be recorded with cost None. Source:
# https://platform.claude.com/docs/en/about-claude/pricing (checked
# 2026-10-21): input $2, output $10, 5-minute cache write $2.50, 1-hour cache
# write $4, cache hit $0.10 per MTok; no long-context surcharge. Matches the
# remote litellm list on the same date.
ANTHROPIC_PRICES = {
    "claude-sonnet-5-5": {
        "input_cost_per_token": 2e-06,
        "output_cost_per_token": 1e-05,
        "cache_creation_input_token_cost": 2.5e-06,
        "cache_creation_input_token_cost_above_1hr": 4e-06,
        "cache_read_input_token_cost": 1e-07,
    },
}

# Capability flags DSPy reads from litellm's model list, registered with the
# prices for the same reason. Without an entry `supports_response_schema` is
# False and DSPy's JSONAdapter skips its structured-output attempt (a forced
# tool call for Anthropic, which Claude 5.x rejects with a 400) and goes
# straight to JSON mode; with the remote list it makes that attempt first.
# The values are the remote list's on 2026-10-21, so the fallback path no
# longer depends on whether the list could be fetched.
ANTHROPIC_CAPABILITIES = {
    "claude-sonnet-5-5": {
        "supports_response_schema": True,
        "supports_function_calling": True,
        "supports_tool_choice": True,
        "supports_reasoning": True,
    },
}

# Substrings identifying a reasoning model. Missing an entry fails silently and
# expensively: the model draws answer and hidden reasoning from one budget, so
# it truncates mid-answer while the run still reports success. glm-5.2 was run
# for a full dataset at max_tokens=4096 before the truncation warnings were
# noticed, and the damage fell unevenly across pipelines. deepseek-v4-flash then
# repeated it as a judge, truncating 2 of its first 14 verdicts.
#
# Prefer listing a model that turns out not to need it: max_tokens is a ceiling,
# so raising it for an ordinary model changes nothing, while leaving a reasoning
# model out corrupts the run. Because a model is selected by environment
# variable, LLM_REASONING_MODELS overrides this list so that adding one never
# requires editing code here.
REASONING_MODEL_MARKERS = ("gpt-5", "gpt-oss", "glm", "deepseek")


class ModelSettings(BaseSettings):
    """LLM and embedding model configuration."""

    model_config = SettingsConfigDict(
        env_prefix="", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM models
    preprocess_model: str = Field("gpt-4o-mini", alias="PREPROCESS_MODEL")
    evaluate_model: str = Field("gpt-4o-mini", alias="EVALUATE_MODEL")
    generate_model: str = Field("gpt-4o", alias="GENERATE_MODEL")
    agent_model: str = Field("gpt-4o", alias="AGENT_MODEL")

    # Embedding
    embedding_model: str = Field("text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dimension: int = 1536

    # Anthropic API key, chosen by slot: ANTHROPIC_KEY_SLOT=<n> reads
    # ANTHROPIC_API_KEY_<n> from the keys file, so that a run states which
    # key (and budget) it spent without the key itself being recorded. An
    # Anthropic model does not start without a slot.
    anthropic_key_slot: int | None = Field(None, alias="ANTHROPIC_KEY_SLOT")
    anthropic_keys_file: Path = Field(PROJECT_ROOT / ".env.anthropic", alias="ANTHROPIC_KEYS_FILE")

    # LLM parameters
    temperature: float = 0.0
    max_tokens: int = Field(4096, alias="LLM_MAX_TOKENS")
    # Reasoning budget for reasoning-capable models. "default" leaves the
    # provider's own setting alone, which some models need in order to emit a
    # complete structured response.
    reasoning_effort: str = Field("low", alias="LLM_REASONING_EFFORT")
    num_retries: int = Field(3, alias="LLM_NUM_RETRIES")
    # Comma-separated substrings replacing REASONING_MODEL_MARKERS. Lets a new
    # reasoning model be added alongside the *_MODEL variables that select it,
    # instead of in code that is easy to forget.
    reasoning_models: str = Field("", alias="LLM_REASONING_MODELS")
    # Snapshot each model string is expected to answer with, as reported in
    # `response.model`, e.g. "azure/gpt-5-mini=gpt-5-mini-2025-08-07". A run
    # whose first response names a different snapshot is aborted before any
    # question is asked, because its numbers could not be placed in a table
    # next to runs made against the declared one.
    expected_response_models: str = Field("", alias="LLM_EXPECTED_RESPONSE_MODELS")

    def expected_response_model(self, model: str) -> str | None:
        """Declared snapshot for `model`, or None when the model is unpinned."""
        for entry in self.expected_response_models.split(","):
            spec, sep, snapshot = entry.strip().partition("=")
            if sep and spec.strip() == model and snapshot.strip():
                return snapshot.strip()
        return None

    def is_reasoning_model(self, model: str) -> bool:
        """Whether `model` draws its answer and hidden reasoning from one budget."""
        markers = (
            tuple(m.strip().lower() for m in self.reasoning_models.split(",") if m.strip())
            or REASONING_MODEL_MARKERS
        )
        model_lower = model.lower()
        return any(marker in model_lower for marker in markers)


class RetrievalSettings(BaseSettings):
    """Retrieval hyperparameters."""

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")

    # Search parameters
    top_k: int = 50
    text_top_k: int = 40
    query_method: str = "rrf"  # rrf | vector_only | text_only | combined
    hybrid_weight: float = 0.48  # vector search weight in RRF

    # Passage accumulation
    max_passages: int = 30  # FIFO eviction above this
    # Apply max_passages to the single-shot baselines (naive, CRAG-style)
    # too. Off by default: the March 2026 runs handed those pipelines all
    # top_k=50 passages while the accumulating ones were capped at 30, and
    # that difference is preserved as the "uncontrolled" condition. On, every
    # pipeline sees the same M and the comparison is controlled for context
    # budget (see BasePipeline.passage_cap).
    max_passages_all_pipelines: bool = False


class EvaluationSettings(BaseSettings):
    """4D quality evaluation hyperparameters."""

    model_config = SettingsConfigDict(env_prefix="EVAL_")

    # Quality threshold
    quality_threshold: int = 40  # score >= threshold → "output"

    # Max retry
    max_retry_count: int = 3

    # 4D score ranges (for reference / validation)
    relevance_max: int = 30
    coverage_max: int = 25
    specificity_max: int = 25
    sufficiency_max: int = 20
    total_max: int = 100


class ExperimentSettings(BaseSettings):
    """Experiment control parameters."""

    model_config = SettingsConfigDict(env_prefix="EXPERIMENT_")

    seed: int = Field(42, alias="EXPERIMENT_SEED")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    # Which phase of the campaign a run belongs to, fixed before it starts
    # and written to its manifest, so that verification reads it from the
    # record instead of having it chosen after the results are seen. Set by
    # run.py --campaign-phase or CAMPAIGN_PHASE.
    campaign_phase: Literal["smoke", "pilot", "grid"] | None = Field(None, alias="CAMPAIGN_PHASE")

    # Datasets
    datasets: list[str] = ["hotpotqa", "2wikimultihopqa", "musique", "financebench"]

    # Ablation flags — toggle individual pipeline components on/off
    enable_iteration: bool = True  # iterative loop
    enable_accumulation: bool = True  # passage accumulation
    enable_4d_evaluation: bool = True  # 4D quality assessment
    enable_refinement: bool = True  # targeted query refinement
    enable_agent_routing: bool = True  # 3-way agent routing
    enable_dspy: bool = True  # DSPy pipeline (vs manual prompt)
    enable_agentic_refinement: bool = False  # ReAct-based agentic refinement (opt-in)

    # LLM-call budget per question, for budget-matched comparisons. None keeps
    # each pipeline's own stopping rule. With a budget the loop pipeline keeps
    # refining until the budget is spent instead of stopping on a score, and
    # the agentic pipeline caps its ReAct iterations (best effort, since tool
    # calls that use the LLM are decided by the agent). See
    # LoopRAGPipeline.loop_retry_budget / AgenticRAGPipeline.effective_max_iters.
    llm_call_budget: int | None = Field(None, alias="EXPERIMENT_LLM_CALL_BUDGET")


class AgentSettings(BaseSettings):
    """ReAct agent hyperparameters for agentic retrieval refinement."""

    model_config = SettingsConfigDict(env_prefix="AGENT_")

    max_iterations: int = 5  # ReAct reasoning-action iterations

    # Tool-level ablation: None = all tools enabled
    # Valid names (7): search, decompose, structure, terminology, evaluate, inspect, calculate
    enabled_tools: list[str] | None = None


class IRCoTSettings(BaseSettings):
    """IRCoT (Trivedi et al., 2023) parameters; see pipeline/ircot.py for the paper's."""

    model_config = SettingsConfigDict(env_prefix="IRCOT_")

    max_steps: int = 8  # paper: 8 (footnote 4)
    per_step_k: int = 8  # paper: tuned per dataset from {2, 4, 6, 8}


class Settings(BaseSettings):
    """Root settings aggregating all sub-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    ollama_api_key: str = Field("", alias="OLLAMA_API_KEY")
    ollama_api_base: str = Field("https://ollama.com/v1", alias="OLLAMA_API_BASE")

    # Disable the LLM response cache. Required whenever latency is measured:
    # at temperature=0 a rerun returns cached completions in milliseconds,
    # which deflates latency without changing any answer.
    disable_llm_cache: bool = Field(False, alias="DISABLE_LLM_CACHE")

    # Sub-settings
    model: ModelSettings = Field(default_factory=ModelSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    experiment: ExperimentSettings = Field(default_factory=ExperimentSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    ircot: IRCoTSettings = Field(default_factory=IRCoTSettings)

    # Paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    raw_dir: Path = RAW_DIR
    processed_dir: Path = PROCESSED_DIR
    index_dir: Path = INDEX_DIR
    results_dir: Path = Field(RESULTS_DIR, alias="RESULTS_DIR")


# ---------------------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------------------
settings = Settings()


def make_lm(model: str, **kwargs):
    """Create a dspy.LM with default retry/temperature from settings.

    Centralizes LM creation so rate-limit retry settings are consistent.
    """
    import dspy
    import litellm

    # gpt-5 reasoning models don't support temperature; drop unsupported params
    litellm.drop_params = True

    # max_tokens must be passed explicitly. Without it the provider default
    # applies, and a provider with a small default truncates the response
    # mid-structure — which surfaces as unparsable tool calls rather than as
    # an obvious error.
    defaults = {
        "temperature": settings.model.temperature,
        "num_retries": settings.model.num_retries,
        "max_tokens": settings.model.max_tokens,
    }
    # Reasoning models: hold effort at "low" so every model is compared at a
    # similar compute budget. Left at its default, gpt-oss spends thousands of
    # reasoning tokens on a single preprocessing call and truncates mid-answer.
    if settings.model.is_reasoning_model(model):
        if "reasoning_effort" not in kwargs and settings.model.reasoning_effort != "default":
            defaults["reasoning_effort"] = settings.model.reasoning_effort
        # Reasoning tokens are drawn from the same budget as the answer, so a
        # limit sized for ordinary completions truncates mid-structure. DSPy
        # also rejects anything under 16000 for the gpt-5 family outright.
        defaults["max_tokens"] = max(settings.model.max_tokens, REASONING_MIN_MAX_TOKENS)

    if model.startswith(ANTHROPIC_PREFIX):
        _use_anthropic_key_slot()
        _register_anthropic_price(model)
        defaults.pop("temperature", None)
        defaults["max_tokens"] = max(settings.model.max_tokens, ANTHROPIC_MIN_MAX_TOKENS)
        if "output_config" not in kwargs and settings.model.reasoning_effort != "default":
            defaults["output_config"] = {"effort": settings.model.reasoning_effort}

    # Ollama Cloud is reached through its OpenAI-compatible endpoint so the
    # API key travels as a bearer token. Written as "ollama_cloud/<model>"
    # to stay distinct from litellm's own local-Ollama provider.
    if model.startswith(OLLAMA_CLOUD_PREFIX):
        defaults.setdefault("api_base", settings.ollama_api_base)
        defaults.setdefault("api_key", settings.ollama_api_key)
        model = f"openai/{model[len(OLLAMA_CLOUD_PREFIX) :]}"

    defaults.update(kwargs)
    if model.startswith(ANTHROPIC_PREFIX):
        retries = defaults.pop("num_retries", 0)
        return _anthropic_lm_class()(model, num_retries=0, transient_retries=retries, **defaults)
    return dspy.LM(model, **defaults)


def _use_anthropic_key_slot() -> None:
    """Export the slot's key as ANTHROPIC_API_KEY, where litellm reads it.

    Raises when no slot is set or the slot has no key, so an Anthropic run
    never falls back to whatever key the environment happens to hold.
    """
    import os

    from dotenv import dotenv_values

    slot = settings.model.anthropic_key_slot
    if slot is None:
        raise ValueError(
            "Anthropic model requested without ANTHROPIC_KEY_SLOT: set it to the "
            "ANTHROPIC_API_KEY_<n> slot this run may spend"
        )
    path = Path(settings.model.anthropic_keys_file)
    key = dotenv_values(path).get(f"ANTHROPIC_API_KEY_{slot}") if path.is_file() else None
    if not key:
        raise ValueError(f"No ANTHROPIC_API_KEY_{slot} in {path}")
    os.environ["ANTHROPIC_API_KEY"] = key


def _register_anthropic_price(model: str) -> None:
    import litellm

    name = model[len(ANTHROPIC_PREFIX) :]
    prices = ANTHROPIC_PRICES.get(name)
    if prices is None:
        return
    entry = {
        "litellm_provider": "anthropic",
        "mode": "chat",
        **prices,
        **ANTHROPIC_CAPABILITIES.get(name, {}),
    }
    litellm.register_model({name: entry, model: entry})


@functools.cache
def _anthropic_lm_class():
    """dspy.LM that retries transient errors but not a rejected request.

    litellm retries every APIError, a 400 included, and with exponential
    backoff a request the provider will always reject costs ~85 s at
    LLM_NUM_RETRIES=12. DSPy's JSONAdapter fallback sends such a request to
    Claude 5.x (a forced tool_choice, which thinking models reject), so the
    fallback alone would add that delay to a question's latency. Retrying
    here, with litellm's own retries off, keeps every other error on
    litellm's schedule: first retry at once, then 1, 2, 4, 8 and 10 s.
    BadRequestError covers its subclasses (content policy, context window).
    Only `forward` is overridden; nothing in this codebase calls the async
    path.

    A refusal in a category of RETRIED_REFUSAL_CATEGORIES is answered by
    sending the same request again, at once, up to REFUSAL_RETRIES times. It
    arrives as HTTP 200 with stop_reason "refusal", which litellm returns as
    an ordinary "stop", so the decision reads the raw body through
    RawStopWatcher. Resends skip DSPy's cache, which would otherwise serve
    the refusal back. Whatever the last send returned is returned, refused
    or not; the meter records each send (see cost_tracker.empty_usage).
    """
    import dspy
    import litellm
    import tenacity

    from agentic_rag.evaluation.cost_tracker import (
        install_raw_stop_watcher,
        last_refusal_category,
        refusal_retry,
    )
    from experiments.exclusions import REFUSAL_RETRIES, RETRIED_REFUSAL_CATEGORIES

    class AnthropicLM(dspy.LM):
        def __init__(self, *args, transient_retries: int = 0, **kwargs):
            super().__init__(*args, **kwargs)
            self.transient_retries = transient_retries

        def _send(self, prompt, messages, **kwargs):
            retryer = tenacity.Retrying(
                retry=tenacity.retry_if_not_exception_type(litellm.BadRequestError),
                wait=tenacity.wait_chain(*(tenacity.wait_fixed(s) for s in (0, 1, 2, 4, 8, 10))),
                stop=tenacity.stop_after_attempt(self.transient_retries + 1),
                reraise=True,
            )
            return retryer(super().forward, prompt, messages, **kwargs)

        def forward(self, prompt=None, messages=None, **kwargs):
            install_raw_stop_watcher()
            for resend in range(REFUSAL_RETRIES + 1):
                with refusal_retry(resend):
                    if resend:
                        result = self._send(prompt, messages, **{**kwargs, "cache": False})
                    else:
                        result = self._send(prompt, messages, **kwargs)
                    category = last_refusal_category()
                if category not in RETRIED_REFUSAL_CATEGORIES:
                    break
            return result

    return AnthropicLM
