"""Read `exclusions.json`: questions excluded from analysis, per dataset and model.

Two layouts are read. The original keys entries by dataset only; such an
entry has no model and is refused for a Claude model (see excluded_ids):

    {"_schema": {...}, "hotpotqa": [{"id": ..., "reason": ...}, ...]}

The model-dimension layout adds the model tag (as in run directory names,
e.g. "gpt-5-mini") under the dataset, so an entry applies to that model
only:

    {"_schema": {...}, "hotpotqa": {"gpt-5-mini": [{"id": ...}, ...]}}

A list entry may also name its model in a "model" field. Keys starting
with "_" are metadata and skipped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass(frozen=True)
class Exclusion:
    dataset: str
    id: str
    # Model tag the exclusion applies to; None means every model.
    model: str | None = None
    reason: str = ""
    pipeline: str = ""
    first_seen: str = ""
    note: str = ""


def _entry(dataset: str, model: str | None, raw: dict) -> Exclusion:
    return Exclusion(
        dataset=dataset,
        id=str(raw["id"]),
        model=raw.get("model", model),
        reason=str(raw.get("reason") or ""),
        pipeline=str(raw.get("pipeline") or ""),
        first_seen=str(raw.get("first_seen") or ""),
        note=str(raw.get("note") or ""),
    )


def load_exclusions(path: Path | str) -> list[Exclusion]:
    """Every entry in the file, in either layout."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[Exclusion] = []
    for dataset, value in data.items():
        if dataset.startswith("_"):
            continue
        if isinstance(value, list):
            out.extend(_entry(dataset, None, raw) for raw in value)
        elif isinstance(value, dict):
            for model, entries in value.items():
                if model.startswith("_"):
                    continue
                out.extend(_entry(dataset, model, raw) for raw in entries)
        else:
            raise ValueError(f"exclusions for {dataset!r} must be a list or a dict by model")
    return out


def excluded_ids(exclusions: list[Exclusion], dataset: str, model: str) -> set[str]:
    """Ids of `dataset` excluded for `model`, including entries for every model.

    An entry without a model was written before the file had a model
    dimension, when every exclusion came from gpt-5-mini runs. Applying such
    an entry to a Claude model would drop questions from its analysis that
    it never failed, so that lookup raises; for other models it is applied
    with a warning until the file is migrated.
    """
    unscoped = [e for e in exclusions if e.dataset == dataset and e.model is None]
    if unscoped:
        if model.startswith("claude"):
            raise ValueError(
                f"exclusions for {dataset!r} have no model; give them one before "
                f"looking up {model!r}"
            )
        logger.warning(
            f"exclusions for {dataset!r} have no model; applying {len(unscoped)} to {model!r}"
        )
    return {
        e.id for e in exclusions if e.dataset == dataset and (e.model is None or e.model == model)
    }


# ---------------------------------------------------------------------------
# Refusal policy and triggers. A question is excluded for a model when a refusal stood
# (main analysis); the sensitivity analysis excludes it on any first refusal,
# recovered by a resend or not. See cost_tracker.empty_usage for the fields.
# ---------------------------------------------------------------------------

# Refusal categories retried by sending the same request again, and how many
# times (make_lm's Anthropic LM). "reasoning_extraction" is a judgement on the
# output format the prompt asks for, not on the question, so a resend is a
# second draw of the same request; every other category is final on the
# first refusal, as fixed before the grid ran. Kept here, free of litellm, so analysis
# code can read the policy without importing the client stack.
RETRIED_REFUSAL_CATEGORIES = frozenset({"reasoning_extraction"})
REFUSAL_RETRIES = 3


def refusal_final(usage: dict | None) -> bool:
    """Whether a refusal in `usage` stood: not retried, or still refused at the last resend.

    Refusals in a retried category (RETRIED_REFUSAL_CATEGORIES)
    that a resend recovered do not count. Usage written before resends were
    metered has no refusals_final_by_category; every refusal there stood.
    """
    usage = usage or {}
    if "refusals_final_by_category" not in usage:
        return int(usage.get("refused_calls") or 0) > 0
    if sum(int(n) for n in (usage["refusals_final_by_category"] or {}).values()) > 0:
        return True
    return any(
        int(n) > 0 and category not in RETRIED_REFUSAL_CATEGORIES
        for category, n in (usage.get("refusal_categories") or {}).items()
    )


def row_refusal_final(row: dict) -> bool:
    """A result row's main-analysis trigger, earlier attempts at its item included."""
    return (
        bool(row.get("refusal_seen"))
        or refusal_final(row)
        or refusal_final(row.get("prior_attempt_usage"))
    )


def row_refusal_first(row: dict) -> bool:
    """A result row's sensitivity-analysis trigger: any refused call at its item.

    Resends follow only a refusal, so any refused call implies a refused
    first send.
    """
    prior = row.get("prior_attempt_usage") or {}
    return (
        bool(row.get("refusal_first_seen"))
        or int(row.get("refused_calls") or 0) > 0
        or int(prior.get("refused_calls") or 0) > 0
    )


def refusal_ids(rows: list[dict], *, first: bool = False) -> list[str]:
    """Ids of the questions whose rows trigger a refusal exclusion.

    first=False: a refusal stood (main analysis). first=True: any refusal,
    recovered or not (sensitivity analysis).
    """
    trigger = row_refusal_first if first else row_refusal_final
    return sorted({str(r.get("id")) for r in rows if trigger(r)})
