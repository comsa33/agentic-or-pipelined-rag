"""The refusal pattern set only grows, and the recount reports what grew.

Every refusal number reported before 2026-10-23 used the legacy set. Adding
Sonnet's surface forms must not drop or alter any legacy pattern, or the
earlier numbers could no longer be reproduced from the code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.analysis.refusal_analysis import (
    HEDGE_REFUSAL_PATTERNS,
    LEGACY_REFUSAL_PATTERNS,
    REFUSAL_PATTERNS,
    SONNET_REFUSAL_PATTERNS,
    is_refusal,
)
from experiments.analysis.refusal_pattern_recount import main, recount

# REFUSAL_PATTERNS before the Sonnet forms were added (the legacy set).
LEGACY_FROZEN = (
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


# Sonnet surface forms, added 2026-10-23.
SONNET_FROZEN = (
    r"do(?:es)? not say",
    "don[\u2019']?t say",
    r"do(?:es)? not state",
    "don[\u2019']?t give this",
    r"do(?:es)? not give enough information",
    r"cannot be computed from the passages",
)

# Generic hedge forms, added once on 2026-10-25; the set is frozen after this.
HEDGE_FROZEN = (
    r"passages do not (?:\w+ly )?(?:give|establish|confirm|link|explain)",
    r"not in the passages",
    r"outside knowledge",
    r"\bgive no\b",
)


def test_legacy_patterns_are_unchanged():
    assert LEGACY_REFUSAL_PATTERNS == LEGACY_FROZEN


def test_legacy_patterns_are_a_subset_of_the_current_set():
    assert set(LEGACY_FROZEN) <= set(REFUSAL_PATTERNS)
    assert REFUSAL_PATTERNS[: len(LEGACY_FROZEN)] == LEGACY_FROZEN
    assert len(REFUSAL_PATTERNS) == (
        len(LEGACY_FROZEN) + len(SONNET_REFUSAL_PATTERNS) + len(HEDGE_REFUSAL_PATTERNS)
    )


def test_the_pattern_set_is_frozen():
    """Each addition is appended in order; nothing after 2026-10-25."""
    assert SONNET_REFUSAL_PATTERNS == SONNET_FROZEN
    assert HEDGE_REFUSAL_PATTERNS == HEDGE_FROZEN
    assert REFUSAL_PATTERNS == LEGACY_FROZEN + SONNET_FROZEN + HEDGE_FROZEN


# The nine hedge forms listed on 2026-10-23, each in
# the sentence frame the smoke answers used.
@pytest.mark.parametrize(
    "prediction",
    [
        "Both directors are named, but the passages give no birth dates.",
        "The passages do not give either birth year.",
        "DepEd, though the passages do not clearly establish the boundary link.",
        "This is the best match, but the passages do not confirm the boundary country.",
        "The passages do not establish the Laos link.",
        "Note that the passages do not link the Philippines to Laos.",
        "The passages do not explain this for the Mekong.",
        "1993-2003, because the passages do not give it.",
        "This date range is not in the passages; it comes from outside knowledge.",
    ],
)
def test_c4_hedge_forms_are_detected(prediction):
    assert is_refusal(prediction)


@pytest.mark.parametrize(
    "prediction",
    [
        "Christopher Nolan",
        "The Department of Education",
        "The passages establish that he was born in 1916.",
        "The passages give the date as 1993.",
        "They forgive nothing and give nothing back.",
        "Not in the 1990s, but in 2003.",
    ],
)
def test_c4_patterns_leave_plain_answers_alone(prediction):
    assert not is_refusal(prediction)


def _run(tmp_path: Path, name: str, rows: dict[str, list[dict]]) -> Path:
    run = tmp_path / name
    run.mkdir()
    for pipeline, items in rows.items():
        (run / f"rq1_{pipeline}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in items), encoding="utf-8"
        )
    (run / "rq1_naive_rag_judged_x.jsonl").write_text('{"id": "ignored"}\n')
    return run


def test_recount_by_dataset_pipeline_k_and_changed_rows(tmp_path, capsys):
    k1 = _run(
        tmp_path,
        "20261021_000000_rq1_hotpotqa_n3_gpt-5-mini_k1",
        {
            "naive_rag": [
                {"id": "a", "prediction": "Cannot determine from the passages."},
                {"id": "b", "prediction": "No. But the passages do not state how many."},
                {"id": "c", "prediction": "Paris"},
                {"id": "d", "error": "boom", "prediction": "do not say"},
            ],
        },
    )
    k2 = _run(
        tmp_path,
        "20261022_000000_rq1_hotpotqa_n3_gpt-5-mini_k2",
        {"single-pass": [{"id": "b", "prediction": "They don't say."}]},
    )
    counts, changed = recount([k1, k2])
    assert counts[("hotpotqa", "naive_rag", "k1")] == [3, 1, 2]
    assert counts[("hotpotqa", "single-pass", "k2")] == [1, 0, 1]
    assert [(c.id, c.k, c.match) for c in changed] == [
        ("b", "k1", "do not state"),
        ("b", "k2", "don't say"),
    ]

    assert main([str(k1), str(k2), "--csv", str(tmp_path / "out.csv")]) == 0
    out = capsys.readouterr().out
    assert "Rows whose verdict differs: 2 over 1 question(s)" in out
    assert (tmp_path / "out.csv").read_text().count("\n") == 3


def test_refusal_table_reads_none_as_empty_after_exclusions(tmp_path):
    """Rows saved by earlier code versions carry None; str() would score it as "None".

    Excluded questions leave in the loader, before any refusal is counted.
    """
    from experiments.analysis.refusal_analysis import RefusalAnalyzer

    run = tmp_path / "s_rq1_hotpotqa_n3_claude-sonnet-5-5_k1"
    run.mkdir()
    rows = [
        {"id": "q0", "prediction": "Cannot determine.", "reference": "none"},
        {"id": "q1", "prediction": None, "reference": "none", "refusal_seen": True},
        {"id": "q2", "prediction": None, "reference": "none"},
    ]
    (run / "rq1_naive_rag.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "exclusions.json").write_text(
        json.dumps({"hotpotqa": {"claude-sonnet-5-5": [{"id": "q1", "reason": "refused"}]}})
    )
    (rec,) = RefusalAnalyzer.from_results_dir(tmp_path).compute()
    assert rec["total"] == 2, "the excluded question is gone before counting"
    assert rec["refusals"] == 1
    assert rec["f1_non_refused"] == 0.0, "None scored as the text 'None' would match"
