"""The verifier must reject every defect that has invalidated a run before.

Each test plants one defect in an otherwise valid run directory and expects
exactly that check to FAIL; the clean directory must pass with no FAIL.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.manifest import REQUIRED_ROW_FIELDS, RunManifest
from experiments.verify import FAIL, WARN, verify_judged, verify_repeat_set, verify_run_dir

SNAPSHOT = "gpt-5-mini-2025-08-07"
RUN_ID = "run"
COMMIT = "0" * 40


def _row(i: int, **overrides) -> dict:
    row = {f: None for f in REQUIRED_ROW_FIELDS}
    row.update(
        run_id=RUN_ID,
        repeat_index=None,
        git_commit=COMMIT,
        llm_cache_disabled=True,
        id=f"q{i}",
        question=f"question {i}",
        reference="a",
        all_references=["a"],
        prediction="a",
        pipeline="naive_rag",
        passages_used=50,
        total_passages_retrieved=50,
        llm_calls=1,
        latency_seconds=3.2,
        metered_calls=1,
        usage_complete=True,
        cost_usd=0.001,
        response_models=[SNAPSHOT],
    )
    row.update(overrides)
    return row


def _manifest(**overrides) -> dict:
    slot = {
        "slot": "generate",
        "spec": "azure/gpt-5-mini",
        "resolved": "azure/gpt-5-mini",
        "provider": "azure",
        "expected_response_model": SNAPSHOT,
        "observed_response_model": SNAPSHOT,
    }
    m = {
        "run_id": RUN_ID,
        "created_at": "t",
        "finished_at": "t",
        "command": ["run.py"],
        "experiment": "RQ1",
        "dataset": "2wikimultihopqa",
        "n": 2,
        "variants": ["Naive RAG"],
        "seed": 42,
        "git": {"commit": COMMIT, "branch": "main", "dirty": False, "modified_files": []},
        "packages": {"dspy": "3.1.3"},
        "cache": {"llm_cache_disabled": True},
        "model_defaults": {},
        "retrieval": {},
        "evaluation": {},
        "agent": {},
        "models": [{**slot, "slot": s} for s in ("preprocess", "evaluate", "generate", "agent")],
        "max_passages_by_pipeline": {"Naive RAG": None},
        "preflight": {"status": "ok", "mismatches": [], "unpinned": []},
        "observed_response_models": [SNAPSHOT],
    }
    m.update(overrides)
    return m


def _write_run(tmp_path: Path, rows: list[dict], manifest: dict | None, summary: dict | None):
    if manifest is not None:
        RunManifest.model_validate(manifest).save(tmp_path)
    (tmp_path / "rq1_naive_rag.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    if summary is not None:
        (tmp_path / "rq1_naive_rag_summary.json").write_text(json.dumps(summary))
    return tmp_path


def _summary(variant: str = "Naive RAG") -> dict:
    return {
        "extra": {"variant": variant},
        "metrics": {"f1": 0.5},
        "settings": {
            "max_passages": 30,
            "passage_cap": None,
            "enabled_tools": ["search"],
            "models": {},
            "seed": 42,
            "top_k": 50,
        },
    }


def _failed(checks) -> set[str]:
    return {c.name.split(": ")[-1] for c in checks if c.status == FAIL}


@pytest.fixture
def clean_run(tmp_path):
    return _write_run(tmp_path, [_row(0), _row(1)], _manifest(), _summary())


def test_clean_run_passes(clean_run):
    checks = verify_run_dir(clean_run)
    assert not _failed(checks), [c for c in checks if c.status == FAIL]


def test_missing_manifest_fails(tmp_path):
    run = _write_run(tmp_path, [_row(0), _row(1)], None, _summary())
    assert "manifest" in _failed(verify_run_dir(run, n=2))


@pytest.mark.parametrize(
    ("rows", "expected_check"),
    [
        ([_row(0)], "n"),
        ([_row(0), _row(0)], "unique ids"),
        (
            [_row(0), {"id": "q1", "question": "x", "error": "rate limit", "pipeline": "p"}],
            "errors",
        ),
        ([_row(0), _row(1, latency_seconds=0.01)], "latency < 1s share"),
        ([_row(0), _row(1, metered_calls=0)], "usage metered"),
        ([_row(0), _row(1, usage_complete=False)], "usage complete"),
        ([_row(0), _row(1, response_models=["gpt-5-mini-2099"])], "row snapshots"),
    ],
)
def test_row_defects_fail(tmp_path, rows, expected_check):
    run = _write_run(tmp_path, rows, _manifest(), _summary())
    assert expected_check in _failed(verify_run_dir(run))


def test_missing_required_field_fails(tmp_path):
    rows = [_row(0), _row(1)]
    del rows[1]["cost_usd"]
    run = _write_run(tmp_path, rows, _manifest(), _summary())
    checks = verify_run_dir(run)
    assert "required fields" in _failed(checks)
    assert any("cost_usd" in c.detail for c in checks if c.status == FAIL)


def test_snapshot_mismatch_in_manifest_fails(tmp_path):
    m = _manifest()
    m["models"][2]["observed_response_model"] = "gpt-5-mini-2099"
    run = _write_run(tmp_path, [_row(0), _row(1)], m, _summary())
    assert "snapshot generate" in _failed(verify_run_dir(run))


def test_stray_snapshot_during_run_fails(tmp_path):
    m = _manifest(observed_response_models=[SNAPSHOT, "gpt-5-mini-2099"])
    run = _write_run(tmp_path, [_row(0), _row(1)], m, _summary())
    assert "snapshot during run" in _failed(verify_run_dir(run))


def test_unpinned_snapshot_only_warns(tmp_path):
    m = _manifest()
    for slot in m["models"]:
        slot["expected_response_model"] = None
    m["preflight"]["unpinned"] = ["azure/gpt-5-mini"]
    run = _write_run(tmp_path, [_row(0), _row(1)], m, _summary())
    checks = verify_run_dir(run)
    assert not _failed(checks)
    assert any(c.name == "snapshot pinned" and c.status == WARN for c in checks)


def test_cache_on_fails(tmp_path):
    run = _write_run(
        tmp_path, [_row(0), _row(1)], _manifest(cache={"llm_cache_disabled": False}), _summary()
    )
    assert "cache disabled" in _failed(verify_run_dir(run))


def test_dirty_tree_fails_unless_allowed(tmp_path):
    m = _manifest(
        git={"commit": "0" * 40, "branch": "main", "dirty": True, "modified_files": ["x"]}
    )
    run = _write_run(tmp_path, [_row(0), _row(1)], m, _summary())
    assert "git clean" in _failed(verify_run_dir(run))
    assert "git clean" not in _failed(verify_run_dir(run, allow_dirty=True))


def test_unfinished_or_mismatched_preflight_fails(tmp_path):
    m = _manifest(finished_at=None, preflight={"status": "mismatch", "mismatches": ["x"]})
    run = _write_run(tmp_path, [_row(0), _row(1)], m, _summary())
    failed = _failed(verify_run_dir(run))
    assert {"preflight", "run finished"} <= failed


def test_summary_without_settings_fails(tmp_path):
    run = _write_run(tmp_path, [_row(0), _row(1)], _manifest(), {"metrics": {"f1": 1}})
    assert "summary" in _failed(verify_run_dir(run))


def test_variant_count_must_match_manifest(tmp_path):
    run = _write_run(tmp_path, [_row(0), _row(1)], _manifest(variants=["A", "B"]), _summary())
    assert "variants complete" in _failed(verify_run_dir(run))


# ---------------------------------------------------------------------------
# Resumed runs: rows written by an earlier process
# ---------------------------------------------------------------------------
def test_resumed_rows_from_same_commit_only_warn(tmp_path):
    rows = [_row(0, run_id="earlier_run"), _row(1)]
    run = _write_run(tmp_path, rows, _manifest(), _summary())
    checks = verify_run_dir(run)
    assert not _failed(checks)
    warned = [c for c in checks if c.status == WARN and c.name.endswith("resumed rows")]
    assert warned and "earlier_run" in warned[0].detail and "1 row(s)" in warned[0].detail


def test_resumed_rows_from_another_commit_fail(tmp_path):
    rows = [_row(0, run_id="earlier_run", git_commit="f" * 40), _row(1)]
    run = _write_run(tmp_path, rows, _manifest(), _summary())
    assert "resumed rows commit" in _failed(verify_run_dir(run))


def test_rows_without_provenance_fail(tmp_path):
    rows = [_row(0, run_id=None, git_commit=None), _row(1)]
    run = _write_run(tmp_path, rows, _manifest(), _summary())
    assert "provenance" in _failed(verify_run_dir(run))


def test_rows_made_with_cache_on_fail(tmp_path):
    rows = [_row(0, run_id="earlier_run", llm_cache_disabled=False), _row(1)]
    run = _write_run(tmp_path, rows, _manifest(), _summary())
    assert "row cache" in _failed(verify_run_dir(run))


def test_march_2026_paper_run_is_rejected():
    """The directories the paper was built from must not pass: that is the point."""
    paper = Path("data/results/paper")
    if not paper.exists():
        pytest.skip("paper results not on this machine")
    run = next(paper.glob("*_rq1_2wikimultihopqa_*"))
    assert "manifest" in _failed(verify_run_dir(run, n=200))


# ---------------------------------------------------------------------------
# Repeated runs checked as a set
# ---------------------------------------------------------------------------
def _repeat_dirs(tmp_path, ks, ns=None):
    dirs = []
    for i, k in enumerate(ks):
        n = ns[i] if ns else 2
        d = tmp_path / f"run_k{k}"
        d.mkdir()
        _write_run(
            d,
            [_row(0), _row(1)],
            _manifest(
                run_id=d.name,
                run_key=f"rq1|2wikimultihopqa|gpt-5-mini|n{n}|k{k}",
                repeat_index=k,
                n=n,
            ),
            _summary(),
        )
        dirs.append(d)
    return dirs


def _set_failed(checks) -> set[str]:
    return {c.name for c in checks if c.status == FAIL}


def test_complete_repeat_set_passes(tmp_path):
    assert not _set_failed(verify_repeat_set(_repeat_dirs(tmp_path, [1, 2, 3])))


def test_gap_in_repeat_indices_fails(tmp_path):
    assert "repeat index contiguous" in _set_failed(
        verify_repeat_set(_repeat_dirs(tmp_path, [1, 3]))
    )


def test_duplicate_repeat_index_fails(tmp_path):
    dirs = _repeat_dirs(tmp_path, [1, 2])
    # a second directory claiming k=2
    d = tmp_path / "run_k2_again"
    d.mkdir()
    _write_run(
        d,
        [_row(0), _row(1)],
        _manifest(run_id=d.name, run_key="rq1|2wikimultihopqa|gpt-5-mini|n2|k2", repeat_index=2),
        _summary(),
    )
    assert "repeat index unique" in _set_failed(verify_repeat_set([*dirs, d]))


def test_uneven_n_across_repeats_fails(tmp_path):
    assert "same n" in _set_failed(verify_repeat_set(_repeat_dirs(tmp_path, [1, 2], ns=[2, 3])))


def test_single_run_is_not_a_repeat_set(tmp_path):
    assert verify_repeat_set(_repeat_dirs(tmp_path, [1])) == []


# ---------------------------------------------------------------------------
# Passage cap (controlled-M scope)
# ---------------------------------------------------------------------------
def test_capped_variant_must_not_exceed_cap(tmp_path):
    m = _manifest(max_passages_by_pipeline={"Naive RAG": 30})
    rows = [_row(0, passages_used=30), _row(1, passages_used=31)]
    run = _write_run(tmp_path, rows, m, _summary())
    assert "passage cap" in _failed(verify_run_dir(run))


def test_capped_variant_within_cap_passes(tmp_path):
    m = _manifest(max_passages_by_pipeline={"Naive RAG": 30})
    rows = [_row(0, passages_used=30), _row(1, passages_used=30)]
    run = _write_run(tmp_path, rows, m, _summary())
    assert not _failed(verify_run_dir(run))


def test_uncapped_variant_must_use_everything_retrieved(tmp_path):
    rows = [_row(0), _row(1, passages_used=30, total_passages_retrieved=50)]
    run = _write_run(tmp_path, rows, _manifest(), _summary())
    assert "passage cap" in _failed(verify_run_dir(run))


def test_cap_never_reached_only_warns(tmp_path):
    m = _manifest(max_passages_by_pipeline={"Naive RAG": 30})
    rows = [_row(0, passages_used=12), _row(1, passages_used=9)]
    run = _write_run(tmp_path, rows, m, _summary())
    checks = verify_run_dir(run)
    assert not _failed(checks)
    assert any(c.name.endswith("passage cap reached") and c.status == WARN for c in checks)


# ---------------------------------------------------------------------------
# LLM-call budget
# ---------------------------------------------------------------------------
def _budget_manifest(pipeline: str, budget: int) -> dict:
    return _manifest(
        llm_call_budget_by_pipeline={"Naive RAG": budget},
        pipeline_by_variant={"Naive RAG": pipeline},
    )


def test_loop_over_budget_fails(tmp_path):
    rows = [_row(0, llm_calls=10), _row(1, llm_calls=11)]
    run = _write_run(tmp_path, rows, _budget_manifest("loop", 10), _summary())
    assert "call budget" in _failed(verify_run_dir(run))


def test_loop_on_budget_passes(tmp_path):
    rows = [_row(0, llm_calls=10), _row(1, llm_calls=10)]
    run = _write_run(tmp_path, rows, _budget_manifest("loop", 10), _summary())
    assert not _failed(verify_run_dir(run))


def test_agentic_over_budget_only_warns(tmp_path):
    rows = [_row(0, llm_calls=4), _row(1, llm_calls=6)]
    run = _write_run(tmp_path, rows, _budget_manifest("agentic", 4), _summary())
    checks = verify_run_dir(run)
    assert not _failed(checks)
    assert any(c.name.endswith("call budget") and c.status == WARN for c in checks)


def test_unspent_budget_warns(tmp_path):
    rows = [_row(0, llm_calls=4), _row(1, llm_calls=5)]
    run = _write_run(tmp_path, rows, _budget_manifest("loop", 10), _summary())
    checks = verify_run_dir(run)
    assert not _failed(checks)
    assert any(c.name.endswith("call budget spent") and c.status == WARN for c in checks)


def test_pipeline_count_drift_only_warns(tmp_path):
    rows = [_row(0), _row(1, llm_calls=6, metered_calls=4)]
    run = _write_run(tmp_path, rows, _manifest(), _summary())
    checks = verify_run_dir(run)
    assert not _failed(checks)
    assert any(
        c.name.endswith("pipeline count drifts from metered") and c.status == WARN for c in checks
    )


def test_one_call_of_drift_is_tolerated(tmp_path):
    rows = [_row(0), _row(1, llm_calls=5, metered_calls=4)]
    checks = verify_run_dir(_write_run(tmp_path, rows, _manifest(), _summary()))
    assert not any(c.name.endswith("drifts from metered") and c.status == WARN for c in checks)


def test_adapter_resends_explain_the_drift(tmp_path):
    rows = [_row(0), _row(1, llm_calls=3, metered_calls=5, structured_calls={"json_schema": 2})]
    checks = verify_run_dir(_write_run(tmp_path, rows, _manifest(), _summary()))
    assert not any(c.name.endswith("drifts from metered") and c.status == WARN for c in checks)


def test_unexplained_metered_calls_still_drift(tmp_path):
    rows = [_row(0), _row(1, llm_calls=3, metered_calls=5, structured_calls={})]
    checks = verify_run_dir(_write_run(tmp_path, rows, _manifest(), _summary()))
    assert any(c.name.endswith("drifts from metered") and c.status == WARN for c in checks)


# ---------------------------------------------------------------------------
# IRCoT step bound
# ---------------------------------------------------------------------------
def _ircot_manifest(max_steps: int) -> dict:
    return _manifest(
        pipeline_by_variant={"Naive RAG": "ircot"},
        ircot_params_by_pipeline={"Naive RAG": {"max_steps": max_steps, "per_step_k": 8}},
    )


def test_ircot_rows_over_the_step_bound_fail(tmp_path):
    rows = [_row(0, llm_calls=5), _row(1, llm_calls=6)]
    run = _write_run(tmp_path, rows, _ircot_manifest(4), _summary())
    assert "ircot steps" in _failed(verify_run_dir(run))


def test_ircot_rows_within_the_step_bound_pass(tmp_path):
    rows = [_row(0, llm_calls=5), _row(1, llm_calls=3)]
    run = _write_run(tmp_path, rows, _ircot_manifest(4), _summary())
    assert not _failed(verify_run_dir(run))


def test_ircot_resends_do_not_count_as_steps(tmp_path):
    rows = [
        _row(0, llm_calls=5, metered_calls=9, structured_calls={"json_schema": 4}),
        _row(1, llm_calls=3, metered_calls=3),
    ]
    checks = verify_run_dir(_write_run(tmp_path, rows, _ircot_manifest(4), _summary()))
    assert not _failed(checks)
    info = next(c for c in checks if c.name.endswith("ircot metered calls (info)"))
    assert (
        info.detail.startswith("12 metered for 8 pipeline calls") and "4 JSONAdapter" in info.detail
    )


# ---------------------------------------------------------------------------
# Judge re-scoring (opt-in)
# ---------------------------------------------------------------------------
def _write_judged(run: Path, ids: list[str], tag: str = "gpt") -> None:
    (run / f"rq1_naive_rag_judged_{tag}.jsonl").write_text(
        "".join(json.dumps({"id": i, "judge": 1}) + "\n" for i in ids)
    )


def test_missing_judged_file_fails(clean_run):
    assert "judged" in _failed(verify_judged(clean_run, "gpt"))


def test_judged_with_same_ids_passes(clean_run):
    _write_judged(clean_run, ["q0", "q1"])
    assert not _failed(verify_judged(clean_run, "gpt"))


def test_judged_with_different_ids_fails(clean_run):
    _write_judged(clean_run, ["q0", "q9"])
    checks = verify_judged(clean_run, "gpt")
    assert "judged" in _failed(checks)
    assert any("id sets differ" in c.detail for c in checks)


def test_judged_with_fewer_rows_fails(clean_run):
    _write_judged(clean_run, ["q0"])
    assert "judged" in _failed(verify_judged(clean_run, "gpt"))


def test_all_references_is_required(tmp_path):
    rows = [_row(0), _row(1)]
    del rows[0]["all_references"]
    run = _write_run(tmp_path, rows, _manifest(), _summary())
    checks = verify_run_dir(run)
    assert "required fields" in _failed(checks)
    assert any("all_references" in c.detail for c in checks if c.status == FAIL)


# ---------------------------------------------------------------------------
# Call outcomes: truncation fails, the rest is information
# ---------------------------------------------------------------------------
def test_truncated_call_fails(tmp_path):
    rows = [_row(0, truncated_calls=0), _row(1, truncated_calls=1)]
    run = _write_run(tmp_path, rows, _manifest(), _summary())
    assert "no truncated calls" in _failed(verify_run_dir(run))


def test_failed_refused_and_fallback_counts_are_information_only(tmp_path):
    extra = dict(
        truncated_calls=0,
        failed_calls=1,
        failed_by_status={"400": 1},
        timeout_calls=0,
        refused_calls=1,
        structured_calls={"json_schema": 1},
    )
    run = _write_run(tmp_path, [_row(0, **extra), _row(1, **extra)], _manifest(), _summary())
    checks = verify_run_dir(run)
    assert not _failed(checks), [c for c in checks if c.status == FAIL]
    info = {c.name.split(": ")[-1]: c.detail for c in checks}
    assert info["failed calls (info)"] == "2 row(s); by status {'400': 2}"
    assert info["adapter fallbacks (info)"] == "2 row(s); calls by mode {'json_schema': 2}"


def test_refusal_categories_and_raw_usage_fields_are_information_only(tmp_path):
    new_fields = dict(
        refused_calls=1,
        refusal_categories={"reasoning_extraction": 1},
        refusal_explained=1,
        usage_keys_seen=["input_tokens", "output_tokens"],
        output_tokens_details_seen=False,
        thinking_blocks=2,
    )
    run = _write_run(tmp_path, [_row(0, **new_fields), _row(1)], _manifest(), _summary())
    checks = verify_run_dir(run)
    assert not _failed(checks), [c for c in checks if c.status == FAIL]
    info = {c.name.split(": ")[-1]: c.detail for c in checks}
    assert info["refusal categories (info)"] == "{'reasoning_extraction': 1}; 1 with an explanation"
    assert info["raw usage fields (info)"] == (
        "keys ['input_tokens', 'output_tokens']; output_tokens_details absent; 2 thinking block(s)"
    )


def test_rows_written_before_the_new_fields_get_no_line(tmp_path):
    run = _write_run(tmp_path, [_row(0), _row(1)], _manifest(), _summary())
    names = {c.name.split(": ")[-1] for c in verify_run_dir(run)}
    assert "refusal categories (info)" not in names
    assert "raw usage fields (info)" not in names


def test_unpriced_anthropic_call_fails(tmp_path):
    claude = ["claude-sonnet-5-5"]
    rows = [
        _row(0, response_models=claude, calls_without_cost=0),
        _row(1, response_models=claude, calls_without_cost=1),
    ]
    run = _write_run(tmp_path, rows, _manifest(), _summary())
    assert "anthropic calls priced" in _failed(verify_run_dir(run))


def test_unpriced_call_of_another_provider_is_not_this_check(tmp_path):
    run = _write_run(tmp_path, [_row(0, calls_without_cost=1), _row(1)], _manifest(), _summary())
    assert "anthropic calls priced" not in _failed(verify_run_dir(run))


def test_attempts_are_reported_as_information(tmp_path):
    rows = [_row(0, failed_calls=0, attempts=3), _row(1, failed_calls=0, attempts=1)]
    run = _write_run(tmp_path, rows, _manifest(), _summary())
    checks = verify_run_dir(run)
    assert not _failed(checks)
    info = {c.name.split(": ")[-1]: c.detail for c in checks}
    assert info["attempts (info)"] == "4 request(s) sent for 2 metered call(s)"


# ---------------------------------------------------------------------------
# Error rows count for refusals and pricing; phase and created_at
# ---------------------------------------------------------------------------
def _claude_error_row(i: int, **extra) -> dict:
    return {
        "id": f"q{i}",
        "question": "x",
        "error": "parse failure",
        "error_class": "refused",
        "pipeline": "p",
        "response_models": ["claude-sonnet-5-5"],
        **extra,
    }


def test_refusal_on_an_error_row_is_counted(tmp_path):
    rows = [_row(0, refused_calls=0), _claude_error_row(1, refused_calls=1, calls_without_cost=0)]
    run = _write_run(tmp_path, rows, _manifest(), _summary())
    info = {c.name.split(": ")[-1]: c.detail for c in verify_run_dir(run)}
    assert info["refused calls (info)"] == "1 row(s) with a provider refusal"
    assert info["error classes (info)"] == "{'refused': 1}"


def test_unpriced_anthropic_call_on_an_error_row_fails(tmp_path):
    rows = [_row(0), _claude_error_row(1, refused_calls=0, calls_without_cost=1)]
    run = _write_run(tmp_path, rows, _manifest(), _summary())
    assert "anthropic calls priced" in _failed(verify_run_dir(run))


def _warned(checks) -> set[str]:
    return {c.name.split(": ")[-1] for c in checks if c.status == WARN}


def test_missing_campaign_phase_warns(tmp_path):
    run = _write_run(tmp_path, [_row(0), _row(1)], _manifest(), _summary())
    assert "campaign phase fixed" in _warned(verify_run_dir(run))
    run2 = _write_run(
        tmp_path / "b", [_row(0), _row(1)], _manifest(campaign_phase="smoke"), _summary()
    )
    assert "campaign phase fixed" not in _warned(verify_run_dir(run2))


def test_created_at_change_during_run_warns(tmp_path):
    m = _manifest()
    m["models"][0] |= {
        "provider_model_created_at": "2026-09-28T00:00:00Z",
        "provider_model_created_at_end": "2026-11-01T00:00:00Z",
    }
    run = _write_run(tmp_path, [_row(0), _row(1)], m, _summary())
    checks = verify_run_dir(run)
    assert "model created_at stable preprocess" in _warned(checks)
    assert "model created_at stable preprocess" not in _failed(checks)


# ---------------------------------------------------------------------------
# Grid mode: refused and truncated error rows are outcomes, not defects
# ---------------------------------------------------------------------------
def _err(i: int, cls: str | None) -> dict:
    row = {"id": f"q{i}", "question": "x", "error": "e", "pipeline": "p"}
    if cls is not None:
        row["error_class"] = cls
    return row


def _grid_rows(errors: list[str | None], n_ok: int) -> list[dict]:
    return [_row(i) for i in range(n_ok)] + [_err(n_ok + j, c) for j, c in enumerate(errors)]


def test_grid_keeps_refused_and_truncated_out_of_errors(tmp_path):
    rows = _grid_rows(["refused", "truncated"], n_ok=40)
    run = _write_run(tmp_path, rows, _manifest(campaign_phase="grid", n=42), _summary())
    checks = verify_run_dir(run, n=42)
    assert "errors" not in _failed(checks)
    info = {c.name.split(": ")[-1]: c.detail for c in checks}
    assert info["final error rows (info)"] == (
        "to exclude: refused 1, platform_filtered 0; scored wrong: truncated 1"
    )
    assert "truncated share" not in _warned(checks)


def test_grid_still_fails_other_and_unclassified_errors(tmp_path):
    for cls in ("other", None):
        d = tmp_path / str(cls)
        rows = _grid_rows([cls], n_ok=2)
        run = _write_run(d, rows, _manifest(campaign_phase="grid", n=3), _summary())
        assert "errors" in _failed(verify_run_dir(run, n=3)), cls


def test_grid_warns_when_truncation_exceeds_five_percent(tmp_path):
    rows = _grid_rows(["truncated"] * 2, n_ok=18)  # 10%
    run = _write_run(tmp_path, rows, _manifest(campaign_phase="grid", n=20), _summary())
    checks = verify_run_dir(run, n=20)
    assert "truncated share" in _warned(checks)
    assert "errors" not in _failed(checks)


def test_smoke_and_pilot_fail_every_error_row(tmp_path):
    for phase in ("smoke", "pilot"):
        d = tmp_path / phase
        rows = _grid_rows(["refused"], n_ok=2)
        run = _write_run(d, rows, _manifest(campaign_phase=phase, n=3), _summary())
        assert "errors" in _failed(verify_run_dir(run, n=3)), phase


def test_refusal_in_a_prior_attempt_or_answered_row_is_counted(tmp_path):
    rows = [
        _row(0, refused_calls=0, refusal_seen=True),
        _row(1, refused_calls=0, prior_attempt_usage={"refused_calls": 1}),
        _row(2, refused_calls=0),
    ]
    run = _write_run(tmp_path, rows, _manifest(n=3), _summary())
    info = {c.name.split(": ")[-1]: c.detail for c in verify_run_dir(run, n=3)}
    assert info["refused calls (info)"] == "2 row(s) with a provider refusal"


def test_grid_reports_truncated_answered_rows_without_failing(tmp_path):
    rows = [_row(i, truncated_calls=0) for i in range(19)] + [_row(19, truncated_calls=1)]
    run = _write_run(tmp_path, rows, _manifest(campaign_phase="grid", n=20), _summary())
    checks = verify_run_dir(run, n=20)
    assert "no truncated calls" not in _failed(checks)
    info = {c.name.split(": ")[-1]: c.detail for c in checks}
    assert info["truncated calls (info)"] == "1 row(s) with a call stopped at max_tokens"


def test_pilot_and_smoke_still_fail_truncated_answered_rows(tmp_path):
    for phase in ("smoke", "pilot", None):
        d = tmp_path / str(phase)
        rows = [_row(0, truncated_calls=0), _row(1, truncated_calls=1)]
        run = _write_run(d, rows, _manifest(campaign_phase=phase), _summary())
        assert "no truncated calls" in _failed(verify_run_dir(run)), phase


def test_truncated_share_counts_answered_and_error_rows(tmp_path):
    # 18 clean + 1 answered-but-truncated + 1 truncated error row = 2/20 = 10%
    rows = [_row(i, truncated_calls=0) for i in range(18)]
    rows += [_row(18, truncated_calls=1), _err(19, "truncated")]
    run = _write_run(tmp_path, rows, _manifest(campaign_phase="grid", n=20), _summary())
    checks = verify_run_dir(run, n=20)
    assert "truncated share" in _warned(checks)
    detail = next(c.detail for c in checks if c.name.endswith("truncated share"))
    assert detail == "2/20 = 10.0% (answered 1, error 1; limit 5%)"


def test_truncated_share_within_limit_does_not_warn(tmp_path):
    rows = [_row(i, truncated_calls=0) for i in range(19)] + [_row(19, truncated_calls=1)]
    run = _write_run(tmp_path, rows, _manifest(campaign_phase="grid", n=20), _summary())
    assert "truncated share" not in _warned(verify_run_dir(run, n=20))


# ---------------------------------------------------------------------------
# V4: a refused question must be excluded for the run's model
# ---------------------------------------------------------------------------
def _refusal_run(tmp_path, phase: str, exclusions: dict | None):
    tree = tmp_path / phase
    (tree / "run").mkdir(parents=True)
    if exclusions is not None:
        (tree / "exclusions.json").write_text(json.dumps(exclusions))
    rows = [_row(0, refusal_seen=False), _row(1, refusal_seen=True)]
    m = _manifest(campaign_phase=phase, run_key="rq1|2wikimultihopqa|claude-sonnet-5-5|n2|k1")
    return _write_run(tree / "run", rows, m, _summary())


def test_grid_fails_a_refusal_missing_from_exclusions(tmp_path):
    run = _refusal_run(tmp_path, "grid", {"2wikimultihopqa": {"gpt-5-mini": [{"id": "q1"}]}})
    assert "refusal not in exclusions" in _failed(verify_run_dir(run))


def test_grid_passes_a_refusal_excluded_for_its_model(tmp_path):
    excl = {"2wikimultihopqa": {"claude-sonnet-5-5": [{"id": "q1"}]}}
    run = _refusal_run(tmp_path, "grid", excl)
    assert "refusal not in exclusions" not in _failed(verify_run_dir(run))


def test_grid_fails_when_exclusions_have_no_model(tmp_path):
    run = _refusal_run(tmp_path, "grid", {"2wikimultihopqa": [{"id": "q1"}]})
    assert "refusal not in exclusions" in _failed(verify_run_dir(run))


def test_smoke_reports_a_missing_refusal_as_information(tmp_path):
    run = _refusal_run(tmp_path, "smoke", None)
    checks = verify_run_dir(run)
    assert not any("refusal not in exclusions" in c.name and c.status == FAIL for c in checks)
    info = {c.name.split(": ")[-1]: c.detail for c in checks}
    assert info["refusal not in exclusions (info)"].startswith("1 of 1")


# ---------------------------------------------------------------------------
# V14: a question blocked by the platform filter to the end
# ---------------------------------------------------------------------------
def _filtered_run(tmp_path, phase: str, exclusions: dict | None):
    tree = tmp_path / phase
    (tree / "run").mkdir(parents=True)
    if exclusions is not None:
        (tree / "exclusions.json").write_text(json.dumps(exclusions))
    rows = [_row(0), _row(1), _err(2, "platform_filtered")]
    m = _manifest(campaign_phase=phase, n=3, run_key="rq1|2wikimultihopqa|gpt-5-mini|n3|k1")
    return _write_run(tree / "run", rows, m, _summary())


def test_grid_takes_an_excluded_filter_block_as_information(tmp_path):
    run = _filtered_run(tmp_path, "grid", {"2wikimultihopqa": {"gpt-5-mini": [{"id": "q2"}]}})
    checks = verify_run_dir(run, n=3)
    assert not {"errors", "platform filter not in exclusions"} & _failed(checks)
    info = {c.name.split(": ")[-1]: c.detail for c in checks}
    assert "platform_filtered 1" in info["final error rows (info)"]


def test_grid_fails_a_filter_block_missing_from_exclusions(tmp_path):
    run = _filtered_run(tmp_path, "grid", {"2wikimultihopqa": {"gpt-5-mini": []}})
    failed = _failed(verify_run_dir(run, n=3))
    assert "platform filter not in exclusions" in failed
    assert "errors" not in failed


def test_smoke_and_pilot_fail_a_filter_block(tmp_path):
    for phase in ("smoke", "pilot"):
        run = _filtered_run(tmp_path, phase, {"2wikimultihopqa": {"gpt-5-mini": [{"id": "q2"}]}})
        assert "errors" in _failed(verify_run_dir(run, n=3)), phase


def test_grid_needs_only_refusals_that_stood_in_exclusions(tmp_path):
    """A refusal recovered by a resend is listed for the sensitivity analysis only."""
    tree = tmp_path / "grid"
    (tree / "run").mkdir(parents=True)
    (tree / "exclusions.json").write_text(
        json.dumps({"2wikimultihopqa": {"claude-sonnet-5-5": [{"id": "q1"}]}})
    )
    recovered = dict(
        refused_calls=1,
        refusal_categories={"reasoning_extraction": 1},
        refusal_retries=1,
        refusals_recovered=1,
        refusals_final_by_category={},
        refusal_seen=False,
        refusal_first_seen=True,
    )
    stood = dict(
        refused_calls=4,
        refusal_categories={"reasoning_extraction": 4},
        refusal_retries=3,
        refusals_recovered=0,
        refusals_final_by_category={"reasoning_extraction": 1},
        refusal_seen=True,
        refusal_first_seen=True,
    )
    m = _manifest(campaign_phase="grid", run_key="rq1|2wikimultihopqa|claude-sonnet-5-5|n2|k1")
    run = _write_run(tree / "run", [_row(0, **recovered), _row(1, **stood)], m, _summary())
    checks = verify_run_dir(run)
    assert "refusal not in exclusions" not in _failed(checks)
    info = {c.name.split(": ")[-1]: c.detail for c in checks}
    assert info["refused calls (info)"] == "1 row(s) with a provider refusal"
    assert info["refusal resends (info)"] == (
        "4 resend(s), 1 recovered; first refusals (sensitivity analysis) on 2 question(s): "
        "['q0', 'q1']"
    )


# ---------------------------------------------------------------------------
# Per-variant settings: manifest (declared before the run) vs summary (as run)
VARIANT_CHECK = "variant settings vs manifest"


def _status(checks, name: str) -> str:
    return next(c.status for c in checks if c.name.endswith(name))


def test_variant_settings_matching_the_manifest_pass(tmp_path):
    summary = _summary()
    run = _write_run(
        tmp_path,
        [_row(0), _row(1)],
        _manifest(settings_by_variant={"Naive RAG": summary["settings"]}),
        summary,
    )
    assert _status(verify_run_dir(run), VARIANT_CHECK) == "OK"


@pytest.mark.parametrize(
    ("key", "declared"),
    [
        ("agent_max_iterations", 5),
        ("enabled_tools", None),
        ("max_passages", 50),
        ("llm_call_budget", 10),
    ],
)
def test_variant_settings_differing_from_the_manifest_fail(tmp_path, key, declared):
    summary = _summary()
    summary["settings"].update(agent_max_iterations=10, llm_call_budget=None)
    run = _write_run(
        tmp_path,
        [_row(0), _row(1)],
        _manifest(settings_by_variant={"Naive RAG": {**summary["settings"], key: declared}}),
        summary,
    )
    checks = verify_run_dir(run)
    assert VARIANT_CHECK in _failed(checks)
    assert key in next(c.detail for c in checks if c.name.endswith(VARIANT_CHECK))


def test_variant_missing_from_the_manifest_fails(tmp_path):
    run = _write_run(
        tmp_path,
        [_row(0), _row(1)],
        _manifest(settings_by_variant={"Loop Refinement": {}}),
        _summary(),
    )
    assert VARIANT_CHECK in _failed(verify_run_dir(run))


def test_version1_manifest_without_variant_settings_warns(clean_run):
    """Old manifests stay as recorded: their defaults are flagged, not trusted."""
    assert _status(verify_run_dir(clean_run), VARIANT_CHECK) == WARN
