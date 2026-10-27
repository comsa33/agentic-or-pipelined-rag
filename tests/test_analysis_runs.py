"""The analysis loader: error classes, exclusions, and the campaign layout.

The regression test reads the gpt-5-mini k=1 campaign runs, when present,
and checks that every cell's n and mean F1 match the summaries written at
run time: rows from before error_class existed must load exactly as before.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from experiments.analysis.runs import _model_from_stamped_name, _split_stamped_name, load_runs


def _run(tree: Path, name: str, pipeline: str, rows: list[dict]) -> None:
    d = tree / name
    d.mkdir(parents=True, exist_ok=True)
    (d / f"rq1_{pipeline}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _ok(i: int, pred: str = "a") -> dict:
    return {"id": f"q{i}", "prediction": pred, "reference": "a"}


def test_campaign_names_are_recognised():
    assert _split_stamped_name("x_rq1_hotpotqa_n200_gpt-5-mini_k3") == (
        "x_rq1_hotpotqa_n200_gpt-5-mini",
        3,
    )
    assert _split_stamped_name("x_rq1_hotpotqa_n3_claude-sonnet-5-5_qabcd1234") == (
        "x_rq1_hotpotqa_n3_claude-sonnet-5-5",
        None,
    )
    assert _model_from_stamped_name("x_rq1_musique_n2_claude-sonnet-5-5_qabcd1234_k1") == (
        "claude-sonnet-5-5"
    )
    assert _model_from_stamped_name("20260324_rq1_hotpotqa_n200_gpt-5-mini") == "gpt-5-mini"


def test_error_rows_by_class(tmp_path):
    rows = [
        _ok(0),
        {"id": "q1", "error": "cut", "error_class": "truncated", "reference": "a"},
        {"id": "q2", "error": "429", "error_class": "other", "reference": "a"},
        {"id": "q3", "error": "old row without a class", "reference": "a"},
    ]
    _run(tmp_path, "s_rq1_hotpotqa_n4_gpt-5-mini_k1", "naive_rag", rows)
    (rec,) = load_runs(tmp_path)
    assert [r["id"] for r in rec["rows"]] == ["q0", "q1"]
    wrong = rec["rows"][1]
    assert wrong["prediction"] == "" and wrong["scored_wrong"] is True
    assert rec["repeat_index"] == 1 and rec["n_excluded"] == 0


def test_tree_exclusions_apply_to_their_model_only(tmp_path):
    (tmp_path / "exclusions.json").write_text(
        json.dumps({"hotpotqa": {"claude-sonnet-5-5": [{"id": "q1", "reason": "refused"}]}})
    )
    refused = {"id": "q1", "error": "x", "error_class": "refused", "refusal_seen": True}
    _run(tmp_path, "s_rq1_hotpotqa_n2_claude-sonnet-5-5_k1", "naive_rag", [_ok(0), refused])
    _run(tmp_path, "s_rq1_hotpotqa_n2_gpt-5-mini_k1", "naive_rag", [_ok(0), _ok(1)])
    by_model = {r["model"]: r for r in load_runs(tmp_path)}
    assert [r["id"] for r in by_model["claude-sonnet-5-5"]["rows"]] == ["q0"]
    assert by_model["claude-sonnet-5-5"]["n_excluded"] == 1
    assert len(by_model["gpt-5-mini"]["rows"]) == 2
    assert by_model["gpt-5-mini"]["n_excluded"] == 0
    kept = {r["model"]: r for r in load_runs(tmp_path, apply_exclusions=False)}
    assert kept["claude-sonnet-5-5"]["n_excluded"] == 0


def test_unscoped_exclusions_are_refused_for_claude(tmp_path):
    (tmp_path / "exclusions.json").write_text(json.dumps({"hotpotqa": [{"id": "q1"}]}))
    _run(tmp_path, "s_rq1_hotpotqa_n1_claude-sonnet-5-5_k1", "naive_rag", [_ok(0)])
    with pytest.raises(ValueError, match="no model"):
        load_runs(tmp_path)


def test_judge_scores_join_by_id_and_never_decide_rows(tmp_path):
    """A judged file has no error key: its empty-prediction rows must not count."""
    run = "s_rq1_hotpotqa_n3_gpt-5-mini_k1"
    rows = [
        _ok(0),
        {"id": "q1", "error": "filter", "error_class": "other", "reference": "a"},
        {"id": "q2", "error": "cut", "error_class": "truncated", "reference": "a"},
    ]
    _run(tmp_path, run, "naive_rag", rows)
    judged = [
        {"id": "q0", "judge_status": "ok", "llm_judge": 1.0},
        {"id": "q1", "prediction": "", "judge_status": "ok", "llm_judge": 0.0},
        {"id": "q2", "prediction": "", "judge_status": "ok", "llm_judge": 1.0},
    ]
    (tmp_path / run / "rq1_naive_rag_judged_tag1.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in judged)
    )
    (rec,) = load_runs(tmp_path, judge="tag1")
    assert [(r["id"], r["llm_judge"]) for r in rec["rows"]] == [("q0", 1.0), ("q2", 0.0)]
    assert rec["n_judge_missing"] == 0


# ---------------------------------------------------------------------------
# Regression on the gpt-5-mini k=1 campaign runs (read only; *_k1 only)
# ---------------------------------------------------------------------------
# Campaign results are not in the repository; point CAMPAIGN_RESULTS_DIR at
# a results-campaign directory to run these tests, which only read it.
CAMPAIGN = Path(os.environ.get("CAMPAIGN_RESULTS_DIR", "/nonexistent"))
needs_campaign = pytest.mark.skipif(
    not CAMPAIGN.is_dir(), reason="CAMPAIGN_RESULTS_DIR not set to campaign results"
)
JUDGE = "gpt-4-1-nano_7bcf05"


def _k1_tree(tmp_path: Path, exclusions: dict | None = None) -> Path:
    """The full-n gpt-5-mini k1 runs, linked into a tree of their own."""
    tree = tmp_path / "tree"
    tree.mkdir()
    for d in sorted(CAMPAIGN.iterdir()):
        if d.is_dir() and d.name.endswith("_gpt-5-mini_k1") and "_n2_" not in d.name:
            (tree / d.name).symlink_to(d)
    if exclusions is not None:
        (tree / "exclusions.json").write_text(json.dumps(exclusions))
    return tree


def _summary_metrics(rec: dict, tree: Path) -> dict:
    (run_dir,) = [d for d in tree.iterdir() if d.is_dir() and f"_rq1_{rec['dataset']}_" in d.name]
    summary = json.loads((run_dir / f"rq1_{rec['pipeline']}_summary.json").read_text())
    return summary["metrics"]


def _metrics(rows: list[dict]) -> dict:
    from agentic_rag.evaluation.metrics import evaluate_batch

    return evaluate_batch(
        [r.get("prediction", "") for r in rows],
        [r.get("reference", "") for r in rows],
        compute_bert_score=False,
    )


@needs_campaign
def test_k1_cells_without_error_rows_match_their_summaries(tmp_path):
    """(1) n, F1 and EM exactly as written at run time."""
    tree = _k1_tree(tmp_path)
    checked = 0
    for rec in load_runs(tree, apply_exclusions=False, require_paired_n=False):
        metrics = _summary_metrics(rec, tree)
        run_dir = next(d for d in tree.iterdir() if f"_rq1_{rec['dataset']}_" in d.name)
        raw = (run_dir / f"rq1_{rec['pipeline']}.jsonl").read_text().splitlines()
        if any('"error"' in line for line in raw):
            continue
        got = _metrics(rec["rows"])
        assert (got["n"], got["f1"], got["exact_match"]) == (
            metrics["n"],
            metrics["f1"],
            metrics["exact_match"],
        ), (rec["dataset"], rec["pipeline"])
        checked += 1
    assert checked == 17  # 20 cells less the three CRAG cells with filter errors


@needs_campaign
def test_k1_crag_error_rows_are_dropped_not_scored_wrong(tmp_path):
    """(2) Old error rows carry no error_class: dropped as before, n as in the summary."""
    tree = _k1_tree(tmp_path)
    crag = [
        r
        for r in load_runs(tree, apply_exclusions=False, require_paired_n=False)
        if r["pipeline"] == "crag_replica"
    ]
    n_by_dataset = {r["dataset"]: len(r["rows"]) for r in crag}
    assert n_by_dataset == {
        "2wikimultihopqa": 199,
        "hotpotqa": 197,
        "musique": 198,
        "financebench": 150,
    }
    for rec in crag:
        assert not any(r.get("scored_wrong") for r in rec["rows"])
        got = _metrics(rec["rows"])
        assert (got["n"], got["f1"]) == (
            _summary_metrics(rec, tree)["n"],
            _summary_metrics(rec, tree)["f1"],
        )


# The exclusions as corrected by the 2Wiki audit (2026-10-21), for this test
# only: the campaign file itself is corrected after the chain ends.
CORRECTED_EXCLUSIONS = {
    "_schema": {"note": "test copy of the audit's correction"},
    "2wikimultihopqa": {"gpt-5-mini": [{"id": "2wiki_e7f8e98808a311ebbd7cac1f6bf848b6"}]},
    "hotpotqa": {
        "gpt-5-mini": [
            {"id": "hotpotqa_5ae5cf625542996de7b71a22"},
            {"id": "hotpotqa_5a840e395542992ef85e239d"},
            {"id": "hotpotqa_5add0f635542994ed6169c56"},
        ]
    },
    "musique": {
        "gpt-5-mini": [
            {"id": "musique_2hop__789565_416401"},
            {"id": "musique_2hop__575188_342798"},
        ]
    },
}


@needs_campaign
def test_k1_exclusions_apply_to_every_pipeline_of_the_dataset(tmp_path):
    """(3) Paired n once the excluded questions leave every pipeline."""
    tree = _k1_tree(tmp_path, CORRECTED_EXCLUSIONS)
    expected = {"2wikimultihopqa": 199, "hotpotqa": 197, "musique": 198, "financebench": 150}
    recs = load_runs(tree, judge=JUDGE)
    assert len(recs) == 20
    for rec in recs:
        assert len(rec["rows"]) == expected[rec["dataset"]], (rec["dataset"], rec["pipeline"])
        assert (
            rec["n_excluded"] == 200 - expected[rec["dataset"]]
            if rec["dataset"] != ("financebench")
            else rec["n_excluded"] == 0
        )
        assert rec["n_judge_missing"] == 0
        assert all(r["llm_judge"] is not None for r in rec["rows"])


def test_significance_skips_judged_files_and_applies_exclusions(tmp_path):
    """Read alone, a judged file would turn the error row into an answered one."""
    from experiments.analysis.significance import SignificanceAnalyzer

    run = "s_rq1_hotpotqa_n3_gpt-5-mini_k1"
    rows = [
        {**_ok(0), "pipeline": "naive_rag"},
        {**_ok(1), "pipeline": "naive_rag"},
        {"id": "q2", "error": "filter", "pipeline": "naive_rag", "reference": "a"},
    ]
    _run(tmp_path, run, "naive_rag", rows)
    judged = [
        {"id": f"q{i}", "pipeline": "naive_rag", "prediction": "", "reference": "a"}
        for i in range(3)
    ]
    (tmp_path / run / "rq1_naive_rag_judged_tag1.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in judged)
    )
    (tmp_path / "exclusions.json").write_text(
        json.dumps({"hotpotqa": {"gpt-5-mini": [{"id": "q1"}]}})
    )
    scores = SignificanceAnalyzer.from_results_dir(tmp_path / run).scores
    assert scores["naive_rag"]["ids"] == ["q0"]
    assert scores["naive_rag"]["f1"] == [1.0]


def test_platform_filtered_rows_leave_the_analysis(tmp_path):
    rows = [_ok(0), {"id": "q1", "error": "blocked", "error_class": "platform_filtered"}]
    _run(tmp_path, "s_rq1_hotpotqa_n2_gpt-5-mini_k1", "naive_rag", rows)
    (rec,) = load_runs(tmp_path)
    assert [r["id"] for r in rec["rows"]] == ["q0"]


# ---------------------------------------------------------------------------
# Paired n and order-independent bootstrap
# ---------------------------------------------------------------------------
def test_unpaired_cell_is_refused(tmp_path):
    from experiments.analysis.runs import UnpairedCellError

    run = "s_rq1_hotpotqa_n3_gpt-5-mini_k1"
    _run(tmp_path, run, "naive_rag", [_ok(0), _ok(1), _ok(2)])
    _run(tmp_path, run, "crag_replica", [_ok(0), _ok(1), {"id": "q2", "error": "x"}])
    with pytest.raises(UnpairedCellError, match="crag_replica"):
        load_runs(tmp_path)
    assert len(load_runs(tmp_path, require_paired_n=False)) == 2


def test_significance_refuses_an_unpaired_cell(tmp_path):
    from experiments.analysis.runs import UnpairedCellError
    from experiments.analysis.significance import SignificanceAnalyzer

    run = "s_rq1_hotpotqa_n3_gpt-5-mini_k1"
    _run(tmp_path, run, "naive_rag", [{**_ok(i), "pipeline": "naive_rag"} for i in range(3)])
    _run(tmp_path, run, "crag_replica", [{**_ok(i), "pipeline": "crag_replica"} for i in range(2)])
    with pytest.raises(UnpairedCellError):
        SignificanceAnalyzer.from_results_dir(tmp_path / run)


def test_bootstrap_does_not_depend_on_row_or_pipeline_order(tmp_path):
    import random

    from experiments.analysis.significance import SignificanceAnalyzer

    rng = random.Random(0)
    preds = {p: [rng.choice(["a", "b"]) for _ in range(40)] for p in ("p1", "p2", "p3")}

    def write(d, order, pipeline_order):
        for p in pipeline_order:
            rows = [
                {"id": f"q{i:02d}", "pipeline": p, "prediction": preds[p][i], "reference": "a"}
                for i in order
            ]
            _run(d.parent, d.name, p, rows)

    a, b = (
        tmp_path / "a" / "s_rq1_hotpotqa_n40_gpt-5-mini_k1",
        tmp_path / "b" / "s_rq1_hotpotqa_n40_gpt-5-mini_k1",
    )
    shuffled = list(range(40))
    rng.shuffle(shuffled)
    write(a, range(40), ("p1", "p2", "p3"))
    write(b, shuffled, ("p3", "p1", "p2"))
    sa, sb = SignificanceAnalyzer.from_results_dir(a), SignificanceAnalyzer.from_results_dir(b)
    pa = sa.pairwise_tests(baseline="p1", n_boot=2000)
    pb = sb.pairwise_tests(baseline="p1", n_boot=2000)
    assert {k: v["boot_p"] for k, v in pa.items()} == {k: v["boot_p"] for k, v in pb.items()}
    assert sa.bootstrap_ci(n_boot=2000) == sb.bootstrap_ci(n_boot=2000)
