"""A refusal that stood leaves the pipeline's answer None; reporting must survive it.

litellm hands an Anthropic refusal back as an ordinary "stop", so no exception
is raised and the pipeline returns answer=None. A Sonnet MuSiQue run finished
all 1,000 rows and then died in ROUGE tokenisation on that None, and --resume
died the same way because the row was already in the checkpoint.
"""

from __future__ import annotations

import json

from agentic_rag.pipeline.base import PipelineResult
from experiments.common import print_comparison_table, run_pipeline_on_dataset, save_results


class RefusingPipeline:
    def run(self, question: str) -> PipelineResult:
        return PipelineResult(question=question, answer=None)


def _dataset(n: int) -> list[dict]:
    return [{"id": f"q{i}", "question": f"question {i}", "answer": f"answer {i}"} for i in range(n)]


def test_a_none_answer_is_recorded_as_empty(tmp_path):
    (row,) = run_pipeline_on_dataset(RefusingPipeline(), _dataset(1), "p", checkpoint_dir=tmp_path)
    assert row["prediction"] == ""
    assert "error" not in row


def test_none_predictions_in_a_checkpoint_are_scored_as_empty(tmp_path):
    rows = [
        {"id": "q0", "question": "question 0", "reference": "answer 0", "prediction": None},
        {"id": "q1", "question": "question 1", "reference": "answer 1", "prediction": "answer 1"},
    ]
    checkpoint = tmp_path / "ckpt" / "p_checkpoint.jsonl"
    checkpoint.parent.mkdir()
    checkpoint.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    before = checkpoint.read_bytes()

    results = run_pipeline_on_dataset(
        RefusingPipeline(), _dataset(2), "p", checkpoint_dir=checkpoint.parent
    )
    assert [r["prediction"] for r in results] == [None, "answer 1"], "resume reads rows as stored"

    print_comparison_table({"p": results})
    save_results(results, "none", run_dir=tmp_path / "a")
    empty = [{**r, "prediction": r["prediction"] or ""} for r in results]
    save_results(empty, "empty", run_dir=tmp_path / "b")

    got = json.loads((tmp_path / "a" / "none_summary.json").read_text(encoding="utf-8"))
    want = json.loads((tmp_path / "b" / "empty_summary.json").read_text(encoding="utf-8"))
    assert got["metrics"] == want["metrics"]
    assert got["metrics"]["exact_match"] == 0.5
    assert checkpoint.read_bytes() == before, "the checkpoint file is not rewritten"
