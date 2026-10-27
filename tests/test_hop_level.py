"""hop_level's paired bootstrap follows significance.py's rules."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from experiments.analysis.hop_level import HopLevelAnalyzer
from experiments.analysis.runs import UnpairedCellError

HOPS = [2 if i % 2 else 4 for i in range(40)]


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """A raw MuSiQue file whose gold n_hops give the hop reference HOPS."""
    raw = tmp_path / "raw"
    raw.mkdir()
    items = [
        {"id": f"q{i:02d}", "question": "q?", "metadata": {"n_hops": hop}}
        for i, hop in enumerate(HOPS)
    ]
    (raw / "musique.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    return raw


def _rows(preds: list[str], hops: list[int]) -> list[dict]:
    return [
        {
            "id": f"q{i:02d}",
            "prediction": p,
            "reference": "a",
            "question_difficulty": {"hop_count": hops[i]},
        }
        for i, p in enumerate(preds)
    ]


def _records(order: list[int], record_order: list[str], k: int | None = 1) -> list[dict]:
    rng = random.Random(0)
    hops = HOPS
    preds = {
        p: [rng.choice(["a", "b"]) for _ in range(40)]
        for p in ("agentic_(react)", "loop_refinement", "naive_rag")
    }
    out = []
    for pipeline in record_order:
        rows = _rows(preds[pipeline], hops)
        out.append(
            {
                "model": "m",
                "dataset": "musique",
                "pipeline": pipeline,
                "repeat_index": k,
                "rows": [rows[i] for i in order],
            }
        )
    return out


def test_paired_deltas_do_not_depend_on_row_or_record_order(raw_dir):
    shuffled = list(range(40))
    random.Random(1).shuffle(shuffled)
    a = HopLevelAnalyzer(
        _records(list(range(40)), ["agentic_(react)", "loop_refinement", "naive_rag"]),
        raw_dir=raw_dir,
    )
    b = HopLevelAnalyzer(
        _records(shuffled, ["naive_rag", "loop_refinement", "agentic_(react)"]), raw_dir=raw_dir
    )
    for baseline in ("loop_refinement", "naive_rag"):
        assert a.paired_deltas(baseline, n_boot=2000) == b.paired_deltas(baseline, n_boot=2000)


def test_unpaired_comparison_is_refused(raw_dir):
    records = _records(list(range(40)), ["agentic_(react)", "loop_refinement"])
    records[1]["rows"] = records[1]["rows"][:-1]
    with pytest.raises(UnpairedCellError):
        HopLevelAnalyzer(records, raw_dir=raw_dir).paired_deltas(n_boot=100)


def test_repeats_are_compared_within_their_own_run(raw_dir):
    """k=1 and k=2 records of one cell used to overwrite each other."""
    records = _records(list(range(40)), ["agentic_(react)", "loop_refinement"], k=1)
    records += _records(list(range(40)), ["agentic_(react)", "loop_refinement"], k=2)
    deltas = HopLevelAnalyzer(records, raw_dir=raw_dir).paired_deltas(n_boot=100)
    assert sorted({(d["repeat_index"], d["hop_count"]) for d in deltas}) == [
        (1, 2),
        (1, 4),
        (2, 2),
        (2, 4),
    ]


# ---------------------------------------------------------------------------
# Hop comes from the reference, not from the row stamp
# ---------------------------------------------------------------------------
def test_unstamped_error_row_stays_in_its_hop_cell(raw_dir):
    """A truncated error row is kept, scored wrong; its hop comes from the reference."""
    records = _records(list(range(40)), ["agentic_(react)", "loop_refinement"])
    row = records[0]["rows"][3]
    records[0]["rows"][3] = {
        "id": row["id"],
        "prediction": "",
        "reference": "a",
        "error": "truncated",
        "error_class": "truncated",
        "scored_wrong": True,
    }
    cells = HopLevelAnalyzer(records, raw_dir=raw_dir).per_cell()
    n = {(c["pipeline"], c["hop_count"]): c["n"] for c in cells}
    assert n[("agentic_(react)", HOPS[3])] == n[("loop_refinement", HOPS[3])] == 20
    deltas = HopLevelAnalyzer(records, raw_dir=raw_dir).paired_deltas(n_boot=100)
    assert {d["hop_count"]: d["n"] for d in deltas} == {2: 20, 4: 20}


def test_a_stamp_that_disagrees_with_the_reference_stops_the_analysis(raw_dir):
    from experiments.difficulty import HopMismatchError

    records = _records(list(range(40)), ["agentic_(react)", "loop_refinement"])
    records[1]["rows"][0]["question_difficulty"] = {"hop_count": 3}
    with pytest.raises(HopMismatchError, match="q00"):
        HopLevelAnalyzer(records, raw_dir=raw_dir).per_cell()


def test_a_question_missing_from_the_reference_stops_the_analysis(raw_dir):
    records = _records(list(range(40)), ["agentic_(react)", "loop_refinement"])
    records[0]["rows"][0]["id"] = "unknown"
    with pytest.raises(KeyError, match="unknown"):
        HopLevelAnalyzer(records, raw_dir=raw_dir).per_cell()
