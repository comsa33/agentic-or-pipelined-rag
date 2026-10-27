"""The hop reference reproduces every stamp the campaign runs wrote.

Hop-level analyses read the hop count from the raw dataset by question id
(experiments.difficulty) instead of from the row. That is only a change of
source if the two agree; these tests check it on the campaign results, which
are not in the repository: point CAMPAIGN_RESULTS_DIR at a results-campaign
directory (and RAW_DATA_DIR at data/raw, default its sibling) to run them.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pytest

from experiments.analysis.runs import DATASETS
from experiments.difficulty import hop_reference

CAMPAIGN = Path(os.environ.get("CAMPAIGN_RESULTS_DIR", "/nonexistent"))
RAW = Path(os.environ.get("RAW_DATA_DIR", str(CAMPAIGN.parent / "raw")))
needs_campaign = pytest.mark.skipif(
    not (CAMPAIGN.is_dir() and RAW.is_dir()),
    reason="CAMPAIGN_RESULTS_DIR not set to campaign results",
)


def _full_runs() -> list[Path]:
    """The controlled-grid runs: rq1, full sample (the n2 smoke runs are left out)."""
    return sorted(
        d
        for d in CAMPAIGN.glob("2026*_rq1_*")
        if d.is_dir() and ("_n200_" in d.name or "_n150_" in d.name)
    )


def _rows(run: Path):
    for path in sorted(run.glob("rq1_*.jsonl")):
        if "_judged" in path.name or "checkpoint" in path.name:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield path, json.loads(line)


@needs_campaign
def test_reference_matches_every_stamp_in_the_campaign():
    runs = _full_runs()
    # 2 model families x 4 workloads x 3 runs
    assert len(runs) == 24
    stamped = unstamped = 0
    for run in runs:
        dataset = next(d for d in DATASETS if d in run.name)
        reference = hop_reference(dataset, RAW)
        for path, row in _rows(run):
            qid = str(row["id"])
            assert qid in reference, f"{path.name}: {qid} not in the raw dataset"
            stamp = (row.get("question_difficulty") or {}).get("hop_count")
            if stamp is None:
                unstamped += 1
                continue
            stamped += 1
            assert int(stamp) == reference[qid], f"{path.name}: {qid}"
    assert stamped > 9000
    # Only error rows lack a stamp in these runs.
    assert unstamped < 100


@needs_campaign
def test_2wiki_full_sample_is_150_two_hop_and_50_four_hop():
    reference = hop_reference("2wikimultihopqa", RAW)
    for run in _full_runs():
        if "2wikimultihopqa" not in run.name:
            continue
        ids = {str(r["id"]) for path, r in _rows(run) if path.stem == "rq1_naive_rag"}
        assert len(ids) == 200
        assert Counter(reference[i] for i in ids) == {2: 150, 4: 50}, run.name
