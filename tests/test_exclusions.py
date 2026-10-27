"""exclusions.json gains a model dimension; the dataset-only layout must still read the same."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from experiments.exclusions import Exclusion, excluded_ids, load_exclusions

SCHEMA = {"description": "test"}
ENTRIES = {
    "2wikimultihopqa": [{"id": "2wiki_a", "reason": "azure_content_filter", "pipeline": "crag"}],
    "hotpotqa": [
        {"id": "hp_a", "reason": "azure_content_filter"},
        {"id": "hp_b", "reason": "azure_content_filter"},
    ],
}


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_dataset_only_layout_still_reads_for_gpt5_mini(tmp_path):
    ex = load_exclusions(_write(tmp_path, "old.json", {"_schema": SCHEMA, **ENTRIES}))
    assert {e.model for e in ex} == {None}
    assert excluded_ids(ex, "hotpotqa", "gpt-5-mini") == {"hp_a", "hp_b"}
    assert excluded_ids(ex, "musique", "gpt-5-mini") == set()


def test_entries_without_a_model_are_refused_for_claude(tmp_path):
    """gpt-5-mini's exclusions must not silently drop questions from Sonnet."""
    ex = load_exclusions(_write(tmp_path, "old.json", {"_schema": SCHEMA, **ENTRIES}))
    with pytest.raises(ValueError, match="no model"):
        excluded_ids(ex, "hotpotqa", "claude-sonnet-5-5")
    # A dataset with no unscoped entry is fine.
    assert excluded_ids(ex, "musique", "claude-sonnet-5-5") == set()


def test_migrating_under_a_model_leaves_that_model_unchanged(tmp_path):
    """Moving the existing entries under gpt-5-mini must not change its analysis."""
    old = load_exclusions(_write(tmp_path, "old.json", {"_schema": SCHEMA, **ENTRIES}))
    migrated = {"_schema": SCHEMA} | {ds: {"gpt-5-mini": v} for ds, v in ENTRIES.items()}
    new = load_exclusions(_write(tmp_path, "new.json", migrated))
    for ds in ("2wikimultihopqa", "hotpotqa", "musique", "financebench"):
        assert sorted(excluded_ids(new, ds, "gpt-5-mini")) == sorted(
            excluded_ids(old, ds, "gpt-5-mini")
        )
        assert excluded_ids(new, ds, "claude-sonnet-5-5") == set()


def test_model_named_in_the_entry(tmp_path):
    data = {"hotpotqa": [{"id": "hp_c", "model": "claude-sonnet-5-5", "reason": "refused"}]}
    ex = load_exclusions(_write(tmp_path, "e.json", data))
    assert ex == [
        Exclusion(dataset="hotpotqa", id="hp_c", model="claude-sonnet-5-5", reason="refused")
    ]
    assert excluded_ids(ex, "hotpotqa", "gpt-5-mini") == set()


def test_malformed_dataset_value_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="hotpotqa"):
        load_exclusions(_write(tmp_path, "bad.json", {"hotpotqa": "hp_a"}))


# Campaign results are not in the repository (see tests/test_analysis_runs.py).
CAMPAIGN_FILE = Path(os.environ.get("CAMPAIGN_RESULTS_DIR", "/nonexistent")) / "exclusions.json"


@pytest.mark.skipif(
    not CAMPAIGN_FILE.is_file(), reason="CAMPAIGN_RESULTS_DIR not set to campaign results"
)
def test_campaign_file_reads():
    ex = load_exclusions(CAMPAIGN_FILE)
    assert ex and all(e.id and e.dataset for e in ex)
