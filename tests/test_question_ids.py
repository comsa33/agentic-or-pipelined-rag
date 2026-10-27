"""Smoke runs on named questions: never in pilot or grid, never mixed with seeded samples."""

from __future__ import annotations

import json

import pytest

import experiments.run as run_mod
from agentic_rag.config.settings import settings
from experiments.manifest import (
    load_question_set,
    make_run_key,
    plan_run,
    select_questions,
)

DATASET = [{"id": f"q{i}", "question": f"question {i}"} for i in range(5)]


def _ids_file(tmp_path, data: dict):
    path = tmp_path / "hardest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_question_set_records_file_hashes_and_order(tmp_path):
    path = _ids_file(tmp_path, {"hotpotqa": ["q3", "q0", "q4"], "musique": ["m1"]})
    qs = load_question_set(path, "hotpotqa")
    assert qs["ids"] == ["q3", "q0", "q4"]
    assert qs["file"] == str(path.resolve())
    assert len(qs["file_sha256"]) == 64 and len(qs["ids_sha256"]) == 64
    assert qs["key"] == qs["ids_sha256"][:8]
    # Another order is another run.
    other = load_question_set(_ids_file(tmp_path, {"hotpotqa": ["q0", "q3", "q4"]}), "hotpotqa")
    assert other["key"] != qs["key"]


def test_question_set_must_name_the_dataset_once_each(tmp_path):
    with pytest.raises(ValueError, match="no questions"):
        load_question_set(_ids_file(tmp_path, {"musique": ["m1"]}), "hotpotqa")
    with pytest.raises(ValueError, match="twice"):
        load_question_set(_ids_file(tmp_path, {"hotpotqa": ["q1", "q1"]}), "hotpotqa")


def test_selection_keeps_order_and_refuses_unknown_ids():
    assert [q["id"] for q in select_questions(DATASET, ["q4", "q1"])] == ["q4", "q1"]
    with pytest.raises(ValueError, match="q9"):
        select_questions(DATASET, ["q1", "q9"])


def test_named_questions_are_keyed_apart_from_a_sample():
    sampled = make_run_key("rq1", "hotpotqa", "claude-sonnet-5-5", 3, None)
    named = make_run_key("rq1", "hotpotqa", "claude-sonnet-5-5", 3, None, "abcd1234")
    assert named != sampled
    assert named.split("|")[1] == "hotpotqa@qabcd1234", "kept out of the sample's repeat set"


def test_run_id_carries_the_question_set(tmp_path):
    plan = plan_run(
        tmp_path,
        stem="rq1",
        dataset="hotpotqa",
        model_tag="claude-sonnet-5-5",
        n=3,
        question_set_hash="abcd1234",
    )
    assert plan.run_id.endswith("_n3_claude-sonnet-5-5_qabcd1234")
    assert "@qabcd1234" in plan.run_key


@pytest.fixture
def _dataset(monkeypatch):
    monkeypatch.setattr(run_mod, "load_dataset", lambda name, sample_size=None: list(DATASET))


def test_smoke_runs_the_named_questions(tmp_path, monkeypatch, _dataset):
    monkeypatch.setattr(settings.experiment, "campaign_phase", "smoke")
    path = _ids_file(tmp_path, {"hotpotqa": ["q2", "q0"]})
    dataset, qs = run_mod._load_questions("hotpotqa", None, str(path))
    assert [q["id"] for q in dataset] == ["q2", "q0"]
    assert qs["ids"] == ["q2", "q0"]


@pytest.mark.parametrize("phase", ["pilot", "grid", None])
def test_named_questions_are_refused_outside_smoke(tmp_path, monkeypatch, _dataset, phase):
    monkeypatch.setattr(settings.experiment, "campaign_phase", phase)
    path = _ids_file(tmp_path, {"hotpotqa": ["q2"]})
    with pytest.raises(ValueError, match="smoke runs only"):
        run_mod._load_questions("hotpotqa", None, str(path))


def test_named_questions_and_sample_are_exclusive(tmp_path, monkeypatch, _dataset):
    monkeypatch.setattr(settings.experiment, "campaign_phase", "smoke")
    path = _ids_file(tmp_path, {"hotpotqa": ["q2"]})
    with pytest.raises(ValueError, match="--sample"):
        run_mod._load_questions("hotpotqa", 2, str(path))


def test_unknown_id_stops_before_the_run(tmp_path, monkeypatch, _dataset):
    monkeypatch.setattr(settings.experiment, "campaign_phase", "smoke")
    path = _ids_file(tmp_path, {"hotpotqa": ["q2", "nope"]})
    with pytest.raises(ValueError, match="nope"):
        run_mod._load_questions("hotpotqa", None, str(path))


def test_without_named_questions_the_sample_is_unchanged(monkeypatch):
    calls = []
    monkeypatch.setattr(
        run_mod, "load_dataset", lambda name, sample_size=None: calls.append(sample_size) or []
    )
    monkeypatch.setattr(settings.experiment, "campaign_phase", "grid")
    assert run_mod._load_questions("hotpotqa", 200, None) == ([], None)
    assert calls == [200]
