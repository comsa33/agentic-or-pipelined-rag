"""Batch judging must send what sync judging sends and record what it records.

The Sonnet runs are judged through the OpenAI Batch API, but their verdicts
are compared with rows judged synchronously, and `verify_campaign.py
--require-judge` reads both the same way. Nothing here reaches the network:
the judge's HTTP requests are answered by a fake server, and the Batch API by
a fake client that runs each submitted batch through that same server.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import pytest

judge_module = pytest.importorskip("scripts.run_llm_judge", reason="requires scripts package")

from experiments.verify import OK, verify_judged

JUDGE = "openai/gpt-4.1-nano"
TAG = "gpt-4-1-nano_7bcf05"

# One row per behaviour of the judge: a correct and a wrong verdict, a
# truncated completion, and a reply ChatAdapter cannot parse, which makes
# DSPy retry the row with JSONAdapter — a second request with another body.
ROWS = [
    {"id": "q1", "question": "Capital of France?", "reference": "Paris", "prediction": "right"},
    {"id": "q2", "question": "Capital of Italy?", "reference": "Rome", "prediction": "wrong"},
    {"id": "q3", "question": "Longest river?", "reference": "Nile", "prediction": "long"},
    {"id": "q4", "question": "Highest peak?", "reference": "Everest", "prediction": "garbled"},
    {"id": "q4", "question": "Highest peak?", "reference": "Everest", "prediction": "right"},
]


def _prediction(body: dict) -> str:
    user = next(m["content"] for m in body["messages"] if m["role"] == "user")
    return user.split("predicted_answer")[1].split("\n")[1].strip(' "')


def serve(body: dict) -> dict:
    """What the judge model answers to one request body."""
    prediction = _prediction(body)
    json_mode = "response_format" in body
    finish = "stop"
    if prediction == "long":
        content, finish = "", "length"
    elif prediction == "garbled" and not json_mode:
        content = "no fields here"
    else:
        verdict = prediction in ("right", "garbled")
        content = (
            json.dumps({"is_correct": verdict})
            if json_mode
            else f"[[ ## is_correct ## ]]\n{verdict}\n\n[[ ## completed ## ]]"
        )
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4.1-nano-2025-04-14",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class FakeOpenAI:
    """The Batch API surface BatchJudge uses, answering through `serve`."""

    def __init__(self, fail: set[str] | None = None, status: str = "completed") -> None:
        self.fail = fail or set()
        self.status = status
        self.inputs: dict[str, str] = {}
        self.contents: dict[str, str] = {}
        self.batches_by_id: dict[str, SimpleNamespace] = {}
        self.created = 0
        self.interrupt_on: str | None = None
        self.files = SimpleNamespace(create=self._create_file, content=self._content)
        self.batches = SimpleNamespace(create=self._create_batch, retrieve=self._retrieve)

    def _create_file(self, file, purpose):
        assert purpose == "batch"
        name, data = file
        assert name.endswith(".jsonl")
        file_id = f"file-{len(self.inputs)}"
        self.inputs[file_id] = data.decode("utf-8")
        return SimpleNamespace(id=file_id)

    def _content(self, file_id):
        return SimpleNamespace(text=self.contents[file_id])

    def _create_batch(self, input_file_id, endpoint, completion_window):
        assert endpoint == "/v1/chat/completions"
        self.created += 1
        batch_id = f"batch-{self.created}"
        out, err = [], []
        for line in self.inputs[input_file_id].splitlines():
            req = json.loads(line)
            cid = req["custom_id"]
            if cid in self.fail:
                self.fail.discard(cid)  # fails once, as a transient error would
                err.append({"custom_id": cid, "response": {"status_code": 500, "body": {}}})
            elif self.status == "completed":
                resp = {"status_code": 200, "body": serve(req["body"])}
                out.append({"id": "r", "custom_id": cid, "response": resp, "error": None})
            # An expired batch leaves unfinished requests out of both files.
        self.contents[f"{batch_id}-out"] = "".join(json.dumps(r) + "\n" for r in out)
        self.contents[f"{batch_id}-err"] = "".join(json.dumps(r) + "\n" for r in err)
        self.batches_by_id[batch_id] = SimpleNamespace(
            id=batch_id,
            status=self.status,
            output_file_id=f"{batch_id}-out" if out else None,
            error_file_id=f"{batch_id}-err" if err else None,
            request_counts=None,
        )
        return SimpleNamespace(id=batch_id)

    def _retrieve(self, batch_id):
        if self.interrupt_on == batch_id:
            self.interrupt_on = None
            raise KeyboardInterrupt
        return self.batches_by_id[batch_id]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Route every judge request through the batch transport, never the network."""
    import litellm

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # A client litellm built before the transport was installed would bypass it.
    litellm.in_memory_llm_clients_cache.flush_cache()
    transport = judge_module._batch_transport()
    litellm.in_memory_llm_clients_cache.flush_cache()
    transport.calls = 0
    return transport


def _judge_sync(run_dir: Path) -> tuple[Path, dict, list[dict]]:
    """Judge synchronously with `serve` answering, recording each body sent."""
    sent: list[dict] = []

    def handle(self, request):
        self.calls += 1
        body = json.loads(request.content)
        sent.append(body)
        return httpx.Response(200, json=serve(body))

    path = _write_results(run_dir)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(judge_module._BatchTransport, "handle", handle)
        summary = judge_module.judge_jsonl(path, judge_model=JUDGE)
    return path, summary, sent


def _write_results(run_dir: Path, rows=ROWS) -> Path:
    run_dir.mkdir(parents=True)
    path = run_dir / "rq1_agentic_(react).jsonl"
    path.write_text(
        "".join(json.dumps({**r, "pipeline": "agentic_(react)"}) + "\n" for r in rows),
        encoding="utf-8",
    )
    return path


def _batch_inputs(client: FakeOpenAI) -> list[dict]:
    return [json.loads(line) for text in client.inputs.values() for line in text.splitlines()]


def test_batch_requests_are_the_bodies_sync_sends(tmp_path, offline):
    _, _, sync_bodies = _judge_sync(tmp_path / "sync")
    assert offline.calls == len(sync_bodies)

    client = FakeOpenAI()
    path = _write_results(tmp_path / "batch")
    judge_module.BatchJudge(tmp_path / "batch" / "state", JUDGE, client=client).run([path])

    inputs = _batch_inputs(client)
    # Same bodies, one per request sync makes; batches group them by round.
    canon = lambda bodies: sorted(json.dumps(b, sort_keys=True) for b in bodies)  # noqa: E731
    assert canon(r["body"] for r in inputs) == canon(sync_bodies)
    assert len(sync_bodies) == len(ROWS) + 1  # q4's unparsable reply is retried once
    assert all(r["url"] == "/v1/chat/completions" and r["method"] == "POST" for r in inputs)
    assert set(sync_bodies[0]) == {"model", "messages", "max_tokens", "temperature"}
    assert sync_bodies[0]["model"] == "gpt-4.1-nano"
    # The JSONAdapter retry could only be known once the first answer was back.
    assert client.created == 2
    assert len({r["custom_id"] for r in inputs}) == len(inputs)


def test_batch_rows_and_summary_match_sync_byte_for_byte(tmp_path):
    sync_path, sync_summary, _ = _judge_sync(tmp_path / "sync")

    path = _write_results(tmp_path / "batch")
    [batch_summary] = judge_module.BatchJudge(
        tmp_path / "batch" / "state", JUDGE, client=FakeOpenAI()
    ).run([path])

    name = f"{path.stem}_judged_{TAG}.jsonl"
    sync_bytes = (sync_path.parent / name).read_bytes()
    assert (path.parent / name).read_bytes() == sync_bytes
    assert batch_summary == sync_summary
    statuses = [json.loads(line)["judge_status"] for line in sync_bytes.splitlines()]
    assert statuses == ["ok", "ok", "truncated", "ok", "ok"]
    # Stub verdicts are False; a leaked stub would turn these into zeros.
    assert batch_summary["judge_accuracy"] == pytest.approx(3 / 5)
    assert all(c.status == OK for c in verify_judged(path.parent, TAG))


def _judged_rows(path: Path) -> list[dict]:
    judged = path.parent / f"{path.stem}_judged_{TAG}.jsonl"
    return [json.loads(line) for line in judged.read_text().splitlines()]


def test_a_failed_request_is_sent_again_in_the_next_round(tmp_path):
    path = _write_results(tmp_path / "run", ROWS[:2])
    probe = FakeOpenAI()
    judge_module.BatchJudge(tmp_path / "probe", JUDGE, client=probe).run([path])
    first_cid = json.loads(next(iter(probe.inputs.values())).splitlines()[0])["custom_id"]
    (path.parent / f"{path.stem}_judged_{TAG}.jsonl").unlink()

    client = FakeOpenAI(fail={first_cid})
    judge_module.BatchJudge(tmp_path / "state", JUDGE, client=client).run([path])
    assert client.created == 2
    assert [r["judge_status"] for r in _judged_rows(path)] == ["ok", "ok"]
    assert [r["llm_judge"] for r in _judged_rows(path)] == [1.0, 0.0]


def test_rows_never_answered_are_errors_that_the_same_command_redoes(tmp_path):
    path = _write_results(tmp_path / "run", ROWS[:3])
    state = tmp_path / "run" / "state"
    client = FakeOpenAI(status="expired")
    judge_module.BatchJudge(state, JUDGE, client=client).run([path])
    assert client.created == judge_module.BATCH_MAX_ROUNDS
    assert [(r["llm_judge"], r["judge_status"]) for r in _judged_rows(path)] == [(0.0, "error")] * 3
    assert all(c.status == OK for c in verify_judged(path.parent, TAG))

    # Same state directory: the expired requests are resubmitted, not
    # answered from the earlier rounds' failures.
    client.status = "completed"
    [summary] = judge_module.BatchJudge(state, JUDGE, client=client).run([path])
    assert client.created == judge_module.BATCH_MAX_ROUNDS + 1
    rows = _judged_rows(path)
    assert [r["judge_status"] for r in rows] == ["ok", "ok", "truncated"]
    assert summary["unusable"] == 1


def test_interrupted_run_resumes_the_same_batch(tmp_path):
    path = _write_results(tmp_path / "run", ROWS[:2])
    state = tmp_path / "run" / "state"
    client = FakeOpenAI()
    client.interrupt_on = "batch-1"
    with pytest.raises(KeyboardInterrupt):
        judge_module.BatchJudge(state, JUDGE, client=client).run([path])
    assert client.created == 1
    assert json.loads((state / "state.json").read_text())["rounds"][0]["batch_id"] == "batch-1"

    [summary] = judge_module.BatchJudge(state, JUDGE, client=client).run([path])
    assert client.created == 1, "resume submitted a second batch"
    assert summary["judge_accuracy"] == pytest.approx(1 / 2)


def test_interrupted_second_round_resumes_too(tmp_path):
    """Interrupted while waiting for the JSONAdapter retries of round 0."""
    sync_path, sync_summary, _ = _judge_sync(tmp_path / "sync")
    path = _write_results(tmp_path / "run")
    state = tmp_path / "run" / "state"
    client = FakeOpenAI()
    client.interrupt_on = "batch-2"
    with pytest.raises(KeyboardInterrupt):
        judge_module.BatchJudge(state, JUDGE, client=client).run([path])
    assert client.created == 2

    [summary] = judge_module.BatchJudge(state, JUDGE, client=client).run([path])
    assert client.created == 2, "resume submitted another batch"
    name = f"{path.stem}_judged_{TAG}.jsonl"
    assert (path.parent / name).read_bytes() == (sync_path.parent / name).read_bytes()
    assert summary == sync_summary


def test_batch_id_reattaches_when_state_is_lost(tmp_path):
    path = _write_results(tmp_path / "run", ROWS[:2])
    client = FakeOpenAI()
    client.interrupt_on = "batch-1"
    with pytest.raises(KeyboardInterrupt):
        judge_module.BatchJudge(tmp_path / "state", JUDGE, client=client).run([path])

    other = tmp_path / "elsewhere"
    [summary] = judge_module.BatchJudge(other, JUDGE, client=client).run([path], batch_id="batch-1")
    assert client.created == 1
    assert summary["status"] == "complete" and summary["unusable"] == 0


def test_batch_artifacts_are_not_result_files(tmp_path):
    path = _write_results(tmp_path / "run", ROWS[:1])
    state = path.parent / f"judge_batch_{TAG}"
    judge_module.BatchJudge(state, JUDGE, client=FakeOpenAI()).run([path])
    assert not list(state.glob("*.jsonl"))
    assert sorted(p.name for p in path.parent.glob("*.jsonl")) == [
        path.name,
        f"{path.stem}_judged_{TAG}.jsonl",
    ]


def test_batch_id_unknown_to_existing_state_is_refused(tmp_path):
    path = _write_results(tmp_path / "run", ROWS[:1])
    client = FakeOpenAI()
    client.interrupt_on = "batch-1"
    with pytest.raises(KeyboardInterrupt):
        judge_module.BatchJudge(tmp_path / "state", JUDGE, client=client).run([path])
    with pytest.raises(ValueError, match="other batches"):
        judge_module.BatchJudge(tmp_path / "state", JUDGE, client=client).run(
            [path], batch_id="batch-9"
        )


def test_a_judge_outside_openai_is_refused_before_submitting(tmp_path, monkeypatch):
    path = _write_results(tmp_path / "run", ROWS[:1])
    client = FakeOpenAI()
    monkeypatch.setenv("OPENAI_API_BASE", "https://proxy.example/v1")
    with pytest.raises(ValueError, match="only covers OpenAI"):
        judge_module.BatchJudge(tmp_path / "state", "openai/gpt-4.1-nano", client=client).run(
            [path]
        )
    assert client.created == 0


def test_state_for_another_judge_is_refused(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "state.json").write_text(json.dumps({"judge_model": "openai/gpt-4o", "rounds": []}))
    with pytest.raises(ValueError, match="belongs to"):
        judge_module.BatchJudge(state, JUDGE, client=FakeOpenAI())


def test_batch_without_an_api_key_exits_before_anything_runs(tmp_path, monkeypatch):
    path = _write_results(tmp_path / "run", ROWS[:1])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["run_llm_judge.py", str(path), "--batch"])
    monkeypatch.setattr(
        judge_module, "BatchJudge", lambda *a, **k: pytest.fail("batch judge was started")
    )
    with pytest.raises(SystemExit) as exc:
        judge_module.main()
    assert exc.value.code not in (0, None)
    assert "OPENAI_API_KEY" in str(exc.value.code)
    assert not list(path.parent.glob("*_judged_*"))


def test_a_none_prediction_is_judged_as_empty(tmp_path):
    """Rows saved by earlier code versions can carry None where a refusal stood."""

    def judge(run_dir: Path, prediction) -> tuple[list[dict], bytes, dict]:
        sent: list[dict] = []

        def handle(self, request):
            body = json.loads(request.content)
            sent.append(body)
            return httpx.Response(200, json=serve(body))

        row = {"id": "q1", "question": "Capital of France?", "reference": "Paris"}
        path = _write_results(run_dir, [{**row, "prediction": prediction}])
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(judge_module._BatchTransport, "handle", handle)
            summary = judge_module.judge_jsonl(path, judge_model=JUDGE)
        return sent, (path.parent / f"{path.stem}_judged_{TAG}.jsonl").read_bytes(), summary

    none_sent, none_rows, none_summary = judge(tmp_path / "none", None)
    empty_sent, empty_rows, empty_summary = judge(tmp_path / "empty", "")
    assert none_sent == empty_sent
    assert none_rows == empty_rows
    assert json.loads(none_rows)["prediction"] == ""
    assert {k: v for k, v in none_summary.items() if "file" not in k} == {
        k: v for k, v in empty_summary.items() if "file" not in k
    }

    path = _write_results(tmp_path / "batch", [{"id": "q1", "prediction": None}])
    assert judge_module._load_items(path)[0]["prediction"] == ""
