"""Run LLM-as-Judge on experiment result JSONL files.

Reads existing JSONL files, calls LLM-as-Judge for each prediction,
and saves results to a new file with _judged suffix.

Supports checkpoint/resume: skips already-judged items.

Usage:
    # Single run directory (the paper's judge is gpt-4.1)
    uv run python scripts/run_llm_judge.py data/results-campaign/<run_dir>/ --judge-model openai/gpt-4.1

    # All rq1 directories
    uv run python scripts/run_llm_judge.py --pattern "rq1" --judge-model openai/gpt-4.1

    # Specific JSONL file only
    uv run python scripts/run_llm_judge.py "data/results-campaign/<run_dir>/rq1_agentic_(react).jsonl" --judge-model openai/gpt-4.1

    # Through the OpenAI Batch API: same requests, rows and summary as above.
    # State is kept in <dir>/judge_batch_<tag>/; rerun the same command to resume.

The API key must be in the shell environment (e.g. `set -a; source .env; set +a`):
settings read .env but do not export it, and without OPENAI_API_KEY every call
fails and is recorded as an "error" verdict.
    uv run python scripts/run_llm_judge.py data/results-campaign/<run_dir>/ --judge-model openai/gpt-4.1 --batch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import httpx

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from agentic_rag.evaluation.metrics import llm_judge_correctness
from experiments.common import _prediction_text, setup_experiment


def _judge_suffix(judge_model: str | None, *, legacy: bool = False) -> str:
    """Filename suffix identifying which judge produced a set of verdicts.

    The readable part drops the provider prefix and flattens ":" and "." so the
    name survives a filesystem, which means distinct models can flatten to the
    same tag — "openai/qwen3-5" and "ollama_cloud/qwen3.5" both become
    "qwen3-5". Two judges sharing a suffix is worse than an ugly name: the
    second run resumes from the first one's checkpoint, skips every item it
    thinks is done, and overwrites its summary. A short digest of the full
    model string is appended so the suffix is unique even when the tag is not.

    Falls back to plain "_judged" when no model is given, so files written by
    earlier runs keep their names.
    """
    if not judge_model:
        return "_judged"
    tag = judge_model.split("/")[-1].replace(":", "-").replace(".", "-")
    if legacy:
        return f"_judged_{tag}"
    digest = hashlib.sha256(judge_model.encode()).hexdigest()[:6]
    return f"_judged_{tag}_{digest}"


def _judged_path(base: Path, stem: str, judge_model: str | None, ext: str) -> Path:
    """Artifact path for this judge, reusing a digest-less file already on disk.

    The digest was added after a first round of judging had been written, and
    those verdicts cost real API calls. Renaming them is not safe either: the
    summaries never recorded the model string, so the digest for an existing
    file cannot be recomputed. Preferring an existing legacy name keeps that
    round addressable while every new judge gets a collision-free one.
    """
    legacy = base / f"{stem}{_judge_suffix(judge_model, legacy=True)}{ext}"
    if legacy.exists():
        return legacy
    return base / f"{stem}{_judge_suffix(judge_model)}{ext}"


def _key(item: dict) -> tuple[str, str]:
    """Identity of a judged row.

    A results file can hold the same question once per pipeline, so the id on
    its own is not unique within a file — the judge panel carries 200 rows over
    184 ids. Pairing it with the pipeline keeps those rows distinct.
    """
    return (str(item.get("pipeline", "")), str(item["id"]))


def _load_items(jsonl_path: Path) -> list[dict]:
    items = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            # A refusal that stood was checkpointed as None by earlier code versions;
            # judge it as the empty answer it is, not as the text "None".
            if "prediction" in item:
                item["prediction"] = _prediction_text(item["prediction"])
            items.append(item)
    return items


def _load_checkpoint(jsonl_path: Path, output_path: Path) -> dict:
    """Verdicts already on disk, keyed by _key, with unusable ones removed.

    A verdict that came from a truncated or failed completion is not a
    judgement — drop it so this run redoes it. That is what makes a larger
    ceiling recoverable: rerun with more room and only the bad verdicts are
    re-scored, everything else is skipped.

    Keyed on (pipeline, id): a file may hold the same question once per
    pipeline, so id alone collapses distinct rows into one. The judge panel is
    exactly that shape — 200 rows over 184 ids — and keying on id would drop 16
    of them on the first resume.
    """
    judged, kept, retry = {}, [], 0
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("judge_status", "ok") != "ok":
                    retry += 1
                    continue
                judged[_key(item)] = item
                kept.append(item)
    if retry:
        logger.warning(f"[{jsonl_path.name}] {retry} verdicts were not usable — re-judging them")
        # Rewrite without them before appending, or the file would carry both the
        # discarded verdict and its replacement and stop satisfying "one row per
        # question". Written from the row list, not the lookup, so rows sharing an
        # id survive.
        with open(output_path, "w", encoding="utf-8") as f:
            for item in kept:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return judged


def _skipped_summary(jsonl_path: Path, items: list[dict], judged: dict) -> dict:
    logger.info(f"[{jsonl_path.name}] Already complete, skipping")
    scores = [judged[_key(i)]["llm_judge"] for i in items if _key(i) in judged]
    return {
        "file": jsonl_path.name,
        "n": len(items),
        "judge_accuracy": sum(scores) / len(scores) if scores else 0.0,
        "status": "skipped",
    }


def _judged_row(item: dict, score: float, status: str) -> dict:
    # judge_status records why the verdict ended. Anything but "ok" is
    # not a judgement, and a later run re-scores it rather than trusting
    # the 0.0 that a truncated completion would otherwise leave behind.
    return {
        "id": item["id"],
        "question": item.get("question", ""),
        "reference": item.get("reference", ""),
        "prediction": item.get("prediction", ""),
        "llm_judge": score,
        "judge_status": status,
        "pipeline": item.get("pipeline", ""),
    }


def _complete_summary(
    jsonl_path: Path, output_path: Path, items: list[dict], scores: list[float]
) -> dict:
    accuracy = sum(scores) / len(scores) if scores else 0.0
    # Count from the file rather than this run: earlier passes may have left
    # unusable verdicts behind, and what matters is whether any remain.
    unusable = 0
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            if line.strip() and json.loads(line).get("judge_status", "ok") != "ok":
                unusable += 1
    logger.info(f"[{jsonl_path.name}] Done: judge_accuracy={accuracy:.3f}, unusable={unusable}")
    if unusable:
        logger.warning(
            f"[{jsonl_path.name}] {unusable} verdicts still unusable — rerun to re-score "
            f"just those; raise LLM_MAX_TOKENS first if they are truncated rather than unparsed"
        )

    return {
        "file": jsonl_path.name,
        "n": len(items),
        "judge_accuracy": accuracy,
        "unusable": unusable,
        "status": "complete",
    }


def judge_jsonl(jsonl_path: Path, delay: float = 0.0, judge_model: str | None = None) -> dict:
    """Run LLM-as-Judge on a single JSONL file.

    Returns summary dict with judge accuracy.
    """
    # Tag the output with the judge model. Multiple judges are run over the
    # same results to check that conclusions survive a change of judge, and a
    # shared filename would make each run silently overwrite the last.
    output_path = _judged_path(jsonl_path.parent, jsonl_path.stem, judge_model, ".jsonl")

    items = _load_items(jsonl_path)
    judged = _load_checkpoint(jsonl_path, output_path)

    logger.info(
        f"[{jsonl_path.name}] {len(items)} items, "
        f"{len(judged)} already judged, "
        f"{len(items) - len(judged)} remaining"
    )

    if len(judged) >= len(items):
        return _skipped_summary(jsonl_path, items, judged)

    # Open output in append mode
    scores = []
    with open(output_path, "a", encoding="utf-8") as out:
        for i, item in enumerate(items):
            if _key(item) in judged:
                scores.append(judged[_key(item)]["llm_judge"])
                continue

            prediction = item.get("prediction", "")
            reference = item.get("reference", "")
            question = item.get("question", "")

            try:
                score, status = llm_judge_correctness(
                    prediction, reference, question, judge_model=judge_model, with_status=True
                )
            except Exception as e:
                logger.warning(f"  [{i + 1}/{len(items)}] Error: {e}")
                score, status = 0.0, "error"

            if status != "ok":
                logger.warning(
                    f"  [{i + 1}/{len(items)}] verdict unusable ({status}) — id={item['id']}"
                )
            scores.append(score)

            out.write(json.dumps(_judged_row(item, score, status), ensure_ascii=False) + "\n")
            out.flush()

            if (i + 1) % 20 == 0:
                running_acc = sum(scores) / len(scores)
                logger.info(f"  [{i + 1}/{len(items)}] running judge_acc={running_acc:.3f}")

            if delay > 0:
                time.sleep(delay)

    return _complete_summary(jsonl_path, output_path, items, scores)


# ---------------------------------------------------------------------------
# Batch mode (OpenAI Batch API)
#
# The judge's prompt, parameters and parsing live in llm_judge_correctness,
# through DSPy and litellm. Rebuilding the request by hand would be a second
# copy of all three that can drift, so batch mode runs that same function
# behind an HTTP transport instead. Each pass sends every pending row through
# it: a request the transport has an answer for gets that answer back, and one
# it has not seen is recorded for the next batch and answered with a stub
# whose verdict is thrown away. The first pass therefore collects the bodies
# the synchronous path would send, and once the batch is back the same pass
# replays its responses into the same parser. A row that needs a request
# nobody has answered yet — DSPy retries a failed parse with JSONAdapter, a
# different body — is held back and its request goes into another round.
# ---------------------------------------------------------------------------

BATCH_ENDPOINT = "/v1/chat/completions"
BATCH_TERMINAL = {"completed", "failed", "expired", "cancelled"}
BATCH_MAX_ROUNDS = 3


def _body_digest(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _custom_id(source: Path, item: dict, ordinal: int, body: dict) -> str:
    """Batch request id for one request of one row.

    Built from the row (file, pipeline, id), the request's position in that
    row's run and the body itself. Rows are never merged even when their
    bodies match — sync mode makes one call per row. Including the body
    means a replayed request that differs from the recorded one does not
    match its answer and is sent again instead.
    """
    pipeline, qid = _key(item)
    # Resolved, so a rerun from another directory derives the same ids.
    raw = f"{source.resolve()}|{pipeline}|{qid}|{ordinal}|{_body_digest(body)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _stub_completion(body: dict) -> dict:
    """A completion that parses, so the judge stops without another request."""
    if "response_format" in body:
        content = '{"is_correct": false}'
    else:
        content = "[[ ## is_correct ## ]]\nFalse\n\n[[ ## completed ## ]]"
    return {
        "id": "batch-stub",
        "object": "chat.completion",
        "created": 0,
        "model": body.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class _BatchTransport:
    """Answers the judge's HTTP requests from batch results.

    litellm builds its OpenAI client once and keeps the transport it was
    given, so one instance is installed per process and switched between
    rows through `begin`, never replaced.
    """

    def __init__(self) -> None:
        self.answers: dict[str, dict] = {}
        self.unanswered: dict[str, dict] = {}
        self.misrouted: set[str] = set()
        self.calls = 0
        self.begin(Path(), {})

    def begin(self, source: Path, item: dict) -> None:
        self.source, self.item = source, item
        self.ordinal = 0
        # Set once a request had no answer; the row is then held for a later round.
        self.pending = False

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        body = json.loads(request.content)
        if request.url.host != "api.openai.com" or request.url.path != BATCH_ENDPOINT:
            # Raising here would go through litellm's retries and DSPy's
            # fallback and end as an "error" row; run() refuses instead.
            self.misrouted.add(str(request.url))
            return httpx.Response(200, json=_stub_completion(body))
        ordinal, self.ordinal = self.ordinal, self.ordinal + 1
        if self.pending:
            # Whatever follows a stub is a reaction to the stub, not to the judge.
            return httpx.Response(200, json=_stub_completion(body))
        cid = _custom_id(self.source, self.item, ordinal, body)
        answer = self.answers.get(cid)
        if answer is None:
            self.unanswered[cid] = body
            self.pending = True
            return httpx.Response(200, json=_stub_completion(body))
        return httpx.Response(200, json=answer)


_TRANSPORT: _BatchTransport | None = None


def _batch_transport() -> _BatchTransport:
    """Route every litellm request in this process through the batch transport."""
    import dspy

    global _TRANSPORT
    # A cached stub would be served back in place of the real answer.
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    if _TRANSPORT is None:
        import litellm

        _TRANSPORT = _BatchTransport()
        litellm.client_session = httpx.Client(
            transport=httpx.MockTransport(lambda request: _TRANSPORT.handle(request))
        )
    return _TRANSPORT


def _answers_from(text: str) -> dict[str, dict]:
    """Completions from batch output lines, by custom_id.

    Only requests that succeeded count as answered. One that failed, or that
    an expired or cancelled batch never ran, stays unanswered and is sent
    again in the next round, as sync mode retries a failed call.
    """
    out = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        response = row.get("response") or {}
        if response.get("status_code") == 200 and response.get("body"):
            out[row["custom_id"]] = response["body"]
    return out


class BatchJudge:
    """Judge result files through the OpenAI Batch API.

    State lives in `state_dir`: state.json names each round's batch and the
    requests it carries, and each round's results are downloaded once beside
    it. A rerun reads that state and picks up the open batch rather than
    submitting a new one, so an interrupted run loses nothing.
    """

    def __init__(
        self,
        state_dir: Path,
        judge_model: str,
        *,
        client=None,
        poll_interval: float = 60.0,
        max_rounds: int = BATCH_MAX_ROUNDS,
    ) -> None:
        self.state_dir = state_dir
        self.judge_model = judge_model
        self._client = client
        self.poll_interval = poll_interval
        self.max_rounds = max_rounds
        self.state_path = state_dir / "state.json"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state["judge_model"] != judge_model:
                raise ValueError(
                    f"{self.state_path} belongs to {self.state['judge_model']}, not {judge_model}"
                )
        else:
            self.state = {"judge_model": judge_model, "rounds": []}

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def _save(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.state_path)

    def _replay(self, work: list[tuple[Path, dict]]) -> tuple[dict, dict]:
        """Run every row through the judge; return verdicts and unanswered requests."""
        transport = _batch_transport()
        transport.answers = self._answers()
        transport.unanswered = {}
        transport.misrouted = set()
        verdicts = {}
        for source, item in work:
            transport.begin(source, item)
            try:
                score, status = llm_judge_correctness(
                    item.get("prediction", ""),
                    item.get("reference", ""),
                    item.get("question", ""),
                    judge_model=self.judge_model,
                    with_status=True,
                )
            except Exception as e:
                logger.warning(f"  [{source.name}] Error: {e}")
                score, status = 0.0, "error"
            if not transport.pending:
                verdicts[(str(source), _key(item))] = (score, status)
        if transport.misrouted:
            raise ValueError(
                f"batch mode only covers OpenAI chat completions, not {sorted(transport.misrouted)}"
            )
        return verdicts, dict(transport.unanswered)

    def _answers(self) -> dict[str, dict]:
        answers: dict[str, dict] = {}
        for rnd in self.state["rounds"]:
            if rnd.get("results"):
                text = (self.state_dir / rnd["results"]).read_text(encoding="utf-8")
                answers.update(_answers_from(text))
        return answers

    def _submit(self, index: int, requests: dict[str, dict]) -> dict:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # Kept out of *.jsonl so result-file globs never mistake it for results.
        name = f"round{index}_input.ndjson"
        lines = "".join(
            json.dumps(
                {"custom_id": cid, "method": "POST", "url": BATCH_ENDPOINT, "body": body},
                ensure_ascii=False,
            )
            + "\n"
            for cid, body in requests.items()
        )
        (self.state_dir / name).write_text(lines, encoding="utf-8")
        uploaded = self.client.files.create(
            file=(f"round{index}_input.jsonl", lines.encode("utf-8")), purpose="batch"
        )
        batch = self.client.batches.create(
            input_file_id=uploaded.id, endpoint=BATCH_ENDPOINT, completion_window="24h"
        )
        rnd = {"batch_id": batch.id, "input": name, "custom_ids": sorted(requests), "results": None}
        self.state["rounds"].append(rnd)
        self._save()
        logger.info(f"Round {index}: submitted {len(requests)} requests as batch {batch.id}")
        return rnd

    def _collect(self, index: int, rnd: dict) -> None:
        """Wait for a round's batch and download its results."""
        while True:
            batch = self.client.batches.retrieve(rnd["batch_id"])
            counts = getattr(batch, "request_counts", None)
            logger.info(f"Round {index}: batch {batch.id} {batch.status} {counts or ''}")
            if batch.status in BATCH_TERMINAL:
                break
            time.sleep(self.poll_interval)
        if batch.status == "failed":
            # Nothing ran; forget the round so a rerun submits it again.
            self.state["rounds"].remove(rnd)
            self._save()
            raise RuntimeError(f"batch {batch.id} failed: {getattr(batch, 'errors', None)}")
        text = ""
        for file_id in (batch.output_file_id, batch.error_file_id):
            if file_id:
                text += self.client.files.content(file_id).text.rstrip("\n") + "\n"
        name = f"round{index}_results.ndjson"
        (self.state_dir / name).write_text(text, encoding="utf-8")
        rnd["results"] = name
        rnd["status"] = batch.status
        self._save()

    def run(self, files: list[Path], batch_id: str | None = None) -> list[dict]:
        plans, work = [], []
        for path in files:
            output_path = _judged_path(path.parent, path.stem, self.judge_model, ".jsonl")
            items = _load_items(path)
            judged = _load_checkpoint(path, output_path)
            logger.info(
                f"[{path.name}] {len(items)} items, "
                f"{len(judged)} already judged, "
                f"{len(items) - len(judged)} remaining"
            )
            plans.append((path, output_path, items, judged))
            work.extend((path, i) for i in items if _key(i) not in judged)

        if batch_id and batch_id not in {r["batch_id"] for r in self.state["rounds"]}:
            if self.state["rounds"]:
                raise ValueError(f"{self.state_path} already tracks other batches, not {batch_id}")
            # State lost but the batch is known: request ids are derived from
            # the rows, so the first pass reproduces the ids the batch carries.
            _, requests = self._replay(work)
            self.state["rounds"].append(
                {
                    "batch_id": batch_id,
                    "input": None,
                    "custom_ids": sorted(requests),
                    "results": None,
                }
            )
            self._save()

        # Batches submitted before an interruption are waited for, not resubmitted.
        for index, rnd in enumerate(self.state["rounds"]):
            if not rnd.get("results"):
                self._collect(index, rnd)

        verdicts: dict = {}
        submitted = 0
        while True:
            new, requests = self._replay(
                [w for w in work if (str(w[0]), _key(w[1])) not in verdicts]
            )
            verdicts.update(new)
            if not requests:
                break
            if submitted == self.max_rounds:
                # Recorded as failed calls below, which a rerun re-judges.
                logger.warning(f"{len(requests)} rows still unanswered after {submitted} rounds")
                break
            index = len(self.state["rounds"])
            self._collect(index, self._submit(index, requests))
            submitted += 1

        results = []
        for path, output_path, items, judged in plans:
            if len(judged) >= len(items):
                results.append(_skipped_summary(path, items, judged))
                continue
            scores = []
            with open(output_path, "a", encoding="utf-8") as out:
                for i, item in enumerate(items):
                    if _key(item) in judged:
                        scores.append(judged[_key(item)]["llm_judge"])
                        continue
                    # Rows left unanswered after the last round are recorded as
                    # failed calls, which a rerun re-judges.
                    score, status = verdicts.get((str(path), _key(item)), (0.0, "error"))
                    if status != "ok":
                        logger.warning(
                            f"  [{i + 1}/{len(items)}] verdict unusable ({status}) — "
                            f"id={item['id']}"
                        )
                    scores.append(score)
                    out.write(
                        json.dumps(_judged_row(item, score, status), ensure_ascii=False) + "\n"
                    )
            results.append(_complete_summary(path, output_path, items, scores))
        return results


def find_jsonl_files(base_dir: Path, pattern: str | None = None) -> list[Path]:
    """Find all JSONL files to process."""
    results_dir = base_dir / "data" / "results"
    jsonl_files = []

    dirs = sorted(results_dir.glob("202603*/"))
    if pattern:
        dirs = [d for d in dirs if any(d.name.find(p) >= 0 for p in pattern.split(","))]

    for d in dirs:
        for f in sorted(d.glob("*.jsonl")):
            # Skip already-judged files and checkpoint files
            if "_judged" in f.name or "checkpoint" in f.name:
                continue
            jsonl_files.append(f)

    return jsonl_files


def main():
    parser = argparse.ArgumentParser(description="Run LLM-as-Judge on result JSONL files")
    parser.add_argument(
        "path",
        nargs="?",
        help="Result directory or specific JSONL file",
    )
    parser.add_argument(
        "--pattern",
        help="Filter directories by pattern (e.g., 'rq1', 'rq1_hotpotqa')",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay between LLM calls in seconds (rate limiting)",
    )
    parser.add_argument(
        "--judge-model",
        default="openai/gpt-4.1-nano",
        help="Model to use for judging (default: openai/gpt-4.1-nano)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Judge through the OpenAI Batch API; rerun the same command to resume",
    )
    parser.add_argument(
        "--batch-id",
        help="With --batch: attach to this batch when the local batch state is lost",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=60.0,
        help="With --batch: seconds between batch status checks",
    )
    args = parser.parse_args()
    if args.batch_id and not args.batch:
        parser.error("--batch-id requires --batch")
    if args.batch and not os.environ.get("OPENAI_API_KEY"):
        # Without it every row fails before reaching the batch and is written
        # as an "error" verdict, slowly, after the retry backoff for each one.
        sys.exit("OPENAI_API_KEY is not set in the environment; export it before --batch")

    setup_experiment()

    project_root = Path(__file__).resolve().parent.parent

    if args.path:
        path = Path(args.path)
        if path.is_file() and path.suffix == ".jsonl":
            files = [path]
        elif path.is_dir():
            files = sorted(
                f
                for f in path.glob("*.jsonl")
                if "_judged" not in f.name and "checkpoint" not in f.name
            )
        else:
            logger.error(f"Invalid path: {path}")
            sys.exit(1)
    else:
        files = find_jsonl_files(project_root, pattern=args.pattern)

    if not files:
        logger.error("No JSONL files found")
        sys.exit(1)

    logger.info(f"Found {len(files)} JSONL files to judge")
    logger.info(f"Judge model: {args.judge_model}")
    total_items = 0
    for f in files:
        with open(f) as fh:
            n = sum(1 for _ in fh)
        total_items += n
    logger.info(f"Total predictions: {total_items}")

    # The summary goes beside the results it covers; batch state goes under it.
    if args.path:
        summary_dir = Path(args.path) if Path(args.path).is_dir() else Path(args.path).parent
    else:
        summary_dir = project_root / "data" / "results"

    results = []
    if args.batch:
        state_dir = (
            summary_dir / f"judge_batch{_judge_suffix(args.judge_model).removeprefix('_judged')}"
        )
        logger.info(f"Batch state: {state_dir}")
        judge = BatchJudge(state_dir, args.judge_model, poll_interval=args.poll_interval)
        try:
            results = judge.run(files, batch_id=args.batch_id)
        except KeyboardInterrupt:
            logger.warning(f"Interrupted; rerun the same command to resume from {state_dir}")
            sys.exit(130)
    else:
        for f in files:
            try:
                r = judge_jsonl(f, delay=args.delay, judge_model=args.judge_model)
                results.append(r)
            except KeyboardInterrupt:
                logger.warning("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error processing {f}: {e}")
                results.append({"file": f.name, "status": "error", "error": str(e)})

    # Print summary
    print("\n" + "=" * 60)
    print("LLM-as-Judge Summary")
    print("=" * 60)
    for r in results:
        status = r.get("status", "unknown")
        acc = r.get("judge_accuracy", 0)
        n = r.get("n", 0)
        print(f"  {r['file']:50s} n={n:4d} acc={acc:.3f} [{status}]")

    # Save summary
    # The summary is named after the judge too, minus the "_judged" marker that
    # only makes sense on per-file verdicts. Same legacy preference as
    # _judged_path so an earlier round's summary keeps its name.
    def summary_name(*, legacy: bool) -> str:
        tag = _judge_suffix(args.judge_model, legacy=legacy).removeprefix("_judged")
        return f"llm_judge_summary{tag}.json"

    summary_path = summary_dir / summary_name(legacy=True)
    if not summary_path.exists():
        summary_path = summary_dir / summary_name(legacy=False)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
