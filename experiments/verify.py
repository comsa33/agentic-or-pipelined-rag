"""Checks a result directory must pass before its numbers enter the paper.

Every check here corresponds to a way a past run was silently wrong:

  manifest / preflight     the March 2026 runs recorded no conditions at all
  snapshot                 a re-run on another model snapshot is not a re-run
  cache                    a warm response cache deflates latency to ms
  git dirty                the commit hash does not identify the code that ran
  n / unique ids           a resumed run can drop or duplicate questions
  errors                   an error row has an empty prediction and counts
  latency < 1 s share      cache contamination (see latency_analysis)
  required fields          a table needing a value that was never stored
  metered calls            usage that was not captured cannot be re-measured
  usage complete           a late callback would bill one question to the next
  provenance               a resumed row was made by another process
  passage cap / budget     the controlled scope (M, calls) a table claims

`verify_run_dir` returns the checks; `scripts/verify_campaign.py` prints
them as a table and exits non-zero when any FAIL remains.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from experiments.analysis.latency_analysis import (
    CACHE_CONTAMINATION_LIMIT,
    CACHE_HIT_THRESHOLD_SECONDS,
)
from experiments.exclusions import excluded_ids, load_exclusions, refusal_ids, row_refusal_final
from experiments.manifest import REQUIRED_ROW_FIELDS, RunManifest, load_manifest_or_none

FAIL = "FAIL"
WARN = "WARN"
OK = "OK"

# Keys `summary.json` must carry so a table can be rebuilt from disk alone.
REQUIRED_SUMMARY_SETTINGS = (
    "max_passages",
    "passage_cap",
    "enabled_tools",
    "models",
    "seed",
    "top_k",
)


@dataclass
class Check:
    run: str
    name: str
    status: str
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.status == FAIL


def _result_files(run_dir: Path) -> list[Path]:
    """The per-variant result files, without judge re-scorings or checkpoints."""
    return sorted(
        p for p in run_dir.glob("*.jsonl") if "_judged" not in p.name and "checkpoint" not in p.name
    )


def _read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def verify_run_dir(
    run_dir: str | Path,
    *,
    n: int | None = None,
    allow_dirty: bool = False,
) -> list[Check]:
    """Run every check on one result directory."""
    run_dir = Path(run_dir)
    run = run_dir.name
    checks: list[Check] = []

    def add(name: str, ok: bool, detail: str = "", *, warn: bool = False) -> None:
        status = OK if ok else (WARN if warn else FAIL)
        checks.append(Check(run, name, status, detail))

    if not run_dir.is_dir():
        add("run dir", False, f"not a directory: {run_dir}")
        return checks

    manifest = load_manifest_or_none(run_dir)
    add("manifest", manifest is not None, "" if manifest else "manifest.json missing or unreadable")
    if manifest is not None:
        checks.extend(_manifest_checks(run, manifest, allow_dirty))
    target_n = n if n is not None else (manifest.n if manifest else None)
    expected = _expected_snapshots(manifest)

    files = _result_files(run_dir)
    add("result files", bool(files), "" if files else "no *.jsonl result file")
    if manifest is not None:
        add(
            "variants complete",
            len(files) == len(manifest.variants),
            f"{len(files)} result files for {len(manifest.variants)} variants",
        )

    for path in files:
        checks.extend(_rows_checks(run, path, target_n, expected, manifest))
        checks.extend(_summary_checks(run, path, manifest))

    return checks


def verify_judged(run_dir: str | Path, judge_tag: str) -> list[Check]:
    """Every result file must have a judge re-scoring covering exactly its rows.

    Judging is a separate, later step, so this is opt-in
    (`--require-judge <tag>`). The judged file is `<stem>_judged_<tag>.jsonl`
    and must contain the same ids as the result file, no more and no fewer.
    """
    run_dir = Path(run_dir)
    run = run_dir.name
    out: list[Check] = []
    for path in _result_files(run_dir):
        tag = path.stem
        judged = path.with_name(f"{tag}_judged_{judge_tag}.jsonl")
        if not judged.exists():
            out.append(Check(run, f"{tag}: judged", FAIL, f"{judged.name} missing"))
            continue
        ids = [str(r.get("id")) for r in _read_rows(path)]
        judged_ids = [str(r.get("id")) for r in _read_rows(judged)]
        same = len(judged_ids) == len(ids) and set(judged_ids) == set(ids)
        out.append(
            Check(
                run,
                f"{tag}: judged",
                OK if same else FAIL,
                f"{len(judged_ids)} judged rows for {len(ids)} result rows"
                + ("" if set(judged_ids) == set(ids) else "; id sets differ"),
            )
        )
    return out


def verify_repeat_set(run_dirs: list[str | Path]) -> list[Check]:
    """Checks across several result directories that form repeated runs.

    Runs are grouped by config, dataset and model. Within a group the
    repeat indices must be unique and contiguous from 1 and every run must
    have the same n, or a mean over the repeats would be computed over an
    incomplete or uneven set.
    """
    groups: dict[str, list[tuple[str, RunManifest]]] = {}
    for run_dir in run_dirs:
        m = load_manifest_or_none(Path(run_dir))
        if m is None or not m.run_key:
            continue
        base = "|".join(m.run_key.split("|")[:3])  # stem|dataset|model
        groups.setdefault(base, []).append((Path(run_dir).name, m))

    out: list[Check] = []
    for base, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        label = f"repeat set {base}"
        ks = [m.repeat_index for _, m in members]
        missing_k = [name for name, m in members if m.repeat_index is None]
        out.append(
            Check(label, "repeat index present", FAIL if missing_k else OK, f"no k: {missing_k}")
        )
        present = sorted(k for k in ks if k is not None)
        unique = len(set(present)) == len(present)
        out.append(Check(label, "repeat index unique", OK if unique else FAIL, f"k={present}"))
        contiguous = present == list(range(1, len(present) + 1))
        out.append(
            Check(label, "repeat index contiguous", OK if contiguous else FAIL, f"k={present}")
        )
        ns = sorted({m.n for _, m in members})
        out.append(Check(label, "same n", OK if len(ns) == 1 else FAIL, f"n={ns}"))
    return out


def _manifest_checks(run: str, m: RunManifest, allow_dirty: bool) -> list[Check]:
    out: list[Check] = []

    def add(name, ok, detail="", *, warn=False):
        out.append(Check(run, name, OK if ok else (WARN if warn else FAIL), detail))

    add("preflight", m.preflight.get("status") == "ok", f"status={m.preflight.get('status')}")
    add("run finished", m.finished_at is not None, "finished_at missing: run did not complete")

    unpinned = [r.spec for r in m.models if not r.expected_response_model]
    add(
        "snapshot pinned",
        not unpinned,
        f"unpinned: {sorted(set(unpinned))}" if unpinned else "",
        warn=True,
    )
    for r in m.models:
        if r.expected_response_model:
            add(
                f"snapshot {r.slot}",
                r.observed_response_model == r.expected_response_model,
                f"observed={r.observed_response_model!r} expected={r.expected_response_model!r}",
            )
    expected = _expected_snapshots(m)
    if expected and m.observed_response_models:
        stray = sorted(set(m.observed_response_models) - expected)
        add("snapshot during run", not stray, f"unexpected response.model: {stray}")
    # The model's listed created_at, read at start and end of the run. A
    # change is weak evidence of a model update and does not void the run.
    for r in m.models:
        if r.provider_model_created_at and r.provider_model_created_at_end:
            add(
                f"model created_at stable {r.slot}",
                r.provider_model_created_at == r.provider_model_created_at_end,
                f"start={r.provider_model_created_at} end={r.provider_model_created_at_end}",
                warn=True,
            )
    add(
        "campaign phase fixed",
        m.campaign_phase is not None,
        "manifest has no campaign_phase (smoke|pilot|grid)",
        warn=True,
    )

    add(
        "cache disabled",
        bool(m.cache.get("llm_cache_disabled")),
        "DISABLE_LLM_CACHE was off: latency is not measurable",
    )
    add(
        "git clean",
        allow_dirty or not m.git.get("dirty"),
        f"commit {str(m.git.get('commit'))[:10]} with uncommitted changes: "
        f"{m.git.get('modified_files')}",
    )
    return out


def _expected_snapshots(m: RunManifest | None) -> set[str]:
    if m is None:
        return set()
    return {r.expected_response_model for r in m.models if r.expected_response_model}


def _exclusion_checks(
    run: str, tag: str, path: Path, ids: list[str], m: RunManifest, grid: bool, what: str
) -> list[Check]:
    """Every question in `ids` must be excluded for this run's model.

    exclusions.json sits beside the run directories. A missing entry fails a
    grid run and is information in smoke and pilot runs.
    """
    name = f"{tag}: {what} not in exclusions" + ("" if grid else " (info)")
    model = m.run_key.split("|")[2] if m.run_key.count("|") >= 2 else None
    exclusions_path = path.parent.parent / "exclusions.json"
    try:
        exclusions = load_exclusions(exclusions_path) if exclusions_path.is_file() else []
        excluded = excluded_ids(exclusions, m.dataset, model) if model else set()
    except ValueError as e:  # e.g. entries without a model, looked up for Claude
        return [Check(run, name, FAIL if grid else OK, str(e))]
    missing = [i for i in ids if i not in excluded]
    detail = f"{len(missing)} of {len(ids)} {what} question(s) not excluded: {missing}"
    return [Check(run, name, FAIL if grid and missing else OK, detail)]


def _refusal_seen(row: dict) -> bool:
    """A refusal stood at the row's item (prior attempts too): the exclusion trigger."""
    return row_refusal_final(row)


# Share of a cell's rows that may end truncated before verify warns (grid).
TRUNCATED_SHARE_LIMIT = 0.05


def _rows_checks(
    run: str,
    path: Path,
    n: int | None,
    expected: set[str],
    manifest: RunManifest | None,
) -> list[Check]:
    out: list[Check] = []
    tag = path.stem

    def add(name, ok, detail="", *, warn=False):
        out.append(Check(run, f"{tag}: {name}", OK if ok else (WARN if warn else FAIL), detail))

    rows = _read_rows(path)
    valid = [r for r in rows if "error" not in r]
    n_err = len(rows) - len(valid)
    # In the grid, refused and truncated error rows are outcomes, not
    # defects: refusals are excluded and truncations scored wrong, as the
    # analysis plan fixed before the grid ran. The phase comes from the manifest,
    # fixed before the run.
    grid = manifest is not None and manifest.campaign_phase == "grid"
    final_errors = {"refused": 0, "truncated": 0, "platform_filtered": 0}
    if grid:
        for r in rows:
            if "error" in r and r.get("error_class") in final_errors:
                final_errors[r["error_class"]] += 1

    if n is None:
        add("n", True, f"{len(rows)} rows (no target given)", warn=True)
    else:
        add("n", len(rows) == n, f"{len(rows)} rows, target {n}")
    ids = [str(r.get("id")) for r in rows]
    add("unique ids", len(set(ids)) == len(ids), f"{len(set(ids))} unique of {len(ids)}")
    n_open = n_err - sum(final_errors.values())
    add("errors", n_open == 0, f"{n_open} error row(s)")
    if grid:
        add(
            "final error rows (info)",
            True,
            f"to exclude: refused {final_errors['refused']}, "
            f"platform_filtered {final_errors['platform_filtered']}; "
            f"scored wrong: truncated {final_errors['truncated']}",
        )
        # Rows touched by truncation, answered or not, over the cell's rows.
        cut_ok = sum(1 for r in valid if int(r.get("truncated_calls") or 0) > 0)
        cut_err = sum(
            1
            for r in rows
            if "error" in r
            and (r.get("error_class") == "truncated" or int(r.get("truncated_calls") or 0) > 0)
        )
        share = (cut_ok + cut_err) / len(rows) if rows else 0.0
        add(
            "truncated share",
            share <= TRUNCATED_SHARE_LIMIT,
            f"{cut_ok + cut_err}/{len(rows)} = {share:.1%} "
            f"(answered {cut_ok}, error {cut_err}; limit {TRUNCATED_SHARE_LIMIT:.0%})",
            warn=True,
        )

    missing = sorted({f for r in valid for f in REQUIRED_ROW_FIELDS if f not in r})
    add("required fields", not missing, f"missing in some rows: {missing}")

    latencies = [r["latency_seconds"] for r in valid if "latency_seconds" in r]
    if latencies:
        fast = sum(1 for x in latencies if x < CACHE_HIT_THRESHOLD_SECONDS)
        share = fast / len(latencies)
        add(
            "latency < 1s share",
            share <= CACHE_CONTAMINATION_LIMIT,
            f"{fast}/{len(latencies)} = {share:.0%} (limit {CACHE_CONTAMINATION_LIMIT:.0%})",
        )

    unmetered = sum(1 for r in valid if not r.get("metered_calls"))
    add("usage metered", unmetered == 0, f"{unmetered} row(s) with no metered call")
    # Rows written before finish reasons were recorded carry no
    # `truncated_calls`; the check applies only where the field exists.
    # In the grid a truncated call is an outcome (counted in "truncated
    # share"); in smoke and pilot it fails, to stop and look before the grid.
    if any("truncated_calls" in r for r in valid):
        truncated = [r for r in valid if int(r.get("truncated_calls") or 0) > 0]
        detail = f"{len(truncated)} row(s) with a call stopped at max_tokens"
        if grid:
            add("truncated calls (info)", True, detail)
        else:
            add("no truncated calls", not truncated, detail)
    # Information only, never a failure: failed calls are retried or fall
    # back inside DSPy, and a refusal is scored as the answer it produced.
    # The counts let the paper say how often either happened.
    if any("failed_calls" in r for r in valid):
        by_status: dict[str, int] = {}
        for r in valid:
            for status, n in (r.get("failed_by_status") or {}).items():
                by_status[status] = by_status.get(status, 0) + int(n)
        rows_failed = sum(1 for r in valid if int(r.get("failed_calls") or 0) > 0)
        add("failed calls (info)", True, f"{rows_failed} row(s); by status {by_status}")
        timeouts = sum(int(r.get("timeout_calls") or 0) for r in valid)
        add("timeout calls (info)", True, f"{timeouts} call(s)")
    # Refusals and unpriced calls are counted over every row, error rows
    # included: a refusal usually ends in a parse failure, i.e. an error row.
    if any("refused_calls" in r or "refusal_seen" in r for r in rows):
        refused = sum(1 for r in rows if _refusal_seen(r))
        add("refused calls (info)", True, f"{refused} row(s) with a provider refusal")
        if any("refusal_retries" in r for r in rows):
            first_ids = refusal_ids(rows, first=True)
            retries = sum(int(r.get("refusal_retries") or 0) for r in rows)
            recovered = sum(int(r.get("refusals_recovered") or 0) for r in rows)
            add(
                "refusal resends (info)",
                True,
                f"{retries} resend(s), {recovered} recovered; first refusals "
                f"(sensitivity analysis) on {len(first_ids)} question(s): {first_ids}",
            )
        if manifest is not None and refused:
            refused_ids = refusal_ids(rows)
            out.extend(_exclusion_checks(run, tag, path, refused_ids, manifest, grid, "refusal"))
    filtered_ids = sorted(
        str(r.get("id"))
        for r in rows
        if "error" in r and r.get("error_class") == "platform_filtered"
    )
    if manifest is not None and filtered_ids:
        out.extend(
            _exclusion_checks(run, tag, path, filtered_ids, manifest, grid, "platform filter")
        )
    if n_err:
        classes: dict[str, int] = {}
        for r in rows:
            if "error" in r:
                c = r.get("error_class") or "unclassified"
                classes[c] = classes.get(c, 0) + 1
        add("error classes (info)", True, f"{classes}")
    if any("attempts" in r for r in valid):
        attempts = sum(int(r.get("attempts") or 0) for r in valid)
        metered = sum(int(r.get("metered_calls") or 0) for r in valid)
        add("attempts (info)", True, f"{attempts} request(s) sent for {metered} metered call(s)")
    if any("structured_calls" in r for r in valid):
        modes: dict[str, int] = {}
        for r in valid:
            for mode, n in (r.get("structured_calls") or {}).items():
                modes[mode] = modes.get(mode, 0) + int(n)
        fell_back = sum(1 for r in valid if r.get("structured_calls"))
        add("adapter fallbacks (info)", True, f"{fell_back} row(s); calls by mode {modes}")
    # Written by runs after stop_details and the raw usage keys were metered;
    # older rows carry neither and get no line.
    if any("refusal_categories" in r for r in rows):
        categories: dict[str, int] = {}
        explained = 0
        for r in rows:
            for category, n in (r.get("refusal_categories") or {}).items():
                categories[category] = categories.get(category, 0) + int(n)
            explained += int(r.get("refusal_explained") or 0)
        add(
            "refusal categories (info)",
            True,
            f"{categories or 'none'}; {explained} with an explanation",
        )
    if any("usage_keys_seen" in r for r in rows):
        keys = sorted({k for r in rows for k in r.get("usage_keys_seen") or []})
        thinking = sum(int(r.get("thinking_blocks") or 0) for r in rows)
        details = any(r.get("output_tokens_details_seen") for r in rows)
        add(
            "raw usage fields (info)",
            True,
            f"keys {keys}; output_tokens_details {'present' if details else 'absent'}; "
            f"{thinking} thinking block(s)",
        )
    # Anthropic prices are registered in code (settings.ANTHROPIC_PRICES), so
    # an Anthropic call without a cost means the registration did not apply.
    claude_rows = [
        r for r in rows if any(str(m).startswith("claude") for m in r.get("response_models", []))
    ]
    if claude_rows:
        claude_uncosted = [r for r in claude_rows if int(r.get("calls_without_cost") or 0) > 0]
        add(
            "anthropic calls priced",
            not claude_uncosted,
            f"{len(claude_uncosted)} row(s) with an Anthropic call that has no cost",
        )
    incomplete = sum(1 for r in valid if r.get("usage_complete") is False)
    add("usage complete", incomplete == 0, f"{incomplete} row(s) with unreported calls")
    # The paper reports metered_calls. llm_calls is the pipeline's own
    # estimate and is kept so a drift between the two shows where the
    # estimate breaks (it overcounts ReAct steps at max_iters=1). A call that
    # failed to parse and was re-sent through JSONAdapter is metered twice but
    # counted once by the pipeline; those re-sends are in structured_calls.
    drift = [
        r
        for r in valid
        if abs(int(r.get("llm_calls") or 0) + _resends(r) - int(r.get("metered_calls") or 0)) > 1
    ]
    add(
        "pipeline count drifts from metered",
        not drift,
        f"{len(drift)} row(s) with |llm_calls - metered_calls| > 1",
        warn=True,
    )

    if expected:
        stray = sorted(
            {m for r in valid for m in r.get("response_models", []) if m not in expected}
        )
        add("row snapshots", not stray, f"unexpected response.model in rows: {stray}")

    if manifest is not None:
        out.extend(_provenance_checks(run, tag, valid, manifest))
        out.extend(_passage_cap_checks(run, tag, path, valid, manifest))
        out.extend(_call_budget_checks(run, tag, path, valid, manifest))
        out.extend(_ircot_checks(run, tag, path, valid, manifest))
    return out


def _resends(row: dict) -> int:
    """Calls re-sent through JSONAdapter after a parse failure (metered, not counted)."""
    return sum(int(v or 0) for v in (row.get("structured_calls") or {}).values())


def _ircot_checks(run: str, tag: str, path: Path, rows: list[dict], m: RunManifest) -> list[Check]:
    """An IRCoT row makes at most max_steps reasoning calls plus one generation.

    The bound is on the pipeline's own count: a JSONAdapter re-send is not a
    step. The metered count the paper reports is shown next to it.
    """
    variant = _variant_of(path)
    params = m.ircot_params_by_pipeline.get(variant) if variant else None
    if not params or "max_steps" not in params:
        return []
    bound = int(params["max_steps"]) + 1
    calls = [int(r.get("llm_calls") or 0) for r in rows]
    over = sum(1 for c in calls if c > bound)
    metered = sum(int(r.get("metered_calls") or 0) for r in rows)
    resends = sum(_resends(r) for r in rows)
    return [
        Check(
            run,
            f"{tag}: ircot steps",
            OK if over == 0 else FAIL,
            f"max_steps {params['max_steps']} -> at most {bound} calls; {over} row(s) over",
        ),
        Check(
            run,
            f"{tag}: ircot metered calls (info)",
            OK,
            f"{metered} metered for {sum(calls)} pipeline calls"
            f" ({metered / max(len(rows), 1):.2f}/question); {resends} JSONAdapter re-send(s)",
        ),
    ]


# Below this share of the budget a "budget-matched" run did not match anything.
BUDGET_SPENT_MIN_SHARE = 0.8


def _call_budget_checks(
    run: str, tag: str, path: Path, rows: list[dict], m: RunManifest
) -> list[Check]:
    """A budget-matched variant must have spent its budget and never exceeded it.

    The loop pipeline spends a budget deterministically, so it is held to it
    exactly. The agentic pipeline can only cap its iterations, since the
    agent decides which LLM-backed tools to call, so an overrun there is
    reported rather than failed.
    """
    out: list[Check] = []

    def add(name, ok, detail="", *, warn=False):
        out.append(Check(run, f"{tag}: {name}", OK if ok else (WARN if warn else FAIL), detail))

    variant = _variant_of(path)
    budget = m.llm_call_budget_by_pipeline.get(variant) if variant else None
    if budget is None:
        return out
    calls = [r.get("llm_calls") for r in rows if r.get("llm_calls") is not None]
    if not calls:
        return out
    strict = m.pipeline_by_variant.get(variant) != "agentic"
    over = sum(1 for c in calls if c > budget)
    add(
        "call budget",
        over == 0,
        f"budget {budget}: {over} row(s) over, max {max(calls)}"
        + ("" if strict else " (agentic: best-effort cap)"),
        warn=not strict,
    )
    mean = sum(calls) / len(calls)
    add(
        "call budget spent",
        mean >= BUDGET_SPENT_MIN_SHARE * budget,
        f"mean {mean:.1f} of budget {budget} ({mean / budget:.0%})",
        warn=True,
    )
    return out


def _variant_of(path: Path) -> str | None:
    """Variant name of a result file, from its summary.json."""
    summary_path = path.with_name(f"{path.stem}_summary.json")
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8")).get("extra", {}).get("variant")
    except json.JSONDecodeError:
        return None


def _passage_cap_checks(
    run: str, tag: str, path: Path, rows: list[dict], m: RunManifest
) -> list[Check]:
    """Every row must respect the passage cap the manifest declares for its variant.

    This is the mechanical basis of the paper's "controlled for M" scope: a
    capped pipeline may never hand the generator more than M passages, and
    an uncapped one must have used everything it retrieved.
    """
    out: list[Check] = []

    def add(name, ok, detail="", *, warn=False):
        out.append(Check(run, f"{tag}: {name}", OK if ok else (WARN if warn else FAIL), detail))

    variant = _variant_of(path)
    if variant is None or variant not in m.max_passages_by_pipeline:
        add("passage cap", False, f"no cap declared for variant {variant!r}", warn=True)
        return out
    cap = m.max_passages_by_pipeline[variant]
    used = [r.get("passages_used") for r in rows if r.get("passages_used") is not None]
    if not used:
        return out
    if cap is None:
        truncated = sum(
            1 for r in rows if r.get("passages_used") != r.get("total_passages_retrieved")
        )
        add("passage cap", truncated == 0, f"uncapped, but {truncated} row(s) used < retrieved")
    else:
        over = sum(1 for u in used if u > cap)
        add("passage cap", over == 0, f"cap {cap}: {over} row(s) over, max used {max(used)}")
        add("passage cap reached", max(used) >= cap, f"cap {cap} never reached", warn=True)
    return out


def _provenance_checks(run: str, tag: str, rows: list[dict], m: RunManifest) -> list[Check]:
    """Rows reused from a checkpoint must have been made under this manifest's code.

    A resumed run carries rows written by an earlier process. They are
    acceptable only if that process ran the same commit with the cache off
    and got answers from the pinned snapshot (checked with the other rows);
    otherwise the manifest would vouch for conditions it did not produce.
    """
    out: list[Check] = []

    def add(name, ok, detail="", *, warn=False):
        out.append(Check(run, f"{tag}: {name}", OK if ok else (WARN if warn else FAIL), detail))

    unstamped = sum(1 for r in rows if not r.get("run_id") or not r.get("git_commit"))
    add("provenance", unstamped == 0, f"{unstamped} row(s) without run_id/git_commit stamp")

    cached = sum(1 for r in rows if r.get("llm_cache_disabled") is False)
    add("row cache", cached == 0, f"{cached} row(s) made with the response cache on")

    foreign = [r for r in rows if r.get("run_id") and r.get("run_id") != m.run_id]
    if not foreign:
        return out
    sources = sorted({str(r.get("run_id")) for r in foreign})
    other_commit = sorted(
        {str(r.get("git_commit")) for r in foreign if r.get("git_commit") != m.git.get("commit")}
    )
    add(
        "resumed rows commit",
        not other_commit,
        f"{len(foreign)} resumed row(s) from {sources}; commits differ from manifest: "
        f"{[c[:10] for c in other_commit]}",
    )
    if not other_commit:
        add("resumed rows", False, f"{len(foreign)} row(s) resumed from {sources}", warn=True)
    return out


def _summary_checks(run: str, path: Path, manifest: RunManifest | None = None) -> list[Check]:
    out: list[Check] = []
    tag = path.stem
    summary_path = path.with_name(f"{tag}_summary.json")
    if not summary_path.exists():
        out.append(Check(run, f"{tag}: summary", FAIL, f"{summary_path.name} missing"))
        return out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    has_metrics = bool(summary.get("metrics"))
    settings_used = summary.get("settings") or {}
    missing = [k for k in REQUIRED_SUMMARY_SETTINGS if k not in settings_used]
    ok = has_metrics and not missing
    detail = (
        "" if ok else f"metrics={'ok' if has_metrics else 'missing'} settings missing={missing}"
    )
    out.append(Check(run, f"{tag}: summary", OK if ok else FAIL, detail))
    if manifest is not None:
        variant = (summary.get("extra") or {}).get("variant")
        out.append(_variant_settings_check(run, tag, variant, settings_used, manifest))
    return out


def _variant_settings_check(
    run: str, tag: str, variant: str | None, settings_used: dict, m: RunManifest
) -> Check:
    """The settings a variant ran under must be those the manifest declared for it.

    The manifest records each variant's settings before the first question
    (settings_by_variant); the summary records them again while the variant
    runs. Every key either side carries must agree (agent iterations and
    tools, retrieval, passage cap, call budget, models, ...). Version 1
    manifests have no per-variant record, only the process defaults, which
    are not the settings any variant ran under: a warning, not a pass.
    """
    name = f"{tag}: variant settings vs manifest"
    if not m.settings_by_variant:
        return Check(
            run,
            name,
            WARN,
            "manifest has no settings_by_variant (version 1): its run-level blocks are "
            "process defaults, not the applied settings; only summary.json records them",
        )
    declared = m.settings_by_variant.get(variant) if variant else None
    if declared is None:
        return Check(run, name, FAIL, f"no settings declared for variant {variant!r}")
    differ = {
        k: (declared.get(k), settings_used.get(k))
        for k in sorted(set(declared) | set(settings_used))
        if declared.get(k) != settings_used.get(k)
    }
    detail = "; ".join(f"{k}: manifest {a!r}, summary {b!r}" for k, (a, b) in differ.items())
    return Check(run, name, FAIL if differ else OK, detail)
