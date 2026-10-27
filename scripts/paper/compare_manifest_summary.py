#!/usr/bin/env python3
"""Manifest fields against the settings each variant actually ran under.

For every campaign run directory (controlled grid rq1, budget_matched,
external; n200/n150, smoke n2 excluded), compare the run-level manifest
(written once, before any variant's settings are applied) with the
`settings` block of each `<prefix>_<variant>_summary.json`, which run.py
snapshots while that variant runs (run.py _run_variant). Also counts the
ReAct iterations in the result rows (action_history length) and the metered
calls of the loop budget variant. Prints one line per (field, manifest value,
applied value) with the runs and variants it covers. Reads only; no API call;
no prediction is printed.

    uv run python scripts/paper/compare_manifest_summary.py [results-dir]
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(__file__).resolve().parents[2] / "data/results-campaign"
)
RUN = re.compile(
    r"^\d{8}_\d{6}_(rq1|budget_matched|external)_(\w+?)_n(200|150)_(gpt-5-mini|claude-sonnet-5-5)_k\d$"
)

# manifest path -> summary settings key
PAIRS = [
    (("model_defaults", "temperature"), "temperature"),
    (("model_defaults", "max_tokens"), "max_tokens"),
    (("retrieval", "top_k"), "top_k"),
    (("retrieval", "query_method"), "query_method"),
    (("retrieval", "hybrid_weight"), "hybrid_weight"),
    (("retrieval", "max_passages"), "max_passages"),
    (("retrieval", "max_passages_all_pipelines"), "max_passages_all_pipelines"),
    (("evaluation", "quality_threshold"), "quality_threshold"),
    (("evaluation", "max_retry"), "max_retry"),
    (("agent", "max_iterations"), "agent_max_iterations"),
    (("agent", "enabled_tools"), "enabled_tools"),
    (("seed",), "seed"),
]

table: dict = defaultdict(
    lambda: defaultdict(set)
)  # field -> (manifest, applied) -> {exp/model/variant}
per_variant: dict = defaultdict(lambda: defaultdict(set))
iters: dict = defaultdict(Counter)
budget_calls: dict = defaultdict(Counter)
runs = 0
for d in sorted(ROOT.iterdir()):
    m = RUN.match(d.name)
    if not m or not (d / "manifest.json").is_file():
        continue
    runs += 1
    exp, model = m.group(1), m.group(4)
    man = json.loads((d / "manifest.json").read_text())
    models_man = {r["slot"]: r["spec"] for r in man["models"]}
    for s in sorted(d.glob("*_summary.json")):
        summ = json.loads(s.read_text())
        st, variant = summ["settings"], summ["extra"]["variant"]
        tag = f"{exp}/{model}/{variant}"
        for path, key in PAIRS:
            mv = man
            for p in path:
                mv = mv.get(p) if isinstance(mv, dict) else None
            table[".".join(path)][(json.dumps(mv), json.dumps(st.get(key)))].add(tag)
        table["models (4 slots)"][(json.dumps(models_man == st["models"]), "")].add(tag)
        table["max_passages_by_pipeline[variant]"][
            (
                json.dumps(man["max_passages_by_pipeline"].get(variant)),
                json.dumps(st.get("passage_cap")),
            )
        ].add(tag)
        table["llm_call_budget_by_pipeline[variant]"][
            (
                json.dumps(man["llm_call_budget_by_pipeline"].get(variant)),
                json.dumps(st.get("llm_call_budget")),
            )
        ].add(tag)
        if variant in man.get("ircot_params_by_pipeline", {}):
            ip = man["ircot_params_by_pipeline"][variant]
            table["ircot_params_by_pipeline[variant]"][
                (
                    json.dumps({k: ip.get(k) for k in ("max_steps", "per_step_k")}),
                    json.dumps(
                        {
                            "max_steps": st.get("ircot_max_steps"),
                            "per_step_k": st.get("ircot_per_step_k"),
                        }
                    ),
                )
            ].add(tag)
        per_variant[tag]["react_max_iters"].add(json.dumps(st.get("react_max_iters")))
        rows_file = s.with_name(s.name.replace("_summary.json", ".jsonl"))
        if rows_file.is_file():
            for line in rows_file.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if "error" in r:
                    continue
                if "agentic" in variant.lower():
                    iters[tag][len(r.get("action_history") or [])] += 1
                if st.get("llm_call_budget") and "loop" in variant.lower():
                    budget_calls[tag][r.get("metered_calls")] += 1

print(f"runs {runs}")
for field, combos in table.items():
    for (mv, av), tags in sorted(combos.items()):
        groups = Counter(t.rsplit("/", 1)[0] for t in tags)
        same = "SAME" if mv == av or field == "models (4 slots)" else "DIFF"
        print(f"{field:42s} manifest={mv:40s} applied={av:40s} {same} n={len(tags)} {dict(groups)}")
print("\nreact_max_iters per agentic variant:")
for tag, v in sorted(per_variant.items()):
    if "agentic" in tag.lower():
        print(" ", tag, sorted(v["react_max_iters"]))
print("\naction_history length, max per agentic variant group:")
agg = defaultdict(Counter)
for tag, c in iters.items():
    agg[tag] += c
for tag, c in sorted(agg.items()):
    print(" ", tag, "max", max(c), "dist", dict(sorted(c.items())))
print("\nmetered_calls of loop budget variants:")
for tag, c in sorted(budget_calls.items()):
    print(" ", tag, dict(sorted(c.items(), key=lambda kv: (kv[0] is None, kv[0]))))
