"""Sonnet MuSiQue 2-hop Agentic-Loop: same delta and CI under gpt-4.1-nano and gpt-4.1, different p.

Compares the per-question difference vectors (k-mean) of the two judges, recomputes the
bootstrap with the cell's own comparison key, and counts changed verdicts per pipeline and k.
Prints the result. No API call.

    uv run python scripts/paper/judge_identical_delta.py
"""

import collections
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import shutil
import tempfile

from loguru import logger

from experiments.analysis.hop_level import _paired_bootstrap
from experiments.analysis.runs import comparison_rng, load_runs
from experiments.difficulty import hop_for_row, hop_reference

logger.remove()
RUN = re.compile(r"^\d{8}_\d{6}_rq1_musique_n200_claude-sonnet-5-5_k[123]$")
res = REPO / "data/results-campaign"
stage = Path(tempfile.mkdtemp())
for p in res.iterdir():
    if RUN.match(p.name):
        (stage / p.name).symlink_to(p.resolve())
(stage / "exclusions.json").symlink_to((res / "exclusions.json").resolve())
ref = hop_reference("musique", REPO / "data/raw")
S = {}
for j, tag in (("nano", "gpt-4-1-nano_7bcf05"), ("gpt41", "gpt-4-1_39118f")):
    recs = load_runs(stage, prefix="rq1", judge=tag, require_paired_n=True)
    d = collections.defaultdict(dict)
    for r in recs:
        if r["model"] != "claude-sonnet-5-5":
            continue
        for row in r["rows"]:
            d[(r["pipeline"], r["repeat_index"])][str(row["id"])] = (
                float(row["llm_judge"]),
                hop_for_row(row, ref, "x"),
            )
    S[j] = d
shutil.rmtree(stage)
A, L = "agentic_(react)", "loop_refinement"
qids = sorted(q for q, (s, h) in S["nano"][(A, 1)].items() if h == 2)
print("n", len(qids))
vec = {}
for j in S:
    a = np.array([np.mean([S[j][(A, k)][q][0] for k in (1, 2, 3)]) for q in qids])
    b = np.array([np.mean([S[j][(L, k)][q][0] for k in (1, 2, 3)]) for q in qids])
    vec[j] = (a, b, a - b)
    p, lo, hi, obs = _paired_bootstrap(
        a, b, comparison_rng(42, "F1", "judge", "claude-sonnet-5-5", "musique", "2", A, L), 10000
    )
    print(j, "A", a.mean(), "L", b.mean(), "delta", obs, "ci", lo, hi, "p", p)
dn, dg = vec["nano"][2], vec["gpt41"][2]
print("diff vectors equal:", np.array_equal(dn, dg), "positions differing:", int((dn != dg).sum()))
print("sorted equal (same multiset):", np.array_equal(np.sort(dn), np.sort(dg)))
print("nano diff value counts", collections.Counter(np.round(dn, 3)).most_common())
print("g41  diff value counts", collections.Counter(np.round(dg, 3)).most_common())
ch = collections.Counter()
for pipe in (A, L):
    for k in (1, 2, 3):
        for q in qids:
            x, y = S["nano"][(pipe, k)][q][0], S["gpt41"][(pipe, k)][q][0]
            if x != y:
                ch[(pipe, k, f"{int(x)}->{int(y)}")] += 1
print(sorted(ch.items()))
# where p comes from
for j in vec:
    rng = comparison_rng(42, "F1", "judge", "claude-sonnet-5-5", "musique", "2", A, L)
    d = vec[j][2]
    idx = rng.integers(0, len(d), size=(10000, len(d)))
    boot = d[idx].mean(axis=1)
    print(
        j,
        "P(boot<=0)",
        np.mean(boot <= 0),
        "P(boot>=0)",
        np.mean(boot >= 0),
        "n boot==0",
        int((boot == 0).sum()),
    )
