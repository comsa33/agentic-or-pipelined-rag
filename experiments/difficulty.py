"""Question difficulty metadata, and the per-dataset reference built from it.

`question_difficulty` is what the runner stamps on every result row
(`question_difficulty` field). Analyses that split by hop count read the hop
from `hop_reference` instead: it applies the same function to the raw
dataset, so every row of a question gets the same hop whether or not the row
carries a stamp (error rows written before 2026-10-23 do not). Where a row
does carry one, `hop_for_row` checks it against the reference.

Free of litellm and DSPy, so analysis code can import it.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path


def question_difficulty(item: dict) -> dict:
    """Extract question difficulty metadata from a dataset item.

    Extracts hop count, entity count, and question type for use
    as moderating variables (hop bins of the analysis).
    """
    import re

    question = item.get("question", "")
    q_lower = question.lower()
    metadata = item.get("metadata", {})

    # Hop count: prefer dataset gold metadata, fallback to heuristic
    # MuSiQue: metadata.n_hops (2-4)
    # HotpotQA: always 2-hop by design
    # 2WikiMultiHopQA: infer from len(supporting_facts)
    hop_count = (
        metadata.get("n_hops")  # MuSiQue gold standard
        or item.get("hop_count")
        or item.get("num_hops")
    )
    if not hop_count:
        supporting_facts = metadata.get("supporting_facts", [])
        if supporting_facts:
            # Number of distinct supporting docs ≈ hop count
            titles = {sf.get("title", sf) for sf in supporting_facts if isinstance(sf, dict)}
            hop_count = max(2, len(titles))  # multi-hop datasets are ≥2
        else:
            # Last resort heuristic
            bridge_words = ["who", "where", "when", "which", "whose"]
            hop_count = max(1, sum(1 for w in bridge_words if w in q_lower))

    # Entity count: named entities (capitalized multi-word spans)
    entities = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", question)
    entity_count = len(set(entities))

    # Question type: prefer dataset metadata, fallback to keyword heuristic
    # HotpotQA/2Wiki: metadata.type = comparison/bridge/compositional/inference
    dataset_type = metadata.get("type", "")
    if dataset_type in ("comparison",):
        question_type = "comparison"
    elif dataset_type in ("bridge", "compositional", "inference"):
        question_type = "bridge"
    elif any(w in q_lower for w in ["how many", "how much", "how long"]):
        question_type = "numerical"
    elif any(w in q_lower for w in ["where", "what place", "born in"]):
        question_type = "location"
    elif any(w in q_lower for w in ["when", "what year", "what date"]):
        question_type = "temporal"
    else:
        question_type = "factoid"

    return {
        "hop_count": hop_count,
        "entity_count": entity_count,
        "question_type": question_type,
    }


class HopMismatchError(ValueError):
    """A row's stamped hop count differs from the reference for its question."""


@functools.cache
def difficulty_reference(dataset: str, raw_dir: Path | None = None) -> dict[str, dict]:
    """id -> question_difficulty for every item of `data/raw/<dataset>.jsonl`."""
    if raw_dir is None:
        from agentic_rag.config.settings import settings

        raw_dir = settings.raw_dir
    path = Path(raw_dir) / f"{dataset}.jsonl"
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                item = json.loads(line)
                out[str(item.get("id", i))] = question_difficulty(item)
    return out


def hop_reference(dataset: str, raw_dir: Path | None = None) -> dict[str, int]:
    """id -> hop count for every item of the dataset."""
    return {qid: int(d["hop_count"]) for qid, d in difficulty_reference(dataset, raw_dir).items()}


def hop_for_row(row: dict, reference: dict[str, int], source: str = "") -> int:
    """The row's hop count from `reference`, checked against its stamp if it has one.

    Raises KeyError when the question is not in the reference and
    HopMismatchError when the stamp disagrees with it.
    """
    qid = str(row["id"])
    if qid not in reference:
        raise KeyError(f"{source}: question {qid} is not in the hop reference")
    hop = reference[qid]
    stamped = (row.get("question_difficulty") or {}).get("hop_count")
    if stamped is not None and int(stamped) != hop:
        raise HopMismatchError(
            f"{source}: question {qid} is stamped hop {stamped}, reference says {hop}"
        )
    return hop
