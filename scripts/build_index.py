"""Build retrieval indices from dataset passages.

Extracts passages from prepared datasets (HotpotQA / 2WikiMultiHopQA /
MuSiQue context paragraphs, FinanceBench evidence pages) and builds
FAISS + BM25 hybrid indices along with auxiliary section/term indices
for agent tools.

Usage:
  uv run python scripts/build_index.py --dataset hotpotqa
  uv run python scripts/build_index.py --dataset 2wikimultihopqa
  uv run python scripts/build_index.py --dataset musique
  uv run python scripts/build_index.py --dataset financebench
  uv run python scripts/build_index.py --dataset all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentic_rag.config.settings import settings
from agentic_rag.retriever.indexer import DocumentIndexer, Passage


def extract_passages_hotpotqa(raw_path: Path) -> list[Passage]:
    """Extract context paragraphs from HotpotQA dataset."""
    passages = []
    seen_titles = set()

    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            for p in item.get("passages", []):
                title = p.get("title", "")
                content = p.get("content", "")
                if not content or title in seen_titles:
                    continue
                seen_titles.add(title)
                passages.append(
                    Passage(
                        id=f"hotpotqa_{title}",
                        title=title,
                        content=content,
                        source=f"hotpotqa/{title}",
                    )
                )

    logger.info(f"Extracted {len(passages)} unique passages from HotpotQA")
    return passages


def extract_passages_2wikimultihopqa(raw_path: Path) -> list[Passage]:
    """Extract context paragraphs from 2WikiMultiHopQA dataset.

    Same format as HotpotQA — context with title + content.
    """
    passages = []
    seen_titles = set()

    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            for p in item.get("passages", []):
                title = p.get("title", "")
                content = p.get("content", "")
                if not content or title in seen_titles:
                    continue
                seen_titles.add(title)
                passages.append(
                    Passage(
                        id=f"2wiki_{title}",
                        title=title,
                        content=content,
                        source=f"2wikimultihopqa/{title}",
                    )
                )

    logger.info(f"Extracted {len(passages)} unique passages from 2WikiMultiHopQA")
    return passages


def extract_passages_musique(raw_path: Path) -> list[Passage]:
    """Extract paragraphs from MuSiQue dataset."""
    passages = []
    seen_titles = set()

    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            for p in item.get("passages", []):
                title = p.get("title", "")
                content = p.get("content", "")
                if not content or title in seen_titles:
                    continue
                seen_titles.add(title)
                passages.append(
                    Passage(
                        id=f"musique_{title}",
                        title=title,
                        content=content,
                        source=f"musique/{title}",
                    )
                )

    logger.info(f"Extracted {len(passages)} unique passages from MuSiQue")
    return passages


def _extract_section_titles_llm(passages_data: list[dict], model: str) -> dict[str, str]:
    """Extract semantic section titles from passage contents using LLM.

    Sends batch requests to classify each passage into a document section.
    Domain-agnostic — works for financial filings, legal docs, technical manuals, etc.

    Args:
        passages_data: List of {"pid": ..., "source": ..., "content": ...} dicts.
        model: LiteLLM model identifier (e.g., "gemini/gemini-3.1-flash-lite-preview").

    Returns:
        Dict mapping pid → section title string.
    """
    import litellm

    results: dict[str, str] = {}
    batch_size = 10  # passages per LLM call for efficiency

    for i in range(0, len(passages_data), batch_size):
        batch = passages_data[i : i + batch_size]

        # Build batch prompt
        entries = []
        for idx, p in enumerate(batch):
            # Use first 400 chars of content for classification
            snippet = p["content"][:400].replace("\n", " ").strip()
            entries.append(f"[{idx}] Source: {p['source']}\nContent: {snippet}")

        prompt = (
            "You are a document structure analyst. For each passage below, "
            "identify the document section it belongs to.\n\n"
            "Return ONLY a numbered list in this exact format:\n"
            "[0] Section Name\n"
            "[1] Section Name\n"
            "...\n\n"
            "Rules:\n"
            "- Use standard section names (e.g., 'Balance Sheet', 'Income Statement', "
            "'Cash Flow Statement', 'Risk Factors', 'Business Overview')\n"
            "- Keep section names concise (2-5 words)\n"
            "- If the content is a financial statement, use the canonical name\n"
            "- If unsure, describe the content topic briefly\n\n" + "\n\n".join(entries)
        )

        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
                num_retries=3,
            )
            reply = response.choices[0].message.content.strip()

            # Parse response lines like "[0] Balance Sheet"
            for line in reply.split("\n"):
                line = line.strip()
                if not line or not line.startswith("["):
                    continue
                try:
                    bracket_end = line.index("]")
                    idx = int(line[1:bracket_end])
                    section = line[bracket_end + 1 :].strip().lstrip(":-. ")
                    if 0 <= idx < len(batch) and section:
                        results[batch[idx]["pid"]] = section
                except (ValueError, IndexError):
                    continue

        except Exception as e:
            logger.warning(f"LLM section extraction failed for batch {i // batch_size}: {e}")
            # Fallback: use first meaningful content line
            for p in batch:
                if p["pid"] not in results:
                    lines = [
                        ln.strip()
                        for ln in p["content"].split("\n")
                        if ln.strip() and ln.strip() != "Table of Contents" and len(ln.strip()) > 5
                    ]
                    results[p["pid"]] = lines[0][:60] if lines else "Unknown Section"

        if (i + batch_size) % 50 == 0 or i + batch_size >= len(passages_data):
            logger.info(
                f"  Section extraction progress: {min(i + batch_size, len(passages_data))}/{len(passages_data)}"
            )

    return results


def extract_passages_financebench(raw_path: Path) -> list[Passage]:
    """Extract evidence pages from FinanceBench dataset.

    Uses LLM-based section title extraction for domain-agnostic
    document structure analysis, enabling meaningful section_index
    for structure-aware retrieval.
    """
    # Phase 1: Read all passages
    raw_passages = []
    seen_ids = set()

    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            for p in item.get("passages", []):
                original_title = p.get("title", "")
                content = p.get("content", "")
                source = p.get("source", "")
                page_num = p.get("page_num", 0)
                pid = f"fb_{source}_p{page_num}"

                if not content or pid in seen_ids:
                    continue
                seen_ids.add(pid)
                raw_passages.append(
                    {
                        "pid": pid,
                        "original_title": original_title,
                        "content": content,
                        "source": source,
                        "page_num": page_num,
                    }
                )

    logger.info(f"Read {len(raw_passages)} unique passages from FinanceBench")

    # Phase 2: Extract semantic section titles via LLM
    logger.info("Extracting semantic section titles via LLM...")
    section_map = _extract_section_titles_llm(raw_passages, model=settings.model.preprocess_model)

    # Phase 3: Build Passage objects with semantic titles
    passages = []
    section_stats: dict[str, int] = {}

    for p in raw_passages:
        section = section_map.get(p["pid"], "")
        if section:
            title = f"{p['source']} — {section}"
            section_stats[section] = section_stats.get(section, 0) + 1
        else:
            title = p["original_title"]
            section_stats["(unclassified)"] = section_stats.get("(unclassified)", 0) + 1

        passages.append(
            Passage(
                id=p["pid"],
                title=title,
                content=p["content"],
                source=p["source"],
                metadata={"page_num": p["page_num"], "original_title": p["original_title"]},
            )
        )

    logger.info(f"Extracted {len(passages)} unique passages from FinanceBench")
    logger.info(f"Section distribution: {dict(sorted(section_stats.items(), key=lambda x: -x[1]))}")
    return passages


# ---------------------------------------------------------------------------
# Registry and main
# ---------------------------------------------------------------------------

DATASET_EXTRACTORS = {
    "hotpotqa": ("hotpotqa.jsonl", extract_passages_hotpotqa),
    "2wikimultihopqa": ("2wikimultihopqa.jsonl", extract_passages_2wikimultihopqa),
    "musique": ("musique.jsonl", extract_passages_musique),
    "financebench": ("financebench.jsonl", extract_passages_financebench),
}


def build_index_for_dataset(dataset_name: str) -> None:
    """Build hybrid retrieval index for a dataset."""
    if dataset_name not in DATASET_EXTRACTORS:
        logger.error(
            f"No passage extractor for '{dataset_name}'. "
            f"Available: {list(DATASET_EXTRACTORS.keys())}"
        )
        return

    filename, extractor = DATASET_EXTRACTORS[dataset_name]
    raw_path = settings.raw_dir / filename
    if not raw_path.exists():
        logger.error(
            f"Dataset not found: {raw_path}\n"
            f"Run: uv run python scripts/prepare_datasets.py --dataset {dataset_name}"
        )
        return

    # Extract passages
    passages = extractor(raw_path)
    if not passages:
        logger.error(f"No passages extracted from {dataset_name}")
        return

    # Build index
    index_dir = settings.index_dir / dataset_name
    index_dir.mkdir(parents=True, exist_ok=True)

    indexer = DocumentIndexer()
    indexer.passages = indexer.chunk_passages(passages)
    indexer.retriever.build_index(indexer.passages)
    indexer.section_index.build(indexer.passages)
    indexer.term_index.build(indexer.passages)
    indexer.save(index_dir)

    logger.info(
        f"Index built for {dataset_name}: {len(indexer.passages)} chunks, saved to {index_dir}"
    )


def main():
    all_datasets = list(DATASET_EXTRACTORS.keys())
    parser = argparse.ArgumentParser(description="Build retrieval indices from datasets")
    parser.add_argument(
        "--dataset",
        choices=[*all_datasets, "all"],
        default="all",
        help="Which dataset to build index for",
    )
    args = parser.parse_args()

    datasets = all_datasets if args.dataset == "all" else [args.dataset]
    for name in datasets:
        build_index_for_dataset(name)

    logger.info("Index building complete!")


if __name__ == "__main__":
    main()
