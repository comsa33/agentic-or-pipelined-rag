# Agentic or Pipelined?

Code for the paper *Agentic or Pipelined? A Confound-Controlled Empirical Study of an Architectural Decision in Retrieval-Augmented LLM Applications* (Ruo Lee, Taeyeon Oh).

The paper compares two kinds of retrieval-augmented generation (RAG) configuration:

- **Agentic:** a ReAct agent that chooses its own retrieval tools.
- **Pipelined:** Naive RAG, Single-Pass, Loop Refinement and a CRAG-style pipeline.

All five configurations share the retriever, the passage cap, the evaluator and the generation signature. They differ only in control flow.

This repository holds:

- the five configurations;
- the run manifest and the pre-flight model-snapshot check;
- the validator that checks every run directory;
- the budget-matched cells and the external IRCoT instance;
- the scripts that build the paper's tables from the recorded runs (`scripts/paper/`).

The recorded runs themselves are not in this repository (see [Data and analysis](#data-and-analysis)).

## Install

You need Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen --extra dev
cp .env.example .env        # model names and provider keys; see below
```

### Provider keys

Put the keys in `.env`. Only the template `.env.example` is tracked; every other `.env*` file is ignored by git.

| Purpose | Variables |
|---|---|
| gpt-5-mini through Azure OpenAI (the paper's runs) | `AZURE_API_KEY`, `AZURE_API_BASE` (your resource endpoint), `AZURE_API_VERSION` (the runs used `2024-12-01-preview`; each manifest records it) |
| The judge (`openai/gpt-4.1`) | `OPENAI_API_KEY` |
| Claude Sonnet 5.5 | a separate keys file, below |

Anthropic keys live in `.env.anthropic` (or the file named by `ANTHROPIC_KEYS_FILE`), one per slot:

```env
ANTHROPIC_API_KEY_1=<your key>
```

A run names the slot it may spend with `ANTHROPIC_KEY_SLOT=1`. The manifest records the slot, never the key, and an Anthropic model does not start without a slot.

## Prepare the data

The runs drew their questions from these files with seed 42 (200 per multi-hop workload, all 150 for FinanceBench). HotpotQA is prepared in full (7,405 validation questions). 2WikiMultiHopQA keeps its first 500 questions, and MuSiQue its first 500 answerable questions (`prepare_datasets.py` drops the unanswerable ones before taking 500). Each file is also the retrieval corpus of its workload.

```bash
uv run python scripts/prepare_datasets.py --dataset hotpotqa
uv run python scripts/prepare_datasets.py --dataset 2wikimultihopqa --sample 500
uv run python scripts/prepare_datasets.py --dataset musique --sample 500
uv run python scripts/prepare_datasets.py --dataset financebench
```

These commands reproduce the files the runs used byte for byte, and the seed-42 draw from them gives the question set of every recorded run.

Build the hybrid (FAISS + BM25) indices with the paper's embedding model:

```bash
EMBEDDING_MODEL=all-MiniLM-L6-v2 uv run python scripts/build_index.py --dataset all
```

The default `EMBEDDING_MODEL` in the code is `text-embedding-3-small` (OpenAI, paid). The paper used `all-MiniLM-L6-v2`, which runs locally. `.env.example` sets it.

**FinanceBench passage titles.** While building the FinanceBench index, `build_index.py` asks `PREPROCESS_MODEL` (an LLM) to name the section of each evidence page, and the passage title is `<document> — <section>`. Retrieval scores only the passage text, so the titles are not used for the ranking. The other three indices are rebuilt from the same passages as the indices the runs used.

## Run one cell of the controlled grid

A run is one model family on one workload, repeated as run `k`. The command below uses the paper's settings:

- the response cache is off;
- every configuration is capped at the same 30 passages;
- the four model slots point at one model.

```bash
DISABLE_LLM_CACHE=true RETRIEVAL_MAX_PASSAGES_ALL_PIPELINES=true \
PREPROCESS_MODEL=azure/gpt-5-mini EVALUATE_MODEL=azure/gpt-5-mini \
GENERATE_MODEL=azure/gpt-5-mini AGENT_MODEL=azure/gpt-5-mini \
LLM_EXPECTED_RESPONSE_MODELS=azure/gpt-5-mini=gpt-5-mini-2025-08-07 \
RESULTS_DIR=data/results-campaign \
uv run python experiments/run.py --config configs/experiment/rq1.yaml \
    --dataset 2wikimultihopqa --sample 200 --repeat-index 1 --campaign-phase grid
```

**Workloads**

| `--dataset` | `--sample` |
|---|---|
| `hotpotqa` | 200 |
| `2wikimultihopqa` | 200 |
| `musique` | 200 |
| `financebench` | 150 |

For the Claude Sonnet runs, set the four model slots to `anthropic/claude-sonnet-5-5` and set `ANTHROPIC_KEY_SLOT`.

The configuration sets `temperature: 0.0`, but neither model family receives it. litellm drops it for gpt-5-mini, which accepts only its default, and the Anthropic client removes it for Claude Sonnet 5.5. Each manifest records the parameters actually sent (`sent_params`).

**Other configuration files**

| File | Runs |
|---|---|
| `configs/experiment/budget_matched.yaml` | the budget-matched cells |
| `configs/experiment/external.yaml` | IRCoT under the same protocol |
| `configs/experiment/rq3.yaml`, `rq4.yaml`, `configs/ablation/` | the tool variants reported descriptively |

File names map as follows: `rq1` = controlled grid, `rq3`/`rq4` = uncontrolled instance variants. They are kept because they name the result files.

**What each run directory holds**

- `manifest.json`, written before the first question. It records the models, the parameters sent, the snapshot the provider returned, the code commit and the settings of every configuration.
- One `<experiment>_<configuration>.jsonl` per configuration, with one row per question.
- One `_summary.json` per configuration.

## Validate and judge

```bash
uv run python scripts/verify_campaign.py data/results-campaign/<run_dir> --n 200
uv run python scripts/run_llm_judge.py data/results-campaign/<run_dir> --judge-model openai/gpt-4.1 --batch
```

`verify_campaign.py` fails a run when any of the following happens:

- rows are missing or duplicated;
- a returned snapshot differs from the declared one;
- the passage cap or the call budget is broken;
- the response cache was on;
- the settings a configuration ran under differ from those its manifest declared.

The recorded runs have version-1 manifests. Those record the process defaults rather than the settings of each configuration, so the validator reports that check as a warning on them, not a pass. The settings each configuration ran under are in its `_summary.json`. Runs made with this code write version-2 manifests (`settings_by_variant`), which the validator checks.

## Rebuild the paper's tables

```bash
scripts/paper/run_all.sh <judge41_vs_human_hedge167_rows.csv>
```

The script runs the analyses in dependency order and writes to `scripts/paper/out/`. It reads `data/results-campaign` and `data/raw` and makes no model call. The argument is the file of 167 human-labelled rows from the paper's supplementary data. Parts that need the uncontrolled-grid trees (`data/results/paper`, `data/results-openweight/final`) or the Sonnet smoke tree (`data/results-smoke-f23-sonnet`) run only when those trees are present.

The following are results only, because they need files that are not released (the human annotators' label sheets and the uncontrolled judge-panel verdicts):

- the judge-human κ per workload and on all 200 items;
- the uncontrolled-grid judge sweep;
- the literature-audit coding.

Their outputs are in the paper's supplementary data.

## Code provenance

Each run's manifest records the commit of the development repository the run was made from (`git.commit`). The campaign used five commits: `6c307f3`, `d24f8d5`, `27542f9`, `cac25e4` and `cc845ba`. That repository is not public.

This repository is the final state of the development code (`a0d70e0`, made after the last run) with the following changes:

- names, comments and documentation tidied;
- unused modules removed (prompt optimisation, an old human-evaluation sheet, old analyses, PopQA, Natural Questions and Wikipedia loaders);
- an import cycle in `agentic_rag/tools` fixed.

Against the run commits, later development commits changed:

- for every run commit: the run manifest (version 2 adds the per-configuration settings record), the validator and the run driver that writes the manifest;
- for the two runs at `cac25e4`: how the runner records a refused answer (an empty string instead of `None`; the analysis reads both as empty);
- for the runs at `6c307f3` (gpt-5-mini): the runner's error classification, resume rules and per-row usage fields, the Anthropic client (not used by those runs) and several analysis modules.

The pipelines, retriever, signatures and tools are the same at all five commits.

Every string that reaches a model prompt is identical to each of the five commits:

- the Signature instructions and field descriptions;
- the docstrings of the tools given to the agent;
- the prompt constants.

The only difference is one unused Korean prompt constant, which was removed. To check, dump both trees and compare:

```bash
uv run python scripts/dump_prompts.py <development-code-tree> > run.json
uv run python scripts/dump_prompts.py > here.json
diff run.json here.json
```

## Tests

```bash
uv run pytest tests/
```

Seven tests are skipped without the recorded runs:

- Six read the campaign. To run them, point `CAMPAIGN_RESULTS_DIR` at a `results-campaign` directory (and `RAW_DATA_DIR` at `data/raw` if it is not next to it).
- One checks that the validator rejects an uncontrolled March run. It needs `data/results/paper`.

## Data and analysis

The recorded runs will be deposited on Zenodo (**DOI: 10.5281/zenodo.XXXXXXX**, deposit planned). The deposit will contain:

- the run manifests, result rows, judge verdicts and exclusion lists of the campaign (`data/results-campaign`);
- the prepared question files the runs drew from.

Datasets keep their own licences (below). Result rows that quote FinanceBench text are under its non-commercial terms.

## Datasets and licences

`scripts/prepare_datasets.py` downloads the datasets from the Hugging Face Hub. They are not redistributed here. Two of them are read from community mirrors of the original release. The licence is the one of the original dataset.

| Workload | Loaded from | Original release | Licence |
|---|---|---|---|
| HotpotQA | `hotpot_qa` (distractor, validation), now `hotpotqa/hotpot_qa` | Yang et al. (2018) | CC BY-SA 4.0 |
| 2WikiMultiHopQA | `framolfese/2WikiMultihopQA` (mirror, validation) | Ho et al. (2020), Alab-NII/2wikimultihop | Apache 2.0 |
| MuSiQue | `bdsaglam/musique` (mirror, validation, answerable) | Trivedi et al. (2022), StonyBrookNLP/musique | CC BY 4.0 |
| FinanceBench | `PatronusAI/financebench` | Islam et al. (2023) | CC BY-NC 4.0 (non-commercial) |

The mirrors do not state a licence on the Hub.

## Citation

```bibtex
@unpublished{lee2026agentic,
  title  = {Agentic or Pipelined? A Confound-Controlled Empirical Study of an
            Architectural Decision in Retrieval-Augmented LLM Applications},
  author = {Lee, Ruo and Oh, Taeyeon},
  note   = {Manuscript under review},
  year   = {2026}
}
```

See also `CITATION.cff`.

## Licence

The code is released under the MIT licence (see `LICENSE`).
