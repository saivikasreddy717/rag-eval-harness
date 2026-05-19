# rag-eval-harness

> Benchmark RAG retrieval strategies head-to-head with RAGAS metrics.  
> Swap LLMs and embedding models via YAML config — zero code changes required.

[![CI](https://github.com/saivikasreddy717/rag-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/saivikasreddy717/rag-eval-harness/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why this exists

Most teams ship RAG systems without knowing which retrieval strategy actually performs best on *their* data. The common trap: pick `naive` dense retrieval because it's the default, ship to production, and discover six months later that a hybrid or re-ranking approach would have cut hallucinations in half.

This harness runs **5 retrieval strategies** against the same question set, scores each with **5 RAGAS metrics**, and produces a side-by-side scorecard with cost and latency data. The goal is to make "which strategy should I use?" an empirical question with a three-command answer.

**Design principles:**
- **Model-agnostic** — swap Groq → OpenAI → Ollama → Gemini by changing two lines in a YAML file
- **Free by default** — ships with a Groq + local-embeddings config that costs $0 to run the full 500-question benchmark
- **CI-safe tests** — all 115 tests mock out LLM/embeddings calls; no API keys required in CI
- **Reproducible** — fixed random seed, pinned dependencies via `uv.lock`

---

## The 5 strategies

### 1. Naive (dense-only)

The baseline. Embed the question with BGE-large, run FAISS cosine similarity search, pass the top-k chunks to the LLM.

**When it works well:** semantically-phrased questions where the answer passage uses similar vocabulary to the question.  
**When it breaks:** keyword-heavy queries (proper names, acronyms, code), where the question and answer embeddings land far apart despite being topically identical.

### 2. Hybrid (BM25 + dense, RRF fusion)

Runs both a dense FAISS search and a BM25 sparse search simultaneously, then merges the two ranked lists using **Reciprocal Rank Fusion**:

```
RRF_score(d) = Σ  1 / (k + rank(d))
              lists
```

Each document accumulates `1/(k + rank)` contributions from every list it appears in. Documents that rank well in both systems score highest. The constant `k=60` prevents a rank-1 result from dominating when the other list disagrees.

**Why this beats naive:** BM25 is an exact-match powerhouse — it finds passages containing the exact words from the query. Dense retrieval finds paraphrases and semantic neighbors. RRF combines both signals without requiring their incompatible score scales to be calibrated.  
**Tradeoff:** two retrieval calls instead of one (~15 ms extra latency).

### 3. Rerank (dense → Cohere cross-encoder)

Fetches a large candidate pool (top-20) from FAISS, then sends all 20 chunks to Cohere's `rerank-english-v3.0` cross-encoder. The cross-encoder reads the question and each chunk *together* and produces a true relevance score — far more accurate than a bi-encoder's dot product, but too slow to run over the whole index.

**Why this is accurate:** bi-encoders compress question and passage into separate vectors before comparing them. Cross-encoders see both simultaneously, which lets them catch fine-grained relevance cues like negations, pronoun referents, and multi-hop reasoning.  
**Tradeoff:** one extra network round-trip to Cohere per query; requires `COHERE_API_KEY` (free tier: 1,000 reranks/month).

### 4. HyDE (Hypothetical Document Embeddings)

Before touching the index, HyDE asks the LLM to write a *hypothetical* answer passage:

```
"Write a short factual passage that directly answers: {question}"
```

It then embeds that hypothetical passage (not the original question) and uses it for FAISS search. The intuition: a generated answer-like passage lives in the same embedding neighbourhood as real answer passages — much closer than a short factual question.

**Why this helps on factoid QA:** short questions ("What caused the 2008 financial crisis?") and long explanatory passages embed very differently. A 3-sentence hypothetical answer bridges that gap.  
**Tradeoff:** one extra LLM call per query (+~300 ms, ~$0.00004 on Groq). Can backfire if the hypothesis hallucinates a wrong direction.

*Reference: Gao et al. (2022) — [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)*

### 5. Multi-Query (query expansion + deduplication)

Asks the LLM to rephrase the question into 3 alternative formulations targeting different facets. Then runs FAISS search for each phrasing, merges the result sets, and deduplicates by `chunk_id`. The original question is always queried first so no primary signal is lost.

**Why this increases recall:** a multi-faceted question like "What were the political and economic effects of the French Revolution?" benefits from phrasings that emphasise different sub-topics. Each phrasing retrieves a different set of passages; merging them fills in gaps that any single query misses.  
**Tradeoff:** N+1 embedding calls + one extra LLM call per query.

---

## Agent Evaluation

Beyond benchmarking retrieval strategies, the harness can score **agent traces** — sequences of tool calls and reasoning steps produced by an AI agent. Enable this with `eval-agent`:

```bash
rag-eval --config configs/groq_llama4.yaml eval-agent \
    --traces examples/agent_traces_sample.jsonl
```

This writes `results/scorecard_agent.csv` with one row per trace.

### The 3 agent metrics

| Metric | What it measures | How | Target |
|--------|-----------------|-----|--------|
| **Tool Selection Accuracy** | Did the agent pick the correct tool at each labelled decision point? | Label comparison against `expected_tool` — no LLM required | >0.92 (binary), >0.85 (5+ tools) |
| **Tool Execution Success** | What fraction of tool calls completed without error? | Reads `tool_call.success` from the trace | >0.98 in production |
| **Multi-Step Coherence** | Does each step build logically on prior steps? Does the final answer reflect the full chain? | LLM-as-judge scores 0–1 per trace using the configured judge model | >0.85 on 4+ step traces |

Tool selection and execution are pure label comparisons — zero LLM calls, zero cost. Coherence is the only metric that calls the judge LLM (one call per trace).

Thresholds sourced from: Pratik R, *"Building an Evaluation Harness for Production AI Agents: A 12-Metric Framework From 100+ Deployments"*, Towards Data Science, May 2026.

### Trace format

Each line in the JSONL file is an `AgentTrace`. See [`examples/agent_traces_sample.jsonl`](examples/agent_traces_sample.jsonl) for 10 realistic examples covering single-step, multi-step, success, and failure cases.

To enable labelled eval (tool selection accuracy), add `expected_tool` to each tool-calling step:

```jsonl
{
  "trace_id": "t-001",
  "steps": [
    {
      "step_id": 0,
      "input": "Search for recent LLM papers.",
      "tool_call": {"name": "web_search", "args": {"query": "..."}, "success": true, "error_type": null},
      "tool_result": "[\"Paper A\", \"Paper B\"]",
      "output": "Found 2 papers.",
      "timestamp": "2026-05-18T10:01:00Z",
      "expected_tool": "web_search"
    }
  ],
  "final_output": "Found 2 recent papers."
}
```

### Enabling in config

```yaml
agent_eval:
  enabled: true
  metrics:
    - tool_selection
    - tool_execution
    - coherence
  coherence_min_steps: 2   # skip traces shorter than this (no cross-step reasoning)
```

---

## Online Evaluation

The `eval-online` command samples a stream of **production traces** and scores them in near-real-time. It supports mixed streams — agent traces and RAG prediction records are automatically routed to the appropriate eval path.

```bash
rag-eval --config configs/groq_llama4.yaml eval-online \
    --traces-stream examples/agent_traces_sample.jsonl \
    --sample-rate 0.05
```

This writes:
- `results/online/YYYY-MM-DD.jsonl` — scored records for the current day
- `results/online/rollup_YYYY-MM-DD.csv` — daily aggregates: mean, p50, p95, p99 per metric

The default 5% sample rate matches the recommendation from the TDS "12-Metric Framework" article. The sampler uses reservoir sampling so memory stays bounded regardless of stream length. Passing `--sample-rate 1.0` scores every record (useful for offline re-evaluation).

### Config

```yaml
online:
  enabled: true
  sample_rate: 0.05       # fraction of records to score
  output_dir: results/online
  rollup_interval: daily
  seed: 42                # fixed seed for deterministic sampling
```

---

## Metrics scored

### Core RAGAS metrics (RAG benchmark)

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| **Faithfulness** | Is every claim in the answer grounded in the retrieved context? | Catches hallucinations — answers that sound correct but aren't supported by sources |
| **Answer Relevancy** | Does the answer address the question asked? | Catches off-topic or evasive answers |
| **Context Precision** | Are the retrieved chunks actually relevant to the question? | Measures retrieval signal-to-noise ratio |
| **Context Recall** | Did retrieval capture all the information needed to answer? | Measures retrieval completeness |
| **Answer Correctness** | Is the final answer factually correct (vs. ground truth)? | End-to-end quality: retrieval × generation combined |

All metrics are scored 0–1 by a separate judge LLM (Llama 3.3 70B on Groq by default). Per-question scores are written to a scorecard CSV; the `compare` command aggregates across all strategies and generates an HTML report with charts.

### Optional extra metrics (opt-in via `metrics_extra` config)

Enable any combination in your YAML:

```yaml
metrics_extra:
  hallucination_rate: true    # fraction of answer claims not in context or common knowledge
  context_relevance: true     # per-chunk LLM relevance score (distinct from RAGAS precision)
  retrieval_latency: true     # adds p50/p95/p99 of retrieval-only time to AGGREGATE row
```

| Metric | What it measures | Notes |
|--------|-----------------|-------|
| **Hallucination Rate** | Fraction of answer claims that are neither grounded in retrieved context nor verifiable as common knowledge | LLM judge; target <2% general, <0.5% regulated domains |
| **Context Relevance** | Mean LLM-judge relevance score across all retrieved chunks, per query | Distinct from RAGAS context_precision (MRR-based); gives an absolute per-chunk score |
| **Retrieval Latency p95** | 95th-percentile of retrieval-only time (embed + search, excluding LLM generation) | Surfaced as `retrieval_latency_p95_ms` in the AGGREGATE scorecard row |

---

## Metrics coverage table

The table below maps this harness to the 12-metric framework from Pratik R's TDS article (May 2026). ✅ = implemented, 🔜 = planned.

| Category | Metric | Status | Command |
|----------|--------|--------|---------|
| **Retrieval** | Context Precision | ✅ | `eval` |
| **Retrieval** | Context Recall | ✅ | `eval` |
| **Retrieval** | Context Relevance (per-chunk) | ✅ | `eval` (opt-in) |
| **Retrieval** | Retrieval Latency p95 | ✅ | `eval` (opt-in) |
| **Generation** | Faithfulness | ✅ | `eval` |
| **Generation** | Answer Relevancy | ✅ | `eval` |
| **Generation** | Hallucination Rate | ✅ | `eval` (opt-in) |
| **Agent** | Tool Selection Accuracy | ✅ | `eval-agent` |
| **Agent** | Tool Execution Success | ✅ | `eval-agent` |
| **Agent** | Multi-Step Coherence | ✅ | `eval-agent` |
| **Production cost** | Cost per Query | 🔜 | — |
| **Production cost** | Cost per Correct Answer | 🔜 | — |

**Coverage after this release: 10 of 12.** Cost-per-query tracking is planned as future work.

---

## Quick start

### Prerequisites

```bash
# Free accounts needed:
# Groq (generator + judge):  https://console.groq.com  → GROQ_API_KEY
# Cohere (reranker only):     https://cohere.com        → COHERE_API_KEY
```

### Install

```bash
git clone https://github.com/saivikasreddy717/rag-eval-harness
cd rag-eval-harness

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Set API keys

```bash
cp .env.example .env
# Edit .env and add:
#   GROQ_API_KEY=gsk_...
#   COHERE_API_KEY=...   # only needed if running the rerank strategy
```

### Run the full benchmark (Makefile)

```bash
make benchmark          # runs all 4 steps end-to-end
```

Or step by step:

```bash
make index    # download HotpotQA, chunk docs, build FAISS + BM25 index
make run      # run all 5 strategies, write predictions/*.jsonl
make eval     # score each strategy with RAGAS → results/scorecard_*.csv
make compare  # merge scorecards → results/results.csv + results/report.html
```

### Or use the CLI directly

```bash
# RAG benchmark pipeline
rag-eval --config configs/groq_llama4.yaml index
rag-eval --config configs/groq_llama4.yaml run
rag-eval --config configs/groq_llama4.yaml eval
rag-eval --config configs/groq_llama4.yaml compare

# Run only specific strategies
rag-eval --config configs/groq_llama4.yaml run --strategy naive --strategy hybrid

# Limit questions for a quick test
rag-eval --config configs/groq_llama4.yaml run --max-questions 20
rag-eval --config configs/groq_llama4.yaml eval --max-questions 20

# Agent evaluation (tool selection, execution, coherence)
rag-eval --config configs/groq_llama4.yaml eval-agent \
    --traces examples/agent_traces_sample.jsonl

# Online evaluation (sampled production stream)
rag-eval --config configs/groq_llama4.yaml eval-online \
    --traces-stream examples/agent_traces_sample.jsonl \
    --sample-rate 1.0
```

---

## Swapping providers

Change two lines in your YAML. No code changes.

### Config presets included

| Config | Generator | Judge | Embeddings | Est. cost (500 Qs) |
|--------|-----------|-------|------------|---------------------|
| `groq_llama4.yaml` | Llama 4 Scout (Groq) | Llama 3.3 70B (Groq) | BGE-large-en-v1.5 (local) | **~$0** |
| `openai.yaml` | GPT-4o-mini | GPT-4o | text-embedding-3-small | ~$12–18 |
| `ollama_local.yaml` | Llama 3.1 8B (local) | Llama 3.1 8B (local) | BGE-large-en-v1.5 (local) | **$0** |
| `google_gemini.yaml` | Gemini 2.0 Flash | Gemini 2.0 Flash | BGE-large-en-v1.5 (local) | ~$0 |

### Install provider extras

```bash
uv sync --extra openai       # GPT-4o / text-embedding-3-small
uv sync --extra anthropic    # Claude 3.5 Sonnet / Haiku
uv sync --extra ollama       # Llama 3.1 local
uv sync --extra google       # Gemini 2.0
uv sync --all-extras         # everything
```

### Write your own config

```yaml
# configs/my_setup.yaml
dataset:
  name: hotpotqa        # or "custom" for your own corpus
  split: validation
  sample_size: 100      # number of questions to benchmark
  seed: 42

retrieval:
  chunk_size: 512
  chunk_overlap: 50
  top_k: 5              # chunks passed to the LLM
  top_k_rerank: 20      # candidates fetched before reranking

generator:
  provider: groq         # groq | openai | anthropic | ollama | google
  model: meta-llama/llama-4-scout-17b-16e-instruct
  temperature: 0.1
  max_tokens: 512

judge:
  provider: groq
  model: llama-3.3-70b-versatile
  temperature: 0.0

embeddings:
  provider: local        # local | openai
  model: BAAI/bge-large-en-v1.5

reranker:
  provider: cohere
  model: rerank-english-v3.0
  enabled: true          # set false to skip reranking in the rerank strategy

strategies:
  - naive
  - hybrid
  - rerank
  - hyde
  - multi_query

output:
  dir: results
```

---

## Output

After `make compare`, the `results/` directory contains:

```
results/
├── scorecard_naive.csv        # per-question scores: faithfulness, relevancy, ...
├── scorecard_hybrid.csv
├── scorecard_rerank.csv
├── scorecard_hyde.csv
├── scorecard_multi_query.csv
├── results.csv                # one row per strategy, AGGREGATE scores only
└── report.html                # interactive Plotly dashboard
```

`report.html` is a self-contained file (Plotly loaded from CDN) with:
- Grouped bar chart: all 5 metrics × all strategies
- Radar chart: per-strategy quality profile
- Latency bar chart: p50 response time comparison
- Cost scatter: answer correctness vs. cost per 1K queries
- Summary table: sortable, highlights the best strategy per metric

---

## Benchmark results

*Full 500-question run on HotpotQA (validation split) — coming soon.*

Preliminary results on 50-question spot check (Groq free tier):

| Strategy | Faithfulness | Ans. Relevancy | Ctx. Precision | Ctx. Recall | Correctness | Latency p50 |
|----------|:-----------:|:--------------:|:--------------:|:-----------:|:-----------:|:-----------:|
| naive | — | — | — | — | — | — |
| hybrid | — | — | — | — | — | — |
| rerank | — | — | — | — | — | — |
| hyde | — | — | — | — | — | — |
| multi_query | — | — | — | — | — | — |

*Numbers will be filled in after the full benchmark run is published.*

---

## Project structure

```
rag-eval-harness/
├── configs/                    # Provider presets (groq, openai, ollama, gemini)
├── examples/
│   └── agent_traces_sample.jsonl  # 10 realistic agent traces for eval-agent
├── src/rag_eval/
│   ├── providers/
│   │   ├── llm.py              # LLM factory — returns LangChain ChatModel
│   │   ├── embeddings.py       # Embeddings factory — local BGE or OpenAI
│   │   └── reranker.py         # Cohere cross-encoder reranker factory
│   ├── strategies/
│   │   ├── base.py             # BaseStrategy: generate(), embed_query(), raw_llm_call()
│   │   ├── naive.py            # Dense-only FAISS retrieval
│   │   ├── hybrid.py           # BM25 + dense + RRF fusion
│   │   ├── rerank.py           # Dense candidates → Cohere cross-encoder
│   │   ├── hyde.py             # Hypothetical Document Embeddings
│   │   └── multi_query.py      # Query expansion + deduplication
│   ├── agent_eval/             # Agent evaluation sub-package (eval-agent command)
│   │   ├── trace.py            # Pydantic v2 AgentTrace / AgentStep / ToolCall models
│   │   ├── fixtures.py         # Mock trace generators for tests
│   │   ├── evaluator.py        # Orchestrates agent metrics → scorecard_agent.csv
│   │   └── metrics/
│   │       ├── tool_selection.py   # Accuracy vs expected_tool labels
│   │       ├── tool_execution.py   # Success rate + error categorisation
│   │       └── coherence.py        # LLM-as-judge multi-step coherence
│   ├── online/                 # Online evaluation sub-package (eval-online command)
│   │   ├── sampler.py          # Reservoir sampler (deterministic, bounded memory)
│   │   ├── runner.py           # Mixed-stream router → scored JSONL
│   │   └── storage.py          # Append JSONL + daily rollup CSV
│   ├── config.py               # Pydantic v2 config schema + YAML loader
│   ├── cli.py                  # Click CLI: index / run / eval / compare / eval-agent / eval-online
│   ├── datasets.py             # HuggingFace dataset loading + caching
│   ├── indexer.py              # FAISS IndexFlatIP + BM25Okapi builder
│   ├── runner.py               # Orchestrate strategy runs, write JSONL predictions
│   ├── evaluator.py            # RAGAS + extra metrics → scorecard CSV
│   └── reporter.py             # load_scorecards → comparison matrix → HTML report
├── tests/
│   ├── conftest.py             # CI safety: TOKENIZERS_PARALLELISM=false
│   ├── test_phase1.py          # Config, indexer, dataset tests
│   ├── test_phase2.py          # Strategy registry, base class, naive strategy
│   ├── test_phase3.py          # Evaluator + RAGAS scoring (fully mocked)
│   ├── test_phase4.py          # Hybrid/rerank/HyDE/multi-query strategies
│   ├── test_phase5.py          # Reporter: load, matrix, charts, HTML, CLI compare
│   ├── test_smoke.py           # CLI smoke tests
│   ├── test_agent_eval.py      # Agent trace schema + 3 agent metrics + orchestrator
│   ├── test_online_eval.py     # Sampler + storage + runner
│   └── test_new_rag_metrics.py # Hallucination rate + context relevance + latency p95
├── .github/workflows/ci.yml    # Python 3.10 + 3.12 matrix, uv cache, HF model cache
├── Makefile                    # index / run / eval / compare / benchmark / test
└── pyproject.toml              # Dependencies, ruff config, pytest filterwarnings
```

---

## Running tests

```bash
# Run all ~145 tests (no API keys needed)
make test

# Verbose with short tracebacks
uv run pytest tests/ -v --tb=short

# Run only the new agent / online / extra-metrics tests
uv run pytest tests/test_agent_eval.py tests/test_online_eval.py tests/test_new_rag_metrics.py -v
```

All tests mock out LLM and embeddings calls. The BGE-large model is loaded once during tests that exercise the real embedding path (Phase 1 index tests); subsequent runs use the HuggingFace local cache.

CI runs on every push to `main` across **Python 3.10 and 3.12** via GitHub Actions.

---

## Tech stack

| Component | Library | Why |
|-----------|---------|-----|
| Dense retrieval | `faiss-cpu` (IndexFlatIP + L2-norm = cosine) | Fast exact search; no approximate index needed at <100k chunks |
| Sparse retrieval | `rank-bm25` (BM25Okapi) | Pure Python, zero deps, good enough for benchmarking |
| Embeddings | `sentence-transformers` (BGE-large-en-v1.5) | Strong open-source bi-encoder; free, runs locally |
| Reranking | `cohere` ClientV2 (rerank-english-v3.0) | Best free-tier cross-encoder available via API |
| LLM routing | `langchain` + `langchain-groq/openai/...` | Swap providers without touching strategy code |
| RAGAS scoring | `ragas>=0.2.0` (EvaluationDataset API) | Industry-standard RAG evaluation metrics |
| Config | `pydantic v2` + YAML | Type-validated, self-documenting, IDE-friendly |
| Charts | `plotly` (CDN-embedded HTML) | Self-contained report, no server required |
| Package mgmt | `uv` + `uv.lock` | Fast, reproducible installs |

---

## License

MIT. See [LICENSE](LICENSE).

---

Built by [Sai Vikas Reddy Yeddulamala](https://github.com/saivikasreddy717)
