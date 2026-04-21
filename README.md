# rag-eval-harness

> Benchmark RAG retrieval strategies head-to-head with RAGAS metrics.
> Swap LLMs and embedding models via config — zero code changes required.

[![CI](https://github.com/saivikasreddy717/rag-eval-harness/actions/workflows/benchmark.yml/badge.svg)](https://github.com/saivikasreddy717/rag-eval-harness/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What this is

Most teams ship RAG systems without knowing which retrieval strategy actually works best on their data. This harness runs **5 strategies** on the same dataset, scores each with **RAGAS metrics**, and outputs a cost/latency/quality scorecard so you can make an informed decision before going to production.

## Strategies compared

| Strategy | Description |
|----------|-------------|
| `naive` | Dense retrieval only, cosine similarity, top-k |
| `hybrid` | BM25 + dense retrieval fused with Reciprocal Rank Fusion |
| `rerank` | Hybrid retrieval, then Cohere/ColBERT reranker |
| `hyde` | Hypothetical Document Embeddings: embed a generated answer, not the query |
| `multi_query` | Generate 3 query variants, union results, deduplicate |

## Metrics scored

| Metric | What it measures |
|--------|-----------------|
| Faithfulness | Is the answer grounded in retrieved context? (0-1) |
| Answer Relevancy | Is the answer on-topic? (0-1) |
| Context Precision | Are retrieved chunks actually relevant? (0-1) |
| Context Recall | Did retrieval capture everything needed? (0-1) |
| Latency (p50/p95) | Per-query response time in ms |
| Cost (USD/1K queries) | Estimated API cost |

## Quick start

```bash
git clone https://github.com/saivikasreddy717/rag-eval-harness
cd rag-eval-harness

# Install dependencies
uv sync

# Add your API keys (Groq + Cohere, both free)
cp .env.example .env

# Run full benchmark
make benchmark

# Or step by step:
make index    # build FAISS index from dataset
make run      # generate predictions with all 5 strategies
make eval     # score with RAGAS
make compare  # generate results.csv + report.html
```

## Swapping models

Change 2 lines in your config YAML. No code changes.

```bash
# Use OpenAI instead of Groq
python -m rag_eval --config configs/openai.yaml benchmark

# Run fully local with Ollama (no API keys needed)
python -m rag_eval --config configs/ollama_local.yaml benchmark

# Use Google Gemini
python -m rag_eval --config configs/google_gemini.yaml benchmark
```

### Available config presets

| Config | Generator | Judge | Embeddings | Cost |
|--------|-----------|-------|------------|------|
| `groq_llama4.yaml` | Llama 4 Scout (Groq) | Llama 3.3 70B (Groq) | BGE-large (local) | ~$0 |
| `openai.yaml` | GPT-4o-mini | GPT-4o | text-embedding-3-small | ~$12-18 |
| `ollama_local.yaml` | Llama 3.1 (local) | Llama 3.1 (local) | BGE-large (local) | $0 |
| `google_gemini.yaml` | Gemini 2.0 Flash | Gemini 2.0 Flash | BGE-large (local) | ~$0 |

### Run on your own corpus

```yaml
# configs/my_corpus.yaml
dataset:
  name: custom         # points to your dataset
  sample_size: 200

generator:
  provider: groq
  model: meta-llama/llama-4-scout-17b-16e-instruct

embeddings:
  provider: local
  model: BAAI/bge-large-en-v1.5
```

## Results

*Benchmark results on HotpotQA (500 questions) — coming after full run.*

## Project structure

```
rag-eval-harness/
├── configs/             # Swap providers here, no code changes needed
├── src/rag_eval/
│   ├── providers/       # LLM + embeddings factories (model-agnostic layer)
│   ├── strategies/      # naive, hybrid, rerank, hyde, multi_query
│   ├── config.py        # Pydantic config validation
│   ├── cli.py           # CLI entry point
│   ├── datasets.py      # Dataset loading + caching
│   ├── indexer.py       # FAISS + BM25 index building
│   ├── runner.py        # Benchmark orchestration
│   ├── evaluator.py     # RAGAS scoring
│   └── report.py        # HTML report generation
├── tests/
└── results/             # Scorecard CSV + HTML report (gitignored)
```

## License

MIT. See [LICENSE](LICENSE).

---

Built by [Sai Vikas Reddy Yeddulamala](https://github.com/saivikasreddy717)
