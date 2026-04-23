"""
RAG strategy implementations.

Each strategy subclasses BaseStrategy and implements answer().
New strategies are registered here in STRATEGY_REGISTRY — the CLI
and runner resolve strategy names through this dict.

Phase 4 complete:
  naive      — dense-only (FAISS cosine)
  hybrid     — BM25 + dense, RRF fusion
  rerank     — dense candidates → Cohere cross-encoder reranking
  hyde       — Hypothetical Document Embeddings (extra LLM call)
  multi_query — query expansion + deduplicated dense retrieval
"""

from __future__ import annotations

from rag_eval.config import Config
from rag_eval.indexer import RAGIndex
from rag_eval.strategies.base import BaseStrategy
from rag_eval.strategies.hybrid import HybridRAG
from rag_eval.strategies.hyde import HyDERAG
from rag_eval.strategies.multi_query import MultiQueryRAG
from rag_eval.strategies.naive import NaiveRAG
from rag_eval.strategies.rerank import RerankRAG

# Registry — maps strategy name → class
STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "naive": NaiveRAG,
    "hybrid": HybridRAG,
    "rerank": RerankRAG,
    "hyde": HyDERAG,
    "multi_query": MultiQueryRAG,
}


def get_strategy(name: str, cfg: Config, index: RAGIndex) -> BaseStrategy:
    """
    Instantiate a strategy by name.

    Args:
        name:  Strategy name (naive | hybrid | rerank | hyde | multi_query).
        cfg:   Full Config from loaded YAML.
        index: Loaded RAGIndex (FAISS + BM25).

    Raises:
        ValueError: If the strategy name is unknown.
    """
    if name not in STRATEGY_REGISTRY:
        implemented = sorted(STRATEGY_REGISTRY)
        raise ValueError(f"Unknown strategy '{name}'.\nAvailable: {implemented}")
    return STRATEGY_REGISTRY[name](cfg, index)
