"""
RAG strategy implementations.

Each strategy subclasses BaseStrategy and implements answer().
New strategies are registered here in STRATEGY_REGISTRY — the CLI
and runner resolve strategy names through this dict.
"""
from __future__ import annotations

from rag_eval.config import Config
from rag_eval.indexer import RAGIndex
from rag_eval.strategies.base import BaseStrategy
from rag_eval.strategies.naive import NaiveRAG

# Registry — add new strategies here as they are implemented
STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "naive": NaiveRAG,
    # Phase 4: hybrid, rerank, hyde, multi_query
}


def get_strategy(name: str, cfg: Config, index: RAGIndex) -> BaseStrategy:
    """
    Instantiate a strategy by name.

    Args:
        name:  Strategy name (naive | hybrid | rerank | hyde | multi_query).
        cfg:   Full Config from loaded YAML.
        index: Loaded RAGIndex (FAISS + BM25).

    Raises:
        ValueError: If the strategy name is not yet implemented.
    """
    if name not in STRATEGY_REGISTRY:
        implemented = sorted(STRATEGY_REGISTRY)
        raise ValueError(
            f"Strategy '{name}' is not implemented yet.\n"
            f"Available: {implemented}"
        )
    return STRATEGY_REGISTRY[name](cfg, index)
