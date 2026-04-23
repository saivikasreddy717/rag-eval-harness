"""
Cohere reranker provider for rag-eval-harness.

Returns a callable that takes (query, documents, top_n) and returns a list of
(original_index, relevance_score) tuples sorted by descending score.

Only Cohere is supported today; the interface is kept provider-agnostic so a
cross-encoder reranker (e.g. via sentence-transformers) can be added later.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from rag_eval.config import RerankerConfig

# Type alias for the reranker callable
Reranker = Callable[[str, list[str], int], list[tuple[int, float]]]


def get_reranker(cfg: RerankerConfig) -> Reranker:
    """
    Build and return a reranker callable from the config.

    Args:
        cfg: RerankerConfig — provider, model, top_n.

    Returns:
        A function ``rerank(query, documents, top_n) -> [(index, score), ...]``
        sorted by relevance_score descending.

    Raises:
        ValueError:       If the provider is unsupported or cfg.enabled is False.
        EnvironmentError: If the required API key is missing.
        ImportError:      If the provider SDK is not installed.
    """
    if not cfg.enabled:
        raise ValueError(
            "Reranker is disabled (reranker.enabled = false in config). "
            "Enable it or use a strategy that does not require reranking."
        )

    if cfg.provider == "cohere":
        return _build_cohere_reranker(cfg.model)

    raise ValueError(f"Unknown reranker provider: '{cfg.provider}'. Supported: ['cohere']")


def _build_cohere_reranker(model: str) -> Reranker:
    """Build a Cohere reranker callable (requires COHERE_API_KEY)."""
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise OSError(
            "COHERE_API_KEY environment variable is not set.\n"
            "Add COHERE_API_KEY=<your-key> to your .env file.\n"
            "Free keys available at https://cohere.com (10M tokens/month)"
        )

    try:
        import cohere
    except ImportError as exc:
        raise ImportError("cohere package is not installed.\nRun: uv add cohere") from exc

    # cohere >= 5.0 uses ClientV2
    client = cohere.ClientV2(api_key=api_key)

    def _rerank(
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[tuple[int, float]]:
        """
        Rerank documents for a query.

        Returns:
            List of (original_index, relevance_score) sorted by score
            descending, length == min(top_n, len(documents)).
        """
        response = client.rerank(
            model=model,
            query=query,
            documents=documents,
            top_n=top_n,
        )
        return [(r.index, r.relevance_score) for r in response.results]

    return _rerank
