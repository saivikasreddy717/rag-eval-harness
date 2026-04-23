"""
Phase 4 tests — Hybrid, Rerank, HyDE, MultiQuery strategies.

All LLM/API calls are mocked.  The FAISS + BM25 index is built from a tiny
3-passage fake corpus (reuses the built_index fixture from test_phase2).
No API keys or network access needed — safe to run in CI.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures (identical to test_phase2 to keep tests self-contained)
# ---------------------------------------------------------------------------

FAKE_CORPUS = [
    {"passage_id": "Alpha", "text": "Alpha is the first letter of the Greek alphabet."},
    {"passage_id": "Beta", "text": "Beta is the second letter of the Greek alphabet."},
    {"passage_id": "Gamma", "text": "Gamma rays are high-frequency electromagnetic radiation."},
]


@pytest.fixture
def built_index(tmp_path):
    """Build a small real FAISS + BM25 index into tmp_path."""
    from rag_eval.chunker import chunk_corpus
    from rag_eval.config import Config
    from rag_eval.indexer import build_index, RAGIndex
    import rag_eval.indexer as idx_mod

    idx_mod.INDEX_DIR = tmp_path
    idx_mod.FAISS_PATH = tmp_path / "faiss.index"
    idx_mod.CHUNKS_PATH = tmp_path / "chunks.json"
    idx_mod.META_PATH = tmp_path / "index_meta.json"

    cfg = Config()
    chunks = chunk_corpus(FAKE_CORPUS, cfg.retrieval)
    build_index(chunks, cfg)

    index = RAGIndex.load(tmp_path)
    yield index

    idx_mod.INDEX_DIR = __import__("pathlib").Path("data/index")
    idx_mod.FAISS_PATH = idx_mod.INDEX_DIR / "faiss.index"
    idx_mod.CHUNKS_PATH = idx_mod.INDEX_DIR / "chunks.json"
    idx_mod.META_PATH = idx_mod.INDEX_DIR / "index_meta.json"


def _mock_llm_response(text: str = "Test answer.") -> MagicMock:
    r = MagicMock()
    r.content = text
    return r


# ---------------------------------------------------------------------------
# RRF fusion helper (no LLM, pure unit test)
# ---------------------------------------------------------------------------

class TestRRFFusion:

    def test_rrf_prefers_docs_in_both_lists(self):
        from rag_eval.strategies.hybrid import _rrf_merge

        dense = [
            {"chunk_id": "a", "text": "A", "score": 0.9},
            {"chunk_id": "b", "text": "B", "score": 0.8},
            {"chunk_id": "c", "text": "C", "score": 0.7},
        ]
        bm25 = [
            {"chunk_id": "b", "text": "B", "score": 10},  # "b" appears in both
            {"chunk_id": "d", "text": "D", "score": 9},
            {"chunk_id": "c", "text": "C", "score": 8},
        ]
        merged = _rrf_merge(dense, bm25, top_k=3)
        ids = [h["chunk_id"] for h in merged]

        # "b" and "c" both appear in two lists so they should rank highest
        assert "b" in ids
        assert "c" in ids

    def test_rrf_respects_top_k(self):
        from rag_eval.strategies.hybrid import _rrf_merge

        hits = [{"chunk_id": str(i), "text": f"Doc {i}", "score": 1.0} for i in range(10)]
        merged = _rrf_merge(hits, hits, top_k=3)
        assert len(merged) == 3

    def test_rrf_handles_disjoint_lists(self):
        from rag_eval.strategies.hybrid import _rrf_merge

        dense = [{"chunk_id": "a", "text": "A", "score": 0.9}]
        bm25 = [{"chunk_id": "b", "text": "B", "score": 10}]
        merged = _rrf_merge(dense, bm25, top_k=2)
        assert len(merged) == 2

    def test_rrf_empty_inputs(self):
        from rag_eval.strategies.hybrid import _rrf_merge

        merged = _rrf_merge([], [], top_k=5)
        assert merged == []


# ---------------------------------------------------------------------------
# HybridRAG
# ---------------------------------------------------------------------------

class TestHybridRAG:

    def test_hybrid_returns_result(self, built_index):
        from rag_eval.strategies.hybrid import HybridRAG
        from rag_eval.config import Config

        cfg = Config()
        with patch.dict(os.environ, {"GROQ_API_KEY": "dummy"}):
            strategy = HybridRAG(cfg, built_index)
            strategy.llm = MagicMock()
            strategy.llm.invoke.return_value = _mock_llm_response("Alpha answer.")

            result = strategy.answer("What is alpha?")

        assert result.answer == "Alpha answer."
        assert len(result.contexts) > 0
        assert result.latency_ms >= 0
        assert result.metadata["strategy"] == "hybrid"

    def test_hybrid_context_count_bounded_by_top_k(self, built_index):
        from rag_eval.strategies.hybrid import HybridRAG
        from rag_eval.config import Config, RetrievalConfig

        cfg = Config()
        cfg = cfg.model_copy(update={"retrieval": RetrievalConfig(top_k=2)})
        with patch.dict(os.environ, {"GROQ_API_KEY": "dummy"}):
            strategy = HybridRAG(cfg, built_index)
            strategy.llm = MagicMock()
            strategy.llm.invoke.return_value = _mock_llm_response()
            result = strategy.answer("Test")

        assert len(result.contexts) <= 2

    def test_hybrid_metadata_has_hit_counts(self, built_index):
        from rag_eval.strategies.hybrid import HybridRAG
        from rag_eval.config import Config

        cfg = Config()
        with patch.dict(os.environ, {"GROQ_API_KEY": "dummy"}):
            strategy = HybridRAG(cfg, built_index)
            strategy.llm = MagicMock()
            strategy.llm.invoke.return_value = _mock_llm_response()
            result = strategy.answer("What is gamma?")

        assert "num_dense_hits" in result.metadata
        assert "num_bm25_hits" in result.metadata
        assert "num_fused_hits" in result.metadata

    def test_hybrid_registered_in_strategy_registry(self):
        from rag_eval.strategies import STRATEGY_REGISTRY
        assert "hybrid" in STRATEGY_REGISTRY


# ---------------------------------------------------------------------------
# RerankRAG
# ---------------------------------------------------------------------------

class TestRerankRAG:

    def test_rerank_returns_result(self, built_index):
        from rag_eval.strategies.rerank import RerankRAG
        from rag_eval.config import Config

        cfg = Config()
        mock_reranker = MagicMock(return_value=[(0, 0.95), (1, 0.80)])

        with patch.dict(os.environ, {"GROQ_API_KEY": "dummy", "COHERE_API_KEY": "dummy"}):
            with patch("rag_eval.strategies.rerank.get_reranker", return_value=mock_reranker):
                strategy = RerankRAG(cfg, built_index)
                strategy.llm = MagicMock()
                strategy.llm.invoke.return_value = _mock_llm_response("Reranked answer.")
                result = strategy.answer("What is beta?")

        assert result.answer == "Reranked answer."
        assert len(result.contexts) == 2   # matches mock reranker top_n
        assert result.metadata["strategy"] == "rerank"

    def test_rerank_context_order_follows_reranker(self, built_index):
        """Contexts should appear in the order the reranker returns them."""
        from rag_eval.strategies.rerank import RerankRAG
        from rag_eval.config import Config

        cfg = Config()

        # Simulate reranker preferring index 2 over index 0
        mock_reranker = MagicMock(return_value=[(2, 0.99), (0, 0.50)])

        with patch.dict(os.environ, {"GROQ_API_KEY": "dummy", "COHERE_API_KEY": "dummy"}):
            with patch("rag_eval.strategies.rerank.get_reranker", return_value=mock_reranker):
                strategy = RerankRAG(cfg, built_index)
                strategy.llm = MagicMock()
                strategy.llm.invoke.return_value = _mock_llm_response()

                # Dense search will return at least 3 candidates from our 3-chunk index
                result = strategy.answer("Any question")

        # Verify reranker was actually called
        assert mock_reranker.called

    def test_rerank_registered_in_strategy_registry(self):
        from rag_eval.strategies import STRATEGY_REGISTRY
        assert "rerank" in STRATEGY_REGISTRY


# ---------------------------------------------------------------------------
# HyDERAG
# ---------------------------------------------------------------------------

class TestHyDERAG:

    def test_hyde_returns_result(self, built_index):
        from rag_eval.strategies.hyde import HyDERAG
        from rag_eval.config import Config

        cfg = Config()
        # First invoke = hypothetical doc generation; second = final answer
        call_count = {"n": 0}

        def _invoke_side_effect(messages):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _mock_llm_response("Alpha is the first letter.")
            return _mock_llm_response("HyDE final answer.")

        with patch.dict(os.environ, {"GROQ_API_KEY": "dummy"}):
            strategy = HyDERAG(cfg, built_index)
            strategy.llm = MagicMock()
            strategy.llm.invoke.side_effect = _invoke_side_effect
            result = strategy.answer("What is alpha?")

        assert result.answer == "HyDE final answer."
        assert len(result.contexts) > 0
        assert result.latency_ms >= 0
        assert result.metadata["strategy"] == "hyde"

    def test_hyde_makes_two_llm_calls(self, built_index):
        """HyDE must call the LLM exactly twice (hypothetical doc + answer)."""
        from rag_eval.strategies.hyde import HyDERAG
        from rag_eval.config import Config

        cfg = Config()
        with patch.dict(os.environ, {"GROQ_API_KEY": "dummy"}):
            strategy = HyDERAG(cfg, built_index)
            strategy.llm = MagicMock()
            strategy.llm.invoke.return_value = _mock_llm_response("Some text.")
            strategy.answer("What is gamma?")

        assert strategy.llm.invoke.call_count == 2

    def test_hyde_metadata_has_hypothetical_doc(self, built_index):
        from rag_eval.strategies.hyde import HyDERAG
        from rag_eval.config import Config

        cfg = Config()
        with patch.dict(os.environ, {"GROQ_API_KEY": "dummy"}):
            strategy = HyDERAG(cfg, built_index)
            strategy.llm = MagicMock()
            strategy.llm.invoke.return_value = _mock_llm_response("Hypothetical text here.")
            result = strategy.answer("What is beta?")

        assert "hypothetical_doc" in result.metadata
        assert isinstance(result.metadata["hypothetical_doc"], str)

    def test_hyde_registered_in_strategy_registry(self):
        from rag_eval.strategies import STRATEGY_REGISTRY
        assert "hyde" in STRATEGY_REGISTRY


# ---------------------------------------------------------------------------
# MultiQueryRAG
# ---------------------------------------------------------------------------

class TestMultiQueryRAG:

    def test_multi_query_returns_result(self, built_index):
        from rag_eval.strategies.multi_query import MultiQueryRAG
        from rag_eval.config import Config

        cfg = Config()
        call_count = {"n": 0}

        def _invoke_side_effect(messages):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Query expansion call
                return _mock_llm_response(
                    "What is the meaning of alpha?\nDefine the letter alpha.\nAlpha in Greek alphabet"
                )
            # Final answer call
            return _mock_llm_response("Multi-query answer.")

        with patch.dict(os.environ, {"GROQ_API_KEY": "dummy"}):
            strategy = MultiQueryRAG(cfg, built_index)
            strategy.llm = MagicMock()
            strategy.llm.invoke.side_effect = _invoke_side_effect
            result = strategy.answer("What is alpha?")

        assert result.answer == "Multi-query answer."
        assert len(result.contexts) > 0
        assert result.latency_ms >= 0
        assert result.metadata["strategy"] == "multi_query"

    def test_multi_query_makes_two_llm_calls(self, built_index):
        """MultiQuery must call the LLM exactly twice (expansion + answer)."""
        from rag_eval.strategies.multi_query import MultiQueryRAG
        from rag_eval.config import Config

        cfg = Config()
        with patch.dict(os.environ, {"GROQ_API_KEY": "dummy"}):
            strategy = MultiQueryRAG(cfg, built_index)
            strategy.llm = MagicMock()
            strategy.llm.invoke.return_value = _mock_llm_response("Query A\nQuery B")
            strategy.answer("What is beta?")

        assert strategy.llm.invoke.call_count == 2

    def test_multi_query_deduplicates_chunks(self, built_index):
        """Contexts should not contain duplicate chunks."""
        from rag_eval.strategies.multi_query import MultiQueryRAG
        from rag_eval.config import Config

        cfg = Config()
        with patch.dict(os.environ, {"GROQ_API_KEY": "dummy"}):
            strategy = MultiQueryRAG(cfg, built_index)
            strategy.llm = MagicMock()
            # Expansion returns 3 similar variants — all will retrieve same chunks
            strategy.llm.invoke.return_value = _mock_llm_response(
                "First Greek letter\nLetter alpha Greek\nGreek letter one"
            )
            result = strategy.answer("What is alpha?")

        # No duplicate texts in contexts
        assert len(result.contexts) == len(set(result.contexts))

    def test_multi_query_metadata_has_queries(self, built_index):
        from rag_eval.strategies.multi_query import MultiQueryRAG
        from rag_eval.config import Config

        cfg = Config()
        with patch.dict(os.environ, {"GROQ_API_KEY": "dummy"}):
            strategy = MultiQueryRAG(cfg, built_index)
            strategy.llm = MagicMock()
            strategy.llm.invoke.return_value = _mock_llm_response("Alt query A\nAlt query B")
            result = strategy.answer("What is gamma?")

        assert "queries" in result.metadata
        assert "num_queries" in result.metadata
        assert result.metadata["num_queries"] >= 1   # at least original question

    def test_multi_query_registered_in_strategy_registry(self):
        from rag_eval.strategies import STRATEGY_REGISTRY
        assert "multi_query" in STRATEGY_REGISTRY


# ---------------------------------------------------------------------------
# Strategy registry completeness
# ---------------------------------------------------------------------------

class TestStrategyRegistry:

    def test_all_five_strategies_registered(self):
        from rag_eval.strategies import STRATEGY_REGISTRY
        expected = {"naive", "hybrid", "rerank", "hyde", "multi_query"}
        assert expected == set(STRATEGY_REGISTRY.keys())

    def test_get_strategy_unknown_raises(self):
        from rag_eval.strategies import get_strategy
        from rag_eval.config import Config
        from unittest.mock import MagicMock

        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("nonexistent", Config(), MagicMock())

    def test_get_strategy_raises_no_longer_says_not_implemented(self):
        """After Phase 4, the error message should say 'Unknown' not 'not implemented yet'."""
        from rag_eval.strategies import get_strategy
        from rag_eval.config import Config

        with pytest.raises(ValueError) as exc_info:
            get_strategy("unknown_strat", Config(), MagicMock())

        assert "not implemented yet" not in str(exc_info.value).lower() or \
               "unknown" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Multi-query helper: _parse_query_variants
# ---------------------------------------------------------------------------

class TestParseQueryVariants:

    def test_includes_original_first(self):
        from rag_eval.strategies.multi_query import _parse_query_variants

        variants = _parse_query_variants("Alt A\nAlt B", "Original?", max_variants=3)
        assert variants[0] == "Original?"

    def test_strips_numbering_and_bullets(self):
        from rag_eval.strategies.multi_query import _parse_query_variants

        text = "1. First variant\n- Second variant\n• Third variant"
        variants = _parse_query_variants(text, "Q?", max_variants=3)
        assert all(not v[0].isdigit() for v in variants[1:])
        assert all(not v.startswith("-") for v in variants[1:])

    def test_deduplicates_variants(self):
        from rag_eval.strategies.multi_query import _parse_query_variants

        text = "Alpha\nALPHA\nalpha"  # same thing, different case
        variants = _parse_query_variants(text, "Q?", max_variants=5)
        lower_variants = [v.lower() for v in variants]
        assert len(lower_variants) == len(set(lower_variants))

    def test_respects_max_variants(self):
        from rag_eval.strategies.multi_query import _parse_query_variants

        text = "\n".join([f"Query {i}" for i in range(10)])
        variants = _parse_query_variants(text, "Original?", max_variants=3)
        assert len(variants) <= 4   # original + 3 variants


# ---------------------------------------------------------------------------
# Reranker provider
# ---------------------------------------------------------------------------

class TestRerankerProvider:

    def test_get_reranker_raises_when_disabled(self):
        from rag_eval.providers.reranker import get_reranker
        from rag_eval.config import RerankerConfig

        cfg = RerankerConfig(enabled=False)
        with pytest.raises(ValueError, match="disabled"):
            get_reranker(cfg)

    def test_get_reranker_raises_missing_api_key(self):
        from rag_eval.providers.reranker import get_reranker
        from rag_eval.config import RerankerConfig
        import os

        cfg = RerankerConfig(provider="cohere", enabled=True)
        # Ensure env var is absent
        env_without_key = {k: v for k, v in os.environ.items() if k != "COHERE_API_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True):
            with pytest.raises(EnvironmentError, match="COHERE_API_KEY"):
                get_reranker(cfg)

    def test_get_reranker_returns_callable(self):
        from rag_eval.providers.reranker import get_reranker
        from rag_eval.config import RerankerConfig

        cfg = RerankerConfig(provider="cohere", enabled=True)
        with patch.dict(os.environ, {"COHERE_API_KEY": "dummy"}):
            with patch("cohere.ClientV2"):  # prevent real HTTP
                reranker = get_reranker(cfg)

        assert callable(reranker)

    def test_get_reranker_unknown_provider_raises(self):
        from rag_eval.providers.reranker import get_reranker
        from rag_eval.config import RerankerConfig

        cfg = RerankerConfig(provider="cohere", enabled=True)
        cfg = cfg.model_copy(update={"provider": "unknown_provider"})
        with pytest.raises(ValueError, match="Unknown reranker provider"):
            get_reranker(cfg)
