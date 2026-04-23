"""
Phase 2 tests — telemetry, strategy base, NaiveRAG, runner.

All tests use mocked LLM responses and a tiny fake index so no API
keys or network access are required. Safe to run in CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FAKE_CORPUS = [
    {"passage_id": "Alpha", "text": "Alpha is the first letter of the Greek alphabet."},
    {"passage_id": "Beta", "text": "Beta is the second letter of the Greek alphabet."},
    {"passage_id": "Gamma", "text": "Gamma rays are high-frequency electromagnetic radiation."},
]

FAKE_QA_PAIRS = [
    {
        "id": "q1",
        "question": "What is alpha?",
        "reference_answer": "The first letter of the Greek alphabet",
        "reference_contexts": [FAKE_CORPUS[0]["text"]],
    }
]


@pytest.fixture
def built_index(tmp_path):
    """Build a small real FAISS + BM25 index into tmp_path."""
    import rag_eval.indexer as idx_mod
    from rag_eval.chunker import chunk_corpus
    from rag_eval.config import Config
    from rag_eval.indexer import RAGIndex, build_index

    # Redirect index dir to tmp_path
    idx_mod.INDEX_DIR = tmp_path
    idx_mod.FAISS_PATH = tmp_path / "faiss.index"
    idx_mod.CHUNKS_PATH = tmp_path / "chunks.json"
    idx_mod.META_PATH = tmp_path / "index_meta.json"

    cfg = Config()
    chunks = chunk_corpus(FAKE_CORPUS, cfg.retrieval)
    build_index(chunks, cfg)

    index = RAGIndex.load(tmp_path)
    yield index

    # Restore defaults
    idx_mod.INDEX_DIR = Path("data/index")
    idx_mod.FAISS_PATH = idx_mod.INDEX_DIR / "faiss.index"
    idx_mod.CHUNKS_PATH = idx_mod.INDEX_DIR / "chunks.json"
    idx_mod.META_PATH = idx_mod.INDEX_DIR / "index_meta.json"


# ---------------------------------------------------------------------------
# Telemetry tests
# ---------------------------------------------------------------------------


class TestTelemetry:
    def test_token_count_nonzero(self):
        from rag_eval.telemetry import count_tokens

        assert count_tokens("Hello world this is a test") > 0

    def test_price_lookup_groq(self):
        from rag_eval.telemetry import _lookup_price

        input_p, output_p = _lookup_price("groq", "meta-llama/llama-4-scout-17b-16e-instruct")
        assert input_p > 0
        assert output_p > 0

    def test_price_lookup_unknown_is_zero(self):
        from rag_eval.telemetry import _lookup_price

        input_p, output_p = _lookup_price("ollama", "some-local-model")
        assert input_p == 0.0
        assert output_p == 0.0

    def test_tracker_summary(self):
        from rag_eval.telemetry import QueryRecord, TelemetryTracker

        tracker = TelemetryTracker(strategy="naive")
        for i in range(10):
            tracker.add(
                QueryRecord(
                    strategy="naive",
                    latency_ms=float(100 + i * 10),
                    prompt_tokens=500,
                    completion_tokens=50,
                    cost_usd=0.0001,
                )
            )

        summary = tracker.summary()
        assert summary["n_queries"] == 10
        assert summary["latency_p50_ms"] > 0
        assert summary["latency_p95_ms"] >= summary["latency_p50_ms"]
        assert summary["total_cost_usd"] == pytest.approx(0.001, rel=1e-3)

    def test_empty_tracker_returns_empty_summary(self):
        from rag_eval.telemetry import TelemetryTracker

        tracker = TelemetryTracker(strategy="naive")
        assert tracker.summary() == {}


# ---------------------------------------------------------------------------
# Strategy base tests
# ---------------------------------------------------------------------------


class TestStrategyBase:
    def test_build_rag_prompt_contains_question_and_contexts(self):
        from rag_eval.strategies.base import build_rag_prompt

        prompt = build_rag_prompt(
            "What is alpha?",
            ["Alpha is the first letter.", "More info here."],
        )
        assert "What is alpha?" in prompt
        assert "Alpha is the first letter." in prompt
        assert "[1]" in prompt
        assert "[2]" in prompt

    def test_build_messages_returns_list(self):
        from rag_eval.strategies.base import build_messages

        messages = build_messages("Test question?", ["Context A", "Context B"])
        assert len(messages) == 2
        assert messages[0].__class__.__name__ == "SystemMessage"
        assert messages[1].__class__.__name__ == "HumanMessage"

    def test_strategy_registry_has_naive(self):
        from rag_eval.strategies import STRATEGY_REGISTRY

        assert "naive" in STRATEGY_REGISTRY

    def test_get_strategy_raises_for_unknown(self):
        """Requesting a strategy name that does not exist raises ValueError."""
        from rag_eval.config import Config
        from rag_eval.strategies import get_strategy

        with pytest.raises(ValueError):
            get_strategy("totally_unknown_strategy", Config(), MagicMock())


# ---------------------------------------------------------------------------
# NaiveRAG tests (LLM is mocked — no API calls)
# ---------------------------------------------------------------------------


class TestNaiveRAG:
    def test_naive_rag_returns_result(self, built_index):
        """NaiveRAG.answer() returns a populated RAGResult."""
        import os

        from rag_eval.config import Config
        from rag_eval.strategies.naive import NaiveRAG

        cfg = Config()
        mock_response = MagicMock()
        mock_response.content = "Alpha is the first letter of the Greek alphabet."

        # Groq validates GROQ_API_KEY at instantiation — set a dummy key
        with patch.dict(os.environ, {"GROQ_API_KEY": "dummy-key-for-testing"}):
            strategy = NaiveRAG(cfg, built_index)
            strategy.llm = MagicMock()
            strategy.llm.invoke.return_value = mock_response

            result = strategy.answer("What is alpha?")

        assert result.answer == "Alpha is the first letter of the Greek alphabet."
        # Fake corpus has only 3 chunks — FAISS returns min(top_k, available)
        assert len(result.contexts) == min(cfg.retrieval.top_k, built_index.num_chunks)
        assert result.latency_ms >= 0
        assert result.prompt_tokens > 0
        assert result.completion_tokens > 0
        assert result.metadata["strategy"] == "naive"

    def test_naive_rag_retrieves_correct_top_k(self, built_index):
        import os

        from rag_eval.config import Config, RetrievalConfig
        from rag_eval.strategies.naive import NaiveRAG

        cfg = Config()
        cfg = cfg.model_copy(update={"retrieval": RetrievalConfig(top_k=2)})
        mock_response = MagicMock()
        mock_response.content = "Some answer."

        with patch.dict(os.environ, {"GROQ_API_KEY": "dummy-key-for-testing"}):
            strategy = NaiveRAG(cfg, built_index)
            strategy.llm = MagicMock()
            strategy.llm.invoke.return_value = mock_response
            result = strategy.answer("Test question")

        assert len(result.contexts) == 2


# ---------------------------------------------------------------------------
# Runner tests
# ---------------------------------------------------------------------------


class TestRunner:
    def test_runner_writes_jsonl(self, built_index, tmp_path):
        import os

        from rag_eval.config import Config
        from rag_eval.runner import run_strategy
        from rag_eval.strategies.naive import NaiveRAG

        cfg = Config()
        with patch.dict(os.environ, {"GROQ_API_KEY": "dummy-key-for-testing"}):
            strategy = NaiveRAG(cfg, built_index)
            strategy.llm = MagicMock()
            strategy.llm.invoke.return_value = MagicMock(content="Test answer.")
            run_strategy(strategy, FAKE_QA_PAIRS, tmp_path)

        predictions_file = tmp_path / "predictions_naive.jsonl"
        assert predictions_file.exists()

        lines = predictions_file.read_text().strip().split("\n")
        assert len(lines) == len(FAKE_QA_PAIRS)

        record = json.loads(lines[0])
        assert "question" in record
        assert "answer" in record
        assert "contexts" in record
        assert "reference_answer" in record
        assert "reference_contexts" in record
        assert "latency_ms" in record
        assert "cost_usd" in record
        assert record["strategy"] == "naive"

    def test_runner_handles_errors_gracefully(self, built_index, tmp_path):
        """Runner logs errors per-query but completes the full run."""
        import os

        from rag_eval.config import Config
        from rag_eval.runner import run_strategy
        from rag_eval.strategies.naive import NaiveRAG

        cfg = Config()
        with patch.dict(os.environ, {"GROQ_API_KEY": "dummy-key-for-testing"}):
            strategy = NaiveRAG(cfg, built_index)
            strategy.llm = MagicMock()
            strategy.llm.invoke.side_effect = Exception("Simulated API error")
            run_strategy(strategy, FAKE_QA_PAIRS, tmp_path)

        predictions_file = tmp_path / "predictions_naive.jsonl"
        assert predictions_file.exists()
        record = json.loads(predictions_file.read_text().strip())
        assert "error" in record["metadata"]
