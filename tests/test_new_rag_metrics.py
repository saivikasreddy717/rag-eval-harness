"""
Tests for the three new RAG/production metrics:
  - hallucination_rate (LLM judge, mocked)
  - context_relevance  (LLM judge, mocked)
  - retrieval_latency p95 (pure computation, no mocks needed)

No API keys or real network calls required. CI-safe.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_judge(content: str) -> MagicMock:
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content=content)
    return mock


def _make_predictions(n: int = 3) -> list[dict]:
    return [
        {
            "id": f"q{i}",
            "question": f"What is concept {i}?",
            "answer": f"Concept {i} is defined as X.",
            "contexts": [f"Context passage A for {i}.", f"Context passage B for {i}."],
            "reference_answer": f"X is concept {i}.",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Hallucination rate
# ---------------------------------------------------------------------------


class TestHallucinationRate:
    def test_zero_hallucination_when_all_grounded(self):
        from rag_eval.evaluator import score_hallucination_rate

        judge = _make_judge(
            json.dumps(
                {
                    "total_claims": 3,
                    "hallucinated_claims": 0,
                    "hallucination_rate": 0.0,
                    "examples": [],
                }
            )
        )
        preds = _make_predictions(2)
        rates = score_hallucination_rate(preds, judge)
        assert len(rates) == 2
        assert all(r == pytest.approx(0.0) for r in rates)

    def test_full_hallucination_rate_parsed(self):
        from rag_eval.evaluator import score_hallucination_rate

        judge = _make_judge(
            json.dumps(
                {
                    "total_claims": 4,
                    "hallucinated_claims": 4,
                    "hallucination_rate": 1.0,
                    "examples": ["claim1"],
                }
            )
        )
        preds = _make_predictions(1)
        rates = score_hallucination_rate(preds, judge)
        assert rates[0] == pytest.approx(1.0)

    def test_partial_hallucination_rate(self):
        from rag_eval.evaluator import score_hallucination_rate

        judge = _make_judge(
            json.dumps(
                {
                    "total_claims": 4,
                    "hallucinated_claims": 1,
                    "hallucination_rate": 0.25,
                    "examples": [],
                }
            )
        )
        preds = _make_predictions(1)
        rates = score_hallucination_rate(preds, judge)
        assert rates[0] == pytest.approx(0.25)

    def test_judge_api_failure_returns_none(self):
        from rag_eval.evaluator import score_hallucination_rate

        judge = MagicMock()
        judge.invoke.side_effect = RuntimeError("API error")
        preds = _make_predictions(1)
        rates = score_hallucination_rate(preds, judge)
        assert rates[0] is None

    def test_malformed_json_returns_none(self):
        from rag_eval.evaluator import score_hallucination_rate

        judge = _make_judge("not json at all!!!")
        preds = _make_predictions(1)
        rates = score_hallucination_rate(preds, judge)
        assert rates[0] is None

    def test_score_clamped_to_0_1(self):
        from rag_eval.evaluator import score_hallucination_rate

        judge = _make_judge(
            json.dumps(
                {
                    "total_claims": 1,
                    "hallucinated_claims": 0,
                    "hallucination_rate": 1.5,
                    "examples": [],
                }
            )
        )
        preds = _make_predictions(1)
        rates = score_hallucination_rate(preds, judge)
        assert rates[0] <= 1.0


# ---------------------------------------------------------------------------
# Context relevance (per-chunk)
# ---------------------------------------------------------------------------


class TestContextRelevance:
    def test_mean_of_chunk_scores(self):
        from rag_eval.evaluator import score_context_relevance

        # Two chunks: scores 0.8 and 0.6 → mean 0.7
        call_count = [0]

        def mock_invoke(msgs):
            score = 0.8 if call_count[0] % 2 == 0 else 0.6
            call_count[0] += 1
            return MagicMock(content=json.dumps({"score": score}))

        judge = MagicMock()
        judge.invoke.side_effect = mock_invoke

        preds = _make_predictions(1)
        scores = score_context_relevance(preds, judge)
        assert scores[0] == pytest.approx(0.7, rel=1e-4)

    def test_empty_contexts_returns_none(self):
        from rag_eval.evaluator import score_context_relevance

        judge = _make_judge(json.dumps({"score": 0.9}))
        preds = [{"question": "Q?", "answer": "A.", "contexts": []}]
        scores = score_context_relevance(preds, judge)
        assert scores[0] is None

    def test_per_prediction_score_count_matches(self):
        from rag_eval.evaluator import score_context_relevance

        judge = _make_judge(json.dumps({"score": 0.75}))
        preds = _make_predictions(4)
        scores = score_context_relevance(preds, judge)
        assert len(scores) == 4

    def test_judge_failure_returns_none(self):
        from rag_eval.evaluator import score_context_relevance

        judge = MagicMock()
        judge.invoke.side_effect = RuntimeError("LLM down")
        preds = _make_predictions(1)
        scores = score_context_relevance(preds, judge)
        assert scores[0] is None


# ---------------------------------------------------------------------------
# Retrieval latency p95
# ---------------------------------------------------------------------------


class TestRetrievalLatencyPercentiles:
    def test_p95_computation(self):
        latencies = list(range(1, 101))  # 1..100 ms
        p95 = float(np.percentile(latencies, 95))
        assert p95 == pytest.approx(95.05, rel=1e-3)

    def test_p95_in_aggregate_row(self, tmp_path):
        """Full-stack: evaluate_predictions adds retrieval_latency_p95_ms to AGGREGATE."""
        from unittest.mock import patch

        from rag_eval.config import Config, MetricsExtraConfig, OutputConfig
        from rag_eval.evaluator import evaluate_predictions

        # Build a minimal predictions file with retrieval_latency_ms populated
        lines = []
        for i in range(10):
            lines.append(
                json.dumps(
                    {
                        "id": f"q{i}",
                        "question": f"Q{i}?",
                        "answer": f"A{i}.",
                        "contexts": [f"C{i}."],
                        "reference_answer": f"R{i}.",
                        "reference_contexts": [f"C{i}."],
                        "strategy": "naive",
                        "latency_ms": 200.0 + i * 10,
                        "retrieval_latency_ms": 50.0 + i * 5,
                        "cost_usd": 0.0001,
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "metadata": {},
                    }
                )
            )
        pred_file = tmp_path / "predictions_naive.jsonl"
        pred_file.write_text("\n".join(lines), encoding="utf-8")

        ragas_df = pd.DataFrame(
            {
                "user_input": [f"Q{i}?" for i in range(10)],
                "response": [f"A{i}." for i in range(10)],
                "faithfulness": [0.9] * 10,
                "answer_relevancy": [0.85] * 10,
                "context_precision": [0.80] * 10,
                "context_recall": [0.82] * 10,
                "answer_correctness": [0.75] * 10,
            }
        )
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = ragas_df

        cfg = Config()
        cfg = cfg.model_copy(
            update={
                "output": OutputConfig(dir=str(tmp_path)),
                "metrics_extra": MetricsExtraConfig(retrieval_latency=True),
            }
        )

        with (
            patch("rag_eval.evaluator.evaluate", return_value=mock_result),
            patch("rag_eval.evaluator.get_llm"),
            patch("rag_eval.evaluator.get_embeddings"),
            patch("rag_eval.evaluator.LangchainLLMWrapper"),
            patch("rag_eval.evaluator.LangchainEmbeddingsWrapper"),
        ):
            scorecard = evaluate_predictions(pred_file, cfg)

        df = pd.read_csv(scorecard)
        agg = df[df["question_id"] == "AGGREGATE"].iloc[0]
        assert "retrieval_latency_p95_ms" in df.columns
        assert not pd.isna(agg["retrieval_latency_p95_ms"])
        # p95 of 50, 55, 60, ..., 95 ms (10 values)
        expected = float(np.percentile([50.0 + i * 5 for i in range(10)], 95))
        assert agg["retrieval_latency_p95_ms"] == pytest.approx(expected, rel=1e-3)
