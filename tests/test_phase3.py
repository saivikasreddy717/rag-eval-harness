"""
Phase 3 tests — RAGAS evaluation layer (evaluator.py) and CLI eval command.

All RAGAS calls are mocked so no API keys or network access are required.
Safe to run in CI.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_predictions_file(tmp_path: Path, strategy: str = "naive", n: int = 3) -> Path:
    """Write a minimal predictions JSONL with n valid records."""
    lines = []
    for i in range(n):
        lines.append(
            json.dumps(
                {
                    "id": f"q{i}",
                    "question": f"Question {i}?",
                    "answer": f"Answer {i}.",
                    "contexts": [f"Context {i}A.", f"Context {i}B."],
                    "reference_answer": f"Reference {i}.",
                    "reference_contexts": [f"Context {i}A."],
                    "strategy": strategy,
                    "latency_ms": 100.0 + i * 10,
                    "cost_usd": 0.0001,
                    "prompt_tokens": 200 + i,
                    "completion_tokens": 30 + i,
                    "metadata": {"strategy": strategy},
                }
            )
        )
    pred_file = tmp_path / f"predictions_{strategy}.jsonl"
    pred_file.write_text("\n".join(lines), encoding="utf-8")
    return pred_file


def _make_error_predictions_file(tmp_path: Path) -> Path:
    """Write a predictions JSONL where every record is an error."""
    record = json.dumps(
        {
            "id": "q0",
            "question": "What?",
            "answer": "",
            "contexts": [],
            "reference_answer": "Ref.",
            "reference_contexts": [],
            "strategy": "naive",
            "latency_ms": 0.0,
            "cost_usd": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "metadata": {"error": "Simulated API error"},
        }
    )
    pred_file = tmp_path / "predictions_naive.jsonl"
    pred_file.write_text(record, encoding="utf-8")
    return pred_file


def _mock_ragas_result(n_rows: int = 3) -> MagicMock:
    """Return a mock RAGAS result whose .to_pandas() returns a plausible DataFrame."""
    df = pd.DataFrame(
        {
            "user_input": [f"Question {i}?" for i in range(n_rows)],
            "response": [f"Answer {i}." for i in range(n_rows)],
            "faithfulness": [0.9, 0.8, 0.85][:n_rows],
            "answer_relevancy": [0.88, 0.79, 0.84][:n_rows],
            "context_precision": [0.75, 0.70, 0.72][:n_rows],
            "context_recall": [0.80, 0.77, 0.78][:n_rows],
            "answer_correctness": [0.70, 0.65, 0.68][:n_rows],
        }
    )
    mock_result = MagicMock()
    mock_result.to_pandas.return_value = df
    return mock_result


# Consistent set of patches used by most evaluate_predictions tests.
# Order: outermost → last decorator → first argument.
_EVAL_PATCHES = [
    "rag_eval.evaluator.LangchainEmbeddingsWrapper",
    "rag_eval.evaluator.LangchainLLMWrapper",
    "rag_eval.evaluator.get_embeddings",
    "rag_eval.evaluator.get_llm",
    "rag_eval.evaluator.evaluate",
]


def _apply_eval_patches(test_fn):
    """Stack all five evaluator patches onto a test function."""
    for target in reversed(_EVAL_PATCHES):
        test_fn = patch(target)(test_fn)
    return test_fn


# ---------------------------------------------------------------------------
# evaluator._load_predictions
# ---------------------------------------------------------------------------


class TestLoadPredictions:
    def test_loads_all_lines(self, tmp_path):
        from rag_eval.evaluator import _load_predictions

        pred_file = _make_predictions_file(tmp_path, n=5)
        preds = _load_predictions(pred_file)
        assert len(preds) == 5

    def test_skips_blank_lines(self, tmp_path):
        from rag_eval.evaluator import _load_predictions

        pred_file = tmp_path / "preds.jsonl"
        pred_file.write_text(
            '{"id": "q1", "answer": "A"}\n\n{"id": "q2", "answer": "B"}\n',
            encoding="utf-8",
        )
        preds = _load_predictions(pred_file)
        assert len(preds) == 2

    def test_raises_for_missing_file(self, tmp_path):
        from rag_eval.evaluator import _load_predictions

        with pytest.raises(FileNotFoundError):
            _load_predictions(tmp_path / "nonexistent.jsonl")


# ---------------------------------------------------------------------------
# evaluator._build_ragas_dataset
# ---------------------------------------------------------------------------


class TestBuildRagasDataset:
    def test_builds_dataset_from_valid_predictions(self):
        from rag_eval.evaluator import _build_ragas_dataset

        preds = [
            {
                "question": "What is alpha?",
                "answer": "The first Greek letter.",
                "contexts": ["Alpha is the first letter."],
                "reference_answer": "The first letter of the Greek alphabet.",
                "reference_contexts": ["Alpha is the first letter."],
                "metadata": {},
            }
        ]
        dataset = _build_ragas_dataset(preds)
        assert len(dataset.samples) == 1

    def test_skips_error_predictions(self):
        from rag_eval.evaluator import _build_ragas_dataset

        preds = [
            {
                "question": "What is alpha?",
                "answer": "",  # empty — should be skipped
                "contexts": [],
                "reference_answer": "Ref.",
                "reference_contexts": [],
                "metadata": {"error": "API timeout"},
            },
            {
                "question": "What is beta?",
                "answer": "The second Greek letter.",
                "contexts": ["Beta context."],
                "reference_answer": "Second letter.",
                "reference_contexts": [],
                "metadata": {},
            },
        ]
        dataset = _build_ragas_dataset(preds)
        assert len(dataset.samples) == 1

    def test_sample_fields_are_set_correctly(self):
        from rag_eval.evaluator import _build_ragas_dataset

        preds = [
            {
                "question": "Q?",
                "answer": "A.",
                "contexts": ["C1", "C2"],
                "reference_answer": "R.",
                "reference_contexts": ["C1"],
                "metadata": {},
            }
        ]
        dataset = _build_ragas_dataset(preds)
        sample = dataset.samples[0]
        assert sample.user_input == "Q?"
        assert sample.response == "A."
        assert sample.retrieved_contexts == ["C1", "C2"]
        assert sample.reference == "R."
        assert sample.reference_contexts == ["C1"]


# ---------------------------------------------------------------------------
# evaluator.evaluate_predictions  (all ragas I/O mocked)
# ---------------------------------------------------------------------------


class TestEvaluatePredictions:
    @patch("rag_eval.evaluator.LangchainEmbeddingsWrapper")
    @patch("rag_eval.evaluator.LangchainLLMWrapper")
    @patch("rag_eval.evaluator.get_embeddings")
    @patch("rag_eval.evaluator.get_llm")
    @patch("rag_eval.evaluator.evaluate")
    def test_scorecard_csv_is_written(
        self,
        mock_evaluate,
        mock_get_llm,
        mock_get_embeddings,
        mock_llm_wrapper,
        mock_emb_wrapper,
        tmp_path,
    ):
        from rag_eval.config import Config, OutputConfig
        from rag_eval.evaluator import evaluate_predictions

        pred_file = _make_predictions_file(tmp_path, n=3)
        mock_evaluate.return_value = _mock_ragas_result(n_rows=3)

        cfg = Config()
        cfg = cfg.model_copy(update={"output": OutputConfig(dir=str(tmp_path))})

        result_path = evaluate_predictions(pred_file, cfg)

        assert result_path.exists()
        assert result_path.suffix == ".csv"

    @patch("rag_eval.evaluator.LangchainEmbeddingsWrapper")
    @patch("rag_eval.evaluator.LangchainLLMWrapper")
    @patch("rag_eval.evaluator.get_embeddings")
    @patch("rag_eval.evaluator.get_llm")
    @patch("rag_eval.evaluator.evaluate")
    def test_scorecard_has_strategy_column(
        self,
        mock_evaluate,
        mock_get_llm,
        mock_get_embeddings,
        mock_llm_wrapper,
        mock_emb_wrapper,
        tmp_path,
    ):
        from rag_eval.config import Config, OutputConfig
        from rag_eval.evaluator import evaluate_predictions

        pred_file = _make_predictions_file(tmp_path, strategy="naive", n=3)
        mock_evaluate.return_value = _mock_ragas_result(n_rows=3)

        cfg = Config()
        cfg = cfg.model_copy(update={"output": OutputConfig(dir=str(tmp_path))})

        result_path = evaluate_predictions(pred_file, cfg)
        df = pd.read_csv(result_path)

        assert "strategy" in df.columns
        assert (df["strategy"] == "naive").all()

    @patch("rag_eval.evaluator.LangchainEmbeddingsWrapper")
    @patch("rag_eval.evaluator.LangchainLLMWrapper")
    @patch("rag_eval.evaluator.get_embeddings")
    @patch("rag_eval.evaluator.get_llm")
    @patch("rag_eval.evaluator.evaluate")
    def test_scorecard_has_aggregate_row(
        self,
        mock_evaluate,
        mock_get_llm,
        mock_get_embeddings,
        mock_llm_wrapper,
        mock_emb_wrapper,
        tmp_path,
    ):
        from rag_eval.config import Config, OutputConfig
        from rag_eval.evaluator import evaluate_predictions

        pred_file = _make_predictions_file(tmp_path, n=3)
        mock_evaluate.return_value = _mock_ragas_result(n_rows=3)

        cfg = Config()
        cfg = cfg.model_copy(update={"output": OutputConfig(dir=str(tmp_path))})

        result_path = evaluate_predictions(pred_file, cfg)
        df = pd.read_csv(result_path)

        assert "AGGREGATE" in df["question_id"].values

    @patch("rag_eval.evaluator.LangchainEmbeddingsWrapper")
    @patch("rag_eval.evaluator.LangchainLLMWrapper")
    @patch("rag_eval.evaluator.get_embeddings")
    @patch("rag_eval.evaluator.get_llm")
    @patch("rag_eval.evaluator.evaluate")
    def test_scorecard_has_ragas_metric_columns(
        self,
        mock_evaluate,
        mock_get_llm,
        mock_get_embeddings,
        mock_llm_wrapper,
        mock_emb_wrapper,
        tmp_path,
    ):
        from rag_eval.config import Config, OutputConfig
        from rag_eval.evaluator import _METRIC_NAMES, evaluate_predictions

        pred_file = _make_predictions_file(tmp_path, n=3)
        mock_evaluate.return_value = _mock_ragas_result(n_rows=3)

        cfg = Config()
        cfg = cfg.model_copy(update={"output": OutputConfig(dir=str(tmp_path))})

        result_path = evaluate_predictions(pred_file, cfg)
        df = pd.read_csv(result_path)

        for metric in _METRIC_NAMES:
            assert metric in df.columns, f"Missing column: {metric}"

    @patch("rag_eval.evaluator.LangchainEmbeddingsWrapper")
    @patch("rag_eval.evaluator.LangchainLLMWrapper")
    @patch("rag_eval.evaluator.get_embeddings")
    @patch("rag_eval.evaluator.get_llm")
    @patch("rag_eval.evaluator.evaluate")
    def test_max_questions_limits_input(
        self,
        mock_evaluate,
        mock_get_llm,
        mock_get_embeddings,
        mock_llm_wrapper,
        mock_emb_wrapper,
        tmp_path,
    ):
        """With max_questions=2, only 2 samples should reach ragas.evaluate."""
        from rag_eval.config import Config, OutputConfig
        from rag_eval.evaluator import evaluate_predictions

        pred_file = _make_predictions_file(tmp_path, n=5)
        mock_evaluate.return_value = _mock_ragas_result(n_rows=2)

        cfg = Config()
        cfg = cfg.model_copy(update={"output": OutputConfig(dir=str(tmp_path))})

        evaluate_predictions(pred_file, cfg, max_questions=2)

        # Inspect the EvaluationDataset passed to ragas.evaluate
        call_kwargs = mock_evaluate.call_args
        dataset_arg = call_kwargs.kwargs.get("dataset") or call_kwargs.args[0]
        assert len(dataset_arg.samples) == 2

    def test_raises_for_missing_predictions_file(self, tmp_path):
        from rag_eval.config import Config
        from rag_eval.evaluator import evaluate_predictions

        with pytest.raises(FileNotFoundError):
            evaluate_predictions(tmp_path / "nonexistent.jsonl", Config())

    @patch("rag_eval.evaluator.LangchainEmbeddingsWrapper")
    @patch("rag_eval.evaluator.LangchainLLMWrapper")
    @patch("rag_eval.evaluator.get_embeddings")
    @patch("rag_eval.evaluator.get_llm")
    @patch("rag_eval.evaluator.evaluate")
    def test_all_error_predictions_raises(
        self,
        mock_evaluate,
        mock_get_llm,
        mock_get_embeddings,
        mock_llm_wrapper,
        mock_emb_wrapper,
        tmp_path,
    ):
        from rag_eval.config import Config
        from rag_eval.evaluator import evaluate_predictions

        pred_file = _make_error_predictions_file(tmp_path)
        cfg = Config()

        with pytest.raises(ValueError, match="No valid predictions"):
            evaluate_predictions(pred_file, cfg)

    @patch("rag_eval.evaluator.LangchainEmbeddingsWrapper")
    @patch("rag_eval.evaluator.LangchainLLMWrapper")
    @patch("rag_eval.evaluator.get_embeddings")
    @patch("rag_eval.evaluator.get_llm")
    @patch("rag_eval.evaluator.evaluate")
    def test_telemetry_columns_merged(
        self,
        mock_evaluate,
        mock_get_llm,
        mock_get_embeddings,
        mock_llm_wrapper,
        mock_emb_wrapper,
        tmp_path,
    ):
        """latency_ms and cost_usd should appear in the scorecard."""
        from rag_eval.config import Config, OutputConfig
        from rag_eval.evaluator import evaluate_predictions

        pred_file = _make_predictions_file(tmp_path, n=3)
        mock_evaluate.return_value = _mock_ragas_result(n_rows=3)

        cfg = Config()
        cfg = cfg.model_copy(update={"output": OutputConfig(dir=str(tmp_path))})

        result_path = evaluate_predictions(pred_file, cfg)
        df = pd.read_csv(result_path)

        assert "latency_ms" in df.columns
        assert "cost_usd" in df.columns

    @patch("rag_eval.evaluator.LangchainEmbeddingsWrapper")
    @patch("rag_eval.evaluator.LangchainLLMWrapper")
    @patch("rag_eval.evaluator.get_embeddings")
    @patch("rag_eval.evaluator.get_llm")
    @patch("rag_eval.evaluator.evaluate")
    def test_aggregate_scores_are_means(
        self,
        mock_evaluate,
        mock_get_llm,
        mock_get_embeddings,
        mock_llm_wrapper,
        mock_emb_wrapper,
        tmp_path,
    ):
        """AGGREGATE row values should equal the column means of the data rows."""
        from rag_eval.config import Config, OutputConfig
        from rag_eval.evaluator import evaluate_predictions

        pred_file = _make_predictions_file(tmp_path, n=3)
        mock_evaluate.return_value = _mock_ragas_result(n_rows=3)

        cfg = Config()
        cfg = cfg.model_copy(update={"output": OutputConfig(dir=str(tmp_path))})

        result_path = evaluate_predictions(pred_file, cfg)
        df = pd.read_csv(result_path)

        data_rows = df[df["question_id"] != "AGGREGATE"]
        agg_row = df[df["question_id"] == "AGGREGATE"].iloc[0]

        expected_faith = data_rows["faithfulness"].mean()
        assert agg_row["faithfulness"] == pytest.approx(expected_faith, rel=1e-4)


# ---------------------------------------------------------------------------
# CLI eval command
# ---------------------------------------------------------------------------


class TestEvalCLI:
    @patch("rag_eval.evaluator.LangchainEmbeddingsWrapper")
    @patch("rag_eval.evaluator.LangchainLLMWrapper")
    @patch("rag_eval.evaluator.get_embeddings")
    @patch("rag_eval.evaluator.get_llm")
    @patch("rag_eval.evaluator.evaluate")
    def test_eval_command_succeeds(
        self,
        mock_evaluate,
        mock_get_llm,
        mock_get_embeddings,
        mock_llm_wrapper,
        mock_emb_wrapper,
        tmp_path,
    ):
        from click.testing import CliRunner

        from rag_eval.cli import main

        _make_predictions_file(tmp_path, strategy="naive", n=2)
        mock_evaluate.return_value = _mock_ragas_result(n_rows=2)

        # Minimal config YAML pointing output to tmp_path
        config_yaml = tmp_path / "test_config.yaml"
        config_yaml.write_text(
            textwrap.dedent(f"""\
                generator:
                  provider: groq
                  model: meta-llama/llama-4-scout-17b-16e-instruct
                judge:
                  provider: groq
                  model: llama-3.3-70b-versatile
                embeddings:
                  provider: local
                  model: BAAI/bge-large-en-v1.5
                strategies:
                  - naive
                output:
                  dir: "{str(tmp_path).replace(chr(92), "/")}"
            """),
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(config_yaml), "eval", "--strategy", "naive"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output

    def test_eval_command_missing_predictions_warns(self, tmp_path):
        """eval with no prediction files should print a warning and exit 1."""
        from click.testing import CliRunner

        from rag_eval.cli import main

        config_yaml = tmp_path / "test_config.yaml"
        config_yaml.write_text(
            textwrap.dedent(f"""\
                generator:
                  provider: groq
                  model: meta-llama/llama-4-scout-17b-16e-instruct
                judge:
                  provider: groq
                  model: llama-3.3-70b-versatile
                embeddings:
                  provider: local
                  model: BAAI/bge-large-en-v1.5
                strategies:
                  - naive
                output:
                  dir: "{str(tmp_path).replace(chr(92), "/")}"
            """),
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(config_yaml), "eval"],
        )

        assert result.exit_code != 0
        assert "No predictions found" in result.output or "Nothing to evaluate" in result.output

    @patch("rag_eval.evaluator.LangchainEmbeddingsWrapper")
    @patch("rag_eval.evaluator.LangchainLLMWrapper")
    @patch("rag_eval.evaluator.get_embeddings")
    @patch("rag_eval.evaluator.get_llm")
    @patch("rag_eval.evaluator.evaluate")
    def test_eval_command_max_questions_flag(
        self,
        mock_evaluate,
        mock_get_llm,
        mock_get_embeddings,
        mock_llm_wrapper,
        mock_emb_wrapper,
        tmp_path,
    ):
        """--max-questions 2 should limit evaluation to 2 samples."""
        from click.testing import CliRunner

        from rag_eval.cli import main

        _make_predictions_file(tmp_path, strategy="naive", n=5)
        mock_evaluate.return_value = _mock_ragas_result(n_rows=2)

        config_yaml = tmp_path / "test_config.yaml"
        config_yaml.write_text(
            textwrap.dedent(f"""\
                generator:
                  provider: groq
                  model: meta-llama/llama-4-scout-17b-16e-instruct
                judge:
                  provider: groq
                  model: llama-3.3-70b-versatile
                embeddings:
                  provider: local
                  model: BAAI/bge-large-en-v1.5
                strategies:
                  - naive
                output:
                  dir: "{str(tmp_path).replace(chr(92), "/")}"
            """),
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(config_yaml), "eval", "--strategy", "naive", "--max-questions", "2"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output

        # Verify only 2 samples were passed to ragas.evaluate
        call_kwargs = mock_evaluate.call_args
        dataset_arg = call_kwargs.kwargs.get("dataset") or call_kwargs.args[0]
        assert len(dataset_arg.samples) == 2
