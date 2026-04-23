"""
Phase 5 tests — comparison report (reporter.py) and CLI compare command.

All tests are self-contained: scorecard CSVs are written into tmp_path
so no real evaluation run is required. Safe to run in CI.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers — create fake scorecard CSVs
# ---------------------------------------------------------------------------

_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]

_STRATEGY_SCORES = {
    "naive": [0.82, 0.80, 0.75, 0.78, 0.70],
    "hybrid": [0.87, 0.85, 0.80, 0.83, 0.76],
    "rerank": [0.90, 0.88, 0.84, 0.86, 0.80],
    "hyde": [0.85, 0.83, 0.78, 0.81, 0.74],
    "multi_query": [0.86, 0.84, 0.79, 0.82, 0.75],
}


def _make_scorecard(
    tmp_path: Path,
    strategy: str,
    scores: list[float] | None = None,
    n_questions: int = 3,
) -> Path:
    """Write a realistic scorecard CSV (n_questions rows + AGGREGATE row)."""
    if scores is None:
        scores = _STRATEGY_SCORES.get(strategy, [0.8] * 5)

    rows = []
    for i in range(n_questions):
        row = {"strategy": strategy, "question_id": f"q{i}"}
        for m, base in zip(_METRICS, scores):
            row[m] = round(base + (i - 1) * 0.01, 4)  # tiny variation per row
        row["latency_ms"] = 300.0 + i * 20
        row["cost_usd"] = 0.00012
        row["prompt_tokens"] = 250 + i
        row["completion_tokens"] = 40 + i
        rows.append(row)

    # AGGREGATE row = means of per-question rows
    agg = {"strategy": strategy, "question_id": "AGGREGATE"}
    df_data = pd.DataFrame(rows)
    for col in _METRICS + ["latency_ms", "cost_usd", "prompt_tokens", "completion_tokens"]:
        agg[col] = df_data[col].mean()
    rows.append(agg)

    df = pd.DataFrame(rows)
    path = tmp_path / f"scorecard_{strategy}.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# reporter.load_scorecards
# ---------------------------------------------------------------------------


class TestLoadScorecards:
    def test_loads_single_scorecard(self, tmp_path):
        from rag_eval.reporter import load_scorecards

        _make_scorecard(tmp_path, "naive")
        df = load_scorecards(tmp_path)
        assert "naive" in df["strategy"].values

    def test_loads_multiple_scorecards(self, tmp_path):
        from rag_eval.reporter import load_scorecards

        for s in ["naive", "hybrid", "rerank"]:
            _make_scorecard(tmp_path, s)

        df = load_scorecards(tmp_path)
        assert set(df["strategy"].unique()) == {"naive", "hybrid", "rerank"}

    def test_raises_for_missing_output_dir(self, tmp_path):
        from rag_eval.reporter import load_scorecards

        with pytest.raises(FileNotFoundError):
            load_scorecards(tmp_path / "nonexistent")

    def test_raises_when_no_scorecards_present(self, tmp_path):
        from rag_eval.reporter import load_scorecards

        # Directory exists but has no scorecard_*.csv files
        with pytest.raises(FileNotFoundError):
            load_scorecards(tmp_path)

    def test_row_count_is_sum_of_all_scorecards(self, tmp_path):
        """Each 3-question scorecard has 4 rows (3 + AGGREGATE)."""
        from rag_eval.reporter import load_scorecards

        for s in ["naive", "hybrid"]:
            _make_scorecard(tmp_path, s, n_questions=3)

        df = load_scorecards(tmp_path)
        assert len(df) == 8  # 2 strategies × (3 + 1)


# ---------------------------------------------------------------------------
# reporter.build_comparison_matrix
# ---------------------------------------------------------------------------


class TestBuildComparisonMatrix:
    def test_matrix_has_one_row_per_strategy(self, tmp_path):
        from rag_eval.reporter import build_comparison_matrix, load_scorecards

        for s in ["naive", "hybrid", "rerank"]:
            _make_scorecard(tmp_path, s)

        combined = load_scorecards(tmp_path)
        matrix = build_comparison_matrix(combined)

        assert len(matrix) == 3

    def test_matrix_has_all_metric_columns(self, tmp_path):
        from rag_eval.reporter import build_comparison_matrix, load_scorecards

        _make_scorecard(tmp_path, "naive")
        combined = load_scorecards(tmp_path)
        matrix = build_comparison_matrix(combined)

        for m in _METRICS:
            assert m in matrix.columns, f"Missing metric column: {m}"

    def test_matrix_scores_match_aggregate_row(self, tmp_path):
        """Values in the matrix must equal the AGGREGATE row from the scorecard."""
        from rag_eval.reporter import build_comparison_matrix, load_scorecards

        _make_scorecard(tmp_path, "naive")
        combined = load_scorecards(tmp_path)
        matrix = build_comparison_matrix(combined)

        # Pick faithfulness value from matrix
        faith_matrix = matrix.loc[matrix["strategy"] == "naive", "faithfulness"].iloc[0]

        # Compare against the AGGREGATE row from combined df
        faith_agg = combined.loc[
            (combined["strategy"] == "naive") & (combined["question_id"] == "AGGREGATE"),
            "faithfulness",
        ].iloc[0]

        assert faith_matrix == pytest.approx(faith_agg, rel=1e-4)

    def test_matrix_is_sorted_by_strategy(self, tmp_path):
        from rag_eval.reporter import build_comparison_matrix, load_scorecards

        for s in ["rerank", "naive", "hybrid"]:  # intentionally unsorted
            _make_scorecard(tmp_path, s)

        combined = load_scorecards(tmp_path)
        matrix = build_comparison_matrix(combined)

        strategies = matrix["strategy"].tolist()
        assert strategies == sorted(strategies)


# ---------------------------------------------------------------------------
# reporter._make_metrics_bar / _make_radar / _make_latency_bar / _make_cost_scatter
# ---------------------------------------------------------------------------


class TestChartBuilders:
    def _matrix(self, tmp_path, strategies=("naive", "hybrid")):
        from rag_eval.reporter import build_comparison_matrix, load_scorecards

        for s in strategies:
            _make_scorecard(tmp_path, s)
        combined = load_scorecards(tmp_path)
        return build_comparison_matrix(combined)

    def test_metrics_bar_returns_figure(self, tmp_path):
        from rag_eval.reporter import _make_metrics_bar

        matrix = self._matrix(tmp_path)
        fig = _make_metrics_bar(matrix)
        assert fig is not None
        assert fig.data  # has traces

    def test_radar_returns_figure(self, tmp_path):
        from rag_eval.reporter import _make_radar

        matrix = self._matrix(tmp_path)
        fig = _make_radar(matrix)
        assert fig is not None
        assert fig.data

    def test_latency_bar_returns_figure(self, tmp_path):
        from rag_eval.reporter import _make_latency_bar

        matrix = self._matrix(tmp_path)
        fig = _make_latency_bar(matrix)
        assert fig is not None

    def test_cost_scatter_returns_figure(self, tmp_path):
        from rag_eval.reporter import _make_cost_scatter

        matrix = self._matrix(tmp_path)
        fig = _make_cost_scatter(matrix)
        assert fig is not None

    def test_latency_bar_returns_none_without_column(self, tmp_path):
        from rag_eval.reporter import _make_latency_bar

        # Matrix without latency_ms column
        matrix = pd.DataFrame(
            {
                "strategy": ["naive"],
                "faithfulness": [0.8],
            }
        )
        result = _make_latency_bar(matrix)
        assert result is None

    def test_metrics_bar_has_one_trace_per_strategy(self, tmp_path):
        from rag_eval.reporter import _make_metrics_bar

        matrix = self._matrix(tmp_path, strategies=("naive", "hybrid", "rerank"))
        fig = _make_metrics_bar(matrix)
        assert len(fig.data) == 3


# ---------------------------------------------------------------------------
# reporter.generate_html_report
# ---------------------------------------------------------------------------


class TestGenerateHtmlReport:
    def test_html_file_is_created(self, tmp_path):
        from rag_eval.reporter import build_comparison_matrix, generate_html_report, load_scorecards

        for s in ["naive", "hybrid"]:
            _make_scorecard(tmp_path, s)

        combined = load_scorecards(tmp_path)
        matrix = build_comparison_matrix(combined)

        output_file = tmp_path / "report.html"
        generate_html_report(matrix, output_file)

        assert output_file.exists()
        assert output_file.stat().st_size > 5_000  # non-trivial file

    def test_html_contains_strategy_names(self, tmp_path):
        from rag_eval.reporter import build_comparison_matrix, generate_html_report, load_scorecards

        for s in ["naive", "hybrid"]:
            _make_scorecard(tmp_path, s)

        combined = load_scorecards(tmp_path)
        matrix = build_comparison_matrix(combined)
        output_file = tmp_path / "report.html"
        generate_html_report(matrix, output_file)

        html = output_file.read_text(encoding="utf-8")
        assert "naive" in html
        assert "hybrid" in html

    def test_html_contains_plotly(self, tmp_path):
        from rag_eval.reporter import build_comparison_matrix, generate_html_report, load_scorecards

        _make_scorecard(tmp_path, "naive")
        combined = load_scorecards(tmp_path)
        matrix = build_comparison_matrix(combined)
        output_file = tmp_path / "report.html"
        generate_html_report(matrix, output_file)

        html = output_file.read_text(encoding="utf-8")
        # Plotly CDN or inline script should be present
        assert "plotly" in html.lower()

    def test_html_contains_summary_table(self, tmp_path):
        from rag_eval.reporter import build_comparison_matrix, generate_html_report, load_scorecards

        _make_scorecard(tmp_path, "naive")
        combined = load_scorecards(tmp_path)
        matrix = build_comparison_matrix(combined)
        output_file = tmp_path / "report.html"
        generate_html_report(matrix, output_file)

        html = output_file.read_text(encoding="utf-8")
        assert "summary-table" in html
        assert "faithfulness" in html


# ---------------------------------------------------------------------------
# reporter.compare_strategies  (end-to-end)
# ---------------------------------------------------------------------------


class TestCompareStrategies:
    def test_compare_writes_results_csv_and_report(self, tmp_path):
        from rag_eval.config import Config, OutputConfig
        from rag_eval.reporter import compare_strategies

        for s in ["naive", "hybrid"]:
            _make_scorecard(tmp_path, s)

        cfg = Config()
        cfg = cfg.model_copy(update={"output": OutputConfig(dir=str(tmp_path))})

        results_csv, report_html = compare_strategies(cfg)

        assert results_csv.exists()
        assert report_html.exists()

    def test_results_csv_has_correct_shape(self, tmp_path):
        from rag_eval.config import Config, OutputConfig
        from rag_eval.reporter import compare_strategies

        strategies = ["naive", "hybrid", "rerank"]
        for s in strategies:
            _make_scorecard(tmp_path, s)

        cfg = Config()
        cfg = cfg.model_copy(update={"output": OutputConfig(dir=str(tmp_path))})

        results_csv, _ = compare_strategies(cfg)
        df = pd.read_csv(results_csv)

        assert len(df) == len(strategies)
        assert "strategy" in df.columns
        for m in _METRICS:
            assert m in df.columns

    def test_compare_raises_for_missing_scorecards(self, tmp_path):
        from rag_eval.config import Config, OutputConfig
        from rag_eval.reporter import compare_strategies

        cfg = Config()
        cfg = cfg.model_copy(update={"output": OutputConfig(dir=str(tmp_path))})

        # tmp_path exists but has no scorecard CSVs
        with pytest.raises(FileNotFoundError):
            compare_strategies(cfg)


# ---------------------------------------------------------------------------
# CLI compare command
# ---------------------------------------------------------------------------


class TestCompareCLI:
    def test_compare_command_succeeds(self, tmp_path):
        from click.testing import CliRunner

        from rag_eval.cli import main

        for s in ["naive", "hybrid"]:
            _make_scorecard(tmp_path, s)

        config_yaml = tmp_path / "cfg.yaml"
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
                  - hybrid
                output:
                  dir: "{str(tmp_path).replace(chr(92), "/")}"
            """),
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(config_yaml), "compare"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert (tmp_path / "results.csv").exists()
        assert (tmp_path / "report.html").exists()

    def test_compare_command_exits_1_with_no_scorecards(self, tmp_path):
        from click.testing import CliRunner

        from rag_eval.cli import main

        config_yaml = tmp_path / "cfg.yaml"
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
            ["--config", str(config_yaml), "compare"],
        )

        assert result.exit_code == 1

    def test_compare_output_mentions_report_html(self, tmp_path):
        from click.testing import CliRunner

        from rag_eval.cli import main

        for s in ["naive"]:
            _make_scorecard(tmp_path, s)

        config_yaml = tmp_path / "cfg.yaml"
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
            ["--config", str(config_yaml), "compare"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "report.html" in result.output


# ---------------------------------------------------------------------------
# Smoke test: update placeholder check for compare command
# ---------------------------------------------------------------------------


class TestSmokeCompareNoLongerPlaceholder:
    """compare is now implemented; make sure the old placeholder is gone."""

    def test_compare_is_not_a_placeholder(self, tmp_path):
        from click.testing import CliRunner

        from rag_eval.cli import main

        # Without scorecards compare should exit 1 (real error), not 0 (placeholder)
        config_yaml = tmp_path / "cfg.yaml"
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
        result = runner.invoke(main, ["--config", str(config_yaml), "compare"])
        # A placeholder exits 0; a real command finding no files should exit 1
        assert result.exit_code != 0, (
            "compare still behaves like a placeholder (exit 0 on missing input)"
        )
