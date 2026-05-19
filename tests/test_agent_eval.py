"""
Tests for the agent evaluation module.

All LLM calls (coherence judge) are mocked — no API keys required.
CI-safe.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from rag_eval.agent_eval.fixtures import (
    all_sample_traces,
    make_coherent_multi_step_trace,
    make_multi_step_trace,
    make_schema_mismatch_trace,
    make_single_step_tool_trace,
    make_single_step_trace,
    make_wrong_tool_trace,
)
from rag_eval.agent_eval.trace import AgentStep, AgentTrace, ToolCall

# ---------------------------------------------------------------------------
# Trace schema validation
# ---------------------------------------------------------------------------


class TestTraceSchema:
    def test_valid_single_step_trace(self):
        trace = make_single_step_trace()
        assert trace.trace_id == "t-001"
        assert trace.num_steps == 1
        assert trace.final_output == "The capital of France is Paris."

    def test_valid_tool_call(self):
        trace = make_single_step_tool_trace()
        step = trace.steps[0]
        assert step.tool_call is not None
        assert step.tool_call.name == "web_search"
        assert step.tool_call.success is True
        assert step.tool_call.error_type is None

    def test_tool_call_failure_with_error_type(self):
        trace = make_schema_mismatch_trace()
        tc = trace.steps[0].tool_call
        assert tc.success is False
        assert tc.error_type == "schema_mismatch"

    def test_error_type_must_be_none_when_success(self):
        with pytest.raises(Exception):
            ToolCall(name="tool", args={}, success=True, error_type="api_error")

    def test_steps_must_be_ordered(self):
        with pytest.raises(Exception):
            AgentTrace(
                trace_id="bad",
                steps=[
                    AgentStep(
                        step_id=1,  # should be 0
                        input="x",
                        output="y",
                        timestamp="2026-01-01T00:00:00Z",
                    )
                ],
                final_output="done",
            )

    def test_empty_steps_rejected(self):
        with pytest.raises(Exception):
            AgentTrace(trace_id="empty", steps=[], final_output="done")

    def test_labelled_tool_steps_property(self):
        trace = make_multi_step_trace("t-x", num_steps=4, all_success=True)
        labelled = trace.labelled_tool_steps
        assert len(labelled) == 4
        assert all(s.expected_tool is not None for s in labelled)

    def test_tool_steps_property_excludes_no_call_steps(self):
        trace = make_single_step_trace()  # no tool call
        assert trace.tool_steps == []

    def test_trace_from_jsonl_round_trip(self, tmp_path):
        trace = make_coherent_multi_step_trace()
        line = trace.model_dump_json()
        restored = AgentTrace.model_validate_json(line)
        assert restored.trace_id == trace.trace_id
        assert restored.num_steps == trace.num_steps

    def test_optional_fields_default_to_none(self):
        trace = make_single_step_trace()
        step = trace.steps[0]
        assert step.tool_call is None
        assert step.tool_result is None
        assert step.expected_tool is None
        assert step.expected_args is None


# ---------------------------------------------------------------------------
# Tool selection accuracy
# ---------------------------------------------------------------------------


class TestToolSelectionAccuracy:
    def test_perfect_accuracy(self):
        from rag_eval.agent_eval.metrics.tool_selection import score_tool_selection

        traces = [make_multi_step_trace("t-ok", num_steps=4, all_success=True)]
        result = score_tool_selection(traces)
        assert result.accuracy == pytest.approx(1.0)
        assert result.num_labelled == 4

    def test_wrong_tool_lowers_accuracy(self):
        from rag_eval.agent_eval.metrics.tool_selection import score_tool_selection

        # t-004: 1 step, expected "document_retriever", got "web_search"
        traces = [make_wrong_tool_trace()]
        result = score_tool_selection(traces)
        assert result.accuracy == pytest.approx(0.0)
        assert result.num_labelled == 1

    def test_mixed_accuracy(self):
        from rag_eval.agent_eval.metrics.tool_selection import score_tool_selection

        correct = make_multi_step_trace("c", num_steps=2, all_success=True)
        wrong = make_wrong_tool_trace("w")
        result = score_tool_selection([correct, wrong])
        # 2 correct out of 3 labelled
        assert result.accuracy == pytest.approx(2 / 3, rel=1e-4)

    def test_no_labelled_steps_returns_zero(self):
        from rag_eval.agent_eval.metrics.tool_selection import score_tool_selection

        # single-step trace with no expected_tool
        trace = make_single_step_trace()
        result = score_tool_selection([trace])
        assert result.accuracy == 0.0
        assert result.num_labelled == 0

    def test_per_tool_breakdown_present(self):
        from rag_eval.agent_eval.metrics.tool_selection import score_tool_selection

        traces = [make_multi_step_trace("t", num_steps=4, all_success=True)]
        result = score_tool_selection(traces)
        assert len(result.per_tool) > 0
        for tool, stats in result.per_tool.items():
            assert "accuracy" in stats
            assert 0.0 <= stats["accuracy"] <= 1.0

    def test_confusion_matrix_off_diagonal_for_wrong_tool(self):
        from rag_eval.agent_eval.metrics.tool_selection import score_tool_selection

        trace = make_wrong_tool_trace()
        result = score_tool_selection([trace])
        # predicted=web_search, expected=document_retriever
        assert result.confusion_matrix["web_search"]["document_retriever"] == 1


# ---------------------------------------------------------------------------
# Tool execution success
# ---------------------------------------------------------------------------


class TestToolExecutionSuccess:
    def test_all_successful(self):
        from rag_eval.agent_eval.metrics.tool_execution import score_tool_execution

        traces = [make_multi_step_trace("t", num_steps=4, all_success=True)]
        result = score_tool_execution(traces)
        assert result.success_rate == pytest.approx(1.0)
        assert result.total_calls == 4
        assert result.successful_calls == 4

    def test_schema_mismatch_counted(self):
        from rag_eval.agent_eval.metrics.tool_execution import score_tool_execution

        traces = [make_schema_mismatch_trace()]
        result = score_tool_execution(traces)
        assert result.success_rate == pytest.approx(0.0)
        assert result.error_breakdown.get("schema_mismatch", 0) == 1

    def test_no_tool_calls_returns_zero(self):
        from rag_eval.agent_eval.metrics.tool_execution import score_tool_execution

        traces = [make_single_step_trace()]  # no tool call
        result = score_tool_execution(traces)
        assert result.total_calls == 0
        assert result.success_rate == 0.0

    def test_mixed_success_failure(self):
        from rag_eval.agent_eval.metrics.tool_execution import score_tool_execution

        traces = [
            make_single_step_tool_trace("ok", success=True),
            make_schema_mismatch_trace("fail"),
        ]
        result = score_tool_execution(traces)
        assert result.total_calls == 2
        assert result.successful_calls == 1
        assert result.success_rate == pytest.approx(0.5)

    def test_unknown_error_type_maps_to_other(self):
        from rag_eval.agent_eval.metrics.tool_execution import score_tool_execution

        trace = AgentTrace(
            trace_id="unk",
            steps=[
                AgentStep(
                    step_id=0,
                    input="x",
                    tool_call=ToolCall(
                        name="foo",
                        args={},
                        success=False,
                        error_type="completely_unknown_type",
                    ),
                    output="failed",
                    timestamp="2026-01-01T00:00:00Z",
                )
            ],
            final_output="failed",
        )
        result = score_tool_execution([trace])
        assert result.error_breakdown.get("other", 0) == 1


# ---------------------------------------------------------------------------
# Multi-step coherence (LLM judge mocked)
# ---------------------------------------------------------------------------


class TestCoherence:
    def _make_mock_judge(self, score: float = 0.9) -> MagicMock:
        mock = MagicMock()
        mock.invoke.return_value = MagicMock(
            content=json.dumps({"score": score, "reason": "Test mock response."})
        )
        return mock

    def test_high_coherence_trace_scores_well(self):
        from rag_eval.agent_eval.metrics.coherence import score_coherence

        trace = make_coherent_multi_step_trace()
        judge = self._make_mock_judge(score=0.95)
        result = score_coherence([trace], judge)
        assert result.num_evaluated == 1
        assert result.mean_score == pytest.approx(0.95)

    def test_single_step_trace_skipped(self):
        from rag_eval.agent_eval.metrics.coherence import score_coherence

        trace = make_single_step_trace()
        judge = self._make_mock_judge()
        result = score_coherence([trace], judge, min_steps=2)
        assert trace.trace_id in result.skipped_traces
        assert result.num_evaluated == 0
        judge.invoke.assert_not_called()

    def test_min_steps_respected(self):
        from rag_eval.agent_eval.metrics.coherence import score_coherence

        two_step = make_multi_step_trace("t", num_steps=2)
        judge = self._make_mock_judge(score=0.8)
        result = score_coherence([two_step], judge, min_steps=3)
        assert result.num_evaluated == 0
        assert len(result.skipped_traces) == 1

    def test_parse_failure_returns_none_for_that_trace(self):
        from rag_eval.agent_eval.metrics.coherence import score_coherence

        trace = make_coherent_multi_step_trace()
        judge = MagicMock()
        judge.invoke.return_value = MagicMock(content="not valid json at all")
        result = score_coherence([trace], judge)
        assert result.scores[trace.trace_id] is None
        assert result.num_evaluated == 0

    def test_llm_exception_does_not_crash(self):
        from rag_eval.agent_eval.metrics.coherence import score_coherence

        trace = make_coherent_multi_step_trace()
        judge = MagicMock()
        judge.invoke.side_effect = RuntimeError("API down")
        result = score_coherence([trace], judge)
        assert result.scores[trace.trace_id] is None


# ---------------------------------------------------------------------------
# Agent evaluator orchestrator (evaluator.py)
# ---------------------------------------------------------------------------


class TestAgentEvaluatorOrchestrator:
    def test_load_traces_from_jsonl(self, tmp_path):
        from rag_eval.agent_eval.evaluator import load_traces

        traces_file = tmp_path / "traces.jsonl"
        traces_file.write_text(
            "\n".join(t.model_dump_json() for t in all_sample_traces()),
            encoding="utf-8",
        )
        loaded = load_traces(traces_file)
        assert len(loaded) == 8

    def test_load_traces_skips_blank_lines(self, tmp_path):
        from rag_eval.agent_eval.evaluator import load_traces

        traces_file = tmp_path / "traces.jsonl"
        lines = [t.model_dump_json() for t in all_sample_traces()[:3]]
        traces_file.write_text("\n\n".join(lines) + "\n\n", encoding="utf-8")
        loaded = load_traces(traces_file)
        assert len(loaded) == 3

    def test_load_traces_raises_for_missing_file(self, tmp_path):
        from rag_eval.agent_eval.evaluator import load_traces

        with pytest.raises(FileNotFoundError):
            load_traces(tmp_path / "nope.jsonl")

    @patch("rag_eval.agent_eval.evaluator.get_llm")
    def test_scorecard_csv_written(self, mock_get_llm, tmp_path):
        from rag_eval.agent_eval.evaluator import evaluate_agent_traces
        from rag_eval.config import AgentEvalConfig, Config, OutputConfig

        mock_judge = MagicMock()
        mock_judge.invoke.return_value = MagicMock(content='{"score": 0.85, "reason": "ok"}')
        mock_get_llm.return_value = mock_judge

        traces_file = tmp_path / "traces.jsonl"
        traces_file.write_text(
            "\n".join(t.model_dump_json() for t in all_sample_traces()),
            encoding="utf-8",
        )

        cfg = Config()
        cfg = cfg.model_copy(
            update={
                "output": OutputConfig(dir=str(tmp_path)),
                "agent_eval": AgentEvalConfig(enabled=True),
            }
        )

        out = evaluate_agent_traces(traces_file, cfg)
        assert out.exists()
        assert out.suffix == ".csv"

    @patch("rag_eval.agent_eval.evaluator.get_llm")
    def test_scorecard_has_aggregate_row(self, mock_get_llm, tmp_path):
        import pandas as pd

        from rag_eval.agent_eval.evaluator import evaluate_agent_traces
        from rag_eval.config import AgentEvalConfig, Config, OutputConfig

        mock_get_llm.return_value = MagicMock(
            invoke=MagicMock(return_value=MagicMock(content='{"score": 0.9, "reason": "good"}'))
        )

        traces_file = tmp_path / "traces.jsonl"
        traces_file.write_text(
            "\n".join(t.model_dump_json() for t in all_sample_traces()),
            encoding="utf-8",
        )
        cfg = Config()
        cfg = cfg.model_copy(
            update={
                "output": OutputConfig(dir=str(tmp_path)),
                "agent_eval": AgentEvalConfig(enabled=True),
            }
        )
        out = evaluate_agent_traces(traces_file, cfg)
        df = pd.read_csv(out)
        assert "AGGREGATE" in df["trace_id"].values
