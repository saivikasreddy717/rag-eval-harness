"""
Tool Execution Success metric.

Measures the fraction of tool calls that completed without error and categorises
failure modes to guide debugging.

No LLM calls required — uses tool_call.success and tool_call.error_type labels
already present in the trace.

Error categories (from TDS "12-Metric Framework" article, Pratik R, May 2026):
  schema_mismatch  — wrong argument types / missing required fields
  timeout          — tool did not respond within the time limit
  api_error        — upstream service returned an error
  other            — everything else

Target threshold: >0.98 success rate in production.

Output
------
ToolExecutionResult:
  success_rate      — fraction of tool calls that succeeded (0.0–1.0)
  error_breakdown   — {error_category: count}
  total_calls       — total tool calls evaluated
  successful_calls  — count of successful tool calls
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from rag_eval.agent_eval.trace import AgentTrace

_VALID_ERROR_TYPES = frozenset({"schema_mismatch", "timeout", "api_error", "other"})


@dataclass
class ToolExecutionResult:
    """Output of score_tool_execution()."""

    success_rate: float
    error_breakdown: dict[str, int]
    total_calls: int
    successful_calls: int


def score_tool_execution(traces: list[AgentTrace]) -> ToolExecutionResult:
    """
    Compute tool execution success rate across a set of traces.

    Only steps that contain a tool_call are evaluated.  Steps without a
    tool_call (pure reasoning steps) are skipped.

    Args:
        traces: List of AgentTrace objects.

    Returns:
        ToolExecutionResult with success rate and error categorisation.
    """
    total_calls = 0
    successful_calls = 0
    error_counts: dict[str, int] = defaultdict(int)

    for trace in traces:
        for step in trace.tool_steps:
            tc = step.tool_call
            total_calls += 1
            if tc.success:
                successful_calls += 1
            else:
                # Normalise error type: map unknown/None values to "other"
                raw = (tc.error_type or "other").lower()
                category = raw if raw in _VALID_ERROR_TYPES else "other"
                error_counts[category] += 1

    if total_calls == 0:
        return ToolExecutionResult(
            success_rate=0.0,
            error_breakdown={},
            total_calls=0,
            successful_calls=0,
        )

    success_rate = successful_calls / total_calls

    return ToolExecutionResult(
        success_rate=round(success_rate, 4),
        error_breakdown=dict(error_counts),
        total_calls=total_calls,
        successful_calls=successful_calls,
    )
