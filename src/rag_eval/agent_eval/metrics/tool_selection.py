"""
Tool Selection Accuracy metric.

Measures how often the agent chose the correct tool at labelled decision points.

No LLM calls required — pure label comparison against expected_tool ground truth.

Target thresholds (from TDS "12-Metric Framework" article, Pratik R, May 2026):
  >0.92  for binary tool choices (2 tools)
  >0.85  for 5+ tool vocabulary

Output
------
ToolSelectionResult:
  accuracy          — overall fraction of correct selections (0.0–1.0)
  per_tool          — per-tool breakdown {tool_name: {"correct": int, "total": int, "accuracy": float}}
  confusion_matrix  — {predicted_tool: {expected_tool: count}} raw counts
  num_labelled      — total labelled decision points evaluated
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from rag_eval.agent_eval.trace import AgentStep, AgentTrace


@dataclass
class ToolSelectionResult:
    """Output of score_tool_selection()."""

    accuracy: float
    per_tool: dict[str, dict[str, float | int]]
    confusion_matrix: dict[str, dict[str, int]]
    num_labelled: int


def score_tool_selection(traces: list[AgentTrace]) -> ToolSelectionResult:
    """
    Compute tool selection accuracy across a set of traces.

    Only steps with an expected_tool label are evaluated.  Steps without
    expected_tool are silently skipped.

    Args:
        traces: List of AgentTrace objects, at least some of which have
                expected_tool labels on their tool-calling steps.

    Returns:
        ToolSelectionResult with overall accuracy, per-tool breakdown,
        and raw confusion matrix.
    """
    labelled_steps: list[AgentStep] = []
    for trace in traces:
        labelled_steps.extend(trace.labelled_tool_steps)

    if not labelled_steps:
        return ToolSelectionResult(
            accuracy=0.0,
            per_tool={},
            confusion_matrix={},
            num_labelled=0,
        )

    # Accumulate per-tool counts and confusion matrix
    per_tool_correct: dict[str, int] = defaultdict(int)
    per_tool_total: dict[str, int] = defaultdict(int)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for step in labelled_steps:
        expected = step.expected_tool  # guaranteed non-None by labelled_tool_steps
        predicted = step.tool_call.name if step.tool_call else "__no_call__"

        per_tool_total[expected] += 1
        confusion[predicted][expected] += 1

        if predicted == expected:
            per_tool_correct[expected] += 1

    total_correct = sum(per_tool_correct.values())
    accuracy = total_correct / len(labelled_steps)

    per_tool = {
        tool: {
            "correct": per_tool_correct[tool],
            "total": per_tool_total[tool],
            "accuracy": per_tool_correct[tool] / per_tool_total[tool],
        }
        for tool in per_tool_total
    }

    return ToolSelectionResult(
        accuracy=round(accuracy, 4),
        per_tool=per_tool,
        confusion_matrix={k: dict(v) for k, v in confusion.items()},
        num_labelled=len(labelled_steps),
    )
