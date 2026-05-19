"""
Multi-Step Coherence metric.

LLM-as-judge evaluation of whether each step in a trace builds logically on
prior steps and whether the final output reflects the complete reasoning chain.

Uses the judge LLM configured in the root Config — no model is hardcoded.

Configurable
------------
min_steps : int (default 2)
    Traces with fewer steps than this threshold are skipped with a log message.

Target threshold (TDS "12-Metric Framework" article, Pratik R, May 2026):
  >0.85 coherence score on traces of 4+ steps

Output
------
CoherenceResult:
  scores          — {trace_id: score}  (0.0–1.0, or None if skipped)
  mean_score      — mean across evaluated traces
  skipped_traces  — list of trace IDs skipped (below min_steps)
  num_evaluated   — count of traces scored
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from rag_eval.agent_eval.trace import AgentTrace

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an objective evaluator of AI agent reasoning quality.
Score the coherence of the agent's multi-step reasoning on a scale from 0.0 to 1.0,
where 1.0 means perfectly coherent (every step builds on prior context, the final
answer fully reflects the reasoning chain) and 0.0 means completely incoherent
(steps are disconnected, the final answer ignores prior steps or contradicts them).

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{"score": <float between 0.0 and 1.0>, "reason": "<one sentence>"}
"""

_USER_TEMPLATE = """\
Trace ID: {trace_id}

Steps:
{steps_block}

Final output: {final_output}

Score the coherence of this trace."""


@dataclass
class CoherenceResult:
    """Output of score_coherence()."""

    scores: dict[str, float | None]
    mean_score: float
    skipped_traces: list[str]
    num_evaluated: int


def _format_steps(trace: AgentTrace) -> str:
    parts = []
    for step in trace.steps:
        tool_line = ""
        if step.tool_call:
            tc = step.tool_call
            status = "OK" if tc.success else f"FAILED ({tc.error_type})"
            tool_line = f"  Tool: {tc.name}({tc.args}) → {status}\n"
            if step.tool_result:
                tool_line += f"  Result: {step.tool_result}\n"
        parts.append(f"[{step.step_id}] Input: {step.input}\n{tool_line}  Output: {step.output}")
    return "\n\n".join(parts)


def _parse_score(raw: str) -> float | None:
    """Extract the float score from the judge's JSON response."""
    import json
    import re

    # Try direct JSON parse first
    try:
        data = json.loads(raw.strip())
        score = float(data["score"])
        return max(0.0, min(1.0, score))
    except Exception:
        pass

    # Fallback: regex on "score": 0.xx
    match = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', raw)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))

    return None


def score_coherence(
    traces: list[AgentTrace],
    judge_llm: BaseChatModel,
    min_steps: int = 2,
) -> CoherenceResult:
    """
    Score multi-step coherence for each trace using an LLM judge.

    Traces with fewer than min_steps steps are skipped — single-step traces
    have no cross-step reasoning to evaluate.

    Args:
        traces:    List of AgentTrace objects to evaluate.
        judge_llm: Instantiated LangChain BaseChatModel used as the judge.
                   Should come from get_llm(cfg.judge) so it stays model-agnostic.
        min_steps: Minimum number of steps required to evaluate a trace.
                   Traces below this threshold are skipped. Default: 2.

    Returns:
        CoherenceResult with per-trace scores and aggregate mean.
    """
    scores: dict[str, float | None] = {}
    skipped: list[str] = []

    for trace in traces:
        if trace.num_steps < min_steps:
            logger.info(
                "Skipping trace %s: %d steps < min_steps=%d",
                trace.trace_id,
                trace.num_steps,
                min_steps,
            )
            skipped.append(trace.trace_id)
            scores[trace.trace_id] = None
            continue

        prompt = _USER_TEMPLATE.format(
            trace_id=trace.trace_id,
            steps_block=_format_steps(trace),
            final_output=trace.final_output,
        )

        try:
            response = judge_llm.invoke(
                [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
            raw = response.content.strip()
            score = _parse_score(raw)
            if score is None:
                logger.warning(
                    "Could not parse coherence score for trace %s: %r", trace.trace_id, raw
                )
            scores[trace.trace_id] = score
        except Exception as exc:
            logger.error("Coherence judge failed for trace %s: %s", trace.trace_id, exc)
            scores[trace.trace_id] = None

    evaluated_scores = [v for v in scores.values() if v is not None]
    mean_score = sum(evaluated_scores) / len(evaluated_scores) if evaluated_scores else 0.0

    return CoherenceResult(
        scores=scores,
        mean_score=round(mean_score, 4),
        skipped_traces=skipped,
        num_evaluated=len(evaluated_scores),
    )
