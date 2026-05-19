"""
Agent evaluation orchestrator.

Runs all enabled agent metrics over a set of AgentTrace objects and writes
a per-trace scorecard CSV to results/scorecard_agent.csv.

Scorecard columns
-----------------
  trace_id, num_steps, tool_selection_accuracy, tool_execution_success,
  coherence_score

An AGGREGATE row at the bottom contains column means.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from rag_eval.agent_eval.metrics.coherence import score_coherence
from rag_eval.agent_eval.metrics.tool_execution import score_tool_execution
from rag_eval.agent_eval.metrics.tool_selection import score_tool_selection
from rag_eval.agent_eval.trace import AgentTrace
from rag_eval.config import Config
from rag_eval.providers.llm import get_llm

console = Console()
logger = logging.getLogger(__name__)


def load_traces(traces_file: Path) -> list[AgentTrace]:
    """
    Load AgentTrace objects from a JSONL file (one trace per line).

    Args:
        traces_file: Path to a JSONL file.

    Returns:
        List of validated AgentTrace objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no valid traces are found.
    """
    traces_file = Path(traces_file)
    if not traces_file.exists():
        raise FileNotFoundError(
            f"Traces file not found: {traces_file}\nPass a JSONL file with one AgentTrace per line."
        )

    traces: list[AgentTrace] = []
    for i, line in enumerate(traces_file.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            traces.append(AgentTrace.model_validate(json.loads(line)))
        except Exception as exc:
            logger.warning("Skipping malformed trace on line %d: %s", i, exc)

    if not traces:
        raise ValueError(f"No valid traces found in {traces_file}.")
    return traces


def evaluate_agent_traces(
    traces_file: Path,
    cfg: Config,
) -> Path:
    """
    Evaluate agent traces with tool selection, tool execution, and coherence metrics.

    Args:
        traces_file: Path to a JSONL file of AgentTrace objects.
        cfg:         Root Config — used for judge LLM and output directory.

    Returns:
        Path to the written scorecard CSV.
    """
    traces = load_traces(traces_file)
    agent_cfg = cfg.agent_eval
    enabled_metrics = set(agent_cfg.metrics)

    console.print(
        f"[cyan]Agent eval:[/] {len(traces)} traces | "
        f"metrics: [cyan]{', '.join(sorted(enabled_metrics))}[/]"
    )

    # --- Tool selection (no LLM) ---
    sel_result = None
    if "tool_selection" in enabled_metrics:
        sel_result = score_tool_selection(traces)
        console.print(
            f"Tool selection accuracy: [bold green]{sel_result.accuracy:.4f}[/] "
            f"({sel_result.num_labelled} labelled steps)"
        )

    # --- Tool execution (no LLM) ---
    exec_result = None
    if "tool_execution" in enabled_metrics:
        exec_result = score_tool_execution(traces)
        console.print(
            f"Tool execution success: [bold green]{exec_result.success_rate:.4f}[/] "
            f"({exec_result.total_calls} tool calls)"
        )

    # --- Coherence (LLM judge) ---
    coh_result = None
    if "coherence" in enabled_metrics:
        judge_llm = get_llm(cfg.judge)
        min_steps = agent_cfg.coherence_min_steps
        console.print(
            f"[dim]Running coherence judge (min_steps={min_steps}) — "
            f"uses {cfg.judge.provider}/{cfg.judge.model}[/]"
        )
        coh_result = score_coherence(traces, judge_llm, min_steps=min_steps)
        console.print(
            f"Coherence mean score: [bold green]{coh_result.mean_score:.4f}[/] "
            f"({coh_result.num_evaluated} evaluated, "
            f"{len(coh_result.skipped_traces)} skipped)"
        )

    # Build per-trace DataFrame
    rows = []
    for trace in traces:
        row: dict = {
            "trace_id": trace.trace_id,
            "num_steps": trace.num_steps,
        }
        # Tool selection: accuracy is aggregate; per-trace we record whether each
        # labelled step was correct.  For the CSV we store overall accuracy once
        # in the aggregate row; per-trace gets the fraction of correct steps in
        # this trace.
        if sel_result is not None:
            trace_labelled = trace.labelled_tool_steps
            if trace_labelled:
                correct = sum(
                    1 for s in trace_labelled if s.tool_call and s.tool_call.name == s.expected_tool
                )
                row["tool_selection_accuracy"] = round(correct / len(trace_labelled), 4)
            else:
                row["tool_selection_accuracy"] = None

        if exec_result is not None:
            tool_steps = trace.tool_steps
            if tool_steps:
                ok = sum(1 for s in tool_steps if s.tool_call and s.tool_call.success)
                row["tool_execution_success"] = round(ok / len(tool_steps), 4)
            else:
                row["tool_execution_success"] = None

        if coh_result is not None:
            row["coherence_score"] = coh_result.scores.get(trace.trace_id)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Append AGGREGATE row
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    agg_row = df[numeric_cols].mean(skipna=True).to_dict()
    agg_row["trace_id"] = "AGGREGATE"
    agg_row["num_steps"] = df["num_steps"].mean()
    df = pd.concat([df, pd.DataFrame([agg_row])], ignore_index=True)

    # Write scorecard
    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "scorecard_agent.csv"
    df.to_csv(output_file, index=False)

    _print_agent_summary(df)
    console.print(f"[green]Agent scorecard saved:[/] {output_file}")

    return output_file


def _print_agent_summary(df: pd.DataFrame) -> None:
    agg = df[df["trace_id"] == "AGGREGATE"].iloc[0]
    table = Table(
        title="Agent evaluation (aggregate)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="cyan", min_width=28)
    table.add_column("Score", justify="right", style="bold green")

    for col in ["tool_selection_accuracy", "tool_execution_success", "coherence_score"]:
        if col in agg and not pd.isna(agg.get(col, float("nan"))):
            val = agg[col]
            table.add_row(col.replace("_", " ").title(), f"{val:.4f}")

    console.print(table)
