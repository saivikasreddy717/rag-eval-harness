"""
RAGAS evaluation layer for rag-eval-harness.

Scores predictions from runner.py with five RAGAS metrics and writes
a per-question scorecard CSV alongside aggregate summary stats.

RAGAS version: 0.4.x (uses ragas.metrics.collections, EvaluationDataset,
SingleTurnSample — the stable post-0.4 API).

Output
------
results/scorecard_naive.csv   — per-question scores + latency + cost
results/scorecard_hybrid.csv  — same for each strategy

Columns
-------
  strategy, question_id, faithfulness, answer_relevancy,
  context_precision, context_recall, answer_correctness,
  latency_ms, cost_usd, prompt_tokens, completion_tokens

An AGGREGATE row at the bottom contains column means.

Rate-limit note
---------------
RAGAS makes ~2 LLM judge calls per metric per question.
For 500 questions × 5 metrics = ~5,000 judge calls per strategy.
On Groq free tier (14,400 req/day for Llama 3.3 70B) that is ~8 hours.
Use --max-questions to evaluate a subset first, or upgrade to paid tier.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.collections import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from rich.console import Console
from rich.table import Table

from rag_eval.config import Config
from rag_eval.providers.embeddings import get_embeddings
from rag_eval.providers.llm import get_llm

console = Console()

# Ordered list of metric names — used for display and column checks
_METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]

# Metric objects matching the order above
_METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
]


def _load_predictions(predictions_file: Path) -> list[dict]:
    """Read a predictions JSONL file into a list of dicts."""
    predictions_file = Path(predictions_file)
    if not predictions_file.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {predictions_file}\nRun: python -m rag_eval run first."
        )
    predictions = []
    with open(predictions_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions


def _build_ragas_dataset(predictions: list[dict]) -> EvaluationDataset:
    """Build a RAGAS EvaluationDataset from a list of prediction dicts."""
    samples = []
    for pred in predictions:
        # Skip error predictions — RAGAS can't score empty answers
        if pred.get("metadata", {}).get("error") or not pred.get("answer"):
            continue

        samples.append(
            SingleTurnSample(
                user_input=pred["question"],
                response=pred["answer"],
                retrieved_contexts=pred["contexts"],
                reference=pred.get("reference_answer", ""),
                reference_contexts=pred.get("reference_contexts") or None,
            )
        )

    return EvaluationDataset(samples=samples)


def evaluate_predictions(
    predictions_file: Path,
    cfg: Config,
    max_questions: int | None = None,
) -> Path:
    """
    Score a predictions JSONL file with RAGAS and write a scorecard CSV.

    Args:
        predictions_file: Path to predictions_{strategy}.jsonl from runner.py.
        cfg:              Full Config — used for judge LLM and embeddings.
        max_questions:    Evaluate only the first N questions (useful for testing
                          or when rate limits are a concern).

    Returns:
        Path to the scorecard CSV file.
    """
    predictions_file = Path(predictions_file)

    # Load predictions
    predictions = _load_predictions(predictions_file)
    strategy_name = predictions[0]["strategy"] if predictions else "unknown"

    if max_questions:
        predictions = predictions[:max_questions]

    valid_predictions = [
        p for p in predictions if p.get("answer") and not p.get("metadata", {}).get("error")
    ]
    skipped = len(predictions) - len(valid_predictions)

    console.print(
        f"[cyan]Evaluating[/] [bold]{strategy_name}[/]: "
        f"{len(valid_predictions)} questions"
        + (f" ([yellow]{skipped} skipped — errors[/])" if skipped else "")
    )

    if not valid_predictions:
        raise ValueError(
            f"No valid predictions in {predictions_file}. "
            "All predictions had errors. Re-run the strategy first."
        )

    # Build RAGAS dataset
    dataset = _build_ragas_dataset(valid_predictions)

    # Set up judge LLM and embeddings
    judge_llm = LangchainLLMWrapper(get_llm(cfg.judge))
    judge_embeddings = LangchainEmbeddingsWrapper(get_embeddings(cfg.embeddings))

    # Run config: generous timeout + retries for rate-limited APIs
    run_config = RunConfig(
        timeout=180,  # seconds per individual LLM call
        max_retries=5,
        max_wait=90,  # max seconds to wait between retries
    )

    console.print(
        f"Judge: [cyan]{cfg.judge.provider}/{cfg.judge.model}[/] | "
        f"Metrics: [cyan]{', '.join(_METRIC_NAMES)}[/]"
    )
    console.print("[dim]This may take a while — RAGAS makes multiple LLM calls per question.[/]")

    # Run RAGAS evaluate
    result = evaluate(
        dataset=dataset,
        metrics=_METRICS,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
        raise_exceptions=False,  # return NaN for failures, don't abort
        show_progress=True,
    )

    # Convert to DataFrame
    scores_df = result.to_pandas()

    # Merge telemetry columns (latency, cost) from our predictions
    telemetry = pd.DataFrame(
        [
            {
                "question_id": p["id"],
                "latency_ms": p.get("latency_ms", 0.0),
                "cost_usd": p.get("cost_usd", 0.0),
                "prompt_tokens": p.get("prompt_tokens", 0),
                "completion_tokens": p.get("completion_tokens", 0),
            }
            for p in valid_predictions
        ]
    )

    # Merge on position (RAGAS preserves insertion order)
    if len(telemetry) == len(scores_df):
        scores_df = pd.concat(
            [scores_df.reset_index(drop=True), telemetry.reset_index(drop=True)],
            axis=1,
        )
    else:
        console.print(
            "[yellow]Warning:[/] Telemetry row count mismatch — "
            "latency/cost columns omitted from scorecard."
        )

    scores_df.insert(0, "strategy", strategy_name)

    # Append AGGREGATE row (column means across numeric columns)
    numeric_cols = scores_df.select_dtypes(include="number").columns.tolist()
    agg_row = scores_df[numeric_cols].mean().to_dict()
    agg_row["strategy"] = strategy_name
    agg_row["question_id"] = "AGGREGATE"
    scores_df = pd.concat(
        [scores_df, pd.DataFrame([agg_row])],
        ignore_index=True,
    )

    # Save scorecard CSV
    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"scorecard_{strategy_name}.csv"
    scores_df.to_csv(output_file, index=False)

    _print_aggregate_table(strategy_name, agg_row)
    console.print(f"[green]Scorecard saved:[/] {output_file}")

    return output_file


def _print_aggregate_table(strategy_name: str, agg: dict) -> None:
    """Print a rich table of aggregate RAGAS scores."""
    table = Table(
        title=f"RAGAS scores: {strategy_name} (aggregate)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="cyan", min_width=22)
    table.add_column("Score", justify="right", style="bold green")

    for metric in _METRIC_NAMES:
        val = agg.get(metric)
        score_str = f"{val:.4f}" if val is not None and not pd.isna(val) else "N/A"
        table.add_row(metric.replace("_", " ").title(), score_str)

    if "latency_ms" in agg:
        table.add_row("Latency (mean ms)", f"{agg['latency_ms']:.1f}")
    if "cost_usd" in agg:
        table.add_row("Cost / query (USD)", f"${agg['cost_usd']:.6f}")

    console.print(table)
