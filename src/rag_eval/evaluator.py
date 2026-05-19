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

Columns (base)
--------------
  strategy, question_id, faithfulness, answer_relevancy,
  context_precision, context_recall, answer_correctness,
  latency_ms, cost_usd, prompt_tokens, completion_tokens

Optional columns (enabled in config under metrics_extra)
---------------------------------------------------------
  hallucination_rate    — fraction of answer claims not grounded in context
  context_relevance     — mean LLM-judge relevance score across retrieved chunks
  retrieval_latency_ms  — per-query retrieval-only latency (from runner)

AGGREGATE row additions (when retrieval_latency enabled)
--------------------------------------------------------
  retrieval_latency_p50_ms, retrieval_latency_p95_ms, retrieval_latency_p99_ms

Rate-limit note
---------------
RAGAS makes ~2 LLM judge calls per metric per question.
For 500 questions × 5 metrics = ~5,000 judge calls per strategy.
On Groq free tier (14,400 req/day for Llama 3.3 70B) that is ~8 hours.
Use --max-questions to evaluate a subset first, or upgrade to paid tier.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
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

logger = logging.getLogger(__name__)

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
                "retrieval_latency_ms": p.get("retrieval_latency_ms", 0.0),
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

    # --- Optional extra metrics (gated by metrics_extra config) ---
    extra_cfg = cfg.metrics_extra

    if extra_cfg.hallucination_rate:
        console.print(
            f"[dim]Scoring hallucination rate ({len(valid_predictions)} predictions)...[/]"
        )
        judge_llm_for_extras = get_llm(cfg.judge)
        hal_rates = score_hallucination_rate(valid_predictions, judge_llm_for_extras)
        scores_df["hallucination_rate"] = hal_rates

    if extra_cfg.context_relevance:
        console.print(
            f"[dim]Scoring context relevance ({len(valid_predictions)} predictions)...[/]"
        )
        if not extra_cfg.hallucination_rate:
            judge_llm_for_extras = get_llm(cfg.judge)
        ctx_scores = score_context_relevance(valid_predictions, judge_llm_for_extras)
        scores_df["context_relevance"] = ctx_scores

    scores_df.insert(0, "strategy", strategy_name)

    # Append AGGREGATE row (column means + retrieval latency percentiles)
    numeric_cols = scores_df.select_dtypes(include="number").columns.tolist()
    agg_row = scores_df[numeric_cols].mean().to_dict()
    agg_row["strategy"] = strategy_name
    agg_row["question_id"] = "AGGREGATE"

    # Add retrieval latency percentiles to aggregate when the column is present
    if extra_cfg.retrieval_latency and "retrieval_latency_ms" in scores_df.columns:
        ret_lat = scores_df["retrieval_latency_ms"].dropna()
        if len(ret_lat) > 0:
            agg_row["retrieval_latency_p50_ms"] = round(float(np.percentile(ret_lat, 50)), 2)
            agg_row["retrieval_latency_p95_ms"] = round(float(np.percentile(ret_lat, 95)), 2)
            agg_row["retrieval_latency_p99_ms"] = round(float(np.percentile(ret_lat, 99)), 2)

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

    for extra_col in ("hallucination_rate", "context_relevance"):
        if extra_col in agg and not pd.isna(agg.get(extra_col, float("nan"))):
            table.add_row(extra_col.replace("_", " ").title(), f"{agg[extra_col]:.4f}")

    if "latency_ms" in agg:
        table.add_row("Latency (mean ms)", f"{agg['latency_ms']:.1f}")
    if "retrieval_latency_p95_ms" in agg:
        table.add_row("Retrieval p95 (ms)", f"{agg['retrieval_latency_p95_ms']:.1f}")
    if "cost_usd" in agg:
        table.add_row("Cost / query (USD)", f"${agg['cost_usd']:.6f}")

    console.print(table)


# ---------------------------------------------------------------------------
# Hallucination rate
# ---------------------------------------------------------------------------

_HALLUCINATION_SYSTEM = """\
You are a factual accuracy evaluator. Your task is to assess whether the claims
in an AI-generated answer are grounded.

For each atomic claim in the answer, determine whether it is:
  (a) Supported by the retrieved context passages, OR
  (b) Verifiable common knowledge (e.g. the sky is blue, water is H2O)

A claim is hallucinated if it is NEITHER supported by the context NOR common knowledge.

Respond with ONLY a JSON object in this exact format:
{
  "total_claims": <int>,
  "hallucinated_claims": <int>,
  "hallucination_rate": <float 0.0-1.0>,
  "examples": ["<hallucinated claim 1>", ...]
}
No markdown. No extra text."""

_HALLUCINATION_USER = """\
Question: {question}

Retrieved context:
{context_block}

Answer to evaluate:
{answer}

Evaluate the factual grounding of each claim in the answer."""


def score_hallucination_rate(
    predictions: list[dict],
    judge_llm,
) -> list[float | None]:
    """
    Score hallucination rate for each prediction using an LLM judge.

    The judge extracts atomic claims from the answer and flags any claim that
    is not supported by the retrieved context AND not verifiable as common
    knowledge.  This is distinct from RAGAS faithfulness, which uses an NLI
    decomposition approach.

    Args:
        predictions: List of prediction dicts (with question, answer, contexts).
        judge_llm:   Instantiated LangChain BaseChatModel.

    Returns:
        List of hallucination rates (0.0–1.0) or None on parse failure,
        one entry per prediction.
    """
    rates: list[float | None] = []
    for pred in predictions:
        context_block = "\n\n".join(
            f"[{i + 1}] {ctx}" for i, ctx in enumerate(pred.get("contexts", []))
        )
        prompt = _HALLUCINATION_USER.format(
            question=pred.get("question", ""),
            context_block=context_block or "(no context)",
            answer=pred.get("answer", ""),
        )
        try:
            response = judge_llm.invoke(
                [
                    SystemMessage(content=_HALLUCINATION_SYSTEM),
                    HumanMessage(content=prompt),
                ]
            )
            raw = response.content.strip()
            rate = _parse_hallucination_rate(raw)
            rates.append(rate)
        except Exception as exc:
            logger.warning("Hallucination judge failed: %s", exc)
            rates.append(None)
    return rates


def _parse_hallucination_rate(raw: str) -> float | None:
    """Extract hallucination_rate float from judge JSON response."""
    try:
        data = json.loads(raw)
        return max(0.0, min(1.0, float(data["hallucination_rate"])))
    except Exception:
        pass
    match = re.search(r'"hallucination_rate"\s*:\s*([0-9]*\.?[0-9]+)', raw)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    return None


# ---------------------------------------------------------------------------
# Context relevance (per-chunk LLM judge)
# ---------------------------------------------------------------------------

_CONTEXT_RELEVANCE_SYSTEM = """\
You are a retrieval quality evaluator. Score how relevant the given passage is
to answering the given question on a scale from 0.0 to 1.0.

1.0 = The passage directly answers or strongly supports answering the question.
0.5 = The passage is partially related but does not fully address the question.
0.0 = The passage is irrelevant to the question.

Respond with ONLY a JSON object: {"score": <float 0.0-1.0>}
No markdown. No extra text."""

_CONTEXT_RELEVANCE_USER = "Question: {question}\n\nPassage: {chunk}"


def score_context_relevance(
    predictions: list[dict],
    judge_llm,
) -> list[float | None]:
    """
    Score mean context relevance per prediction using an LLM judge.

    Each retrieved chunk is scored 0–1 for its relevance to the question,
    then scores are averaged across all chunks for that prediction.  This is
    distinct from RAGAS context_precision (which uses an MRR-style ranking
    approach); this metric produces an absolute per-chunk relevance score.

    Args:
        predictions: List of prediction dicts (with question and contexts).
        judge_llm:   Instantiated LangChain BaseChatModel.

    Returns:
        List of mean context relevance scores (0.0–1.0) or None,
        one entry per prediction.
    """
    mean_scores: list[float | None] = []
    for pred in predictions:
        question = pred.get("question", "")
        chunks = pred.get("contexts", [])
        if not chunks:
            mean_scores.append(None)
            continue
        chunk_scores: list[float] = []
        for chunk in chunks:
            prompt = _CONTEXT_RELEVANCE_USER.format(question=question, chunk=chunk)
            try:
                response = judge_llm.invoke(
                    [
                        SystemMessage(content=_CONTEXT_RELEVANCE_SYSTEM),
                        HumanMessage(content=prompt),
                    ]
                )
                raw = response.content.strip()
                score = _parse_context_relevance_score(raw)
                if score is not None:
                    chunk_scores.append(score)
            except Exception as exc:
                logger.warning("Context relevance judge failed for chunk: %s", exc)
        mean_scores.append(
            round(sum(chunk_scores) / len(chunk_scores), 4) if chunk_scores else None
        )
    return mean_scores


def _parse_context_relevance_score(raw: str) -> float | None:
    """Extract score float from judge JSON response."""
    try:
        data = json.loads(raw)
        return max(0.0, min(1.0, float(data["score"])))
    except Exception:
        pass
    match = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', raw)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    return None
