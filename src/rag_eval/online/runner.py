"""
Online evaluation runner.

Reads a JSONL stream of production traces (one record per line), samples
them via ReservoirSampler, routes each record to the appropriate eval path,
and writes scored results to a dated JSONL file.

Record routing
--------------
Each line is parsed as a dict and classified by shape:

  Agent trace  — has "trace_id" and "steps" keys → AgentTrace
  RAG prediction — has "question" and "answer" keys → RAG path (scores
                   hallucination_rate + context_relevance when enabled)
  Unknown       — logged as warning, skipped

Output: results/online/YYYY-MM-DD.jsonl (one scored record per line)
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from rag_eval.agent_eval.metrics.tool_execution import score_tool_execution
from rag_eval.agent_eval.metrics.tool_selection import score_tool_selection
from rag_eval.agent_eval.trace import AgentTrace
from rag_eval.config import Config
from rag_eval.online.sampler import ReservoirSampler
from rag_eval.online.storage import append_result, daily_output_path

logger = logging.getLogger(__name__)


def _is_agent_record(record: dict) -> bool:
    return "trace_id" in record and "steps" in record


def _is_rag_record(record: dict) -> bool:
    return "question" in record and "answer" in record


def _score_agent_record(record: dict, cfg: Config) -> dict[str, Any]:
    """Score a single agent trace record and return a result dict."""
    try:
        trace = AgentTrace.model_validate(record)
    except Exception as exc:
        logger.warning("Malformed agent trace %s: %s", record.get("trace_id", "?"), exc)
        return {"record_type": "agent", "error": str(exc), **_id_fields(record)}

    agent_cfg = cfg.agent_eval
    enabled = set(agent_cfg.metrics)
    result: dict[str, Any] = {
        "record_type": "agent",
        "trace_id": trace.trace_id,
        "num_steps": trace.num_steps,
    }

    if "tool_selection" in enabled:
        sel = score_tool_selection([trace])
        result["tool_selection_accuracy"] = sel.accuracy

    if "tool_execution" in enabled:
        exc_r = score_tool_execution([trace])
        result["tool_execution_success"] = exc_r.success_rate

    # Coherence requires an LLM call — skip in online runner for single records
    # to avoid per-record LLM overhead.  Coherence is batched in storage rollup.
    result["coherence_score"] = None

    return result


def _score_rag_record(record: dict, cfg: Config) -> dict[str, Any]:
    """Score a single RAG prediction record and return a result dict."""
    extra = cfg.metrics_extra
    result: dict[str, Any] = {
        "record_type": "rag",
        "question_id": record.get("id", "unknown"),
        "strategy": record.get("strategy", "unknown"),
        "latency_ms": record.get("latency_ms"),
        "retrieval_latency_ms": record.get("retrieval_latency_ms"),
    }

    # Hallucination rate and context relevance require an LLM judge.
    # In the online runner we record placeholders; the daily rollup can batch
    # these for cost efficiency, or they can be computed lazily.
    if extra.hallucination_rate:
        result["hallucination_rate"] = None  # deferred to rollup
    if extra.context_relevance:
        result["context_relevance"] = None  # deferred to rollup

    return result


def _id_fields(record: dict) -> dict:
    return {k: record.get(k) for k in ("trace_id", "id", "question_id") if k in record}


def run_online_eval(
    traces_stream_file: Path,
    cfg: Config,
    sample_rate: float | None = None,
    run_date: date | None = None,
) -> Path:
    """
    Sample and score records from a JSONL stream file.

    Args:
        traces_stream_file: Path to the JSONL stream (production traces).
        cfg:                Root Config — used for output dir and enabled metrics.
        sample_rate:        Override the sample rate from config (0 < rate <= 1.0).
        run_date:           Date for output file naming. Defaults to today.

    Returns:
        Path to the dated JSONL output file.
    """
    traces_stream_file = Path(traces_stream_file)
    if not traces_stream_file.exists():
        raise FileNotFoundError(f"Stream file not found: {traces_stream_file}")

    online_cfg = cfg.online
    rate = sample_rate if sample_rate is not None else online_cfg.sample_rate
    today = run_date or date.today()

    sampler = ReservoirSampler(sample_rate=rate, seed=online_cfg.seed)
    output_dir = Path(online_cfg.output_dir)
    output_path = daily_output_path(output_dir, today)

    raw_records: list[dict] = []
    with open(traces_stream_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                raw_records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON line: %s", exc)

    sampled = sampler.add_all(raw_records)
    logger.info("Sampled %d / %d records (rate=%.2f)", len(sampled), len(raw_records), rate)

    scored_records = []
    for record in sampled:
        if _is_agent_record(record):
            scored = _score_agent_record(record, cfg)
        elif _is_rag_record(record):
            scored = _score_rag_record(record, cfg)
        else:
            logger.warning("Unknown record shape, skipping: %s", list(record.keys())[:5])
            continue
        scored_records.append(scored)

    for scored in scored_records:
        append_result(output_path, scored)

    logger.info("Wrote %d scored records to %s", len(scored_records), output_path)
    return output_path
