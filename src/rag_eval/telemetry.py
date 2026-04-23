"""
Cost and latency tracking for rag-eval-harness.

Tracks per-query token usage and estimates cost in USD so the
comparison report can show a cost/quality tradeoff scatter plot.

Price table uses public list prices as of 2025. Override by setting
custom prices in your config if needed. Providers with no API cost
(ollama, local embeddings) are assigned $0.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

import tiktoken

# ---------------------------------------------------------------------------
# Price table: (provider, model_prefix) -> (input $/1M tokens, output $/1M tokens)
# Model prefix matching: checks if model startswith the key.
# ---------------------------------------------------------------------------

_PRICE_TABLE: dict[tuple[str, str], tuple[float, float]] = {
    # Groq-hosted models
    ("groq", "meta-llama/llama-4-scout"): (0.11, 0.34),
    ("groq", "meta-llama/llama-4-maverick"): (0.50, 0.77),
    ("groq", "llama-3.3-70b"): (0.59, 0.79),
    ("groq", "llama-3.1-70b"): (0.59, 0.79),
    ("groq", "llama-3.1-8b"): (0.05, 0.08),
    ("groq", "mixtral-8x7b"): (0.24, 0.24),
    # OpenAI
    ("openai", "gpt-4o-mini"): (0.15, 0.60),
    ("openai", "gpt-4o"): (2.50, 10.00),
    ("openai", "o3-mini"): (1.10, 4.40),
    # Anthropic
    ("anthropic", "claude-haiku"): (0.25, 1.25),
    ("anthropic", "claude-sonnet"): (3.00, 15.00),
    # Google
    ("google", "gemini-2.0-flash"): (0.0, 0.0),  # free tier
    ("google", "gemini-1.5-flash"): (0.075, 0.30),
    # Ollama — fully local, no cost
    ("ollama", ""): (0.0, 0.0),
}


def _lookup_price(provider: str, model: str) -> tuple[float, float]:
    """Return (input_price, output_price) per 1M tokens for a given model."""
    provider = provider.lower()
    model = model.lower()
    for (p, m_prefix), prices in _PRICE_TABLE.items():
        if p == provider and model.startswith(m_prefix):
            return prices
    return (0.0, 0.0)  # unknown provider/model: assume free (e.g. ollama)


def _count_tokens(text: str) -> int:
    """Approximate token count using cl100k_base (GPT-4 tokeniser)."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback: rough word-based approximation
        return int(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# Per-query record
# ---------------------------------------------------------------------------


@dataclass
class QueryRecord:
    """Telemetry for a single RAG query."""

    strategy: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ---------------------------------------------------------------------------
# Tracker — accumulates records, computes aggregates
# ---------------------------------------------------------------------------


@dataclass
class TelemetryTracker:
    """Accumulates QueryRecords for a single strategy run."""

    strategy: str
    records: list[QueryRecord] = field(default_factory=list)

    def add(self, record: QueryRecord) -> None:
        self.records.append(record)

    def summary(self) -> dict:
        if not self.records:
            return {}
        latencies = [r.latency_ms for r in self.records]
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        total_cost = sum(r.cost_usd for r in self.records)
        total_tokens = sum(r.total_tokens for r in self.records)

        return {
            "strategy": self.strategy,
            "n_queries": n,
            "latency_p50_ms": round(latencies_sorted[n // 2], 1),
            "latency_p95_ms": round(latencies_sorted[int(n * 0.95)], 1),
            "latency_mean_ms": round(sum(latencies) / n, 1),
            "total_cost_usd": round(total_cost, 4),
            "cost_per_1k_queries_usd": round(total_cost / n * 1000, 4),
            "total_tokens": total_tokens,
        }


# ---------------------------------------------------------------------------
# Context manager for timing a block + estimating cost
# ---------------------------------------------------------------------------


@contextmanager
def timed_llm_call(
    tracker: TelemetryTracker,
    prompt: str,
    provider: str,
    model: str,
) -> Generator[QueryRecord, None, None]:
    """
    Context manager that times a block and estimates cost from token counts.

    Usage:
        with timed_llm_call(tracker, prompt_text, "groq", "llama-4-scout...") as record:
            response = llm.invoke(prompt)
            record.completion_tokens = count_tokens(response.content)
    """
    record = QueryRecord(strategy=tracker.strategy)
    record.prompt_tokens = _count_tokens(prompt)

    t0 = time.perf_counter()
    yield record
    record.latency_ms = (time.perf_counter() - t0) * 1000

    input_price, output_price = _lookup_price(provider, model)
    record.cost_usd = (
        record.prompt_tokens * input_price / 1_000_000
        + record.completion_tokens * output_price / 1_000_000
    )

    tracker.add(record)


def count_tokens(text: str) -> int:
    """Public helper — count tokens in a string."""
    return _count_tokens(text)
