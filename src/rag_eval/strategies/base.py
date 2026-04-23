"""
Abstract base class for all RAG strategies.

Every strategy (naive, hybrid, rerank, hyde, multi_query) subclasses
BaseStrategy and implements a single method: answer().

The shared infrastructure (LLM, embeddings, prompt template, cost tracking)
lives here so strategies stay focused on their retrieval logic only.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from rag_eval.config import Config
from rag_eval.indexer import RAGIndex
from rag_eval.providers.embeddings import get_embeddings
from rag_eval.providers.llm import get_llm
from rag_eval.telemetry import TelemetryTracker, count_tokens, _lookup_price

# ---------------------------------------------------------------------------
# Shared prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise question-answering assistant.
Answer the question using ONLY the information in the provided context passages.
If the context does not contain enough information to answer, respond with exactly: "I don't know."
Be concise. Do not add information not present in the context.\
"""


def build_rag_prompt(question: str, contexts: list[str]) -> str:
    """Build the full prompt string for token counting purposes."""
    context_block = "\n\n".join(
        f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)
    )
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}"
    )


def build_messages(question: str, contexts: list[str]) -> list:
    """Build LangChain message list for the RAG prompt."""
    context_block = "\n\n".join(
        f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)
    )
    return [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Context:\n{context_block}\n\nQuestion: {question}"
        ),
    ]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RAGResult:
    """Output of BaseStrategy.answer() — everything needed for RAGAS eval."""
    question: str
    answer: str                    # generated answer
    contexts: list[str]            # retrieved chunk texts passed to the LLM
    latency_ms: float
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base strategy
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """
    Abstract base for all RAG strategies.

    Subclasses must set `name` (class attribute) and implement `answer()`.
    The LLM and embeddings are instantiated once in __init__ and reused.
    """

    name: str = "base"   # override in each subclass

    def __init__(self, cfg: Config, index: RAGIndex) -> None:
        self.cfg = cfg
        self.index = index
        self.llm = get_llm(cfg.generator)
        self.embeddings = get_embeddings(cfg.embeddings)
        self.tracker = TelemetryTracker(strategy=self.name)

    @abstractmethod
    def answer(self, question: str) -> RAGResult:
        """
        Retrieve relevant chunks for `question` and generate an answer.

        Args:
            question: The question to answer.

        Returns:
            RAGResult with answer, contexts, and telemetry.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers available to all strategies
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> list[float]:
        """Embed a single text string (query or hypothetical document)."""
        return self.embeddings.embed_query(text)

    def raw_llm_call(self, prompt: str) -> tuple[str, int, int, float]:
        """
        Call the LLM with a plain-text prompt (no RAG context).

        Used by strategies that need an extra LLM call before retrieval,
        such as HyDE (hypothetical document generation) or multi-query
        expansion. Cost is estimated from token counts.

        Args:
            prompt: Plain text prompt to send to the LLM.

        Returns:
            (response_text, prompt_tokens, completion_tokens, cost_usd)
        """
        from langchain_core.messages import HumanMessage

        prompt_tokens = count_tokens(prompt)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        text = response.content.strip()
        completion_tokens = count_tokens(text)

        input_price, output_price = _lookup_price(
            self.cfg.generator.provider,
            self.cfg.generator.model,
        )
        cost_usd = (
            prompt_tokens * input_price / 1_000_000
            + completion_tokens * output_price / 1_000_000
        )
        return text, prompt_tokens, completion_tokens, cost_usd

    def generate(
        self,
        question: str,
        contexts: list[str],
    ) -> tuple[str, int, int, float]:
        """
        Call the LLM with question + contexts, measure latency + cost.

        Returns:
            (answer_text, prompt_tokens, completion_tokens, latency_ms)
        """
        messages = build_messages(question, contexts)
        prompt_str = build_rag_prompt(question, contexts)   # for token count

        prompt_tokens = count_tokens(prompt_str)
        t0 = time.perf_counter()
        response = self.llm.invoke(messages)
        latency_ms = (time.perf_counter() - t0) * 1000

        answer = response.content.strip()
        completion_tokens = count_tokens(answer)

        input_price, output_price = _lookup_price(
            self.cfg.generator.provider,
            self.cfg.generator.model,
        )
        cost_usd = (
            prompt_tokens * input_price / 1_000_000
            + completion_tokens * output_price / 1_000_000
        )

        return answer, prompt_tokens, completion_tokens, latency_ms, cost_usd
