"""
Naive RAG strategy — dense retrieval only.

Pipeline:
  1. Embed the query with the configured embeddings model
  2. FAISS cosine similarity search, return top-k chunks
  3. Pass chunks + question to the LLM, return the answer

This is the baseline all other strategies are compared against.
It represents the simplest possible RAG implementation.
"""

from __future__ import annotations

import time

from rag_eval.strategies.base import BaseStrategy, RAGResult


class NaiveRAG(BaseStrategy):
    """
    Baseline: dense-only retrieval, no hybrid fusion, no reranking.

    Strengths  : fast, simple, low latency
    Weaknesses : misses keyword-heavy queries that BM25 would catch,
                 no way to recover from poor embedding similarity
    """

    name = "naive"

    def answer(self, question: str) -> RAGResult:
        # Step 1 + 2: embed then retrieve (timed as retrieval phase)
        t_ret = time.perf_counter()
        query_embedding = self.embed_query(question)
        hits = self.index.dense_search(
            query_embedding,
            top_k=self.cfg.retrieval.top_k,
        )
        retrieval_latency_ms = (time.perf_counter() - t_ret) * 1000
        contexts = [h["text"] for h in hits]

        # Step 3: generate
        answer, prompt_tokens, completion_tokens, latency_ms, cost_usd = self.generate(
            question, contexts
        )

        return RAGResult(
            question=question,
            answer=answer,
            contexts=contexts,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            retrieval_latency_ms=retrieval_latency_ms,
            metadata={
                "strategy": self.name,
                "num_hits": len(hits),
                "top_scores": [round(h["score"], 4) for h in hits[:3]],
            },
        )
