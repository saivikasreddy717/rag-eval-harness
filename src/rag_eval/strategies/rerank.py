"""
Rerank RAG strategy — dense retrieval followed by Cohere cross-encoder reranking.

Pipeline:
  1. Embed the query and run FAISS cosine search for top_k_rerank candidates
     (default 20 — a much wider net than the 5 we ultimately return).
  2. Send all candidate texts + question to Cohere's rerank endpoint.
     The cross-encoder reads every (question, passage) pair together, giving
     much richer relevance signal than the inner-product embedding similarity.
  3. Keep the top_k passages with the highest rerank scores.
  4. Pass those passages + question to the generator LLM.

Why reranking helps:
  Bi-encoder embeddings (FAISS) score query and passage independently —
  they never see each other. A cross-encoder reads both simultaneously and
  captures fine-grained semantic interactions that the bi-encoder misses.
  On most open-domain QA benchmarks this stage gives the largest single
  accuracy boost at a modest extra cost.

Requirements:
  COHERE_API_KEY environment variable must be set.
  Free tier: 10M tokens/month — enough for full HotpotQA evaluation.
"""
from __future__ import annotations

import os

from rag_eval.config import Config
from rag_eval.indexer import RAGIndex
from rag_eval.providers.reranker import get_reranker
from rag_eval.strategies.base import BaseStrategy, RAGResult


class RerankRAG(BaseStrategy):
    """
    Dense retrieval + Cohere cross-encoder reranking.

    Strengths  : highest retrieval precision of all strategies; cross-encoder
                 directly models query–passage interaction
    Weaknesses : extra network latency for the rerank API call; requires
                 COHERE_API_KEY; cost scales with top_k_rerank × query count
    """

    name = "rerank"

    def __init__(self, cfg: Config, index: RAGIndex) -> None:
        super().__init__(cfg, index)
        # Build the reranker at construction time so missing keys fail fast
        self.reranker = get_reranker(cfg.reranker)

    def answer(self, question: str) -> RAGResult:
        top_k = self.cfg.retrieval.top_k
        candidate_k = self.cfg.retrieval.top_k_rerank   # wider net (default 20)

        # Step 1: dense retrieval — more candidates than we'll return
        query_embedding = self.embed_query(question)
        candidates = self.index.dense_search(query_embedding, top_k=candidate_k)

        if not candidates:
            # Edge case: empty index (shouldn't happen in production)
            contexts = []
        else:
            # Step 2: cross-encoder reranking
            docs = [c["text"] for c in candidates]
            ranked_pairs = self.reranker(question, docs, top_n=min(top_k, len(docs)))

            # Step 3: build context list in rerank order
            contexts = [docs[idx] for idx, _score in ranked_pairs]

        # Step 4: generate answer
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
            metadata={
                "strategy": self.name,
                "num_candidates": len(candidates),
                "num_reranked": len(contexts),
            },
        )
