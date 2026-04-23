"""
Hybrid RAG strategy — dense + BM25 with Reciprocal Rank Fusion.

Pipeline:
  1. Dense search:  embed question → FAISS cosine similarity (top_k × 2 candidates)
  2. Sparse search: BM25 over chunk texts (top_k × 2 candidates)
  3. RRF fusion:    merge the two ranked lists with Reciprocal Rank Fusion
  4. Generate:      pass top-k fused chunks + question to the LLM

Why hybrid beats naive:
  Dense vectors are great for semantic / paraphrase matching but fail on
  keyword-heavy queries (rare nouns, proper names, code snippets).
  BM25 is the opposite: exact-match powerhouse, poor on paraphrases.
  RRF gives each system a vote proportional to its rank confidence
  without needing to tune incompatible score scales.

Reference: Cormack et al. (2009) "Reciprocal Rank Fusion outperforms
  Condorcet and individual Rank Learning Methods."
"""
from __future__ import annotations

from rag_eval.strategies.base import BaseStrategy, RAGResult


def _rrf_merge(
    dense_hits: list[dict],
    bm25_hits: list[dict],
    top_k: int,
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion of two ranked result lists.

    Each document gets a score of 1/(k + rank) from each list it appears in.
    Documents appearing in both lists receive contributions from both.

    Args:
        dense_hits: Ranked hits from FAISS dense search.
        bm25_hits:  Ranked hits from BM25 sparse search.
        top_k:      Number of results to return after fusion.
        k:          RRF constant (60 is the standard choice).

    Returns:
        Merged and re-ranked list of hit dicts, length ≤ top_k.
    """
    scores: dict[str, float] = {}
    all_chunks: dict[str, dict] = {}

    for rank, hit in enumerate(dense_hits):
        cid = hit["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        all_chunks[cid] = hit

    for rank, hit in enumerate(bm25_hits):
        cid = hit["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        all_chunks.setdefault(cid, hit)   # keep dense hit if already seen

    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    return [all_chunks[cid] for cid in ranked_ids[:top_k]]


class HybridRAG(BaseStrategy):
    """
    Hybrid BM25 + dense retrieval with RRF fusion.

    Strengths  : covers both keyword-heavy and semantic queries; generally
                 outperforms either component alone on factoid QA benchmarks
    Weaknesses : slightly higher latency (two retrieval calls); RRF weight is
                 fixed (not learned); BM25 ignores token morphology
    """

    name = "hybrid"

    def answer(self, question: str) -> RAGResult:
        top_k = self.cfg.retrieval.top_k
        # Fetch 2× candidates so RRF has room to work
        candidate_k = top_k * 2

        # Step 1: dense retrieval
        query_embedding = self.embed_query(question)
        dense_hits = self.index.dense_search(query_embedding, top_k=candidate_k)

        # Step 2: sparse BM25 retrieval
        bm25_hits = self.index.bm25_search(question, top_k=candidate_k)

        # Step 3: RRF fusion
        fused_hits = _rrf_merge(dense_hits, bm25_hits, top_k=top_k)
        contexts = [h["text"] for h in fused_hits]

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
                "num_dense_hits": len(dense_hits),
                "num_bm25_hits": len(bm25_hits),
                "num_fused_hits": len(fused_hits),
            },
        )
