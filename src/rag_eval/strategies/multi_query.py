"""
Multi-Query RAG strategy — query expansion with deduplication.

Pipeline:
  1. Ask the LLM to rewrite the original question into N alternative phrasings
     (default 3) that might surface different relevant passages.
  2. Embed each variant (plus the original) and run FAISS search for each.
  3. Merge the candidate pools, deduplicate by chunk_id, and take the top_k
     unique chunks (ordered by first appearance to preserve relevance ranking).
  4. Pass the merged context + original question to the LLM.

Why multi-query helps:
  A single embedding captures one semantic direction. A question like
  "What are the consequences of the French Revolution?" can be approached
  from at least three angles: political, social, economic.  Generating
  multiple phrasings increases the chance that each sub-topic's best
  passages appear in at least one search, without needing a smarter index.

Tradeoffs:
  + Higher recall on multi-faceted questions; no index changes required
  - One extra LLM call per query; total retrieval latency scales with N+1
  - If the LLM produces near-duplicate variants, diversity gain is low
  - Context window fills up faster with merged results (mitigated by dedup)

Reference: Inspired by LangChain's MultiQueryRetriever and the
  "Query Expansion" approach from Ma et al. (2023).
"""
from __future__ import annotations

import time

from rag_eval.strategies.base import BaseStrategy, RAGResult

# Number of alternative query phrasings to generate
_N_VARIANTS = 3

_EXPAND_PROMPT = """\
You are a search query expert.  Given the original question below, write \
{n} alternative phrasings that could help retrieve different relevant \
passages from a document corpus.  Write one query per line.  Output ONLY \
the queries — no numbering, no explanation.

Original question: {question}

Alternative queries:"""


def _parse_query_variants(text: str, original: str, max_variants: int) -> list[str]:
    """
    Parse LLM output into a deduplicated list of queries.

    Always includes the original question first so we never lose the
    primary search signal even if the LLM produces garbage.
    """
    seen: set[str] = {original.strip().lower()}
    queries = [original]

    for line in text.splitlines():
        variant = line.strip().lstrip("-•* 1234567890.)").strip()
        if not variant:
            continue
        key = variant.lower()
        if key not in seen:
            seen.add(key)
            queries.append(variant)
        if len(queries) >= max_variants + 1:
            break

    return queries


class MultiQueryRAG(BaseStrategy):
    """
    Multi-query expansion with deduplicated retrieval.

    Strengths  : higher recall on multi-faceted questions; leverages the
                 LLM's paraphrase ability to bridge vocabulary gaps
    Weaknesses : N+1 embedding calls; one extra LLM call; risk of diluted
                 context if expanded queries pull in off-topic passages
    """

    name = "multi_query"

    def answer(self, question: str) -> RAGResult:
        t0 = time.perf_counter()
        top_k = self.cfg.retrieval.top_k

        # Step 1: generate query variants
        expand_prompt = _EXPAND_PROMPT.format(n=_N_VARIANTS, question=question)
        variants_text, exp_pt, exp_ct, exp_cost = self.raw_llm_call(expand_prompt)
        queries = _parse_query_variants(variants_text, question, max_variants=_N_VARIANTS)

        # Step 2: retrieve for each query, dedup by chunk_id
        seen_ids: set[str] = set()
        merged_hits: list[dict] = []

        for q in queries:
            q_embedding = self.embed_query(q)
            hits = self.index.dense_search(q_embedding, top_k=top_k)
            for hit in hits:
                cid = hit["chunk_id"]
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    merged_hits.append(hit)

        # Step 3: take top_k unique chunks (first-appearance order)
        contexts = [h["text"] for h in merged_hits[:top_k]]

        # Step 4: generate answer from merged context
        answer, prompt_tokens, completion_tokens, _, rag_cost = self.generate(
            question, contexts
        )

        total_latency_ms = (time.perf_counter() - t0) * 1000
        total_cost_usd = exp_cost + rag_cost

        return RAGResult(
            question=question,
            answer=answer,
            contexts=contexts,
            latency_ms=total_latency_ms,
            cost_usd=total_cost_usd,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            metadata={
                "strategy": self.name,
                "num_queries": len(queries),
                "queries": queries,
                "num_unique_chunks": len(merged_hits),
            },
        )
