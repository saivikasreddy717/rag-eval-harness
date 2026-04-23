"""
HyDE RAG strategy — Hypothetical Document Embeddings.

Pipeline:
  1. Ask the LLM to write a *hypothetical* passage that would answer the
     question (without any retrieved context — a zero-shot generation).
  2. Embed the hypothetical passage instead of the original question.
  3. Use that embedding for FAISS cosine similarity search.
  4. Pass the *real* retrieved chunks + original question to the LLM to
     produce the final answer.

Why HyDE can beat naive:
  Short questions ("What causes thunder?") and long answer-like passages
  live in very different parts of embedding space. By generating a
  document that looks like an answer, we move the query vector much
  closer to real answer passages in the corpus. This tends to recover
  longer, more answer-like chunks that keyword or short-query embeddings
  would miss.

Tradeoffs:
  + Better recall for factoid questions where the question form ≠ answer form
  - One extra LLM call per query (adds ~200–400 ms and a few cents/K queries)
  - If the hypothetical document hallucinates a wrong direction the
    retrieval can degrade (HyDE is not always better than naive)
  - Does not benefit queries that already use answer-like phrasing

Reference: Gao et al. (2022) "Precise Zero-Shot Dense Retrieval without
  Relevance Labels." https://arxiv.org/abs/2212.10496
"""

from __future__ import annotations

import time

from rag_eval.strategies.base import BaseStrategy, RAGResult

_HYDE_PROMPT = """\
Write a short, factual passage (2-4 sentences) that directly answers the question below.
Write only the passage — no preamble, no "answer:", no citation markers.

Question: {question}

Passage:"""


class HyDERAG(BaseStrategy):
    """
    Hypothetical Document Embedding retrieval.

    Strengths  : strong on factoid QA where question phrasing ≠ answer phrasing;
                 no index changes needed
    Weaknesses : adds one extra LLM call; can backfire on ambiguous questions
    """

    name = "hyde"

    def answer(self, question: str) -> RAGResult:
        t0 = time.perf_counter()

        # Step 1: generate a hypothetical document for the question
        hyp_prompt = _HYDE_PROMPT.format(question=question)
        hyp_doc, hyp_pt, hyp_ct, hyp_cost = self.raw_llm_call(hyp_prompt)

        # Step 2: embed the hypothetical document (not the question)
        hyp_embedding = self.embed_query(hyp_doc)

        # Step 3: dense retrieval using the hypothetical embedding
        hits = self.index.dense_search(hyp_embedding, top_k=self.cfg.retrieval.top_k)
        contexts = [h["text"] for h in hits]

        # Step 4: generate the real answer from retrieved contexts
        answer, prompt_tokens, completion_tokens, _, rag_cost = self.generate(question, contexts)

        total_latency_ms = (time.perf_counter() - t0) * 1000
        total_cost_usd = hyp_cost + rag_cost

        return RAGResult(
            question=question,
            answer=answer,
            contexts=contexts,
            latency_ms=total_latency_ms,
            cost_usd=total_cost_usd,
            # Report tokens for the final RAG call (standard across strategies)
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            metadata={
                "strategy": self.name,
                "hypothetical_doc": hyp_doc[:200],  # truncated for logging
                "hyp_prompt_tokens": hyp_pt,
                "hyp_completion_tokens": hyp_ct,
                "num_hits": len(hits),
            },
        )
