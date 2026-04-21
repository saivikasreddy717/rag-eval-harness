"""
Text chunking for rag-eval-harness.

Splits corpus passages into overlapping chunks sized by token count
(not character count) for accurate context window management.

Uses tiktoken cl100k_base (same tokenizer as GPT-4 / text-embedding-3)
as the length function so chunk_size in config maps directly to LLM tokens.
"""
from __future__ import annotations

from typing import TypedDict

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_eval.config import RetrievalConfig
from rag_eval.datasets import Passage


class Chunk(TypedDict):
    chunk_id: str          # globally unique: "{passage_id}__chunk_{n}"
    text: str
    passage_id: str        # which passage this came from
    chunk_index: int       # position within the passage (0-based)


def chunk_corpus(corpus: list[Passage], config: RetrievalConfig) -> list[Chunk]:
    """
    Split all corpus passages into token-bounded overlapping chunks.

    Args:
        corpus: List of passages from load_hotpotqa().
        config: RetrievalConfig with chunk_size and chunk_overlap (in tokens).

    Returns:
        Flat list of Chunk dicts ready for embedding and indexing.
    """
    enc = tiktoken.get_encoding("cl100k_base")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=lambda text: len(enc.encode(text)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []

    for passage in corpus:
        splits = splitter.split_text(passage["text"])

        for i, text in enumerate(splits):
            # Skip empty/whitespace chunks that can appear after splitting
            if not text.strip():
                continue

            chunks.append(
                Chunk(
                    chunk_id=f"{passage['passage_id']}__chunk_{i}",
                    text=text.strip(),
                    passage_id=passage["passage_id"],
                    chunk_index=i,
                )
            )

    return chunks


def chunk_stats(chunks: list[Chunk], config: RetrievalConfig) -> dict:
    """Return summary stats for a set of chunks (useful for logging)."""
    enc = tiktoken.get_encoding("cl100k_base")
    token_counts = [len(enc.encode(c["text"])) for c in chunks]

    return {
        "num_chunks": len(chunks),
        "avg_tokens": round(sum(token_counts) / len(token_counts), 1) if token_counts else 0,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "chunk_size_cfg": config.chunk_size,
        "chunk_overlap_cfg": config.chunk_overlap,
    }
