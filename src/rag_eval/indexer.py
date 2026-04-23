"""
FAISS + BM25 index building and loading for rag-eval-harness.

Artifacts written to data/index/:
  faiss.index      Binary FAISS flat inner-product index (L2-normalised = cosine)
  chunks.json      All chunks with metadata — source of truth for BM25 + retrieval
  index_meta.json  Config snapshot used at build time (for cache invalidation)

BM25 is rebuilt from chunks.json on each load (rank-bm25 objects
are not serialisable, and rebuilding from 10-15K chunks is instantaneous).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from rag_eval.chunker import Chunk
from rag_eval.config import Config

console = Console()

INDEX_DIR = Path("data/index")
FAISS_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
META_PATH = INDEX_DIR / "index_meta.json"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def index_exists() -> bool:
    """Return True if a built index is present on disk."""
    return FAISS_PATH.exists() and CHUNKS_PATH.exists() and META_PATH.exists()


def build_index(chunks: list[Chunk], cfg: Config) -> None:
    """
    Embed all chunks, build a FAISS flat index, and persist to disk.

    Args:
        chunks: Output of chunker.chunk_corpus().
        cfg:    Full Config — used for embeddings provider and metadata.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    from rag_eval.providers.embeddings import get_embeddings

    embed_model = get_embeddings(cfg.embeddings)
    texts = [c["text"] for c in chunks]

    # Embed in batches with a progress bar
    console.print(
        f"[cyan]Embedding {len(texts)} chunks[/] "
        f"with [bold]{cfg.embeddings.provider}/{cfg.embeddings.model}[/] "
        f"(batch_size={cfg.embeddings.batch_size})"
    )

    batch_size = cfg.embeddings.batch_size
    all_embeddings: list[list[float]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding chunks...", total=len(texts))

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            batch_embeddings = embed_model.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            progress.advance(task, len(batch))

    # Stack into numpy array and L2-normalise (cosine via inner product)
    vectors = np.array(all_embeddings, dtype=np.float32)
    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on unit vectors = cosine similarity
    index.add(vectors)

    # Persist
    faiss.write_index(index, str(FAISS_PATH))
    console.print(f"  Saved FAISS index ({index.ntotal} vectors, dim={dim})")

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    console.print(f"  Saved chunks manifest ({len(chunks)} chunks)")

    meta = {
        "num_chunks": len(chunks),
        "dimension": dim,
        "embeddings_provider": cfg.embeddings.provider,
        "embeddings_model": cfg.embeddings.model,
        "chunk_size": cfg.retrieval.chunk_size,
        "chunk_overlap": cfg.retrieval.chunk_overlap,
        "dataset": cfg.dataset.name,
        "sample_size": cfg.dataset.sample_size,
        "seed": cfg.dataset.seed,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    console.print("  Saved index metadata")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


class RAGIndex:
    """
    Loaded index ready for dense and BM25 retrieval.

    Instantiate once per benchmark run and reuse across strategies.

    Usage:
        index = RAGIndex.load()
        dense_hits  = index.dense_search(query_embedding, top_k=20)
        bm25_hits   = index.bm25_search(query, top_k=20)
    """

    def __init__(
        self,
        faiss_index: faiss.Index,
        chunks: list[Chunk],
        bm25: BM25Okapi,
        meta: dict,
    ) -> None:
        self._faiss = faiss_index
        self._chunks = chunks
        self._bm25 = bm25
        self.meta = meta

    @classmethod
    def load(cls, index_dir: str | Path = INDEX_DIR) -> RAGIndex:
        """
        Load index from disk.

        Raises:
            FileNotFoundError: If index has not been built yet.
        """
        index_dir = Path(index_dir)
        faiss_path = index_dir / "faiss.index"
        chunks_path = index_dir / "chunks.json"
        meta_path = index_dir / "index_meta.json"

        for p in (faiss_path, chunks_path, meta_path):
            if not p.exists():
                raise FileNotFoundError(f"Index file not found: {p}\nRun: python -m rag_eval index")

        faiss_index = faiss.read_index(str(faiss_path))

        with open(chunks_path, encoding="utf-8") as f:
            chunks: list[Chunk] = json.load(f)

        with open(meta_path) as f:
            meta = json.load(f)

        # Rebuild BM25 in memory (instantaneous for 10-15K chunks)
        tokenized = [c["text"].lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized)

        console.print(
            f"[dim]Index loaded:[/] "
            f"{faiss_index.ntotal} vectors, "
            f"dim={faiss_index.d}, "
            f"built {meta.get('built_at', 'unknown')[:10]}"
        )

        return cls(faiss_index=faiss_index, chunks=chunks, bm25=bm25, meta=meta)

    # -----------------------------------------------------------------------
    # Retrieval primitives — used by strategies
    # -----------------------------------------------------------------------

    def dense_search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict]:
        """
        Return top-k chunks by cosine similarity.

        Args:
            query_embedding: Already-embedded query vector (un-normalised OK).
            top_k: Number of results to return.

        Returns:
            List of chunk dicts enriched with {"score": float, "retrieval": "dense"}.
        """
        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        scores, indices = self._faiss.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = dict(self._chunks[idx])
            chunk["score"] = float(score)
            chunk["retrieval"] = "dense"
            results.append(chunk)
        return results

    def bm25_search(self, query: str, top_k: int) -> list[dict]:
        """
        Return top-k chunks by BM25 score.

        Args:
            query: Raw query string (tokenised internally).
            top_k: Number of results to return.

        Returns:
            List of chunk dicts enriched with {"score": float, "retrieval": "bm25"}.
        """
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = dict(self._chunks[idx])
            chunk["score"] = float(scores[idx])
            chunk["retrieval"] = "bm25"
            results.append(chunk)
        return results

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)
