"""
Phase 1 tests — dataset loading, chunking, and index build/load.

All tests use a small in-memory fixture so no network calls or
API keys are required. Safe to run in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared fixture: minimal fake dataset
# ---------------------------------------------------------------------------

FAKE_CORPUS = [
    {
        "passage_id": "Alpha Article",
        "text": (
            "Alpha is the first letter of the Greek alphabet. "
            "It is used in mathematics and science to denote various quantities. "
            "Alpha particles are emitted during radioactive decay."
        ),
    },
    {
        "passage_id": "Beta Article",
        "text": (
            "Beta is the second letter of the Greek alphabet. "
            "Beta testing is a phase of software development. "
            "Beta waves are observed in brain activity."
        ),
    },
    {
        "passage_id": "Gamma Article",
        "text": (
            "Gamma rays are a form of electromagnetic radiation. "
            "They have the highest frequency in the electromagnetic spectrum. "
            "Gamma radiation is used in cancer treatment."
        ),
    },
]

FAKE_QA_PAIRS = [
    {
        "id": "q1",
        "question": "What is the first letter of the Greek alphabet?",
        "reference_answer": "Alpha",
        "reference_contexts": [FAKE_CORPUS[0]["text"]],
    },
    {
        "id": "q2",
        "question": "What type of radiation is used in cancer treatment?",
        "reference_answer": "Gamma rays",
        "reference_contexts": [FAKE_CORPUS[2]["text"]],
    },
]


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------


class TestChunker:
    def test_basic_chunking(self):
        from rag_eval.chunker import chunk_corpus
        from rag_eval.config import RetrievalConfig

        cfg = RetrievalConfig(chunk_size=50, chunk_overlap=10)
        chunks = chunk_corpus(FAKE_CORPUS, cfg)

        assert len(chunks) > 0
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "passage_id" in chunk
            assert "chunk_index" in chunk
            assert chunk["text"].strip() != ""

    def test_chunk_ids_are_unique(self):
        from rag_eval.chunker import chunk_corpus
        from rag_eval.config import RetrievalConfig

        cfg = RetrievalConfig(chunk_size=50, chunk_overlap=10)
        chunks = chunk_corpus(FAKE_CORPUS, cfg)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_passage_id_preserved(self):
        from rag_eval.chunker import chunk_corpus
        from rag_eval.config import RetrievalConfig

        cfg = RetrievalConfig(chunk_size=50, chunk_overlap=10)
        chunks = chunk_corpus(FAKE_CORPUS, cfg)
        passage_ids = {c["passage_id"] for c in chunks}
        corpus_ids = {p["passage_id"] for p in FAKE_CORPUS}
        assert passage_ids.issubset(corpus_ids)

    def test_large_chunk_size_gives_fewer_chunks(self):
        from rag_eval.chunker import chunk_corpus
        from rag_eval.config import RetrievalConfig

        small_cfg = RetrievalConfig(chunk_size=20, chunk_overlap=5)
        large_cfg = RetrievalConfig(chunk_size=512, chunk_overlap=50)

        small_chunks = chunk_corpus(FAKE_CORPUS, small_cfg)
        large_chunks = chunk_corpus(FAKE_CORPUS, large_cfg)

        assert len(small_chunks) >= len(large_chunks)

    def test_chunk_stats(self):
        from rag_eval.chunker import chunk_corpus, chunk_stats
        from rag_eval.config import RetrievalConfig

        cfg = RetrievalConfig(chunk_size=512, chunk_overlap=50)
        chunks = chunk_corpus(FAKE_CORPUS, cfg)
        stats = chunk_stats(chunks, cfg)

        assert stats["num_chunks"] == len(chunks)
        assert stats["avg_tokens"] > 0
        assert stats["min_tokens"] <= stats["avg_tokens"] <= stats["max_tokens"]


# ---------------------------------------------------------------------------
# Index build + load tests (uses a temp dir, no GPU/API needed)
# ---------------------------------------------------------------------------


class TestIndexer:
    @pytest.fixture
    def fake_chunks(self):
        from rag_eval.chunker import chunk_corpus
        from rag_eval.config import RetrievalConfig

        return chunk_corpus(FAKE_CORPUS, RetrievalConfig(chunk_size=512, chunk_overlap=50))

    @pytest.fixture
    def fake_config(self):
        """Config using local BGE embeddings (no API key needed)."""
        from rag_eval.config import Config

        return Config()  # defaults: local BGE-large embeddings

    def test_index_builds_and_loads(self, fake_chunks, fake_config, tmp_path):
        # Patch INDEX_DIR to use tmp_path so tests don't pollute data/
        import rag_eval.indexer as indexer_mod
        from rag_eval.indexer import RAGIndex, build_index

        original_dir = indexer_mod.INDEX_DIR
        indexer_mod.INDEX_DIR = tmp_path
        indexer_mod.FAISS_PATH = tmp_path / "faiss.index"
        indexer_mod.CHUNKS_PATH = tmp_path / "chunks.json"
        indexer_mod.META_PATH = tmp_path / "index_meta.json"

        try:
            build_index(fake_chunks, fake_config)

            # All artifacts should exist
            assert (tmp_path / "faiss.index").exists()
            assert (tmp_path / "chunks.json").exists()
            assert (tmp_path / "index_meta.json").exists()

            # Load and verify
            rag_index = RAGIndex.load(tmp_path)
            assert rag_index.num_chunks == len(fake_chunks)
            assert rag_index.meta["embeddings_provider"] == "local"

        finally:
            indexer_mod.INDEX_DIR = original_dir
            indexer_mod.FAISS_PATH = original_dir / "faiss.index"
            indexer_mod.CHUNKS_PATH = original_dir / "chunks.json"
            indexer_mod.META_PATH = original_dir / "index_meta.json"

    def test_dense_search_returns_top_k(self, fake_chunks, fake_config, tmp_path):
        import rag_eval.indexer as indexer_mod
        from rag_eval.indexer import RAGIndex, build_index

        original_dir = indexer_mod.INDEX_DIR
        indexer_mod.INDEX_DIR = tmp_path
        indexer_mod.FAISS_PATH = tmp_path / "faiss.index"
        indexer_mod.CHUNKS_PATH = tmp_path / "chunks.json"
        indexer_mod.META_PATH = tmp_path / "index_meta.json"

        try:
            build_index(fake_chunks, fake_config)
            rag_index = RAGIndex.load(tmp_path)

            # Use a random query embedding of correct dimension
            dim = rag_index._faiss.d
            query_vec = np.random.rand(dim).tolist()

            top_k = min(2, rag_index.num_chunks)
            results = rag_index.dense_search(query_vec, top_k=top_k)

            assert len(results) == top_k
            assert all("score" in r for r in results)
            assert all("text" in r for r in results)
            assert all(r["retrieval"] == "dense" for r in results)

        finally:
            indexer_mod.INDEX_DIR = original_dir
            indexer_mod.FAISS_PATH = original_dir / "faiss.index"
            indexer_mod.CHUNKS_PATH = original_dir / "chunks.json"
            indexer_mod.META_PATH = original_dir / "index_meta.json"

    def test_bm25_search_returns_relevant_results(self, fake_chunks, fake_config, tmp_path):
        import rag_eval.indexer as indexer_mod
        from rag_eval.indexer import RAGIndex, build_index

        original_dir = indexer_mod.INDEX_DIR
        indexer_mod.INDEX_DIR = tmp_path
        indexer_mod.FAISS_PATH = tmp_path / "faiss.index"
        indexer_mod.CHUNKS_PATH = tmp_path / "chunks.json"
        indexer_mod.META_PATH = tmp_path / "index_meta.json"

        try:
            build_index(fake_chunks, fake_config)
            rag_index = RAGIndex.load(tmp_path)

            results = rag_index.bm25_search("gamma radiation cancer treatment", top_k=2)

            # Should find something related to Gamma Article
            assert len(results) > 0
            assert all("score" in r for r in results)
            assert all(r["retrieval"] == "bm25" for r in results)
            # Top result should be from the Gamma article
            assert "Gamma" in results[0]["passage_id"]

        finally:
            indexer_mod.INDEX_DIR = original_dir
            indexer_mod.FAISS_PATH = original_dir / "faiss.index"
            indexer_mod.CHUNKS_PATH = original_dir / "chunks.json"
            indexer_mod.META_PATH = original_dir / "index_meta.json"


# ---------------------------------------------------------------------------
# Dataset structure tests (no network — validates TypedDict keys)
# ---------------------------------------------------------------------------


class TestDatasetStructure:
    def test_qa_pair_has_required_keys(self):
        for qa in FAKE_QA_PAIRS:
            assert "id" in qa
            assert "question" in qa
            assert "reference_answer" in qa
            assert "reference_contexts" in qa
            assert isinstance(qa["reference_contexts"], list)

    def test_corpus_passage_has_required_keys(self):
        for passage in FAKE_CORPUS:
            assert "passage_id" in passage
            assert "text" in passage
            assert passage["text"].strip() != ""
