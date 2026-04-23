"""
Dataset loading and caching for rag-eval-harness.

Currently supports HotpotQA (distractor setting) — multi-hop QA with
ground-truth supporting passages, ideal for stressing retrieval.

Output format
-------------
{
    "qa_pairs": [
        {
            "id": "5a7a06935542990198eaf050",
            "question": "Which magazine was started first?",
            "reference_answer": "Arthur's Magazine",
            "reference_contexts": ["passage text A", "passage text B"],
        },
        ...
    ],
    "corpus": [
        {"passage_id": "Arthur's Magazine", "text": "Arthur's Magazine ..."},
        ...
    ]
}

qa_pairs  : what the runner iterates — question + ground-truth for RAGAS
corpus    : unique passages across all QA pairs — what gets indexed
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TypedDict

from rich.console import Console

from rag_eval.config import DatasetConfig

console = Console()

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------


class QAPair(TypedDict):
    id: str
    question: str
    reference_answer: str
    reference_contexts: list[str]  # ground-truth passages (for RAGAS context_recall)


class Passage(TypedDict):
    passage_id: str  # unique key (passage title in HotpotQA)
    text: str


class BenchmarkData(TypedDict):
    qa_pairs: list[QAPair]
    corpus: list[Passage]


# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
QA_PAIRS_CACHE = DATA_DIR / "qa_pairs.json"
CORPUS_CACHE = DATA_DIR / "corpus.json"


def _cache_exists() -> bool:
    return QA_PAIRS_CACHE.exists() and CORPUS_CACHE.exists()


def _load_cache() -> BenchmarkData:
    with open(QA_PAIRS_CACHE) as f:
        qa_pairs = json.load(f)
    with open(CORPUS_CACHE) as f:
        corpus = json.load(f)
    return {"qa_pairs": qa_pairs, "corpus": corpus}


def _save_cache(data: BenchmarkData) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(QA_PAIRS_CACHE, "w") as f:
        json.dump(data["qa_pairs"], f, indent=2)
    with open(CORPUS_CACHE, "w") as f:
        json.dump(data["corpus"], f, indent=2)


# ---------------------------------------------------------------------------
# HotpotQA loader
# ---------------------------------------------------------------------------


def load_hotpotqa(config: DatasetConfig, force: bool = False) -> BenchmarkData:
    """
    Load and sample HotpotQA (distractor setting) from HuggingFace Datasets.

    Results are cached to data/qa_pairs.json and data/corpus.json so
    subsequent runs skip the download.

    Args:
        config: DatasetConfig specifying split, sample_size, and seed.
        force:  Re-download and re-sample even if cache exists.

    Returns:
        BenchmarkData with qa_pairs and deduplicated corpus.
    """
    if _cache_exists() and not force:
        data = _load_cache()
        console.print(
            f"[dim]Loaded from cache:[/] "
            f"{len(data['qa_pairs'])} QA pairs, "
            f"{len(data['corpus'])} passages"
        )
        return data

    console.print(f"[cyan]Downloading HotpotQA ({config.split} split)...[/]")

    # Lazy import — datasets is a heavy dep, only load when needed
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("datasets is not installed. Run: uv sync")

    ds = hf_load("hotpot_qa", "distractor", split=config.split, trust_remote_code=True)

    # Reproducible sampling
    rng = random.Random(config.seed)
    total = len(ds)
    sample_size = min(config.sample_size, total)
    indices = sorted(rng.sample(range(total), sample_size))
    sampled = ds.select(indices)

    console.print(f"[green]Sampled[/] {sample_size}/{total} questions (seed={config.seed})")

    qa_pairs: list[QAPair] = []
    corpus_map: dict[str, str] = {}  # passage_id -> text (deduplicates across questions)

    for item in sampled:
        # Build passage map for this question (10 passages: 2 relevant + 8 distractors)
        passage_texts: dict[str, str] = {}
        for title, sentences in zip(
            item["context"]["title"],
            item["context"]["sentences"],
        ):
            passage_texts[title] = " ".join(sentences)
            corpus_map[title] = passage_texts[title]

        # Ground-truth contexts = the supporting passages (2 per question typically)
        supporting_titles = set(item["supporting_facts"]["title"])
        reference_contexts = [passage_texts[t] for t in supporting_titles if t in passage_texts]

        qa_pairs.append(
            QAPair(
                id=item["id"],
                question=item["question"],
                reference_answer=item["answer"],
                reference_contexts=reference_contexts,
            )
        )

    corpus: list[Passage] = [Passage(passage_id=pid, text=text) for pid, text in corpus_map.items()]

    data = BenchmarkData(qa_pairs=qa_pairs, corpus=corpus)
    _save_cache(data)

    console.print(
        f"[green]Built corpus:[/] {len(qa_pairs)} QA pairs, {len(corpus)} unique passages"
    )
    return data
