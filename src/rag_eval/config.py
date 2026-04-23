"""
Configuration system for rag-eval-harness.

All settings live in YAML files under configs/.
Swap providers and models without touching any code.

Supported LLM providers   : groq, openai, anthropic, ollama, google
Supported embed providers : local, openai, cohere, ollama, google
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

VALID_STRATEGIES = frozenset({"naive", "hybrid", "rerank", "hyde", "multi_query"})
VALID_LLM_PROVIDERS = frozenset({"groq", "openai", "anthropic", "ollama", "google"})
VALID_EMBED_PROVIDERS = frozenset({"local", "openai", "cohere", "ollama", "google"})
VALID_RERANK_PROVIDERS = frozenset({"cohere", "none"})


class LLMConfig(BaseModel):
    """Config for the generator LLM (produces answers in the RAG pipeline)."""

    provider: str = "groq"
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        if v not in VALID_LLM_PROVIDERS:
            raise ValueError(
                f"Invalid LLM provider '{v}'. Choose from: {sorted(VALID_LLM_PROVIDERS)}"
            )
        return v


class JudgeConfig(BaseModel):
    """Config for the judge LLM (scores answers in RAGAS evaluation)."""

    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        if v not in VALID_LLM_PROVIDERS:
            raise ValueError(
                f"Invalid judge provider '{v}'. Choose from: {sorted(VALID_LLM_PROVIDERS)}"
            )
        return v


class EmbeddingsConfig(BaseModel):
    """Config for the embedding model used to build and query the FAISS index."""

    provider: str = "local"
    model: str = "BAAI/bge-large-en-v1.5"
    batch_size: int = Field(default=32, gt=0)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        if v not in VALID_EMBED_PROVIDERS:
            raise ValueError(
                f"Invalid embeddings provider '{v}'. Choose from: {sorted(VALID_EMBED_PROVIDERS)}"
            )
        return v


class RerankerConfig(BaseModel):
    """Config for the optional reranker (used by the 'rerank' strategy)."""

    provider: str = "cohere"
    model: str = "rerank-english-v3.0"
    top_n: int = Field(default=5, gt=0)
    enabled: bool = True


class RetrievalConfig(BaseModel):
    """Chunking and retrieval hyperparameters shared across strategies."""

    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)
    top_k: int = Field(default=5, gt=0, description="Chunks returned to the LLM")
    top_k_rerank: int = Field(default=20, gt=0, description="Candidates before reranking")


class DatasetConfig(BaseModel):
    """Dataset selection and sampling."""

    name: str = "hotpotqa"
    split: str = "validation"
    sample_size: int = Field(default=500, gt=0)
    seed: int = 42


class OutputConfig(BaseModel):
    """Where to write benchmark artifacts."""

    dir: str = "results"
    predictions: bool = True
    scorecard: bool = True
    report: bool = True


class Config(BaseModel):
    """Root config — validated on load, all fields have sensible defaults."""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generator: LLMConfig = Field(default_factory=LLMConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    strategies: list[str] = Field(
        default_factory=lambda: ["naive", "hybrid", "rerank", "hyde", "multi_query"]
    )
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("strategies")
    @classmethod
    def validate_strategies(cls, v: list[str]) -> list[str]:
        invalid = set(v) - VALID_STRATEGIES
        if invalid:
            raise ValueError(
                f"Unknown strategies: {sorted(invalid)}. Valid options: {sorted(VALID_STRATEGIES)}"
            )
        return v


def load_config(path: str | Path = "configs/groq_llama4.yaml") -> Config:
    """
    Load and validate config from a YAML file.

    Args:
        path: Path to a YAML config file. Defaults to the Groq + Llama 4 config.

    Returns:
        Validated Config object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config contains invalid values.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Available configs: {list(Path('configs').glob('*.yaml'))}"
        )

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    return Config(**data)
