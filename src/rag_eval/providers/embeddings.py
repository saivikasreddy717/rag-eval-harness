"""
Embeddings provider factory.

Usage:
    from rag_eval.providers.embeddings import get_embeddings
    from rag_eval.config import load_config

    cfg = load_config("configs/groq_llama4.yaml")
    embeddings = get_embeddings(cfg.embeddings)  # Local BGE model

Supported providers
-------------------
local       sentence-transformers model running on your machine.
            Default: BAAI/bge-large-en-v1.5 (outperforms OpenAI text-embedding-3-small
            on MTEB, 335MB, downloads once and caches to ~/.cache/huggingface)
            No API key needed.

openai      OpenAI Embeddings API (text-embedding-3-small, text-embedding-3-large)
            Install extra: uv sync --extra openai
            Env: OPENAI_API_KEY

cohere      Cohere Embeddings API (embed-english-v3.0, embed-multilingual-v3.0)
            Env: COHERE_API_KEY

ollama      Local Ollama embeddings (nomic-embed-text, mxbai-embed-large, ...)
            Install extra: uv sync --extra ollama
            No API key needed.

google      Google Generative AI Embeddings
            Install extra: uv sync --extra google
            Env: GOOGLE_API_KEY
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from rag_eval.config import EmbeddingsConfig


def get_embeddings(config: EmbeddingsConfig) -> Embeddings:
    """
    Return a LangChain Embeddings instance for the given provider config.

    All returned embeddings implement the same LangChain interface so
    FAISS, BM25 fusion, and HyDE never need to branch on provider.

    Args:
        config: EmbeddingsConfig from the loaded YAML.

    Returns:
        A LangChain-compatible embeddings model.

    Raises:
        ImportError: If the required provider package is not installed.
        ValueError: If an unsupported provider is specified.
    """
    provider = config.provider.lower()

    # --------------------------------------------------------------- local
    if provider == "local":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-huggingface is not installed.\n"
                "It is included in core deps — run: uv sync"
            )
        return HuggingFaceEmbeddings(
            model_name=config.model,
            encode_kwargs={
                "batch_size": config.batch_size,
                "normalize_embeddings": True,  # required for cosine similarity
            },
        )

    # --------------------------------------------------------------- openai
    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-openai is not installed.\nInstall with: uv sync --extra openai"
            )
        return OpenAIEmbeddings(model=config.model)

    # --------------------------------------------------------------- cohere
    if provider == "cohere":
        try:
            from langchain_cohere import CohereEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-cohere is not installed.\n"
                "Install with: uv sync --extra openai  (bundled with cohere extra)"
            )
        return CohereEmbeddings(model=config.model)

    # --------------------------------------------------------------- ollama
    if provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-ollama is not installed.\nInstall with: uv sync --extra ollama"
            )
        return OllamaEmbeddings(model=config.model)

    # --------------------------------------------------------------- google
    if provider == "google":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-google-genai is not installed.\nInstall with: uv sync --extra google"
            )
        return GoogleGenerativeAIEmbeddings(model=config.model)

    raise ValueError(
        f"Unsupported embeddings provider: '{provider}'.\n"
        f"Valid options: local, openai, cohere, ollama, google"
    )
