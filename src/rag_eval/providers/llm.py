"""
LLM provider factory — the core of the model-agnostic design.

Usage:
    from rag_eval.providers.llm import get_llm
    from rag_eval.config import load_config

    cfg = load_config("configs/groq_llama4.yaml")
    llm = get_llm(cfg.generator)      # Groq / Llama 4 Scout
    judge = get_llm(cfg.judge)        # Groq / Llama 3.3 70B

Switching providers requires zero code changes.
Edit your YAML config and re-run.

Supported providers
-------------------
groq        Groq API  (Llama 4 Scout, Llama 3.3 70B, Mixtral, ...)
            Free tier at https://console.groq.com
            Env: GROQ_API_KEY

openai      OpenAI API (GPT-4o, GPT-4o-mini, o3-mini, ...)
            Install extra: uv sync --extra openai
            Env: OPENAI_API_KEY

anthropic   Anthropic API (Claude Haiku, Sonnet, Opus, ...)
            Install extra: uv sync --extra anthropic
            Env: ANTHROPIC_API_KEY

ollama      Local Ollama server (Llama 3, Mistral, Phi-3, Gemma, ...)
            Install Ollama at https://ollama.com, then: ollama pull llama3.1
            Install extra: uv sync --extra ollama
            No API key needed.

google      Google AI / Gemini (Gemini 2.0 Flash, Gemini Pro, ...)
            Free tier at https://aistudio.google.com
            Install extra: uv sync --extra google
            Env: GOOGLE_API_KEY
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from rag_eval.config import JudgeConfig, LLMConfig


def get_llm(config: LLMConfig | JudgeConfig) -> BaseChatModel:
    """
    Return a LangChain BaseChatModel for the given provider config.

    All returned models implement the same LangChain interface so the
    rest of the codebase never needs to branch on provider.

    Args:
        config: LLMConfig or JudgeConfig (both share the same fields).

    Returns:
        A LangChain-compatible chat model.

    Raises:
        ImportError: If the required provider package is not installed.
        ValueError: If an unsupported provider is specified.
    """
    provider = config.provider.lower()

    # ------------------------------------------------------------------ groq
    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError(
                "langchain-groq is not installed.\nIt is included in core deps — run: uv sync"
            )
        return ChatGroq(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    # --------------------------------------------------------------- openai
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai is not installed.\nInstall with: uv sync --extra openai"
            )
        return ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    # ------------------------------------------------------------- anthropic
    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic is not installed.\nInstall with: uv sync --extra anthropic"
            )
        return ChatAnthropic(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    # --------------------------------------------------------------- ollama
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-ollama is not installed.\n"
                "Install with: uv sync --extra ollama\n"
                "Also ensure Ollama is running: https://ollama.com"
            )
        return ChatOllama(
            model=config.model,
            temperature=config.temperature,
        )

    # --------------------------------------------------------------- google
    if provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai is not installed.\nInstall with: uv sync --extra google"
            )
        return ChatGoogleGenerativeAI(
            model=config.model,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        )

    raise ValueError(
        f"Unsupported LLM provider: '{provider}'.\n"
        f"Valid options: groq, openai, anthropic, ollama, google"
    )
