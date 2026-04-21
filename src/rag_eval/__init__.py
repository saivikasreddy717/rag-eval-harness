"""
rag-eval-harness
================
Benchmark RAG retrieval strategies head-to-head with RAGAS metrics.
Model-agnostic: swap LLMs and embedding models via config, zero code changes.

Strategies: naive | hybrid | rerank | hyde | multi_query
Providers : groq  | openai | anthropic | ollama | google
"""

__version__ = "0.1.0"
__author__ = "Sai Vikas Reddy Yeddulamala"
