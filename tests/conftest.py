"""
Shared pytest configuration for rag-eval-harness.

This file is loaded automatically by pytest before any tests run.

What lives here
---------------
- Session-level environment variable setup that must happen before any
  import of sentence-transformers or HuggingFace tokenizers.

What does NOT live here
------------------------
- Test fixtures that are only used in one file (keep them in that file).
- The built_index fixture is intentionally duplicated across test files
  to keep each file self-contained and independently runnable.

CI notes
--------
All tests are CI-safe: external LLM / embedding API calls are mocked via
unittest.mock.patch.  The only real network dependency is the initial
download of the BAAI/bge-large-en-v1.5 embedding model (~1.3 GB), which
the GitHub Actions workflow caches under ~/.cache/huggingface/hub after
the first run.  Subsequent CI runs reuse the cache and stay fast.

No API keys (GROQ_API_KEY, COHERE_API_KEY, etc.) are needed to run the
test suite.  Tests that instantiate provider clients do so inside
`patch.dict(os.environ, {"KEY": "dummy"})` blocks.
"""

from __future__ import annotations

import os

# Silence the HuggingFace tokenizers parallelism fork warning.
# Must be set before the `tokenizers` package is imported (i.e. before
# sentence-transformers / langchain-huggingface loads).  Setting it here
# at module import time — before any test collection — is safe.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
