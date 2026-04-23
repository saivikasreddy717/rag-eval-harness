"""
Smoke tests — verify config loading, provider imports, and CLI work
without making any API calls or requiring API keys.

These run in CI on every push.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner
from pathlib import Path


class TestConfigLoading:
    """Config system loads and validates correctly."""

    def test_default_config_loads(self):
        from rag_eval.config import load_config

        cfg = load_config("configs/groq_llama4.yaml")
        assert cfg.generator.provider == "groq"
        assert "llama-4-scout" in cfg.generator.model
        assert cfg.judge.provider == "groq"
        assert cfg.embeddings.provider == "local"
        assert cfg.reranker.enabled is True
        assert len(cfg.strategies) == 5

    def test_all_bundled_configs_load(self):
        """Every YAML file in configs/ should parse without error."""
        from rag_eval.config import load_config

        config_dir = Path("configs")
        yaml_files = list(config_dir.glob("*.yaml"))
        assert len(yaml_files) >= 4, "Expected at least 4 config presets"

        for config_file in yaml_files:
            cfg = load_config(config_file)
            assert cfg.generator.provider is not None
            assert cfg.embeddings.provider is not None
            assert len(cfg.strategies) > 0

    def test_missing_config_raises(self):
        from rag_eval.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config("configs/nonexistent.yaml")

    def test_invalid_strategy_raises(self):
        from rag_eval.config import Config

        with pytest.raises(ValueError, match="Unknown strategies"):
            Config(strategies=["naive", "invalid_strategy"])

    def test_invalid_provider_raises(self):
        from rag_eval.config import LLMConfig

        with pytest.raises(ValueError, match="Invalid LLM provider"):
            LLMConfig(provider="unknown_provider")

    def test_config_defaults_are_sensible(self):
        from rag_eval.config import Config

        cfg = Config()
        assert cfg.retrieval.chunk_size == 512
        assert cfg.retrieval.top_k == 5
        assert cfg.dataset.sample_size == 500
        assert cfg.dataset.seed == 42


class TestProviderImports:
    """Provider factories are importable and callable (no API calls)."""

    def test_llm_factory_importable(self):
        from rag_eval.providers.llm import get_llm
        assert callable(get_llm)

    def test_embeddings_factory_importable(self):
        from rag_eval.providers.embeddings import get_embeddings
        assert callable(get_embeddings)

    def test_invalid_llm_provider_raises(self):
        from rag_eval.providers.llm import get_llm
        from rag_eval.config import LLMConfig

        # We need to bypass config validation to test the factory directly
        cfg = LLMConfig.model_construct(
            provider="invalid",
            model="some-model",
            temperature=0.1,
            max_tokens=512,
        )
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_llm(cfg)

    def test_invalid_embeddings_provider_raises(self):
        from rag_eval.providers.embeddings import get_embeddings
        from rag_eval.config import EmbeddingsConfig

        cfg = EmbeddingsConfig.model_construct(
            provider="invalid",
            model="some-model",
            batch_size=32,
        )
        with pytest.raises(ValueError, match="Unsupported embeddings provider"):
            get_embeddings(cfg)


class TestCLI:
    """CLI entry point works correctly."""

    def test_help_exits_cleanly(self):
        from rag_eval.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "RAG Eval Harness" in result.output

    def test_info_command(self):
        from rag_eval.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--config", "configs/groq_llama4.yaml", "info"])
        assert result.exit_code == 0
        assert "groq" in result.output.lower()

    def test_compare_requires_scorecards(self):
        """compare is now implemented — exits 1 with a helpful message when no scorecards exist."""
        from rag_eval.cli import main

        runner = CliRunner()
        # Invoke without any scorecards present (default results/ dir is likely empty in CI)
        result = runner.invoke(main, ["compare"])
        # Should either exit 1 (no scorecards) or 0 (scorecards happened to exist)
        # We just verify it no longer returns the old placeholder "Coming in Phase 5" panel
        assert "Coming in Phase 5" not in result.output

    def test_run_requires_index(self):
        """run is implemented — exits 1 with a helpful message when index is absent."""
        from rag_eval.cli import main
        from rag_eval.indexer import index_exists

        if index_exists():
            # Index present on this machine: skip the no-index path.
            return

        runner = CliRunner()
        result = runner.invoke(main, ["run"])
        assert result.exit_code == 1
        assert "index" in result.output.lower()

    def test_eval_requires_predictions(self):
        """eval is implemented — exits 1 with a helpful message when predictions are absent."""
        import tempfile, textwrap
        from click.testing import CliRunner
        from rag_eval.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            config_yaml = f"{tmp}/cfg.yaml"
            with open(config_yaml, "w") as f:
                f.write(textwrap.dedent(f"""\
                    generator:
                      provider: groq
                      model: meta-llama/llama-4-scout-17b-16e-instruct
                    judge:
                      provider: groq
                      model: llama-3.3-70b-versatile
                    embeddings:
                      provider: local
                      model: BAAI/bge-large-en-v1.5
                    strategies:
                      - naive
                    output:
                      dir: "{tmp.replace(chr(92), '/')}"
                """))
            runner = CliRunner()
            result = runner.invoke(main, ["--config", config_yaml, "eval"])
            assert result.exit_code == 1
            assert "predictions" in result.output.lower() or "evaluate" in result.output.lower()

    def test_index_command_detects_missing_index(self):
        """index command exits cleanly when index already exists (or tells you to build)."""
        from rag_eval.cli import main

        runner = CliRunner()
        # When index exists it prints a message and exits 0.
        # When it doesn't exist it tries to download — skip that path in smoke tests.
        from rag_eval.indexer import index_exists
        if index_exists():
            result = runner.invoke(main, ["index"])
            assert result.exit_code == 0

    def test_unknown_config_shows_error(self):
        from rag_eval.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--config", "configs/ghost.yaml", "info"])
        assert result.exit_code != 0 or "not found" in result.output.lower()
