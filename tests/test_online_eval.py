"""
Tests for the online evaluation module (sampler, runner, storage).

No external API calls — all LLM/embedding calls are mocked.
CI-safe.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from rag_eval.agent_eval.fixtures import all_sample_traces, make_multi_step_trace
from rag_eval.online.sampler import ReservoirSampler
from rag_eval.online.storage import (
    append_result,
    build_daily_rollup,
    daily_output_path,
    load_daily_records,
    rollup_path,
)

# ---------------------------------------------------------------------------
# ReservoirSampler
# ---------------------------------------------------------------------------


class TestReservoirSampler:
    def test_deterministic_with_fixed_seed(self):
        items = list(range(100))
        s1 = ReservoirSampler(sample_rate=0.2, seed=42)
        s2 = ReservoirSampler(sample_rate=0.2, seed=42)
        s1.add_all(items)
        s2.add_all(items)
        assert s1.reservoir == s2.reservoir

    def test_different_seeds_produce_different_samples(self):
        items = list(range(200))
        s1 = ReservoirSampler(sample_rate=0.3, seed=1)
        s2 = ReservoirSampler(sample_rate=0.3, seed=999)
        s1.add_all(items)
        s2.add_all(items)
        assert s1.reservoir != s2.reservoir

    def test_sample_rate_1_keeps_all(self):
        items = list(range(50))
        s = ReservoirSampler(sample_rate=1.0, seed=0)
        s.add_all(items)
        assert s.n_sampled == 50

    def test_sample_rate_near_zero_keeps_few(self):
        items = list(range(10_000))
        s = ReservoirSampler(sample_rate=0.001, seed=7)
        s.add_all(items)
        # With 10k items at 0.1% rate, expect roughly 10 items; allow ±30
        assert 0 < s.n_sampled < 50

    def test_n_seen_tracks_all_offered_items(self):
        s = ReservoirSampler(sample_rate=0.5, seed=1)
        items = list(range(20))
        s.add_all(items)
        assert s.n_seen == 20

    def test_invalid_sample_rate_raises(self):
        with pytest.raises(ValueError):
            ReservoirSampler(sample_rate=0.0)
        with pytest.raises(ValueError):
            ReservoirSampler(sample_rate=1.5)

    def test_clear_resets_state(self):
        s = ReservoirSampler(sample_rate=1.0, seed=0)
        s.add_all(list(range(10)))
        s.clear()
        assert s.n_seen == 0
        assert s.n_sampled == 0
        assert s.reservoir == []

    def test_add_returns_bool(self):
        s = ReservoirSampler(sample_rate=1.0, seed=0)
        result = s.add("item")
        assert isinstance(result, bool)
        assert result is True


# ---------------------------------------------------------------------------
# Storage: append_result + load_daily_records
# ---------------------------------------------------------------------------


class TestStorage:
    def test_append_creates_file(self, tmp_path):
        p = tmp_path / "2026-05-18.jsonl"
        append_result(p, {"metric": "tool_selection", "score": 0.9})
        assert p.exists()

    def test_append_is_idempotent_accumulation(self, tmp_path):
        p = tmp_path / "out.jsonl"
        append_result(p, {"score": 0.8})
        append_result(p, {"score": 0.9})
        records = load_daily_records(p)
        assert len(records) == 2

    def test_load_daily_records_empty_for_missing_file(self, tmp_path):
        records = load_daily_records(tmp_path / "missing.jsonl")
        assert records == []

    def test_daily_output_path_format(self, tmp_path):
        p = daily_output_path(tmp_path, date(2026, 5, 18))
        assert p.name == "2026-05-18.jsonl"
        assert p.parent == tmp_path

    def test_rollup_path_format(self, tmp_path):
        p = rollup_path(tmp_path, date(2026, 5, 18))
        assert p.name == "rollup_2026-05-18.csv"


# ---------------------------------------------------------------------------
# Storage: daily rollup math
# ---------------------------------------------------------------------------


class TestDailyRollup:
    def _write_records(self, path: Path, records: list[dict]) -> None:
        for r in records:
            append_result(path, r)

    def test_rollup_computes_correct_mean(self, tmp_path):
        p = tmp_path / "2026-05-18.jsonl"
        scores = [0.8, 0.9, 1.0]
        for s in scores:
            append_result(p, {"tool_selection_accuracy": s})
        rollup_csv = build_daily_rollup(p, date(2026, 5, 18))
        df = pd.read_csv(rollup_csv)
        row = df[df["metric"] == "tool_selection_accuracy"].iloc[0]
        assert row["mean"] == pytest.approx(sum(scores) / len(scores), rel=1e-5)

    def test_rollup_computes_correct_p95(self, tmp_path):
        import numpy as np

        p = tmp_path / "2026-05-18.jsonl"
        # 20 values; p95 is deterministic with numpy
        values = [float(i) / 20 for i in range(1, 21)]
        for v in values:
            append_result(p, {"latency_ms": v})
        rollup_csv = build_daily_rollup(p, date(2026, 5, 18))
        df = pd.read_csv(rollup_csv)
        row = df[df["metric"] == "latency_ms"].iloc[0]
        expected_p95 = np.percentile(values, 95)
        assert row["p95"] == pytest.approx(expected_p95, rel=1e-5)

    def test_rollup_raises_for_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ValueError):
            build_daily_rollup(p, date(2026, 5, 18))

    def test_rollup_date_column_present(self, tmp_path):
        p = tmp_path / "2026-05-18.jsonl"
        append_result(p, {"tool_execution_success": 0.99})
        rollup_csv = build_daily_rollup(p, date(2026, 5, 18))
        df = pd.read_csv(rollup_csv)
        assert "date" in df.columns
        assert df["date"].iloc[0] == "2026-05-18"


# ---------------------------------------------------------------------------
# Online runner
# ---------------------------------------------------------------------------


class TestOnlineRunner:
    def _make_stream_file(self, tmp_path: Path, traces=None, rag_records=None) -> Path:
        p = tmp_path / "stream.jsonl"
        lines = []
        for t in traces or []:
            lines.append(t.model_dump_json())
        for r in rag_records or []:
            lines.append(json.dumps(r))
        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def test_runner_writes_jsonl(self, tmp_path):
        from rag_eval.config import Config, OnlineConfig, OutputConfig
        from rag_eval.online.runner import run_online_eval

        traces = all_sample_traces()
        stream = self._make_stream_file(tmp_path, traces=traces)

        cfg = Config()
        cfg = cfg.model_copy(
            update={
                "output": OutputConfig(dir=str(tmp_path)),
                "online": OnlineConfig(
                    enabled=True, sample_rate=1.0, output_dir=str(tmp_path / "online"), seed=42
                ),
            }
        )
        out = run_online_eval(stream, cfg, sample_rate=1.0, run_date=date(2026, 5, 18))
        assert out.exists()
        records = load_daily_records(out)
        assert len(records) == len(traces)

    def test_runner_samples_subset(self, tmp_path):
        from rag_eval.config import Config, OnlineConfig
        from rag_eval.online.runner import run_online_eval

        traces = [make_multi_step_trace(f"t-{i}", num_steps=3) for i in range(20)]
        stream = self._make_stream_file(tmp_path, traces=traces)
        cfg = Config()
        cfg = cfg.model_copy(
            update={
                "online": OnlineConfig(
                    enabled=True, sample_rate=0.5, output_dir=str(tmp_path / "online"), seed=42
                )
            }
        )
        out = run_online_eval(stream, cfg, sample_rate=0.5, run_date=date(2026, 5, 18))
        records = load_daily_records(out)
        # With seed=42 and rate=0.5 over 20 items, should be significantly less than 20
        assert len(records) < 20

    def test_runner_routes_rag_records(self, tmp_path):
        from rag_eval.config import Config, OnlineConfig
        from rag_eval.online.runner import run_online_eval

        rag_records = [
            {
                "id": f"q{i}",
                "question": f"Q{i}?",
                "answer": f"A{i}.",
                "contexts": ["c1", "c2"],
                "strategy": "naive",
                "latency_ms": 100.0,
                "retrieval_latency_ms": 30.0,
            }
            for i in range(5)
        ]
        stream = self._make_stream_file(tmp_path, rag_records=rag_records)
        cfg = Config()
        cfg = cfg.model_copy(
            update={
                "online": OnlineConfig(output_dir=str(tmp_path / "online"), sample_rate=1.0, seed=0)
            }
        )
        out = run_online_eval(stream, cfg, sample_rate=1.0, run_date=date(2026, 5, 18))
        records = load_daily_records(out)
        assert all(r["record_type"] == "rag" for r in records)

    def test_runner_raises_for_missing_file(self, tmp_path):
        from rag_eval.config import Config
        from rag_eval.online.runner import run_online_eval

        with pytest.raises(FileNotFoundError):
            run_online_eval(tmp_path / "nope.jsonl", Config())
