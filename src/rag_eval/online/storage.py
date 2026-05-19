"""
Online evaluation storage: append to rolling JSONL + daily rollup CSV.

File layout
-----------
results/online/YYYY-MM-DD.jsonl   — raw scored records for that day
results/online/rollup_YYYY-MM-DD.csv — aggregated stats (mean, p50, p95, p99)

Rollup metrics computed for any numeric column present in the day's JSONL:
  mean, p50 (median), p95, p99

The rollup is written fresh each time to allow re-runs to update it without
duplicate rows.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


def daily_output_path(output_dir: Path | str, run_date: date) -> Path:
    """
    Return the path to the dated JSONL file for a given day.

    Args:
        output_dir: Base directory for online eval output.
        run_date:   Date of the run.

    Returns:
        Path like output_dir/YYYY-MM-DD.jsonl
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{run_date.isoformat()}.jsonl"


def rollup_path(output_dir: Path | str, run_date: date) -> Path:
    """
    Return the path to the daily rollup CSV for a given day.

    Args:
        output_dir: Base directory for online eval output.
        run_date:   Date of the rollup.

    Returns:
        Path like output_dir/rollup_YYYY-MM-DD.csv
    """
    return Path(output_dir) / f"rollup_{run_date.isoformat()}.csv"


def append_result(jsonl_path: Path, record: dict[str, Any]) -> None:
    """
    Append a single scored record to the daily JSONL file.

    Idempotent with respect to the file: always appends a new line.
    The parent directory is created if it does not exist.

    Args:
        jsonl_path: Path to the .jsonl file to append to.
        record:     Dict to serialise as a JSON line.
    """
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_daily_records(jsonl_path: Path) -> list[dict[str, Any]]:
    """
    Load all records from a dated JSONL file.

    Args:
        jsonl_path: Path to the .jsonl file.

    Returns:
        List of dicts, one per scored record.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        return []
    records = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def build_daily_rollup(jsonl_path: Path, run_date: date) -> Path:
    """
    Aggregate numeric metrics from a daily JSONL into a rollup CSV.

    Computes mean, p50, p95, p99 for each numeric column.  The rollup is
    written to rollup_YYYY-MM-DD.csv in the same directory as the JSONL.

    Args:
        jsonl_path: Path to the dated JSONL file.
        run_date:   Date label written into the rollup's "date" column.

    Returns:
        Path to the written rollup CSV.
    """
    records = load_daily_records(jsonl_path)
    if not records:
        raise ValueError(f"No records found in {jsonl_path}.")

    df = pd.DataFrame(records)
    numeric_cols = [
        c
        for c in df.select_dtypes(include="number").columns
        if c not in ("num_steps",)  # exclude purely structural columns
    ]

    if not numeric_cols:
        raise ValueError("No numeric metric columns found in daily JSONL.")

    rows = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        rows.append(
            {
                "date": run_date.isoformat(),
                "metric": col,
                "mean": round(series.mean(), 6),
                "p50": round(series.quantile(0.50), 6),
                "p95": round(series.quantile(0.95), 6),
                "p99": round(series.quantile(0.99), 6),
                "n": len(series),
            }
        )

    rollup_df = pd.DataFrame(rows)
    out_path = rollup_path(jsonl_path.parent, run_date)
    rollup_df.to_csv(out_path, index=False)
    return out_path
