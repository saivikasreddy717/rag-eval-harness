"""
Comparison report generator for rag-eval-harness.

Loads all per-strategy scorecard CSVs, builds a compact strategy × metric
summary table, and writes two artefacts to the output directory:

  results/results.csv   — strategy × metric comparison matrix (one row per strategy)
  results/report.html   — interactive Plotly dashboard (self-contained HTML)

Charts in report.html
---------------------
  1. Grouped bar  — all 5 RAGAS metrics side-by-side per strategy
  2. Radar / spider — multi-metric shape per strategy (at-a-glance winner)
  3. Latency bar  — mean latency in milliseconds per strategy
  4. Cost vs accuracy scatter — cost_usd (x) vs answer_correctness (y)

The HTML embeds Plotly from CDN — no extra JS files needed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from rag_eval.config import Config
from rag_eval.evaluator import _METRIC_NAMES

console = Console()

# Columns to keep in the comparison matrix (order matters for display)
_SUMMARY_COLS = _METRIC_NAMES + ["latency_ms", "cost_usd"]

# Friendly display names for charts
_METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "answer_correctness": "Answer Correctness",
    "latency_ms": "Latency (ms)",
    "cost_usd": "Cost / query (USD)",
}

# Plotly colour palette (one per strategy, colour-blind friendly)
_STRATEGY_COLOURS = [
    "#4C72B0",  # navy blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_scorecards(output_dir: Path) -> pd.DataFrame:
    """
    Load all scorecard_*.csv files found in output_dir.

    Args:
        output_dir: Directory containing scorecard CSVs written by evaluator.py.

    Returns:
        Combined DataFrame with all rows from all scorecards.

    Raises:
        FileNotFoundError: If output_dir doesn't exist or contains no scorecards.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(
            f"Output directory not found: {output_dir}\nRun: python -m rag_eval eval first."
        )

    scorecard_files = sorted(output_dir.glob("scorecard_*.csv"))
    if not scorecard_files:
        raise FileNotFoundError(
            f"No scorecard CSVs found in {output_dir}\nRun: python -m rag_eval eval first."
        )

    frames = []
    for path in scorecard_files:
        df = pd.read_csv(path)
        frames.append(df)
        console.print(f"  [dim]Loaded[/] {path.name} ({len(df) - 1} questions + AGGREGATE)")

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Matrix building
# ---------------------------------------------------------------------------


def build_comparison_matrix(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a compact strategy × metric comparison matrix.

    Extracts the AGGREGATE row from each strategy and returns a DataFrame
    with one row per strategy and columns for each RAGAS metric plus
    latency and cost.

    Args:
        combined_df: Output of load_scorecards().

    Returns:
        DataFrame with columns [strategy] + _SUMMARY_COLS.
        Indexed by strategy name for easy lookup.
    """
    agg = combined_df[combined_df["question_id"] == "AGGREGATE"].copy()

    # Keep only the columns we care about
    keep_cols = ["strategy"] + [c for c in _SUMMARY_COLS if c in agg.columns]
    matrix = agg[keep_cols].reset_index(drop=True)
    matrix = matrix.sort_values("strategy").reset_index(drop=True)

    return matrix


# ---------------------------------------------------------------------------
# Plotly chart builders (each returns a Plotly Figure)
# ---------------------------------------------------------------------------


def _strategy_colours(strategies: list[str]) -> dict[str, str]:
    """Map strategy names to colours."""
    return {s: _STRATEGY_COLOURS[i % len(_STRATEGY_COLOURS)] for i, s in enumerate(strategies)}


def _make_metrics_bar(matrix: pd.DataFrame):
    """Grouped bar chart: RAGAS metrics (x) × score (y), grouped by strategy."""
    import plotly.graph_objects as go

    strategies = matrix["strategy"].tolist()
    colours = _strategy_colours(strategies)

    fig = go.Figure()
    for _, row in matrix.iterrows():
        strat = row["strategy"]
        scores = [row.get(m, float("nan")) for m in _METRIC_NAMES]
        fig.add_trace(
            go.Bar(
                name=strat,
                x=[_METRIC_LABELS.get(m, m) for m in _METRIC_NAMES],
                y=scores,
                marker_color=colours[strat],
                text=[f"{v:.3f}" if pd.notna(v) else "N/A" for v in scores],
                textposition="outside",
            )
        )

    fig.update_layout(
        title="RAGAS Metrics by Strategy",
        xaxis_title="Metric",
        yaxis_title="Score (0–1)",
        yaxis=dict(range=[0, 1.15]),
        barmode="group",
        legend_title="Strategy",
        template="plotly_white",
        height=450,
    )
    return fig


def _make_radar(matrix: pd.DataFrame):
    """Radar / spider chart: one polygon per strategy covering all 5 RAGAS metrics."""
    import plotly.graph_objects as go

    strategies = matrix["strategy"].tolist()
    colours = _strategy_colours(strategies)
    labels = [_METRIC_LABELS.get(m, m) for m in _METRIC_NAMES]
    # Close the polygon by repeating the first point
    theta = labels + [labels[0]]

    fig = go.Figure()
    for _, row in matrix.iterrows():
        strat = row["strategy"]
        r = [row.get(m, 0.0) for m in _METRIC_NAMES]
        r_closed = r + [r[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=r_closed,
                theta=theta,
                fill="toself",
                name=strat,
                line=dict(color=colours[strat]),
                fillcolor=colours[strat],
                opacity=0.25,
            )
        )

    fig.update_layout(
        title="Strategy Radar: All RAGAS Metrics",
        polar=dict(radialaxis=dict(range=[0, 1], tickformat=".2f")),
        legend_title="Strategy",
        template="plotly_white",
        height=450,
    )
    return fig


def _make_latency_bar(matrix: pd.DataFrame):
    """Horizontal bar chart of mean query latency per strategy."""
    import plotly.graph_objects as go

    if "latency_ms" not in matrix.columns:
        return None

    strategies = matrix["strategy"].tolist()
    colours = _strategy_colours(strategies)
    latencies = matrix["latency_ms"].tolist()

    fig = go.Figure(
        go.Bar(
            x=latencies,
            y=strategies,
            orientation="h",
            marker_color=[colours[s] for s in strategies],
            text=[f"{v:.0f} ms" for v in latencies],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Mean Query Latency by Strategy",
        xaxis_title="Latency (ms)",
        yaxis_title="Strategy",
        template="plotly_white",
        height=350,
        xaxis=dict(range=[0, max(latencies) * 1.25 if latencies else 1000]),
    )
    return fig


def _make_cost_scatter(matrix: pd.DataFrame):
    """Scatter plot: cost per query (x) vs answer_correctness (y)."""
    import plotly.graph_objects as go

    if "cost_usd" not in matrix.columns or "answer_correctness" not in matrix.columns:
        return None

    strategies = matrix["strategy"].tolist()
    colours = _strategy_colours(strategies)

    fig = go.Figure()
    for _, row in matrix.iterrows():
        strat = row["strategy"]
        x = row.get("cost_usd", float("nan"))
        y = row.get("answer_correctness", float("nan"))
        if pd.isna(x) or pd.isna(y):
            continue
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(size=18, color=colours[strat]),
                text=[strat],
                textposition="top center",
                name=strat,
            )
        )

    fig.update_layout(
        title="Cost vs Answer Correctness",
        xaxis_title="Cost per query (USD)",
        yaxis_title="Answer Correctness (0–1)",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
        height=400,
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# HTML report assembly
# ---------------------------------------------------------------------------


def generate_html_report(matrix: pd.DataFrame, output_file: Path) -> None:
    """
    Generate a self-contained interactive HTML report.

    Args:
        matrix:      Output of build_comparison_matrix().
        output_file: Where to write the HTML file.
    """
    import plotly.io as pio

    charts = [
        ("ragas_metrics", _make_metrics_bar(matrix)),
        ("radar", _make_radar(matrix)),
        ("latency", _make_latency_bar(matrix)),
        ("cost_scatter", _make_cost_scatter(matrix)),
    ]

    # Convert each figure to an HTML div (no full-page wrapper, no duplicate Plotly JS)
    chart_divs = []
    plotlyjs_included = False
    for _name, fig in charts:
        if fig is None:
            continue
        include_js = "cdn" if not plotlyjs_included else False
        div = pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs=include_js,
            config={"displayModeBar": True, "responsive": True},
        )
        chart_divs.append(div)
        plotlyjs_included = True

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    strategies_run = ", ".join(matrix["strategy"].tolist())

    # Build the summary table HTML
    table_html = matrix.to_html(
        index=False,
        float_format=lambda x: f"{x:.4f}",
        classes="summary-table",
        border=0,
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>RAG Eval Harness — Strategy Comparison Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f8f9fa;
      color: #212529;
      margin: 0;
      padding: 0;
    }}
    .header {{
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
      color: white;
      padding: 2rem 2.5rem;
    }}
    .header h1 {{ margin: 0 0 0.25rem 0; font-size: 1.8rem; }}
    .header p  {{ margin: 0; opacity: 0.75; font-size: 0.9rem; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem; }}
    .card {{
      background: white;
      border-radius: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.1);
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }}
    .card h2 {{ margin-top: 0; font-size: 1.1rem; color: #495057; }}
    .summary-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }}
    .summary-table th {{
      background: #343a40;
      color: white;
      padding: 0.5rem 0.75rem;
      text-align: left;
    }}
    .summary-table td {{
      padding: 0.45rem 0.75rem;
      border-bottom: 1px solid #dee2e6;
    }}
    .summary-table tr:nth-child(even) {{ background: #f8f9fa; }}
    .charts-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
    }}
    @media (max-width: 768px) {{
      .charts-grid {{ grid-template-columns: 1fr; }}
    }}
    .footer {{
      text-align: center;
      padding: 1.5rem;
      color: #6c757d;
      font-size: 0.8rem;
    }}
  </style>
</head>
<body>

<div class="header">
  <h1>RAG Eval Harness — Strategy Comparison</h1>
  <p>Generated: {timestamp} &nbsp;|&nbsp; Strategies: {strategies_run}</p>
</div>

<div class="container">

  <div class="card">
    <h2>Summary Table (AGGREGATE scores)</h2>
    {table_html}
  </div>

  <div class="charts-grid">
    <div class="card">{"</div><div class='card'>".join(chart_divs)}</div>
  </div>

</div>

<div class="footer">
  Built with
  <a href="https://github.com/saivikasreddy717/rag-eval-harness" target="_blank">
    rag-eval-harness
  </a>
  &bull; RAGAS evaluation &bull; Plotly charts
</div>

</body>
</html>"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Rich console summary table
# ---------------------------------------------------------------------------


def _print_comparison_table(matrix: pd.DataFrame) -> None:
    """Print a rich comparison table to the console."""
    table = Table(
        title="Strategy Comparison (AGGREGATE)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Strategy", style="cyan", min_width=12)
    for col in _METRIC_NAMES:
        table.add_column(col.replace("_", " ").title(), justify="right", min_width=10)
    if "latency_ms" in matrix.columns:
        table.add_column("Latency ms", justify="right", min_width=10)
    if "cost_usd" in matrix.columns:
        table.add_column("Cost USD", justify="right", min_width=10)

    for _, row in matrix.iterrows():
        cells = [str(row["strategy"])]
        for col in _METRIC_NAMES:
            v = row.get(col)
            cells.append(f"{v:.4f}" if pd.notna(v) else "N/A")
        if "latency_ms" in matrix.columns:
            cells.append(f"{row['latency_ms']:.0f}")
        if "cost_usd" in matrix.columns:
            cells.append(f"${row['cost_usd']:.6f}")
        table.add_row(*cells)

    console.print(table)

    # Highlight the winner on each metric
    console.print("\n[bold]Best per metric:[/]")
    for col in _METRIC_NAMES:
        if col not in matrix.columns:
            continue
        col_vals = matrix[col].dropna()
        if col_vals.empty:
            continue
        best_idx = col_vals.idxmax()
        best_strategy = matrix.loc[best_idx, "strategy"]
        best_val = col_vals.max()
        console.print(
            f"  [green]{col.replace('_', ' ').title()}[/]: "
            f"[bold]{best_strategy}[/] ({best_val:.4f})"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compare_strategies(cfg: Config) -> tuple[Path, Path]:
    """
    Load all scorecard CSVs, build a comparison matrix, and write reports.

    Args:
        cfg: Full Config — used for output directory.

    Returns:
        (results_csv_path, report_html_path)
    """
    output_dir = Path(cfg.output.dir)

    console.print(f"[cyan]Loading scorecards from[/] {output_dir}/")
    combined_df = load_scorecards(output_dir)

    console.print("[cyan]Building comparison matrix...[/]")
    matrix = build_comparison_matrix(combined_df)

    console.print(
        f"Comparing [bold]{len(matrix)}[/] strateg{'y' if len(matrix) == 1 else 'ies'}: "
        f"[cyan]{', '.join(matrix['strategy'].tolist())}[/]"
    )

    # Save results.csv
    results_csv = output_dir / "results.csv"
    matrix.to_csv(results_csv, index=False)
    console.print(f"[green]Saved:[/] {results_csv}")

    # Generate HTML report
    report_html = output_dir / "report.html"
    console.print("[cyan]Generating HTML report...[/]")
    generate_html_report(matrix, report_html)
    console.print(f"[green]Saved:[/] {report_html}")

    # Print comparison table + winners to console
    _print_comparison_table(matrix)

    return results_csv, report_html
