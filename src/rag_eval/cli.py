"""
CLI entry point for rag-eval-harness.

Usage:
    python -m rag_eval --help
    python -m rag_eval info
    python -m rag_eval --config configs/openai.yaml info
    python -m rag_eval index
    python -m rag_eval run --strategy naive --strategy hybrid
    python -m rag_eval eval
    python -m rag_eval compare
"""
from __future__ import annotations

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
@click.option(
    "--config", "-c",
    default="configs/groq_llama4.yaml",
    show_default=True,
    help="Path to config YAML. See configs/ for available presets.",
)
@click.pass_context
def main(ctx: click.Context, config: str) -> None:
    """
    RAG Eval Harness — benchmark RAG retrieval strategies head-to-head.

    Swap LLMs, embedding models, and datasets via --config.
    Zero code changes required.

    \b
    Quick start:
        cp .env.example .env        # add GROQ_API_KEY and COHERE_API_KEY
        python -m rag_eval index    # build FAISS index
        python -m rag_eval run      # generate predictions
        python -m rag_eval eval     # score with RAGAS
        python -m rag_eval compare  # generate scorecard + HTML report
    """
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


@main.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """Show loaded config: providers, models, dataset, strategies."""
    from dotenv import load_dotenv
    from rag_eval.config import load_config

    load_dotenv()

    cfg = load_config(ctx.obj["config_path"])

    # Model slot table
    table = Table(title="Model Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Slot", style="cyan", min_width=12)
    table.add_column("Provider", style="green", min_width=10)
    table.add_column("Model")

    table.add_row("Generator", cfg.generator.provider, cfg.generator.model)
    table.add_row("Judge", cfg.judge.provider, cfg.judge.model)
    table.add_row("Embeddings", cfg.embeddings.provider, cfg.embeddings.model)
    table.add_row(
        "Reranker",
        cfg.reranker.provider if cfg.reranker.enabled else "disabled",
        cfg.reranker.model if cfg.reranker.enabled else "-",
    )
    console.print(table)

    # Dataset + retrieval summary
    console.print(
        f"\nDataset : [cyan]{cfg.dataset.name}[/] | "
        f"Split: [cyan]{cfg.dataset.split}[/] | "
        f"Samples: [cyan]{cfg.dataset.sample_size}[/] | "
        f"Seed: {cfg.dataset.seed}"
    )
    console.print(
        f"Chunks  : size=[cyan]{cfg.retrieval.chunk_size}[/], "
        f"overlap=[cyan]{cfg.retrieval.chunk_overlap}[/], "
        f"top_k=[cyan]{cfg.retrieval.top_k}[/]"
    )
    console.print(
        f"Strategies: [cyan]{', '.join(cfg.strategies)}[/] "
        f"({len(cfg.strategies)} total)"
    )
    console.print(f"Output  : [cyan]{cfg.output.dir}/[/]")


# ---------------------------------------------------------------------------
# Placeholder commands — implemented in later phases.
# Each prints a helpful message rather than silently doing nothing.
# ---------------------------------------------------------------------------

@main.command()
@click.pass_context
def index(ctx: click.Context) -> None:
    """Build FAISS index from dataset chunks. [Phase 1]"""
    console.print(Panel(
        "[yellow]Coming in Phase 1[/yellow]\n\n"
        "This command will:\n"
        "  1. Download the HotpotQA dataset\n"
        "  2. Chunk passages with overlap\n"
        "  3. Embed chunks with your configured embeddings model\n"
        "  4. Persist a FAISS index to data/index/",
        title="index",
        border_style="yellow",
    ))


@main.command()
@click.option(
    "--strategy", "-s",
    multiple=True,
    help="Strategy to run. Repeat to run multiple. Default: all strategies in config.",
)
@click.pass_context
def run(ctx: click.Context, strategy: tuple[str, ...]) -> None:
    """Run RAG strategies and collect predictions. [Phase 2-4]"""
    console.print(Panel(
        "[yellow]Coming in Phase 2-4[/yellow]\n\n"
        "This command will:\n"
        "  1. Load index from data/index/\n"
        "  2. Run each strategy on the dataset\n"
        "  3. Track cost and latency per query\n"
        "  4. Save predictions to results/predictions_<strategy>.jsonl",
        title="run",
        border_style="yellow",
    ))


@main.command("eval")
@click.pass_context
def eval_cmd(ctx: click.Context) -> None:
    """Score predictions with RAGAS metrics. [Phase 3]"""
    console.print(Panel(
        "[yellow]Coming in Phase 3[/yellow]\n\n"
        "This command will:\n"
        "  1. Load predictions from results/\n"
        "  2. Score with RAGAS: faithfulness, relevancy, context precision, recall\n"
        "  3. Aggregate per-strategy scores\n"
        "  4. Save scorecard_<strategy>.csv",
        title="eval",
        border_style="yellow",
    ))


@main.command()
@click.pass_context
def compare(ctx: click.Context) -> None:
    """Compare all strategies and generate scorecard + HTML report. [Phase 5]"""
    console.print(Panel(
        "[yellow]Coming in Phase 5[/yellow]\n\n"
        "This command will:\n"
        "  1. Load all scorecard CSVs\n"
        "  2. Build strategy x metric comparison matrix\n"
        "  3. Generate results.csv + report.html with plotly charts",
        title="compare",
        border_style="yellow",
    ))
