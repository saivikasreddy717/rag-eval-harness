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
@click.option("--rebuild", is_flag=True, default=False, help="Force rebuild even if index exists.")
@click.pass_context
def index(ctx: click.Context, rebuild: bool) -> None:
    """Build FAISS + BM25 index from dataset chunks."""
    from dotenv import load_dotenv
    load_dotenv()

    from rag_eval.config import load_config
    from rag_eval.datasets import load_hotpotqa
    from rag_eval.chunker import chunk_corpus, chunk_stats
    from rag_eval.indexer import build_index, index_exists

    cfg = load_config(ctx.obj["config_path"])

    if index_exists() and not rebuild:
        console.print(
            "[green]Index already exists.[/] "
            "Use [bold]--rebuild[/] to force a fresh build."
        )
        return

    # Step 1: load dataset
    console.rule("[cyan]Step 1/3  Dataset[/]")
    data = load_hotpotqa(cfg.dataset, force=rebuild)

    # Step 2: chunk
    console.rule("[cyan]Step 2/3  Chunking[/]")
    chunks = chunk_corpus(data["corpus"], cfg.retrieval)
    stats = chunk_stats(chunks, cfg.retrieval)
    console.print(
        f"[green]{stats['num_chunks']} chunks[/] "
        f"from {len(data['corpus'])} passages "
        f"(avg {stats['avg_tokens']} tokens, "
        f"range {stats['min_tokens']}-{stats['max_tokens']})"
    )

    # Step 3: embed + build index
    console.rule("[cyan]Step 3/3  Indexing[/]")
    build_index(chunks, cfg)

    console.rule("[green]Done[/]")
    console.print(
        f"[bold green]Index ready.[/] "
        f"{stats['num_chunks']} chunks indexed. "
        f"Run [bold]python -m rag_eval run[/] next."
    )


@main.command()
@click.option(
    "--strategy", "-s",
    multiple=True,
    help="Strategy to run. Repeat for multiple. Default: all strategies in config.",
)
@click.pass_context
def run(ctx: click.Context, strategy: tuple[str, ...]) -> None:
    """Run RAG strategies over the dataset and collect predictions."""
    from dotenv import load_dotenv
    load_dotenv()

    from pathlib import Path
    from rag_eval.config import load_config
    from rag_eval.datasets import load_hotpotqa
    from rag_eval.indexer import RAGIndex, index_exists
    from rag_eval.runner import run_strategy
    from rag_eval.strategies import get_strategy, STRATEGY_REGISTRY

    cfg = load_config(ctx.obj["config_path"])

    if not index_exists():
        console.print(
            "[red]No index found.[/] Run [bold]python -m rag_eval index[/] first."
        )
        raise SystemExit(1)

    # Resolve which strategies to run
    strategies_to_run = list(strategy) if strategy else cfg.strategies
    not_implemented = [s for s in strategies_to_run if s not in STRATEGY_REGISTRY]
    if not_implemented:
        console.print(
            f"[yellow]Skipping not-yet-implemented strategies:[/] "
            f"{', '.join(not_implemented)}"
        )
        strategies_to_run = [s for s in strategies_to_run if s in STRATEGY_REGISTRY]

    if not strategies_to_run:
        console.print("[red]No implemented strategies to run.[/]")
        raise SystemExit(1)

    # Load shared resources once
    console.print("[cyan]Loading dataset and index...[/]")
    data = load_hotpotqa(cfg.dataset)
    rag_index = RAGIndex.load()
    output_dir = Path(cfg.output.dir)

    console.print(
        f"Running [bold]{len(strategies_to_run)}[/] "
        f"strateg{'y' if len(strategies_to_run) == 1 else 'ies'}: "
        f"[cyan]{', '.join(strategies_to_run)}[/]"
    )
    console.print(f"Dataset: [cyan]{len(data['qa_pairs'])} questions[/]\n")

    for strategy_name in strategies_to_run:
        console.rule(f"[cyan]{strategy_name}[/]")
        strat = get_strategy(strategy_name, cfg, rag_index)
        run_strategy(strat, data["qa_pairs"], output_dir)

    console.rule("[green]All strategies complete[/]")
    console.print(
        f"Predictions saved to [cyan]{cfg.output.dir}/[/]. "
        "Run [bold]python -m rag_eval eval[/] next."
    )


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
