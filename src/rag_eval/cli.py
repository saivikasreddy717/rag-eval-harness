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
    python -m rag_eval eval-agent --traces examples/agent_traces_sample.jsonl
    python -m rag_eval eval-online --traces-stream examples/agent_traces_sample.jsonl --sample-rate 1.0
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.option(
    "--config",
    "-c",
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
    console.print(f"Strategies: [cyan]{', '.join(cfg.strategies)}[/] ({len(cfg.strategies)} total)")
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

    from rag_eval.chunker import chunk_corpus, chunk_stats
    from rag_eval.config import load_config
    from rag_eval.datasets import load_hotpotqa
    from rag_eval.indexer import build_index, index_exists

    cfg = load_config(ctx.obj["config_path"])

    if index_exists() and not rebuild:
        console.print(
            "[green]Index already exists.[/] Use [bold]--rebuild[/] to force a fresh build."
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
    "--strategy",
    "-s",
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
    from rag_eval.strategies import STRATEGY_REGISTRY, get_strategy

    cfg = load_config(ctx.obj["config_path"])

    if not index_exists():
        console.print("[red]No index found.[/] Run [bold]python -m rag_eval index[/] first.")
        raise SystemExit(1)

    # Resolve which strategies to run
    strategies_to_run = list(strategy) if strategy else cfg.strategies
    not_implemented = [s for s in strategies_to_run if s not in STRATEGY_REGISTRY]
    if not_implemented:
        console.print(
            f"[yellow]Skipping not-yet-implemented strategies:[/] {', '.join(not_implemented)}"
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
@click.option(
    "--strategy",
    "-s",
    multiple=True,
    help="Strategy to evaluate. Repeat for multiple. Default: all strategies in config.",
)
@click.option(
    "--max-questions",
    "-n",
    default=None,
    type=int,
    show_default=True,
    help=(
        "Evaluate only the first N questions. Useful for quick sanity checks "
        "or when API rate limits are a concern."
    ),
)
@click.pass_context
def eval_cmd(ctx: click.Context, strategy: tuple[str, ...], max_questions: int | None) -> None:
    """Score predictions with RAGAS (faithfulness, relevancy, precision, recall, correctness)."""
    from dotenv import load_dotenv

    load_dotenv()

    from pathlib import Path

    from rag_eval.config import load_config
    from rag_eval.evaluator import evaluate_predictions

    cfg = load_config(ctx.obj["config_path"])
    output_dir = Path(cfg.output.dir)

    # Resolve which strategies to evaluate
    strategies_to_eval = list(strategy) if strategy else cfg.strategies

    # Find available prediction files
    available = []
    missing = []
    for s in strategies_to_eval:
        pred_file = output_dir / f"predictions_{s}.jsonl"
        if pred_file.exists():
            available.append((s, pred_file))
        else:
            missing.append(s)

    if missing:
        console.print(
            f"[yellow]No predictions found for:[/] {', '.join(missing)}\n"
            "Run [bold]python -m rag_eval run[/] first."
        )

    if not available:
        console.print("[red]Nothing to evaluate.[/]")
        raise SystemExit(1)

    console.print(
        f"Evaluating [bold]{len(available)}[/] "
        f"strateg{'y' if len(available) == 1 else 'ies'}: "
        f"[cyan]{', '.join(s for s, _ in available)}[/]"
    )
    if max_questions:
        console.print(f"[dim]Capped at {max_questions} questions per strategy.[/]")

    scorecard_paths = []
    for _strategy_name, pred_file in available:
        console.rule(f"[cyan]{_strategy_name}[/]")
        scorecard = evaluate_predictions(pred_file, cfg, max_questions=max_questions)
        scorecard_paths.append(scorecard)

    console.rule("[green]Evaluation complete[/]")
    console.print(
        f"Scorecards saved to [cyan]{cfg.output.dir}/[/]. "
        "Run [bold]python -m rag_eval compare[/] next."
    )


@main.command("eval-agent")
@click.option(
    "--traces",
    "-t",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a JSONL file of AgentTrace objects (one per line).",
)
@click.pass_context
def eval_agent_cmd(ctx: click.Context, traces: str) -> None:
    """Score agent traces: tool selection, tool execution, and multi-step coherence."""
    from dotenv import load_dotenv

    load_dotenv()

    from pathlib import Path

    from rag_eval.agent_eval.evaluator import evaluate_agent_traces
    from rag_eval.config import load_config

    cfg = load_config(ctx.obj["config_path"])

    # Force agent_eval enabled for this command even if config says disabled
    cfg = cfg.model_copy(update={"agent_eval": cfg.agent_eval.model_copy(update={"enabled": True})})

    try:
        scorecard = evaluate_agent_traces(Path(traces), cfg)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise SystemExit(1)

    console.rule("[green]Agent eval complete[/]")
    console.print(f"[bold green]Scorecard:[/] [cyan]{scorecard}[/]")


@main.command("eval-online")
@click.option(
    "--traces-stream",
    "-t",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a JSONL stream of production traces (agent or RAG records).",
)
@click.option(
    "--sample-rate",
    "-r",
    default=None,
    type=float,
    show_default=True,
    help="Fraction of records to sample (overrides config). Default: value from config (0.05).",
)
@click.pass_context
def eval_online_cmd(ctx: click.Context, traces_stream: str, sample_rate: float | None) -> None:
    """
    Sample and score a stream of production traces, writing dated JSONL + rollup CSV.

    Supports mixed streams: agent traces (with trace_id + steps) and RAG predictions
    (with question + answer) are routed to the appropriate eval path automatically.
    """
    from dotenv import load_dotenv

    load_dotenv()

    from datetime import date
    from pathlib import Path

    from rag_eval.config import load_config
    from rag_eval.online.runner import run_online_eval
    from rag_eval.online.storage import build_daily_rollup

    cfg = load_config(ctx.obj["config_path"])
    today = date.today()

    try:
        jsonl_out = run_online_eval(
            Path(traces_stream),
            cfg,
            sample_rate=sample_rate,
            run_date=today,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise SystemExit(1)

    console.print(f"[green]Online eval JSONL:[/] [cyan]{jsonl_out}[/]")

    # Build daily rollup
    try:
        rollup_out = build_daily_rollup(jsonl_out, today)
        console.print(f"[green]Daily rollup CSV:[/] [cyan]{rollup_out}[/]")
    except ValueError as exc:
        console.print(f"[yellow]Rollup skipped:[/] {exc}")

    console.rule("[green]Online eval complete[/]")


@main.command()
@click.pass_context
def compare(ctx: click.Context) -> None:
    """Load all scorecards, build comparison matrix, and generate results.csv + report.html."""
    from dotenv import load_dotenv

    load_dotenv()

    from rag_eval.config import load_config
    from rag_eval.reporter import compare_strategies

    cfg = load_config(ctx.obj["config_path"])

    try:
        results_csv, report_html = compare_strategies(cfg)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/]")
        raise SystemExit(1)

    console.rule("[green]Done[/]")
    console.print(
        f"[bold green]Comparison complete.[/]\n"
        f"  Results CSV : [cyan]{results_csv}[/]\n"
        f"  HTML report : [cyan]{report_html}[/]\n\n"
        "Open [bold]report.html[/] in any browser to explore the interactive charts."
    )
