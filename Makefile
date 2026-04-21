.PHONY: install test lint index run eval compare report benchmark clean

## Setup
install:
	uv sync

install-all:
	uv sync --all-extras

## Development
test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:
	uv run ruff format src/ tests/

## Benchmark pipeline (run in order)
index:
	uv run python -m rag_eval --config configs/groq_llama4.yaml index

run:
	uv run python -m rag_eval --config configs/groq_llama4.yaml run

eval:
	uv run python -m rag_eval --config configs/groq_llama4.yaml eval

compare:
	uv run python -m rag_eval --config configs/groq_llama4.yaml compare

## Run full pipeline end-to-end
benchmark: index run eval compare

## Utility
info:
	uv run python -m rag_eval --config configs/groq_llama4.yaml info

clean:
	rm -rf data/ results/ __pycache__ .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
