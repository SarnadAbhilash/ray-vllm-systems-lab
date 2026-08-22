.PHONY: install test lint analyze charts verify

install:
	uv sync --extra dev --extra charts

test:
	uv run pytest --cov=ray_vllm_lab --cov-report=term-missing

lint:
	uv run ruff check .

analyze:
	uv run prefix-cache-lab analyze data/sample/agentic_requests.jsonl \
		--output results/sample/prefix-cache-report.json \
		--markdown results/sample/prefix-cache-report.md

charts:
	uv run python scripts/plot_analyzer.py results/sample/prefix-cache-report.json --output-dir charts
	uv run python scripts/plot_training.py results/phase2/raw/modal-results.json --output-dir charts/phase2

verify: lint test analyze charts
