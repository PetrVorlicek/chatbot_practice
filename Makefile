.PHONY: install dev seed-docs

install:
	uv sync

seed-docs: install
	uv run python -m ai.seed_semantic_docs

dev: install
	uv run uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
