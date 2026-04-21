.PHONY: install dev

install:
	uv sync

dev: install
	uv run uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
