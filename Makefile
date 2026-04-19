.PHONY: install dev

install:
	uv sync

dev: install
	uv run python main.py
