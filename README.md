# Chatbot

An AI support assistant built to handle customer conversations through a simple REST API.

This is a demo project using a Warhammer 40,000-inspired knowledge base (Space Marine weaponry) as the documentation corpus — a fun way to make the sample data a bit more interesting than lorem ipsum.

## Vision

- LLM-powered support replies
- Documentation search for grounded answers
- Ticket creation as a fallback path

## Roadmap

- Semantic search over support documentation
- Support ticket model and validation
- Agentic routing for different issue types
- Human-in-the-loop feedback flow
- Lightweight frontend chat UI

## Run locally

```bash
make seed-docs
make dev
```

The development server runs with hot reload enabled.

Server default: `http://localhost:8000`

## Semantic search over support documentation

In this demo, vectors are stored as BLOBs in SQLite. At query time, vectors are loaded into memory and ranked with cosine similarity using `numpy`.

### Seed documentation vectors

```bash
make seed-docs
```

This command:

- reads `documents/*.txt`
- chunks text with overlap
- calls the embeddings endpoint
- stores text + vector blobs in `ai/semantic_docs.sqlite3`

### Test the chat endpoint

```bash
curl -X POST "http://localhost:8000/chat" \
	-H "Content-Type: application/json" \
	-d '{"user_input":"What is a bolter and when should it be used?"}'
```

## Simple support tickets store (SQLite)

There is a small Python ticket store in `db/ticket_store.py` with:

- `id` (autoincrement primary key)
- `title`
- `description`
- `status` (enum: `open`, `in_progress`, `resolved`, `closed`)
- `created_at` (auto timestamp)
