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
make dev
```

The development server runs with hot reload enabled.

Server default: `http://localhost:8000`

# Semantic search over support documentation
In this demo, we store the vectors as BLOBs in a standard SQLite table. When a query comes in, we pull the vectors into memory and calculate the similarity using *numpy* - the gold standard of Python math libraries.
