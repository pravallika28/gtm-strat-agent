# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
cd tools
pip install -r requirements.txt
pip install langgraph llama-index llama-index-llms-anthropic
```

`langgraph`, `llama-index`, and `llama-index-llms-anthropic` are imported in source but missing from `requirements.txt`.

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_key_here
```

The retriever expects a `./data` directory containing GTM-related documents. This directory is not committed to the repo and must be created manually.

## Running

```bash
# Run the agent
python tools/main.py

# Query documents via retriever
python -c "from tools.retriever import query_gtm_docs; print(query_gtm_docs('your query'))"
```

## Architecture

The project has two independent entry points that both use Claude claude-3-5-sonnet:

**`tools/main.py` — Conversational agent**
- Defines `AgentState` (TypedDict with message history) and a LangGraph workflow
- Single node (`researcher`) calls Claude via `ChatAnthropic` and appends the response to state
- Graph: `START → researcher → END`; exposed as `app = workflow.compile()`

**`tools/retriever.py` — RAG over local documents**
- Loads documents from `./data` using LlamaIndex `SimpleDirectoryReader`
- Builds a vector index and wraps it in a query engine backed by Claude
- Exposes `query_gtm_docs(query: str)` for semantic search + generation

The two components are not wired together yet — `main.py` is a pure chat agent with no document retrieval, and `retriever.py` is a standalone RAG interface.

## Code Formatting & Linting

`black` and `ruff` are in `requirements.txt`:

```bash
black tools/
ruff check tools/
```
