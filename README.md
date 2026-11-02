# Empório da Música — Customer Support Agent

A Portuguese-language conversational agent designed to support customer service for the fictional Empório da Música store. It answers questions about the catalog, prices, availability, orders, store information, and policies using the sources provided in Artefact's technical challenge.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally
- `qwen3.5:9b` chat model and `nomic-embed-text` embedding model

## Setup

From the `emporio-musica-agent/` directory, install the dependencies:

```powershell
uv sync
```

Download the models (once):

```powershell
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
```

The defaults are `http://localhost:11434` and `qwen3.5:9b`. To change them, set `OLLAMA_BASE_URL` and/or `OLLAMA_MODEL` in the environment or in a `.env` file at the project root:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.5:9b
```

## Policy indexing

The `data/políticas_da_loja.pdf` handbook is extracted, cleaned, split into overlapping chunks, and stored locally in Chroma. Generate or update the index before starting the API:

```powershell
uv run python scripts/ingest_policies.py
```

The script runs retrieval checks for important rules. The first indexing run may take a few minutes because embeddings are computed locally. The database is stored in `data/chroma/`.

## Run the API

```powershell
uv run uvicorn emporio_musica_agent.main:app --host 127.0.0.1 --port 8000
```

Interactive documentation: <http://127.0.0.1:8000/docs>. Health check: `GET /health`.

Example conversation:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/chat `
  -ContentType 'application/json' `
  -Body '{"session_id":"customer-001","message":"What acoustic guitars are available for up to R$1000?"}'
```

Each request requires `session_id` and `message`. Messages with the same identifier share in-memory history for the lifetime of the process. The API returns `{ "answer": "..." }`.

To use the chat in the terminal through the CLI instead of the API:

```powershell
uv run emporio-musica-agent
```

## Technical decisions

### Framework(s) and agent approach

I used LangChain `create_agent` to coordinate tool calls (function calling) in a ReAct flow: the model interprets the question, selects the required sources, and drafts a response based on the results. The architecture is hybrid: structured queries against the CSV files handle the catalog and orders, while RAG retrieves excerpts from the PDF for policies and store information. Questions that depend on a deadline can also call the current-date tool.

### Model and provider

The chat model is Qwen 3.5 9B, running locally through Ollama with zero temperature. This choice requires no API key and avoids sending conversations to an external provider, at the cost of depending on the machine's memory and processing capacity. For RAG embeddings, I use `nomic-embed-text`, also through Ollama.

### Interaction interface

The primary interface is an HTTP API built with FastAPI: `POST /chat` receives `session_id` and `message`, and `GET /health` checks the service. FastAPI also exposes OpenAPI documentation at `/docs`. An optional CLI reuses the agent and allows conversations in the terminal.

### Conversation history persistence

History is kept in memory, separated by `session_id`, with up to 12 final messages (six turns) per session. This keeps context during a conversation without adding database infrastructure to the prototype. History is lost when the process restarts and is not shared across instances; durable persistence and synchronization would be next steps for production use by multiple users.

### Data handling

Pandas loads the operational CSV files. Product search considers active products with positive stock, combines names, descriptions, and categories to find items, applies category and price filters, and calculates promotional prices based on active promotions. Orders are associated with their respective items, and statuses are presented in Portuguese.

For policies, the PDF is extracted and its text cleaned, split into 1,200-character chunks with a 200-character overlap, and indexed locally in Chroma. Semantic search retrieves relevant excerpts for the agent; the ingestion script includes checks for important rules.

A tool returns the current date in the `America/Sao_Paulo` time zone for questions requiring deadline comparisons. Under the remorse policy, the deadline is seven calendar days after receipt; the agent does not assume this starting point when the receipt date has not been provided. The prompt instructs the agent to consult tools for store-specific facts, avoid inventing information, and use a fallback response when the sources are insufficient.

## Interaction examples

The five records in [`examples/`](examples/) document conversations made through the API, including the requests and responses returned by the agent. They cover the catalog, pricing, store information, an order, and application of the remorse policy.

## Limitations and next steps

- Data and the catalog are static files; there is no connection to a production inventory, order, or payment system.
- History is lost on restart, and sessions are not persisted or synchronized for concurrent access.
- RAG quality depends on PDF extraction, the local model, and semantic retrieval. Responses do not replace human review in ambiguous cases.
- In the recorded catalog run, the response also mentioned a ukulele as a false positive. A possible next adjustment is to require the structured category filter and limit the response to records returned by that filter.
- There is no authentication, customer-level authorization, observability, automated response evaluation, or production deployment.
- With more time: integrate real transactional sources with access controls, persist sessions, create an evaluation set for routing/retrieval/policies, and add monitoring and load testing.

## Use of coding assistants

I used Codex as a coding assistant to help read the challenge brief and repository, structure this documentation, and run checks and API calls. I reviewed the decisions and changes in the context of the provided code and data.
