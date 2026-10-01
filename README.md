# LLM Agent Template

A production-oriented RAG agent template built on LlamaIndex and FastAPI. Designed to be a reusable starting point for LLM-powered applications with a clean, modular architecture that separates concerns across configuration, ingestion, retrieval, and agent construction.

---

## Features

- Multi-provider LLM support: Ollama (local), OpenAI, Anthropic, Google GenAI
- ChromaDB-backed vector store with environment-isolated persistence
- Document ingestion pipeline with manifest-based change tracking
- FastAPI web interface with streaming chat (SSE), evaluation endpoints, and Jinja2 templates
- LlamaIndex workflow-based agents (`FunctionAgent` / `ReActAgent`) with long-term memory blocks
- RAG evaluation via `FaithfulnessEvaluator`, `RelevancyEvaluator`, and `GuidelineEvaluator`
- Optional judge LLM for LLM-as-evaluator workflows
- Dev / test / prod environment isolation
- CLI with `serve`, `chat`, and `ingest` subcommands

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) (for local models) or API keys for cloud providers
- A running Ollama instance at `http://localhost:11434` if using local models

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/spookymvlder/llm-agent-template.git
cd llm-agent-template

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
# for tests: pip install -r requirements-dev.txt

# 4. Create your .env file
cp .env.example .env
# Edit .env with your settings
```

---

## Configuration

All runtime configuration is controlled via `.env`. See `.env.example` for a full template.

### Minimum configuration (Ollama)

```env
LLM_PROVIDER=ollama
LLM_MODEL=qwen3
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

### Minimum configuration (OpenAI)

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

### Key settings

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama`, `openai`, `anthropic`, `gemini` |
| `LLM_MODEL` | `qwen3` | Model name for the selected provider |
| `LLM_TEMPERATURE` / `LLM_CONTEXT_WINDOW` / `LLM_MAX_TOKENS` / `LLM_RATE_LIMIT_RPM` | `0.2` / `8192` / provider default / unlimited | Primary LLM tuning. Context window applies to Ollama only |
| `ROUTER_*`, `JUDGE_*` | fall back to `LLM_*` | Same suffixes as `LLM_*` (`_PROVIDER`, `_MODEL`, `_TEMPERATURE`, ...). The judge (evaluation) is disabled unless `JUDGE_MODEL` is set |
| `ENABLE_ROUTER` | `false` | Classify queries with the router LLM before running an agent |
| `EMBEDDING_PROVIDER` | `huggingface` | `huggingface` or `ollama` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model name |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `50` | Splitter settings, in tokens |
| `USE_CUDA` | `false` | Run HuggingFace embeddings on GPU if available |
| `COLLECTIONS` | `documents` | Comma-separated collection names; the first is the default |
| `COLLECTION_<NAME>_RAW_DIR` / `_TOP_K` / `_DESCRIPTION` | `data/raw/<name>` / `RETRIEVAL_TOP_K` / generic | Per-collection overrides. The description tells the agent what the collection contains |
| `ENVIRONMENT` | `dev` | `dev`, `test`, or `prod` (dev enables hot reload) |
| `AUTO_INGEST` | `true` | Ingest new (and, if enabled, changed) files on startup |
| `REINDEX_CHANGED_FILES` | `true` | Re-embed files whose content hash changed since ingestion; `false` only logs them |
| `RETRIEVAL_TOP_K` | `5` | Number of chunks retrieved per query |
| `MAX_ITERATIONS` | `3` | Agent reasoning loop cap |
| `MEMORY_TOKEN_LIMIT` | `4096` | Total token budget for short + long term memory |
| `ENABLE_FACT_EXTRACTION` | `true` | Extract facts from conversation into long-term memory |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `HF_TOKEN` | _(unset)_ | HuggingFace token (suppresses rate limit warnings) |
| `LOG_LEVEL` | `INFO` | Python logging level |

Configuration is parsed when `src.config` is imported and validated at startup; missing API keys and invalid values are all reported together.

---

## Usage

### Start the web server

```bash
python -m src.main serve
# Navigate to http://localhost:8000
```

### Interactive CLI chat

```bash
python -m src.main chat
```

### Ingest documents without starting the server

```bash
python -m src.main ingest                      # all collections
python -m src.main ingest -c rules             # one collection
python -m src.main ingest -c rules --reindex   # delete and re-embed (after changing embedding/chunk settings)
```

Place documents in `data/raw/<collection>/` (by default `data/raw/documents/`) before ingesting. Supported formats: `.txt`, `.md`, `.pdf`, `.html`, `.htm`, `.json`, `.csv`.

---

## Project Structure

```
src/
  agent_setup/
    agent_factory.py        # build_rag_agent(): one search tool per collection
    agent_tools.py          # Generic tools (datetime, calculator, list documents)
    memory_factory.py         # Build Memory with optional memory blocks
  indexing/
    chroma_index_manager.py # ChromaDB-backed VectorStoreIndex for one collection
    collections.py          # CollectionHandle: store + files + index + postprocessors
    index_manager.py        # File discovery, manifest tracking
  llm/
    llamaindex_setup.py     # Configures LlamaIndex Settings globals
    llm_factory.py          # Builds LLM instances from LlmSettings
  app_context.py            # AppContext (what bootstrap builds) and Profile
  config.py                 # AppConfig dataclass, loaded from .env
  config_helpers.py         # Coercion helpers, settings dataclasses
  evaluation.py             # EvaluatorBundle, evaluate_response()
  fastapi_app.py            # FastAPI app, routes, SSE streaming
  logging_setup.py          # Logging configuration
  main.py                   # CLI entry point (serve / chat / ingest)
  models.py                 # Pydantic models for API schemas
  providers.py              # LLMProvider and EmbeddingProvider enums
  startup.py                # bootstrap(profile) — builds an AppContext

data/
  raw/<collection>/         # Drop documents here for ingestion
  dev/                      # Dev environment data (chroma/, manifests/<collection>.json)
  test/
  prod/

templates/
  index.html                # Web UI (chat, evaluate, chat+eval tabs)
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Liveness check (does not call the LLM) |
| `POST` | `/chat` | Non-streaming chat (JSON response) |
| `POST` | `/chat/stream` | Streaming chat via SSE |
| `POST` | `/evaluate` | Evaluate a question/answer pair |
| `POST` | `/chat_and_evaluate` | Chat then evaluate the response |
| `POST` | `/clear` | Reset conversation memory |
| `GET` | `/docs` | Auto-generated Swagger UI |

---

## Adding Documents

Place any supported file in `data/raw/<collection>/`. On next startup (when `AUTO_INGEST=true`) or by running `ingest`, new files are chunked, embedded, and stored in ChromaDB. Files are never moved or deleted — a manifest per collection records each file's SHA-256 (plus size and modification time, so unchanged files aren't re-hashed).

When a file's content changes, its old chunks are replaced on the next ingest (set `REINDEX_CHANGED_FILES=false` to only log changed files). Files removed from disk are reported but their chunks are kept; reindex the collection to drop them:

```bash
python -m src.main ingest -c documents --reindex
```

Each document gets a stable id (`<path relative to the collection folder>#<position in file>`), so re-ingesting never duplicates it.

### Collections

`COLLECTIONS=rules,transcripts` creates two collections, each with its own folder, search tool and optional `COLLECTION_<NAME>_DESCRIPTION` (which tells the agent when to search it). Use separate collections for different kinds of content; use metadata for variations within one kind (e.g. rulebook edition).

### Retrieval postprocessors

Per-collection LlamaIndex node postprocessors run after retrieval and before the LLM sees the chunks — the place for deterministic handling based on metadata:

```python
ctx = bootstrap(Profile.SERVE, postprocessors={"rules": [MyEditionPostprocessor()]})
```

---

## Adding Tools

Add tool functions to `src/agent_setup/agent_tools.py`. Tools with no external dependencies are plain functions; tools that need a collection use factory functions that close over it:

```python
def _make_my_tool(collection: CollectionHandle):
    async def my_tool(query: str) -> str:
        """Describe what this tool does — the LLM reads this."""
        return str(await collection.as_query_engine().aquery(query))
    return my_tool
```

Register it in `build_generic_tools()` and it will be passed to the agent as `extra_tools` via `startup.py`.

---

## Evaluation

If `JUDGE_PROVIDER` and `JUDGE_MODEL` are set, the evaluator is enabled. The judge LLM should differ from the primary LLM to avoid a model evaluating its own outputs.

```env
JUDGE_PROVIDER=openai
JUDGE_MODEL=gpt-4o
```

Evaluation runs faithfulness, relevancy, and guideline checks against each response. Results are accessible via `/evaluate` and `/chat_and_evaluate`, and structured as:

```json
{
  "passing": true,
  "faithfulness": { "passing": true, "score": 1.0, "feedback": "..." },
  "relevancy":    { "passing": true, "score": 1.0, "feedback": "..." },
  "guidelines":   [{ "passing": true, "feedback": "..." }]
}
```

---

## Environment Isolation

Each environment (`dev`, `test`, `prod`) maintains its own ChromaDB collection and ingestion manifest under `data/{env}/`. Raw documents in `data/raw/` are shared across environments. Set `ENVIRONMENT=prod` in your deployment environment to use the production data directory.

---

## Dependencies

Core dependencies. See `requirements.txt` for pinned versions.

- `llama-index-core` — agent workflows, vector store index, evaluation
- `llama-index-vector-stores-chroma` — ChromaDB integration
- `llama-index-llms-ollama / openai / anthropic / google-genai` — LLM providers
- `llama-index-embeddings-huggingface / ollama` — embedding models
- `chromadb` — vector database
- `fastapi` + `uvicorn` — web server
- `sentence-transformers` — HuggingFace embedding backend
- `python-dotenv` — environment variable loading