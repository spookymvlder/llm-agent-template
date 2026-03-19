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
| `EMBEDDING_PROVIDER` | `huggingface` | `huggingface` or `ollama` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model name |
| `ENVIRONMENT` | `dev` | `dev`, `test`, or `prod` |
| `AUTO_INGEST` | `true` | Ingest raw documents on startup if index is empty |
| `RETRIEVAL_TOP_K` | `5` | Number of documents retrieved per query |
| `MAX_ITERATIONS` | `5` | Agent reasoning loop cap |
| `MEMORY_TOKEN_LIMIT` | `40000` | Total token budget for short + long term memory |
| `ENABLE_FACT_EXTRACTION` | `true` | Extract facts from conversation into long-term memory |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `JUDGE_PROVIDER` | _(unset)_ | Provider for the evaluator LLM |
| `JUDGE_MODEL` | _(unset)_ | Model for the evaluator LLM |
| `HF_TOKEN` | _(unset)_ | HuggingFace token (suppresses rate limit warnings) |
| `DEBUG` | `false` | Enables uvicorn hot reload |
| `LOG_LEVEL` | `INFO` | Python logging level |

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
python -m src.main ingest
```

Place documents in `data/raw/` before ingesting. Supported formats: `.txt`, `.md`, `.pdf`, `.html`, `.htm`, `.json`, `.csv`.

---

## Project Structure

```
src/
  agent_setup/
    agent_factory.py        # build_rag_agent(), build_evaluator_agent()
    agent_tools.py          # Generic tools (datetime, calculator, etc.)
    memory_factory.py         # Build Memory with optional memory blocks
  indexing/
    chroma_index_manager.py # ChromaDB-backed VectorStoreIndex
    index_manager.py        # File discovery, manifest tracking
  llm/
    llamaindex_setup.py     # Configures LlamaIndex Settings globals
    llm_factory.py          # Builds LLM instances from LlmSettings
  config.py                 # AppConfig dataclass, loaded from .env
  config_helpers.py         # Coercion helpers, settings dataclasses
  evaluation.py             # EvaluatorBundle, evaluate_response()
  fastapi_app.py            # FastAPI app, routes, SSE streaming
  logging_setup.py          # Logging configuration
  main.py                   # CLI entry point (serve / chat / ingest)
  models.py                 # Pydantic models for API schemas
  providers.py              # LLMProvider and EmbeddingProvider enums
  startup.py                # bootstrap() — full application startup sequence

data/
  raw/                      # Drop documents here for ingestion
  dev/                      # Dev environment data (chroma/, manifest.json)
  test/
  prod/

templates/
  index.html                # Web UI (chat, analyze, evaluate, chat+eval tabs)
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Agent health check |
| `POST` | `/chat` | Non-streaming chat (JSON response) |
| `POST` | `/chat/stream` | Streaming chat via SSE |
| `POST` | `/analyze` | Analyze a URL |
| `POST` | `/evaluate` | Evaluate a question/answer pair |
| `POST` | `/chat_and_evaluate` | Chat then evaluate the response |
| `POST` | `/clear` | Reset conversation memory |
| `GET` | `/docs` | Auto-generated Swagger UI |

---

## Adding Documents

Place any supported file in `data/raw/`. On next startup (or by running `ingest`), the file will be chunked, embedded, and stored in ChromaDB. Files are never moved or deleted — the manifest tracks what each environment has processed.

To force a full re-ingest, delete the environment's `manifest.json`:

```bash
# Dev environment
rm data/dev/manifest.json
```

---

## Adding Tools

Add tool functions to `src/agent/agent_tools.py`. Tools with no external dependencies are plain functions; tools that need the query engine or ChromaDB client use factory functions that close over the dependency:

```python
def _make_my_tool(query_engine):
    def my_tool(query: str) -> str:
        """Describe what this tool does — the LLM reads this."""
        return str(query_engine.query(query))
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
- `sse-starlette` — server-sent events for streaming
- `sentence-transformers` — HuggingFace embedding backend
- `python-dotenv` — environment variable loading