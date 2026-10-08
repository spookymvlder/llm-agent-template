# LLM Agent Template

A production-oriented RAG agent template built on LlamaIndex and FastAPI. Designed to be a reusable starting point for LLM-powered applications with a clean, modular architecture that separates concerns across configuration, ingestion, retrieval, and agent construction.

---

## Features

- Multi-provider LLMs: Ollama (local), OpenAI, Anthropic, Google GenAI — separately configurable for the agent, an optional router and an evaluation judge, with rate limits and Ollama thinking control
- Multiple named ChromaDB collections, each with its own folder, search tool and description
- File ingestion with SHA-256 change detection, folder-level `_metadata.json`, static/manual update modes, and tabular (CSV/JSONL/Parquet) collections
- Runtime ingestion: append-only **streams** (e.g. live transcripts) searchable immediately and re-chunked on close, plus ad-hoc documents
- Search tools return chunks *with metadata*; every chat response lists its `sources`
- Optional LLM router with pluggable routes (agents, fixed replies, or any custom handler)
- Per-collection preprocessing steps and retrieval postprocessors
- FastAPI app with streaming chat (SSE), per-conversation memory, retrieval/ingest/stream/summary endpoints and a small web UI
- Golden-set evaluation harness (route accuracy, retrieval metrics, judged answers, run-to-run deltas; optional RAGAS)
- Dev / test / prod environment isolation; CLI with `serve`, `chat`, `ingest` and `eval`
- Test suite that runs offline (mock embedder and scripted LLMs)

## How it fits together

```
.env → AppConfig (config.py) ──► bootstrap(profile) (startup.py) ──► AppContext (app_context.py)
                                                                      ├─ collections: {name: CollectionHandle}
data/raw/<collection>/  ── IndexManager (files, manifest) ─┐         │    store (Chroma) · files · streams
data/<env>/streams/     ── StreamManager (JSONL) ──────────┼─► sync ─┤    preprocess → embed → search
POST /documents ───────────────────────────────────────────┘         ├─ agents: {"rag", "direct", ...}
                                                                      ├─ router (optional) · evaluator (judge)
                                                                      └─ conversations (memory per id)
FastAPI (fastapi_app.py) / CLI (main.py) ── chat.respond(): router → route handler → agent → answer + sources
```

`bootstrap()` builds only what a profile needs: `INGEST` (collections, no LLM), `CHAT`, `SERVE` (adds the evaluator), `EVAL`. Everything it builds lives on the `AppContext`, which FastAPI keeps on `app.state.ctx`.

### Extension points

| To... | Use |
|---|---|
| Add a corpus | `COLLECTIONS=...` + `COLLECTION_<NAME>_DESCRIPTION` |
| Tag documents | `_metadata.json` in folders, `metadata` on `/documents` and `/streams`, or a preprocessing step |
| Read a table as documents | `CollectionOptions(schema=DocumentSchema(tabular=True, ...))` |
| Transform rows before embedding | `CollectionOptions(steps=[fn])` |
| Act on retrieved chunks (e.g. by metadata) | `CollectionOptions(postprocessors=[...])`, or branch on `sources` in your code |
| Add a tool | `agent_tools.py`, passed to `build_rag_agent(extra_tools=...)` |
| Add a specialised agent | `build_rag_agent([ctx.collection("rules")])` → `ctx.agents["rules"]` |
| Add a route | `bootstrap(routes=[...RouteSpec(...)])` with `ENABLE_ROUTER=true` |
| Measure a change | `python -m src.main eval golden.json --compare <previous run>` |

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
# for RAGAS in eval runs: pip install -r requirements-eval.txt

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
| `LLM_THINKING` | model default | Ollama reasoning models: `true`/`false` turns thinking on/off |
| `ROUTER_*`, `JUDGE_*` | fall back to `LLM_*` | Same suffixes as `LLM_*` (`_PROVIDER`, `_MODEL`, `_TEMPERATURE`, `_THINKING`, ...). The judge (evaluation) is disabled unless `JUDGE_MODEL` is set |
| `ENABLE_ROUTER` | `false` | Classify queries with the router LLM before running an agent |
| `ROUTER_MIN_CONFIDENCE` / `ROUTER_DEFAULT_ROUTE` | `0.5` / `rag` | Below this confidence (or on unusable output) the default route is used |
| `EMBEDDING_PROVIDER` | `huggingface` | `huggingface` or `ollama` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model name |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `50` | Splitter settings, in tokens |
| `USE_CUDA` | `false` | Run HuggingFace embeddings on GPU if available |
| `COLLECTIONS` | `documents` | Comma-separated collection names; the first is the default |
| `COLLECTION_<NAME>_RAW_DIR` / `_TOP_K` / `_DESCRIPTION` | `data/raw/<name>` / `RETRIEVAL_TOP_K` / from `_metadata.json` | Per-collection overrides. The description is better kept in the collection's `_metadata.json` (see Collections); the `.env` value overrides it |
| `ENVIRONMENT` | `dev` | `dev`, `test`, or `prod` (dev enables hot reload) |
| `AUTO_INGEST` | `true` | Ingest new (and, if enabled, changed) files on startup |
| `DEFAULT_CHANGE_MODE` | `static` | `static`: new/changed files embed on every ingest; `manual`: they wait for `ingest --manual`. Per folder: `{"_mode": ...}` in `_metadata.json` |
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
python -m src.main ingest -c rules --reindex   # delete and re-embed files + streams (after changing embedding/chunk settings)
python -m src.main ingest --manual             # also apply pending changes in manual folders
```

Stop the server before `--reindex`: it deletes and recreates the collections, and a running server keeps handles to the old ones. A reindex re-reads manual folders as they are on disk, drafts included. (Plain `ingest` and `POST /ingest` are safe while the server runs.)

Place documents in `data/raw/<collection>/` (by default `data/raw/documents/`) before ingesting. Supported formats: `.txt`, `.md`, `.pdf`, `.html`, `.htm`, `.json`, `.csv`.

PDFs are read with PyMuPDF (in `requirements.txt`), which keeps word spacing on typeset layouts where LlamaIndex's default reader glues words together (`Oneofeacharmourtype...`) — text like that embeds poorly and retrieval quietly suffers. Ingest logs a warning when a file's extracted text looks glued; check with the Retrieve tab or `POST /retrieve` that chunks read naturally.

---

## Project Structure

```
src/
  agent_setup/
    agent_factory.py        # build_rag_agent(): one search tool per collection
    agent_tools.py          # Generic tools (datetime, calculator, list documents)
    agent_runner.py         # stream_agent() / run_agent(): deltas, tool calls, answer + sources
    memory_factory.py       # build_memory(), ConversationStore (memory per conversation id)
    router.py               # QueryRouter, RouteSpec, default routes
    summarize.py            # summarize_documents(): tree-summarise chunks matching a filter
  indexing/
    chroma_index_manager.py # ChromaDB-backed VectorStoreIndex for one collection
    collections.py          # CollectionHandle: store + files + index + postprocessors
    index_manager.py        # File discovery, change detection, _metadata.json, static/manual modes
    streams.py              # Stream files (append-only JSONL) and their status
  preprocessing/
    pipeline.py             # preprocess(df, schema, steps)
    sanitizer.py            # Text normalisation, Chroma-safe metadata, dedupe
  llm/
    llamaindex_setup.py     # Configures LlamaIndex Settings globals
    llm_factory.py          # Builds LLM instances from LlmSettings
    parsing.py              # Tolerant JSON parsing of LLM output
  app_context.py            # AppContext (what bootstrap builds) and Profile
  chat.py                   # respond() / answer(): route (if enabled) then run the handler
  config.py                 # AppConfig dataclass, loaded from .env
  config_helpers.py         # Coercion helpers, settings dataclasses
  evaluation/
    live.py                 # EvaluatorBundle, evaluate_response() / evaluate_answer()
    harness.py              # Golden-set runs: route, retrieval, answer, judge → report
    metrics.py              # Hit rate, precision, recall, MRR, nDCG
    ragas_eval.py           # Optional RAGAS metrics
  fastapi_app.py            # FastAPI app, routes, SSE streaming
  logging_setup.py          # Logging configuration
  main.py                   # CLI entry point (serve / chat / ingest / eval)
  models.py                 # Pydantic models for API schemas
  providers.py              # LLMProvider and EmbeddingProvider enums
  schema.py                 # DocumentSchema: text/metadata columns, tabular mode, metadata visibility
  startup.py                # bootstrap(profile) — builds an AppContext

data/
  raw/<collection>/         # Drop documents here for ingestion (shared by all environments)
  dev/                      # Dev environment data (chroma/, manifests/, streams/, evaluation/)
  test/
  prod/

templates/
  index.html                # Web UI (chat, retrieve, evaluate, chat+eval tabs)
examples/
  golden.example.json       # Golden-set format for `python -m src.main eval`
tests/                      # Offline pytest suite (mock embedder, scripted LLMs)
requirements.txt            # + requirements-dev.txt (tests), requirements-eval.txt (RAGAS)
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI (chat, retrieve, evaluate tabs) |
| `GET` | `/health` | Liveness: the process is up |
| `GET` | `/ready` | What was built: collections with chunk counts, evaluator, active conversations |
| `POST` | `/chat` | Chat. Returns `response`, `conversation_id` and `sources` |
| `POST` | `/chat/stream` | Chat via SSE: `delta`, `tool_call`, `sources`, `done` (with `conversation_id`), `error` events |
| `POST` | `/clear` | Forget a conversation (`{"conversation_id": ...}`) |
| `POST` | `/retrieve` | Search a collection directly: chunks, scores, metadata. Optional `filters`, `top_k` |
| `POST` | `/documents` | Add or replace documents in a collection at runtime (not kept across a reindex) |
| `POST` | `/streams/{collection}/{stream_id}` | Append a fragment to a stream; embedded immediately |
| `POST` | `/streams/{collection}/{stream_id}/close` | Close a stream (re-chunks it by default) |
| `GET` | `/streams/{collection}` | List a collection's streams |
| `POST` | `/ingest` | Sync collection folders now; `include_manual` applies pending manual changes |
| `POST` | `/route` | Classify a message with the router without answering it (router enabled) |
| `POST` | `/summarize` | Summarise every chunk matching `filters`, in order (e.g. `{"stream_id": "session-12"}`) |
| `POST` | `/evaluate` | Evaluate a supplied answer against retrieved context (needs `JUDGE_MODEL`) |
| `POST` | `/chat_and_evaluate` | Answer with a query engine, then evaluate it (needs `JUDGE_MODEL`) |
| `GET` | `/docs` | Auto-generated Swagger UI |

**Conversations.** Send `conversation_id` from a previous response to continue a conversation; omit it to start a new one. Histories live in server memory (lost on restart), capped by `MAX_CONVERSATIONS` and expired after `CONVERSATION_TTL_S` idle seconds.

**Sources.** The agent's search tools return chunks (not a pre-written summary), and every chunk the agent was given is returned as `sources` with its collection, document id, text, score and metadata — so calling code can act on metadata (e.g. `edition`) directly.

**Runtime documents.** `POST /documents`:

```json
{"collection": "transcripts",
 "documents": [{"id": "s12-0042", "text": "GM: The dragon flees north.", "metadata": {"session": 12}}]}
```

Documents go through the collection's preprocessing steps; re-posting an id replaces it. They have no source file, so `ingest --reindex` deletes them — use a stream, or a file in `data/raw/<collection>/`, for anything that must survive a reindex.

**Errors.** In the `dev` environment error responses include the exception message; otherwise they say to check the server logs.

---

## Adding Documents

Place any supported file in `data/raw/<collection>/`. On next startup (when `AUTO_INGEST=true`) or by running `ingest`, new files are chunked, embedded, and stored in ChromaDB. Files are never moved or deleted — a manifest per collection records each file's SHA-256 (plus size and modification time, so unchanged files aren't re-hashed).

When a file's content changes, its old chunks are replaced on the next ingest. Files removed from disk are reported but their chunks are kept; reindex the collection to drop them:

```bash
python -m src.main ingest -c documents --reindex
```

Each document gets a stable id (`<path relative to the collection folder>#<position in file>`), so re-ingesting never duplicates it.

### Collections

`COLLECTIONS=rules,transcripts` creates two collections, each with its own folder and search tool. Use separate collections for different kinds of content; use metadata for variations within one kind (e.g. rulebook edition).

Give each collection a description: it is part of the collection's search tool and is how the agent decides whether, and where, to search. Without one the agent may answer from general knowledge instead. Put it in the collection's root `_metadata.json` (a collection with no files, like one filled by streams, can still have its folder and this file):

```json
// data/raw/rules/_metadata.json
{"_description": "Rules of Mythic Bastionland, a tabletop RPG about knights: the Knights and their abilities, Seers, combat, travel and Realms."}
```

It is read at startup (restart to apply an edit; nothing is re-embedded). `COLLECTION_<NAME>_DESCRIPTION` in `.env` overrides it, e.g. to try out wording. Startup logs a warning for collections without one.

### Static and manual folders

Files are **static** by default: new and changed files are embedded on every ingest, including startup. Mark a folder **manual** for files you edit by hand and don't want picked up half-finished:

```json
{"_mode": "manual"}
```

Changes there are logged as pending until you apply them with `python -m src.main ingest --manual` or `POST /ingest {"include_manual": true}`. `DEFAULT_CHANGE_MODE` sets the default for folders that don't say.

Keys starting with `_` configure ingestion and are never stored as metadata: `_mode` (any folder) and `_description` (collection root only). Unknown `_` keys are logged, so a typo doesn't fail silently.

### Streams (e.g. live transcripts)

For text that arrives in pieces while the app runs:

```
POST /streams/transcripts/session-12          {"text": "GM: The dragon flees north.", "metadata": {"session": 12}}
POST /streams/transcripts/session-12          {"text": "Alice: I follow it.", "metadata": {"session": 12}}
POST /streams/transcripts/session-12/close    {"metadata": {"date": "2026-10-02"}}
```

Each fragment is appended to `data/<env>/streams/<collection>/<stream_id>.jsonl` (per environment, so test sessions don't reach prod) and embedded at once — searchable immediately. Closing re-chunks by default: the fragment chunks are replaced by the whole stream as one document (carrying the close metadata plus any metadata every fragment shared), which retrieves better than many short fragments; pass `"rechunk": false` to keep the fragments. A reindex replays every stream the way it was last stored, so it reproduces the live index. `GET /streams/<collection>` lists streams.

### Folder metadata

A `_metadata.json` object in any folder under `data/raw/<collection>/` is added to every file in that folder and below (nearer folders override). Editing it re-embeds the files it covers.

```
data/raw/rules/
  2014/_metadata.json   {"edition": "2014"}
  2014/PHB.pdf
  2024/_metadata.json   {"edition": "2024"}
  2024/PHB.pdf
```

Metadata is shown to the LLM with each retrieved chunk and can be used in filters, but is **not** embedded by default, so tags don't skew similarity. File bookkeeping (hashes, sizes, dates) is hidden from the LLM.

### Per-collection options (schema, preprocessing, postprocessors)

Code-level customisation is passed to `bootstrap()` per collection:

```python
from src.indexing import CollectionOptions
from src.schema import DocumentSchema

def tag_speaker(df):                     # preprocessing step: DataFrame -> DataFrame
    df["speaker"] = df["text"].str.extract(r"^(\w+):")
    return df

options = {
    "rules": CollectionOptions(postprocessors=[MyEditionPostprocessor()]),
    "papers": CollectionOptions(
        schema=DocumentSchema(tabular=True, id_column="id", text_columns=("title", "abstract")),
    ),
    "transcripts": CollectionOptions(steps=[tag_speaker]),
}
ctx = bootstrap(Profile.SERVE, options=options)
```

- **`DocumentSchema`** — which columns are text, which metadata is embedded (`embed_metadata_keys`) or hidden from the LLM, and `tabular=True` to read `.csv` / `.jsonl` / `.parquet` (needs `pyarrow`) as one document per row.
- **`steps`** — run before sanitisation, which normalises whitespace, drops empty rows and duplicate ids, and coerces metadata (lists, dicts, enums, NaN) to Chroma-safe values.
- **`postprocessors`** — LlamaIndex node postprocessors run after retrieval and before the LLM sees the chunks: the place for deterministic handling based on metadata.

Changing a schema or steps doesn't re-embed existing documents; run `ingest -c <name> --reindex`.

---

## Router (optional)

With `ENABLE_ROUTER=true`, each message is first classified by the router LLM (`ROUTER_*` settings — a small, fast model with `ROUTER_THINKING=false` works well), then handled by the chosen route. The default routes are `rag` (agent with the search tools), `direct` (no tools: small talk, general knowledge) and `clarify` (asks the user to rephrase). Unparseable answers, unknown routes and confidence below `ROUTER_MIN_CONFIDENCE` fall back to `ROUTER_DEFAULT_ROUTE`. The decision is returned as `route` in chat responses and as the first SSE event; `POST /route` classifies without answering.

A route is a name, a description the router reads, and an async-generator handler. Handlers can run any agent, reply with a fixed message, or do anything else:

```python
from src.agent_setup import AgentFinished, RouteSpec, agent_route, build_rag_agent, default_routes

open_questions: list[str] = []

async def note_question(rc):              # rc: RouteContext (ctx, message, memory, decision)
    open_questions.append(rc.message)
    yield AgentFinished(response="Added to the open questions list.")

routes = [
    agent_route("rules", "Questions about game rules.", agent_name="rules"),
    agent_route("lore", "Questions about the world, characters or past sessions.", agent_name="lore"),
    RouteSpec("offtopic", "Questions unrelated to the game.", note_question),
]
ctx = bootstrap(Profile.SERVE, routes=routes)       # with ROUTER_DEFAULT_ROUTE=rules
ctx.agents["rules"] = build_rag_agent([ctx.collection("rules")])
ctx.agents["lore"] = build_rag_agent([ctx.collection("transcripts"), ctx.collection("notes")])
```

## Summaries

`summarize_documents(collection, filters)` (and `POST /summarize`) reads every chunk matching exact-match metadata filters in document order — not a similarity search — and tree-summarises them, so input isn't limited by the context window. For end-of-session notes: `{"collection": "transcripts", "filters": {"stream_id": "session-12"}}`.

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

If `JUDGE_MODEL` is set (optionally `JUDGE_PROVIDER`), the evaluator is enabled. The judge LLM should differ from the primary LLM to avoid a model evaluating its own outputs.

```env
JUDGE_PROVIDER=openai
JUDGE_MODEL=gpt-4o
```

### Golden-set regression runs

Write a golden set — questions with whatever you know about the right answer — and run it after changing a model, chunking, prompt or route to see whether things got better or worse:

```bash
python -m src.main eval examples/golden.example.json --name baseline
python -m src.main eval examples/golden.example.json --name smaller-chunks --compare baseline
python -m src.main eval golden.json --no-answer       # route + retrieval only: fast, no agent runs
python -m src.main eval golden.json --ragas           # + RAGAS (pip install -r requirements-eval.txt)
```

Each case needs only a `question`; each check runs when its field is present (see `examples/golden.example.json`):

| Field | Check |
|---|---|
| `expected_route` | Router accuracy (router enabled) |
| `expected_sources` (+ `collection`, `filters`) | Retrieval hit rate, precision, recall, MRR, nDCG @k — the collection is searched directly, so it measures retrieval, not the agent's tool choice. Sources match doc ids, source paths (`2024/PHB.pdf` matches all its chunks) or stream ids |
| always (unless `--no-answer`) | Full pipeline answer (fresh conversation per case), latency, sources used; with a judge: faithfulness, relevancy, guideline pass rate |
| `reference_answer` | Judge correctness score (1–5) |

Reports go to `data/<env>/evaluation/<run>/report.json` and `report.md` (settings, aggregate with deltas against `--compare`, one row per case, retrieval misses). `EVAL_CONCURRENCY` runs cases in parallel; judge calls respect `JUDGE_RATE_LIMIT_RPM`. RAGAS (faithfulness, response relevancy, context precision) needs a capable judge — small local models often fail its structured output; failed samples are counted, not fatal.

### Live evaluation

The `/evaluate` and `/chat_and_evaluate` endpoints run faithfulness, relevancy, and guideline checks against a single response, structured as:

```json
{
  "passing": true,
  "faithfulness": { "passing": true, "score": 1.0, "feedback": "..." },
  "relevancy":    { "passing": true, "score": 1.0, "feedback": "..." },
  "guidelines":   [{ "passing": true, "feedback": "..." }]
}
```

---

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The suite runs offline in a temporary data directory: LlamaIndex's mock embedder replaces the embedding model, and scripted mock LLMs stand in for the agent, router and judge, so no Ollama instance, model download or API key is needed. `tests/test_api.py` drives the real FastAPI app with a hand-built `AppContext`; use the same pattern (override `bootstrap`, or the `get_ctx` dependency) to test project routes.

---

## Environment Isolation

Each environment (`dev`, `test`, `prod`) maintains its own ChromaDB collections, ingestion manifests, stream files and evaluation reports under `data/{env}/`. Raw documents in `data/raw/` are shared across environments. Set `ENVIRONMENT=prod` in your deployment environment to use the production data directory.

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