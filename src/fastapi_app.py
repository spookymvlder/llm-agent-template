from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.templating import Jinja2Templates
from fastapi.sse import EventSourceResponse, ServerSentEvent

from src.agent_setup import AgentDelta, AgentFinished, AgentToolCall, RouteDecision, summarize_documents
from src.agent_setup.summarize import DEFAULT_INSTRUCTION
from src.chat import answer, respond
from src.app_context import AppContext, Profile
from src.config import CONFIG as cfg
from src.evaluation import evaluate_answer, evaluate_response
from src.indexing import CollectionHandle, RetrievedChunk, StreamClosed, StreamError, StreamInfo, StreamNotFound
from src.models import (
    AddDocumentsRequest,
    AddDocumentsResponse,
    ChatEvalRequest,
    ChatEvalResponse,
    ChatRequest,
    ChatResponse,
    ClearRequest,
    CollectionStatus,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    IngestResult,
    ReadyResponse,
    RetrieveRequest,
    RouteInfo,
    RouteRequest,
    SummarizeRequest,
    SummarizeResponse,
    RetrieveResponse,
    Source,
    StreamAppendRequest,
    StreamAppendResponse,
    StreamCloseRequest,
    StreamStatus,
)
from src.startup import bootstrap


log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan & dependencies
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ctx = bootstrap(Profile.SERVE)
    log.info(
        "Startup complete. collections=%s evaluator=%s",
        list(app.state.ctx.collections), "enabled" if app.state.ctx.evaluator_bundle else "disabled",
    )
    yield
    log.info("Shutting down.")


def get_ctx(request: Request) -> AppContext:
    """The AppContext built at startup. Tests can override this dependency with a stub context."""
    return request.app.state.ctx


Ctx = Annotated[AppContext, Depends(get_ctx)]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LLM Agent",
    description="A LlamaIndex-backed LLM agent with FastAPI.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# ---------------------------------------------------------------------------
# Routes — status
# ---------------------------------------------------------------------------

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness: the process is up. Does not touch the LLM or vector store."""
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse)
async def ready(ctx: Ctx):
    """Readiness: what was built at startup, with chunk counts per collection."""
    return ReadyResponse(
        ready=bool(ctx.agents),
        collections=[CollectionStatus(name=h.name, description=h.description, chunks=h.count())
                     for h in ctx.collections.values()],
        evaluator_ready=ctx.evaluator_bundle is not None,
        conversations=len(ctx.conversations) if ctx.conversations is not None else 0,
        routes=list(ctx.router.routes) if ctx.router else None,
    )

# ---------------------------------------------------------------------------
# Routes — chat
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, ctx: Ctx):
    """Non-streaming chat. Useful for programmatic clients and testing."""
    conversation_id, memory = ctx.conversations.get(body.conversation_id)
    try:
        decision, result = await answer(ctx, body.message, memory, cfg.max_iterations)
    except Exception as e:
        raise _internal_error("Chat", e)
    return ChatResponse(
        response=result.response,
        conversation_id=conversation_id,
        sources=_sources(result.sources),
        route=RouteInfo(**decision.model_dump()) if decision else None,
    )


@app.post("/chat/stream", response_class=EventSourceResponse)
async def chat_stream(body: ChatRequest, ctx: Ctx):
    """Streaming chat via SSE. Used by the browser UI.

    Events (data is JSON):
        route      — {"route", "confidence", "reason", "fallback"} first, when the router is enabled
        delta      — a piece of answer text (string)
        tool_call  — {"tool": name, "args": {...}} when the agent calls a tool
        sources    — list of retrieved chunks the agent was given (see Source)
        done       — {"conversation_id": id, "response": full answer}; pass the id back to continue
        error      — error message (string)
    """
    conversation_id, memory = ctx.conversations.get(body.conversation_id)
    try:
        async for event in respond(ctx, body.message, memory, cfg.max_iterations):
            if isinstance(event, RouteDecision):
                yield ServerSentEvent(event="route", data=event.model_dump())
            elif isinstance(event, AgentDelta):
                yield ServerSentEvent(event="delta", data=event.text)
            elif isinstance(event, AgentToolCall):
                yield ServerSentEvent(event="tool_call", data={"tool": event.tool, "args": event.args})
            elif isinstance(event, AgentFinished):
                yield ServerSentEvent(event="sources", data=[s.model_dump() for s in _sources(event.sources)])
                yield ServerSentEvent(event="done", data={"conversation_id": conversation_id, "response": event.response})
    except Exception as e:
        log.exception("Chat stream error")
        yield ServerSentEvent(event="error", data=_error_detail("Chat", e))


@app.post("/route", response_model=RouteInfo)
async def route(body: RouteRequest, ctx: Ctx):
    """Classify a message without answering it (e.g. to sort incoming questions or tickets)."""
    if ctx.router is None:
        raise HTTPException(status_code=503, detail="Router is not enabled. Set ENABLE_ROUTER=true.")
    history = []
    if body.conversation_id:
        _, memory = ctx.conversations.get(body.conversation_id)
        history = await memory.aget_all()
    try:
        decision = await ctx.router.classify(body.message, history)
    except Exception as e:
        raise _internal_error("Routing", e)
    return RouteInfo(**decision.model_dump())


@app.post("/clear")
async def clear_conversation(body: ClearRequest, ctx: Ctx):
    """Forget a conversation's history."""
    found = await ctx.conversations.reset(body.conversation_id)
    return {"status": "ok", "cleared": found}

# ---------------------------------------------------------------------------
# Routes — retrieval & documents
# ---------------------------------------------------------------------------

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(body: RetrieveRequest, ctx: Ctx):
    """Search a collection without the agent: the chunks, scores and metadata an agent would see."""
    handle = _collection(ctx, body.collection)
    try:
        chunks = await handle.search(body.query, top_k=body.top_k, filters=body.filters)
    except Exception as e:
        raise _internal_error("Retrieval", e)
    return RetrieveResponse(collection=handle.name, results=_sources(chunks))


@app.post("/documents", response_model=AddDocumentsResponse)
async def add_documents(body: AddDocumentsRequest, ctx: Ctx):
    """Add or replace documents in a collection at runtime (e.g. a support ticket or a note).

    Re-posting a document with the same id replaces it. Runtime documents have no source file, so
    `ingest --reindex` deletes them; use a stream (POST /streams/...) or a file in the collection
    folder for anything that must survive one.
    """
    handle = _collection(ctx, body.collection)
    docs = [d.model_dump() for d in body.documents]
    try:
        # Embedding is blocking (and CPU-bound for local models); keep it off the event loop.
        ids = await asyncio.to_thread(handle.add_documents, docs)
    except Exception as e:
        raise _internal_error("Adding documents", e)
    return AddDocumentsResponse(collection=handle.name, ids=ids)

@app.post("/ingest", response_model=IngestResponse)
async def ingest(body: IngestRequest, ctx: Ctx):
    """Sync collection folders now (as startup auto-ingest does): embed new/changed files and report
    pending manual files. With include_manual, also embed changed files in manual folders."""
    handles = [_collection(ctx, body.collection)] if body.collection else list(ctx.collections.values())
    results = []
    try:
        for handle in handles:
            result = await asyncio.to_thread(handle.sync, body.include_manual)
            results.append(IngestResult(collection=handle.name, **asdict(result)))
    except Exception as e:
        raise _internal_error("Ingest", e)
    return IngestResponse(results=results)

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(body: SummarizeRequest, ctx: Ctx):
    """Summarise every chunk matching `filters`, in document order (e.g. a finished meeting's notes)."""
    handle = _collection(ctx, body.collection)
    try:
        result = await summarize_documents(handle, body.filters, body.instruction or DEFAULT_INSTRUCTION)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise _internal_error("Summarize", e)
    return SummarizeResponse(collection=handle.name, summary=result.text, chunks=result.chunks, doc_ids=result.doc_ids)

# ---------------------------------------------------------------------------
# Routes — streams
# ---------------------------------------------------------------------------

@app.get("/streams/{collection}", response_model=list[StreamStatus])
async def list_streams(collection: str, ctx: Ctx):
    handle = _collection(ctx, collection)
    return [_stream_status(handle, info) for info in handle.streams.list()]


@app.post("/streams/{collection}/{stream_id}", response_model=StreamAppendResponse)
async def append_to_stream(collection: str, stream_id: str, body: StreamAppendRequest, ctx: Ctx):
    """Append a fragment (e.g. one utterance or log entry) to a stream, creating the stream on first use.
    It is saved to the stream's file and embedded immediately, so it is searchable at once."""
    handle = _collection(ctx, collection)
    try:
        doc_id = await asyncio.to_thread(handle.append_to_stream, stream_id, body.text, body.metadata)
    except StreamError as e:
        raise _stream_error(e)
    except Exception as e:
        raise _internal_error("Stream append", e)
    return StreamAppendResponse(collection=handle.name, stream_id=stream_id, doc_id=doc_id)


@app.post("/streams/{collection}/{stream_id}/close", response_model=StreamStatus)
async def close_stream(collection: str, stream_id: str, body: StreamCloseRequest, ctx: Ctx):
    """Close a stream. By default its fragments are re-embedded as one normally-chunked document."""
    handle = _collection(ctx, collection)
    try:
        info = await asyncio.to_thread(handle.close_stream, stream_id, body.rechunk, body.metadata)
    except StreamError as e:
        raise _stream_error(e)
    except Exception as e:
        raise _internal_error("Stream close", e)
    return _stream_status(handle, info)

# ---------------------------------------------------------------------------
# Routes — evaluation
# ---------------------------------------------------------------------------

@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(body: EvaluateRequest, ctx: Ctx):
    """Evaluate a supplied answer against the documents retrieved for the question."""
    bundle = _require_evaluator(ctx)
    handle = _collection(ctx, body.collection)
    try:
        nodes = await handle.aretrieve(body.question)
        eval_result = await evaluate_answer(
            bundle=bundle,
            query=body.question,
            answer=body.answer,
            contexts=[n.get_content() for n in nodes],
        )
    except Exception as e:
        raise _internal_error("Evaluation", e)
    return EvaluateResponse(question=body.question, answer=body.answer, evaluation=eval_result.summary())


@app.post("/chat_and_evaluate", response_model=ChatEvalResponse)
async def chat_and_evaluate(body: ChatEvalRequest, ctx: Ctx):
    """Answer from one collection with a query engine (keeps source_nodes), then evaluate the answer."""
    bundle = _require_evaluator(ctx)
    handle = _collection(ctx, body.collection)
    try:
        rag_response = await handle.as_query_engine().aquery(body.message)
        eval_result = await evaluate_response(bundle=bundle, query=body.message, response=rag_response)
    except Exception as e:
        raise _internal_error("Chat and evaluate", e)
    return ChatEvalResponse(
        question=body.message,
        answer=str(rag_response),
        evaluation=eval_result.summary(),
        sources=_sources(RetrievedChunk.from_node(handle.name, n) for n in rag_response.source_nodes),
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sources(chunks) -> list[Source]:
    return [Source(**asdict(c)) for c in chunks]


def _collection(ctx: AppContext, name: str | None) -> CollectionHandle:
    if name is None:
        return ctx.default_collection
    try:
        return ctx.collection(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e.args[0]))


def _stream_status(handle: CollectionHandle, info: StreamInfo) -> StreamStatus:
    return StreamStatus(collection=handle.name, stream_id=info.stream_id, open=info.open,
                        fragments=info.fragments, granularity=str(info.granularity))


def _stream_error(e: StreamError) -> HTTPException:
    status = 404 if isinstance(e, StreamNotFound) else 409 if isinstance(e, StreamClosed) else 422
    return HTTPException(status_code=status, detail=str(e))


def _require_evaluator(ctx: AppContext):
    """Raise 503 if the evaluator bundle is not configured."""
    if ctx.evaluator_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Evaluator is not configured. Set JUDGE_MODEL (and optionally JUDGE_PROVIDER) in your .env file.",
        )
    return ctx.evaluator_bundle


def _error_detail(action: str, e: Exception) -> str:
    """Full error text in dev; a generic message elsewhere (details can include paths or provider responses)."""
    return f"{action} failed: {e}" if cfg.debug else f"{action} failed. See server logs for details."


def _internal_error(action: str, e: Exception) -> HTTPException:
    log.exception("%s failed", action)
    return HTTPException(status_code=500, detail=_error_detail(action, e))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.fastapi_app:app", host="127.0.0.1", port=8000, reload=cfg.debug)
