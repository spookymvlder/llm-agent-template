from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.templating import Jinja2Templates
from fastapi.sse import EventSourceResponse, ServerSentEvent

from llama_index.core.agent.workflow import AgentStream
from sse_starlette.sse import EventSourceResponse

from src.evaluation import EvaluatorBundle, evaluate_response
from src.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatEvalResponse,
    ChatRequest,
    ChatResponse,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
)
from src.startup import bootstrap




log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

agent = None
query_engine = None
evaluator_bundle: EvaluatorBundle | None = None
memory = None                   # built after bootstrap

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, query_engine, evaluator_bundle, max_iterations, memory
    result = bootstrap()
    agent = result.agent
    query_engine = result.query_engine
    evaluator_bundle = result.evaluator_bundle
    max_iterations = result.max_iterations
    memory = result.memory
    log.info(
        "Startup complete. evaluator=%s",
        "enabled" if evaluator_bundle else "disabled",
    )
    yield
    log.info("Shutting down.")

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
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        test = await agent.run(user_msg="Hello")
        return HealthResponse(
            status="healthy",
            agent_responsive=True,
            message=str(test),
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    """Non-streaming chat. Useful for programmatic clients and testing."""
    if not body.message.strip():
        raise HTTPException(status_code=422, detail="Message cannot be empty.")
    try:
        response = await agent.run(user_msg=body.message, max_iterations=max_iterations)
        return ChatResponse(response=str(response))
    except Exception as e:
        log.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(body: ChatRequest):
    """Streaming chat via SSE. Used by the browser UI."""
    if not body.message.strip():
        raise HTTPException(status_code=422, detail="Message cannot be empty.")

    async def generate():
        try:
            handler = agent.run(user_msg=body.message, max_iterations=max_iterations)
            async for event in handler.stream_events():
                if isinstance(event, AgentStream):
                    yield {"data": event.delta}
        except Exception as e:
            log.exception("Chat stream error")
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(generate())


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_url(body: AnalyzeRequest):
    prompt = (
        f"Analyze this URL focusing on the last {body.time_limit_days} days: {body.url}"
        if body.time_limit_days
        else f"Using the media analyzer tool, extract themes from this content: {body.url}"
    )
    try:
        response = await agent.run(user_msg=prompt, max_iterations=max_iterations)
        return AnalyzeResponse(url=body.url, analysis=str(response))
    except Exception as e:
        log.exception("URL analysis error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(body: EvaluateRequest):
    _require_evaluator()
    try:
        # Query the index directly to get a Response object with source_nodes.
        rag_response = query_engine.query(body.question)
        eval_result = await evaluate_response(
            bundle=evaluator_bundle,
            query=body.question,
            response=rag_response,
        )
        return EvaluateResponse(
            question=body.question,
            answer=body.answer,
            evaluation=eval_result.summary(),
        )
    except Exception as e:
        log.exception("Evaluation error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat_and_evaluate", response_model=ChatEvalResponse)
async def chat_and_evaluate(body: ChatRequest):
    _require_evaluator()
    try:
        # Use query_engine directly to preserve source_nodes for evaluation.
        rag_response = query_engine.query(body.message)
        answer_str = str(rag_response)

        eval_result = await evaluate_response(
            bundle=evaluator_bundle,
            query=body.message,
            response=rag_response,
        )

        return ChatEvalResponse(
            question=body.message,
            answer=answer_str,
            evaluation=eval_result.summary(),
        )
    except Exception as e:
        log.exception("Chat and evaluate error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
async def clear_conversation():
    """Reset the agent's conversation memory."""
    agent.reset()
    return {"status": "ok", "message": "Conversation cleared."}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_evaluator() -> None:
    """Raise 503 if the evaluator bundle is not configured."""
    if evaluator_bundle is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Evaluator is not configured. "
                "Set JUDGE_PROVIDER and JUDGE_MODEL in your .env file."
            ),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    # TODO update reload to be a flag.
    uvicorn.run("src.fastapi_app:app", host="127.0.0.1", port=8000, reload=True)