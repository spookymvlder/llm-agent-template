from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from src.startup import bootstrap
from src.models import ChatRequest, ChatResponse, AnalyzeRequest, AnalyzeResponse, HealthResponse, EvaluateRequest, EvaluateResponse, ChatEvalResponse

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Lifespan — replaces @app.before_first_request / teardown
# ---------------------------------------------------------------------------

agent = None
evaluator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, evaluator
    result = bootstrap()
    agent = result.agent
    evaluator = result.evaluator
    yield

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

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
    
@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        test = await agent.run("Hello")
        return HealthResponse(status="healthy", agent_responsive=True, message=str(test))
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    if not body.message.strip():
        raise HTTPException(status_code=422, detail="Message cannot be empty.")
    try:
        response = await agent.run(body.message)
        return ChatResponse(response=str(response))
    except Exception as e:
        log.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_url(body: AnalyzeRequest):
    if body.time_limit_days:
        prompt = f"Analyze this URL focusing on the last {body.time_limit_days} days: {body.url}"
    else:
        prompt = f"Using the media analyzer tool, extract themes from this content: {body.url}"
    try:
        response = await agent.run(user_msg=prompt)
        return AnalyzeResponse(url=body.url, analysis=str(response))
    except Exception as e:
        log.exception("URL analysis error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(body: EvaluateRequest):
    if evaluator is None:
        raise HTTPException(
            status_code=503,
            detail="Evaluator is not configured. Set JUDGE_PROVIDER and JUDGE_MODEL in your .env file."
        )
    prompt = (
        f"Please evaluate the following response to a user question.\n\n"
        f"Question: {body.question}\n"
        f"Response: {body.answer}\n\n"
        f"Use the retrieve_source_documents tool to fetch relevant context "
        f"from the document corpus before making your assessment."
    )
    try:
        result = await evaluator.run(user_msg=prompt)
        return EvaluateResponse(
            question=body.question,
            answer=body.answer,
            evaluation=str(result),
        )
    except Exception as e:
        log.exception("Evaluation error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat_and_evaluate", response_model=ChatEvalResponse)
async def chat_and_evaluate(body: ChatRequest):
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Evaluator is not configured.")
    try:
        answer = await agent.run(user_msg=body.message)
        answer_str = str(answer)

        prompt = (
            f"Please evaluate the following response to a user question.\n\n"
            f"Question: {body.message}\n"
            f"Response: {answer_str}\n\n"
            f"Use the retrieve_source_documents tool to fetch relevant context "
            f"before making your assessment."
        )
        evaluation = await evaluator.run(user_msg=prompt)

        return ChatEvalResponse(
            question=body.message,
            answer=answer_str,
            evaluation=str(evaluation),
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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
