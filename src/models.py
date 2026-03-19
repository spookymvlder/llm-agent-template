from pydantic import BaseModel

class EvaluationResult(BaseModel):
    faithfulness: int
    relevance: int
    completeness: int
    fairness: int
    justification: str

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

class AnalyzeRequest(BaseModel):
    url: str
    time_limit_days: int | None = None

class AnalyzeResponse(BaseModel):
    url: str
    analysis: str

class HealthResponse(BaseModel):
    status: str
    agent_responsive: bool
    message: str = ""
    error: str = ""

class EvaluateRequest(BaseModel):
    question: str
    answer: str

class EvaluateResponse(BaseModel):
    question: str
    answer: str
    evaluation: str

class ChatEvalResponse(BaseModel):
    question: str
    answer: str
    evaluation: str