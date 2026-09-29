from typing import Annotated, Any

from pydantic import BaseModel, StringConstraints

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ChatRequest(BaseModel):
    message: NonEmptyStr

class ChatResponse(BaseModel):
    response: str

class HealthResponse(BaseModel):
    status: str
    agent_ready: bool
    evaluator_ready: bool

class EvaluateRequest(BaseModel):
    question: str
    answer: str

class EvaluateResponse(BaseModel):
    question: str
    answer: str
    evaluation: dict[str, Any]

class ChatEvalResponse(BaseModel):
    question: str
    answer: str
    evaluation: dict[str, Any]
