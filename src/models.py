from typing import Annotated, Any

from pydantic import BaseModel, Field, StringConstraints

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
MetadataValue = str | int | float | bool


class Source(BaseModel):
    """A retrieved chunk: which collection and document it came from, its text, score and metadata."""
    collection: str
    doc_id: str | None
    text: str
    score: float | None
    metadata: dict[str, Any]


# ---- Chat ------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: NonEmptyStr
    conversation_id: str | None = Field(
        default=None, description="Continue a conversation. Omit to start a new one; the response returns its id.",
    )

class RouteInfo(BaseModel):
    """The router's decision. `fallback` means the default route was used (unclear or unparseable answer)."""
    route: str
    confidence: float
    reason: str
    fallback: bool

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    sources: list[Source]
    route: RouteInfo | None = Field(default=None, description="Set when the router is enabled.")

class RouteRequest(BaseModel):
    message: NonEmptyStr
    conversation_id: str | None = Field(default=None, description="Give the router this conversation's recent turns.")

class ClearRequest(BaseModel):
    conversation_id: NonEmptyStr


# ---- Retrieval & documents -------------------------------------------------

class RetrieveRequest(BaseModel):
    query: NonEmptyStr
    collection: str | None = Field(default=None, description="Defaults to the first collection in COLLECTIONS.")
    top_k: int | None = Field(default=None, ge=1, le=100)
    filters: dict[str, MetadataValue] | None = Field(
        default=None, description='Exact-match metadata filters, e.g. {"version": "2024"}.',
    )

class RetrieveResponse(BaseModel):
    collection: str
    results: list[Source]

class DocumentIn(BaseModel):
    text: NonEmptyStr
    id: str | None = Field(default=None, description="Stable id; re-posting the same id replaces the document.")
    metadata: dict[str, Any] = Field(default_factory=dict)

class AddDocumentsRequest(BaseModel):
    collection: str | None = Field(default=None, description="Defaults to the first collection in COLLECTIONS.")
    documents: list[DocumentIn] = Field(min_length=1, max_length=1000)

class AddDocumentsResponse(BaseModel):
    collection: str
    ids: list[str]


class IngestRequest(BaseModel):
    collection: str | None = Field(default=None, description="Omit to sync every collection.")
    include_manual: bool = Field(default=False, description='Also embed changed files in {"_mode": "manual"} folders.')

class IngestResult(BaseModel):
    collection: str
    new_files: list[str]
    changed_files: list[str]
    pending_files: list[str]
    missing_files: list[str]
    streams_embedded: list[str]
    documents_added: int
    chunks_removed: int

class IngestResponse(BaseModel):
    results: list[IngestResult]


class SummarizeRequest(BaseModel):
    collection: str | None = Field(default=None, description="Defaults to the first collection in COLLECTIONS.")
    filters: dict[str, MetadataValue] | None = Field(
        default=None, description='Exact-match filters selecting what to summarise, e.g. {"stream_id": "standup-2026-10-02"}.',
    )
    instruction: str | None = Field(default=None, description="What the summary should focus on.")

class SummarizeResponse(BaseModel):
    collection: str
    summary: str
    chunks: int
    doc_ids: list[str]


# ---- Streams ---------------------------------------------------------------

class StreamAppendRequest(BaseModel):
    text: NonEmptyStr
    metadata: dict[str, Any] = Field(default_factory=dict)

class StreamAppendResponse(BaseModel):
    collection: str
    stream_id: str
    doc_id: str

class StreamCloseRequest(BaseModel):
    rechunk: bool = Field(default=True, description="Replace fragment chunks with the whole stream as one document.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Added to the re-chunked document.")

class StreamStatus(BaseModel):
    collection: str
    stream_id: str
    open: bool
    fragments: int
    granularity: str


# ---- Status ----------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str

class CollectionStatus(BaseModel):
    name: str
    description: str
    chunks: int

class ReadyResponse(BaseModel):
    ready: bool
    collections: list[CollectionStatus]
    routes: list[str] | None = Field(default=None, description="Router routes; None when the router is disabled.")
    evaluator_ready: bool
    conversations: int


# ---- Evaluation ------------------------------------------------------------

class EvaluateRequest(BaseModel):
    question: NonEmptyStr
    answer: NonEmptyStr
    collection: str | None = None

class EvaluateResponse(BaseModel):
    question: str
    answer: str
    evaluation: dict[str, Any]

class ChatEvalRequest(BaseModel):
    message: NonEmptyStr
    collection: str | None = None

class ChatEvalResponse(BaseModel):
    question: str
    answer: str
    evaluation: dict[str, Any]
    sources: list[Source]
