"""Plain helpers shared by the tests (fixtures live in conftest.py)."""
from __future__ import annotations

import json
from pathlib import Path

from llama_index.core.base.llms.types import ChatMessage, CompletionResponse, MessageRole, ToolCallBlock
from llama_index.core.llms.mock import MockFunctionCallingLLM

from src.indexing import CollectionHandle


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def stored(handle: CollectionHandle) -> list[tuple[str, str, dict]]:
    """(document_id, text, metadata) for every chunk in a collection, sorted by document id."""
    got = handle.store.client.get_collection(handle.name).get(include=["documents", "metadatas"])
    return sorted((m["document_id"], d, m) for d, m in zip(got["documents"], got["metadatas"]))


def scripted_agent_llm(answer: str = "Employees get 25 vacation days.") -> MockFunctionCallingLLM:
    """A tool-calling LLM: the first turn calls search_<collection> (collection named in the message after
    'in:' — default 'handbook'), the turn after a tool result answers. Tool-less agents answer directly."""
    def generate(messages):
        last = messages[-1]
        if any("no access to documents" in (m.content or "") for m in messages if m.role == MessageRole.SYSTEM):
            return ChatMessage(role="assistant", content="Hello!")
        if last.role == MessageRole.TOOL:
            return ChatMessage(role="assistant", content=answer)
        text = last.content or ""
        collection = text.split("in:", 1)[1].strip().split()[0] if "in:" in text else "handbook"
        return ChatMessage(role="assistant", blocks=[
            ToolCallBlock(tool_call_id="t1", tool_name=f"search_{collection}", tool_kwargs={"query": text}),
        ])
    return MockFunctionCallingLLM(response_generator=generate)


class FakeRouterLLM:
    """Router LLM returning canned replies: `replies` maps a substring of the message to raw LLM text."""
    def __init__(self, replies: dict[str, str] | None = None, default_route: str = "rag"):
        self.replies = replies or {}
        self.default = json.dumps({"route": default_route, "confidence": 0.9, "reason": "default"})
        self.prompts: list[str] = []

    async def acomplete(self, prompt: str):
        self.prompts.append(prompt)
        message = prompt.rsplit("Message: ", 1)[1].split("\n")[0].lower()
        return CompletionResponse(text=next((r for k, r in self.replies.items() if k in message), self.default))
