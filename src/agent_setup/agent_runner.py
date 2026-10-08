"""Run an agent and collect what callers need: streamed text, tool calls, and the chunks it used."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from llama_index.core.agent.workflow import AgentOutput, AgentStream, FunctionAgent, ToolCall, ToolCallResult
from llama_index.core.memory import Memory

from src.agent_setup.agent_tools import SearchResult
from src.indexing import RetrievedChunk

log = logging.getLogger(__name__)


@dataclass
class AgentDelta:
    """A piece of the answer text as the LLM streams it."""
    text: str


@dataclass
class AgentToolCall:
    """The agent decided to call a tool."""
    tool: str
    args: dict[str, Any]


@dataclass
class AgentFinished:
    """The final answer, plus every chunk returned by search tools during the run (deduplicated)."""
    response: str
    sources: list[RetrievedChunk] = field(default_factory=list)


AgentEvent = AgentDelta | AgentToolCall | AgentFinished


async def stream_agent(
    agent: FunctionAgent,
    message: str,
    memory: Memory | None,
    max_iterations: int,
) -> AsyncIterator[AgentEvent]:
    """Run the agent, yielding AgentDelta / AgentToolCall events and finally one AgentFinished.

    Logs (INFO) each tool call, what it returned, and a timing summary per answer — the quickest way to
    see why an answer was slow or didn't use the documents.
    """
    start = time.perf_counter()
    first_token: float | None = None
    llm_calls = tool_calls = 0
    handler = agent.run(user_msg=message, memory=memory, max_iterations=max_iterations)
    sources: dict[tuple[str, str | None, str], RetrievedChunk] = {}

    async for event in handler.stream_events():
        if isinstance(event, ToolCallResult):  # subclass of ToolCall: check first
            raw = event.tool_output.raw_output
            if isinstance(raw, SearchResult):
                chunks = raw.chunks()
                for chunk in chunks:
                    sources.setdefault((chunk.collection, chunk.doc_id, chunk.text), chunk)
                log.info("Tool result: %s returned %d chunk(s): %s", event.tool_name, len(chunks),
                         ", ".join(_chunk_label(c) for c in chunks) or "none")
            else:
                log.info("Tool result: %s -> %.120r", event.tool_name, str(event.tool_output))
        elif isinstance(event, ToolCall):
            tool_calls += 1
            log.info("Tool call: %s(%s)", event.tool_name, event.tool_kwargs)
            yield AgentToolCall(tool=event.tool_name, args=dict(event.tool_kwargs))
        elif isinstance(event, AgentOutput):
            llm_calls += 1
        elif isinstance(event, AgentStream) and event.delta:
            if first_token is None:
                first_token = time.perf_counter() - start
            yield AgentDelta(event.delta)

    response = await handler
    log.info(
        "Answered in %.1fs (first token after %s, %d LLM call(s), %d tool call(s), %d source chunk(s)).",
        time.perf_counter() - start, f"{first_token:.1f}s" if first_token is not None else "n/a",
        llm_calls, tool_calls, len(sources),
    )
    if tool_calls == 0:
        log.info("No tools were called: the answer did not use the document collections.")
    yield AgentFinished(response=str(response), sources=list(sources.values()))


def _chunk_label(chunk: RetrievedChunk) -> str:
    """Short label for logs, e.g. 'Mythic+Bastionland.pdf p.130 (0.72)'."""
    m = chunk.metadata
    name = m.get("source_path") or m.get("stream_id") or chunk.doc_id or "?"
    page = f" p.{m['page_label']}" if m.get("page_label") else ""
    score = f" ({chunk.score:.2f})" if chunk.score is not None else ""
    return f"{name}{page}{score}"


async def run_agent(agent: FunctionAgent, message: str, memory: Memory | None, max_iterations: int) -> AgentFinished:
    """Run the agent to completion (no streaming) and return the answer with its sources."""
    async for event in stream_agent(agent, message, memory, max_iterations):
        if isinstance(event, AgentFinished):
            return event
    raise RuntimeError("Agent finished without a response.")
