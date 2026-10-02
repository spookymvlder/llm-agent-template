"""Run an agent and collect what callers need: streamed text, tool calls, and the chunks it used."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from llama_index.core.agent.workflow import AgentStream, FunctionAgent, ToolCall, ToolCallResult
from llama_index.core.memory import Memory

from src.agent_setup.agent_tools import SearchResult
from src.indexing import RetrievedChunk


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
    """Run the agent, yielding AgentDelta / AgentToolCall events and finally one AgentFinished."""
    handler = agent.run(user_msg=message, memory=memory, max_iterations=max_iterations)
    sources: dict[tuple[str, str | None, str], RetrievedChunk] = {}

    async for event in handler.stream_events():
        if isinstance(event, ToolCallResult):  # subclass of ToolCall: check first
            raw = event.tool_output.raw_output
            if isinstance(raw, SearchResult):
                for chunk in raw.chunks():
                    sources.setdefault((chunk.collection, chunk.doc_id, chunk.text), chunk)
        elif isinstance(event, ToolCall):
            yield AgentToolCall(tool=event.tool_name, args=dict(event.tool_kwargs))
        elif isinstance(event, AgentStream) and event.delta:
            yield AgentDelta(event.delta)

    response = await handler
    yield AgentFinished(response=str(response), sources=list(sources.values()))


async def run_agent(agent: FunctionAgent, message: str, memory: Memory | None, max_iterations: int) -> AgentFinished:
    """Run the agent to completion (no streaming) and return the answer with its sources."""
    async for event in stream_agent(agent, message, memory, max_iterations):
        if isinstance(event, AgentFinished):
            return event
    raise RuntimeError("Agent finished without a response.")
