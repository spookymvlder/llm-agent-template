"""Answer a message: route it (if the router is enabled), then run the chosen handler.

Shared by the API and the CLI so both behave the same.
"""
from __future__ import annotations

from typing import AsyncIterator

from llama_index.core.memory import Memory

from src.agent_setup import AgentEvent, AgentFinished, stream_agent
from src.agent_setup.router import RouteContext, RouteDecision
from src.app_context import AppContext


async def respond(
    ctx: AppContext,
    message: str,
    memory: Memory | None,
    max_iterations: int,
) -> AsyncIterator[RouteDecision | AgentEvent]:
    """Yield a RouteDecision first (only when the router is enabled), then the handler's events,
    ending with AgentFinished. Without a router, the default agent answers."""
    if ctx.router is None:
        async for event in stream_agent(ctx.agent, message, memory, max_iterations):
            yield event
        return

    history = await memory.aget_all() if memory is not None else []
    decision = await ctx.router.classify(message, history)
    yield decision
    route_ctx = RouteContext(ctx=ctx, message=message, memory=memory, decision=decision, max_iterations=max_iterations)
    async for event in ctx.router.route(decision.route).handler(route_ctx):
        yield event


async def answer(
    ctx: AppContext,
    message: str,
    memory: Memory | None,
    max_iterations: int,
) -> tuple[RouteDecision | None, AgentFinished]:
    """respond() without streaming: the route decision (None without a router) and the final answer."""
    decision = None
    async for event in respond(ctx, message, memory, max_iterations):
        if isinstance(event, RouteDecision):
            decision = event
        elif isinstance(event, AgentFinished):
            return decision, event
    raise RuntimeError("Handler finished without a response.")
