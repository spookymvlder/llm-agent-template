"""
Optional query router: a cheap LLM call that picks how to handle a message before any agent runs.

    decision = await router.classify("How does grappling work?")   # RouteDecision(route="rag", ...)
    async for event in router.route(decision.route).handler(route_ctx): ...

A route is a name, a description the router LLM reads, and a handler. Handlers are async generators
yielding the same events as stream_agent() (AgentDelta / AgentToolCall / AgentFinished), so a handler
can run an agent, return a fixed message, or do anything else (append to a list, call an API) and
report back with a single AgentFinished.

The router asks for JSON in plain text and parses it tolerantly (src/llm/parsing.py) rather than
relying on provider-specific structured output, so it works with small local models. Unparseable
output, unknown routes and low confidence all fall back to the default route.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Sequence

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import Memory
from pydantic import BaseModel, Field

from src.agent_setup.agent_runner import AgentEvent, AgentFinished, stream_agent
from src.llm.parsing import parse_llm_model

if TYPE_CHECKING:
    from src.app_context import AppContext

log = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    route: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    fallback: bool = False   # True when the default route was used instead of the LLM's choice


@dataclass
class RouteContext:
    """Everything a route handler gets."""
    ctx: AppContext
    message: str
    memory: Memory | None
    decision: RouteDecision
    max_iterations: int


RouteHandler = Callable[[RouteContext], AsyncIterator[AgentEvent]]


@dataclass
class RouteSpec:
    """A route the router can choose.

    Args:
        name:        Identifier returned by the router (e.g. "rules").
        description: When to choose this route — the router LLM decides from these descriptions.
        handler:     Async generator producing the response events.
        examples:    Optional example messages, shown to the router to sharpen its choice.
    """
    name: str
    description: str
    handler: RouteHandler
    examples: Sequence[str] = field(default_factory=tuple)


ROUTER_PROMPT = """You route user messages for an assistant. Choose exactly one route.

Routes:
{routes}

Recent conversation (oldest first; may be empty):
{history}

Message: {message}

Reply with only a JSON object, no other text:
{{"route": "<one of: {names}>", "confidence": <number from 0 to 1>, "reason": "<a few words>"}}"""


class QueryRouter:
    """Classifies messages into one of `routes` with an LLM.

    Args:
        llm:            The router LLM (ROUTER_* settings; a small, fast model with thinking off works well).
        routes:         Available routes. Names must be unique.
        default_route:  Used when the LLM's answer is unusable or below min_confidence.
        min_confidence: Decisions below this confidence fall back to default_route.
        history_turns:  How many recent conversation messages to show the router (for follow-ups).
    """

    def __init__(
        self,
        llm: Any,
        routes: Sequence[RouteSpec],
        default_route: str,
        min_confidence: float = 0.5,
        history_turns: int = 4,
    ) -> None:
        names = [r.name for r in routes]
        if len(set(names)) != len(names):
            raise ValueError(f"Route names must be unique: {names}")
        if default_route not in names:
            raise ValueError(f"Default route '{default_route}' is not one of the routes: {', '.join(names)}")
        self.llm = llm
        self.routes = {r.name: r for r in routes}
        self.default_route = default_route
        self.min_confidence = min_confidence
        self.history_turns = history_turns

    def route(self, name: str) -> RouteSpec:
        return self.routes[name]

    async def classify(self, message: str, history: Sequence[ChatMessage] = ()) -> RouteDecision:
        """Pick a route for `message`. Never raises for bad LLM output — falls back to the default route.

        Raises:
            Exception: The LLM call itself failed (e.g. provider unreachable).
        """
        response = await self.llm.acomplete(self._prompt(message, history))
        decision = parse_llm_model(response.text, RouteDecision)
        if decision is None:
            return self._fallback(f"unparseable router output: {response.text[:120]!r}")
        if decision.route not in self.routes:
            return self._fallback(f"unknown route '{decision.route}'")
        if decision.confidence < self.min_confidence:
            return self._fallback(f"low confidence {decision.confidence:.2f} for '{decision.route}' ({decision.reason})")
        log.info("Route: %s (%.2f) — %s", decision.route, decision.confidence, decision.reason)
        return decision

    def _fallback(self, why: str) -> RouteDecision:
        log.info("Route: %s (fallback) — %s", self.default_route, why)
        return RouteDecision(route=self.default_route, confidence=0.0, reason=why, fallback=True)

    def _prompt(self, message: str, history: Sequence[ChatMessage]) -> str:
        lines = []
        for r in self.routes.values():
            lines.append(f"- {r.name}: {r.description}")
            lines.extend(f'    e.g. "{example}"' for example in r.examples)
        recent = [m for m in history if m.role in (MessageRole.USER, MessageRole.ASSISTANT) and m.content][-self.history_turns:]
        return ROUTER_PROMPT.format(
            routes="\n".join(lines),
            history="\n".join(f"{m.role.value}: {m.content[:300]}" for m in recent) or "(none)",
            message=message,
            names=", ".join(self.routes),
        )


# ---------------------------------------------------------------------------
# Default routes — replace or extend via bootstrap(routes=[...])
# ---------------------------------------------------------------------------

CLARIFY_MESSAGE = "I'm not sure what you're asking. Could you rephrase it or add a bit more detail?"


def agent_route(name: str, description: str, agent_name: str, examples: Sequence[str] = ()) -> RouteSpec:
    """A route that runs one of ctx.agents (with the conversation's memory)."""
    async def handler(rc: RouteContext) -> AsyncIterator[AgentEvent]:
        async for event in stream_agent(rc.ctx.agents[agent_name], rc.message, rc.memory, rc.max_iterations):
            yield event
    return RouteSpec(name, description, handler, examples)


def message_route(name: str, description: str, message: str, examples: Sequence[str] = ()) -> RouteSpec:
    """A route that replies with a fixed message (no agent, no LLM call)."""
    async def handler(rc: RouteContext) -> AsyncIterator[AgentEvent]:
        yield AgentFinished(response=message)
    return RouteSpec(name, description, handler, examples)


def default_routes() -> list[RouteSpec]:
    """rag (search the collections), direct (answer without tools) and clarify."""
    return [
        agent_route("rag", "Questions that may be answered by the document collections.", "rag"),
        agent_route(
            "direct", "Greetings, small talk, or general questions that don't need the documents.", "direct",
            examples=("hi there", "thanks!"),
        ),
        message_route("clarify", "Messages too vague or ambiguous to act on.", CLARIFY_MESSAGE,
                      examples=("what about it?",)),
    ]
