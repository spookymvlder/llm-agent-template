import json
import time

import pytest
from llama_index.core.base.llms.types import ChatMessage

from helpers import FakeRouterLLM
from src.agent_setup import AgentFinished, ConversationStore, QueryRouter, RouteSpec, default_routes, message_route


def router(replies, **kwargs):
    return QueryRouter(FakeRouterLLM(replies), default_routes(), default_route="rag", **kwargs)


@pytest.mark.parametrize("message, reply, route, fallback", [
    ("hello", '{"route": "direct", "confidence": 0.95, "reason": "greeting"}', "direct", False),
    ("handbook", '<think>hmm</think>\n```json\n{"route": "rag", "confidence": 0.9, "reason": "r",}\n```', "rag", False),
    ("garbage", "probably rag?", "rag", True),
    ("weather", '{"route": "weather", "confidence": 0.9, "reason": "x"}', "rag", True),
    ("unsure", '{"route": "direct", "confidence": 0.2, "reason": "meh"}', "rag", True),
])
async def test_classify_and_fallbacks(message, reply, route, fallback):
    decision = await router({message: reply}).classify(message)
    assert (decision.route, decision.fallback) == (route, fallback)


async def test_prompt_lists_routes_examples_and_recent_history():
    r = router({})
    history = [ChatMessage(role="user", content=f"turn {i}") for i in range(6)]
    await r.classify("and escaping?", history)
    prompt = r.llm.prompts[-1]
    assert "- clarify: " in prompt and 'e.g. "hi there"' in prompt
    assert "turn 5" in prompt and "turn 1" not in prompt               # only the last 4 turns
    assert "rag, direct, clarify" in prompt


def test_route_validation():
    with pytest.raises(ValueError, match="unique"):
        QueryRouter(FakeRouterLLM(), [*default_routes(), message_route("rag", "dup", "x")], "rag")
    with pytest.raises(ValueError, match="Default route"):
        QueryRouter(FakeRouterLLM(), default_routes(), "missing")


async def test_custom_route_handler():
    seen = []

    async def note(rc):
        seen.append(rc.message)
        yield AgentFinished(response="noted")

    spec = RouteSpec("offtopic", "Not about the game.", note)
    events = [e async for e in spec.handler(type("RC", (), {"message": "pizza?"})())]
    assert seen == ["pizza?"] and events[0].response == "noted"


# ---- conversation store ----------------------------------------------------------

async def test_conversation_store_lru_ttl_and_reset():
    made = []
    store = ConversationStore(factory=lambda cid: made.append(cid) or _FakeMemory(), max_conversations=2, ttl_s=0)
    a, mem_a = store.get(None)
    assert store.get(a)[1] is mem_a and made == [a]                    # same id → same memory
    store.get("b"); store.get("c")                                     # over the cap: 'a' (least recent) dropped
    assert len(store) == 2 and store.get(a)[1] is not mem_a
    assert await store.reset("c") is True and await store.reset("nope") is False

    expiring = ConversationStore(factory=lambda cid: _FakeMemory(), ttl_s=0.01)
    expiring.get("x")
    time.sleep(0.02)
    assert len(expiring) == 0


class _FakeMemory:
    async def areset(self):
        pass
