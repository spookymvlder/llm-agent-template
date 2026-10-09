from __future__ import annotations

import logging
from typing import Sequence

from pydantic import BaseModel

from llama_index.core import Settings
from llama_index.core.agent.workflow import FunctionAgent

from src.agent_setup.agent_tools import build_search_tool
from src.indexing import CollectionHandle

log = logging.getLogger(__name__)


# Search-first: small local models otherwise treat domain questions ("what's our parental leave policy?") as
# general knowledge and answer without searching. Projects with a router can rely on its `direct` route instead.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using searchable document collections.\n"
    "RULES:\n"
    "- The collections are your primary source. If a question could be answered by them, search the "
    "collection whose description matches FIRST, even if you think you already know the answer.\n"
    "- Search results show each passage's metadata (source file, page, tags). Base your answer on the "
    "passages and say which sources you used.\n"
    "- If the passages don't contain the answer, say so plainly. Never claim what the documents do or don't "
    "say without having searched them.\n"
    "- Answer directly without tools only for greetings, small talk, or questions clearly unrelated to the "
    "collections.\n"
    "- Don't repeat the same search. When you have enough information, give your final answer."
)


DIRECT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer directly and concisely from general knowledge and the "
    "conversation so far. You have no access to documents."
)


def build_direct_agent(system_prompt: str = DIRECT_SYSTEM_PROMPT, verbose: bool = False) -> FunctionAgent:
    """An agent with no tools: used by the router's `direct` route for small talk and general questions."""
    return FunctionAgent(tools=[], llm=Settings.llm, system_prompt=system_prompt, verbose=verbose)


def build_rag_agent(
    collections: Sequence[CollectionHandle],
    extra_tools: Sequence = (),
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    output_cls: type[BaseModel] | None = None,
    verbose: bool = False,
) -> FunctionAgent:
    """Create a FunctionAgent with one search tool per collection plus any extra tools.

    Uses LlamaIndex's current workflow-based agent API (0.13+).
    The agent calls tools via the LLM's native function-calling interface,
    which is more reliable than ReAct prompting for most providers.

    Pass a subset of collections to build a specialised agent, e.g. a rules agent
    that can only search the 'rules' collection.

    Args:
        collections:   Collections the agent may search. Each collection's description
                       (COLLECTION_<NAME>_DESCRIPTION) tells the LLM when to use it.
        extra_tools:   Additional LlamaIndex Tool objects (FunctionTool, QueryEngineTool, etc.).
        system_prompt: System prompt controlling agent behavior and tone.
        output_cls:    Optional Pydantic model for structured responses. When provided,
                       the agent will return a validated instance of this class instead
                       of plain text. Defaults to None (unstructured).
        verbose:       Print each workflow step to stdout (startup enables this when LOG_LEVEL=DEBUG).

    Returns:
        FunctionAgent ready for async use via agent.run(user_msg='...').
    """
    tools = [*(build_search_tool(c) for c in collections), *extra_tools]
    log.info("Agent tools: %s", [t.metadata.name for t in tools])
    return FunctionAgent(
        tools=tools,
        llm=Settings.llm,
        system_prompt=system_prompt,
        output_cls=output_cls,
        verbose=verbose,
    )
