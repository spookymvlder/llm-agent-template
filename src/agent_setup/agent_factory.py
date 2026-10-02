from __future__ import annotations

import logging
from typing import Sequence

from pydantic import BaseModel

from llama_index.core import Settings
from llama_index.core.agent.workflow import FunctionAgent

from src.agent_setup.agent_tools import build_search_tool
from src.indexing import CollectionHandle

log = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to searchable document collections.\n"
    "IMPORTANT RULES:\n"
    "- For greetings, casual conversation, or questions you can answer from general knowledge, "
    "respond DIRECTLY without using any tools.\n"
    "- Use a search_* tool when the question may be answered by the documents; pick the "
    "collection whose description matches the question.\n"
    "- Search results show each passage's metadata (source file, page, tags). Base your answer on "
    "the passages and say which sources you used.\n"
    "- Always provide a final answer. Do not loop or repeat tool calls.\n"
    "- When you have enough information to answer, stop and respond immediately."
)


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
