from __future__ import annotations

import logging
from typing import Sequence
from pydantic import BaseModel

from llama_index.core import Settings
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.tools import ToolMetadata, QueryEngineTool
from llama_index.core.agent.workflow import FunctionAgent






log = logging.getLogger(__name__)



DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a searchable document corpus.\n"
    "IMPORTANT RULES:\n"
    "- For greetings, casual conversation, or questions you can answer from general knowledge, "
    "respond DIRECTLY without using any tools.\n"
    "- Only use the rag_search tool when the user explicitly asks about documents or "
    "topics that require searching the document corpus.\n"
    "- Always provide a final answer. Do not loop or repeat tool calls.\n"
    "- When you have enough information to answer, stop and respond immediately."
)

DEFAULT_RAG_TOOL_DESCRIPTION = (
    "Searches the indexed document corpus for relevant context "
    "and returns an answer with citations. Use this tool whenever "
    "the question may be answered by the indexed documents."
)



def build_rag_agent(
    query_engine: BaseQueryEngine,
    extra_tools: Sequence = (),
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    rag_tool_description: str = DEFAULT_RAG_TOOL_DESCRIPTION,
    output_cls: type[BaseModel] | None = None,
    verbose: bool = True,
) -> FunctionAgent:
    """Create a FunctionAgent with a RAG query tool and any extra tools.

    Uses LlamaIndex's current workflow-based agent API (0.13+).
    The agent calls tools via the LLM's native function-calling interface,
    which is more reliable than ReAct prompting for most providers.

    Args:
        query_engine:         Query engine built on the vector store.
        extra_tools:          Additional LlamaIndex Tool objects (FunctionTool, QueryEngineTool, etc.).
        system_prompt:        System prompt controlling agent behavior and tone.
        rag_tool_description: Description the LLM reads to decide when to invoke the RAG tool.
                              Make this specific to your document corpus for best results.
        memory_token_limit:   Max tokens retained in conversation memory. Older messages
                              are dropped when the limit is reached.
        output_cls:           Optional Pydantic model for structured responses. When provided,
                              the agent will return a validated instance of this class instead
                              of plain text. Defaults to None (unstructured).
        verbose:              Log tool calls and intermediate reasoning steps.

    Returns:
        FunctionAgent ready for async use via agent.run(user_msg='...').
    """
    rag_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="rag_search",
            description=rag_tool_description,
        ),
    )


    tools = [rag_tool, *extra_tools]



    return FunctionAgent(
        tools=tools,
        llm=Settings.llm,
        system_prompt=system_prompt,
        output_cls=output_cls,
        verbose=verbose,
    )



