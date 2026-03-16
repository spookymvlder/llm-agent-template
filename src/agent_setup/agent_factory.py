from __future__ import annotations

import logging
from typing import Sequence
from pydantic import BaseModel

from llama_index.core import Settings
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.tools import ToolMetadata, QueryEngineTool, FunctionTool
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import ChatMemoryBuffer



log = logging.getLogger(__name__)


DEFAULT_EVALUATOR_SYSTEM_PROMPT = (
    "You are an impartial evaluator assessing the quality of an AI assistant's responses. "
    "You will be given a user question, the assistant's response, and relevant source documents. "
    "Evaluate the response on the following criteria:\n"
    "- Faithfulness: Does the response accurately reflect the source documents without hallucination?\n"
    "- Relevance: Does the response directly address the user's question?\n"
    "- Completeness: Does the response cover the key information available in the source documents?\n"
    "- Fairness: Is the response free from unsupported bias or assumptions?\n"
    "Provide a score from 1-5 for each criterion and a brief justification. "
    "Be critical and objective — your role is to identify weaknesses, not to flatter."
)


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a searchable document corpus. "
    "Always use the rag_search tool to find relevant information before answering. "
    "Cite the source documents in your response where possible. "
    "If the documents don't contain relevant information, say so rather than guessing."
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
    memory_token_limit: int = 4096,
    output_cls: type[BaseModel] | None = None,
    verbose: bool = False,
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

    memory = ChatMemoryBuffer.from_defaults(token_limit=memory_token_limit)

    return FunctionAgent(
        tools=[rag_tool, *extra_tools],
        llm=Settings.llm,
        system_prompt=system_prompt,
        memory=memory,
        output_cls=output_cls,
        verbose=verbose,
    )


def build_evaluator_agent(
    query_engine: BaseQueryEngine,
    llm: Any,
    extra_tools: Sequence = (),
    system_prompt: str = DEFAULT_EVALUATOR_SYSTEM_PROMPT,
    memory_token_limit: int = 4096,
    output_cls: type[BaseModel] | None = None,
    verbose: bool = False,
) -> FunctionAgent:
    """Create a judge agent that evaluates a RAG agent's responses for quality.

    Shares the same vector store and embeddings as the RAG agent so it can
    retrieve source documents for comparison, but uses an explicitly provided
    LLM rather than Settings.llm — the judge should be a different model than
    the one being evaluated.

    Args:
        query_engine:       Query engine built on the same vector store as the RAG agent.
        llm:                The judge LLM. Should differ from Settings.llm to avoid
                            a model evaluating its own outputs.
        extra_tools:        Additional tools available to the evaluator.
        system_prompt:      Evaluation criteria and instructions for the judge.
        memory_token_limit: Max tokens retained in conversation memory.
        output_cls:         Optional Pydantic model for structured evaluation output.
                            Useful for capturing scores and justifications in a
                            consistent, parseable format.
        verbose:            Log tool calls and intermediate reasoning steps.

    Returns:
        FunctionAgent configured as a judge, ready for async use via agent.run().
    """
    retrieval_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="retrieve_source_documents",
            description=(
                "Retrieves the source documents relevant to a user question. "
                "Use this to obtain the ground truth context when evaluating "
                "whether a response is faithful and complete."
            ),
        ),
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=memory_token_limit)

    return FunctionAgent(
        tools=[retrieval_tool, *extra_tools],
        llm=llm,
        system_prompt=system_prompt,
        memory=memory,
        output_cls=output_cls,
        verbose=verbose,
    )