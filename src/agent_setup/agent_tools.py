from __future__ import annotations

import logging
from datetime import datetime

import chromadb
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.tools import FunctionTool

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone tools — no external dependencies
# ---------------------------------------------------------------------------

def get_current_datetime() -> str:
    """Returns the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """Evaluates a mathematical expression. Example: '2 + 2 * 10'"""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


# ---------------------------------------------------------------------------
# Factories — close over external dependencies
# ---------------------------------------------------------------------------

def _make_summarize_topic(query_engine: BaseQueryEngine):
    def summarize_topic(topic: str) -> str:
        """Summarizes available information on a topic from the document corpus."""
        response = query_engine.query(f"Summarize everything about: {topic}")
        return str(response)
    return summarize_topic


def _make_list_documents(chroma_client: chromadb.ClientAPI, collection_name: str):
    def list_indexed_documents() -> str:
        """Returns a list of document names currently available in the index."""
        try:
            collection = chroma_client.get_collection(collection_name)
            results = collection.get(include=["metadatas"])
            names = sorted({
                m.get("file_name", "unknown")
                for m in results["metadatas"]
                if m
            })
            if not names:
                return "No documents are currently indexed."
            return "\n".join(names)
        except Exception as e:
            log.warning("Failed to list documents: %s", e)
            return "Unable to retrieve document list."
    return list_indexed_documents


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_generic_tools(
    query_engine: BaseQueryEngine,
    chroma_client: chromadb.ClientAPI,
    collection_name: str = "documents",
) -> list[FunctionTool]:
    """Build the standard tool set for a general-purpose RAG agent.

    Args:
        query_engine:     Query engine built on the vector store.
        chroma_client:    ChromaDB client for metadata queries.
        collection_name:  Name of the ChromaDB collection.

    Returns:
        List of FunctionTool objects ready to pass as extra_tools.
    """
    return [
        FunctionTool.from_defaults(fn=get_current_datetime),
        FunctionTool.from_defaults(fn=calculate),
        FunctionTool.from_defaults(fn=_make_summarize_topic(query_engine)),
        FunctionTool.from_defaults(fn=_make_list_documents(chroma_client, collection_name)),
    ]