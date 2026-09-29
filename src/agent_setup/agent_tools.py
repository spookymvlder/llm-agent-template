from __future__ import annotations

import ast
import logging
import operator
from datetime import datetime

import chromadb
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone tools — no external dependencies
# ---------------------------------------------------------------------------

def get_current_datetime() -> str:
    """Returns the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_EXPONENT = 1000


def _eval_node(node: ast.AST) -> int | float:
    """Evaluate an arithmetic AST node. Anything other than numbers and basic operators is rejected."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"exponent larger than {_MAX_EXPONENT}")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> str:
    """Evaluates an arithmetic expression using + - * / // % ** and parentheses. Example: '2 + 2 * 10'"""
    try:
        return str(_eval_node(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Calculation error: {e}"


# ---------------------------------------------------------------------------
# Factories — close over external dependencies
# ---------------------------------------------------------------------------






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
) -> list:
    summarize_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="summarize_topic",
            description=(
                "Summarizes available information on a broad topic from the "
                "document corpus. Use this when the user asks for an overview "
                "or summary rather than a specific answer."
            ),
        ),
    )

    return [
        FunctionTool.from_defaults(fn=get_current_datetime),
        FunctionTool.from_defaults(fn=calculate),
        FunctionTool.from_defaults(fn=_make_list_documents(chroma_client, collection_name)),
        summarize_tool,
    ]