from __future__ import annotations

import ast
import logging
import operator
from datetime import datetime

from typing import Mapping

from llama_index.core.tools import FunctionTool

from src.indexing import CollectionHandle

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

def _make_list_documents(collections: Mapping[str, CollectionHandle]) -> FunctionTool:
    def list_indexed_documents(collection: str) -> str:
        try:
            handle = collections[collection]
        except KeyError:
            return f"Unknown collection '{collection}'. Available: {', '.join(collections)}."
        try:
            # Reads every chunk's metadata — fine for modest corpora; replace with a document registry at scale.
            results = handle.store.client.get_collection(handle.name).get(include=["metadatas"])
            names = sorted({m.get("file_name", "unknown") for m in results["metadatas"] if m})
            return "\n".join(names) if names else f"No documents are indexed in '{collection}'."
        except Exception as e:
            log.warning("Failed to list documents in '%s': %s", collection, e)
            return "Unable to retrieve document list."

    return FunctionTool.from_defaults(
        fn=list_indexed_documents,
        name="list_indexed_documents",
        description=(
            "Lists the source document names in a collection. "
            f"Available collections: {', '.join(collections)}."
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_generic_tools(collections: Mapping[str, CollectionHandle]) -> list[FunctionTool]:
    """Tools every agent can use. Search tools are added per collection by build_rag_agent()."""
    return [
        FunctionTool.from_defaults(fn=get_current_datetime),
        FunctionTool.from_defaults(fn=calculate),
        _make_list_documents(collections),
    ]
