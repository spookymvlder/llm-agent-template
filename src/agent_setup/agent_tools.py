from __future__ import annotations

import ast
import logging
import operator
from datetime import datetime

import re
from dataclasses import dataclass
from typing import Mapping

from llama_index.core.schema import MetadataMode, NodeWithScore
from llama_index.core.tools import FunctionTool

from src.indexing import CollectionHandle, RetrievedChunk

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


@dataclass
class SearchResult:
    """What a search tool returns. The agent reads str(result); the API reads `nodes` for `sources`."""
    collection: str
    query: str
    nodes: list[NodeWithScore]

    def __str__(self) -> str:
        if not self.nodes:
            return f"No results in '{self.collection}' for: {self.query}"
        parts = []
        for i, n in enumerate(self.nodes, start=1):
            score = f" (score {n.score:.2f})" if n.score is not None else ""
            parts.append(f"[{i}]{score}\n{n.node.get_content(metadata_mode=MetadataMode.LLM)}")
        return "\n\n".join(parts)

    def chunks(self) -> list[RetrievedChunk]:
        return [RetrievedChunk.from_node(self.collection, n) for n in self.nodes]


def search_tool_name(collection_name: str) -> str:
    """Tool name for a collection's search tool, e.g. 'rules' -> 'search_rules'."""
    return "search_" + re.sub(r"[^a-zA-Z0-9_]", "_", collection_name)


def build_search_tool(collection: CollectionHandle) -> FunctionTool:
    """A tool that returns a collection's best-matching chunks with their metadata (source, page, tags...).

    The agent reads the chunks directly — no separate LLM call summarises them first — so it can see
    where each passage came from and the API can report the chunks as `sources`.
    """
    async def search(query: str, filters: dict[str, str | int | float | bool] | None = None) -> SearchResult:
        return SearchResult(collection.name, query, await collection.aretrieve(query, filters=filters))

    return FunctionTool.from_defaults(
        async_fn=search,
        name=search_tool_name(collection.name),
        description=(
            f"Searches the '{collection.name}' collection and returns the most relevant passages, each with "
            f"its metadata (source file, page and any tags). Contents: {collection.description} "
            "Optional `filters` restricts results to exact metadata matches, e.g. {\"version\": \"2024\"}."
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
