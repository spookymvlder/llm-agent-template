"""Summarise every chunk matching a metadata filter — e.g. all of a meeting's notes."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from llama_index.core import Settings
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.schema import MetadataMode

from src.indexing import CollectionHandle

log = logging.getLogger(__name__)

DEFAULT_INSTRUCTION = (
    "Summarise these passages in order. Keep names, decisions, open questions and anything that "
    "should be remembered later. Be concise."
)


@dataclass
class Summary:
    text: str
    chunks: int
    doc_ids: list[str] = field(default_factory=list)


async def summarize_documents(
    collection: CollectionHandle,
    filters: Mapping[str, Any] | None = None,
    instruction: str = DEFAULT_INSTRUCTION,
    llm: Any | None = None,
    max_chunks: int = 2000,
) -> Summary:
    """Summarise all chunks in `collection` matching `filters`, in document order.

    Uses LlamaIndex's tree summarisation: chunks are packed into context-window-sized groups, each group
    is summarised, then the summaries are summarised, so input size isn't limited by the context window.
    Chunk text is passed with its LLM-visible metadata (e.g. speaker, source) so the summary can use it.

    Example — minutes for a finished meeting:
        summary = await summarize_documents(ctx.collection("meetings"), {"stream_id": "standup-2026-10-02"})

    Args:
        collection:  Collection to read.
        filters:     Exact-match metadata filters selecting the chunks. None = the whole collection.
        instruction: What the summary should focus on.
        llm:         Defaults to the primary LLM (Settings.llm).
        max_chunks:  Refuse to summarise more than this many chunks (guards against a missing filter).

    Raises:
        ValueError: No chunks, or more than max_chunks, match.
    """
    nodes = collection.get_chunks(filters, limit=max_chunks)
    if not nodes:
        raise ValueError(f"No chunks in '{collection.name}' match {dict(filters or {})}.")

    texts = [n.get_content(metadata_mode=MetadataMode.LLM) for n in nodes]
    log.info("Summarising %d chunk(s) from '%s' (filters=%s).", len(texts), collection.name, dict(filters or {}))
    synthesizer = TreeSummarize(llm=llm or Settings.llm)
    text = await synthesizer.aget_response(query_str=instruction, text_chunks=texts)
    return Summary(
        text=str(text).strip(),
        chunks=len(nodes),
        doc_ids=list(dict.fromkeys(n.ref_doc_id or n.node_id for n in nodes)),
    )
