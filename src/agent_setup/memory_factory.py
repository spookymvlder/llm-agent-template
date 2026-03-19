from __future__ import annotations

import logging
from typing import Any

import chromadb
from llama_index.core.memory import (
    Memory,
    StaticMemoryBlock,
    FactExtractionMemoryBlock,
    VectorMemoryBlock,
)
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore

log = logging.getLogger(__name__)

# Note that enabling vector memory should be done in a separate ChromaDB and would require persistent sessionIDs other than the assigned UUID at startup.
def build_memory(
    session_id: str,
    llm: Any,
    token_limit: int = 40000,
    static_content: str | None = None,
    enable_fact_extraction: bool = True,
    max_facts: int = 50,
    enable_vector_memory: bool = False,
    chroma_client: chromadb.ClientAPI | None = None,
    vector_collection_name: str = "agent_memory",
) -> Memory:
    """Build a Memory object with optional long-term memory blocks.

    Args:
        session_id:               Unique identifier for this conversation session.
        llm:                      LLM used for fact extraction. Should be Settings.llm.
        token_limit:              Total token budget for short + long term memory.
        static_content:           Optional fixed information always injected into context
                                  (e.g. user preferences, application context).
        enable_fact_extraction:   Extract and persist facts from conversation history.
        max_facts:                Maximum facts to retain before summarizing.
        enable_vector_memory:     Store conversation batches in ChromaDB for retrieval.
        chroma_client:            Required if enable_vector_memory is True.
        vector_collection_name:   ChromaDB collection name for vector memory.

    Returns:
        Memory instance ready to pass to agent.run().
    """
    blocks = []

    if static_content:
        blocks.append(
            StaticMemoryBlock(
                name="static_info",
                static_content=static_content,
                priority=0,  # always retained
            )
        )
        log.debug("Memory: static block enabled")

    if enable_fact_extraction:
        blocks.append(
            FactExtractionMemoryBlock(
                name="extracted_facts",
                llm=llm,
                max_facts=max_facts,
                priority=1,
            )
        )
        log.debug("Memory: fact extraction block enabled (max_facts=%d)", max_facts)

    if enable_vector_memory:
        if chroma_client is None:
            log.warning("enable_vector_memory=True but no chroma_client provided — skipping.")
        else:
            collection = chroma_client.get_or_create_collection(vector_collection_name)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            blocks.append(
                VectorMemoryBlock(
                    name="vector_memory",
                    vector_store=vector_store,
                    priority=2,
                )
            )
            log.debug("Memory: vector block enabled (collection=%s)", vector_collection_name)

    memory = Memory.from_defaults(
        session_id=session_id,
        token_limit=token_limit,
        memory_blocks=blocks if blocks else None,
    )

    log.info(
        "Memory built: session=%s blocks=%s",
        session_id,
        [b.name for b in blocks],
    )
    return memory