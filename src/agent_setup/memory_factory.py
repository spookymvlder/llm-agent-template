from __future__ import annotations

import logging
import time
import uuid
from collections import OrderedDict
from typing import Any, Callable

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


class ConversationStore:
    """One Memory per conversation id, so concurrent API clients don't share chat history.

    Memories live in-process: they are lost on restart, and with multiple server workers each worker
    has its own store. Idle conversations expire after ttl_s; beyond max_conversations the least
    recently used is dropped.

    Args:
        factory:           Builds a Memory for a new conversation id (e.g. a partial of build_memory).
        max_conversations: Cap on stored conversations.
        ttl_s:             Seconds of inactivity before a conversation is forgotten. 0 = never.
    """

    def __init__(self, factory: Callable[[str], Memory], max_conversations: int = 100, ttl_s: float = 3600) -> None:
        self._factory = factory
        self._max = max_conversations
        self._ttl = ttl_s
        self._items: OrderedDict[str, tuple[Memory, float]] = OrderedDict()

    def get(self, conversation_id: str | None = None) -> tuple[str, Memory]:
        """Return (id, memory), creating the conversation if the id is new or None (a random id is assigned)."""
        self._evict()
        conversation_id = conversation_id or uuid.uuid4().hex
        if conversation_id in self._items:
            memory, _ = self._items.pop(conversation_id)
        else:
            memory = self._factory(conversation_id)
        self._items[conversation_id] = (memory, time.monotonic())
        while len(self._items) > self._max:
            dropped, _ = self._items.popitem(last=False)
            log.info("Conversation store full; dropped least recently used conversation %s.", dropped)
        return conversation_id, memory

    async def reset(self, conversation_id: str) -> bool:
        """Clear a conversation's history. Returns False if the id is unknown (or expired)."""
        self._evict()
        item = self._items.pop(conversation_id, None)
        if item is None:
            return False
        await item[0].areset()
        return True

    def __len__(self) -> int:
        self._evict()
        return len(self._items)

    def _evict(self) -> None:
        if not self._ttl:
            return
        cutoff = time.monotonic() - self._ttl
        expired = [cid for cid, (_, last_used) in self._items.items() if last_used < cutoff]
        for cid in expired:
            del self._items[cid]

