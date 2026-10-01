from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import chromadb
from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore

from src.config_helpers import CollectionSettings
from src.indexing.chroma_index_manager import ChromaIndexManager
from src.indexing.index_manager import IndexManager

log = logging.getLogger(__name__)

# Chunk metadata key holding the file's path relative to the collection folder (set by IndexManager).
_SOURCE_PATH_KEY = "source_path"


@dataclass
class SyncResult:
    """What sync_files() did. File lists hold paths relative to the collection folder."""
    new_files: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)     # re-embedded
    skipped_changed: list[str] = field(default_factory=list)   # changed but REINDEX_CHANGED_FILES=false
    missing_files: list[str] = field(default_factory=list)
    documents_added: int = 0
    chunks_removed: int = 0


@dataclass
class CollectionHandle:
    """Everything needed to ingest into and search one named collection.

    Node postprocessors run after retrieval and before anything reaches the LLM — the place for
    deterministic, metadata-driven handling (e.g. flagging chunks from an older edition of a rulebook).
    They apply to as_query_engine() and aretrieve(); a raw index.as_retriever() bypasses them.
    """
    settings: CollectionSettings
    store: ChromaIndexManager
    files: IndexManager
    index: VectorStoreIndex
    postprocessors: list[BaseNodePostprocessor] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.settings.name

    @property
    def description(self) -> str:
        return self.settings.description

    def count(self) -> int:
        """Number of stored chunks."""
        return self.store.count()

    def as_query_engine(self, **kwargs: Any) -> BaseQueryEngine:
        """Query engine (retrieve + synthesize an answer) with this collection's top_k and postprocessors."""
        kwargs.setdefault("similarity_top_k", self.settings.top_k)
        kwargs.setdefault("node_postprocessors", list(self.postprocessors))
        return self.index.as_query_engine(**kwargs)

    async def aretrieve(self, query: str, top_k: int | None = None, **retriever_kwargs: Any) -> list[NodeWithScore]:
        """Retrieve chunks (with scores and metadata) without synthesizing an answer, applying postprocessors.

        retriever_kwargs are passed to index.as_retriever(), e.g. filters=MetadataFilters(...).
        """
        retriever = self.index.as_retriever(similarity_top_k=top_k or self.settings.top_k, **retriever_kwargs)
        bundle = QueryBundle(query)
        nodes = await retriever.aretrieve(bundle)
        for postprocessor in self.postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query_bundle=bundle)
        return nodes

    def sync_files(self, reindex_changed: bool = True) -> SyncResult:
        """Embed new files from raw_dir and, if reindex_changed, re-embed files whose content changed.

        A changed file's old chunks are deleted before its new version is embedded. The manifest is
        only updated after embedding succeeds, so an interrupted sync is retried on the next run.
        Files missing from disk are reported but their chunks are kept (reindex to remove them).
        """
        changes = self.files.find_changes()
        to_embed = changes.new + (changes.changed if reindex_changed else [])
        result = SyncResult(
            new_files=[r.rel_path for r in changes.new],
            changed_files=[r.rel_path for r in changes.changed] if reindex_changed else [],
            skipped_changed=[] if reindex_changed else [r.rel_path for r in changes.changed],
            missing_files=changes.missing,
        )

        if result.skipped_changed:
            log.warning(
                "Collection '%s': %d changed file(s) not re-embedded (REINDEX_CHANGED_FILES=false): %s",
                self.name, len(result.skipped_changed), ", ".join(result.skipped_changed),
            )
        if result.missing_files:
            log.warning(
                "Collection '%s': %d file(s) no longer on disk; their chunks are kept until you reindex: %s",
                self.name, len(result.missing_files), ", ".join(result.missing_files),
            )

        if to_embed:
            if result.changed_files:
                result.chunks_removed = self.store.delete_where(_SOURCE_PATH_KEY, result.changed_files)
            built = self.store.load_or_build(self.files.load_as_dataframe(to_embed))
            result.documents_added = len(built.newly_processed)
        self.files.commit(to_embed + changes.touched)
        return result

    def reset(self) -> None:
        """Delete every vector and the file manifest, leaving an empty collection."""
        self.store.delete_collection()
        self.files.reset_manifest()
        self.index = self.store.load_or_build().index


def open_collection(
    settings: CollectionSettings,
    *,
    client: chromadb.ClientAPI,
    chroma_dir: Path,
    manifest_dir: Path,
    distance_metric: str,
    postprocessors: Sequence[BaseNodePostprocessor] = (),
) -> CollectionHandle:
    """Open (creating if needed) a collection. Does not ingest anything."""
    store = ChromaIndexManager(
        chroma_dir=chroma_dir,
        collection_name=settings.name,
        text_column="text",
        id_column="doc_id",
        distance_metric=distance_metric,
        client=client,
    )
    return CollectionHandle(
        settings=settings,
        store=store,
        files=IndexManager(raw_dir=settings.raw_dir, manifest_path=manifest_dir / f"{settings.name}.json"),
        index=store.load_or_build().index,
        postprocessors=list(postprocessors),
    )
