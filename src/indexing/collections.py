from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

import chromadb
from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import BaseNode, MetadataMode, NodeWithScore
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from src.config_helpers import CollectionSettings
from src.indexing.chroma_index_manager import ChromaIndexManager
from src.indexing.index_manager import IndexManager
from src.indexing.streams import Fragment, Granularity, StreamInfo, StreamManager, common_metadata
from src.preprocessing import PreprocessStep, preprocess
from src.schema import (
    DESCRIPTION_KEY,
    DOC_ID,
    ORIGIN,
    ORIGIN_API,
    ORIGIN_STREAM,
    SOURCE_PATH,
    STREAM_ID,
    DocumentSchema,
    FileMode,
)

log = logging.getLogger(__name__)


@dataclass
class CollectionOptions:
    """Code-level customisation for one collection (config covers names, folders, top_k, description).

    Args:
        schema:         How rows become documents: text columns, tabular mode, metadata visibility.
        steps:          Preprocessing functions (DataFrame -> DataFrame) run before sanitisation, e.g.
                        tagging rows or deriving columns. See src/preprocessing/pipeline.py.
        postprocessors: LlamaIndex node postprocessors run after retrieval, before the LLM.

    Changing the schema or steps doesn't re-embed existing documents; run `ingest -c <name> --reindex`.
    """
    schema: DocumentSchema = field(default_factory=DocumentSchema)
    steps: Sequence[PreprocessStep] = ()
    postprocessors: Sequence[BaseNodePostprocessor] = ()


@dataclass
class RetrievedChunk:
    """One retrieved chunk, as returned to API callers and project code (e.g. to branch on metadata)."""
    collection: str
    doc_id: str | None
    text: str
    score: float | None
    metadata: dict[str, Any]

    @classmethod
    def from_node(cls, collection: str, node: NodeWithScore) -> RetrievedChunk:
        return cls(
            collection=collection,
            doc_id=node.node.ref_doc_id,
            text=node.node.get_content(metadata_mode=MetadataMode.NONE),
            score=node.score,
            metadata=dict(node.node.metadata),
        )


def metadata_filters(filters: Mapping[str, Any] | None) -> MetadataFilters | None:
    """Exact-match filters, e.g. {"version": "2024"} -> MetadataFilters. Values must match the stored type."""
    if not filters:
        return None
    return MetadataFilters(filters=[MetadataFilter(key=k, value=v) for k, v in filters.items()])


@dataclass
class SyncResult:
    """What sync() did. File lists hold paths relative to the collection folder."""
    new_files: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)     # re-embedded
    pending_files: list[str] = field(default_factory=list)     # new/changed MANUAL files awaiting an explicit ingest
    missing_files: list[str] = field(default_factory=list)
    streams_embedded: list[str] = field(default_factory=list)  # streams replayed (after a reindex) or re-embedded
    documents_added: int = 0
    chunks_removed: int = 0


@dataclass
class CollectionHandle:
    """Everything needed to ingest into and search one named collection.

    Documents arrive three ways: files in the collection folder (sync), runtime documents
    (add_documents / POST /documents), and streams (append_to_stream / POST /streams/...).

    Node postprocessors run after retrieval and before anything reaches the LLM — the place for
    deterministic, metadata-driven handling (e.g. flagging chunks from a superseded version of a policy).
    They apply to as_query_engine() and aretrieve(); a raw index.as_retriever() bypasses them.
    """
    settings: CollectionSettings
    store: ChromaIndexManager
    files: IndexManager
    streams: StreamManager
    index: VectorStoreIndex
    options: CollectionOptions = field(default_factory=CollectionOptions)

    @property
    def postprocessors(self) -> list[BaseNodePostprocessor]:
        return list(self.options.postprocessors)

    @property
    def name(self) -> str:
        return self.settings.name

    @property
    def description(self) -> str:
        return self.settings.description

    def count(self) -> int:
        """Number of stored chunks."""
        return self.store.count()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def as_query_engine(self, **kwargs: Any) -> BaseQueryEngine:
        """Query engine (retrieve + synthesize an answer) with this collection's top_k and postprocessors."""
        kwargs.setdefault("similarity_top_k", self.settings.top_k)
        kwargs.setdefault("node_postprocessors", list(self.postprocessors))
        return self.index.as_query_engine(**kwargs)

    async def aretrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
        **retriever_kwargs: Any,
    ) -> list[NodeWithScore]:
        """Retrieve chunks (with scores and metadata) without synthesizing an answer, applying postprocessors.

        Args:
            query:   Search text.
            top_k:   Defaults to the collection's top_k.
            filters: Exact-match metadata filters, e.g. {"version": "2024"}.
            retriever_kwargs: Passed to index.as_retriever().
        """
        if filters:
            retriever_kwargs["filters"] = metadata_filters(filters)
        retriever = self.index.as_retriever(similarity_top_k=top_k or self.settings.top_k, **retriever_kwargs)
        bundle = QueryBundle(query)
        nodes = await retriever.aretrieve(bundle)
        for postprocessor in self.postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query_bundle=bundle)
        return nodes

    async def search(self, query: str, top_k: int | None = None, filters: Mapping[str, Any] | None = None) -> list[RetrievedChunk]:
        """aretrieve() as plain RetrievedChunk objects."""
        return [RetrievedChunk.from_node(self.name, n) for n in await self.aretrieve(query, top_k, filters)]

    def get_chunks(self, filters: Mapping[str, Any] | None = None, limit: int | None = None) -> list[BaseNode]:
        """Every chunk matching exact-match `filters`, in document order (not similarity order) —
        e.g. all of a meeting's notes, for summarising. Postprocessors are not applied.

        Raises:
            ValueError: More than `limit` chunks match.
        """
        nodes = self.store.nodes_where(metadata_filters(filters))
        if limit is not None and len(nodes) > limit:
            raise ValueError(f"{len(nodes)} chunks match {dict(filters or {})} in '{self.name}' (limit {limit}); narrow the filters.")
        return sorted(nodes, key=_document_order)

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    def sync(self, include_manual: bool = False) -> SyncResult:
        """Bring the vector store in line with the collection folder and the stream files.

        Files: new and changed STATIC files are embedded (a changed file's old chunks are deleted first);
        MANUAL files are only embedded when include_manual, otherwise they are reported as pending.
        The manifest is only updated after embedding succeeds, so an interrupted sync is retried on the
        next run. Files missing from disk are reported but their chunks are kept (reindex to remove them).

        Streams: replayed after a reindex, or re-embedded if a closed stream's file was edited. Open
        streams are otherwise left alone — their fragments are embedded as they arrive.

        Blocking (embeds synchronously) — call via a thread from async code.
        """
        changes = self.files.find_changes(include_manual=include_manual)
        to_embed = changes.new + changes.changed
        result = SyncResult(
            new_files=[r.rel_path for r in changes.new],
            changed_files=[r.rel_path for r in changes.changed],
            pending_files=[r.rel_path for r in changes.pending],
            missing_files=changes.missing,
        )

        if result.pending_files:
            log.warning(
                "Collection '%s': %d manual file(s) changed and waiting for `ingest --manual`: %s",
                self.name, len(result.pending_files), ", ".join(result.pending_files),
            )
        if result.missing_files:
            log.warning(
                "Collection '%s': %d file(s) no longer on disk; their chunks are kept until you reindex: %s",
                self.name, len(result.missing_files), ", ".join(result.missing_files),
            )

        if to_embed:
            if result.changed_files:
                result.chunks_removed = self.store.delete_where(SOURCE_PATH, result.changed_files)
            df = preprocess(self.files.load_as_dataframe(to_embed), self.options.schema, self.options.steps)
            built = self.store.load_or_build(df)
            result.documents_added = len(built.newly_processed)
        self.files.commit(to_embed + changes.touched)

        for info in self.streams.needing_embedding():
            result.chunks_removed += self.store.delete_where(STREAM_ID, [info.stream_id])
            result.documents_added += self._embed_stream(info)
            self.streams.mark_embedded(info.stream_id)
            result.streams_embedded.append(info.stream_id)
        return result

    # ------------------------------------------------------------------
    # Runtime documents
    # ------------------------------------------------------------------

    def add_documents(self, documents: Sequence[Mapping[str, Any]]) -> list[str]:
        """Add (or replace) documents at runtime. Blocking: embeds synchronously — call via a thread from async code.

        Each document is {"text": str, "id"?: str, "metadata"?: dict}. Documents with an id that already
        exists replace the stored version; documents without one get a random id. They run through the
        collection's preprocessing steps and sanitiser like ingested files, and are tagged
        ingested_via="api". They have no source file, so `ingest --reindex` deletes them — use a stream
        or a file in the collection folder for anything that must survive a reindex.

        Returns:
            The ids of the stored documents (documents whose text is empty after cleaning are dropped).
        """
        rows = [
            self._row(doc["text"], str(doc.get("id") or uuid.uuid4()), doc.get("metadata"), {ORIGIN: ORIGIN_API})
            for doc in documents
        ]
        ids = self._embed_rows(rows, replace=True)
        log.info("Collection '%s': stored %d runtime document(s).", self.name, len(ids))
        return ids

    def runtime_chunk_count(self) -> int:
        """Chunks added via add_documents(); these are lost on reset() because they have no source file."""
        return self.store.count_where(ORIGIN, ORIGIN_API)

    # ------------------------------------------------------------------
    # Streams
    # ------------------------------------------------------------------

    def append_to_stream(self, stream_id: str, text: str, metadata: Mapping[str, Any] | None = None) -> str:
        """Append a fragment to a stream (creating it if new) and embed it at once. Blocking.

        The fragment is written to the stream's file before it is embedded, so the record is kept even
        if embedding fails (the fragment then becomes searchable when the stream is closed/re-chunked).

        Returns:
            The fragment's document id, '<stream_id>#<seq>'.

        Raises:
            StreamError / StreamClosed: Invalid stream id, or the stream is closed.
        """
        fragment = self.streams.append(stream_id, text, dict(metadata or {}))
        doc_id = f"{stream_id}#{fragment.seq}"
        self._embed_rows([self._fragment_row(stream_id, fragment)], replace=False)
        return doc_id

    def close_stream(
        self,
        stream_id: str,
        rechunk: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> StreamInfo:
        """Close a stream. With rechunk (the default), its fragment chunks are replaced by the whole
        stream embedded as one document — longer chunks with context retrieve better than many short
        fragments. The document carries `metadata`, plus any metadata shared by every fragment. Blocking.

        Raises:
            StreamNotFound: No such stream.
        """
        info = self.streams.close(stream_id, Granularity.DOCUMENT if rechunk else Granularity.FRAGMENTS, dict(metadata or {}))
        if rechunk:
            removed = self.store.delete_where(STREAM_ID, [stream_id])
            self._embed_stream(info)
            log.info("Collection '%s': closed stream '%s' and re-chunked it (%d fragment chunk(s) replaced).",
                     self.name, stream_id, removed)
        self.streams.mark_embedded(stream_id)
        return self.streams.info(stream_id)

    def _embed_stream(self, info: StreamInfo) -> int:
        """Embed a stream from its file, per its granularity. Returns documents added."""
        fragments = self.streams.read(info.stream_id)
        if not fragments:
            return 0
        if info.granularity == Granularity.FRAGMENTS:
            rows = [self._fragment_row(info.stream_id, f) for f in fragments]
        else:
            text = "\n".join(f.text for f in fragments)
            metadata = {**common_metadata(fragments), **info.close_metadata}
            rows = [self._row(text, info.stream_id, metadata, {ORIGIN: ORIGIN_STREAM, STREAM_ID: info.stream_id})]
        return len(self._embed_rows(rows, replace=False))

    def _fragment_row(self, stream_id: str, fragment: Fragment) -> dict[str, Any]:
        return self._row(fragment.text, f"{stream_id}#{fragment.seq}", fragment.metadata,
                         {ORIGIN: ORIGIN_STREAM, STREAM_ID: stream_id})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row(self, text: str, doc_id: str, metadata: Mapping[str, Any] | None, system: Mapping[str, Any]) -> dict[str, Any]:
        """A DataFrame row: user metadata, then text and id, then system keys (which always win)."""
        return {**(metadata or {}), self.options.schema.text_columns[0]: text, DOC_ID: doc_id, **system}

    def _embed_rows(self, rows: list[dict[str, Any]], replace: bool) -> list[str]:
        df = preprocess(pd.DataFrame(rows), self.options.schema, self.options.steps)
        if df.empty:
            return []
        ids = df[DOC_ID].tolist()
        if replace:
            self.store.delete_where("document_id", ids)
        self.store.load_or_build(df)
        return ids

    def reset(self) -> None:
        """Delete every vector and the file manifest, leaving an empty collection. The next sync()
        re-embeds files and replays streams.

        Documents added with add_documents (POST /documents) are deleted too and are not restored —
        they have no source file. Streams are kept: they replay from their files.
        """
        runtime = self.runtime_chunk_count()
        if runtime:
            log.warning("Collection '%s': deleting %d chunk(s) added via POST /documents; they have no source file to re-ingest.",
                        self.name, runtime)
        self.store.delete_collection()
        self.files.reset_manifest()
        self.streams.clear_embedded()
        self.index = self.store.load_or_build().index


def _resolve_description(settings: CollectionSettings, files: IndexManager) -> str:
    """COLLECTION_<NAME>_DESCRIPTION from .env, else "_description" in the collection's root _metadata.json,
    else a generic default (logged, since the agent then can't tell what the collection is for)."""
    if settings.description:
        return settings.description
    from_file = files.root_metadata().get(DESCRIPTION_KEY)
    if isinstance(from_file, str) and from_file.strip():
        return from_file.strip()
    log.warning(
        "Collection '%s' has no description, so the agent can't tell what it contains or when to search it. "
        'Add {"%s": "..."} to %s.', settings.name, DESCRIPTION_KEY, settings.raw_dir / "_metadata.json",
    )
    return f"Documents in the '{settings.name}' collection."


def _document_order(node: BaseNode) -> tuple:
    """Sort key: document id with numeric '#n' suffixes compared as numbers (stream fragments, PDF pages),
    then position within the document."""
    doc_id = node.ref_doc_id or node.node_id
    base, _, suffix = doc_id.rpartition("#")
    seq = (base, int(suffix)) if suffix.isdigit() and base else (doc_id, -1)
    return (*seq, getattr(node, "start_char_idx", None) or 0)


def open_collection(
    settings: CollectionSettings,
    *,
    client: chromadb.ClientAPI,
    chroma_dir: Path,
    manifest_dir: Path,
    stream_dir: Path,
    distance_metric: str,
    default_mode: FileMode = FileMode.STATIC,
    options: CollectionOptions | None = None,
) -> CollectionHandle:
    """Open (creating if needed) a collection. Does not ingest anything.

    Args:
        stream_dir: Environment-local folder for this collection's stream files.
    """
    options = options or CollectionOptions()
    files = IndexManager(
        raw_dir=settings.raw_dir,
        manifest_path=manifest_dir / f"{settings.name}.json",
        schema=options.schema,
        default_mode=default_mode,
    )
    settings = replace(settings, description=_resolve_description(settings, files))
    store = ChromaIndexManager(
        chroma_dir=chroma_dir,
        collection_name=settings.name,
        schema=options.schema,
        distance_metric=distance_metric,
        client=client,
    )
    return CollectionHandle(
        settings=settings,
        store=store,
        files=files,
        streams=StreamManager(
            stream_dir=stream_dir,
            manifest_path=manifest_dir / f"{settings.name}.streams.json",
        ),
        index=store.load_or_build().index,
        options=options,
    )
