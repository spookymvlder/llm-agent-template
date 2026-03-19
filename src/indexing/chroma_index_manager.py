from __future__ import annotations

import logging
import pandas as pd
import chromadb

from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

log = logging.getLogger(__name__)


@dataclass
class IndexBuildResult:
    index: VectorStoreIndex
    newly_processed: list[str] = field(default_factory=list)


class ChromaIndexManager:
    """Maintains a ChromaDB-backed VectorStoreIndex.

    - Accepts a pandas DataFrame as input instead of raw files.
    - Uses document IDs to skip already-embedded rows (idempotent upserts).
    - Returns a LlamaIndex VectorStoreIndex so the rest of your agent code
      (query engine, tools) needs no changes.

    Args:
        chroma_dir:     Directory where ChromaDB persists its data.
        collection_name: Name of the ChromaDB collection to use.
        text_column:    DataFrame column containing the text to embed.
        id_column:      DataFrame column to use as the document ID.
                        Must be unique per row. If None, row index is used.
        metadata_columns: Columns to store as document metadata.
                        If None, all columns except text_column are used.
    """

    def __init__(
        self,
        *,
        chroma_dir: Path,
        collection_name: str = "documents",
        text_column: str = "text",
        id_column: str | None = None,
        metadata_columns: list[str] | None = None,
    ) -> None:
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.text_column = text_column
        self.id_column = id_column
        self.metadata_columns = metadata_columns

        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.chroma_dir))


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def client(self) -> chromadb.ClientAPI:
        return self._client

    def load_or_build(self, df: pd.DataFrame | None = None) -> IndexBuildResult:
        """Load the existing index, optionally adding new rows from df.

        - If no collection exists yet, df is required to build from scratch.
        - If a collection exists, only rows whose IDs are not already stored
          will be embedded and inserted.
        - If df is None and the collection already exists, the index is loaded
          as-is (useful when re-attaching to a fully-built index at startup).
        """
        collection = self._client.get_or_create_collection(self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        existing_ids: set[str] = set(collection.get(include=[])["ids"])
        log.info("Collection '%s': %d documents already indexed.", self.collection_name, len(existing_ids))

        newly_processed: list[str] = []

        if df is not None:
            new_docs, new_ids = self._new_documents(df, existing_ids)

            if new_docs:
                log.info("Embedding %d new document(s)...", len(new_docs))
                if existing_ids:
                    # Add to existing index
                    index = VectorStoreIndex.from_vector_store(vector_store)
                    for doc in new_docs:
                        index.insert(doc)
                else:
                    # Build fresh index
                    index = VectorStoreIndex.from_documents(
                        new_docs,
                        storage_context=storage_context,
                        show_progress=True,
                    )
                newly_processed = new_ids
                log.info("Indexed %d new document(s).", len(new_docs))
            else:
                log.info("No new documents to index.")
                index = VectorStoreIndex.from_vector_store(vector_store)
        else:
            if not existing_ids:
                raise ValueError(
                    "Collection is empty and no DataFrame was provided. "
                    "Pass a DataFrame to build_or_load() on first run."
                )
            log.info("Loading existing index (no DataFrame provided).")
            index = VectorStoreIndex.from_vector_store(vector_store)

        return IndexBuildResult(index=index, newly_processed=newly_processed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_metadata_columns(self, df: pd.DataFrame) -> list[str]:
        if self.metadata_columns is not None:
            return self.metadata_columns
        return [c for c in df.columns if c != self.text_column]

    def _row_id(self, row: pd.Series, idx: int) -> str:
        if self.id_column and self.id_column in row.index:
            return str(row[self.id_column])
        return str(idx)

    def _new_documents(
        self, df: pd.DataFrame, existing_ids: set[str]
    ) -> tuple[list[Document], list[str]]:
        meta_cols = self._resolve_metadata_columns(df)
        docs: list[Document] = []
        ids: list[str] = []

        for idx, row in df.iterrows():
            doc_id = self._row_id(row, idx)
            if doc_id in existing_ids:
                continue

            metadata = {col: _safe_meta(row.get(col)) for col in meta_cols}
            docs.append(
                Document(
                    text=str(row[self.text_column]),
                    metadata=metadata,
                    id_=doc_id,
                )
            )
            ids.append(doc_id)

        return docs, ids


def _safe_meta(value) -> str | int | float | bool:
    """ChromaDB metadata values must be scalar — coerce everything else to str."""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)