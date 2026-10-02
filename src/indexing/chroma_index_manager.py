from __future__ import annotations

import logging
import pandas as pd
import chromadb

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.schema import BaseNode
from llama_index.core.storage import StorageContext
from llama_index.core.vector_stores import MetadataFilters
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.preprocessing import safe_meta
from src.schema import DOC_ID, DocumentSchema

log = logging.getLogger(__name__)

# LlamaIndex stores each chunk's parent document id under this Chroma metadata key.
_DOC_ID_KEY = "document_id"
# Max ids per Chroma `$in` lookup when checking which documents already exist.
_LOOKUP_BATCH = 500


@dataclass
class IndexBuildResult:
    index: VectorStoreIndex
    newly_processed: list[str] = field(default_factory=list)


class ChromaIndexManager:
    """Maintains a ChromaDB-backed VectorStoreIndex for one collection.

    - Accepts a pandas DataFrame as input instead of raw files.
    - Uses document IDs to skip already-embedded rows (idempotent inserts).
      Each document is split into chunks at embedding time; Chroma stores one
      entry per chunk, tagged with its parent document id.
    - Returns a LlamaIndex VectorStoreIndex so the rest of your agent code
      (query engine, tools) needs no changes.

    Expects DataFrames shaped by the collection's DocumentSchema (see IndexManager / preprocess()):
    a `doc_id` column, the schema's text columns and metadata columns. Rows without a `doc_id`
    fall back to the row index, which is only stable if the DataFrame is.

    Args:
        chroma_dir:     Directory where ChromaDB persists its data.
        collection_name: Name of the ChromaDB collection to use.
        schema:         Which columns are text vs metadata, and which metadata the embedder and LLM see.
        distance_metric: HNSW space for new collections: 'cosine', 'l2' or 'ip'.
                        Ignored for collections that already exist.
        client:         Shared ChromaDB client. One is created for chroma_dir if not given.
    """

    def __init__(
        self,
        *,
        chroma_dir: Path,
        collection_name: str = "documents",
        schema: DocumentSchema | None = None,
        distance_metric: str = "cosine",
        client: chromadb.ClientAPI | None = None,
    ) -> None:
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.schema = schema or DocumentSchema()
        self.distance_metric = distance_metric

        if client is None:
            self.chroma_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self._client = client


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def client(self) -> chromadb.ClientAPI:
        return self._client

    def count(self) -> int:
        """Number of stored chunks (not documents) in the collection."""
        return self._collection().count()

    def load_or_build(self, df: pd.DataFrame | None = None) -> IndexBuildResult:
        """Load the index, embedding any rows from df whose document IDs are not already stored.

        With df=None (or all rows already stored) the index is loaded as-is. An empty
        collection yields an empty index, which is valid — documents can be added later.
        """
        collection = self._collection()
        vector_store = ChromaVectorStore(chroma_collection=collection)
        log.info("Collection '%s': %d chunk(s) indexed.", self.collection_name, collection.count())

        newly_processed: list[str] = []
        if df is not None and not df.empty:
            new_docs = self._new_documents(df, collection)
            if new_docs:
                log.info("Embedding %d new document(s) into '%s'...", len(new_docs), self.collection_name)
                # from_documents appends to the existing vector store; it does not replace it.
                VectorStoreIndex.from_documents(
                    new_docs,
                    storage_context=StorageContext.from_defaults(vector_store=vector_store),
                    show_progress=len(new_docs) > 20,   # quiet for streamed fragments and small batches
                )
                newly_processed = [d.doc_id for d in new_docs]
                log.info("Indexed %d new document(s).", len(new_docs))
            else:
                log.info("No new documents to index.")

        return IndexBuildResult(
            index=VectorStoreIndex.from_vector_store(vector_store),
            newly_processed=newly_processed,
        )

    def nodes_where(self, filters: MetadataFilters | None) -> list[BaseNode]:
        """Every stored chunk matching `filters` (all chunks if None), as LlamaIndex nodes. No similarity search."""
        return ChromaVectorStore(chroma_collection=self._collection()).get_nodes(node_ids=None, filters=filters)

    def count_where(self, key: str, value: Any) -> int:
        """Number of chunks whose metadata `key` equals `value`."""
        return len(self._collection().get(where={key: value}, include=[])["ids"])

    def delete_where(self, key: str, values: list[str]) -> int:
        """Delete every chunk whose metadata `key` is one of `values` (e.g. all chunks of a changed file).

        Returns:
            Number of chunks deleted.
        """
        collection = self._collection()
        before = collection.count()
        for i in range(0, len(values), _LOOKUP_BATCH):
            collection.delete(where={key: {"$in": values[i:i + _LOOKUP_BATCH]}})
        return before - collection.count()

    def delete_collection(self) -> None:
        """Delete the collection and all its vectors. The next load_or_build() recreates it empty."""
        try:
            self._client.delete_collection(self.collection_name)
            log.info("Deleted collection '%s'.", self.collection_name)
        except Exception as e:  # chromadb raises different types across versions when it doesn't exist
            log.info("Collection '%s' not deleted (%s).", self.collection_name, e)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collection(self):
        return self._client.get_or_create_collection(
            self.collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )

    def _row_id(self, row: pd.Series, idx: int) -> str:
        if DOC_ID in row.index:
            return str(row[DOC_ID])
        return str(idx)

    def _to_document(self, doc_id: str, row: pd.Series, meta_cols: list[str]) -> Document:
        schema = self.schema
        text = schema.text_separator.join(
            str(row[c]) for c in schema.text_columns if c in row.index and str(row[c]).strip()
        )
        metadata = {col: value for col in meta_cols if (value := safe_meta(row.get(col))) is not None}
        return Document(
            text=text,
            metadata=metadata,
            id_=doc_id,
            # Embeddings represent content only (plus any embed_metadata_keys); the LLM sees all but bookkeeping.
            excluded_embed_metadata_keys=[k for k in metadata if k not in schema.embed_metadata_keys],
            excluded_llm_metadata_keys=[k for k in metadata if k in schema.hidden_llm_metadata_keys],
        )

    def _existing_doc_ids(self, collection, doc_ids: list[str]) -> set[str]:
        existing: set[str] = set()
        for i in range(0, len(doc_ids), _LOOKUP_BATCH):
            batch = doc_ids[i:i + _LOOKUP_BATCH]
            found = collection.get(where={_DOC_ID_KEY: {"$in": batch}}, include=["metadatas"])
            existing.update(m[_DOC_ID_KEY] for m in found["metadatas"] if m and _DOC_ID_KEY in m)
        return existing

    def _new_documents(self, df: pd.DataFrame, collection) -> list[Document]:
        meta_cols = self.schema.metadata_for(df.columns)
        rows = [(self._row_id(row, idx), row) for idx, row in df.iterrows()]
        existing = self._existing_doc_ids(collection, [doc_id for doc_id, _ in rows])

        docs: list[Document] = []
        seen: set[str] = set()
        for doc_id, row in rows:
            if doc_id in existing or doc_id in seen:
                continue
            seen.add(doc_id)
            docs.append(self._to_document(doc_id, row, meta_cols))

        skipped = len(rows) - len(docs)
        if skipped:
            log.info("Skipped %d document(s) already in '%s' or duplicated in the input.", skipped, self.collection_name)
        return docs

