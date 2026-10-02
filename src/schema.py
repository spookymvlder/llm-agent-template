from __future__ import annotations

from dataclasses import dataclass

# Column holding each document's stable id (always set by ingestion; not stored as chunk metadata).
DOC_ID = "doc_id"
# Chunk metadata set by file ingestion: path relative to the collection folder, and the file's SHA-256.
SOURCE_PATH = "source_path"
FILE_HASH = "file_hash"

# Metadata the LLM never needs to see. Everything else (file_name, page_label, source_path,
# _metadata.json values, table columns) is shown to the LLM alongside each chunk.
BOOKKEEPING_KEYS: tuple[str, ...] = (
    FILE_HASH, "file_path", "file_size", "file_type",
    "creation_date", "last_modified_date", "last_accessed_date",
)


@dataclass(frozen=True)
class DocumentSchema:
    """How one collection's rows become LlamaIndex Documents.

    File collections (the default) need no configuration: each file is read with LlamaIndex's
    readers (one document per PDF page, one per text file, ...) into a `text` column.

    Tabular collections (`tabular=True`) read every .csv / .jsonl / .parquet file in the collection
    folder as one document per row — e.g. a table of papers with `title` and `abstract` columns.

    Args:
        text_columns:        Columns joined (with text_separator) to form the text that is embedded.
        id_column:           Column with a unique, stable id per row (tabular only). Without one, the row
                             number is used, so reordering a table re-embeds it as different documents.
        metadata_columns:    Columns stored as metadata. None = every column that isn't text.
        embed_metadata_keys: Metadata included in the embedding text. Default none: vectors represent
                             content only, so tags like an edition don't skew similarity.
        hidden_llm_metadata_keys: Metadata kept from the LLM (it is still stored and filterable).
        tabular:             Read tables as one document per row instead of one document per file.
        text_separator:      Joins multiple text columns.
    """
    text_columns: tuple[str, ...] = ("text",)
    id_column: str | None = None
    metadata_columns: tuple[str, ...] | None = None
    embed_metadata_keys: tuple[str, ...] = ()
    hidden_llm_metadata_keys: tuple[str, ...] = BOOKKEEPING_KEYS
    tabular: bool = False
    text_separator: str = "\n\n"

    def metadata_for(self, columns) -> list[str]:
        """The metadata columns to store, given the DataFrame's columns."""
        skip = {*self.text_columns, DOC_ID}
        if self.metadata_columns is None:
            return [c for c in columns if c not in skip]
        # Ingestion bookkeeping is always kept so changed files can be replaced.
        keep = [*self.metadata_columns, SOURCE_PATH, FILE_HASH]
        return [c for c in dict.fromkeys(keep) if c in columns and c not in skip]
