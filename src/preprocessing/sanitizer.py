"""
DataFrame sanitisation for ChromaDB compatibility.

Responsibilities:
  - Validate that the schema's text columns are present
  - Normalise whitespace in text and drop rows with no text
  - Coerce metadata columns to ChromaDB-safe scalar types
    (str, int, float, bool — no None, list, dict, enum, or NaN)
  - Drop duplicate document ids
  - Log a summary of any changes made
"""
from __future__ import annotations

import enum
import json
import logging
import re
from typing import Any

import pandas as pd

from src.schema import DOC_ID, DocumentSchema

log = logging.getLogger(__name__)

_RUNS_OF_SPACES = re.compile(r"[ \t\f\v]+")
_RUNS_OF_BLANK_LINES = re.compile(r"\n\s*\n\s*\n+")


def sanitize(df: pd.DataFrame, schema: DocumentSchema) -> pd.DataFrame:
    """
    Return a sanitised copy of *df*, ready for the embedding pipeline.

    Steps performed:
        1. Validate the schema's text columns exist.
        2. Normalise whitespace in text columns (paragraph breaks are kept — the
           splitter uses them) and drop rows whose text is empty.
        3. Coerce all metadata columns to ChromaDB-safe scalars.
        4. Drop duplicate document ids (keep first occurrence).
        5. Log a change summary.

    Raises:
        ValueError: A text column is missing.
    """
    if df.empty:
        return df
    df = df.copy()
    original_len = len(df)

    # 1. Validate text columns -----------------------------------------------
    missing = [c for c in schema.text_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing text column(s) {missing}. Present columns: {df.columns.tolist()}"
        )

    # 2. Normalise text, drop empty rows -------------------------------------
    for col in schema.text_columns:
        df[col] = df[col].map(normalize_text)
    empty = (df[list(schema.text_columns)] == "").all(axis=1)
    if empty.any():
        log.info("Dropped %d row(s) with no text (e.g. blank PDF pages).", int(empty.sum()))
        df = df[~empty]

    # 3. Coerce metadata to ChromaDB-safe scalars ----------------------------
    for col in schema.metadata_for(df.columns):
        df[col] = df[col].map(safe_meta).astype(object)

    # 4. Drop duplicate ids --------------------------------------------------
    if DOC_ID in df.columns:
        df[DOC_ID] = df[DOC_ID].astype(str)
        before = len(df)
        df = df.drop_duplicates(subset=[DOC_ID], keep="first")
        if before - len(df):
            log.warning("Dropped %d row(s) with a duplicate %s.", before - len(df), DOC_ID)

    # 5. Summary -------------------------------------------------------------
    log.info("Sanitisation complete: %d rows in → %d rows out.", original_len, len(df))
    return df


def normalize_text(value: Any) -> str:
    """Collapse runs of spaces/tabs and of blank lines; keep single newlines and paragraph breaks."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (ValueError, TypeError):
        pass
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = _RUNS_OF_SPACES.sub(" ", text)
    text = _RUNS_OF_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def safe_meta(value: Any) -> str | int | float | bool:
    """
    Coerce *value* to a type that ChromaDB accepts as metadata.

    ChromaDB only accepts str, int, float, or bool as metadata values.
    Handles the richer input types (enums, lists, dicts, pandas NA, numpy scalars)
    that come out of tables and preprocessing steps.
    """
    # --- None / NaN ---------------------------------------------------------
    if value is None:
        return ""
    # Empty collections before pd.isna (which raises on non-scalars)
    if isinstance(value, (list, tuple, set)) and len(value) == 0:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (ValueError, TypeError):
        pass  # pd.isna doesn't work on lists/arrays — continue

    # --- Enum ---------------------------------------------------------------
    if isinstance(value, enum.Enum):
        return safe_meta(value.value)

    # --- Native scalar types (bool before int: bool is an int subclass) -----
    if isinstance(value, (bool, str, int, float)):
        return value

    # --- numpy / pandas scalars ---------------------------------------------
    if hasattr(value, "item") and not hasattr(value, "__len__"):
        return safe_meta(value.item())

    # --- Collections: flatten to comma-separated string --------------------
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)

    # --- Dicts: JSON-encode -------------------------------------------------
    if isinstance(value, dict):
        return json.dumps(value, default=str)

    return str(value)
