"""
Turns ingested rows into a DataFrame ready for embedding.

    preprocess(df, schema, steps) = sanitize(step_n(...step_1(df)))

Project-specific steps are plain functions `DataFrame -> DataFrame` that add or change columns —
e.g. tagging rows with a policy version, classifying a speaker, or splitting a field. They run
before sanitisation, so they may return lists, enums, dicts or NaN; sanitize() makes them
ChromaDB-safe. Register them per collection with CollectionOptions(steps=[...]).
"""
from __future__ import annotations

import logging
from typing import Callable, Sequence

import pandas as pd

from src.preprocessing.sanitizer import sanitize
from src.schema import DocumentSchema

log = logging.getLogger(__name__)

PreprocessStep = Callable[[pd.DataFrame], pd.DataFrame]


def preprocess(
    df: pd.DataFrame,
    schema: DocumentSchema,
    steps: Sequence[PreprocessStep] = (),
) -> pd.DataFrame:
    """Run project steps in order, then sanitise for ChromaDB.

    Args:
        df:     Rows from file or table ingestion (or POST /documents).
        schema: The collection's DocumentSchema.
        steps:  Project-specific transforms, applied in order.

    Returns:
        Sanitised DataFrame.
    """
    if df.empty:
        return df
    for step in steps:
        name = getattr(step, "__name__", type(step).__name__)
        df = step(df)
        log.debug("Preprocessing step '%s' → %d rows, %d columns", name, len(df), len(df.columns))
    return sanitize(df, schema)
