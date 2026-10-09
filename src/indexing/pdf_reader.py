"""PDF reading with PyMuPDF, which keeps word spacing on typeset layouts where pypdf (LlamaIndex's default)
glues words together — e.g. 'Oneofeacharmourtype(coat,plates,helm)canbewornatonce'. Glued text embeds
poorly, so retrieval quietly degrades; looks_glued() lets ingestion warn about it."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

_LONG_WORD = 25          # characters; real words (even German compounds) rarely exceed this
_GLUED_RATIO = 0.05      # share of long "words" that indicates missing spaces
_MIN_WORDS = 50


class PyMuPDFPageReader(BaseReader):
    """One Document per page, with the same `page_label` metadata as LlamaIndex's default PDF reader."""

    def load_data(self, file: Path, extra_info: dict[str, Any] | None = None, **kwargs: Any) -> list[Document]:
        import pymupdf  # only needed when this reader is used

        docs = []
        with pymupdf.open(file) as pdf:
            for page in pdf:
                docs.append(Document(
                    text=page.get_text(),
                    metadata={
                        **(extra_info or {}),
                        # The printed page label if the PDF defines one, else the physical page number.
                        "page_label": page.get_label() or str(page.number + 1),
                    },
                ))
        return docs


def pdf_extractor() -> dict[str, BaseReader]:
    """File-extractor override for SimpleDirectoryReader: PyMuPDF if installed, else LlamaIndex's default."""
    try:
        import pymupdf  # noqa: F401
    except ImportError:
        return {}
    return {".pdf": PyMuPDFPageReader()}


def looks_glued(text: str) -> bool:
    """True if many 'words' are implausibly long — the signature of extraction that dropped spaces."""
    words = text.split()
    if len(words) < _MIN_WORDS:
        return False
    return sum(1 for w in words if len(w) > _LONG_WORD) / len(words) > _GLUED_RATIO
