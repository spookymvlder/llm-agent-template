"""Shared test setup.

src.config loads its CONFIG singleton at import time, so the environment is pinned here — before any
test module imports src — to a throwaway data dir and settings that need no network or API keys.
Tests use LlamaIndex's mock embedder and scripted mock LLMs; nothing calls Ollama or a hosted model.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

_DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-agent-tests-")) / "data"
os.environ.update({
    "ENVIRONMENT": "test",
    "DATA_DIR": str(_DATA_DIR),
    "LOG_TO_FILE": "false",
    "LOG_LEVEL": "WARNING",
    "LLM_PROVIDER": "ollama",
    "LLM_MODEL": "test-model",
    "EMBEDDING_PROVIDER": "huggingface",
    "COLLECTIONS": "handbook,meetings",
    "ENABLE_ROUTER": "false",
    "AUTO_INGEST": "true",
})
for key in ("JUDGE_PROVIDER", "JUDGE_MODEL", "ROUTER_PROVIDER", "ROUTER_MODEL", "DEFAULT_CHANGE_MODE"):
    os.environ.pop(key, None)

from typing import Callable

import chromadb
import pytest
from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM

from src.config_helpers import CollectionSettings
from src.indexing import CollectionHandle, CollectionOptions, open_collection
from src.schema import FileMode


@pytest.fixture(autouse=True)
def mock_models():
    """Every test gets a deterministic mock embedder and a mock LLM in LlamaIndex's global Settings."""
    Settings.embed_model = MockEmbedding(embed_dim=8)
    Settings.llm = MockLLM()
    yield


@pytest.fixture
def make_collection(tmp_path: Path) -> Callable[..., CollectionHandle]:
    """Open a collection in its own temp folders: make_collection("handbook", options=..., default_mode=...).
    description=None behaves like an unset COLLECTION_<NAME>_DESCRIPTION (resolved from _metadata.json)."""
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))

    def _make(name: str = "handbook", options: CollectionOptions | None = None,
              default_mode: FileMode = FileMode.STATIC, top_k: int = 5,
              description: str | None = "default") -> CollectionHandle:
        raw = tmp_path / "raw" / name
        raw.mkdir(parents=True, exist_ok=True)
        if description == "default":
            description = f"The {name} collection."
        return open_collection(
            CollectionSettings(name=name, raw_dir=raw, top_k=top_k, description=description),
            client=client,
            chroma_dir=tmp_path / "chroma",
            manifest_dir=tmp_path / "manifests",
            stream_dir=tmp_path / "streams" / name,
            distance_metric="cosine",
            default_mode=default_mode,
            options=options,
        )

    return _make
