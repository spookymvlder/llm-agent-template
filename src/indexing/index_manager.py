from __future__ import annotations

import json
import logging
import pandas as pd

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from llama_index.core import SimpleDirectoryReader

log = logging.getLogger(__name__)


SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".html", ".htm", ".json", ".csv"}


@dataclass
class NewDocumentsResult:
    df: pd.DataFrame
    file_paths: list[Path] = field(default_factory=list)


class IndexManager:
    """Discovers and reads new documents from raw_dir into a DataFrame.

    Files are never moved — raw_dir is a shared, stable source of truth.
    Each environment maintains its own manifest and indices directory,
    so dev/test/prod each have an independent view of what has been ingested.

    Directory layout:
        data/
          raw/                   # shared, files stay here permanently
          dev/
            indices/
            manifest.json
          test/
            indices/
            manifest.json
          prod/
            indices/
            manifest.json

    Typical usage:
        manager = IndexManager(raw_dir=..., env_dir=...)
        result = manager.load_new_as_dataframe()

        if not result.df.empty:
            chroma_result = chroma_manager.load_or_build(result.df)
            manager.commit_processed(result.file_paths)

    Supported formats: .txt, .md, .pdf, .html, .htm, .json, .csv
    """

    def __init__(self, *, raw_dir: Path, env_dir: Path) -> None:
        """
        Args:
            raw_dir: Shared directory containing source documents. Never modified.
            env_dir: Environment-specific directory (e.g. data/dev, data/prod).
                     Holds the manifest and indices for that environment.
        """
        self.raw_dir = raw_dir
        self.env_dir = env_dir
        self.indices_dir = env_dir / "indices"
        self.manifest_path = env_dir / "manifest.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_new_as_dataframe(self) -> NewDocumentsResult:
        """Return a DataFrame of text chunks from files not yet in this environment's manifest.

        Each row represents one LlamaIndex Document (i.e. one chunk from a
        file). Columns:
            text        — chunk text, ready for embedding
            file_name   — source filename
            file_path   — absolute path of the source file
            <extras>    — any additional metadata LlamaIndex extracted

        If there are no new files, returns an empty DataFrame and an empty
        file list.
        """
        self._ensure_dirs()
        manifest = self._load_manifest()
        new_files = self._discover_new_files(manifest)

        if not new_files:
            log.info("[%s] No new files found in raw directory.", self._env_name)
            return NewDocumentsResult(df=pd.DataFrame(), file_paths=[])

        log.info("[%s] Found %d new file(s) to process.", self._env_name, len(new_files))
        docs = self._read_files(new_files)

        rows = [{"text": doc.text, **doc.metadata} for doc in docs]
        df = pd.DataFrame(rows)
        log.info("[%s] Loaded %d chunk(s) from %d file(s).", self._env_name, len(df), len(new_files))
        return NewDocumentsResult(df=df, file_paths=new_files)

    def commit_processed(self, file_paths: list[Path]) -> None:
        """Record files as processed in this environment's manifest.

        Call only after downstream steps (embedding, etc.) have succeeded.
        Files are NOT moved or deleted — they remain in raw_dir and can be
        re-ingested by other environments.
        """
        if not file_paths:
            return
        manifest = self._load_manifest()
        self._mark_processed(manifest, file_paths)
        log.info("[%s] Committed %d file(s) to manifest.", self._env_name, len(file_paths))

    def reset_manifest(self) -> None:
        """Clear this environment's manifest, forcing a full re-ingest on next run.

        Useful in test environments or when rebuilding an index from scratch.
        Does not affect raw_dir or other environments.
        """
        if self.manifest_path.exists():
            self.manifest_path.unlink()
            log.info("[%s] Manifest cleared.", self._env_name)

    def ingested_count(self) -> int:
        """Return the number of files recorded in this environment's manifest."""
        return len(self._load_manifest())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _env_name(self) -> str:
        return self.env_dir.name

    def _ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.env_dir, self.indices_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _discover_new_files(self, manifest: set[str]) -> list[Path]:
        files = sorted(
            p for p in self.raw_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        )
        return [p for p in files if str(p.resolve()) not in manifest]

    def _read_files(self, files: list[Path]):
        if not files:
            return []
        reader = SimpleDirectoryReader(input_files=[str(p) for p in files])
        return reader.load_data()

    def _load_manifest(self) -> set[str]:
        if not self.manifest_path.exists():
            return set()
        try:
            return set(json.loads(self.manifest_path.read_text(encoding="utf-8")))
        except Exception:
            return set()

    def _save_manifest(self, processed: set[str]) -> None:
        self.manifest_path.write_text(
            json.dumps(sorted(processed), indent=2), encoding="utf-8"
        )

    def _mark_processed(self, manifest: set[str], files: Iterable[Path]) -> None:
        for p in files:
            manifest.add(str(p.resolve()))
        self._save_manifest(manifest)
