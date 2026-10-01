from __future__ import annotations

import hashlib
import json
import logging
import pandas as pd

from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core import SimpleDirectoryReader

log = logging.getLogger(__name__)


SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".html", ".htm", ".json", ".csv"}
# Reserved for folder-level metadata (Phase 3); never ingested as a document.
METADATA_FILENAME = "_metadata.json"
_MANIFEST_VERSION = 2


@dataclass(frozen=True)
class FileRecord:
    """A file on disk and the fingerprint the manifest stores for it."""
    path: Path
    rel_path: str      # relative to raw_dir, '/'-separated; also stored on chunks as `source_path`
    sha256: str
    size: int
    mtime_ns: int

    def to_manifest(self) -> dict:
        return {"sha256": self.sha256, "size": self.size, "mtime_ns": self.mtime_ns}


@dataclass
class FileChanges:
    """Differences between raw_dir and this environment's manifest."""
    new: list[FileRecord] = field(default_factory=list)
    changed: list[FileRecord] = field(default_factory=list)    # content hash differs from the manifest
    touched: list[FileRecord] = field(default_factory=list)    # size/mtime differ but content is identical
    missing: list[str] = field(default_factory=list)           # in the manifest, no longer on disk

    @property
    def has_work(self) -> bool:
        return bool(self.new or self.changed)


class IndexManager:
    """Discovers files in one collection's raw_dir and tracks which versions have been ingested.

    Files are never moved — raw_dir is a shared, stable source of truth.
    Each environment keeps its own manifest per collection, so dev/test/prod
    each have an independent view of what has been ingested. The manifest maps
    each file's relative path to its SHA-256, size and mtime; files whose size and
    mtime are unchanged are not re-hashed.

    Directory layout:
        data/
          raw/
            <collection>/        # shared, files stay here permanently
          dev/
            chroma/
            manifests/
              <collection>.json
          test/ ...
          prod/ ...

    Typical usage:
        manager = IndexManager(raw_dir=..., manifest_path=...)
        changes = manager.find_changes()
        df = manager.load_as_dataframe(changes.new + changes.changed)
        ... embed df (deleting old chunks of changed files first) ...
        manager.commit(changes.new + changes.changed + changes.touched)

    Supported formats: .txt, .md, .pdf, .html, .htm, .json, .csv
    """

    def __init__(self, *, raw_dir: Path, manifest_path: Path) -> None:
        """
        Args:
            raw_dir:       Shared directory containing source documents. Never modified.
            manifest_path: Environment-specific JSON file recording which file versions have been ingested.
        """
        self.raw_dir = raw_dir
        self.manifest_path = manifest_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_changes(self) -> FileChanges:
        """Compare raw_dir against the manifest."""
        self._ensure_dirs()
        manifest = self._load_manifest()
        changes = FileChanges()
        on_disk: set[str] = set()

        for path in self._discover_files():
            rel = self._relative_path(path)
            on_disk.add(rel)
            stat = path.stat()
            known = manifest.get(rel)
            if known and known.get("size") == stat.st_size and known.get("mtime_ns") == stat.st_mtime_ns:
                continue  # unchanged: skip hashing
            record = FileRecord(path, rel, _sha256(path), stat.st_size, stat.st_mtime_ns)
            if known is None:
                changes.new.append(record)
            elif known.get("sha256") != record.sha256:
                changes.changed.append(record)
            else:
                changes.touched.append(record)

        changes.missing = sorted(set(manifest) - on_disk)
        log.info(
            "[%s] %d new, %d changed, %d missing file(s) in %s.",
            self._label, len(changes.new), len(changes.changed), len(changes.missing), self.raw_dir,
        )
        return changes

    def load_as_dataframe(self, records: list[FileRecord]) -> pd.DataFrame:
        """Read files into a DataFrame, one row per LlamaIndex Document (e.g. one page of a PDF,
        or a whole text file). Columns:
            doc_id      — stable id: source_path + '#' + position in the file
            text        — document text, split into chunks at embedding time
            source_path — path relative to raw_dir (used to replace a file's chunks when it changes)
            file_hash   — SHA-256 of the file version that was embedded
            file_name, file_path, ... — metadata LlamaIndex extracted
        """
        if not records:
            return pd.DataFrame()
        by_path = {str(r.path.resolve()): r for r in records}
        docs = SimpleDirectoryReader(input_files=list(by_path)).load_data()

        rows = []
        position: dict[str, int] = {}
        for doc in docs:
            record = by_path.get(str(Path(doc.metadata.get("file_path", "")).resolve()))
            rel = record.rel_path if record else doc.metadata.get("file_name", "unknown")
            n = position.get(rel, 0)
            position[rel] = n + 1
            rows.append({
                "doc_id": f"{rel}#{n}",
                "text": doc.text,
                "source_path": rel,
                "file_hash": record.sha256 if record else "",
                **doc.metadata,
            })
        df = pd.DataFrame(rows)
        log.info("[%s] Loaded %d document(s) from %d file(s).", self._label, len(df), len(records))
        return df

    def commit(self, records: list[FileRecord]) -> None:
        """Record these file versions in the manifest.

        Call only after downstream steps (embedding, etc.) have succeeded, so a failure is retried next run.
        Files are NOT moved or deleted — they remain in raw_dir and can be
        re-ingested by other environments.
        """
        if not records:
            return
        manifest = self._load_manifest()
        manifest.update({r.rel_path: r.to_manifest() for r in records})
        self._save_manifest(manifest)
        log.info("[%s] Committed %d file(s) to manifest.", self._label, len(records))

    def reset_manifest(self) -> None:
        """Clear this collection's manifest, forcing a full re-ingest on next run.

        Useful in test environments or when rebuilding an index from scratch.
        Does not affect raw_dir or other environments.
        """
        if self.manifest_path.exists():
            self.manifest_path.unlink()
            log.info("[%s] Manifest cleared.", self._label)

    def ingested_count(self) -> int:
        """Return the number of files recorded in this environment's manifest."""
        return len(self._load_manifest())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _label(self) -> str:
        """Log prefix: '<env>/<collection>', e.g. 'dev/rules'."""
        return f"{self.manifest_path.parent.parent.name}/{self.manifest_path.stem}"

    def _relative_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.raw_dir.resolve()).as_posix()
        except ValueError:
            return path.name

    def _ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.manifest_path.parent):
            d.mkdir(parents=True, exist_ok=True)

    def _discover_files(self) -> list[Path]:
        return sorted(
            p for p in self.raw_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS and p.name != METADATA_FILENAME
        )

    def _load_manifest(self) -> dict[str, dict]:
        if not self.manifest_path.exists():
            return {}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("[%s] Manifest unreadable (%s); treating every file as new.", self._label, e)
            return {}
        if not isinstance(data, dict) or data.get("version") != _MANIFEST_VERSION:
            log.warning("[%s] Manifest is an older format; treating every file as new. "
                        "Run `ingest --reindex` to avoid duplicate chunks.", self._label)
            return {}
        return data.get("files", {})

    def _save_manifest(self, files: dict[str, dict]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps({"version": _MANIFEST_VERSION, "files": dict(sorted(files.items()))}, indent=2),
            encoding="utf-8",
        )


def _sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
