from __future__ import annotations

import hashlib
import json
import logging
import pandas as pd

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llama_index.core import SimpleDirectoryReader

from src.schema import DOC_ID, FILE_HASH, SOURCE_PATH, DocumentSchema

log = logging.getLogger(__name__)


SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".html", ".htm", ".json", ".csv"}
# Read as one document per row when the collection's schema is tabular.
TABLE_EXTS = {".csv", ".jsonl", ".parquet"}
# Folder-level metadata: applies to every file in its folder and below; never ingested itself.
METADATA_FILENAME = "_metadata.json"
_MANIFEST_VERSION = 3


@dataclass(frozen=True)
class FileRecord:
    """A file on disk and the fingerprint the manifest stores for it."""
    path: Path
    rel_path: str           # relative to raw_dir, '/'-separated; stored on chunks as `source_path`
    sha256: str
    size: int
    mtime_ns: int
    meta_sha256: str        # hash of the merged _metadata.json values that apply to this file
    folder_metadata: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def to_manifest(self) -> dict:
        return {"sha256": self.sha256, "size": self.size, "mtime_ns": self.mtime_ns, "meta_sha256": self.meta_sha256}


@dataclass
class FileChanges:
    """Differences between raw_dir and this environment's manifest."""
    new: list[FileRecord] = field(default_factory=list)
    changed: list[FileRecord] = field(default_factory=list)    # content or folder metadata changed
    touched: list[FileRecord] = field(default_factory=list)    # size/mtime differ but nothing that matters
    missing: list[str] = field(default_factory=list)           # in the manifest, no longer on disk

    @property
    def has_work(self) -> bool:
        return bool(self.new or self.changed)


class IndexManager:
    """Discovers files in one collection's raw_dir and tracks which versions have been ingested.

    Files are never moved — raw_dir is a shared, stable source of truth.
    Each environment keeps its own manifest per collection, so dev/test/prod
    each have an independent view of what has been ingested. The manifest maps
    each file's relative path to its SHA-256, size, mtime and folder-metadata hash;
    files whose size and mtime are unchanged are not re-hashed.

    Folder metadata: a `_metadata.json` object in any folder applies to every file in that folder and
    below, with nearer folders overriding. Editing one marks the files it covers as changed.

        data/raw/rules/
          2014/_metadata.json   -> {"edition": "2014"}
          2014/PHB.pdf
          2024/_metadata.json   -> {"edition": "2024"}
          2024/PHB.pdf

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
        ... preprocess and embed df (deleting old chunks of changed files first) ...
        manager.commit(changes.new + changes.changed + changes.touched)

    Supported formats: .txt, .md, .pdf, .html, .htm, .json, .csv
    (tabular schemas: .csv, .jsonl, .parquet as one document per row; .parquet needs pyarrow).
    """

    def __init__(self, *, raw_dir: Path, manifest_path: Path, schema: DocumentSchema | None = None) -> None:
        """
        Args:
            raw_dir:       Shared directory containing source documents. Never modified.
            manifest_path: Environment-specific JSON file recording which file versions have been ingested.
            schema:        The collection's DocumentSchema (decides whether tables are read per row).
        """
        self.raw_dir = raw_dir
        self.manifest_path = manifest_path
        self.schema = schema or DocumentSchema()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_changes(self) -> FileChanges:
        """Compare raw_dir against the manifest.

        Raises:
            ValueError: A _metadata.json file is not a valid JSON object.
        """
        self._ensure_dirs()
        manifest = self._load_manifest()
        changes = FileChanges()
        on_disk: set[str] = set()
        folder_cache: dict[Path, dict[str, Any]] = {}

        for path in self._discover_files():
            rel = self._relative_path(path)
            on_disk.add(rel)
            folder_metadata = self._folder_metadata(path.parent, folder_cache)
            meta_sha = _sha256_text(json.dumps(folder_metadata, sort_keys=True, default=str))
            stat = path.stat()
            known = manifest.get(rel)

            same_file = bool(known) and known.get("size") == stat.st_size and known.get("mtime_ns") == stat.st_mtime_ns
            if same_file and known.get("meta_sha256") == meta_sha:
                continue  # unchanged: skip hashing
            sha = known["sha256"] if same_file else _sha256_file(path)
            record = FileRecord(path, rel, sha, stat.st_size, stat.st_mtime_ns, meta_sha, folder_metadata)

            if known is None:
                changes.new.append(record)
            elif known.get("sha256") != sha or known.get("meta_sha256") != meta_sha:
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
        """Read files into a DataFrame, one row per document. Columns:
            doc_id      — stable id: source_path + '#' + (position in the file, or the row's id_column)
            text / schema.text_columns — the text to embed
            source_path — path relative to raw_dir (used to replace a file's chunks when it changes)
            file_hash   — SHA-256 of the file version that was embedded
            file_name, page_label, ... — metadata from LlamaIndex's readers or the table's other columns
            _metadata.json values — for keys the reader/table didn't already set
        """
        if not records:
            return pd.DataFrame()
        tables = [r for r in records if self.schema.tabular and r.path.suffix.lower() in TABLE_EXTS]
        files = [r for r in records if r not in tables]

        frames = [self._read_documents(files)] if files else []
        frames += [self._read_table(r) for r in tables]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
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
    # Reading
    # ------------------------------------------------------------------

    def _read_documents(self, records: list[FileRecord]) -> pd.DataFrame:
        """One row per LlamaIndex Document (e.g. one page of a PDF, or a whole text file)."""
        by_path = {str(r.path.resolve()): r for r in records}
        docs = SimpleDirectoryReader(input_files=list(by_path)).load_data()

        text_col = self.schema.text_columns[0]
        rows = []
        position: dict[str, int] = {}
        for doc in docs:
            record = by_path[str(Path(doc.metadata["file_path"]).resolve())]
            n = position.get(record.rel_path, 0)
            position[record.rel_path] = n + 1
            rows.append({
                **record.folder_metadata,
                **doc.metadata,
                text_col: doc.text,
                DOC_ID: f"{record.rel_path}#{n}",
                SOURCE_PATH: record.rel_path,
                FILE_HASH: record.sha256,
            })
        return pd.DataFrame(rows)

    def _read_table(self, record: FileRecord) -> pd.DataFrame:
        """One row per table row. Row values win over _metadata.json values for the same key."""
        suffix = record.path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(record.path)
        elif suffix == ".jsonl":
            df = pd.read_json(record.path, lines=True)
        else:
            try:
                df = pd.read_parquet(record.path)
            except ImportError as e:
                raise ImportError(f"Reading {record.rel_path} needs pyarrow: `pip install pyarrow`.") from e

        id_col = self.schema.id_column
        if id_col and id_col not in df.columns:
            raise ValueError(f"{record.rel_path} has no '{id_col}' column (DocumentSchema.id_column).")
        row_ids = df[id_col].astype(str) if id_col else pd.Series(range(len(df)), index=df.index).astype(str)

        for key, value in record.folder_metadata.items():
            if key not in df.columns:
                df[key] = value
        df[DOC_ID] = record.rel_path + "#" + row_ids
        df[SOURCE_PATH] = record.rel_path
        df[FILE_HASH] = record.sha256
        df["file_name"] = record.path.name
        return df

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
        exts = SUPPORTED_EXTS | (TABLE_EXTS if self.schema.tabular else set())
        return sorted(
            p for p in self.raw_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in exts and p.name != METADATA_FILENAME
        )

    def _folder_metadata(self, folder: Path, cache: dict[Path, dict[str, Any]]) -> dict[str, Any]:
        """Merge _metadata.json from raw_dir down to `folder`; nearer folders override.

        Raises:
            ValueError: A _metadata.json file is not a valid JSON object.
        """
        if folder in cache:
            return cache[folder]
        root = self.raw_dir.resolve()
        resolved = folder.resolve()
        inherited = {} if resolved == root or root not in resolved.parents else self._folder_metadata(folder.parent, cache)

        merged = dict(inherited)
        meta_file = folder / METADATA_FILENAME
        if meta_file.is_file():
            try:
                own = json.loads(meta_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise ValueError(f"{meta_file} is not valid JSON: {e}") from e
            if not isinstance(own, dict):
                raise ValueError(f"{meta_file} must contain a JSON object, e.g. {{\"edition\": \"2024\"}}.")
            merged.update(own)
        cache[folder] = merged
        return merged

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


def _sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
