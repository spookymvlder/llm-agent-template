"""
Streams: documents that arrive in pieces while the app runs — live transcripts (meetings, calls),
chat logs, field notes, event or sensor logs.

Each fragment is appended to an environment-local JSONL file — the durable record — and embedded
immediately so it is searchable at once. Closing a stream (by default) replaces its fragment chunks
with the whole stream embedded as one normally-chunked document, which retrieves better than many
tiny fragments. Reindexing replays every stream the way it was last stored (open or not re-chunked:
fragments; closed and re-chunked: one document), so a rebuilt index matches the live one.

    data/<env>/streams/<collection>/<stream_id>.jsonl      one fragment per line
    data/<env>/manifests/<collection>.streams.json         status of each stream
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum, auto
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_STREAM_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MANIFEST_VERSION = 1
_LIVE = "live"   # embedded_sha256 marker: an open stream whose fragments were embedded as they arrived


class Granularity(StrEnum):
    """How a stream is stored in the vector store."""
    FRAGMENTS = auto()   # one document per fragment (while open, or closed with rechunk=False)
    DOCUMENT  = auto()   # the whole stream as one document, chunked normally


class StreamError(ValueError):
    """Invalid stream id, or an operation that doesn't fit the stream's state."""


class StreamNotFound(StreamError):
    pass


class StreamClosed(StreamError):
    pass


@dataclass
class Fragment:
    seq: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    received_at: str = ""


@dataclass
class StreamInfo:
    stream_id: str
    open: bool = True
    granularity: Granularity = Granularity.FRAGMENTS
    fragments: int = 0
    close_metadata: dict[str, Any] = field(default_factory=dict)
    embedded_sha256: str | None = None   # file hash when the vector store last matched this stream

    def to_manifest(self) -> dict:
        return {
            "open": self.open, "granularity": str(self.granularity), "fragments": self.fragments,
            "close_metadata": self.close_metadata, "embedded_sha256": self.embedded_sha256,
        }

    @classmethod
    def from_manifest(cls, stream_id: str, data: dict) -> StreamInfo:
        return cls(
            stream_id=stream_id,
            open=data.get("open", False),
            granularity=Granularity(data.get("granularity", Granularity.DOCUMENT)),
            fragments=data.get("fragments", 0),
            close_metadata=data.get("close_metadata", {}),
            embedded_sha256=data.get("embedded_sha256"),
        )


class StreamManager:
    """Owns one collection's stream files and their manifest. Thread-safe (appends may come from worker threads)."""

    def __init__(self, *, stream_dir: Path, manifest_path: Path) -> None:
        self.stream_dir = stream_dir
        self.manifest_path = manifest_path
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, stream_id: str, text: str, metadata: dict[str, Any] | None = None) -> Fragment:
        """Append a fragment to the stream's file, creating the stream if needed. Does not embed.

        Raises:
            StreamError:  Invalid stream id.
            StreamClosed: The stream has been closed.
        """
        path = self.path(stream_id)
        with self._lock:
            streams = self._load()
            info = streams.get(stream_id)
            if info is None:
                # A new stream's fragments are embedded as they arrive, so it never needs replaying — until a
                # reindex clears this marker (clear_embedded), after which sync() replays it from the file.
                info = StreamInfo(stream_id, embedded_sha256=_LIVE)
            if not info.open:
                raise StreamClosed(f"Stream '{stream_id}' is closed.")
            fragment = Fragment(
                seq=info.fragments,
                text=text,
                metadata=dict(metadata or {}),
                received_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(fragment.__dict__, ensure_ascii=False, default=str) + "\n")
            info.fragments += 1
            streams[stream_id] = info
            self._save(streams)
        return fragment

    def close(self, stream_id: str, granularity: Granularity, metadata: dict[str, Any] | None = None) -> StreamInfo:
        """Mark a stream closed. The caller embeds it per `granularity`, then calls mark_embedded().

        Raises:
            StreamNotFound: No such stream.
        """
        with self._lock:
            streams = self._load()
            info = self._get(streams, stream_id)
            info.open = False
            info.granularity = granularity
            info.close_metadata = dict(metadata or {})
            info.embedded_sha256 = None
            self._save(streams)
            return info

    def mark_embedded(self, stream_id: str) -> None:
        """Record that the vector store now matches the stream file as it is on disk."""
        with self._lock:
            streams = self._load()
            info = self._get(streams, stream_id)
            info.embedded_sha256 = _sha256_file(self.path(stream_id))
            self._save(streams)

    def needing_embedding(self) -> list[StreamInfo]:
        """Streams whose vectors don't match their file: after a reindex (or an interrupted close),
        or a closed stream file edited by hand. Open streams only need it after a reindex."""
        with self._lock:
            streams = self._load()
            # Stream files without a manifest entry (e.g. manifest deleted): treat as closed documents.
            for path in sorted(self.stream_dir.glob("*.jsonl")) if self.stream_dir.is_dir() else []:
                if path.stem not in streams and _STREAM_ID_RE.match(path.stem):
                    streams[path.stem] = StreamInfo(path.stem, open=False, granularity=Granularity.DOCUMENT,
                                                    fragments=len(self.read(path.stem)))
            result = []
            for info in streams.values():
                if not self.path(info.stream_id).exists():
                    continue
                if info.open:
                    if info.embedded_sha256 is None:
                        result.append(info)
                elif info.embedded_sha256 != _sha256_file(self.path(info.stream_id)):
                    result.append(info)
            return result

    def clear_embedded(self) -> None:
        """Forget what is embedded (after the collection is deleted), so every stream is replayed."""
        with self._lock:
            streams = self._load()
            for info in streams.values():
                info.embedded_sha256 = None
            self._save(streams)

    def read(self, stream_id: str) -> list[Fragment]:
        path = self.path(stream_id)
        if not path.exists():
            raise StreamNotFound(f"Stream '{stream_id}' not found.")
        with path.open(encoding="utf-8") as f:
            return [Fragment(**json.loads(line)) for line in f if line.strip()]

    def info(self, stream_id: str) -> StreamInfo:
        with self._lock:
            return self._get(self._load(), stream_id)

    def list(self) -> list[StreamInfo]:
        with self._lock:
            return list(self._load().values())

    def path(self, stream_id: str) -> Path:
        """The stream's JSONL file. Stream ids become file names, so they are restricted to [A-Za-z0-9._-].

        Raises:
            StreamError: Invalid stream id.
        """
        if not _STREAM_ID_RE.match(stream_id or ""):
            raise StreamError(
                f"Invalid stream id '{stream_id}': use up to 128 letters, digits, '.', '_' or '-', "
                "starting with a letter or digit."
            )
        return self.stream_dir / f"{stream_id}.jsonl"

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _get(self, streams: dict[str, StreamInfo], stream_id: str) -> StreamInfo:
        self.path(stream_id)  # validate
        if stream_id not in streams:
            raise StreamNotFound(f"Stream '{stream_id}' not found.")
        return streams[stream_id]

    def _load(self) -> dict[str, StreamInfo]:
        if not self.manifest_path.exists():
            return {}
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {sid: StreamInfo.from_manifest(sid, d) for sid, d in data.get("streams", {}).items()}

    def _save(self, streams: dict[str, StreamInfo]) -> None:
        # Written on every append, so write-then-rename to never leave a half-written manifest.
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"version": _MANIFEST_VERSION, "streams": {sid: i.to_manifest() for sid, i in sorted(streams.items())}},
            indent=2, default=str,
        ), encoding="utf-8")
        os.replace(tmp, self.manifest_path)


def common_metadata(fragments: list[Fragment]) -> dict[str, Any]:
    """Metadata keys whose value is the same on every fragment (e.g. a session number)."""
    if not fragments:
        return {}
    common = dict(fragments[0].metadata)
    for fragment in fragments[1:]:
        common = {k: v for k, v in common.items() if fragment.metadata.get(k) == v}
    return common


def _sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
