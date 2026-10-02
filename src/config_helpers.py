from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeVar

from src.providers import EmbeddingProvider, LLMProvider


# ---------------------------------------------------------------------------
# Env var coercion helpers
# ---------------------------------------------------------------------------

def as_bool(value: str | None, default: bool = False) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def optional_bool(value: str | None) -> bool | None:
    """Like as_bool, but unset/empty means None ('use the default')."""
    if value is None or value.strip() == "":
        return None
    return as_bool(value)


def as_float(value: str | None, default: float) -> float:
    if value is None or value.strip() == "":
        return default
    return float(value)


def as_int(value: str | None, default: int, min: int | None = None) -> int:
    """Parse an int, clamping to `min` if given (protects against e.g. negative concurrency)."""
    result = default if value is None or value.strip() == "" else int(value)
    if min is not None and result < min:
        return min
    return result


def as_list(
    value: str | None,
    default: list[str] | None = None,
    separator: str = ",",
) -> list[str]:
    if value is None or value.strip() == "":
        return default if default is not None else []
    return [item.strip() for item in value.split(separator) if item.strip()]


def optional_str(value: str | None) -> str | None:
    return value.strip() or None if value else None


def optional_provider(value: str | None) -> LLMProvider | None:
    return parse_enum(value, LLMProvider, "") if optional_str(value) else None


def optional_embedder(
    value: str | None, default: EmbeddingProvider
) -> EmbeddingProvider:
    return parse_enum(value, EmbeddingProvider, default) if optional_str(value) else default


T = TypeVar("T", bound=StrEnum)


def parse_enum(value: str | None, enum_cls: type[T], default: str) -> T:
    """Convert a string to a StrEnum member, case-insensitively.

    Args:
        value:    Raw string (e.g. from os.getenv). Falls back to default if None.
        enum_cls: The StrEnum class to parse into.
        default:  Fallback string value if value is None.

    Returns:
        The matching enum member.

    Raises:
        ValueError: If the string doesn't match any member.
    """
    resolved = value if value is not None else default
    try:
        return enum_cls(resolved.strip().lower())
    except ValueError:
        valid = ", ".join(e.value for e in enum_cls)
        raise ValueError(
            f"Invalid value '{resolved}' for {enum_cls.__name__}. "
            f"Valid options: {valid}"
        ) from None


# ---------------------------------------------------------------------------
# Settings dataclasses — typed slices of AppConfig passed to factories
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LlmSettings:
    """Everything needed to build one LLM. AppConfig holds one per role (primary, router, judge)."""
    provider: LLMProvider
    model: str
    api_key: str | None = None
    base_url: str | None = None          # Ollama only
    request_timeout: float | None = None # Ollama only
    context_window: int | None = None    # Ollama only
    temperature: float | None = None
    max_tokens: int | None = None        # None = provider default; Ollama maps this to num_predict
    rate_limit_rpm: int = 0              # 0 = unlimited
    thinking: bool | None = None         # Ollama only: reasoning-model thinking; None = model default


@dataclass(frozen=True)
class EmbedderSettings:
    provider: EmbeddingProvider
    model: str
    chunk_size: int
    chunk_overlap: int
    embed_batch_size: int
    device: str = "cpu"
    base_url: str | None = None


@dataclass(frozen=True)
class CollectionSettings:
    """A named vector collection. Documents come from raw_dir (if it has files) and/or POST /documents."""
    name: str
    raw_dir: Path
    top_k: int
    description: str       # shown to the agent so it knows when to search this collection
