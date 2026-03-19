from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

from providers import EmbeddingProvider, LLMProvider


# ---------------------------------------------------------------------------
# Env var coercion helpers
# ---------------------------------------------------------------------------

def as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def as_float(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def as_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def as_list(
    value: str | None,
    default: list[str] | None = None,
    separator: str = ",",
) -> list[str]:
    if value is None or value == "":
        return default if default is not None else []
    return [item.strip() for item in value.split(separator) if item.strip()]


def optional_str(value: str | None) -> str | None:
    return value.strip() or None if value else None


def optional_provider(value: str | None) -> LLMProvider | None:
    return LLMProvider(value.lower()) if value else None


def optional_embedder(
    value: str | None, default: EmbeddingProvider
) -> EmbeddingProvider:
    return EmbeddingProvider(value.lower()) if value else default


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

@dataclass
class LlmSettings:
    provider: LLMProvider
    model: str
    api_key: str | None = None
    base_url: str | None = None
    request_timeout: float | None = None
    context_window: int | None = None
    temperature: float | None = None


@dataclass
class EmbedderSettings:
    provider: EmbeddingProvider
    model: str
    chunk_size: int
    base_url: str | None = None


@dataclass
class RetrievalSettings:
    top_k: int
    collection_name: str
    distance_metric: str
    memory_token_limit: int
    max_iterations: int
    enable_fact_extraction: bool
    max_facts: int
