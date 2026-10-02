"""Tolerant parsing of JSON from raw LLM output (local models rarely return clean JSON)."""
from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (e.g. Qwen3, DeepSeek-R1).

    These appear before the actual response and confuse JSON extraction — the think block may itself
    contain braces or JSON-like fragments that match first.
    """
    return _THINK_BLOCK.sub("", text).strip()


def parse_llm_json(text: str) -> dict:
    """Extract and parse a JSON object from raw LLM output. Returns {} if nothing usable is found.

    Handles common model quirks:
      - Reasoning model <think>...</think> preambles
      - Markdown code fences and prose around the object
      - Python literals (None/True/False)
      - Trailing commas before } or ]
      - Invalid backslash escapes (e.g. LaTeX like \\rho, \\mathbb)
      - Unquoted string values where the model dropped the opening quote
    """
    match = _JSON_OBJECT.search(strip_think_blocks(text))
    if not match:
        return {}
    raw = match.group(0)

    # Python literals -> JSON literals
    raw = re.sub(r"\bNone\b", "null", raw)
    raw = re.sub(r"\bTrue\b", "true", raw)
    raw = re.sub(r"\bFalse\b", "false", raw)

    # Trailing commas before } or ]
    raw = re.sub(r",\s*([}\]])", r"\1", raw)

    # Escape backslashes not already part of a valid JSON escape sequence (valid after \: " \ / b f n r t u).
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw)

    # Unquoted string values where the model dropped the opening quote but kept the closing one:
    # `: The text..."` -> `: "The text..."`
    raw = re.sub(
        r':\s+(?!true\b|false\b|null\b|[-\d"\[{])(\w[^"]*)"(\s*[,}\]])',
        lambda m: ': "' + m.group(1).strip().replace('"', '\\"') + '"' + m.group(2),
        raw,
    )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_llm_model(text: str, model: type[M]) -> M | None:
    """parse_llm_json, then validate against a Pydantic model. Returns None if either step fails."""
    data = parse_llm_json(text)
    if not data:
        return None
    try:
        return model.model_validate(data)
    except ValidationError:
        return None
