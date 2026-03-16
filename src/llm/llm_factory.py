from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any

from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.ollama import Ollama

from src.providers import LLMProvider

log = logging.getLogger(__name__)


@dataclass
class _ProviderSpec:
    cls: type
    env_var: str           # env var name to check when api_key is not passed
    api_key_param: str     # kwarg name the LLM class expects for the key


# Registry — add new providers here, nothing else needs to change.
_REGISTRY: dict[str, _ProviderSpec] = {
    LLMProvider.OPENAI:  _ProviderSpec(cls=OpenAI,     env_var="OPENAI_API_KEY",     api_key_param="api_key"),
    LLMProvider.GEMINI:  _ProviderSpec(cls=Gemini,     env_var="GOOGLE_API_KEY",     api_key_param="api_key"),
    LLMProvider.ANTHROPIC: _ProviderSpec(cls=Anthropic, env_var="ANTHROPIC_API_KEY", api_key_param="api_key"),
}


def build_llm(
    provider: str,
    model: str,
    api_key: str | None = None,
    **kwargs: Any,
) -> Any:
    """Build a LlamaIndex-compatible LLM for the given provider.

    Args:
        provider:  One of 'openai', 'gemini', 'anthropic', or 'ollama'.
        model:     The model/version string (e.g. 'gpt-4o', 'claude-sonnet-4-5').
        api_key:   Optional API key. Falls back to the provider's env var.
        **kwargs:  Any extra kwargs forwarded directly to the LLM constructor
                   (e.g. temperature, max_tokens, base_url for Ollama).

    Returns:
        A LlamaIndex LLM instance.

    Raises:
        ValueError: Unknown provider, missing API key, or failed initialization.
    """
    provider = provider.lower()

    # Ollama is local — no API key needed, just pass kwargs through.
    if provider == LLMProvider.OLLAMA:
        try:
            return Ollama(model=model, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to initialize Ollama LLM: {e}") from e

    spec = _REGISTRY.get(provider)
    if spec is None:
        supported = ", ".join(sorted(_REGISTRY) + [LLMProvider.OLLAMA])
        raise ValueError(
            f"Unknown LLM provider '{provider}'. Supported: {supported}"
        )

    resolved_key = api_key or os.getenv(spec.env_var)
    if not resolved_key:
        raise ValueError(
            f"{spec.env_var} is required for the '{provider}' provider. "
            f"Set it in your .env file or pass it as api_key."
        )

    try:
        return spec.cls(model=model, **{spec.api_key_param: resolved_key}, **kwargs)
    except Exception as e:
        raise ValueError(f"Failed to initialize {provider} LLM: {e}") from e
