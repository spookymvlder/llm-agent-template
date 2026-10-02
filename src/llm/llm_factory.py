from __future__ import annotations

import importlib
import os
import logging
from dataclasses import dataclass
from typing import Any

from llama_index.core.rate_limiter import TokenBucketRateLimiter

from src.config_helpers import LlmSettings
from src.providers import API_KEY_ENV_VARS, LLMProvider

log = logging.getLogger(__name__)


@dataclass
class _ProviderSpec:
    module: str            # imported on first use, so uninstalled providers don't break the others
    class_name: str
    package: str           # pip package that provides `module`
    env_var: str | None    # env var name to check when api_key is not passed (None = no key needed)
    api_key_param: str = "api_key"   # kwarg name the LLM class expects for the key

    def load_class(self) -> type:
        try:
            return getattr(importlib.import_module(self.module), self.class_name)
        except ImportError as e:
            raise ValueError(f"The '{self.package}' package is required for this provider: pip install {self.package}") from e


# Registry — add new providers here, nothing else needs to change.
_REGISTRY: dict[str, _ProviderSpec] = {
    LLMProvider.OLLAMA:    _ProviderSpec("llama_index.llms.ollama",       "Ollama",      "llama-index-llms-ollama",       None),
    LLMProvider.OPENAI:    _ProviderSpec("llama_index.llms.openai",       "OpenAI",      "llama-index-llms-openai",       API_KEY_ENV_VARS[LLMProvider.OPENAI]),
    LLMProvider.GEMINI:    _ProviderSpec("llama_index.llms.google_genai", "GoogleGenAI", "llama-index-llms-google-genai", API_KEY_ENV_VARS[LLMProvider.GEMINI]),
    LLMProvider.ANTHROPIC: _ProviderSpec("llama_index.llms.anthropic",    "Anthropic",   "llama-index-llms-anthropic",    API_KEY_ENV_VARS[LLMProvider.ANTHROPIC]),
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
        ValueError: Unknown provider, provider package not installed, missing API key, or failed initialization.
    """
    provider = provider.lower()
    spec = _REGISTRY.get(provider)
    if spec is None:
        raise ValueError(f"Unknown LLM provider '{provider}'. Supported: {', '.join(sorted(_REGISTRY))}")
    cls = spec.load_class()

    if spec.env_var:
        resolved_key = api_key or os.getenv(spec.env_var)
        if not resolved_key:
            raise ValueError(
                f"{spec.env_var} is required for the '{provider}' provider. "
                f"Set it in your .env file or pass it as api_key."
            )
        kwargs[spec.api_key_param] = resolved_key

    try:
        return cls(model=model, **kwargs)
    except Exception as e:
        raise ValueError(f"Failed to initialize {provider} LLM: {e}") from e


def build_llm_from_settings(settings: LlmSettings) -> Any:
    """Build an LLM from an LlmSettings slice of AppConfig.

    Only settings a provider understands are passed to it:
      - Ollama: base_url, request_timeout, context_window, thinking, and max_tokens as num_predict.
      - Hosted providers: max_tokens.
      - All: temperature, and a rate limiter when rate_limit_rpm > 0.

    Raises:
        ValueError: Unknown provider, provider package not installed, missing API key, or failed initialization.
    """
    kwargs: dict[str, Any] = {}
    if settings.temperature is not None:
        kwargs["temperature"] = settings.temperature
    if settings.rate_limit_rpm > 0:
        kwargs["rate_limiter"] = TokenBucketRateLimiter(requests_per_minute=settings.rate_limit_rpm)

    if settings.provider == LLMProvider.OLLAMA:
        ollama_kwargs = {
            "base_url": settings.base_url,
            "request_timeout": settings.request_timeout,
            "context_window": settings.context_window,
            "thinking": settings.thinking,
        }
        kwargs.update({k: v for k, v in ollama_kwargs.items() if v is not None})
        if settings.max_tokens is not None:
            kwargs["additional_kwargs"] = {"num_predict": settings.max_tokens}
    else:
        if settings.max_tokens is not None:
            kwargs["max_tokens"] = settings.max_tokens
        if settings.thinking is not None:
            log.warning("*_THINKING only applies to Ollama models; ignored for %s/%s.", settings.provider, settings.model)

    return build_llm(
        provider=settings.provider,
        model=settings.model,
        api_key=settings.api_key,
        **kwargs,
    )
