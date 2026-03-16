from __future__ import annotations

import logging
from typing import Sequence

from llama_index.core import Settings
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

from src.config import LlmSettings, EmbedderSettings
from src.llm import build_llm
from src.providers import EmbeddingProvider

log = logging.getLogger(__name__)


class LLMConfigurationError(Exception):
    """Raised when LLM configuration fails."""


class EmbeddingConfigurationError(Exception):
    """Raised when embedding model configuration fails."""


# ---------------------------------------------------------------------------
# LlamaIndex global configuration
# ---------------------------------------------------------------------------

def configure_llamaindex(llm_settings: LlmSettings, embedder_settings: EmbedderSettings) -> None:
    """Configure LlamaIndex Settings with LLM and embedding models from AppConfig.

    Raises:
        LLMConfigurationError: Invalid provider, missing API key, or connection error.
        EmbeddingConfigurationError: Embedding model setup failed.
    """
    try:
        Settings.llm = _build_llm(llm_settings)
        Settings.embed_model = _build_embedding_model(embedder_settings)
        Settings.text_splitter = SentenceSplitter(chunk_size=2048)

        log.info("LLM configured: provider=%s model=%s", llm_settings.provider, llm_settings.model)
        log.info("Embeddings configured: provider=%s model=%s", embedder_settings.provider, embedder_settings.model)

    except (LLMConfigurationError, EmbeddingConfigurationError):
        log.error("Failed to configure LlamaIndex.")
        raise
    except Exception as e:
        log.error("Unexpected error during LlamaIndex configuration: %s", e)
        raise LLMConfigurationError(f"Unexpected configuration failure: {e}") from e


def _build_llm(settings: LlmSettings):
    """Delegate to llm_factory — single source of truth for provider wiring."""
    try:
        return build_llm(
            provider=settings.provider,
            model=settings.model,
            # API key falls back to env var inside build_llm if not set on cfg.
            api_key=getattr(settings, f"{settings.provider}_api_key", None),
            # Ollama-specific kwargs — ignored by other providers via **kwargs.
            base_url=getattr(settings, "ollama_base_url", None),
            request_timeout=getattr(settings, "request_timeout_s", None),
            context_window=getattr(settings, "context_window", None),
            temperature=getattr(settings, "temperature", None),
        )
    except ValueError as e:
        raise LLMConfigurationError(str(e)) from e


def _build_embedding_model(embedder_settings: EmbedderSettings):
    """Build the embedding model based on configured provider."""
    try:
        if embedder_settings.provider == EmbeddingProvider.OLLAMA:
            log.info("Building Ollama embeddings at %s...", embedder_settings.ollama_base_url)
            return OllamaEmbedding(model=embedder_settings.model, base_url=embedder_settings.ollama_base_url)
        elif embedder_settings.provider == EmbeddingProvider.HUGGINGFACE:
            log.info("Building HuggingFace embeddings: %s...", embedder_settings.model)
            return HuggingFaceEmbedding(model_name=embedder_settings.model)
        else:
            raise EmbeddingConfigurationError(
                f"Unknown EMBEDDING_PROVIDER '{embedder_settings.provider}'. "
                f"Valid options: {', '.join(e.value for e in EmbeddingProvider)}"
            )
    except EmbeddingConfigurationError:
        raise
    except Exception as e:
        raise EmbeddingConfigurationError(f"Failed to build embedding model: {e}") from e
