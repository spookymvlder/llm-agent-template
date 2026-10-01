from __future__ import annotations

import logging
from typing import Sequence

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

from src.config import LlmSettings, EmbedderSettings
from src.llm.llm_factory import build_llm_from_settings
from src.providers import EmbeddingProvider

log = logging.getLogger(__name__)


class LLMConfigurationError(Exception):
    """Raised when LLM configuration fails."""


class EmbeddingConfigurationError(Exception):
    """Raised when embedding model configuration fails."""


# ---------------------------------------------------------------------------
# LlamaIndex global configuration
# ---------------------------------------------------------------------------

def configure_llamaindex(llm_settings: LlmSettings | None, embedder_settings: EmbedderSettings) -> None:
    """Configure LlamaIndex Settings with LLM and embedding models from AppConfig.

    Args:
        llm_settings:      Primary LLM. None skips the LLM (ingest-only runs), leaving Settings.llm unset.
        embedder_settings: Embedding model and text splitter.

    Raises:
        LLMConfigurationError: Invalid provider, missing API key, or connection error.
        EmbeddingConfigurationError: Embedding model setup failed.
    """
    try:
        if llm_settings is not None:
            Settings.llm = _build_llm(llm_settings)
            log.info("LLM configured: provider=%s model=%s", llm_settings.provider, llm_settings.model)
        Settings.embed_model = _build_embedding_model(embedder_settings)
        Settings.text_splitter = SentenceSplitter(
            chunk_size=embedder_settings.chunk_size,
            chunk_overlap=embedder_settings.chunk_overlap,
        )
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
        return build_llm_from_settings(settings)
    except ValueError as e:
        raise LLMConfigurationError(str(e)) from e


def _build_embedding_model(embedder_settings: EmbedderSettings):
    """Build the embedding model based on configured provider."""
    try:
        if embedder_settings.provider == EmbeddingProvider.OLLAMA:
            log.info("Building Ollama embeddings at %s...", embedder_settings.base_url)
            return OllamaEmbedding(
                model_name=embedder_settings.model,
                base_url=embedder_settings.base_url,
                embed_batch_size=embedder_settings.embed_batch_size,
            )
        elif embedder_settings.provider == EmbeddingProvider.HUGGINGFACE:
            log.info("Building HuggingFace embeddings: %s on %s...", embedder_settings.model, embedder_settings.device)
            return HuggingFaceEmbedding(
                model_name=embedder_settings.model,
                device=embedder_settings.device,
                embed_batch_size=embedder_settings.embed_batch_size,
            )
        else:
            raise EmbeddingConfigurationError(
                f"Unknown EMBEDDING_PROVIDER '{embedder_settings.provider}'. "
                f"Valid options: {', '.join(e.value for e in EmbeddingProvider)}"
            )
    except EmbeddingConfigurationError:
        raise
    except Exception as e:
        raise EmbeddingConfigurationError(f"Failed to build embedding model: {e}") from e
