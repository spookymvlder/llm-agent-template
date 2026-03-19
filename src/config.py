from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
import logging

import numpy as np
from dotenv import load_dotenv

from config_helpers import (
    EmbedderSettings,
    LlmSettings,
    RetrievalSettings,
    as_bool,
    as_float,
    as_int,
    optional_embedder,
    optional_provider,
    optional_str,
    parse_enum,
)
from providers import EmbeddingProvider, LLMProvider

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class AppConfig:
    """Application configuration loaded from the .env file.

    All runtime settings are read once at startup via AppConfig.load().
    Subsequent calls to load() are safe but redundant — use the module-level
    CONFIG singleton instead.
    """

    class Environment(StrEnum):
        DEV  = auto()
        TEST = auto()
        PROD = auto()

    # TODO figure out what parameters can safely be set to None and add as an expected type.
    # General
    seed: int
    rng: np.random.Generator
    environment: AppConfig.Environment
    auto_ingest: bool

    # Paths
    project_root: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    logs_dir: Path
    chroma_dir: Path
    env_dir: Path

    # Logging
    enable_logging: bool
    log_level: str
    log_to_file: bool

    # LLM
    llm_provider: LLMProvider
    llm_model: str

    ollama_base_url: str
    request_timeout: float
    context_window: int
    temperature: float
    max_iterations: int

    openai_api_key: str | None
    google_api_key: str | None
    anthropic_api_key: str | None

    # Judge LLM (optional)
    judge_provider: LLMProvider | None
    judge_model: str | None

    # Embeddings
    embedding_provider: EmbeddingProvider
    embedding_model: str
    chunk_size: int

    # Retrieval
    top_k: int
    collection_name: str
    distance_metric: str
    memory_token_limit: int

    # ---------------------------------------------------------------------------
    # Settings properties — typed slices passed to factories
    # ---------------------------------------------------------------------------

    @property
    def llm_settings(self) -> LlmSettings:
        """
        Returns the config values necessary for the application's primary LLM.

        Allows projects that use this as a template to not rely on the config file, but to instead easily 
        drop in settings based on the needs of their project if these are necessary after initial launch.

        Returns:
            LlmSettings: A subset of config options necessary to run the primary LLM.
        """
        return LlmSettings(
            provider=self.llm_provider,
            model=self.llm_model,
            api_key=getattr(self, f"{self.llm_provider}_api_key", None),
            base_url=self.ollama_base_url,
            request_timeout=self.request_timeout,
            context_window=self.context_window,
            temperature=self.temperature
        )

    @property
    def judge_llm_settings(self) -> LlmSettings | None:
        """
        Returns the config values necessary for the application's evaluator LLM.

        Allows projects that use this as a template to not rely on the config file, but to instead easily 
        drop in settings based on the needs of their project if these are necessary after initial launch.

        Returns:
            LlmSettings: A subset of config options necessary to run the evaluator LLM.
        """
        if not self.judge_provider or not self.judge_model:
            return None
        return LlmSettings(
            provider=self.judge_provider,
            model=self.judge_model,
            api_key=getattr(self, f"{self.judge_provider}_api_key", None),
            base_url=self.ollama_base_url,
            request_timeout=self.request_timeout,
            context_window=self.context_window,
            temperature=self.temperature
        )

    @property
    def embedder_settings(self) -> EmbedderSettings:
        """
        Returns the config values necessary for the application's embedder.

        Allows projects that use this as a template to not rely on the config file, but to instead easily 
        drop in settings based on the needs of their project if these are necessary after initial launch.

        Returns:
            EmbedderSettings: A subset of config options necessary to run embedding for the application.
        """
        return EmbedderSettings(
            provider=self.embedding_provider,
            model=self.embedding_model,
            chunk_size=self.chunk_size,
            base_url=self.ollama_base_url,
        )

    @property
    def retrieval_settings(self) -> RetrievalSettings:
        """
        Bundle of settings for retrieving data from ChromaDB.

        Returns:
            RetrievalSettings: A subset of config options necessary for retrieving data from ChromaDB.
        """
        return RetrievalSettings(
            top_k=self.top_k,
            collection_name=self.collection_name,
            distance_metric=self.distance_metric,
            memory_token_limit=self.memory_token_limit,
            max_iterations=self.max_iterations,
        )

    @property
    def debug(self) -> bool:
        """
        Indicates whether FastAPI should reload or not when code changes.

        Returns:
            bool: True if dev environment.
        """
        return self.environment == AppConfig.Environment.DEV

    # ---------------------------------------------------------------------------
    # Factory
    # ---------------------------------------------------------------------------

    @staticmethod
    def load(
        project_root: Path | None = None,
        dotenv_path: Path | None = None,
    ) -> AppConfig:
        """Load .env and construct AppConfig. Call once at startup.

        Args:
            project_root: Project root directory. Defaults to two levels above config.py.
            dotenv_path:  Path to .env file. Defaults to project_root / '.env'.

        Returns:
            Fully populated AppConfig instance.
        """
        if project_root is None:
            project_root = Path(__file__).resolve().parents[1]
        if dotenv_path is None:
            dotenv_path = project_root / ".env"

        load_dotenv(dotenv_path=dotenv_path, override=False)

        env_mode = parse_enum(
            os.getenv("ENVIRONMENT"),
            AppConfig.Environment,
            AppConfig.Environment.DEV.value,
        )

        data_dir  = project_root / os.getenv("DATA_DIR", "data")
        raw_dir   = (data_dir / "raw").resolve()
        env_dir   = (data_dir / str(env_mode)).resolve()

        seed = as_int(os.getenv("SEED"), default=42)

        cfg = AppConfig(
            # General
            seed=seed,
            rng=np.random.default_rng(seed),
            environment=env_mode,
            auto_ingest=as_bool(os.getenv("AUTO_INGEST"), default=True),

            # Paths
            project_root=project_root,
            data_dir=data_dir,
            raw_dir=raw_dir,
            env_dir=env_dir,
            processed_dir=(env_dir / "processed").resolve(),
            logs_dir=(env_dir / "logs").resolve(),
            chroma_dir=(env_dir / "chroma").resolve(),

            # Logging
            enable_logging=as_bool(os.getenv("ENABLE_LOGGING"), default=True),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            log_to_file=as_bool(os.getenv("LOG_TO_FILE"), default=True),

            # LLM
            llm_provider=parse_enum(os.getenv("LLM_PROVIDER"), LLMProvider, LLMProvider.OLLAMA),
            llm_model=os.getenv("LLM_MODEL", "qwen3").strip(),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            request_timeout=as_float(os.getenv("REQUEST_TIMEOUT"), default=300.0),
            context_window=as_int(os.getenv("CONTEXT_WINDOW"), default=32768),
            temperature=as_float(os.getenv("TEMPERATURE"), default=0.2),
            memory_token_limit=as_int(os.getenv("MEMORY_TOKEN_LIMI"), default=4096),

            # Judge LLM
            judge_provider=optional_provider(os.getenv("JUDGE_PROVIDER")),
            judge_model=optional_str(os.getenv("JUDGE_MODEL")),

            # Embeddings
            embedding_provider=optional_embedder(
                os.getenv("EMBEDDING_PROVIDER"), default=EmbeddingProvider.HUGGINGFACE
            ),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip(),
            chunk_size=as_int(os.getenv("CHUNK_SIZE"), default=2048),

            # Retrieval
            top_k=as_int(os.getenv("RETRIEVAL_TOP_K"), default=5),
            collection_name=os.getenv("COLLECTION_NAME", "documents"),
            distance_metric=os.getenv("DISTANCE_METRIC", "cosine"),
            max_iterations=as_int(os.getenv("MAX_ITERATIONS"), default=3),
            
        )

        # Ensure all runtime directories exist
        for d in (cfg.raw_dir, cfg.env_dir, cfg.processed_dir, cfg.logs_dir, cfg.chroma_dir):
            d.mkdir(parents=True, exist_ok=True)

        log.debug(cfg.to_log_str())
        return cfg


    def to_log_str(self) -> str:
        """Format config fields for debug logging. Redacts API keys."""
        def _redact(value: str | None) -> str:
            if not value:
                return "not set"
            return f"{value[:4]}{'*' * (len(value) - 4)}" if len(value) > 4 else "****"

        lines = [
            "AppConfig:",
            f"  environment     : {self.environment}",
            f"  debug           : {self.debug}",
            f"  auto_ingest     : {self.auto_ingest}",
            "",
            f"  project_root    : {self.project_root}",
            f"  data_dir        : {self.data_dir}",
            f"  raw_dir         : {self.raw_dir}",
            f"  env_dir         : {self.env_dir}",
            f"  chroma_dir      : {self.chroma_dir}",
            f"  logs_dir        : {self.logs_dir}",
            "",
            f"  llm_provider    : {self.llm_provider}",
            f"  llm_model       : {self.llm_model}",
            f"  ollama_base_url : {self.ollama_base_url}",
            f"  context_window  : {self.context_window}",
            f"  temperature     : {self.temperature}",
            f"  request_timeout : {self.request_timeout}",
            "",
            f"  openai_api_key  : {_redact(self.openai_api_key)}",
            f"  google_api_key  : {_redact(self.google_api_key)}",
            f"  anthropic_api_key: {_redact(self.anthropic_api_key)}",
            "",
            f"  judge_provider  : {self.judge_provider or 'not set'}",
            f"  judge_model     : {self.judge_model or 'not set'}",
            "",
            f"  embedding_provider: {self.embedding_provider}",
            f"  embedding_model : {self.embedding_model}",
            f"  chunk_size      : {self.chunk_size}",
            "",
            f"  top_k           : {self.top_k}",
            f"  collection_name : {self.collection_name}",
            f"  distance_metric : {self.distance_metric}",
            f"  memory_token_limit: {self.memory_token_limit}",
            f"  max_iterations: : {self.max_iterations}",
            "",
            f"  enable_logging  : {self.enable_logging}",
            f"  log_level       : {self.log_level}",
            f"  log_to_file     : {self.log_to_file}",
        ]
        return "\n".join(lines)

# TODO - store state of config after load to log
CONFIG = AppConfig.load()
