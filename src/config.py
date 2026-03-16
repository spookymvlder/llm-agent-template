from __future__ import annotations


import os
from dataclasses import dataclass
from pathlib import Path

from enum import Enum, auto

from dotenv import load_dotenv

import numpy as np

from providers import LLMProvider, EmbeddingProvider


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _as_float(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _as_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)

def _as_list(value: str | None, default: list[str] | None = None, separator: str = ",") -> list[str]:
    if value is None or value == "":
        return default if default is not None else []
    
    # Split, strip whitespace, filter empty strings
    return [item.strip() for item in value.split(separator) if item.strip()]

def _optional_provider(value: str | None) -> LLMProvider | None:
    return LLMProvider(value.lower()) if value else None    

def _optional_embedder(value: str | None) -> EmbeddingProvider | None:
    return EmbeddingProvider(value.lower()) if value else None

def _optional_str(value: str | None) -> str | None:
    return value.strip() or None if value else None

@dataclass
class LlmSettings:
    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    request_timeout: float | None = None
    context_window: int | None = None
    temperature: float | None = None

@dataclass
class EmbedderSettings:
    provider: str
    model: str
    chunk_size: int

@dataclass(frozen=True)
class AppConfig:
    """AppConfig controls shared configuration settings that may affect the team. Typically deals with content that is not a part of the 
    version control repo due to .gitignore, such as source files, saved data, or local directories. Also handles loading of model API keys.
    Most content for AppConfig is based on the user's .env file. See .env.example for a .env template for personal use.

    """

    class Environment(Enum):
        DEV = auto()
        TEST = auto()
        PROD = auto()


    # Generic Settings
    seed: int
    rng: np.random
    environment: str

    # Paths
    project_root: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    logs_dir: Path
    chroma_dir: Path


    # Logging / observability
    enable_logging: bool
    log_level: str
    log_to_file: bool

    # LLM provider selection
    llm_provider: LLMProvider  # "openai" | "gemini" | "ollama"
    llm_model: str

    judge_provider: LLMProvider | None = None
    judge_model: str | None = None

    # Keys (optional depending on provider)
    openai_api_key: str | None
    google_api_key: str | None
    anthropic_api_key: str | None

    # Ollama settings
    ollama_base_url: str
    request_timeout_s: float
    context_window: int
    temperature: float
    memory_token_limit: int

    # Embeddings
    embedding_provider: str = EmbeddingProvider.HUGGINGFACE  # "huggingface" | "ollama"
    embedding_model: str
    chunk_size: int

    # Retrieval
    top_k: int

    # Chroma
    collection_name: str
    distance_metric: str


    @property
    def llm_settings(self) -> LlmSettings:
        return LlmSettings(
            provider=self.llm_provider,
            model=self.llm_model,
            api_key=getattr(self, f"{self.llm_provider}_api_key", None),
            base_url=self.ollama_base_url,
            request_timeout=self.request_timeout_s,
            context_window=self.context_window,
            temperature=self.temperature,
        )

    @property
    def judge_llm_settings(self) -> LlmSettings | None:
        if not self.judge_provider or not self.judge_model:
            return None
        return LlmSettings(
            provider=self.judge_provider,
            model=self.judge_model,
            api_key=getattr(self, f"{self.llm_provider}_api_key", None),
            base_url=self.ollama_base_url,
            request_timeout=self.request_timeout_s,
            context_window=self.context_window,
            temperature=self.temperature,
        )

    @property
    def embedding_settings(self) -> EmbedderSettings:
        return EmbedderSettings(
            provider=self.embedding_provider,
            model=self.embedding_model,
            chunk_size=self.chunk_size,
            base_url=self.ollama_base_url,
        )


        


    @staticmethod
    def load(project_root: Path | None = None, dotenv_path: Path | None = None) -> "AppConfig":
        """Loads the .env file contents into the AppConfig class. Intended to be called once at program start.
        Subsequent calls may lead to unexpected outcomes.

        Args:
            project_root (Path | None, optional): The root folder for this project. Defaults to None.
            dotenv_path (Path | None, optional): The directory where the .env file is stored. Defaults to None.

        Returns:
            AppConfig: An AppConfig class that contains all configurable run information.
        """
        if project_root is None:
            # src/config.py -> project root is two parents up (…/rag_starter)
            project_root = Path(__file__).resolve().parents[1]

        if dotenv_path is None:
            dotenv_path = project_root / ".env"

        load_dotenv(dotenv_path=dotenv_path, override=False)

        # Get environment mode (test, dev, prod)
        env_mode = os.getenv("ENVIRONMENT", "dev").lower()
        
        # Base data directory
        data_dir = project_root / os.getenv("DATA_DIR", "data")
        raw_dir = (data_dir / "raw").resolve()

        # Environment-specific subdirectory
        if env_mode.upper() == AppConfig.Environment.TEST:
            env_dir = data_dir / AppConfig.Environment.TEST
        elif env_mode.upper() == AppConfig.Environment.PROD:
            env_dir = data_dir / AppConfig.Environment.PROD
        else:  # dev (default)
            env_dir = data_dir / AppConfig.Environment.DEV
        
        # All subdirectories are relative to data_dir / env, except for raw.
        processed_dir = (env_dir / "processed").resolve()
        logs_dir = (env_dir / "logs").resolve()
        chroma_dir = (env_dir /  "chroma").resolve()

        load_seed=_as_int(os.getenv("SEED"), default=42)

        
        cfg =  AppConfig(

            # General Settings
            seed=load_seed,
            rng=np.random.default_rng(load_seed),
            environment=env_mode,

            # Directories
            project_root=project_root,
            data_dir=data_dir,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            logs_dir=logs_dir,
            chroma_dir=chroma_dir,
            
            # Logging
            enable_logging=_as_bool(os.getenv("ENABLE_LOGGING"), default=True),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            log_to_file=_as_bool(os.getenv("LOG_TO_FILE"), default=True),
            
            # LLM Model
            llm_provider=os.getenv("LLM_PROVIDER", LLMProvider.OLLAMA).lower(),
            llm_model=os.getenv("LLM_MODEL", "llama3.1").strip(),
            judge_provider = _optional_provider(os.getenv("JUDGE_PROVIDER")),
            judge_model = _optional_str(os.getenv("JUDGE_MODEL")),

            # LLM Keys
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),

            # RAG Configuration
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            request_timeout_s=_as_float(os.getenv("REQUEST_TIMEOUT_S"), default=300.0),
            context_window=_as_int(os.getenv("CONTEXT_WINDOW"), default=8000),
            temperature=_as_float(os.getenv("TEMPERATURE"), default=0.2),
            memory_token_limit=_as_int(os.getenv("MEMORY_TOKEN_LIMIT"), default=4096),

            # Embedder Settings
            distance_metric=os.getenv("DISTANCE_METRIC", "cosine"),
            embedding_provider=_optional_embedder(os.getenv("EMBEDDING_PROVIDER")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip(),
            chunk_size=_as_int(os.getenv("CHUNK_SIZE"), default=2048),   

            # Retrieval Settings
            top_k=_as_int(os.getenv("RETRIEVAL_TOP_K"), default=5),
            

            # Filenames
            collection_name=os.getenv("COLLECTION_NAME", "documents"),

        )
        
        cfg.raw_dir.mkdir(parents=True, exist_ok=True)
        cfg.processed_dir.mkdir(parents=True, exist_ok=True)
        cfg.logs_dir.mkdir(parents=True, exist_ok=True)
        cfg.chroma_dir.mkdir(parents=True, exist_ok=True)
        
        return cfg

CONFIG = AppConfig.load()