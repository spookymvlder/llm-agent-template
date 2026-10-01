from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import StrEnum, auto
from pathlib import Path

from dotenv import load_dotenv

from src.config_helpers import (
    CollectionSettings,
    EmbedderSettings,
    LlmSettings,
    as_bool,
    as_float,
    as_int,
    as_list,
    optional_embedder,
    optional_provider,
    optional_str,
    parse_enum,
)
from src.providers import API_KEY_ENV_VARS, EmbeddingProvider, LLMProvider

log = logging.getLogger(__name__)


class ConfigError(ValueError):
    """Raised when the .env configuration is invalid or incomplete."""


# Env vars renamed since earlier versions of the template: old name -> new name.
_RENAMED_ENV_VARS = {
    "TEMPERATURE": "LLM_TEMPERATURE",
    "CONTEXT_WINDOW": "LLM_CONTEXT_WINDOW",
    "COLLECTION_NAME": "COLLECTIONS",
}

_DISTANCE_METRICS = {"cosine", "l2", "ip"}
# ChromaDB collection name rules: 3-512 chars of [a-zA-Z0-9._-], starting and ending alphanumeric.
_COLLECTION_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$")


@dataclass(frozen=True)
class AppConfig:
    """Application configuration loaded from the .env file.

    All runtime settings are read once at startup via AppConfig.load().
    Subsequent calls to load() are safe but redundant — use the module-level
    CONFIG singleton instead. Loading only parses; call validate() (bootstrap does)
    to check that the configuration is usable.
    """

    class Environment(StrEnum):
        DEV  = auto()
        TEST = auto()
        PROD = auto()

    # General
    seed: int
    environment: AppConfig.Environment
    auto_ingest: bool
    reindex_changed_files: bool   # re-embed files whose content hash changed since they were ingested

    # Paths. Directories are created by the components that use them, not at import.
    project_root: Path
    data_dir: Path
    raw_dir: Path          # shared across environments; one subdirectory per collection
    env_dir: Path          # data/{environment}
    processed_dir: Path
    logs_dir: Path
    chroma_dir: Path
    eval_dir: Path

    # Logging
    enable_logging: bool
    log_level: str
    log_to_file: bool

    # LLM roles
    llm_settings: LlmSettings                  # primary agent LLM
    router_llm_settings: LlmSettings           # falls back to primary when ROUTER_* unset
    judge_llm_settings: LlmSettings | None     # None = evaluation disabled
    enable_router: bool
    router_min_confidence: float

    # Embeddings
    embedder_settings: EmbedderSettings
    hf_token: str | None

    # Collections & retrieval
    collections: tuple[CollectionSettings, ...]
    top_k: int
    distance_metric: str

    # Agent
    max_iterations: int
    memory_token_limit: int
    enable_fact_extraction: bool
    max_facts: int

    # Evaluation
    eval_concurrency: int

    # ---------------------------------------------------------------------------
    # Accessors
    # ---------------------------------------------------------------------------

    @property
    def debug(self) -> bool:
        """
        Indicates whether FastAPI should reload or not when code changes.

        Returns:
            bool: True if dev environment.
        """
        return self.environment == AppConfig.Environment.DEV

    @property
    def default_collection(self) -> CollectionSettings:
        """The first collection listed in COLLECTIONS."""
        return self.collections[0]

    def collection(self, name: str) -> CollectionSettings:
        """Look up a configured collection by name.

        Raises:
            KeyError: No collection with that name is configured.
        """
        for c in self.collections:
            if c.name == name:
                return c
        raise KeyError(f"Unknown collection '{name}'. Configured: {', '.join(c.name for c in self.collections)}")

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

        Raises:
            ConfigError / ValueError: A value could not be parsed.
        """
        if project_root is None:
            project_root = Path(__file__).resolve().parents[1]
        if dotenv_path is None:
            dotenv_path = project_root / ".env"

        load_dotenv(dotenv_path=dotenv_path, override=False)

        for old, new in _RENAMED_ENV_VARS.items():
            if os.getenv(old) is not None and os.getenv(new) is None:
                log.warning("%s is no longer read; rename it to %s in your .env.", old, new)

        env_mode = parse_enum(
            os.getenv("ENVIRONMENT"),
            AppConfig.Environment,
            AppConfig.Environment.DEV.value,
        )

        data_dir = (project_root / os.getenv("DATA_DIR", "data")).resolve()
        raw_dir  = data_dir / "raw"
        env_dir  = data_dir / str(env_mode)

        # Should be sufficient to simply set in environ, isn't needed by AppConfig.
        hf_token = optional_str(os.getenv("HF_TOKEN"))
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        llm_settings = _primary_llm()
        top_k = as_int(os.getenv("RETRIEVAL_TOP_K"), default=5, min=1)

        cfg = AppConfig(
            # General
            seed=as_int(os.getenv("SEED"), default=42),
            environment=env_mode,
            auto_ingest=as_bool(os.getenv("AUTO_INGEST"), default=True),
            reindex_changed_files=as_bool(os.getenv("REINDEX_CHANGED_FILES"), default=True),

            # Paths
            project_root=project_root,
            data_dir=data_dir,
            raw_dir=raw_dir,
            env_dir=env_dir,
            processed_dir=env_dir / "processed",
            logs_dir=env_dir / "logs",
            chroma_dir=env_dir / "chroma",
            eval_dir=env_dir / "evaluation",

            # Logging
            enable_logging=as_bool(os.getenv("ENABLE_LOGGING"), default=True),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
            log_to_file=as_bool(os.getenv("LOG_TO_FILE"), default=True),

            # LLM roles
            llm_settings=llm_settings,
            router_llm_settings=_llm_role("ROUTER", llm_settings),
            judge_llm_settings=_llm_role("JUDGE", llm_settings, optional=True),
            enable_router=as_bool(os.getenv("ENABLE_ROUTER"), default=False),
            router_min_confidence=as_float(os.getenv("ROUTER_MIN_CONFIDENCE"), default=0.5),

            # Embeddings
            embedder_settings=EmbedderSettings(
                provider=optional_embedder(os.getenv("EMBEDDING_PROVIDER"), default=EmbeddingProvider.HUGGINGFACE),
                model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip(),
                chunk_size=as_int(os.getenv("CHUNK_SIZE"), default=512, min=32),
                chunk_overlap=as_int(os.getenv("CHUNK_OVERLAP"), default=50, min=0),
                embed_batch_size=as_int(os.getenv("EMBED_BATCH_SIZE"), default=32, min=1),
                device=_resolve_device(as_bool(os.getenv("USE_CUDA"), default=False)),
                base_url=llm_settings.base_url,
            ),
            hf_token=hf_token,

            # Collections & retrieval
            collections=_collections(project_root, raw_dir, top_k),
            top_k=top_k,
            distance_metric=os.getenv("DISTANCE_METRIC", "cosine").strip().lower(),

            # Agent
            max_iterations=as_int(os.getenv("MAX_ITERATIONS"), default=3, min=1),
            memory_token_limit=as_int(os.getenv("MEMORY_TOKEN_LIMIT"), default=4096, min=256),
            enable_fact_extraction=as_bool(os.getenv("ENABLE_FACT_EXTRACTION"), default=True),
            max_facts=as_int(os.getenv("MAX_FACTS"), default=50, min=1),

            # Evaluation
            eval_concurrency=as_int(os.getenv("EVAL_CONCURRENCY"), default=1, min=1),
        )

        log.debug(cfg.to_log_str())
        return cfg

    # ---------------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------------

    def validate(self, check_llm_keys: bool = True) -> None:
        """Check that the configuration is usable. Called by bootstrap before anything is built.

        Args:
            check_llm_keys: Require API keys for the LLM roles in use. Ingest-only runs don't need an LLM.

        Raises:
            ConfigError: Listing every problem found, not just the first.
        """
        problems: list[str] = []

        roles = {"LLM": self.llm_settings, "JUDGE": self.judge_llm_settings} if check_llm_keys else {}
        if self.enable_router and check_llm_keys:
            roles["ROUTER"] = self.router_llm_settings
        for prefix, settings in roles.items():
            if settings and settings.provider in API_KEY_ENV_VARS and not settings.api_key:
                problems.append(
                    f"{prefix}_PROVIDER={settings.provider} requires {API_KEY_ENV_VARS[settings.provider]}."
                )

        if self.distance_metric not in _DISTANCE_METRICS:
            problems.append(f"DISTANCE_METRIC must be one of {sorted(_DISTANCE_METRICS)}, got '{self.distance_metric}'.")
        if not 0.0 <= self.router_min_confidence <= 1.0:
            problems.append(f"ROUTER_MIN_CONFIDENCE must be between 0 and 1, got {self.router_min_confidence}.")
        if self.embedder_settings.chunk_overlap >= self.embedder_settings.chunk_size:
            problems.append("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

        names = [c.name for c in self.collections]
        if len(set(names)) != len(names):
            problems.append(f"COLLECTIONS contains duplicates: {names}.")
        for name in names:
            if not _COLLECTION_NAME_RE.match(name):
                problems.append(
                    f"Collection name '{name}' is invalid: use 3-512 letters, digits, '.', '_' or '-', "
                    "starting and ending with a letter or digit."
                )

        if problems:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(problems))

        judge = self.judge_llm_settings
        if judge and (judge.provider, judge.model) == (self.llm_settings.provider, self.llm_settings.model):
            log.warning("JUDGE model is the same as the primary LLM; it will be grading its own answers.")

        if self.raw_dir.is_dir() and any(p.is_file() for p in self.raw_dir.iterdir()):
            log.warning(
                "Files directly in %s are not ingested. Move them into a collection folder, e.g. %s.",
                self.raw_dir, self.default_collection.raw_dir,
            )

    # ---------------------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------------------

    def to_log_str(self) -> str:
        """Format every config field for debug logging. Redacts API keys and tokens."""
        lines = ["AppConfig:"]
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, tuple) and value and is_dataclass(value[0]):
                lines.append(f"  {f.name}:")
                lines.extend(f"    - {_fmt_settings(v)}" for v in value)
            elif is_dataclass(value):
                lines.append(f"  {f.name:<22}: {_fmt_settings(value)}")
            else:
                shown = _redact(value) if f.name.endswith(("_key", "_token")) else value
                lines.append(f"  {f.name:<22}: {shown}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _api_key(provider: LLMProvider) -> str | None:
    env_var = API_KEY_ENV_VARS.get(provider)
    return optional_str(os.getenv(env_var)) if env_var else None


def _max_tokens(value: str | None, default: int | None) -> int | None:
    """0 or negative means 'use the provider default'."""
    parsed = as_int(value, default=default or 0)
    return parsed if parsed > 0 else None


def _primary_llm() -> LlmSettings:
    provider = parse_enum(os.getenv("LLM_PROVIDER"), LLMProvider, LLMProvider.OLLAMA)
    return LlmSettings(
        provider=provider,
        model=os.getenv("LLM_MODEL", "qwen3").strip(),
        api_key=_api_key(provider),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
        request_timeout=as_float(os.getenv("REQUEST_TIMEOUT_S"), default=300.0),
        context_window=as_int(os.getenv("LLM_CONTEXT_WINDOW"), default=8192, min=512),
        temperature=as_float(os.getenv("LLM_TEMPERATURE"), default=0.2),
        max_tokens=_max_tokens(os.getenv("LLM_MAX_TOKENS"), default=None),
        rate_limit_rpm=as_int(os.getenv("LLM_RATE_LIMIT_RPM"), default=0, min=0),
    )


def _llm_role(prefix: str, fallback: LlmSettings, optional: bool = False) -> LlmSettings | None:
    """Read {PREFIX}_PROVIDER / _MODEL / _TEMPERATURE / _CONTEXT_WINDOW / _MAX_TOKENS / _RATE_LIMIT_RPM.

    Anything unset falls back to the primary LLM's value. If {PREFIX}_MODEL is unset, an optional role
    is disabled (None) and a required role reuses the primary model.

    Raises:
        ConfigError: {PREFIX}_PROVIDER is set without {PREFIX}_MODEL.
    """
    provider = optional_provider(os.getenv(f"{prefix}_PROVIDER"))
    model = optional_str(os.getenv(f"{prefix}_MODEL"))
    if provider and not model:
        raise ConfigError(f"{prefix}_PROVIDER is set but {prefix}_MODEL is not.")
    if model is None and optional:
        return None

    provider = provider or fallback.provider
    return replace(
        fallback,
        provider=provider,
        model=model or fallback.model,
        api_key=_api_key(provider),
        temperature=as_float(os.getenv(f"{prefix}_TEMPERATURE"), default=fallback.temperature),
        context_window=as_int(os.getenv(f"{prefix}_CONTEXT_WINDOW"), default=fallback.context_window, min=512),
        max_tokens=_max_tokens(os.getenv(f"{prefix}_MAX_TOKENS"), default=fallback.max_tokens),
        rate_limit_rpm=as_int(os.getenv(f"{prefix}_RATE_LIMIT_RPM"), default=fallback.rate_limit_rpm, min=0),
    )


def _collections(project_root: Path, raw_dir: Path, default_top_k: int) -> tuple[CollectionSettings, ...]:
    """Parse COLLECTIONS (comma list, first is the default) plus optional per-collection overrides:
    COLLECTION_<NAME>_RAW_DIR (relative to the project root, default data/raw/<name>), COLLECTION_<NAME>_TOP_K and
    COLLECTION_<NAME>_DESCRIPTION.
    <NAME> is the collection name upper-cased with non-alphanumerics replaced by '_'.
    """
    names = as_list(os.getenv("COLLECTIONS"), default=["documents"])
    result = []
    for name in names:
        key = re.sub(r"[^A-Z0-9]", "_", name.upper())
        raw_override = optional_str(os.getenv(f"COLLECTION_{key}_RAW_DIR"))
        result.append(CollectionSettings(
            name=name,
            raw_dir=(project_root / raw_override).resolve() if raw_override else raw_dir / name,
            top_k=as_int(os.getenv(f"COLLECTION_{key}_TOP_K"), default=default_top_k, min=1),
            description=optional_str(os.getenv(f"COLLECTION_{key}_DESCRIPTION"))
                        or f"Documents in the '{name}' collection.",
        ))
    return tuple(result)


def _resolve_device(use_cuda: bool) -> str:
    """
    CUDA lets an NVIDIA GPU run embeddings much faster. To use it, install the CUDA toolkit and a
    CUDA-enabled PyTorch build (https://pytorch.org/get-started/locally/ — the default pip install is CPU-only).
    Falls back to CPU with a warning if CUDA was requested but isn't available.
    """
    if not use_cuda:
        return "cpu"
    import torch
    if torch.cuda.is_available():
        return "cuda"
    log.warning("USE_CUDA=true but CUDA is unavailable on this machine; using CPU.")
    return "cpu"


def _redact(value: str | None) -> str:
    if not value:
        return "not set"
    return f"{value[:4]}{'*' * (len(value) - 4)}" if len(value) > 4 else "****"


def _fmt_settings(settings) -> str:
    parts = []
    for f in fields(settings):
        value = getattr(settings, f.name)
        parts.append(f"{f.name}={_redact(value) if f.name == 'api_key' else value}")
    return ", ".join(parts)


CONFIG = AppConfig.load()
