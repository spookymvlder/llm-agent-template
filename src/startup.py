from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Mapping, Sequence

import chromadb
from llama_index.core import Settings

from src.agent_setup import ConversationStore, build_generic_tools, build_memory, build_rag_agent
from src.app_context import DEFAULT_AGENT, AppContext, Profile
from src.config import CONFIG as cfg
from src.config_helpers import LlmSettings
from src.evaluation import build_evaluator
from src.indexing import CollectionHandle, CollectionOptions, SyncResult, open_collection
from src.llm import build_llm_from_settings, configure_llamaindex
from src.logging_setup import setup_logging
from src.providers import EmbeddingProvider, LLMProvider

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    setup_logging(
        enable=cfg.enable_logging,
        level=cfg.log_level,
        log_to_file=cfg.log_to_file,
        logs_dir=cfg.logs_dir,
    )


def _ollama_models_in_use(profile: Profile) -> set[str]:
    """Ollama model names (LLMs and embedder) this profile will call."""
    roles: list[LlmSettings | None] = []
    if profile != Profile.INGEST:
        roles.append(cfg.llm_settings)
    if profile == Profile.SERVE:
        roles.append(cfg.judge_llm_settings)
    models = {r.model for r in roles if r and r.provider == LLMProvider.OLLAMA}
    if cfg.embedder_settings.provider == EmbeddingProvider.OLLAMA:
        models.add(cfg.embedder_settings.model)
    return models


def _check_ollama(profile: Profile) -> None:
    """Fail fast if Ollama is needed but not running, and warn about models that haven't been pulled.

    Raises:
        RuntimeError: Ollama is unreachable or timed out.
    """
    models = _ollama_models_in_use(profile)
    if not models:
        return
    base_url = cfg.llm_settings.base_url.rstrip("/")
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3.0) as response:
            pulled = {m["name"] for m in json.load(response).get("models", [])}
    except (urllib.error.URLError, TimeoutError) as e:
        raise RuntimeError(
            f"Ollama is not reachable at {base_url} ({e}). Start it with `ollama serve` and try again."
        ) from e
    for model in sorted(models):
        if model not in pulled and f"{model}:latest" not in pulled:
            log.warning("Ollama model '%s' is not pulled. Run `ollama pull %s`.", model, model)


def _open_collections(
    names: Sequence[str] | None,
    options: Mapping[str, CollectionOptions],
) -> dict[str, CollectionHandle]:
    selected = [cfg.collection(n) for n in names] if names else list(cfg.collections)
    unknown = set(options) - {c.name for c in cfg.collections}
    if unknown:
        raise KeyError(f"Options given for unknown collection(s): {', '.join(sorted(unknown))}")

    client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
    return {
        c.name: open_collection(
            c,
            client=client,
            chroma_dir=cfg.chroma_dir,
            manifest_dir=cfg.env_dir / "manifests",
            stream_dir=cfg.streams_dir / c.name,
            distance_metric=cfg.distance_metric,
            default_mode=cfg.default_change_mode,
            options=options.get(c.name),
        )
        for c in selected
    }


def _ingest(collections: Mapping[str, CollectionHandle], reindex: bool, include_manual: bool) -> None:
    for handle in collections.values():
        if reindex:
            log.info("Reindexing '%s'...", handle.name)
            handle.reset()
        log_sync(handle.name, handle.sync(include_manual=include_manual or reindex))


def log_sync(collection: str, result: SyncResult) -> None:
    if result.new_files or result.changed_files or result.streams_embedded:
        log.info(
            "Collection '%s': embedded %d new and %d changed file(s), %d stream(s) "
            "(%d document(s) added, %d old chunk(s) removed).",
            collection, len(result.new_files), len(result.changed_files), len(result.streams_embedded),
            result.documents_added, result.chunks_removed,
        )


def _warn_empty(collections: Mapping[str, CollectionHandle]) -> None:
    for handle in collections.values():
        if handle.count() == 0:
            log.warning(
                "Collection '%s' is empty. Add files to %s and run `python -m src.main ingest`, "
                "or add documents at runtime.", handle.name, handle.settings.raw_dir,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bootstrap(
    profile: Profile = Profile.SERVE,
    *,
    collections: Sequence[str] | None = None,
    reindex: bool = False,
    include_manual: bool = False,
    options: Mapping[str, CollectionOptions] | None = None,
) -> AppContext:
    """Build everything the given profile needs and return it as an AppContext.

    Stages: logging → config validation → Ollama check → LlamaIndex settings → collections
    (+ ingest) → agents → evaluator → conversation store. INGEST stops after collections.

    Files are synced when the profile is INGEST, or when AUTO_INGEST is enabled: new and changed files
    in STATIC folders are embedded; MANUAL folders only with include_manual.

    Args:
        profile:        What to build; see Profile.
        collections:    Names of collections to open. Default: all of COLLECTIONS.
        reindex:        Delete and re-embed the opened collections before ingesting (implies ingest and
                        include_manual: a rebuild includes every file).
        include_manual: Also embed new/changed files in MANUAL folders (see FileMode).
        options:        Per-collection schema, preprocessing steps and retrieval postprocessors, e.g.
                        {"rules": CollectionOptions(steps=[tag_edition], postprocessors=[EditionNote()])}.

    Raises:
        ConfigError:  Invalid configuration.
        RuntimeError: Ollama needed but unreachable.
    """
    _setup_logging()
    needs_llm = profile != Profile.INGEST
    cfg.validate(check_llm_keys=needs_llm)
    log.debug(cfg.to_log_str())
    _check_ollama(profile)
    configure_llamaindex(cfg.llm_settings if needs_llm else None, cfg.embedder_settings)

    handles = _open_collections(collections, options or {})
    if profile == Profile.INGEST or cfg.auto_ingest or reindex:
        _ingest(handles, reindex, include_manual)
    _warn_empty(handles)

    ctx = AppContext(profile=profile, collections=handles)
    if profile == Profile.INGEST:
        return ctx

    ctx.agents[DEFAULT_AGENT] = build_rag_agent(
        collections=list(handles.values()),
        extra_tools=build_generic_tools(handles),
        verbose=cfg.log_level == "DEBUG",
    )
    log.info("Agent ready.")

    if profile == Profile.SERVE and cfg.judge_llm_settings:
        ctx.evaluator_bundle = build_evaluator(build_llm_from_settings(cfg.judge_llm_settings))
        log.info("Evaluator ready.")

    ctx.conversations = ConversationStore(
        factory=lambda conversation_id: build_memory(
            session_id=conversation_id,
            llm=Settings.llm,
            token_limit=cfg.memory_token_limit,
            enable_fact_extraction=cfg.enable_fact_extraction,
            max_facts=cfg.max_facts,
        ),
        max_conversations=cfg.max_conversations,
        ttl_s=cfg.conversation_ttl_s,
    )
    return ctx
