from __future__ import annotations

import logging

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core import Settings

from dataclasses import dataclass

import uuid

from src.agent_setup import build_rag_agent, build_generic_tools
from src.llm import configure_llamaindex, build_llm
from src.config import CONFIG as cfg
from src.indexing import IndexManager, ChromaIndexManager
from src.logging_setup import setup_logging
from src.evaluation import build_evaluator, EvaluatorBundle
from src.agent_setup import build_memory, Memory

log = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    agent: FunctionAgent
    query_engine: BaseQueryEngine
    max_iterations: int
    memory: Memory
    evaluator_bundle: EvaluatorBundle | None = None
    

def _setup_logging() -> None:
    setup_logging(
        enable=cfg.enable_logging,
        level=cfg.log_level,
        log_to_file=cfg.log_to_file,
        logs_dir=cfg.logs_dir,
    )


def _run_ingest(manager: IndexManager, chroma: ChromaIndexManager) -> None:
    """Ingest any new documents from raw_dir into ChromaDB."""
    result = manager.load_new_as_dataframe()
    if result.df.empty:
        log.info("Nothing to ingest — raw directory is empty or fully processed.")
        return
    chroma.load_or_build(result.df)
    manager.commit_processed(result.file_paths)
    log.info("Ingestion complete: %d chunk(s) processed.", len(result.df))


def _build_managers() -> tuple[IndexManager, ChromaIndexManager]:
    return (
        IndexManager(raw_dir=cfg.raw_dir, env_dir=cfg.env_dir),
        ChromaIndexManager(chroma_dir=cfg.chroma_dir, text_column="text", collection_name=cfg.collection_name),
    )


def _build_query_engine(chroma: ChromaIndexManager):
    result = chroma.load_or_build()
    return result.index.as_query_engine(similarity_top_k=cfg.top_k)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_only() -> None:
    """Configure logging and run ingestion. Does not build an agent.

    Used by: main.py `ingest` subcommand.
    """
    _setup_logging()
    configure_llamaindex(cfg.llm_settings, cfg.embedder_settings)
    manager, chroma = _build_managers()
    _run_ingest(manager, chroma)


def bootstrap() -> BootstrapResult:
    """Full startup sequence: logging, LlamaIndex, optional ingest, agent.

    Safe to call from both main.py and the FastAPI lifespan. Always
    configures logging so the app is properly instrumented regardless
    of how it was started (CLI, uvicorn, Docker, etc.).

    Used by: FastAPI lifespan, main.py `serve` and `chat` subcommands.
    """
    _setup_logging()
    configure_llamaindex(cfg.llm_settings, cfg.embedder_settings)

    manager, chroma = _build_managers()

    if cfg.auto_ingest and manager.ingested_count() == 0:
        log.info("Index is empty and auto_ingest is enabled — ingesting now...")
        _run_ingest(manager, chroma)

    if manager.ingested_count() == 0:
        log.warning(
            "ChromaDB collection is empty. The agent will have no documents to search. "
            "Add documents to %s and run ingest.", cfg.raw_dir
        )

    query_engine = _build_query_engine(chroma)
    tools = build_generic_tools(
        query_engine=query_engine,
        chroma_client=chroma.client,      # expose _client as a property on ChromaIndexManager
        collection_name=cfg.collection_name,
    )
    
    agent = build_rag_agent(query_engine=query_engine, extra_tools=tools)
    log.info("Agent ready.")




    evaluator_bundle = None
    if cfg.judge_llm_settings:
        judge_llm = build_llm(cfg.judge_llm_settings)
        evaluator_bundle = build_evaluator(judge_llm)
        log.info("Evaluator ready.")

    memory = build_memory(
        session_id=str(uuid.uuid4()),
        llm=Settings.llm,           # available here naturally
        token_limit=cfg.memory_token_limit,
        enable_fact_extraction=cfg.enable_fact_extraction,
        max_facts=cfg.max_facts,
    )
    

    return BootstrapResult(
        agent=agent, 
        evaluator_bundle=evaluator_bundle, 
        query_engine=query_engine,
        max_iterations=cfg.max_iterations,
        memory=memory,
        )