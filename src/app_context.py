from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto

from llama_index.core.agent.workflow import FunctionAgent

from src.agent_setup import Memory
from src.evaluation import EvaluatorBundle
from src.indexing import CollectionHandle

DEFAULT_AGENT = "rag"


class Profile(StrEnum):
    """What bootstrap() builds.

    INGEST: collections only (no LLM needed).
    CHAT:   collections + agents + memory (CLI; no evaluator).
    SERVE:  CHAT + evaluator if a judge is configured (FastAPI).
    """
    INGEST = auto()
    CHAT   = auto()
    SERVE  = auto()


@dataclass
class AppContext:
    """Runtime objects built by bootstrap(). Configuration values stay on CONFIG.

    Passed explicitly (FastAPI keeps it on app.state.ctx) rather than held in a global,
    so tests can construct one with stubs.
    """
    profile: Profile
    collections: dict[str, CollectionHandle]
    agents: dict[str, FunctionAgent] = field(default_factory=dict)
    evaluator_bundle: EvaluatorBundle | None = None
    memory: Memory | None = None

    @property
    def default_collection(self) -> CollectionHandle:
        """The first collection opened (the first in COLLECTIONS, unless bootstrap was given a subset)."""
        return next(iter(self.collections.values()))

    def collection(self, name: str) -> CollectionHandle:
        try:
            return self.collections[name]
        except KeyError:
            raise KeyError(f"Unknown collection '{name}'. Available: {', '.join(self.collections)}") from None

    @property
    def agent(self) -> FunctionAgent:
        """The default agent, which can search every collection."""
        return self.agents[DEFAULT_AGENT]
