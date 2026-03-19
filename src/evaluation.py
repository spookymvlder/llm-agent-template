from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    GuidelineEvaluator,
    RelevancyEvaluator,
)
from llama_index.core.evaluation.base import EvaluationResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default guidelines — override or extend per project
# ---------------------------------------------------------------------------

DEFAULT_GUIDELINES: list[str] = [
    "The response should be factual and grounded in the retrieved documents.",
    "The response should not make causal claims from correlational data.",
    "The response should acknowledge uncertainty where the source documents are ambiguous.",
]


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

@dataclass
class EvaluatorBundle:
    """Holds all configured evaluators for a RAG pipeline.

    faithfulness and relevancy are always present.
    guidelines is a list of GuidelineEvaluators, one per guideline string,
    and may be empty if no guidelines were provided.
    """
    faithfulness: FaithfulnessEvaluator
    relevancy: RelevancyEvaluator
    guidelines: list[GuidelineEvaluator] = field(default_factory=list)

    @property
    def has_guidelines(self) -> bool:
        return len(self.guidelines) > 0


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

@dataclass
class AggregatedEvaluationResult:
    """Aggregated results from all evaluators for a single query/response pair."""
    query: str
    response: str
    faithfulness: EvaluationResult | None = None
    relevancy: EvaluationResult | None = None
    guidelines: list[EvaluationResult] = field(default_factory=list)

    @property
    def passing(self) -> bool:
        """True only if all evaluators that ran returned passing=True."""
        results = [r for r in [self.faithfulness, self.relevancy] if r is not None]
        results += self.guidelines
        return all(r.passing for r in results if r.passing is not None)

    def summary(self) -> dict[str, Any]:
        """Serializable summary suitable for API responses or logging."""
        def _fmt(r: EvaluationResult | None) -> dict | None:
            if r is None:
                return None
            return {
                "passing": r.passing,
                "score": r.score,
                "feedback": r.feedback,
            }

        return {
            "query": self.query,
            "response": self.response,
            "passing": self.passing,
            "faithfulness": _fmt(self.faithfulness),
            "relevancy": _fmt(self.relevancy),
            "guidelines": [_fmt(r) for r in self.guidelines],
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_evaluator(
    llm: Any,
    guidelines: list[str] | None = None,
) -> EvaluatorBundle:
    """Build a bundle of RAG evaluators backed by the provided judge LLM.

    The judge LLM should differ from the primary agent's LLM to avoid a
    model evaluating its own outputs.

    Args:
        llm:        The judge LLM instance. Built separately from Settings.llm
                    via build_llm(cfg.judge_llm_settings).
        guidelines: Optional list of guideline strings for GuidelineEvaluator.
                    Defaults to DEFAULT_GUIDELINES if not provided.
                    Pass an empty list [] to disable guideline evaluation.

    Returns:
        EvaluatorBundle with faithfulness, relevancy, and optional guidelines.
    """
    resolved_guidelines = guidelines if guidelines is not None else DEFAULT_GUIDELINES

    guideline_evaluators = [
        GuidelineEvaluator(llm=llm, guidelines=g)
        for g in resolved_guidelines
    ]

    log.info(
        "Built evaluator bundle: faithfulness=True, relevancy=True, guidelines=%d",
        len(guideline_evaluators),
    )

    return EvaluatorBundle(
        faithfulness=FaithfulnessEvaluator(llm=llm),
        relevancy=RelevancyEvaluator(llm=llm),
        guidelines=guideline_evaluators,
    )


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

async def evaluate_response(
    bundle: EvaluatorBundle,
    query: str,
    response: Any,  # llama_index Response object from query_engine.query()
) -> AggregatedEvaluationResult:
    """Run all evaluators in the bundle against a query/response pair.

    Individual evaluator failures are caught and logged rather than
    propagating — a single evaluator error should not discard all results.

    Args:
        bundle:   EvaluatorBundle from build_evaluator().
        query:    The original user query string.
        response: The Response object returned by query_engine.query() or
                  agent.run(). Must carry source_nodes for faithfulness
                  and relevancy to work correctly.

    Returns:
        AggregatedEvaluationResult with all available scores and feedback.
    """
    result = AggregatedEvaluationResult(
        query=query,
        response=str(response),
    )

    # Faithfulness
    try:
        result.faithfulness = await bundle.faithfulness.aevaluate_response(
            query=query,
            response=response,
        )
        log.debug("Faithfulness: passing=%s score=%s", result.faithfulness.passing, result.faithfulness.score)
    except Exception as e:
        log.warning("Faithfulness evaluator failed: %s", e)

    # Relevancy
    try:
        result.relevancy = await bundle.relevancy.aevaluate_response(
            query=query,
            response=response,
        )
        log.debug("Relevancy: passing=%s score=%s", result.relevancy.passing, result.relevancy.score)
    except Exception as e:
        log.warning("Relevancy evaluator failed: %s", e)

    # Guidelines
    for i, evaluator in enumerate(bundle.guidelines):
        try:
            g_result = await evaluator.aevaluate_response(
                query=query,
                response=response,
            )
            result.guidelines.append(g_result)
            log.debug("Guideline %d: passing=%s", i, g_result.passing)
        except Exception as e:
            log.warning("Guideline evaluator %d failed: %s", i, e)

    log.info(
        "Evaluation complete for query='%.60s': overall passing=%s",
        query,
        result.passing,
    )
    return result