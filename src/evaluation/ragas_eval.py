"""Optional RAGAS metrics for the eval harness (pip install -r requirements-eval.txt).

Faithfulness       — are the answer's claims supported by the retrieved contexts?
ResponseRelevancy  — does the answer address the question?
Context precision  — are the retrieved contexts relevant (with a reference answer if one is given)?

Written against RAGAS 0.2.x (its LlamaIndex wrappers), so the judge is any LlamaIndex LLM.
RAGAS makes several structured-output calls per sample; small local models (< ~7B) often fail to
produce parseable JSON — failed samples are reported, and aggregates use successful samples only.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class RagasSample:
    question: str
    answer: str
    contexts: list[str]
    reference: str | None = None


def compute_ragas_metrics(samples: list[RagasSample], llm: Any, embed_model: Any) -> dict:
    """Score samples with RAGAS. Returns {"per_sample": [...], "aggregate": {...}}.

    Raises:
        ImportError: RAGAS isn't installed.
    """
    try:
        from ragas import EvaluationDataset, RunConfig, evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.embeddings import LlamaIndexEmbeddingsWrapper
        from ragas.llms import LlamaIndexLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextPrecisionWithoutReference,
            ResponseRelevancy,
        )
    except ImportError as e:
        raise ImportError(f"RAGAS is not installed: pip install -r requirements-eval.txt ({e})") from e

    valid = [s for s in samples if s.answer and s.contexts]
    if not valid:
        log.warning("RAGAS: no samples with both an answer and retrieved contexts.")
        return {"per_sample": [], "aggregate": {}}

    judge = _json_mode(llm)
    llm_wrapper = LlamaIndexLLMWrapper(judge)
    has_reference = all(s.reference for s in valid)
    precision = (LLMContextPrecisionWithReference if has_reference else LLMContextPrecisionWithoutReference)(llm=llm_wrapper)
    metrics = [
        Faithfulness(llm=llm_wrapper),
        ResponseRelevancy(llm=llm_wrapper, embeddings=LlamaIndexEmbeddingsWrapper(embed_model)),
        precision,
    ]

    dataset = EvaluationDataset(samples=[
        SingleTurnSample(user_input=s.question, response=s.answer, retrieved_contexts=s.contexts,
                         reference=s.reference if has_reference else None)
        for s in valid
    ])
    # Faithfulness makes several sequential calls per sample (claim extraction + verification): local models need time.
    is_local = "ollama" in (type(judge).__module__ or "").lower()
    run_config = RunConfig(max_retries=1, max_wait=60, timeout=600 if is_local else 120)
    log.info("RAGAS: scoring %d sample(s) (reference=%s, local=%s).", len(valid), has_reference, is_local)
    scores = evaluate(dataset=dataset, metrics=metrics, run_config=run_config, raise_exceptions=False).to_pandas()

    per_sample = []
    for i, sample in enumerate(valid):
        row = scores.iloc[i]
        per_sample.append({
            "question": sample.question,
            "faithfulness": _float(row.get("faithfulness")),
            # Column names differ slightly between RAGAS minor versions.
            "response_relevancy": _float(_first(row, "response_relevancy", "answer_relevancy")),
            "context_precision": _float(_first(row, "llm_context_precision_with_reference",
                                               "llm_context_precision_without_reference", "context_precision")),
        })
    failed = sum(1 for r in per_sample if all(r[k] is None for k in ("faithfulness", "response_relevancy", "context_precision")))
    if failed:
        log.warning("RAGAS: %d/%d sample(s) produced no parseable scores; try a larger judge model.", failed, len(per_sample))

    aggregate = {k: _mean(r[k] for r in per_sample) for k in ("faithfulness", "response_relevancy", "context_precision")}
    aggregate.update(samples_scored=len(per_sample) - failed, samples_failed=failed)
    return {"per_sample": per_sample, "aggregate": aggregate}


def _json_mode(llm: Any) -> Any:
    """For Ollama, a copy of the judge with json_mode=True (Ollama then forces valid JSON), which avoids
    most RAGAS parse failures. Other providers are returned unchanged."""
    if "ollama" not in (type(llm).__module__ or "").lower():
        return llm
    model = (getattr(llm, "model", "") or "").lower()
    if any(tag in model for tag in (":1b", ":3b", "mini", "tiny")):
        log.warning("RAGAS judge '%s' looks small; RAGAS's JSON schemas usually need a 7B+ model.", model)
    try:
        return llm.model_copy(update={"json_mode": True})
    except Exception as e:
        log.warning("Could not enable json_mode on the Ollama judge (%s); using it as is.", e)
        return llm


def _first(row: Any, *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, 4)


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None
