"""
Golden-set regression harness: did a change (model, chunking, prompt, router) make things better or worse?

A golden set is a JSON list (or {"cases": [...]}, or JSONL) of cases you write by hand:

    {"id": "grapple-2024",
     "question": "How does grappling work?",
     "collection": "rules",                       # for retrieval metrics; default: first collection
     "expected_route": "rules",                   # checked when the router is enabled
     "expected_sources": ["2024/PHB.pdf"],        # doc ids, source paths (prefix of a doc id) or stream ids
     "reference_answer": "An Unarmed Strike option; the target saves.",   # optional, for correctness
     "filters": {"edition": "2024"}}              # optional, applied to the retrieval check

Every field but `question` is optional; each check runs only when its field is present (and, for the
judge checks, when JUDGE_MODEL is configured). Per case:

    route      — router.classify(question) == expected_route
    retrieval  — collection.search(question) vs expected_sources: hit rate, precision, recall, MRR, nDCG @k
                 (searched directly, so the numbers don't depend on which tool the agent picked)
    answer     — the full pipeline (router + agent), fresh conversation per case; latency and sources
    judge      — faithfulness and relevancy of the answer against the sources it used; correctness
                 (1-5) against reference_answer
    ragas      — optional (--ragas): faithfulness, response relevancy, context precision

Results go to data/<env>/evaluation/<run>/report.json and report.md; pass a previous run name as
`compare` to show the change in each aggregate metric.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from llama_index.core import Settings

from src.app_context import AppContext
from src.chat import answer
from src.evaluation.live import evaluate_answer
from src.evaluation.metrics import mean_metrics, retrieval_metrics
from src.indexing import RetrievedChunk
from src.schema import SOURCE_PATH, STREAM_ID

log = logging.getLogger(__name__)


@dataclass
class GoldenCase:
    question: str
    id: str = ""
    collection: str | None = None
    expected_route: str | None = None
    expected_sources: list[str] = field(default_factory=list)
    reference_answer: str | None = None
    filters: dict[str, Any] | None = None


@dataclass
class CaseResult:
    id: str
    question: str
    route: str | None = None
    route_correct: bool | None = None
    retrieved: list[str] = field(default_factory=list)
    retrieval: dict[str, float] = field(default_factory=dict)
    answer: str | None = None
    answer_sources: list[str] = field(default_factory=list)
    latency_s: float | None = None
    judge: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class EvalReport:
    run: str
    started_at: str
    settings: dict[str, Any]
    aggregate: dict[str, Any]
    cases: list[CaseResult]
    ragas: dict[str, Any] | None = None
    compared_to: str | None = None
    deltas: dict[str, float] | None = None


def load_golden(path: Path) -> list[GoldenCase]:
    """Read a golden set (.json list / {"cases": [...]}, or .jsonl). Case ids default to their position."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        raw = data["cases"] if isinstance(data, dict) else data
    cases = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or not str(item.get("question", "")).strip():
            raise ValueError(f"{path}: case {i} needs a non-empty 'question'.")
        known = {k: v for k, v in item.items() if k in GoldenCase.__dataclass_fields__}
        case = GoldenCase(**known)
        case.id = str(case.id or i + 1)
        cases.append(case)
    return cases


def matches(expected: str, chunk: RetrievedChunk) -> bool:
    """An expected source matches a chunk by doc id, by doc id prefix ('2024/PHB.pdf' matches
    '2024/PHB.pdf#12'), by source path, or by stream id."""
    doc_id = chunk.doc_id or ""
    return (
        expected == doc_id
        or doc_id.startswith(expected + "#")
        or expected == chunk.metadata.get(SOURCE_PATH)
        or expected == chunk.metadata.get(STREAM_ID)
    )


def _label(chunk: RetrievedChunk) -> str:
    return chunk.doc_id or chunk.metadata.get(SOURCE_PATH) or "?"


async def run_eval(
    ctx: AppContext,
    cases: list[GoldenCase],
    *,
    run_name: str,
    max_iterations: int,
    concurrency: int = 1,
    top_k: int | None = None,
    include_answer: bool = True,
    include_ragas: bool = False,
    out_dir: Path,
    compare: str | None = None,
) -> EvalReport:
    """Run every case and write report.json / report.md to out_dir/run_name.

    Raises:
        ImportError:       include_ragas without RAGAS installed (checked before any case runs).
        FileNotFoundError: `compare` names a run with no report.
    """
    if include_ragas and importlib.util.find_spec("ragas") is None:
        raise ImportError("--ragas needs RAGAS: pip install -r requirements-eval.txt")
    unknown = sorted({c.collection for c in cases if c.collection and c.collection not in ctx.collections})
    if unknown:
        raise ValueError(f"Golden set names unknown collection(s) {unknown}; configured: {', '.join(ctx.collections)}")
    if compare:
        _load_aggregate(out_dir / compare / "report.json")
    started = datetime.now().isoformat(timespec="seconds")
    semaphore = asyncio.Semaphore(max(1, concurrency))
    ragas_inputs: dict[str, tuple[str, list[str]]] = {}

    async def run_case(case: GoldenCase) -> CaseResult:
        async with semaphore:
            return await _run_case(ctx, case, max_iterations, top_k, include_answer, ragas_inputs)

    results = await asyncio.gather(*(run_case(c) for c in cases))
    report = EvalReport(
        run=run_name,
        started_at=started,
        settings=_settings(ctx, top_k, include_answer),
        aggregate=_aggregate(results),
        cases=list(results),
    )

    if include_ragas:
        report.ragas = _run_ragas(ctx, cases, ragas_inputs)
        report.aggregate.update({f"ragas.{k}": v for k, v in report.ragas.get("aggregate", {}).items()})

    if compare:
        previous = _load_aggregate(out_dir / compare / "report.json")
        report.compared_to = compare
        report.deltas = {
            k: round(v - previous[k], 4) for k, v in report.aggregate.items()
            if isinstance(v, (int, float)) and isinstance(previous.get(k), (int, float))
        }

    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(json.dumps(asdict(report), indent=2, default=str), encoding="utf-8")
    (run_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    log.info("Eval report written to %s", run_dir)
    return report


async def _run_case(
    ctx: AppContext,
    case: GoldenCase,
    max_iterations: int,
    top_k: int | None,
    include_answer: bool,
    ragas_inputs: dict[str, tuple[str, list[str]]],
) -> CaseResult:
    result = CaseResult(id=case.id, question=case.question)
    try:
        if case.expected_route and ctx.router is not None:
            decision = await ctx.router.classify(case.question)
            result.route = decision.route
            result.route_correct = decision.route == case.expected_route

        if case.expected_sources:
            handle = ctx.collection(case.collection) if case.collection else ctx.default_collection
            k = top_k or handle.settings.top_k
            chunks = await handle.search(case.question, top_k=k, filters=case.filters)
            relevant = [any(matches(e, c) for e in case.expected_sources) for c in chunks]
            found = sum(1 for e in case.expected_sources if any(matches(e, c) for c in chunks))
            result.retrieved = [_label(c) for c in chunks]
            result.retrieval = retrieval_metrics(relevant, found, len(case.expected_sources), k)

        if include_answer:
            start = time.perf_counter()
            decision, finished = await answer(ctx, case.question, None, max_iterations)
            result.latency_s = round(time.perf_counter() - start, 3)
            result.answer = finished.response
            result.answer_sources = list(dict.fromkeys(_label(s) for s in finished.sources))
            if result.route is None and decision is not None:
                result.route = decision.route
                if case.expected_route:
                    result.route_correct = decision.route == case.expected_route
            contexts = [s.text for s in finished.sources]
            ragas_inputs[case.id] = (finished.response, contexts)
            if ctx.evaluator_bundle is not None:
                result.judge = await _judge(ctx, case, finished.response, contexts)
    except Exception as e:
        log.exception("Eval case %s failed", case.id)
        result.error = f"{type(e).__name__}: {e}"
    return result


async def _judge(ctx: AppContext, case: GoldenCase, response: str, contexts: list[str]) -> dict[str, Any]:
    bundle = ctx.evaluator_bundle
    judged: dict[str, Any] = {}
    if contexts:
        summary = (await evaluate_answer(bundle, query=case.question, answer=response, contexts=contexts)).summary()
        for name in ("faithfulness", "relevancy"):
            if summary.get(name):
                judged[f"{name}_pass"] = summary[name]["passing"]
                judged[f"{name}_score"] = summary[name]["score"]
        guideline_results = [g["passing"] for g in summary.get("guidelines", []) if g and g.get("passing") is not None]
        if guideline_results:
            judged["guidelines_pass"] = sum(guideline_results) / len(guideline_results)
    if case.reference_answer and bundle.correctness is not None:
        try:
            correctness = await bundle.correctness.aevaluate(
                query=case.question, response=response, reference=case.reference_answer,
            )
            judged["correctness_score"] = correctness.score     # 1-5
            judged["correctness_pass"] = correctness.passing
        except Exception as e:
            log.warning("Correctness evaluator failed for case %s: %s", case.id, e)
    return judged


def _run_ragas(ctx: AppContext, cases: list[GoldenCase], inputs: dict[str, tuple[str, list[str]]]) -> dict[str, Any]:
    from src.evaluation.ragas_eval import RagasSample, compute_ragas_metrics

    if ctx.evaluator_bundle is None:
        log.warning("RAGAS needs a judge LLM: set JUDGE_MODEL.")
        return {"skipped": "no judge configured"}
    samples = [
        RagasSample(c.question, inputs[c.id][0], inputs[c.id][1], c.reference_answer)
        for c in cases if c.id in inputs
    ]
    return compute_ragas_metrics(samples, ctx.evaluator_bundle.llm, Settings.embed_model)


def _aggregate(results: list[CaseResult]) -> dict[str, Any]:
    routed = [r for r in results if r.route_correct is not None]
    agg: dict[str, Any] = {
        "cases": len(results),
        "errors": sum(1 for r in results if r.error),
    }
    if routed:
        agg["route_accuracy"] = round(sum(r.route_correct for r in routed) / len(routed), 4)
        agg["route_cases"] = len(routed)
    agg.update({f"retrieval.{k}": v for k, v in mean_metrics(r.retrieval for r in results if r.retrieval).items()})
    agg.update({f"judge.{k}": v for k, v in mean_metrics(r.judge for r in results if r.judge).items()})
    latencies = [r.latency_s for r in results if r.latency_s is not None]
    if latencies:
        agg["latency_s.mean"] = round(sum(latencies) / len(latencies), 3)
        agg["latency_s.max"] = max(latencies)
    return agg


def _settings(ctx: AppContext, top_k: int | None, include_answer: bool) -> dict[str, Any]:
    from src.config import CONFIG as cfg
    judge = cfg.judge_llm_settings
    return {
        "llm": f"{cfg.llm_settings.provider}/{cfg.llm_settings.model}",
        "router": f"{cfg.router_llm_settings.provider}/{cfg.router_llm_settings.model}" if ctx.router else None,
        "judge": f"{judge.provider}/{judge.model}" if judge else None,
        "embedding": f"{cfg.embedder_settings.provider}/{cfg.embedder_settings.model}",
        "chunk_size": cfg.embedder_settings.chunk_size,
        "chunk_overlap": cfg.embedder_settings.chunk_overlap,
        "top_k": top_k,
        "answers": include_answer,
    }


def _load_aggregate(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No previous report at {path} to compare against.")
    return json.loads(path.read_text(encoding="utf-8"))["aggregate"]


def render_markdown(report: EvalReport) -> str:
    """Human-readable report: settings, aggregate (with deltas), then one row per case."""
    lines = [f"# Eval run `{report.run}`", "", f"Started {report.started_at}", "", "## Settings", ""]
    lines += [f"- **{k}**: {v}" for k, v in report.settings.items()]
    lines += ["", "## Aggregate", ""]
    header = "| metric | value |" + (f" Δ vs `{report.compared_to}` |" if report.deltas is not None else "")
    lines += [header, "|---|---|" + ("---|" if report.deltas is not None else "")]
    for key, value in report.aggregate.items():
        delta = ""
        if report.deltas is not None:
            d = report.deltas.get(key)
            delta = f" {d:+.4f} |" if d is not None else " |"
        lines.append(f"| {key} | {value} |{delta}")

    lines += ["", "## Cases", "", "| id | route | retrieval | judge | latency | answer |", "|---|---|---|---|---|---|"]
    for c in report.cases:
        route = "" if c.route is None else f"{c.route} {'✓' if c.route_correct else '✗' if c.route_correct is False else ''}"
        retrieval = ", ".join(f"{k}={v:.2f}" for k, v in c.retrieval.items() if k.startswith(("hit_rate", "mrr")))
        judge = ", ".join(f"{k}={v}" for k, v in c.judge.items() if k.endswith("score"))
        answer_text = (c.error and f"**ERROR** {c.error}") or (c.answer or "").replace("\n", " ").replace("|", "\\|")[:120]
        lines.append(f"| {c.id} | {route} | {retrieval} | {judge} | {c.latency_s or ''} | {answer_text} |")

    misses = [c for c in report.cases if c.retrieval and not c.retrieval.get(next(iter(c.retrieval)))]
    if misses:
        lines += ["", "## Retrieval misses", ""]
        lines += [f"- **{c.id}** {c.question!r}: retrieved {c.retrieved}" for c in misses]
    return "\n".join(lines) + "\n"
