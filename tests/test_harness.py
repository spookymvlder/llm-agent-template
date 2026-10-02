import json

import pytest
from llama_index.core import Settings

from helpers import scripted_agent_llm, write
from src.agent_setup import build_generic_tools, build_rag_agent
from src.app_context import AppContext, Profile
from src.evaluation.harness import load_golden, matches, run_eval
from src.indexing import RetrievedChunk


def chunk(doc_id, **metadata):
    return RetrievedChunk(collection="c", doc_id=doc_id, text="", score=1.0, metadata=metadata)


def test_matches():
    assert matches("2024/PHB.pdf", chunk("2024/PHB.pdf#12"))
    assert matches("session-12", chunk("session-12#3", stream_id="session-12"))
    assert matches("notes/a.txt", chunk("x", source_path="notes/a.txt"))
    assert not matches("2024/PHB", chunk("2024/PHB.pdf#1"))


def test_load_golden_formats(tmp_path):
    write(tmp_path / "g.json", json.dumps({"cases": [{"question": "q1"}, {"id": "b", "question": "q2", "unknown": 1}]}))
    assert [(c.id, c.question) for c in load_golden(tmp_path / "g.json")] == [("1", "q1"), ("b", "q2")]
    write(tmp_path / "g.jsonl", '{"question": "a"}\n\n{"question": "b"}\n')
    assert len(load_golden(tmp_path / "g.jsonl")) == 2
    write(tmp_path / "bad.json", '[{"id": 1}]')
    with pytest.raises(ValueError, match="question"):
        load_golden(tmp_path / "bad.json")


@pytest.fixture
def ctx(make_collection):
    Settings.llm = scripted_agent_llm()
    rules = make_collection("rules")
    write(rules.settings.raw_dir / "2024" / "grapple.txt", "Grappling is an Unarmed Strike option.")
    rules.sync()
    handles = {"rules": rules}
    return AppContext(profile=Profile.EVAL, collections=handles,
                      agents={"rag": build_rag_agent([rules], build_generic_tools(handles))})


async def test_run_eval_report_and_compare(ctx, tmp_path):
    golden = write(tmp_path / "golden.json", json.dumps([
        {"id": "hit", "question": "How does grappling work?", "expected_sources": ["2024/grapple.txt"]},
        {"id": "miss", "question": "Flanking?", "expected_sources": ["2024/flanking.txt"]},
        {"id": "answer-only", "question": "Anything"},
    ]))
    cases = load_golden(golden)
    out = tmp_path / "evaluation"

    report = await run_eval(ctx, cases, run_name="base", max_iterations=3, concurrency=2, out_dir=out)
    agg = report.aggregate
    assert agg["cases"] == 3 and agg["errors"] == 0
    assert agg["retrieval.hit_rate@5"] == 0.5 and agg["retrieval.mrr"] == 0.5
    assert "judge.faithfulness_pass" not in agg                         # no judge configured
    by_id = {c.id: c for c in report.cases}
    assert by_id["hit"].answer and by_id["hit"].answer_sources == ["2024/grapple.txt#0"]
    md = (out / "base" / "report.md").read_text(encoding="utf-8")
    assert "## Retrieval misses" in md and "**miss**" in md
    assert json.loads((out / "base" / "report.json").read_text(encoding="utf-8"))["aggregate"] == agg

    second = await run_eval(ctx, cases, run_name="next", max_iterations=3, include_answer=False, out_dir=out, compare="base")
    assert second.deltas["retrieval.mrr"] == 0.0 and "latency_s.mean" not in second.aggregate


async def test_run_eval_validates_before_running(ctx, tmp_path):
    cases = load_golden(write(tmp_path / "g.json", json.dumps([{"question": "q", "collection": "nope"}])))
    with pytest.raises(ValueError, match="nope"):
        await run_eval(ctx, cases, run_name="x", max_iterations=3, out_dir=tmp_path)
    ok = load_golden(write(tmp_path / "ok.json", json.dumps([{"question": "q"}])))
    with pytest.raises(FileNotFoundError):
        await run_eval(ctx, ok, run_name="x", max_iterations=3, out_dir=tmp_path, compare="missing")
