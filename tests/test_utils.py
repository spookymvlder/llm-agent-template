"""Small pure functions: LLM output parsing, retrieval metrics, sanitising, the calculator tool, LLM kwargs."""
import enum
import math

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from src.agent_setup.agent_tools import calculate
from src.evaluation.metrics import mean_metrics, retrieval_metrics
from src.llm.parsing import parse_llm_json, parse_llm_model, strip_think_blocks
from src.preprocessing import normalize_text, safe_meta, sanitize
from src.schema import DocumentSchema


# ---- parsing ----------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ('{"a": 1}', {"a": 1}),
    ('<think>{"ignored": 1}</think>\n```json\n{"a": 1,}\n```', {"a": 1}),
    ('Sure: {"a": True, "b": [1, 2,], "c": None}', {"a": True, "b": [1, 2], "c": None}),
    ('{"latex": "\\alpha + \\mathbb{R}"}', {"latex": "\\alpha + \\mathbb{R}"}),   # invalid escapes are repaired
    ("no json here", {}),
    ("[1, 2]", {}),
])
def test_parse_llm_json(raw, expected):
    assert parse_llm_json(raw) == expected


def test_parse_llm_model():
    class Decision(BaseModel):
        route: str
        confidence: float

    assert parse_llm_model('{"route": "rag", "confidence": 0.8}', Decision) == Decision(route="rag", confidence=0.8)
    assert parse_llm_model('{"route": "rag"}', Decision) is None
    assert strip_think_blocks("<think>x</think> answer") == "answer"


# ---- metrics ----------------------------------------------------------------

def test_retrieval_metrics():
    m = retrieval_metrics([False, True, False], matched_expected=1, total_expected=2, k=3)
    assert m["hit_rate@3"] == 1.0 and m["mrr"] == 0.5 and m["recall@3"] == 0.5
    assert m["precision@3"] == pytest.approx(1 / 3)
    assert m["ndcg@3"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert retrieval_metrics([True, True], 2, 2, 2) == {
        "hit_rate@2": 1.0, "precision@2": 1.0, "recall@2": 1.0, "mrr": 1.0, "ndcg@2": 1.0}
    assert retrieval_metrics([], 0, 1, 5)["hit_rate@5"] == 0.0


def test_mean_metrics_skips_missing():
    assert mean_metrics([{"a": 1.0, "b": True}, {"a": 0.0}, {"b": False, "c": None}]) == {"a": 0.5, "b": 0.5}


# ---- sanitizer ----------------------------------------------------------------

def test_normalize_text_keeps_paragraphs():
    assert normalize_text("a  \t b\r\n\n\n\nc\nd ") == "a b\n\nc\nd"
    assert normalize_text(None) == "" and normalize_text(float("nan")) == ""


class Colour(enum.Enum):
    RED = "red"


@pytest.mark.parametrize("value, expected", [
    (None, None), (float("nan"), None), ([], None), (pd.NA, None),
    ("x", "x"), (3, 3), (True, True), (np.int64(4), 4), (Colour.RED, "red"),
    (["a", "b"], "a, b"), ({"k": 1}, '{"k": 1}'),
])
def test_safe_meta(value, expected):
    assert safe_meta(value) == expected


def test_sanitize():
    df = pd.DataFrame([
        {"doc_id": "1", "text": " Hello   world ", "tags": ["a", "b"]},
        {"doc_id": "1", "text": "duplicate id", "tags": None},
        {"doc_id": "2", "text": "   ", "tags": None},
    ])
    out = sanitize(df, DocumentSchema())
    assert out["doc_id"].tolist() == ["1"]
    assert out.iloc[0]["text"] == "Hello world" and out.iloc[0]["tags"] == "a, b"
    with pytest.raises(ValueError, match="title"):
        sanitize(df, DocumentSchema(text_columns=("title",)))


# ---- calculator tool ----------------------------------------------------------

@pytest.mark.parametrize("expression, expected", [
    ("2 + 2 * 10", "22"), ("(1 + 2) ** 3", "27"), ("-7 // 2", "-4"), ("10 % 4", "2"),
])
def test_calculate(expression, expected):
    assert calculate(expression) == expected


@pytest.mark.parametrize("expression", ["__import__('os')", "2 ** 99999", "True + 1", "x + 1", "open('f')"])
def test_calculate_rejects_non_arithmetic(expression):
    assert calculate(expression).startswith("Calculation error")


# ---- LLM factory kwargs ---------------------------------------------------------

def test_build_llm_from_settings_kwargs(monkeypatch):
    import src.llm.llm_factory as factory
    from src.config_helpers import LlmSettings
    from src.providers import LLMProvider

    calls = []
    monkeypatch.setattr(factory, "build_llm", lambda **kw: calls.append(kw))
    base = dict(base_url="http://localhost:11434", request_timeout=30.0, context_window=8192, temperature=0.1)

    factory.build_llm_from_settings(LlmSettings(provider=LLMProvider.OPENAI, model="m", max_tokens=100, **base))
    assert set(calls[-1]) == {"provider", "model", "api_key", "temperature", "max_tokens"}   # no Ollama-only kwargs

    factory.build_llm_from_settings(LlmSettings(provider=LLMProvider.OLLAMA, model="m", max_tokens=100,
                                                thinking=False, rate_limit_rpm=30, **base))
    ollama = calls[-1]
    assert ollama["additional_kwargs"] == {"num_predict": 100} and ollama["thinking"] is False
    assert ollama["base_url"] == base["base_url"] and ollama["rate_limiter"].requests_per_minute == 30


def test_missing_provider_package_has_pip_hint(monkeypatch):
    import src.llm.llm_factory as factory
    from src.providers import LLMProvider

    spec = factory._REGISTRY[LLMProvider.GEMINI]
    monkeypatch.setitem(factory._REGISTRY, LLMProvider.GEMINI,
                        factory._ProviderSpec("not_a_real_module", "X", spec.package, spec.env_var))
    with pytest.raises(ValueError, match="pip install llama-index-llms-google-genai"):
        factory.build_llm("gemini", "m", api_key="k")
