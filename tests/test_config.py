import logging
import os
from pathlib import Path

import pytest

from src.config import AppConfig, ConfigError
from src.providers import LLMProvider
from src.schema import FileMode

_PREFIXES = ("LLM_", "ROUTER_", "JUDGE_", "COLLECTION", "OPENAI_", "GOOGLE_", "ANTHROPIC_", "DISTANCE_",
             "CHUNK_", "TEMPERATURE", "CONTEXT_WINDOW", "DEFAULT_CHANGE_MODE", "REINDEX_CHANGED_FILES", "ENABLE_ROUTER")


@pytest.fixture
def load(monkeypatch, tmp_path):
    """Load AppConfig from only the given env vars (no .env file, no inherited LLM/collection settings)."""
    for key in list(os.environ):
        if key.startswith(_PREFIXES):
            monkeypatch.delenv(key)

    def _load(**env: str) -> AppConfig:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return AppConfig.load(project_root=tmp_path, dotenv_path=tmp_path / "missing.env")
    return _load


def test_defaults(load):
    cfg = load()
    assert cfg.llm_settings.provider == LLMProvider.OLLAMA
    assert cfg.judge_llm_settings is None
    assert cfg.router_llm_settings == cfg.llm_settings          # router falls back to the primary model
    assert [c.name for c in cfg.collections] == ["documents"]
    assert cfg.default_change_mode == FileMode.STATIC
    cfg.validate()


def test_role_fallback_and_overrides(load):
    cfg = load(LLM_PROVIDER="ollama", LLM_MODEL="qwen3", LLM_TEMPERATURE="0.3", LLM_MAX_TOKENS="256",
               JUDGE_MODEL="llama3.1", JUDGE_RATE_LIMIT_RPM="15", ROUTER_MODEL="tiny", ROUTER_THINKING="false")
    judge, router = cfg.judge_llm_settings, cfg.router_llm_settings
    assert (judge.provider, judge.model, judge.temperature, judge.max_tokens, judge.rate_limit_rpm) == \
           (LLMProvider.OLLAMA, "llama3.1", 0.3, 256, 15)
    assert (router.model, router.thinking) == ("tiny", False)
    assert cfg.llm_settings.max_tokens == 256 and cfg.llm_settings.thinking is None


def test_provider_without_model_is_an_error(load):
    with pytest.raises(ConfigError, match="JUDGE_MODEL"):
        load(JUDGE_PROVIDER="openai")


def test_api_key_resolution(load):
    cfg = load(LLM_PROVIDER="gemini", LLM_MODEL="gemini-2.5-flash", GOOGLE_API_KEY="key-123")
    assert cfg.llm_settings.api_key == "key-123"
    assert "key-***" in cfg.to_log_str() and "key-123" not in cfg.to_log_str()


def test_collections_and_overrides(load, tmp_path):
    cfg = load(COLLECTIONS="rules, world-notes", COLLECTION_WORLD_NOTES_TOP_K="9",
               COLLECTION_RULES_RAW_DIR="corpus/rules", COLLECTION_RULES_DESCRIPTION="Rulebooks.")
    rules, notes = cfg.collections
    assert rules.raw_dir == (tmp_path / "corpus/rules").resolve() and rules.description == "Rulebooks."
    assert notes.name == "world-notes" and notes.top_k == 9 and notes.raw_dir.name == "world-notes"
    assert cfg.default_collection is rules and cfg.collection("world-notes") is notes
    with pytest.raises(KeyError):
        cfg.collection("nope")


def test_validate_reports_every_problem(load):
    cfg = load(LLM_PROVIDER="anthropic", LLM_MODEL="claude", DISTANCE_METRIC="dot", COLLECTIONS="ok_name,x,ok_name",
               CHUNK_SIZE="100", CHUNK_OVERLAP="200")
    with pytest.raises(ConfigError) as err:
        cfg.validate()
    message = str(err.value)
    for expected in ("ANTHROPIC_API_KEY", "DISTANCE_METRIC", "duplicates", "'x' is invalid", "CHUNK_OVERLAP"):
        assert expected in message


def test_ingest_only_skips_llm_key_check(load):
    load(LLM_PROVIDER="openai", LLM_MODEL="gpt-4o-mini").validate(check_llm_keys=False)


def test_renamed_env_vars_warn(load, caplog):
    with caplog.at_level(logging.WARNING):
        load(TEMPERATURE="0.9", REINDEX_CHANGED_FILES="false")
    assert "LLM_TEMPERATURE" in caplog.text and "DEFAULT_CHANGE_MODE" in caplog.text


def test_invalid_provider_lists_options(load):
    with pytest.raises(ValueError, match="openai"):
        load(LLM_PROVIDER="mistral")


def test_invalid_change_mode_lists_options(load):
    with pytest.raises(ValueError, match="manual"):
        load(DEFAULT_CHANGE_MODE="sometimes")
