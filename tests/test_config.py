import logging
import os
from pathlib import Path

import pytest

from src.config import AppConfig, ConfigError
from src.providers import LLMProvider
from src.schema import FileMode

_PREFIXES = ("LLM_", "ROUTER_", "JUDGE_", "COLLECTION", "OPENAI_", "GOOGLE_", "ANTHROPIC_", "DISTANCE_",
             "CHUNK_", "TEMPERATURE", "CONTEXT_WINDOW", "DEFAULT_CHANGE_MODE", "REINDEX_CHANGED_FILES", "ENABLE_ROUTER",
             "SERVER_")
_EXACT = ("PORT",)


@pytest.fixture
def load(monkeypatch, tmp_path):
    """Load AppConfig from only the given env vars (no .env file, no inherited LLM/collection settings)."""
    for key in list(os.environ):
        if key.startswith(_PREFIXES) or key in _EXACT:
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
    cfg = load(COLLECTIONS="handbook, world-notes", COLLECTION_WORLD_NOTES_TOP_K="9",
               COLLECTION_HANDBOOK_RAW_DIR="corpus/handbook", COLLECTION_HANDBOOK_DESCRIPTION="Handbooks.")
    handbook, notes = cfg.collections
    assert handbook.raw_dir == (tmp_path / "corpus/handbook").resolve() and handbook.description == "Handbooks."
    assert notes.name == "world-notes" and notes.top_k == 9 and notes.raw_dir.name == "world-notes"
    assert cfg.default_collection is handbook and cfg.collection("world-notes") is notes
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


# ---- server address --------------------------------------------------------------

def test_server_address_defaults_and_precedence(load):
    cfg = load()
    assert (cfg.server_host, cfg.server_port) == ("127.0.0.1", 8000)


def test_platform_port_fallback(load):
    assert load(PORT="9000").server_port == 9000


def test_server_port_beats_platform_port(load):
    assert load(PORT="9000", SERVER_PORT="8100", SERVER_HOST="0.0.0.0").server_port == 8100


def test_invalid_server_port(load):
    with pytest.raises(ConfigError, match="SERVER_PORT"):
        load(SERVER_PORT="70000").validate()


@pytest.mark.parametrize("host, local", [
    ("127.0.0.1", True), ("localhost", True), ("::1", True), ("[::1]", True), ("127.0.1.1", True),
    ("0.0.0.0", False), ("192.168.1.20", False), ("::", False),
])
def test_is_loopback_host(host, local):
    from src.config_helpers import is_loopback_host
    assert is_loopback_host(host) is local


def test_serve_uses_config_flags_override_and_warns_when_exposed(monkeypatch, caplog):
    import sys
    import src.main as main_mod

    calls = []
    monkeypatch.setattr(main_mod.uvicorn, "run", lambda app, **kw: calls.append(kw))
    monkeypatch.setattr(sys, "argv", ["main", "serve"])
    main_mod.main()
    assert (calls[-1]["host"], calls[-1]["port"]) == (main_mod.cfg.server_host, main_mod.cfg.server_port)

    monkeypatch.setattr(sys, "argv", ["main", "serve", "--host", "0.0.0.0", "--port", "8123"])
    with caplog.at_level(logging.WARNING):
        main_mod.main()
    assert (calls[-1]["host"], calls[-1]["port"]) == ("0.0.0.0", 8123)
    assert "no authentication" in caplog.text
