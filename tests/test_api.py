"""The FastAPI app against a real AppContext: real collections in a temp dir, real agents driven by a
scripted tool-calling LLM, and (for routing tests) a fake router LLM."""
import json

import pytest
from fastapi.testclient import TestClient
from llama_index.core import Settings

import src.fastapi_app as api
from helpers import FakeRouterLLM, scripted_agent_llm, write
from src.agent_setup import ConversationStore, QueryRouter, build_direct_agent, build_generic_tools, build_memory, build_rag_agent, default_routes
from src.app_context import AppContext, Profile


@pytest.fixture
def make_client(make_collection, monkeypatch):
    def _make(router_replies: dict[str, str] | None = None) -> tuple[TestClient, AppContext]:
        Settings.llm = scripted_agent_llm()
        rules, transcripts = make_collection("rules"), make_collection("transcripts")
        for ed, text in (("2014", "Grappling uses an Athletics check."), ("2024", "Grappling is an Unarmed Strike option.")):
            write(rules.settings.raw_dir / ed / "_metadata.json", json.dumps({"edition": ed}))
            write(rules.settings.raw_dir / ed / "grapple.txt", text)
        rules.sync()
        handles = {"rules": rules, "transcripts": transcripts}
        ctx = AppContext(
            profile=Profile.SERVE,
            collections=handles,
            agents={"rag": build_rag_agent(list(handles.values()), build_generic_tools(handles))},
            conversations=ConversationStore(lambda cid: build_memory(cid, Settings.llm, enable_fact_extraction=False)),
        )
        if router_replies is not None:
            ctx.agents["direct"] = build_direct_agent()
            ctx.router = QueryRouter(FakeRouterLLM(router_replies), default_routes(), "rag")
        monkeypatch.setattr(api, "bootstrap", lambda profile: ctx)
        return TestClient(api.app), ctx
    return _make


def events(raw: str) -> list[tuple[str, object]]:
    out = []
    for block in raw.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append((lines.get("event", "message"), json.loads(lines["data"])))
    return out


def test_status(make_client):
    client, _ = make_client()
    with client as c:
        assert c.get("/health").json() == {"status": "ok"}
        ready = c.get("/ready").json()
        assert ready["ready"] and ready["routes"] is None
        assert {x["name"]: x["chunks"] for x in ready["collections"]} == {"rules": 2, "transcripts": 0}


def test_chat_returns_sources_and_keeps_conversation(make_client):
    client, ctx = make_client()
    with client as c:
        r = c.post("/chat", json={"message": "How does grappling work?"}).json()
        assert r["response"] == "Grappling is an Unarmed Strike option." and r["route"] is None
        assert sorted(s["metadata"]["edition"] for s in r["sources"]) == ["2014", "2024"]
        cid = r["conversation_id"]
        c.post("/chat", json={"message": "And escaping?", "conversation_id": cid})
        _, memory = ctx.conversations.get(cid)
        assert [m.content for m in memory.get_all() if m.role.value == "user"] == ["How does grappling work?", "And escaping?"]
        assert c.post("/clear", json={"conversation_id": cid}).json()["cleared"] is True
        assert c.post("/chat", json={"message": "  "}).status_code == 422


def test_chat_stream_event_order(make_client):
    client, _ = make_client()
    with client as c:
        names = [e for e, _ in events(c.post("/chat/stream", json={"message": "Grappling?"}).text)]
        assert names[0] == "tool_call" and names[-2:] == ["sources", "done"] and "delta" in names


def test_retrieve(make_client):
    client, _ = make_client()
    with client as c:
        r = c.post("/retrieve", json={"query": "grappling", "filters": {"edition": "2014"}}).json()
        assert r["collection"] == "rules" and [x["metadata"]["edition"] for x in r["results"]] == ["2014"]
        assert c.post("/retrieve", json={"query": "x", "collection": "nope"}).status_code == 404


def test_documents_upsert(make_client):
    client, ctx = make_client()
    with client as c:
        body = {"collection": "transcripts", "documents": [{"id": "d1", "text": "First.", "metadata": {"session": 1}}]}
        assert c.post("/documents", json=body).json()["ids"] == ["d1"]
        body["documents"][0]["text"] = "Second."
        c.post("/documents", json=body)
        assert ctx.collection("transcripts").count() == 1
        assert c.post("/documents", json={"documents": []}).status_code == 422


def test_streams_and_summarize(make_client):
    client, _ = make_client()
    with client as c:
        for line in ("GM: The dragon flees north.", "Alice: I follow it."):
            r = c.post("/streams/transcripts/session-12", json={"text": line, "metadata": {"session": 12}})
        assert r.json()["doc_id"] == "session-12#1"
        assert c.get("/streams/transcripts").json()[0]["open"] is True
        assert c.post("/streams/transcripts/.bad", json={"text": "x"}).status_code == 422
        closed = c.post("/streams/transcripts/session-12/close", json={"metadata": {"date": "2026-10-02"}}).json()
        assert closed["granularity"] == "document" and not closed["open"]
        assert c.post("/streams/transcripts/session-12", json={"text": "late"}).status_code == 409
        assert c.post("/streams/transcripts/nope/close", json={}).status_code == 404

        s = c.post("/summarize", json={"collection": "transcripts", "filters": {"stream_id": "session-12"}}).json()
        assert s["chunks"] == 1 and s["doc_ids"] == ["session-12"] and s["summary"]
        assert c.post("/summarize", json={"collection": "transcripts", "filters": {"stream_id": "x"}}).status_code == 422


def test_ingest_endpoint(make_client):
    client, ctx = make_client()
    with client as c:
        raw = ctx.collection("rules").settings.raw_dir
        write(raw / "notes" / "_metadata.json", '{"_mode": "manual"}')
        write(raw / "notes" / "house.txt", "House rule.")
        write(raw / "new.txt", "New rule.")
        r = c.post("/ingest", json={"collection": "rules"}).json()["results"][0]
        assert r["new_files"] == ["new.txt"] and r["pending_files"] == ["notes/house.txt"]
        r = c.post("/ingest", json={"collection": "rules", "include_manual": True}).json()["results"][0]
        assert r["new_files"] == ["notes/house.txt"]


def test_router(make_client):
    client, _ = make_client({"hello": '{"route": "direct", "confidence": 0.95, "reason": "greeting"}',
                             "vague": '{"route": "clarify", "confidence": 0.8, "reason": "vague"}'})
    with client as c:
        assert c.get("/ready").json()["routes"] == ["rag", "direct", "clarify"]
        r = c.post("/chat", json={"message": "hello there"}).json()
        assert r["route"]["route"] == "direct" and r["response"] == "Hello!" and r["sources"] == []
        assert c.post("/chat", json={"message": "vague?"}).json()["route"]["route"] == "clarify"
        stream = events(c.post("/chat/stream", json={"message": "Grappling?"}).text)
        assert stream[0][0] == "route" and stream[0][1]["route"] == "rag"
        assert c.post("/route", json={"message": "hello"}).json()["route"] == "direct"


def test_route_endpoint_without_router_and_evaluate_without_judge(make_client):
    client, _ = make_client()
    with client as c:
        assert c.post("/route", json={"message": "x"}).status_code == 503
        assert c.post("/evaluate", json={"question": "q", "answer": "a"}).status_code == 503


def test_errors_are_generic_outside_dev(make_client):
    client, ctx = make_client()

    class Exploding:
        def run(self, **kwargs):
            raise RuntimeError("secret provider detail")

    ctx.agents["rag"] = Exploding()
    with client as c:
        r = c.post("/chat", json={"message": "hi"})
        assert r.status_code == 500 and "secret" not in r.json()["detail"]     # ENVIRONMENT=test, not dev
