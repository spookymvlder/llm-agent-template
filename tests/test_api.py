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
        handbook, meetings = make_collection("handbook"), make_collection("meetings")
        for ed, text in (("2023", "Employees get 20 vacation days."), ("2024", "Employees get 25 vacation days.")):
            write(handbook.settings.raw_dir / ed / "_metadata.json", json.dumps({"version": ed}))
            write(handbook.settings.raw_dir / ed / "vacation.txt", text)
        handbook.sync()
        handles = {"handbook": handbook, "meetings": meetings}
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
        assert {x["name"]: x["chunks"] for x in ready["collections"]} == {"handbook": 2, "meetings": 0}


def test_chat_returns_sources_and_keeps_conversation(make_client):
    client, ctx = make_client()
    with client as c:
        r = c.post("/chat", json={"message": "How many vacation days do employees get?"}).json()
        assert r["response"] == "Employees get 25 vacation days." and r["route"] is None
        assert sorted(s["metadata"]["version"] for s in r["sources"]) == ["2023", "2024"]
        cid = r["conversation_id"]
        c.post("/chat", json={"message": "And escaping?", "conversation_id": cid})
        _, memory = ctx.conversations.get(cid)
        assert [m.content for m in memory.get_all() if m.role.value == "user"] == ["How many vacation days do employees get?", "And escaping?"]
        assert c.post("/clear", json={"conversation_id": cid}).json()["cleared"] is True
        assert c.post("/chat", json={"message": "  "}).status_code == 422


def test_answer_is_logged_with_tool_calls_and_timing(make_client, caplog):
    import logging
    client, _ = make_client({"hello": '{"route": "direct", "confidence": 0.95, "reason": "greeting"}'})
    with client as c, caplog.at_level(logging.INFO, logger="src.agent_setup.agent_runner"):
        c.post("/chat", json={"message": "How many vacation days do employees get?"})
        assert "Tool call: search_handbook({'query': 'How many vacation days do employees get?'})" in caplog.text
        assert "Tool result: search_handbook returned 2 chunk(s): 20" in caplog.text     # '2023/vacation.txt (0.xx)'
        assert "1 tool call(s), 2 source chunk(s)" in caplog.text and "2 LLM call(s)" in caplog.text
        caplog.clear()
        c.post("/chat", json={"message": "hello"})                                    # routed to the tool-less agent
        assert "0 tool call(s)" in caplog.text and "did not use the document collections" in caplog.text


def test_chat_stream_event_order(make_client):
    client, _ = make_client()
    with client as c:
        names = [e for e, _ in events(c.post("/chat/stream", json={"message": "Vacation days?"}).text)]
        assert names[0] == "tool_call" and names[-2:] == ["sources", "done"] and "delta" in names


def test_retrieve(make_client):
    client, _ = make_client()
    with client as c:
        r = c.post("/retrieve", json={"query": "vacation", "filters": {"version": "2023"}}).json()
        assert r["collection"] == "handbook" and [x["metadata"]["version"] for x in r["results"]] == ["2023"]
        assert c.post("/retrieve", json={"query": "x", "collection": "nope"}).status_code == 404


def test_documents_upsert(make_client):
    client, ctx = make_client()
    with client as c:
        body = {"collection": "meetings", "documents": [{"id": "d1", "text": "First.", "metadata": {"meeting": 1}}]}
        assert c.post("/documents", json=body).json()["ids"] == ["d1"]
        body["documents"][0]["text"] = "Second."
        c.post("/documents", json=body)
        assert ctx.collection("meetings").count() == 1
        assert c.post("/documents", json={"documents": []}).status_code == 422


def test_streams_and_summarize(make_client):
    client, _ = make_client()
    with client as c:
        for line in ("Ana: We moved the launch to May.", "Ben: I'll update the roadmap."):
            r = c.post("/streams/meetings/standup-12", json={"text": line, "metadata": {"meeting": 12}})
        assert r.json()["doc_id"] == "standup-12#1"
        assert c.get("/streams/meetings").json()[0]["open"] is True
        assert c.post("/streams/meetings/.bad", json={"text": "x"}).status_code == 422
        closed = c.post("/streams/meetings/standup-12/close", json={"metadata": {"date": "2026-10-02"}}).json()
        assert closed["granularity"] == "document" and not closed["open"]
        assert c.post("/streams/meetings/standup-12", json={"text": "late"}).status_code == 409
        assert c.post("/streams/meetings/nope/close", json={}).status_code == 404

        s = c.post("/summarize", json={"collection": "meetings", "filters": {"stream_id": "standup-12"}}).json()
        assert s["chunks"] == 1 and s["doc_ids"] == ["standup-12"] and s["summary"]
        assert c.post("/summarize", json={"collection": "meetings", "filters": {"stream_id": "x"}}).status_code == 422


def test_ingest_endpoint(make_client):
    client, ctx = make_client()
    with client as c:
        raw = ctx.collection("handbook").settings.raw_dir
        write(raw / "notes" / "_metadata.json", '{"_mode": "manual"}')
        write(raw / "notes" / "remote.txt", "Draft policy.")
        write(raw / "new.txt", "New rule.")
        r = c.post("/ingest", json={"collection": "handbook"}).json()["results"][0]
        assert r["new_files"] == ["new.txt"] and r["pending_files"] == ["notes/remote.txt"]
        r = c.post("/ingest", json={"collection": "handbook", "include_manual": True}).json()["results"][0]
        assert r["new_files"] == ["notes/remote.txt"]


def test_router(make_client):
    client, _ = make_client({"hello": '{"route": "direct", "confidence": 0.95, "reason": "greeting"}',
                             "vague": '{"route": "clarify", "confidence": 0.8, "reason": "vague"}'})
    with client as c:
        assert c.get("/ready").json()["routes"] == ["rag", "direct", "clarify"]
        r = c.post("/chat", json={"message": "hello there"}).json()
        assert r["route"]["route"] == "direct" and r["response"] == "Hello!" and r["sources"] == []
        assert c.post("/chat", json={"message": "vague?"}).json()["route"]["route"] == "clarify"
        stream = events(c.post("/chat/stream", json={"message": "Vacation days?"}).text)
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
