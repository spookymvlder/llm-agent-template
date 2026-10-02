import json

import pytest

from helpers import stored
from src.indexing import StreamClosed, StreamError, StreamNotFound
from src.indexing.streams import Fragment, common_metadata


def test_fragments_are_searchable_immediately(make_collection):
    h = make_collection("transcripts")
    for i, line in enumerate(["GM: The dragon flees north.", "Alice: I follow it."]):
        assert h.append_to_stream("session-12", line, {"session": 12}) == f"session-12#{i}"
    assert [d for d, _, _ in stored(h)] == ["session-12#0", "session-12#1"]
    assert all(m["stream_id"] == "session-12" for _, _, m in stored(h))
    lines = h.streams.path("session-12").read_text(encoding="utf-8").splitlines()
    assert [json.loads(l)["text"] for l in lines] == ["GM: The dragon flees north.", "Alice: I follow it."]


def test_close_rechunks_with_shared_and_close_metadata(make_collection):
    h = make_collection("transcripts")
    h.append_to_stream("s12", "GM: The dragon flees.", {"session": 12, "speaker": "GM"})
    h.append_to_stream("s12", "Alice: I follow.", {"session": 12, "speaker": "Alice"})
    info = h.close_stream("s12", metadata={"date": "2026-10-02"})
    assert not info.open and info.granularity == "document"
    (doc_id, text, md), = stored(h)
    assert doc_id == "s12" and text == "GM: The dragon flees.\nAlice: I follow."
    assert md["session"] == 12 and md["date"] == "2026-10-02" and "speaker" not in md
    with pytest.raises(StreamClosed):
        h.append_to_stream("s12", "late")


def test_close_without_rechunk_keeps_fragments(make_collection):
    h = make_collection("transcripts")
    h.append_to_stream("s13", "one")
    h.append_to_stream("s13", "two")
    h.close_stream("s13", rechunk=False)
    assert [d for d, _, _ in stored(h)] == ["s13#0", "s13#1"]


def _content(handle):
    """Stored documents without LlamaIndex's internal fields (which hold fresh random chunk ids after a re-embed)."""
    return [(d, t, {k: v for k, v in m.items() if not k.startswith("_")}) for d, t, m in stored(handle)]


def test_reindex_replays_streams_as_last_stored(make_collection):
    h = make_collection("transcripts")
    h.append_to_stream("closed", "a"); h.append_to_stream("closed", "b"); h.close_stream("closed")
    h.append_to_stream("kept", "c"); h.close_stream("kept", rechunk=False)
    h.append_to_stream("open", "d")
    before = _content(h)

    assert h.sync().streams_embedded == []                            # restart: nothing to re-embed
    h.reset()
    assert h.count() == 0
    r = h.sync()
    assert sorted(r.streams_embedded) == ["closed", "kept", "open"] and _content(h) == before

    h.append_to_stream("open", "e")                                    # an open stream carries on
    h.close_stream("open")
    assert [t for d, t, _ in stored(h) if d == "open"] == ["d\ne"]


def test_edited_closed_stream_is_reembedded(make_collection):
    h = make_collection("transcripts")
    h.append_to_stream("s", "GM: typo")
    h.close_stream("s")
    path = h.streams.path("s")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["text"] = "GM: fixed"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert h.sync().streams_embedded == ["s"] and stored(h)[0][1] == "GM: fixed"


@pytest.mark.parametrize("bad", ["../escape", ".hidden", "has space", "", "a" * 200])
def test_invalid_stream_ids(make_collection, bad):
    with pytest.raises(StreamError):
        make_collection("transcripts").append_to_stream(bad, "x")


def test_unknown_stream(make_collection):
    with pytest.raises(StreamNotFound):
        make_collection("transcripts").close_stream("nope")


def test_common_metadata():
    frags = [Fragment(0, "a", {"session": 1, "speaker": "GM"}), Fragment(1, "b", {"session": 1, "speaker": "A"})]
    assert common_metadata(frags) == {"session": 1}
    assert common_metadata([]) == {}


def test_get_chunks_orders_fragments_numerically(make_collection):
    h = make_collection("transcripts")
    for i in range(12):
        h.append_to_stream("s", f"line {i}")
    assert [n.ref_doc_id for n in h.get_chunks({"stream_id": "s"})] == [f"s#{i}" for i in range(12)]
