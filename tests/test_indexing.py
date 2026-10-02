"""Collections: file sync and change detection, folder metadata and modes, tabular data, runtime documents."""
import os
import time

import pytest
from llama_index.core.schema import MetadataMode

from helpers import stored, write
from src.indexing import CollectionOptions
from src.schema import DocumentSchema, FileMode


def doc_ids(handle):
    return [d for d, _, _ in stored(handle)]


def test_sync_new_unchanged_touched(make_collection):
    h = make_collection()
    write(h.settings.raw_dir / "a.txt", "Alpha.")
    write(h.settings.raw_dir / "sub" / "b.txt", "Bravo.")
    r = h.sync()
    assert sorted(r.new_files) == ["a.txt", "sub/b.txt"] and doc_ids(h) == ["a.txt#0", "sub/b.txt#0"]

    assert not h.sync().new_files                                    # nothing changed
    time.sleep(0.01)
    os.utime(h.settings.raw_dir / "a.txt")                           # same content, new mtime
    r = h.sync()
    assert not r.changed_files and doc_ids(h) == ["a.txt#0", "sub/b.txt#0"]


def test_changed_file_is_replaced_and_missing_file_kept(make_collection):
    h = make_collection()
    path = write(h.settings.raw_dir / "a.txt", "Version one.")
    h.sync()
    write(path, "Version two.")
    r = h.sync()
    assert r.changed_files == ["a.txt"] and r.chunks_removed == 1
    assert [t for _, t, _ in stored(h)] == ["Version two."]

    path.unlink()
    r = h.sync()
    assert r.missing_files == ["a.txt"] and h.count() == 1          # reported, not deleted


def test_folder_metadata_inheritance_and_visibility(make_collection):
    h = make_collection()
    raw = h.settings.raw_dir
    write(raw / "_metadata.json", '{"game": "dnd5e", "edition": "unknown"}')
    write(raw / "2014" / "_metadata.json", '{"edition": "2014"}')
    write(raw / "2014" / "grapple.txt", "Grappling uses an Athletics check.")
    h.sync()
    (_, _, md), = stored(h)
    assert (md["game"], md["edition"], md["source_path"]) == ("dnd5e", "2014", "2014/grapple.txt")

    write(raw / "2014" / "_metadata.json", '{"edition": "2014", "errata": "2018"}')
    assert h.sync().changed_files == ["2014/grapple.txt"]            # metadata edits re-embed
    assert stored(h)[0][2]["errata"] == "2018"


async def test_embeddings_see_content_only_llm_sees_metadata(make_collection):
    h = make_collection()
    write(h.settings.raw_dir / "_metadata.json", '{"edition": "2024"}')
    write(h.settings.raw_dir / "a.txt", "Grappling text.")
    h.sync()
    node = (await h.aretrieve("grappling"))[0].node
    assert node.get_content(metadata_mode=MetadataMode.EMBED) == "Grappling text."
    llm_text = node.get_content(metadata_mode=MetadataMode.LLM)
    assert "edition: 2024" in llm_text and "file_hash" not in llm_text


def test_invalid_metadata_json(make_collection):
    h = make_collection()
    write(h.settings.raw_dir / "_metadata.json", '{"edition": ')
    write(h.settings.raw_dir / "a.txt", "x")
    with pytest.raises(ValueError, match="not valid JSON"):
        h.sync()
    write(h.settings.raw_dir / "_metadata.json", '{"_mode": "sometimes"}')
    with pytest.raises(ValueError, match="_mode"):
        h.sync()


def test_manual_folders_wait_for_explicit_ingest(make_collection):
    h = make_collection()
    raw = h.settings.raw_dir
    write(raw / "notes" / "_metadata.json", '{"_mode": "manual", "kind": "notes"}')
    notes = write(raw / "notes" / "house.txt", "Crits on 19-20.")
    write(raw / "core.txt", "Core rules.")
    r = h.sync()
    assert r.new_files == ["core.txt"] and r.pending_files == ["notes/house.txt"]

    assert h.sync(include_manual=True).new_files == ["notes/house.txt"]
    md = next(m for d, _, m in stored(h) if d.startswith("notes"))
    assert md["kind"] == "notes" and "_mode" not in md

    write(notes, "Crits on 18-20 (draft).")
    assert h.sync().pending_files == ["notes/house.txt"]
    assert "draft" not in " ".join(t for _, t, _ in stored(h))


def test_manual_default_mode(make_collection):
    h = make_collection(default_mode=FileMode.MANUAL)
    write(h.settings.raw_dir / "_metadata.json", '{"_mode": "static"}')
    write(h.settings.raw_dir / "a.txt", "x")
    assert h.sync().new_files == ["a.txt"]                            # folder override beats the default


def test_tabular_schema_and_steps(make_collection):
    def split_tags(df):
        df["tags"] = df["tags"].map(lambda v: v.split("|") if isinstance(v, str) else [])
        return df

    h = make_collection("papers", options=CollectionOptions(
        schema=DocumentSchema(tabular=True, id_column="id", text_columns=("title", "abstract")),
        steps=[split_tags],
    ))
    write(h.settings.raw_dir / "papers.csv", "id,title,abstract,year,tags\n"
          "p1,Fair ranking,Constraints.,2023,ir|fairness\np2,,,2021,\n")
    h.sync()
    (doc_id, text, md), = stored(h)                                  # empty row dropped
    assert doc_id == "papers.csv#p1" and text == "Fair ranking\n\nConstraints."
    assert md["year"] == 2023 and md["tags"] == "ir, fairness"


def test_add_documents_upsert_and_missing_metadata(make_collection):
    h = make_collection("transcripts")
    ids = h.add_documents([
        {"id": "s1", "text": "GM: The dragon flees.", "metadata": {"session": 12, "speakers": ["GM"]}},
        {"text": "Player: I follow.", "metadata": {"session": 12}},
    ])
    assert ids[0] == "s1" and len(ids) == 2
    h.add_documents([{"id": "s1", "text": "GM: The dragon flees south.", "metadata": {"session": 12}}])
    texts = {d: (t, m) for d, t, m in stored(h)}
    assert texts["s1"][0] == "GM: The dragon flees south." and len(texts) == 2
    other = next(m for d, (_, m) in texts.items() if d != "s1")
    assert "speakers" not in other                                   # missing keys are omitted, not ""
    assert h.runtime_chunk_count() == 2


async def test_filters_and_get_chunks_order(make_collection):
    h = make_collection()
    for ed in ("2014", "2024"):
        write(h.settings.raw_dir / ed / "_metadata.json", f'{{"edition": "{ed}"}}')
        write(h.settings.raw_dir / ed / "g.txt", f"Grappling {ed}.")
    h.sync()
    results = await h.search("grappling", filters={"edition": "2014"})
    assert [r.metadata["edition"] for r in results] == ["2014"]
    assert [n.ref_doc_id for n in h.get_chunks()] == ["2014/g.txt#0", "2024/g.txt#0"]
    with pytest.raises(ValueError, match="limit"):
        h.get_chunks(limit=1)


def test_reset_wipes_files_and_runtime_documents(make_collection):
    h = make_collection()
    write(h.settings.raw_dir / "a.txt", "x")
    h.sync()
    h.add_documents([{"text": "runtime"}])
    h.reset()
    assert h.count() == 0
    assert h.sync().new_files == ["a.txt"] and h.count() == 1        # files come back, runtime docs don't
