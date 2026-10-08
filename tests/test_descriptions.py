"""Collection descriptions: .env > "_description" in the collection's root _metadata.json > default."""
import logging

from helpers import stored, write

DESCRIPTION = "Rules of Mythic Bastionland, a tabletop RPG about knights."


def _root_meta(tmp_path, name, text):
    write(tmp_path / "raw" / name / "_metadata.json", text)


def test_description_from_root_metadata_not_stored_on_chunks(make_collection, tmp_path):
    _root_meta(tmp_path, "rules", f'{{"_description": "{DESCRIPTION}", "edition": "2024"}}')
    h = make_collection(description=None)
    assert h.description == DESCRIPTION
    write(h.settings.raw_dir / "a.txt", "Knights.")
    h.sync()
    (_, _, md), = stored(h)
    assert md["edition"] == "2024" and "_description" not in md


def test_env_description_wins(make_collection, tmp_path):
    _root_meta(tmp_path, "rules", f'{{"_description": "{DESCRIPTION}"}}')
    assert make_collection(description="From .env.").description == "From .env."


def test_missing_description_warns_and_defaults(make_collection, caplog):
    with caplog.at_level(logging.WARNING):
        h = make_collection(description=None)
    assert h.description == "Documents in the 'rules' collection."
    assert "has no description" in caplog.text and "_description" in caplog.text


def test_search_tool_uses_the_description(make_collection, tmp_path):
    from src.agent_setup import build_search_tool
    _root_meta(tmp_path, "rules", f'{{"_description": "{DESCRIPTION}"}}')
    assert DESCRIPTION in build_search_tool(make_collection(description=None)).metadata.description


def test_editing_description_does_not_reembed(make_collection, tmp_path):
    _root_meta(tmp_path, "rules", '{"_description": "Old.", "edition": "2024"}')
    h = make_collection(description=None)
    write(h.settings.raw_dir / "a.txt", "Knights.")
    h.sync()
    _root_meta(tmp_path, "rules", '{"_description": "New.", "edition": "2024"}')
    assert h.sync().changed_files == []


def test_nested_description_and_unknown_keys_warn(make_collection, tmp_path, caplog):
    h = make_collection()
    write(h.settings.raw_dir / "2014" / "_metadata.json", '{"_description": "x", "_descripton": "typo", "edition": "2014"}')
    write(h.settings.raw_dir / "2014" / "a.txt", "Knights.")
    with caplog.at_level(logging.WARNING):
        h.sync()
    assert "only the collection's root" in caplog.text and "_descripton" in caplog.text
    (_, _, md), = stored(h)
    assert md["edition"] == "2014" and not any(k.startswith("_d") for k in md)
