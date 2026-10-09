import logging

import pytest

from helpers import stored, write
from src.indexing.pdf_reader import looks_glued

# Like pypdf's output for a two-column typeset book seen in practice, where ~20% of "words" exceeded
# 25 characters (PyMuPDF's extraction of the same book: 0%).
GLUED = "12 EXPENSES Onlymanagerscanapprovetravel.Mostclaimsneeda receiptwithinthirtydays,orareturned. Common items " * 15


def test_looks_glued():
    assert looks_glued(GLUED)
    assert not looks_glued("Only managers can approve travel; most claims need a receipt. " * 20)
    assert not looks_glued("Short.")                                   # too little text to judge


def test_pymupdf_reader_pages_and_labels(make_collection):
    pymupdf = pytest.importorskip("pymupdf")
    h = make_collection()
    pdf = pymupdf.open()
    for text in ("Employees get 25 vacation days.", "Expenses need a receipt."):
        pdf.new_page().insert_text((72, 72), text)
    pdf.save(h.settings.raw_dir / "book.pdf")

    h.sync()
    rows = stored(h)
    assert [d for d, _, _ in rows] == ["book.pdf#0", "book.pdf#1"]
    assert [m["page_label"] for _, _, m in rows] == ["1", "2"]
    assert "Employees get 25 vacation days." in rows[0][1]       # spaces preserved


def test_glued_text_warning(make_collection, caplog):
    h = make_collection()
    write(h.settings.raw_dir / "glued.txt", GLUED)
    with caplog.at_level(logging.WARNING):
        h.sync()
    assert "missing spaces" in caplog.text and "glued.txt" in caplog.text
