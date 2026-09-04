"""ProcessDocumentTool: path sandboxing (mirrors test_fs_reader.py) and a
real PDF end-to-end if PyMuPDF is installed.
"""

from __future__ import annotations

import pytest

from src.tools.builtin.document import ProcessDocumentTool


async def test_missing_file_reports_error(tmp_path):
    tool = ProcessDocumentTool(root=tmp_path)
    result = await tool.run(file_path="ghost.pdf")
    assert result.ok is False
    assert "not a file" in result.error


async def test_path_traversal_is_blocked(tmp_path):
    secret = tmp_path.parent / "secret.pdf"
    secret.write_bytes(b"not a real pdf")
    tool = ProcessDocumentTool(root=tmp_path)
    result = await tool.run(file_path="../secret.pdf")
    assert result.ok is False
    assert "escapes" in result.error


async def test_unsupported_extension_returns_empty_pages_not_an_error(tmp_path):
    # process_document() only branches on .pdf/image extensions; anything
    # else comes back with zero pages rather than raising -- the tool
    # should surface that as ok=True with an empty result, not crash.
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    tool = ProcessDocumentTool(root=tmp_path)
    result = await tool.run(file_path="note.txt")
    assert result.ok is True
    assert result.output["total_pages"] == 0


async def test_real_pdf_is_parsed(tmp_path):
    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Equipment V-101 pressure limit 12.0 bar")
    doc.save(pdf_path)
    doc.close()

    tool = ProcessDocumentTool(root=tmp_path)
    result = await tool.run(file_path="sample.pdf")
    assert result.ok is True
    assert result.output["total_pages"] == 1
    assert "V-101" in result.output["full_text"]
